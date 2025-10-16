# -*- coding: utf-8 -*-
"""
idw_from_xyz_dir.py

【機能 / What this script does】
- 指定フォルダ内の .xyz（または .txt）点群を一括ロードして統合
- 任意でボクセルダウンサンプリング（高速化用）
- KD木を構築し、XY正規格子（GRID_RES）上で IDW 補間（半径R内のk近傍）
- 補間点のみの LAS と、元点群＋補間点を結合した LAS を保存（分類: 2=Ground）
- 大規模データ向けにタイル分割（TILE_SIZE）とバッチ処理でメモリ節約

【前提 / Requirements】
- Python 3.10+
- numpy, scipy, laspy, tqdm（進捗は任意）
  pip install numpy scipy laspy tqdm

【注意 / Notes】
- XYZは "x y z" の空白区切りを想定。ヘッダ行なしを想定（ヘッダがある場合はSKIP_HEADER_LINESで調整）
- 出力LASのスケールは (0.001, 0.001, 0.001) を既定（ミリメートル精度）。必要に応じて変更してください
- 補間はXYグリッド上。水域など広域ではグリッド数が巨大になり得るため、GRID_RES と処理範囲（PADDING/BBOX_CROP）を調整してください
"""

import os
import glob
import math
import numpy as np
from scipy.spatial import cKDTree
import laspy
from tqdm import tqdm

# ====== 入出力設定 / IO settings ======
INPUT_DIR  = r"/data/fulldata/floor_sita_xyz/"   # 統合元のXYZフォルダ
OUTPUT_LAS_INTERP = r"/output/IDW_interp_only.las"   # 補間点のみ
OUTPUT_LAS_MERGED = r"/output/IDW_merged_floor.las"  # 元点群+補間点

# ====== 解析パラメータ / Parameters ======
GRID_RES     = 0.10    # [m] IDW補間グリッド間隔（細かいほど密だが重い）
TILE_SIZE    = 100.0   # [m] タイル処理の一辺長（大きいほど速いがメモリ使用増）
PADDING      = 2.0     # [m] タイル外周のバッファ（境界の補間品質向上）

# IDW設定 / IDW settings
IDW_POWER    = 2.0     # 距離重みの指数 p（一般に 1～3）
K_NEIGHBORS  = 12      # k近傍（半径内でこの数まで使用）
RADIUS       = 5.0     # [m] 参照半径（これより遠い点は無視）
MIN_NEIGHBOR = 3       # 最低参照点数（これを満たさない格子はスキップ）

# 速度・品質調整 / Speed-quality tradeoffs
USE_VOXEL_DOWNSAMPLE = True
VOXEL_SIZE   = 0.20    # [m] ボクセルサイズ（ダウンサンプリング間引き）

# XYZロード設定 / XYZ load options
XYZ_EXTS = (".xyz", ".txt")
SKIP_HEADER_LINES = 0  # 先頭数行にヘッダがある場合はここでスキップ数を指定

# LAS出力スケール / LAS scale & offsets
LAS_SCALE   = np.array([0.001, 0.001, 0.001])  # 軸ごとのスケール
# オフセットは後で自動設定（min座標の切り下げ） / offsets will be set from data mins

# ====== ユーティリティ ======
def load_xyz_dir(input_dir: str) -> np.ndarray:
    """フォルダ内の .xyz/.txt を全て読み込み、(N,3) ndarray を返す。"""
    files = []
    for ext in XYZ_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"No .xyz/.txt found in: {input_dir}")

    pts_list = []
    for path in tqdm(files, desc="📥 Loading XYZ files"):
        arr = np.loadtxt(path, dtype=np.float64, comments=None, skiprows=SKIP_HEADER_LINES)
        if arr.ndim == 1:
            # 単一点の行だけだった場合に (1,3) 化
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            raise ValueError(f"File has <3 columns (x y z): {path}")
        pts_list.append(arr[:, :3])  # x y z のみ
    pts = np.vstack(pts_list)
    return pts

def voxel_downsample_xyz(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """単純なグリッドスナップでのボクセルダウンサンプリング（代表点: 最初の点）。"""
    if voxel <= 0:
        return xyz
    # ボクセルキー（整数グリッド）を作る
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    # 辞書でユニーク化
    seen = {}
    for i, k in enumerate(map(tuple, keys)):
        if k not in seen:
            seen[k] = i
    idx = np.fromiter(seen.values(), dtype=np.int64, count=len(seen))
    return xyz[idx]

def make_grid(min_xy, max_xy, res):
    """XY正規格子の座標（中心）を生成。"""
    x_min, y_min = min_xy
    x_max, y_max = max_xy
    xs = np.arange(x_min, x_max + 1e-9, res)
    ys = np.arange(y_min, y_max + 1e-9, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid_xy = np.column_stack([X.ravel(), Y.ravel()])
    return grid_xy, xs, ys

def idw_interpolate_tile(grid_xy: np.ndarray, tree: cKDTree, xyz: np.ndarray,
                         k: int, radius: float, p: float, min_k: int) -> np.ndarray:
    """
    タイル内の格子点についてIDW補間を行い、z値配列を返す（存在しない場所はNaN）。
    - 半径内の近傍から最大k点取得（距離=0には安全対策）
    """
    if grid_xy.size == 0:
        return np.empty((0,), dtype=np.float64)

    # KD木検索：距離上限ありの k 最近傍
    dists, idxs = tree.query(grid_xy, k=k, distance_upper_bound=radius, workers=-1)

    # dists/idxs は shape=(M,k)。存在しない箇所は idx=tree.n に、dist=inf になる
    M = grid_xy.shape[0]
    z_out = np.full(M, np.nan, dtype=np.float64)

    # ベクトル化のための処理
    # 有効近傍（無限距離でない）を数える
    valid_mask = np.isfinite(dists)
    valid_counts = valid_mask.sum(axis=1)

    # 最低近傍数を満たす格子だけ計算
    ok = valid_counts >= min_k
    if not np.any(ok):
        return z_out

    # ok行だけ抽出
    d_ok = dists[ok]
    i_ok = idxs[ok]

    # d=0（グリッド点が既知点に一致）の場合はその点のZをそのまま採用
    zero_hit = (d_ok == 0.0)
    rows_with_zero = np.any(zero_hit, axis=1)

    # まず rows_with_zero でない行を通常のIDWとして計算
    normal_rows = np.where(~rows_with_zero)[0]
    if normal_rows.size > 0:
        d_n = d_ok[normal_rows]
        i_n = i_ok[normal_rows]
        # 有効な列のみ（距離有限）のマスク
        finite_n = np.isfinite(d_n)
        # 重み w = 1 / d^p
        # 0除算防止：finite_n 以外は0
        w = np.zeros_like(d_n)
        w[finite_n] = 1.0 / np.power(np.maximum(d_n[finite_n], 1e-12), p)
        # 参照点のZ
        z_neighbors = xyz[i_n[:, 0], 2]  # ダミーで初期化（後でブロードキャストで上書き）
        # ↑の一行は形の都合で一旦置く。実際は i_n を使って行ごとにZ配列を集める
        # 行ごとに取り直す：
        z_neighbors = xyz[i_n, 2]  # shape=(nr, k)
        # IDW推定
        z_est = np.sum(w * z_neighbors, axis=1) / np.sum(w, axis=1)
        # 出力へ反映
        z_out[np.where(ok)[0][normal_rows]] = z_est

    # rows_with_zero の行は、距離0の近傍のZをそのまま採用（平均）
    if np.any(rows_with_zero):
        sub = np.where(rows_with_zero)[0]
        d_z = d_ok[sub]
        i_z = i_ok[sub]
        # d=0 の列だけ抽出
        pick = (d_z == 0.0)
        # 行ごとに該当近傍のZの平均
        for rr, (mask_row, idx_row) in enumerate(zip(pick, i_z)):
            z_vals = xyz[idx_row[mask_row], 2]
            z_out[np.where(ok)[0][sub[rr]]] = float(np.mean(z_vals))

    return z_out

def write_las(xyz: np.ndarray, out_path: str):
    """(N,3) の xyz を LAS 1.4 (PointFormat 3) で保存（RGBは未設定）。"""
    if xyz.size == 0:
        raise ValueError("No points to write.")
    mins = xyz.min(axis=0)
    # オフセットは min を 1mm 単位で切り下げ
    offsets = np.floor(mins / LAS_SCALE) * LAS_SCALE

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = offsets
    header.scales  = LAS_SCALE

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # 既定で地面に分類（2）
    try:
        las.classification = np.full(xyz.shape[0], 2, dtype=np.uint8)
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    las.write(out_path)

def main():
    # 1) XYZ統合
    xyz = load_xyz_dir(INPUT_DIR)
    print(f"Loaded points: {xyz.shape[0]:,}")

    # 2) ダウンサンプリング（任意）
    if USE_VOXEL_DOWNSAMPLE:
        xyz_ds = voxel_downsample_xyz(xyz, VOXEL_SIZE)
        print(f"Downsampled: {xyz_ds.shape[0]:,} (voxel={VOXEL_SIZE} m)")
    else:
        xyz_ds = xyz

    # 3) KD木構築（既知点はダウンサンプル版を使用）
    tree = cKDTree(xyz_ds[:, :2])
    print("KDTree built.")

    # 4) グリッド作成に向けてXY範囲を取得
    min_xy = xyz_ds[:, :2].min(axis=0) - PADDING
    max_xy = xyz_ds[:, :2].max(axis=0) + PADDING

    # タイル分割ループ
    interp_pts = []  # 補間点格納（後で結合）
    x0 = min_xy[0]
    y0 = min_xy[1]
    x1 = max_xy[0]
    y1 = max_xy[1]

    nx = int(math.ceil((x1 - x0) / TILE_SIZE))
    ny = int(math.ceil((y1 - y0) / TILE_SIZE))
    total_tiles = nx * ny
    print(f"Tiling: {nx} x {ny} = {total_tiles} tiles (tile {TILE_SIZE} m, padding {PADDING} m)")

    with tqdm(total=total_tiles, desc="🧮 IDW tiles") as pbar:
        for ix in range(nx):
            for iy in range(ny):
                # タイルの外枠（パディング込み）
                tx_min = x0 + ix * TILE_SIZE - PADDING
                tx_max = min(x0 + (ix + 1) * TILE_SIZE + PADDING, x1)
                ty_min = y0 + iy * TILE_SIZE - PADDING
                ty_max = min(y0 + (iy + 1) * TILE_SIZE + PADDING, y1)

                # タイル内グリッド生成
                grid_xy, xs, ys = make_grid((tx_min, ty_min), (tx_max, ty_max), GRID_RES)

                if grid_xy.shape[0] == 0:
                    pbar.update(1)
                    continue

                # まず半径Rの内側に既知点がある場所だけを候補に（粗フィルタ）
                # 最近傍距離 > R の格子は除外（計算節約）
                nn_dist, _ = tree.query(grid_xy, k=1, distance_upper_bound=RADIUS, workers=-1)
                cand_mask = np.isfinite(nn_dist)  # 半径内に少なくとも1点ある
                cand_xy = grid_xy[cand_mask]
                if cand_xy.size == 0:
                    pbar.update(1)
                    continue

                # IDW 補間（タイル単位）
                z_est = idw_interpolate_tile(
                    cand_xy, tree, xyz_ds,
                    k=K_NEIGHBORS, radius=RADIUS, p=IDW_POWER, min_k=MIN_NEIGHBOR
                )
                valid = ~np.isnan(z_est)
                if np.any(valid):
                    pts_tile = np.column_stack([cand_xy[valid], z_est[valid]])
                    # タイル境界での二重生成は気にしなくてOK（同一格子は出ない）
                    interp_pts.append(pts_tile)

                pbar.update(1)

    # 5) 補間点を結合してLAS出力
    if len(interp_pts) == 0:
        raise RuntimeError("No interpolated points produced. Try increasing RADIUS, K, or lowering MIN_NEIGHBOR / GRID_RES.")
    interp_xyz = np.vstack(interp_pts)
    print(f"Interpolated points: {interp_xyz.shape[0]:,}")
    write_las(interp_xyz, OUTPUT_LAS_INTERP)
    print(f"✅ Wrote interp-only LAS: {OUTPUT_LAS_INTERP}")

    # 6) 元点群＋補間点の結合LASも出力（分類はどちらも2=Ground）
    merged_xyz = np.vstack([xyz, interp_xyz])
    write_las(merged_xyz, OUTPUT_LAS_MERGED)
    print(f"✅ Wrote merged LAS:     {OUTPUT_LAS_MERGED}")

if __name__ == "__main__":
    main()
