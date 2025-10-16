# -*- coding: utf-8 -*-
"""
【機能】川の向きに沿った矩形で「船だけ」を囲う（GPU対応・色保持版）
---------------------------------------------------------------------
- centerline.csv（川の中心線）を読み込む
- 各スライス方向ベクトルで川方向座標系を構築
- Z=1.5±0.2m 付近の点を抽出
- cuML.DBSCANで孤立クラスタ（船）を検出
- 各船クラスタについて：
    - スライス座標(u,v)上で最小矩形（川方向に揃えた長方形）を生成
- 元LASの色情報を保持
- 地図全体 + 緑の船バウンディングボックスをLAS出力
---------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import laspy
import cupy as cp
from cuml.cluster import DBSCAN as cuDBSCAN

# ===== 入出力 =====
INPUT_LAS = "/workspace/data/0925_ue_classified.las"
CENTERLINE_CSV = "/workspace/output/centerline.csv"
OUTPUT_LAS = "/workspace/output/1009_ship_bbox_along_river_color_gpu.las"

# ===== パラメータ =====
Z_TARGET = 1.5
Z_TOL = 0.2
DBSCAN_EPS = 2.0
MIN_SAMPLES = 40
RECT_MARGIN = 1.0
RECT_STEP = 0.3

# ===== 関数群 =====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales, header.offsets = src_header.scales, src_header.offsets
    if getattr(src_header, "srs", None):
        header.srs = src_header.srs
    return header

def write_colored_las(path, header_src, xyz_np, colors_np):
    """LAS書き出し（色付き）"""
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = xyz_np[:,0], xyz_np[:,1], xyz_np[:,2]
    las_out.red   = colors_np[:,0].astype(np.uint16)
    las_out.green = colors_np[:,1].astype(np.uint16)
    las_out.blue  = colors_np[:,2].astype(np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)
    print(f"✅ 出力完了: {path} ({len(xyz_np):,}点)")

def make_river_aligned_rectangle(c, t_hat, n_hat, umin, umax, vmin, vmax, step=0.3, z=1.5):
    """川方向に揃った矩形を生成"""
    rect_pts = []
    for v in np.arange(vmin, vmax + step, step):
        for u in [umin, umax]:
            p = c + t_hat * u + n_hat * v
            rect_pts.append([p[0], p[1], z])
    for u in np.arange(umin, umax + step, step):
        for v in [vmin, vmax]:
            p = c + t_hat * u + n_hat * v
            rect_pts.append([p[0], p[1], z])
    return np.array(rect_pts)

# ===== メイン処理 =====
def main():
    # --- 中心線読込 ---
    centers_df = pd.read_csv(CENTERLINE_CSV)
    centers_np = centers_df[["X", "Y"]].to_numpy(float)
    if len(centers_np) < 2:
        raise RuntimeError("中心線が2点未満です。")

    # --- LAS読込 ---
    las = laspy.read(INPUT_LAS)
    X, Y, Z = np.asarray(las.x, np.float64), np.asarray(las.y, np.float64), np.asarray(las.z, np.float64)
    R, G, B = np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)

    # --- GPU転送 ---
    X_cp, Y_cp, Z_cp = cp.asarray(X), cp.asarray(Y), cp.asarray(Z)
    XY_cp = cp.column_stack([X_cp, Y_cp])

    # --- 高さフィルタ（1.5±0.2m） ---
    mask_z = (Z_cp >= (Z_TARGET - Z_TOL)) & (Z_cp <= (Z_TARGET + Z_TOL))
    XY_ship = XY_cp[mask_z]
    if XY_ship.shape[0] == 0:
        raise RuntimeError("指定高さに点群がありません。")

    # --- 川方向ベクトル（平均方向） ---
    t_vecs = np.diff(centers_np, axis=0)
    t_mean = t_vecs.mean(axis=0)
    t_hat = t_mean / np.linalg.norm(t_mean)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    t_hat_cp, n_hat_cp = cp.asarray(t_hat), cp.asarray(n_hat)

    # --- 川方向座標系に変換 ---
    c0_cp = cp.asarray(centers_np[0])
    dxy = XY_ship - c0_cp
    u = dxy @ t_hat_cp
    v = dxy @ n_hat_cp
    uv = cp.column_stack([u, v])

    # --- GPU DBSCAN（孤立クラスタのみ矩形化） ---
    db = cuDBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(uv)
    unique_labels = cp.unique(labels)
    valid_labels = [lbl for lbl in unique_labels.tolist() if int(lbl) != -1]
    print(f"🚢 船クラスタ検出数: {len(valid_labels)}")

    bbox_list = []
    for lbl in valid_labels:
        mask = labels == lbl
        uv_sel = uv[mask]
        umin, vmin = float(uv_sel[:, 0].min()), float(uv_sel[:, 1].min())
        umax, vmax = float(uv_sel[:, 0].max()), float(uv_sel[:, 1].max())
        # 川方向矩形（余白付き）
        rect_pts = make_river_aligned_rectangle(
            centers_np[0], t_hat, n_hat,
            umin - RECT_MARGIN, umax + RECT_MARGIN,
            vmin - RECT_MARGIN, vmax + RECT_MARGIN,
            step=RECT_STEP, z=Z_TARGET
        )
        bbox_list.append(rect_pts)

    bbox_np = np.vstack(bbox_list) if bbox_list else np.empty((0, 3))

    # --- 出力統合（元の色を保持＋緑のBBox追加） ---
    map_xyz = np.column_stack([X, Y, Z])
    n_map, n_box = len(map_xyz), len(bbox_np)
    all_xyz = np.vstack([map_xyz, bbox_np])

    colors = np.zeros((n_map + n_box, 3), np.uint16)
    colors[:n_map, 0] = R
    colors[:n_map, 1] = G
    colors[:n_map, 2] = B
    colors[n_map:, :] = [0, 65535, 0]  # 緑（LAS16bit相当）

    # --- 出力 ---
    write_colored_las(OUTPUT_LAS, las.header, all_xyz, colors)
    print(f"✅ 地図点数: {n_map:,}, 船矩形点数: {n_box:,}")

if __name__ == "__main__":
    main()
