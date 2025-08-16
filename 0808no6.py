# -*- coding: utf-8 -*-
"""
【機能（ワンパス統合 / A版）】
1) 固定Y軸でスライス（幅 BIN_Y）し、各スライス内の Z ≤ Z_LIMIT の点から X中央値を「中心点」として推定
2) スライス中心点をY順に連結し「中心線」を作成（任意で移動平均スムージング）
3) 中心線の各点で接線→法線を求め、接線に垂直な厚み帯（±SLAB_THICK/2）で点群を抽出
4) 帯内点を (法線N, Z) 平面にビットマップ化 → クロージング → 補間セルを緑点として s=0（中心線上）に配置
5) 元点群（Z≤）＋緑点を統合して LAS 出力（スケール/オフセット/SRS/VLR/EVLRを継承、RGBは有無で分岐）
"""

import os
import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS  = r"/data/0725_suidoubasi_sita.las"          # 実在する入力に修正可
OUTPUT_LAS = r"/output/0808_ybin_centerline_perp_10m_green.las"

# === パラメータ ===
Z_LIMIT         = 3.5     # [m] これ以下の点のみ使用（橋上部除去など）
BIN_Y           = 2.0     # [m] Y軸スライス幅
MIN_PTS_PER_BIN = 50      # スライス内最低点数（満たさないと中心点スキップ）
SMOOTH_WINDOW_M = 10.0    # [m] 中心線Xの移動平均窓（0で無効）

# 垂直スライス（帯）→ N-Zラスタ化・補間用
SLAB_THICK    = 10.0      # [m] 接線方向の厚み（±SLAB_THICK/2）
GRID_RES      = 0.10      # [m] N-Z平面ラスタ解像度
MORPH_RADIUS  = 3         # [セル] クロージングの半径
MIN_PIXELS    = 50        # ラスタ化後のユニーク画素数がこれ未満ならスキップ
STEP_EVERY    = 1         # 何点おきに帯スライスするか（1=全点）
VERBOSE       = True

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MORPH_RADIUS + 1, 2 * MORPH_RADIUS + 1))


# === ユーティリティ ===
def moving_average_1d(arr, win):
    if win <= 1 or len(arr) < 2:
        return arr
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(arr_pad, kernel, mode='valid')

def centerline_from_y_slices(points_xyz, z_limit, bin_y, min_pts, smooth_window_m):
    """
    固定Y軸スライスで中心線を推定：
      - 各スライス（Y∈[y0,y1)）で Z≤z_limit を満たす点の X中央値を中心X、スライス中央を中心Yとする
      - Yの昇順に並べ、Xを移動平均で平滑化（任意）
    戻り値: centerline (N,2) = [[Xc,Yc],...]
    """
    pts = points_xyz[points_xyz[:, 2] <= z_limit]
    if len(pts) == 0:
        raise RuntimeError("Z制限後に点がありません（centerline_from_y_slices）")

    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    edges = np.arange(y_min, y_max + bin_y, bin_y)
    y_centers = 0.5 * (edges[:-1] + edges[1:])

    Xc_list, Yc_list = [], []
    for i in range(len(edges) - 1):
        y0, y1 = edges[i], edges[i+1]
        mask = (pts[:,1] >= y0) & (pts[:,1] < y1)
        if not np.any(mask):
            continue
        slab = pts[mask]
        if slab.shape[0] < min_pts:
            continue
        x_med = np.median(slab[:,0])   # 外れ値に強い中心X
        y_ctr = y_centers[i]
        Xc_list.append(x_med)
        Yc_list.append(y_ctr)

    if len(Xc_list) < 2:
        raise RuntimeError("有効なYスライスが不足（中心線を作成できません）")

    Xc = np.array(Xc_list, dtype=float)
    Yc = np.array(Yc_list, dtype=float)

    # Y昇順に整列（念のため）
    order = np.argsort(Yc)
    Xc = Xc[order]
    Yc = Yc[order]

    # 平滑化（メートル→ビン数）
    if smooth_window_m > 0:
        win = max(1, int(round(smooth_window_m / bin_y)))
        if win % 2 == 0:
            win += 1
        Xc = moving_average_1d(Xc, win)

    return np.column_stack([Xc, Yc])

def tangents_normals_from_polyline(xy):
    """中心線XYから、前後差分で接線tと法線nを計算。shape: (N,2),(N,2)"""
    N = xy.shape[0]
    if N < 2:
        raise RuntimeError("中心線点が2点未満です。")
    t = np.zeros((N, 2), dtype=float)
    t[1:-1] = xy[2:] - xy[:-2]
    t[0]    = xy[1] - xy[0]
    t[-1]   = xy[-1] - xy[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    t /= norm
    n = np.stack([-t[:,1], t[:,0]], axis=1)  # 左法線
    return t, n

def copy_header_with_metadata(src_header):
    """
    laspy 2.x 向けの安全なヘッダ継承:
    - copy()は使わない
    - scales/offsets/srs をそのまま継承
    - vlrs/evlrs は None チェックしてから extend
    """
    header = laspy.LasHeader(point_format=src_header.point_format,
                             version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets

    if getattr(src_header, "srs", None):
        header.srs = src_header.srs

    src_vlrs = getattr(src_header, "vlrs", None)
    if src_vlrs:
        header.vlrs.extend(src_vlrs)

    src_evlrs = getattr(src_header, "evlrs", None)
    if src_evlrs:
        header.evlrs.extend(src_evlrs)

    return header


# === メイン ===
def main():
    # 入力存在チェック & 出力Dir作成（空文字対策で or "."）
    if not os.path.isfile(INPUT_LAS):
        raise FileNotFoundError(f"INPUT_LAS not found: {INPUT_LAS}")
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    # 1) LAS読み込み
    las = laspy.read(INPUT_LAS)
    pts_all = np.vstack([las.x, las.y, las.z]).T

    # 入力RGB有無
    in_dims = set(las.point_format.dimension_names)
    has_rgb_in = {"red","green","blue"} <= in_dims
    if has_rgb_in:
        # 0..65535 のまま扱い、最後にそのまま代入（緑点は16bit合成）
        rgb_all = np.vstack([las.red, las.green, las.blue]).T
    else:
        rgb_all = None

    # 2) 固定Yスライスで中心線作成（Z≤限定）
    centerline = centerline_from_y_slices(
        points_xyz=pts_all,
        z_limit=Z_LIMIT,
        bin_y=BIN_Y,
        min_pts=MIN_PTS_PER_BIN,
        smooth_window_m=SMOOTH_WINDOW_M
    )

    # 3) Z≤ の点群だけ残す
    zmask = pts_all[:,2] <= Z_LIMIT
    pts   = pts_all[zmask]
    if has_rgb_in:
        rgb = rgb_all[zmask]
    XY = pts[:, :2]
    Z  = pts[:, 2]

    # 4) 中心線→接線・法線
    t, n = tangents_normals_from_polyline(centerline)

    # 5) 各中心線点で 垂直スライス → N-Z ラスタ → クロージング → 補間セル→緑点(s=0に配置)
    half_thick = SLAB_THICK * 0.5
    green_world = []
    idxs = range(0, len(centerline), STEP_EVERY)

    for i in idxs:
        c  = centerline[i]  # [X,Y]
        ti = t[i]           # 接線 unit
        ni = n[i]           # 法線 unit

        dxy = XY - c
        s = dxy @ ti        # 接線スカラー
        u = dxy @ ni        # 法線スカラー

        band_mask = np.abs(s) <= half_thick
        if not np.any(band_mask):
            continue

        u_slab = u[band_mask]
        z_slab = Z[band_mask]
        if u_slab.size == 0:
            continue

        # ラスタ範囲
        u_min, u_max = u_slab.min(), u_slab.max()
        z_min, z_max = z_slab.min(), z_slab.max()
        gw = int(np.ceil((u_max - u_min) / GRID_RES))
        gh = int(np.ceil((z_max - z_min) / GRID_RES))
        if gw <= 1 or gh <= 1:
            continue

        # 画素化（ユニーク画素で密度判定）
        ui = ((u_slab - u_min) / GRID_RES).astype(int)
        zi = ((z_slab - z_min) / GRID_RES).astype(int)
        ok = (zi >= 0) & (zi < gh) & (ui >= 0) & (ui < gw)
        if not np.any(ok):
            continue

        pix = np.stack([zi[ok], ui[ok]], axis=1)
        uniq = np.unique(pix, axis=0)
        if len(uniq) < MIN_PIXELS:
            continue

        grid = np.zeros((gh, gw), dtype=np.uint8)
        grid[uniq[:,0], uniq[:,1]] = 255

        # クロージング（穴埋め）
        closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, KERNEL)
        diff = (closed > 0) & (grid == 0)
        if not np.any(diff):
            continue

        # 補間セル → s=0（中心線上）に緑点を配置（u:法線方向, z:高さ）
        ii, jj = np.where(diff)
        u_cent = u_min + (jj + 0.5) * GRID_RES
        z_cent = z_min + (ii + 0.5) * GRID_RES

        pxy = c + (u_cent[:, None] * ni[None, :])   # (K,2)
        pz  = z_cent.reshape(-1, 1)
        pw  = np.hstack([pxy, pz])                  # (K,3)
        green_world.append(pw)

    if len(green_world) == 0:
        raise RuntimeError("緑点が生成されませんでした。パラメータ（BIN_Y, SLAB_THICK, GRID_RES, MORPH_RADIUS, MIN_PIXELS）を見直してください。")

    green_world = np.vstack(green_world)

    # 6) 出力統合（Z≤点＋緑点）
    all_pts = np.vstack([pts, green_world])

    # RGB：入力にRGBがあれば維持、緑点は純緑（16bit）
    write_rgb = False
    if has_rgb_in:
        green_rgb16 = np.zeros((len(green_world), 3), dtype=np.uint16)
        green_rgb16[:,1] = 65535  # pure green
        all_rgb16 = np.vstack([rgb, green_rgb16])  # rgbは16bitのまま
        write_rgb = True

    # 7) LAS出力（CRS/VLR/EVLR/SRS継承）
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)

    N = all_pts.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = all_pts[:,0]
    las_out.y = all_pts[:,1]
    las_out.z = all_pts[:,2]

    out_dims = set(las_out.point_format.dimension_names)
    if write_rgb and {"red","green","blue"} <= out_dims:
        las_out.red   = all_rgb16[:,0]
        las_out.green = all_rgb16[:,1]
        las_out.blue  = all_rgb16[:,2]
    # 出力PFにRGBが無ければスキップ（強度のみLAS）

    # 保存
    las_out.write(OUTPUT_LAS)

    if VERBOSE:
        print("✅ 完了: ", OUTPUT_LAS)
        print(f"  中心線点数: {len(centerline)}（BIN_Y={BIN_Y}m, MIN_PTS_PER_BIN={MIN_PTS_PER_BIN}, SMOOTH={SMOOTH_WINDOW_M}m）")
        print(f"  垂直帯: 厚み±{SLAB_THICK*0.5:.2f} m, STEP_EVERY={STEP_EVERY}")
        print(f"  ラスタ: GRID_RES={GRID_RES} m, MORPH_RADIUS={MORPH_RADIUS} px, MIN_PIXELS={MIN_PIXELS}")
        print(f"  Z≤点: {len(pts)} / 緑点: {len(green_world)} / 合計: {N}")
        print(f"  RGB入出力: in={has_rgb_in}, out_rgb_dims={'yes' if {'red','green','blue'} <= out_dims else 'no'}")

if __name__ == "__main__":
    main()
