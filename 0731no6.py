# -*- coding: utf-8 -*-
"""
【機能】
- X方向にオーバーラップ付きスライス（幅10m・9m刻み）
- 各スライスで Y–Z平面を Z方向にビットマップ化（Yごと）
- 両側に1がある連続0（間の空間）を航行可能空間（緑点）とみなす
- 航行可能空間をLASで出力＋最近傍距離統計を表示
"""

import numpy as np
import laspy
from scipy.spatial import cKDTree
from tqdm import tqdm

# === 入出力設定 ===
INPUT_LAS = r"/output/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0731_overlap_zslice_gapspace.las"

# === パラメータ ===
Z_RES = 0.1         # Z方向ビットマップ分解能（m）
Y_RES = 0.1         # Y方向ビン幅（m）
Z_LIMIT = 3.5       # Z制限（m）
X_SLICE_WIDTH = 10.0  # 各スライスの厚み（m）
X_STEP = 9.0          # スライス開始位置の間隔（m）

# === LAS読み込みとZ制限 ===
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
pts = pts[pts[:, 2] <= Z_LIMIT]
if len(pts) == 0:
    raise RuntimeError("⚠ Z制限内の点が存在しません")

x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
x_starts = np.arange(x_min, x_max - X_SLICE_WIDTH, X_STEP)

navigable_pts = []

# === オーバーラップ付きXスライス処理 ===
for x0 in tqdm(x_starts, desc="Xスライス処理"):
    x1 = x0 + X_SLICE_WIDTH
    mask_x = (pts[:, 0] >= x0) & (pts[:, 0] < x1)
    slice_pts = pts[mask_x]
    if len(slice_pts) == 0:
        continue

    y_min, y_max = slice_pts[:, 1].min(), slice_pts[:, 1].max()
    z_min, z_max = slice_pts[:, 2].min(), Z_LIMIT
    y_bins = np.arange(y_min, y_max + Y_RES, Y_RES)
    z_bins = np.arange(z_min, z_max + Z_RES, Z_RES)

    for yi in range(len(y_bins) - 1):
        y0, y1 = y_bins[yi], y_bins[yi + 1]
        mask_y = (slice_pts[:, 1] >= y0) & (slice_pts[:, 1] < y1)
        yz_pts = slice_pts[mask_y]
        if len(yz_pts) == 0:
            continue

        bitmap = np.zeros(len(z_bins) - 1, dtype=np.uint8)
        zi_indices = ((yz_pts[:, 2] - z_min) / Z_RES).astype(int)
        zi_indices = zi_indices[(zi_indices >= 0) & (zi_indices < len(bitmap))]
        bitmap[zi_indices] = 1

        # 両側1に挟まれた0を航行可能空間とする
        inside = False
        for zi in range(1, len(bitmap) - 1):
            if bitmap[zi - 1] == 1 and bitmap[zi + 1] == 1 and bitmap[zi] == 0:
                inside = True
            elif inside and bitmap[zi] == 1:
                inside = False
            if inside and bitmap[zi] == 0:
                x_center = (x0 + x1) / 2
                y_center = (y0 + y1) / 2
                z_center = z_min + (zi + 0.5) * Z_RES
                navigable_pts.append([x_center, y_center, z_center])

# === 出力LAS生成（緑）===
if navigable_pts:
    navigable_pts = np.array(navigable_pts)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = navigable_pts.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las_out = laspy.LasData(header)
    las_out.x = navigable_pts[:, 0]
    las_out.y = navigable_pts[:, 1]
    las_out.z = navigable_pts[:, 2]
    las_out.red   = np.zeros(len(navigable_pts), dtype=np.uint16)
    las_out.green = np.full(len(navigable_pts), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(navigable_pts), dtype=np.uint16)
    las_out.write(OUTPUT_LAS)

    # === 最近傍距離の統計 ===
    tree = cKDTree(navigable_pts)
    dists, _ = tree.query(navigable_pts, k=2)
    print("✅ 最近傍距離の統計:")
    print(f"  平均距離 : {np.mean(dists[:, 1]):.4f} m")
    print(f"  中央値   : {np.median(dists[:, 1]):.4f} m")
    print(f"  最小距離 : {np.min(dists[:, 1]):.4f} m")
    print(f"  最大距離 : {np.max(dists[:, 1]):.4f} m")
    print(f"✅ 出力完了: {OUTPUT_LAS}（点数: {len(navigable_pts)}）")
else:
    print("⚠ 航行可能空間が見つかりませんでした")
