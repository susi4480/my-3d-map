# -*- coding: utf-8 -*-
"""
【機能】
- Z ≤ 3.0m の点群を X方向スライス
- 各YZ断面を2Dビットマップ化
- 点の存在しない最大の空間ラベル（連結成分）を抽出
- 緑点で復元し、元の点群と統合してLAS保存
"""

import numpy as np
import laspy
from scipy.ndimage import label

# === 入出力設定 ===
INPUT_LAS  = "/output/0725_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0728_navi_bitmap.las"

# === パラメータ設定 ===
Z_LIMIT = 3.0
SLICE_INTERVAL = 0.5
SLICE_THICKNESS = 0.2
GRID_RES = 0.1

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0

# === Z制限でフィルタリング ===
mask = xyz[:, 2] <= Z_LIMIT
xyz = xyz[mask]
rgb = rgb[mask]

# === スライス処理（X方向）===
x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
x_bins = np.arange(x_min, x_max, SLICE_INTERVAL)
green_pts = []

for x0 in x_bins:
    x1 = x0 + SLICE_THICKNESS
    mask_slice = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1)
    slice_pts = xyz[mask_slice]
    if len(slice_pts) == 0:
        continue

    y_min, y_max = slice_pts[:, 1].min(), slice_pts[:, 1].max()
    z_min, z_max = slice_pts[:, 2].min(), slice_pts[:, 2].max()
    gw = int(np.ceil((y_max - y_min) / GRID_RES))
    gh = int(np.ceil((z_max - z_min) / GRID_RES))
    grid = np.zeros((gh, gw), dtype=bool)

    for pt in slice_pts:
        yi = int((pt[1] - y_min) / GRID_RES)
        zi = int((pt[2] - z_min) / GRID_RES)
        if 0 <= zi < gh and 0 <= yi < gw:
            grid[zi, yi] = True  # 占有セル

    # 空き領域をラベリング
    empty = ~grid
    labels, num = label(empty)

    if num == 0:
        continue

    # 最大ラベルを航行空間とみなす
    max_label = max(range(1, num + 1), key=lambda l: np.sum(labels == l))
    mask_navi = labels == max_label

    # 緑点として3D復元
    nav_indices = np.argwhere(mask_navi)
    for zi, yi in nav_indices:
        y = y_min + yi * GRID_RES
        z = z_min + zi * GRID_RES
        x = (x0 + x1) / 2
        green_pts.append([x, y, z])

# === 点群結合とLAS保存 ===
green_pts = np.array(green_pts)
green_rgb = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(green_pts), 1))
all_pts = np.vstack([xyz, green_pts])
all_rgb = np.vstack([rgb, green_rgb])
all_rgb16 = (all_rgb * 65535).astype(np.uint16)

header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = all_pts.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
out_las = laspy.LasData(header)
out_las.x, out_las.y, out_las.z = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
out_las.red, out_las.green, out_las.blue = all_rgb16[:, 0], all_rgb16[:, 1], all_rgb16[:, 2]
out_las.write(OUTPUT_LAS)

print(f"✅ 航行可能空間（最大空き領域）を抽出してLAS出力しました: {OUTPUT_LAS}（緑点数: {len(green_pts)}）")
