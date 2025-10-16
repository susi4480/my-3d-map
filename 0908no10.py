# -*- coding: utf-8 -*-
"""
高さスライスごとの2Dモルフォロジー補間 → 積み重ね
- 入力LASは白色点群に変換
- 50cmごとの高さスライスを2D投影
- モルフォロジー閉処理で穴埋め
- 補間セルはスライス中央値Zに赤色点群として追加
"""

import os
import numpy as np
import laspy
from skimage.morphology import binary_closing, disk

# ===== 入出力 =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0908no10_slice_fill.las"
SLICE_DZ = 0.5  # 50cmごと

GRID_RES = 0.2  # 2D格子サイズ[m]
MORPH_RADIUS = 3  # モルフォロジー補間の半径（セル単位）

os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ===== LAS読み込み =====
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T

# 元データを白色点群に変換
orig_colors = {
    "red":   np.full(len(points), 65535, dtype=np.uint16),
    "green": np.full(len(points), 65535, dtype=np.uint16),
    "blue":  np.full(len(points), 65535, dtype=np.uint16)
}

# ===== スライス処理 =====
z_min, z_max = points[:, 2].min(), points[:, 2].max()
zslices = np.arange(z_min, z_max, SLICE_DZ)

interp_points = []
interp_colors = {"red": [], "green": [], "blue": []}

for z0 in zslices:
    z1 = z0 + SLICE_DZ
    mask = (points[:, 2] >= z0) & (points[:, 2] < z1)
    slice_pts = points[mask]
    if len(slice_pts) == 0:
        continue

    # 2D投影（XY座標 → グリッド）
    x_min, y_min = slice_pts[:, 0].min(), slice_pts[:, 1].min()
    x_max, y_max = slice_pts[:, 0].max(), slice_pts[:, 1].max()
    gw = int(np.ceil((x_max - x_min) / GRID_RES))
    gh = int(np.ceil((y_max - y_min) / GRID_RES))

    grid = np.zeros((gh, gw), dtype=np.uint8)
    xi = ((slice_pts[:, 0] - x_min) / GRID_RES).astype(int)
    yi = ((slice_pts[:, 1] - y_min) / GRID_RES).astype(int)
    ok = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)
    grid[yi[ok], xi[ok]] = 1

    # モルフォロジー補間（閉処理）
    closed = binary_closing(grid, disk(MORPH_RADIUS))

    # 補間セルを抽出（元セルとの差分）
    new_cells = np.argwhere((closed > 0) & (grid == 0))

    # 各補間セルを座標に戻す（Zは中央値）
    z_mid = (z0 + z1) / 2
    for yy, xx in new_cells:
        x = x_min + (xx + 0.5) * GRID_RES
        y = y_min + (yy + 0.5) * GRID_RES
        interp_points.append([x, y, z_mid])
        interp_colors["red"].append(65535)
        interp_colors["green"].append(0)
        interp_colors["blue"].append(0)

# ===== 出力データ結合 =====
all_points = np.vstack([points, np.array(interp_points)]) if interp_points else points
N = len(all_points)

header = laspy.LasHeader(point_format=7, version="1.4")
header.offsets = all_points.min(axis=0)
header.scales = [0.001, 0.001, 0.001]

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = all_points[:, 0], all_points[:, 1], all_points[:, 2]

# 色設定（元点群=白、補間点=赤）
las_out.red   = np.concatenate([orig_colors["red"],   np.array(interp_colors["red"],   dtype=np.uint16)])
las_out.green = np.concatenate([orig_colors["green"], np.array(interp_colors["green"], dtype=np.uint16)])
las_out.blue  = np.concatenate([orig_colors["blue"],  np.array(interp_colors["blue"],  dtype=np.uint16)])

las_out.write(OUTPUT_LAS)
print(f"✅ 出力完了: {OUTPUT_LAS}, 点数={N}")
