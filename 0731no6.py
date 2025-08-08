# -*- coding: utf-8 -*-
"""
【機能】
- 入力LASをX方向に10m間隔でスライス
- 各スライス内でZスライス処理（Y方向にビット列化）
- 「1に挟まれた0」を航行可能空間として緑点出力
- 全緑点を統合して1つのLASファイルに保存
"""

import numpy as np
import laspy
from tqdm import tqdm
import os

# === 入出力 ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0725_suidoubasi_ue.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0801_gapfill_all_xslice.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# === パラメータ ===
Z_RES = 0.05      # Z方向ビット解像度
Y_RES = 0.1       # Y方向分割幅
Z_LIMIT = 3.5     # Z上限
X_THICKNESS = 10.0  # X方向スライス幅

# === LAS読み込み＆Z制限 ===
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
pts = pts[pts[:, 2] <= Z_LIMIT]

# === スライス境界 ===
x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
x_edges = np.arange(x_min, x_max + X_THICKNESS, X_THICKNESS)

all_green_pts = []

for xi in tqdm(range(len(x_edges) - 1), desc="Xスライス処理"):
    x0, x1 = x_edges[xi], x_edges[xi + 1]
    slice_mask = (pts[:, 0] >= x0) & (pts[:, 0] < x1)
    slice_pts = pts[slice_mask]
    if len(slice_pts) == 0:
        continue

    y_min, y_max = slice_pts[:, 1].min(), slice_pts[:, 1].max()
    z_min, z_max = slice_pts[:, 2].min(), Z_LIMIT
    y_bins = np.arange(y_min, y_max + Y_RES, Y_RES)
    z_bins = np.arange(z_min, z_max + Z_RES, Z_RES)

    for zi in range(len(z_bins) - 1):
        z0, z1 = z_bins[zi], z_bins[zi + 1]
        z_mask = (slice_pts[:, 2] >= z0) & (slice_pts[:, 2] < z1)
        z_slice_pts = slice_pts[z_mask]
        if len(z_slice_pts) == 0:
            continue

        bitmap = np.zeros(len(y_bins) - 1, dtype=np.uint8)
        yi_indices = ((z_slice_pts[:, 1] - y_min) / Y_RES).astype(int)
        yi_indices = yi_indices[(yi_indices >= 0) & (yi_indices < len(bitmap))]
        bitmap[yi_indices] = 1

        # 航行可能空間の抽出（1に挟まれた0を緑点に）
        inside = False
        for yi in range(1, len(bitmap) - 1):
            prev1 = bitmap[yi - 1]
            next1 = bitmap[yi + 1]
            if prev1 == 1 and next1 == 1 and bitmap[yi] == 0:
                y_center = (y_bins[yi] + y_bins[yi + 1]) / 2
                z_center = (z0 + z1) / 2
                x_center = (x0 + x1) / 2
                all_green_pts.append([x_center, y_center, z_center])

# === LAS出力 ===
if len(all_green_pts) > 0:
    pts_np = np.array(all_green_pts)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = pts_np.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las_out = laspy.LasData(header)
    las_out.x = pts_np[:, 0]
    las_out.y = pts_np[:, 1]
    las_out.z = pts_np[:, 2]
    las_out.red   = np.zeros(len(pts_np), dtype=np.uint16)
    las_out.green = np.full(len(pts_np), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(pts_np), dtype=np.uint16)
    las_out.write(OUTPUT_LAS)
    print(f"✅ 完了：{OUTPUT_LAS}（緑点数: {len(pts_np)}）")
else:
    print("⚠ 航行可能空間が見つかりませんでした")
