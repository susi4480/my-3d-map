# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをX方向にスライス（YZ断面）
- 各スライスでビットマップ化（grid_res）
- 「1と0の境界」線を抽出し、3D座標に変換
- 線上の点を緑点として .las 形式で保存
"""

import os
import numpy as np
import laspy
import cv2
from skimage.measure import find_contours

# === 設定 ===
INPUT_LAS = "/output/0725_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0729xslice_contours.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

slice_thickness = 0.2
slice_interval = 0.5
z_limit = 3.0
grid_res = 0.05

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T

# === Z制限 ===
points = points[points[:, 2] <= z_limit]

# === スライス処理 ===
x_min, x_max = points[:, 0].min(), points[:, 0].max()
x_slices = np.arange(x_min, x_max, slice_interval)
output_pts = []

for x_center in x_slices:
    x_min_s = x_center - slice_thickness / 2
    x_max_s = x_center + slice_thickness / 2
    slice_pts = points[(points[:, 0] >= x_min_s) & (points[:, 0] <= x_max_s)]
    if len(slice_pts) < 50:
        continue

    y_vals, z_vals = slice_pts[:, 1], slice_pts[:, 2]
    y_min, y_max = y_vals.min(), y_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()

    h = int((z_max - z_min) / grid_res) + 1
    w = int((y_max - y_min) / grid_res) + 1
    bitmap = np.zeros((h, w), dtype=np.uint8)

    for y, z in zip(y_vals, z_vals):
        col = int((y - y_min) / grid_res)
        row = int((z_max - z) / grid_res)
        if 0 <= row < h and 0 <= col < w:
            bitmap[row, col] = 1

    # 境界線を抽出（0.5の等高線）
    contours = find_contours(bitmap, 0.5)
    for contour in contours:
        if len(contour) < 2:
            continue
        for r, c in contour:
            y = y_min + c * grid_res
            z = z_max - r * grid_res
            output_pts.append([x_center, y, z])

# === 緑点としてLAS出力 ===
if len(output_pts) > 0:
    output_pts = np.array(output_pts)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = output_pts.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las_out = laspy.LasData(header)
    las_out.x = output_pts[:, 0]
    las_out.y = output_pts[:, 1]
    las_out.z = output_pts[:, 2]
    las_out.red = np.zeros(len(output_pts), dtype=np.uint16)
    las_out.green = np.full(len(output_pts), 65535, dtype=np.uint16)  # 緑
    las_out.blue = np.zeros(len(output_pts), dtype=np.uint16)
    las_out.write(OUTPUT_LAS)
    print(f"✅ 線の保存完了（緑点数: {len(output_pts)}）→ {OUTPUT_LAS}")
else:
    print("⚠ 境界点が見つかりませんでした")
