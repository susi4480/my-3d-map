# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをX方向にスライス（YZ断面）
- 各スライスでビットマップ化（grid_res）
- 「1と0の境界」線を抽出し、3D座標に変換
- 線をPLY / OBJ形式で保存（色は緑）
"""

import os
import numpy as np
import open3d as o3d
import cv2
from skimage.measure import find_contours
import laspy

# === 設定 ===
INPUT_LAS = "/output/0725_suidoubasi_ue.las"
OUTPUT_DIR = "/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
all_lines = []

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
        bitmap[row, col] = 1

    contours = find_contours(bitmap, 0.5)
    for contour in contours:
        if len(contour) < 2:
            continue
        line = []
        for r, c in contour:
            y = y_min + c * grid_res
            z = z_max - r * grid_res
            line.append([x_center, y, z])
        all_lines.append(np.array(line))

# === 輪郭線 → LineSet に変換 ===
all_pts = []
all_edges = []
offset = 0
for line in all_lines:
    all_pts.extend(line)
    edges = [[i + offset, i + 1 + offset] for i in range(len(line) - 1)]
    all_edges.extend(edges)
    offset += len(line)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(np.array(all_pts))
line_set.lines = o3d.utility.Vector2iVector(np.array(all_edges))
line_set.colors = o3d.utility.Vector3dVector(np.tile([[0, 1, 0]], (len(all_edges), 1)))

# === 保存 ===
o3d.io.write_line_set(os.path.join(OUTPUT_DIR, "0729xslice_contours.ply"), line_set)
o3d.io.write_line_set(os.path.join(OUTPUT_DIR, "0729xslice_contours.obj"), line_set)
