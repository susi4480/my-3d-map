# -*- coding: utf-8 -*-
"""
【機能】
- LAS点群をX方向にスライス（0.5m間隔）
- 各スライスをoccupancy grid化し、壁領域を閉じる
- 空き空間の外殻をcv2.findContoursで抽出
- 輪郭点を3Dに復元し、緑点としてLAS出力
"""

import os
import numpy as np
import laspy
import cv2
from skimage.morphology import binary_closing, disk
from tqdm import tqdm

# === 入出力設定 ===
INPUT_LAS = "/output/0725_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0729_morph_contour.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# === パラメータ ===
Z_LIMIT = 3.0
SLICE_INTERVAL = 0.5
SLICE_THICKNESS = 0.2
GRID_RES = 0.05
MORPH_RADIUS = 2  # 壁のギャップ補間用

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T
points = points[points[:, 2] <= Z_LIMIT]  # Z制限

# === スライス処理 ===
x_min, x_max = points[:, 0].min(), points[:, 0].max()
x_bins = np.arange(x_min, x_max, SLICE_INTERVAL)
green_pts = []

for x0 in tqdm(x_bins, desc="スライス処理"):
    x1 = x0 + SLICE_THICKNESS
    mask = (points[:, 0] >= x0) & (points[:, 0] < x1)
    slice_pts = points[mask]
    if len(slice_pts) < 50:
        continue

    y_vals, z_vals = slice_pts[:, 1], slice_pts[:, 2]
    y_min, y_max = y_vals.min(), y_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()
    h = int(np.ceil((z_max - z_min) / GRID_RES))
    w = int(np.ceil((y_max - y_min) / GRID_RES))
    grid = np.zeros((h, w), dtype=bool)

    for y, z in zip(y_vals, z_vals):
        col = int((y - y_min) / GRID_RES)
        row = int((z_max - z) / GRID_RES)  # z軸は上が0行目
        if 0 <= row < h and 0 <= col < w:
            grid[row, col] = True  # 壁としてマーク

    # モルフォロジー補間 → 空間部分を抽出
    closed = binary_closing(grid, disk(MORPH_RADIUS))
    free = ~closed  # 空き空間

    # cv2で輪郭抽出
    free_img = (free.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(free_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        for pt in cnt:
            col, row = pt[0]  # [x, y] = [col, row]
            y = y_min + col * GRID_RES
            z = z_max - row * GRID_RES
            x = (x0 + x1) / 2
            green_pts.append([x, y, z])

# === LAS保存（緑点のみ）
if len(green_pts) > 0:
    green_pts = np.array(green_pts)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = green_pts.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las_out = laspy.LasData(header)
    las_out.x = green_pts[:, 0]
    las_out.y = green_pts[:, 1]
    las_out.z = green_pts[:, 2]
    las_out.red = np.zeros(len(green_pts), dtype=np.uint16)
    las_out.green = np.full(len(green_pts), 65535, dtype=np.uint16)
    las_out.blue = np.zeros(len(green_pts), dtype=np.uint16)
    las_out.write(OUTPUT_LAS)
    print(f"✅ 出力完了: {OUTPUT_LAS}（緑点数: {len(green_pts)}）")
else:
    print("⚠ 緑点（外殻）が見つかりませんでした")
