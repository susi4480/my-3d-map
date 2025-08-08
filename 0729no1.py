# -*- coding: utf-8 -*-
"""
【機能】
- LAS点群をX方向にスライス（0.5m間隔）
- 各YZ断面をoccupancy grid化し、空き空間（0）の最大領域を抽出
- 輪郭を抽出し、B-splineで滑らかに近似
- 曲線上の点を航行可能空間（緑点）としてLAS出力
"""

import os
import numpy as np
import laspy
import cv2
from scipy import interpolate
from scipy.ndimage import label

# === 設定 ===
INPUT_LAS  = r"/output/0725_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0728_bsplinespace.las"
X_INTERVAL = 0.5
GRID_RES   = 0.05
Z_LIMIT    = 3.0

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === 緑点（航行可能空間）格納用 ===
navigable_pts = []

x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
x_slices = np.arange(x_min, x_max + X_INTERVAL, X_INTERVAL)

for x_center in x_slices:
    mask = (points[:, 0] >= x_center - X_INTERVAL/2) & (points[:, 0] < x_center + X_INTERVAL/2)
    slice_pts = points[mask]
    if slice_pts.shape[0] < 10:
        continue
    # Z制限
    slice_pts = slice_pts[slice_pts[:, 2] <= Z_LIMIT]

    # YZビットマップ化
    y_vals = slice_pts[:, 1]
    z_vals = slice_pts[:, 2]
    y_min, y_max = y_vals.min(), y_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()
    h = int(np.ceil((y_max - y_min) / GRID_RES)) + 1
    w = int(np.ceil((z_max - z_min) / GRID_RES)) + 1
    bitmap = np.ones((h, w), dtype=np.uint8) * 255

    for y, z in zip(y_vals, z_vals):
        i = int((y - y_min) / GRID_RES)
        j = int((z - z_min) / GRID_RES)
        bitmap[i, j] = 0

    # ラベリング + 最大空間抽出
    inverted = (bitmap == 255).astype(np.uint8)
    labeled, num = label(inverted)
    if num == 0:
        continue
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    max_label = np.argmax(counts)
    mask = (labeled == max_label).astype(np.uint8)

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        continue

    contour = contours[0][:, 0, :]  # [N,2] 形式
    if len(contour) < 10:
        continue

    # B-splineで近似
    contour = contour.astype(float)
    try:
        tck, u = interpolate.splprep([contour[:, 0], contour[:, 1]], s=3.0)
        u_fine = np.linspace(0, 1, 100)
        x_bs, y_bs = interpolate.splev(u_fine, tck)
    except Exception as e:
        print(f"⚠ B-spline失敗（スキップ）: {e}")
        continue

    # 緑点として保存
    for yi, zi in zip(x_bs, y_bs):
        y = yi * GRID_RES + y_min
        z = zi * GRID_RES + z_min
        navigable_pts.append([x_center, y, z])

# === LAS出力 ===
if len(navigable_pts) > 0:
    navigable_pts = np.array(navigable_pts)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = navigable_pts.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = navigable_pts[:, 0], navigable_pts[:, 1], navigable_pts[:, 2]
    las_out.red = np.zeros(len(navigable_pts), dtype=np.uint16)
    las_out.green = np.full(len(navigable_pts), 65535, dtype=np.uint16)
    las_out.blue = np.zeros(len(navigable_pts), dtype=np.uint16)
    las_out.write(OUTPUT_LAS)
    print(f"✅ 出力完了: {OUTPUT_LAS}")
else:
    print("⚠ 航行可能空間が見つかりませんでした")
