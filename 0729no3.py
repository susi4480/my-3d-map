# -*- coding: utf-8 -*-
"""
【機能】
- 指定LAS点群をX方向スライス（YZ断面）
- 各スライスで Z ≤ Z_LIMIT の点を対象にビットマップ化
- 点が存在しない領域（free space）に複数の最大長方形を詰めて合成
- 合成した空間から外周線を輪郭抽出（共有辺は除去される）
- 輪郭を3D座標に復元し、PLYとOBJで保存
"""

import os
import numpy as np
import laspy
import cv2
import open3d as o3d
from tqdm import tqdm

# === 入出力設定 ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output\0725_suidoubasi_ue.las"  # ←適宜変更
OUTPUT_PLY = r"C:\Users\user\Documents\lab\output\0728_boundary_polygon.ply"
OUTPUT_OBJ = r"C:\Users\user\Documents\lab\output\0728_boundary_polygon.obj"

# === パラメータ ===
Z_LIMIT = 3.0
SLICE_INTERVAL = 0.5
SLICE_THICKNESS = 0.2
GRID_RES = 0.1
MIN_RECT_SIZE = 5  # 小さすぎる長方形を除外

# === 最大内接長方形を求める関数 ===
def find_max_rectangle(bitmap):
    h, w = bitmap.shape
    height = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)

    for i in range(h):
        for j in range(w):
            height[j] = height[j] + 1 if bitmap[i, j] else 0

        stack = []
        j = 0
        while j <= w:
            cur_height = height[j] if j < w else 0
            if not stack or cur_height >= height[stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = height[top] * width
                if area > max_area:
                    max_area = area
                    max_rect = (i - height[top] + 1, stack[-1] + 1 if stack else 0, height[top], width)
    return max_rect

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
xyz = xyz[xyz[:, 2] <= Z_LIMIT]

# === スライス処理 ===
x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
x_bins = np.arange(x_min, x_max, SLICE_INTERVAL)
all_boundary_pts = []

for x0 in tqdm(x_bins, desc="スライス処理"):
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
            grid[zi, yi] = True

    free_bitmap = ~grid
    composite = np.zeros_like(free_bitmap, dtype=np.uint8)

    while np.any(free_bitmap):
        top, left, h, w = find_max_rectangle(free_bitmap)
        if h < MIN_RECT_SIZE or w < MIN_RECT_SIZE:
            break
        composite[top:top + h, left:left + w] = 255
        free_bitmap[top:top + h, left:left + w] = False

    contours, _ = cv2.findContours(composite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        for pt in cnt:
            y = y_min + pt[0][0] * GRID_RES
            z = z_min + pt[0][1] * GRID_RES
            x = (x0 + x1) / 2
            all_boundary_pts.append([x, y, z])

# === 出力 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(all_boundary_pts))
pcd.paint_uniform_color([0, 1, 0])  # 緑色
o3d.io.write_point_cloud(OUTPUT_PLY, pcd)
o3d.io.write_point_cloud(OUTPUT_OBJ, pcd)

print("✅ 完了：スライスごとの境界抽出 → PLY/OBJ出力")
