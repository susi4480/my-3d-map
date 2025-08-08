# -*- coding: utf-8 -*-
"""
【機能】
- Z ≤ 3.0m の点群をX方向スライス
- 各スライスで occupancy grid を構築
- モルフォロジーで壁の隙間を補間（閉領域）
- 空間領域に最大長方形を詰めて合成
- 合成領域の外殻を cv2.findContours で抽出
- 緑点として3Dに復元し、LASで出力
"""

import os
import numpy as np
import laspy
import cv2
from tqdm import tqdm
from skimage.morphology import binary_closing, disk

# === 入出力設定 ===
INPUT_LAS  = "/output/0725_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0730_morph_rect_contour.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# === パラメータ ===
Z_LIMIT = 3.0
SLICE_INTERVAL = 0.5
SLICE_THICKNESS = 0.2
GRID_RES = 0.05
MORPH_RADIUS = 2
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
green_pts = []

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
            grid[zi, yi] = True  # 壁

    # モルフォロジー補間（閉空間化）
    closed = binary_closing(grid, disk(MORPH_RADIUS))
    free = ~closed

    # 長方形詰込み（空間領域）
    composite = np.zeros_like(free, dtype=np.uint8)
    free_bitmap = free.copy()

    while np.any(free_bitmap):
        top, left, h, w = find_max_rectangle(free_bitmap)
        if h < MIN_RECT_SIZE or w < MIN_RECT_SIZE:
            break
        composite[top:top+h, left:left+w] = 255
        free_bitmap[top:top+h, left:left+w] = False

    # 輪郭抽出（合成空間の外周）
    contours, _ = cv2.findContours(composite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        for pt in cnt:
            col, row = pt[0]
            y = y_min + col * GRID_RES
            z = z_min + row * GRID_RES
            x = (x0 + x1) / 2
            green_pts.append([x, y, z])

# === LAS保存 ===
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
    print("⚠ 外殻が見つかりませんでした")
