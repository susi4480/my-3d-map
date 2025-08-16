# -*- coding: utf-8 -*-
"""
【機能】
- Z ≤ 3.0m の点群をX方向スライス
- 各断面（YZ）を2Dビットマップ化し、壁のモルフォロジー補間
- 空間内に作れる「最大の長方形」を検出
- その長方形領域を緑点で再構成して、元の点群と統合
- LAS形式で出力
"""

import numpy as np
import laspy
from skimage.morphology import binary_closing, disk

# === 入出力設定 ===
INPUT_LAS  = "/output/0725_suidoubasi_sita.las"
OUTPUT_LAS = "/output/0728_maxrect_sita.las"

# === パラメータ ===
Z_LIMIT = 3.0
SLICE_INTERVAL = 0.5
SLICE_THICKNESS = 0.2
GRID_RES = 0.1
MORPH_RADIUS = 2

# === 最大内接長方形を求める関数（DP法）===
def find_max_rectangle(bitmap):
    h, w = bitmap.shape
    height = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)  # top, left, height, width

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
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0

# === Z上限フィルタリング ===
mask = xyz[:, 2] <= Z_LIMIT
xyz = xyz[mask]
rgb = rgb[mask]

# === スライス処理 ===
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
            grid[zi, yi] = True

    # モルフォロジー補間で閉空間化
    closed = binary_closing(grid, disk(MORPH_RADIUS))
    free = ~closed

    # 最大内接長方形の検出
    top, left, h, w = find_max_rectangle(free)
    if h == 0 or w == 0:
        continue

    # 緑点として矩形領域を3D復元
    for i in range(h):
        for j in range(w):
            y = y_min + (left + j) * GRID_RES
            z = z_min + (top + i) * GRID_RES
            x = (x0 + x1) / 2
            green_pts.append([x, y, z])

# === 緑点と元点群を統合してLAS保存 ===
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

print(f"✅ 最大内接矩形の緑点を含むLAS出力が完了しました: {OUTPUT_LAS}（緑点数: {len(green_pts)}）")
