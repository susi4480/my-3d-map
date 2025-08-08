# -*- coding: utf-8 -*-
"""
【機能】
- YZ断面の点群からZ上限=3.0m・Z下限=川底の範囲で最大空間長方形を検出
- 四角形の4点とその辺（線分）を緑色で .las ファイルに出力
"""

import numpy as np
import laspy
from scipy.spatial import ConvexHull, Delaunay
import os

# === 入出力設定 ===
input_las = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_387183.00m.las"
output_las_path = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_rect.las"
grid_res = 0.1
Z_MAX = 3.0

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === Z制限（Z ≤ Z_MAX）===
points = points[points[:, 2] <= Z_MAX]
if len(points) == 0:
    raise ValueError("❌ 有効な点群がありません（Z < 3.0）")

Z_MIN = points[:, 2].min()
yz = points[:, [1, 2]]  # Y, Z
y_min, y_max = yz[:, 0].min(), yz[:, 0].max()
z_min, z_max = Z_MIN, Z_MAX

# === グリッド生成 ===
ny = int(np.ceil((y_max - y_min) / grid_res))
nz = int(np.ceil((z_max - z_min) / grid_res))
grid = np.zeros((nz, ny), dtype=np.uint8)

# occupied セルをマーク
iy = ((yz[:, 0] - y_min) / grid_res).astype(int)
iz = ((yz[:, 1] - z_min) / grid_res).astype(int)
grid[iz, iy] = 1

# === 外形マスク作成 ===
hull = ConvexHull(yz)
delaunay = Delaunay(yz[hull.vertices])
yy, zz = np.meshgrid(np.linspace(y_min, y_max, ny),
                     np.linspace(z_min, z_max, nz))
query_points = np.vstack([yy.ravel(), zz.ravel()]).T
mask = delaunay.find_simplex(query_points) >= 0
mask = mask.reshape(nz, ny)

# === 空間マップ生成（点がない＋外形内）===
space = (grid == 0) & mask

# === 最大長方形検出（DP）===
def max_rectangle_area(matrix):
    rows, cols = matrix.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)
    height = [0] * cols

    for r in range(rows):
        for c in range(cols):
            height[c] = 0 if matrix[r][c] == 0 else height[c] + 1

        stack = []
        for c in range(cols + 1):
            while stack and (c == cols or height[stack[-1]] > height[c]):
                h = height[stack.pop()]
                w = c if not stack else c - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    right = c
                    left = c - w
                    bottom = r
                    top = r - h + 1
                    best_rect = (top, left, bottom, right)
            stack.append(c)
    return best_rect

top, left, bottom, right = max_rectangle_area(space)

# === 長方形の4頂点（Y,Z）→ 実座標 ===
y1 = y_min + left * grid_res
y2 = y_min + right * grid_res
z1 = z_min + top * grid_res
z2 = z_min + bottom * grid_res

# === 四角形の点（順：左下 → 右下 → 右上 → 左上）===
rect_corners_yz = np.array([[y1, z1], [y2, z1], [y2, z2], [y1, z2]])
# X座標は元ファイルの中央Xを使用
x_center = np.mean(las.x)
rect_corners_xyz = np.column_stack([np.full(4, x_center), rect_corners_yz])

# === 辺の線分点生成（各辺に50点）===
def interpolate_line(p1, p2, n=50):
    return np.linspace(p1, p2, n)

line_points = []
for i in range(4):
    p1 = rect_corners_xyz[i]
    p2 = rect_corners_xyz[(i + 1) % 4]
    line_points.append(interpolate_line(p1, p2, n=50))
line_points = np.vstack(line_points)

# === 線色（緑）===
line_colors = np.tile(np.array([[0, 255, 0]]), (line_points.shape[0], 1))

# === 元の点群に追加（色はそのまま）===
orig_points = np.vstack([las.x, las.y, las.z]).T
orig_colors = np.vstack([las.red, las.green, las.blue]).T

# === 結合 ===
all_points = np.vstack([orig_points, line_points])
all_colors = np.vstack([orig_colors, line_colors])

# === LAS保存用 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(all_points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
out_las = laspy.LasData(header)

out_las.x, out_las.y, out_las.z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
out_las.red, out_las.green, out_las.blue = all_colors[:, 0], all_colors[:, 1], all_colors[:, 2]

out_las.write(output_las_path)

# === 出力 ===
print(f"✅ 出力完了: {output_las_path}")
print("✅ 最大長方形の4頂点 (Y, Z):")
for i, (y, z) in enumerate(rect_corners_yz):
    print(f"Corner {i+1}: ({y:.3f}, {z:.3f})")
