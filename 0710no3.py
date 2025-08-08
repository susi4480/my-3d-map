# -*- coding: utf-8 -*-
"""
【機能】
- YZ断面から最大空間長方形を検出（Z上限=3.0m、Z下限=川底）
- 最大矩形（白）と、岸壁から1m内側に縮小した矩形（緑）をワイヤー点群で可視化
- .plyとして出力（色情報付き）
"""

import numpy as np
import laspy
from scipy.spatial import ConvexHull, Delaunay
import open3d as o3d
import os

# === 入出力設定 ===
input_las = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_387183.00m.las"
output_ply = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_safezone_yonly.ply"
grid_res = 0.1
Z_MAX = 3.0
SAFE_DIST = 1.0  # 左右（Y方向）の安全距離

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0

# === Z制限ありの点群で矩形検出 ===
limited_points = points[points[:, 2] <= Z_MAX]
if len(limited_points) == 0:
    raise ValueError("❌ 有効な点群がありません（Z < 3.0）")

Z_MIN = limited_points[:, 2].min()
yz = limited_points[:, [1, 2]]
y_min, y_max = yz[:, 0].min(), yz[:, 0].max()
z_min, z_max = Z_MIN, Z_MAX

# === グリッド生成 ===
ny = int(np.ceil((y_max - y_min) / grid_res))
nz = int(np.ceil((z_max - z_min) / grid_res))
grid = np.zeros((nz, ny), dtype=np.uint8)
iy = ((yz[:, 0] - y_min) / grid_res).astype(int)
iz = ((yz[:, 1] - z_min) / grid_res).astype(int)
grid[iz, iy] = 1

# === 凸包マスク作成 ===
hull = ConvexHull(yz)
delaunay = Delaunay(yz[hull.vertices])
yy, zz = np.meshgrid(np.linspace(y_min, y_max, ny),
                     np.linspace(z_min, z_max, nz))
query_points = np.vstack([yy.ravel(), zz.ravel()]).T
mask = delaunay.find_simplex(query_points) >= 0
mask = mask.reshape(nz, ny)
space = (grid == 0) & mask

# === 最大長方形探索 ===
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

# === 最大矩形・安全矩形（Yのみ縮小）の座標 ===
x_center = np.mean(points[:, 0])
y1_raw = y_min + left * grid_res
y2_raw = y_min + right * grid_res
z1 = z_min + top * grid_res
z2 = z_min + bottom * grid_res

# 安全距離分だけYを内側へ（Zはそのまま）
y1_safe = y1_raw + SAFE_DIST
y2_safe = y2_raw - SAFE_DIST

# 最大矩形（白）
rect_max = np.array([
    [x_center, y1_raw, z1],
    [x_center, y2_raw, z1],
    [x_center, y2_raw, z2],
    [x_center, y1_raw, z2]
])

# 安全矩形（緑）
rect_safe = np.array([
    [x_center, y1_safe, z1],
    [x_center, y2_safe, z1],
    [x_center, y2_safe, z2],
    [x_center, y1_safe, z2]
])

# === 補間関数（4辺に点を50個）===
def interpolate_lines(points, n=50):
    lines = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        lines.append(np.linspace(p1, p2, n))
    return np.vstack(lines)

max_lines = interpolate_lines(rect_max)
safe_lines = interpolate_lines(rect_safe)
max_colors = np.tile([[1, 1, 1]], (max_lines.shape[0], 1))  # 白
safe_colors = np.tile([[0, 1, 0]], (safe_lines.shape[0], 1))  # 緑

# === Open3D点群結合 ===
pcd_orig = o3d.geometry.PointCloud()
pcd_orig.points = o3d.utility.Vector3dVector(points)
pcd_orig.colors = o3d.utility.Vector3dVector(colors)

pcd_max = o3d.geometry.PointCloud()
pcd_max.points = o3d.utility.Vector3dVector(max_lines)
pcd_max.colors = o3d.utility.Vector3dVector(max_colors)

pcd_safe = o3d.geometry.PointCloud()
pcd_safe.points = o3d.utility.Vector3dVector(safe_lines)
pcd_safe.colors = o3d.utility.Vector3dVector(safe_colors)

# === 結合して保存 ===
pcd_all = pcd_orig + pcd_max + pcd_safe
o3d.io.write_point_cloud(output_ply, pcd_all)
print(f"✅ 出力完了: {output_ply}")
