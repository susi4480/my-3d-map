# -*- coding: utf-8 -*-
"""
【機能】
- YZ断面から最大空間長方形を検出（Z上限=3.0m、Z下限=川底）
- 元の点群に緑色ワイヤー（4隅+辺を点群として50分割）を加えて .ply 出力
"""

import numpy as np
import laspy
from scipy.spatial import ConvexHull, Delaunay
import open3d as o3d
import os

# === 入出力設定 ===
input_las = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_387183.00m.las"
output_ply = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_rect_with_lines.ply"
grid_res = 0.1
Z_MAX = 3.0

# === LAS読み込み（全点群）===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0  # open3dは0.0~1.0

# === Z制限で最大矩形探索用に限定 ===
limited_points = points[points[:, 2] <= Z_MAX]
if len(limited_points) == 0:
    raise ValueError("❌ 有効な点群がありません（Z < 3.0）")

Z_MIN = limited_points[:, 2].min()
yz = limited_points[:, [1, 2]]
y_min, y_max = yz[:, 0].min(), yz[:, 0].max()
z_min, z_max = Z_MIN, Z_MAX

# === グリッド化 ===
ny = int(np.ceil((y_max - y_min) / grid_res))
nz = int(np.ceil((z_max - z_min) / grid_res))
grid = np.zeros((nz, ny), dtype=np.uint8)
iy = ((yz[:, 0] - y_min) / grid_res).astype(int)
iz = ((yz[:, 1] - z_min) / grid_res).astype(int)
grid[iz, iy] = 1

# === 凸包マスク（有効領域）===
hull = ConvexHull(yz)
delaunay = Delaunay(yz[hull.vertices])
yy, zz = np.meshgrid(np.linspace(y_min, y_max, ny), np.linspace(z_min, z_max, nz))
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

# === 四隅（Y,Z）→ XYZ ===
y1 = y_min + left * grid_res
y2 = y_min + right * grid_res
z1 = z_min + top * grid_res
z2 = z_min + bottom * grid_res
x_center = np.mean(points[:, 0])
rect_corners = np.array([
    [x_center, y1, z1],
    [x_center, y2, z1],
    [x_center, y2, z2],
    [x_center, y1, z2]
])

# === 各辺を補間（50点ずつ）===
def interpolate_line(p1, p2, n=50):
    return np.linspace(p1, p2, n)

wire_points = []
for i in range(4):
    p1 = rect_corners[i]
    p2 = rect_corners[(i + 1) % 4]
    wire_points.append(interpolate_line(p1, p2, n=50))
wire_points = np.vstack(wire_points)
wire_colors = np.tile(np.array([[0, 1, 0]]), (wire_points.shape[0], 1))  # 緑

# === 元の点群をOpen3Dに変換 ===
pcd_orig = o3d.geometry.PointCloud()
pcd_orig.points = o3d.utility.Vector3dVector(points)
pcd_orig.colors = o3d.utility.Vector3dVector(colors)

# === ワイヤー点群を追加 ===
pcd_wire = o3d.geometry.PointCloud()
pcd_wire.points = o3d.utility.Vector3dVector(wire_points)
pcd_wire.colors = o3d.utility.Vector3dVector(wire_colors)

# === 結合して出力 ===
pcd_combined = pcd_orig + pcd_wire
o3d.io.write_point_cloud(output_ply, pcd_combined)
print(f"✅ PLYとして出力しました: {output_ply}")
