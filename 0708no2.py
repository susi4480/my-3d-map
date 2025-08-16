# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをX方向に10mずつ分割
- 各ブロックに対して上面と下面を抽出し、それぞれBPAでメッシュ化
- 最終的に統合し、PLYおよびLASで保存
"""

import numpy as np
import open3d as o3d
import laspy
import os
from pyproj import CRS

# === 入出力設定 ===
input_las = "/output/0707_green_only_ue.las"
output_ply = "/output/0708no1_split_bpa_combined.ply"
output_las = "/output/0708no1_split_bpa_combined.las"
os.makedirs(os.path.dirname(output_ply), exist_ok=True)
crs_utm = CRS.from_epsg(32654)

# === パラメータ ===
x_step = 10.0
z_split_threshold = 2.8
bpa_radii = [0.3, 0.6, 1.2]

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === Xの範囲で分割 ===
x_min, x_max = points[:, 0].min(), points[:, 0].max()
x_bins = np.arange(x_min, x_max + x_step, x_step)

all_vertices = []
all_triangles = []
offset = 0

def mesh_from_points(pts, radii):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return mesh

# === 各スライスで処理 ===
for i in range(len(x_bins) - 1):
    x_start = x_bins[i]
    x_end = x_bins[i + 1]
    mask = (points[:, 0] >= x_start) & (points[:, 0] < x_end)
    block_pts = points[mask]

    if len(block_pts) < 100:
        continue

    upper_pts = block_pts[block_pts[:, 2] >= z_split_threshold]
    lower_pts = block_pts[block_pts[:, 2] < z_split_threshold]

    print(f"▶ Block {i+1}/{len(x_bins)-1} | 点数: {len(block_pts)} (上: {len(upper_pts)}, 下: {len(lower_pts)})")

    if len(upper_pts) > 50:
        mesh_upper = mesh_from_points(upper_pts, bpa_radii)
        v = np.asarray(mesh_upper.vertices)
        t = np.asarray(mesh_upper.triangles) + offset
        all_vertices.append(v)
        all_triangles.append(t)
        offset += len(v)

    if len(lower_pts) > 50:
        mesh_lower = mesh_from_points(lower_pts, bpa_radii)
        v = np.asarray(mesh_lower.vertices)
        t = np.asarray(mesh_lower.triangles) + offset
        all_vertices.append(v)
        all_triangles.append(t)
        offset += len(v)

# === メッシュ統合 ===
if len(all_vertices) == 0:
    raise RuntimeError("❌ メッシュ生成に失敗しました。")

vertices = np.vstack(all_vertices)
triangles = np.vstack(all_triangles)

mesh_all = o3d.geometry.TriangleMesh()
mesh_all.vertices = o3d.utility.Vector3dVector(vertices)
mesh_all.triangles = o3d.utility.Vector3iVector(triangles)
mesh_all.vertex_colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (len(vertices), 1)))

# === 保存（PLY）===
o3d.io.write_triangle_mesh(output_ply, mesh_all)
print(f"✅ メッシュ出力完了: {output_ply}")

# === LAS形式で出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = vertices.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x = vertices[:, 0]
las_out.y = vertices[:, 1]
las_out.z = vertices[:, 2]
las_out.red   = np.zeros(len(vertices), dtype=np.uint16)
las_out.green = np.full(len(vertices), 255, dtype=np.uint16)
las_out.blue  = np.zeros(len(vertices), dtype=np.uint16)
las_out.write(output_las)
print(f"✅ LAS出力完了: {output_las}")
