# -*- coding: utf-8 -*-
"""
BPAで航行可能空間を3Dメッシュ化（安定化版）
"""

import os
import numpy as np
import laspy
import open3d as o3d

# 入出力
INPUT_LAS = r"/output/0817no2_M0_rect_edges_only.las"
OUTPUT_PLY = r"/output/0828mesh_bpa_fixed.ply"
os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)

# パラメータ
Z_LIMIT = 1.9
VOXEL_SIZE = 0.1       # ダウンサンプリングを粗くして安定化
BALL_RADIUS = 0.3

# LAS読み込み
las = laspy.read(INPUT_LAS)
points = np.vstack((las.x, las.y, las.z)).T
points = points[points[:, 2] <= Z_LIMIT]

# Open3D変換
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# ダウンサンプリング
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

# 法線推定（Qhull依存のorient_normals_consistentは使わない）
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
)
pcd.orient_normals_towards_camera_location(camera_location=pcd.get_center())

# BPA
radii = [BALL_RADIUS, BALL_RADIUS*2, BALL_RADIUS*4]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)

# メッシュをクリーニング
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()
mesh.compute_vertex_normals()

# 保存
o3d.io.write_triangle_mesh(OUTPUT_PLY, mesh)
print(f"✅ BPAメッシュ出力完了: {OUTPUT_PLY}")
