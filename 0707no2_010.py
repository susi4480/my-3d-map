# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルから緑の点群を読み込み（航行可能空間）
- 軽いダウンサンプリング（voxel_size=0.1）
- 法線推定＆整列
- Ball Pivoting Algorithm（BPA）でメッシュ化
- PLYとして保存
"""

import os
import numpy as np
import open3d as o3d
import laspy

# === 入出力設定 ===
input_las = "/output/0707_green_only_ue.las"
output_dir = "/output"
os.makedirs(output_dir, exist_ok=True)
output_ply = os.path.join(output_dir, "0707_mesh_bpa_green_only.ply")

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).astype(np.float32).T
colors = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T
print(f"✅ 点数: {len(points):,}")

# === Open3D点群作成 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

# === 軽いダウンサンプリング（0.10m）===
voxel_size = 0.10
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"✅ ダウンサンプリング後点数: {len(pcd.points):,}")

# === 法線推定と整列 ===
print("📏 法線推定中...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
print("🔄 法線整列中...")
#pcd.orient_normals_consistent_tangent_plane(10)

# === BPAメッシュ生成 ===
print("🔧 BPAメッシュ生成中...")
radii = [0.3, 0.5]  # 複数スケールで球を転がす
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector(radii)
)

# === 出力 ===
o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 完了: メッシュ出力 -> {output_ply}")
