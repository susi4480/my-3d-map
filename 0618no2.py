# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d
import laspy

# === 入出力 ===
input_file = "/home/edu3/lab/data/0611_las2_full.las"
output_dir = "/home/edu3/lab/output_strategy"
os.makedirs(output_dir, exist_ok=True)

# === LAS読み込み
print("[BPA] 点群読み込み中...")
las = laspy.read(input_file)
points = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 元の点数: {len(points):,}")

# === Open3D形式に変換
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# === ダウンサンプリング
voxel_size = 0.5
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"✅ ダウンサンプリング後点数: {len(pcd.points):,}")

# === 法線推定（整列なし）
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
# ※ orient_normals_consistent_tangent_plane は使わない

# === BPAメッシュ生成
print("[BPA] メッシュ生成中...")
radii = [0.3, 0.5]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector(radii)
)

# === メッシュ出力
out_path = os.path.join(output_dir, "ball_pivoting_las2_full.ply")
o3d.io.write_triangle_mesh(out_path, mesh)
print(f"🎉 完了: メッシュ出力 -> {out_path}")
