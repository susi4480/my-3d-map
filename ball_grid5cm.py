# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d

# === 入出力設定 ===
input_file = "/home/edu3/lab/data/pond/merged_pond.xyz"
output_dir = "/home/edu3/lab/output_strategy"
os.makedirs(output_dir, exist_ok=True)

# === 点群読み込み ===
print("[BPA] 点群読み込み中...")
pcd = o3d.io.read_point_cloud(input_file)
print(f"✅ 元の点数: {len(pcd.points):,}")

# === 軽いダウンサンプリング ===
voxel_size = 0.05  # ← 点数を多く残す（5 cm グリッド）
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"✅ ダウンサンプリング後点数: {len(pcd.points):,}")

# === 法線推定と整列 ===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(10)

# === BPAメッシュ生成 ===
print("[BPA] メッシュ生成中...")
radii = [0.3, 0.5]  # 少し細かめの球で転がす
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector(radii)
)

# === 出力 ===
out_path = os.path.join(output_dir, "ball_pivoting_fine.ply")
o3d.io.write_triangle_mesh(out_path, mesh)
print(f"🎉 完了: メッシュ出力 -> {out_path}")
