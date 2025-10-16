# -*- coding: utf-8 -*-
"""
【機能】floor補間PLY + lidar LAS を統合し、法線推定＆分類PLY出力（ICP用）
-----------------------------------------------------------------------
1. /output/1009_floor_interp_only.ply を読み込み
2. /output/0925_lidar_sita_merged.las を読み込み
3. 統合 → ダウンサンプリング
4. 0925版方式の法線推定（向き整合なし）
5. 分類（赤=壁・青=床・黄=ビル）
6. 法線付きPLYを保存（ICP地図用）
"""

import numpy as np
import open3d as o3d
import laspy
from pyproj import CRS

# === 入出力 ===
floor_ply_path = "/workspace/output/1009_floor_interp_only.ply"
lidar_las_path = "/workspace/output/0925_lidar_sita_merged.las"
output_ply_path = "/workspace/output/1009_floor_lidar_classified_with_normals.ply"

# === パラメータ ===
down_voxel_size = 0.2
normal_wall_z_max = 3.2
floor_z_max = 1.1
horizontal_threshold = 0.6
search_radius_normals = 1.0
max_neighbors_normals = 500

# === [1] 補間済みfloor PLY 読み込み ===
print("📥 floor補間PLY読み込み中...")
pcd_floor = o3d.io.read_point_cloud(floor_ply_path)
points_floor = np.asarray(pcd_floor.points)
print(f"✅ floor点数: {len(points_floor):,}")

# === [2] LiDAR LAS読み込み ===
print("📥 lidar LAS読み込み中...")
las = laspy.read(lidar_las_path)
points_lidar = np.vstack([las.x, las.y, las.z]).T
print(f"✅ lidar点数: {len(points_lidar):,}")

# === [3] 統合 ===
points_all = np.vstack([points_floor, points_lidar])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_all)

# === [4] ダウンサンプリング ===
print("📏 ダウンサンプリング中...")
pcd = pcd.voxel_down_sample(voxel_size=down_voxel_size)
points = np.asarray(pcd.points)
print(f"✅ ダウンサンプリング後: {len(points):,}")

# === [5] 法線推定（0925方式：整合なし） ===
print("📐 法線推定中 (orient整合なし)...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius_normals, max_nn=max_neighbors_normals)
)
normals = np.asarray(pcd.normals)

# === [6] 分類 ===
colors = np.zeros((len(points), 3))
colors[:] = [1.0, 1.0, 1.0]  # 白 = 未分類
colors[(normals[:, 2] < 0.6) & (points[:, 2] < normal_wall_z_max)] = [1.0, 0.0, 0.0]  # 壁=赤
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0.0, 0.0, 1.0]  # 床=青
colors[points[:, 2] >= normal_wall_z_max] = [1.0, 1.0, 0.0]  # ビル=黄
pcd.colors = o3d.utility.Vector3dVector(colors)

# === [7] PLY出力 ===
print("💾 PLY保存中（法線込み）...")
o3d.io.write_point_cloud(output_ply_path, pcd)
print(f"🎉 出力完了: {output_ply_path} ({len(points):,} 点, 法線付き)")

