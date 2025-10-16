# -*- coding: utf-8 -*-
"""
【機能】
- floorのLAS点群をダウンサンプリング
- lidar点群と統合
- Open3Dで法線推定
- PLYとして保存（必要ならLAS出力も可能）
"""

import laspy
import numpy as np
import open3d as o3d
import os

# === 入力ファイル ===
floor_las_path = r"/output/0725_suidoubasi_floor_ue.las"
lidar_las_path = r"/data/0821_merged_lidar_ue.las"

# === 出力ファイル ===
output_ply_path = r"/output/0821no11_floor_lidar_normals_merged.ply"
voxel_size = 0.1  # floorのダウンサンプリングサイズ
normal_radius = 0.3  # 法線推定用の近傍半径

# === LAS読み込み関数（Open3D用PointCloudに変換） ===
def las_to_o3d_pointcloud(las_path):
    las = laspy.read(las_path)
    points = np.vstack([las.x, las.y, las.z]).T

    # RGBがあればカラー付き
    if hasattr(las, 'red'):
        colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
    else:
        colors = np.zeros_like(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# === 1. floor点群読み込み＋ダウンサンプリング ===
print("📥 floor読み込み＋ダウンサンプリング中...")
pcd_floor = las_to_o3d_pointcloud(floor_las_path)
pcd_floor_down = pcd_floor.voxel_down_sample(voxel_size=voxel_size)

# === 2. lidar点群読み込み（そのまま） ===
print("📥 lidar読み込み中...")
pcd_lidar = las_to_o3d_pointcloud(lidar_las_path)

# === 3. 統合 ===
print("🔗 点群を統合中...")
pcd_merged = pcd_floor_down + pcd_lidar

# === 4. 法線推定 ===
print("📏 法線推定中...")
pcd_merged.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
)
pcd_merged.orient_normals_consistent_tangent_plane(30)

# === 5. 出力 ===
print(f"💾 出力中... {output_ply_path}")
o3d.io.write_point_cloud(output_ply_path, pcd_merged)
print("✅ 完了！")
