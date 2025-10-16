# -*- coding: utf-8 -*-
"""
A版コード（ダウンサンプリングなし・orientなし）
【機能】
- 補間済み川底LASとLiDAR LASを統合
- orient_normals_consistent_tangent_plane() を使用しない
- ダウンサンプリングなし
- 法線Z成分と高さに基づき「床（青）」「壁（赤）」「ビル（黄）」に分類
- LAS出力
"""

import os
import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS

# === 入出力設定 ===
floor_las_path = r"/output/0821_suidoubasi_floor_sita.las"  # 補間済み川底LAS
lidar_las_path = r"/data/0821_merged_lidar_sita.las"         # LiDAR統合LAS
output_las_path = r"/output/0823full_sita_suidoubasi__classified.las"

# === 分類パラメータ ===
normal_wall_z_max = 3.2         # 壁とビルの境界高さ
floor_z_max = 1.1               # 床の上限高さ
horizontal_threshold = 0.7      # 床判定の法線Zしきい値
vertical_threshold = 0.3        # 壁判定の法線Zしきい値

# === 法線推定パラメータ ===
search_radius = 1.0             # 法線推定の近傍半径
max_neighbors = 100              # 法線推定の最大近傍点数

# === [1] 補間済みLAS読み込み ===
print("📥 補間済み川底LAS読み込み中...")
las_floor = laspy.read(floor_las_path)
floor_pts = np.vstack([las_floor.x, las_floor.y, las_floor.z]).T
print(f"✅ 川底点数: {len(floor_pts):,}")

# === [2] LiDAR LAS読み込み ===
print("📥 LiDAR LAS読み込み中...")
las_lidar = laspy.read(lidar_las_path)
lidar_pts = np.vstack([las_lidar.x, las_lidar.y, las_lidar.z]).T
print(f"✅ LiDAR点数: {len(lidar_pts):,}")

# === [3] 点群統合（ダウンサンプリングなし） ===
print("🔗 点群統合中...")
combined_pts = np.vstack([floor_pts, lidar_pts])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(combined_pts)

# === [4] 法線推定（orientなし） ===
print(f"📐 法線推定中... (半径={search_radius}, max_nn={max_neighbors})")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius,
        max_nn=max_neighbors
    )
)
normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)

# === [5] 分類（RGB: 16bitカラー） ===
print("🎨 分類中...")
colors = np.full((len(points), 3), [65535, 65535, 65535], dtype=np.uint16)  # 初期色：白

# 壁（赤）
colors[(normals[:, 2] < vertical_threshold) & (points[:, 2] < normal_wall_z_max)] = [65535, 0, 0]

# 床（青）
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0, 0, 65535]

# ビル群（黄）
colors[points[:, 2] >= normal_wall_z_max] = [65535, 65535, 0]

# === [6] LAS保存 ===
print("💾 LAS出力中...")
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])  # mm精度
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x = points[:, 0]
las_out.y = points[:, 1]
las_out.z = points[:, 2]
las_out.red = colors[:, 0]
las_out.green = colors[:, 1]
las_out.blue = colors[:, 2]
las_out.write(output_las_path)

print(f"🎉 A版分類・LAS出力完了: {output_las_path}")
