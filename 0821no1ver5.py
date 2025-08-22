# -*- coding: utf-8 -*-
"""
【機能】
補間済み川底LASと統合済みLiDAR LASを読み込み、
Z制限（normal_wall_z_max）より上の点は法線推定せずビル群（黄）として分類、
それ以外の点は法線推定して「岸壁（赤）・川底（青）」を分類してLAS出力
（ダウンサンプリングなし・統合済みLASを使用）
"""

import os
import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS

# === 入出力設定 ===
floor_las_path = r"/data/matome/0725_suidoubasi_floor_ue.las"
lidar_las_path = r"/data/matome/0821_merged_lidar_ue.las"
output_las_path = r"/output/0821no1_06_30_suidoubasi_ue.las"

# === 分類パラメータ ===
normal_wall_z_max = 3.3
floor_z_max = 1.1
horizontal_threshold = 0.70

# === 法線推定パラメータ ===
search_radius = 0.6
max_neighbors = 30

# === [1] 補間済み floor LAS 読み込み ===
print("📥 川底LAS読み込み中...")
las_floor = laspy.read(floor_las_path)
floor_pts = np.vstack([las_floor.x, las_floor.y, las_floor.z]).T
print(f"✅ 川底点群数: {len(floor_pts):,}")

# === [2] 統合済みLiDAR LAS 読み込み ===
print("📥 LiDAR LAS読み込み中...")
las_lidar = laspy.read(lidar_las_path)
lidar_pts = np.vstack([las_lidar.x, las_lidar.y, las_lidar.z]).T
print(f"✅ LiDAR点群数: {len(lidar_pts):,}")

# === [3] 点群統合（ダウンサンプリングなし） ===
combined_pts = np.vstack([floor_pts, lidar_pts])
z_vals = combined_pts[:, 2]

# === [4] Z > normal_wall_z_max は法線推定せずビル群として処理 ===
is_high = z_vals > normal_wall_z_max
is_low = ~is_high

high_pts = combined_pts[is_high]
low_pts = combined_pts[is_low]

print(f"🔹 法線推定対象: {len(low_pts):,} 点")
print(f"🔸 ビル分類済み（Z > {normal_wall_z_max}）: {len(high_pts):,} 点")

# === [5] 法線推定（Z制限以下の点のみ） ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(low_pts)

print(f"📐 法線推定中...（radius={search_radius}, max_nn={max_neighbors}）")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=search_radius, max_nn=max_neighbors))
normals = np.asarray(pcd.normals)

# === [6] カラー分類 ===
low_colors = np.full((len(low_pts), 3), fill_value=65535, dtype=np.uint16)  # 白: 未分類

# 壁（赤）
low_colors[(normals[:, 2] < 0.3) & (low_pts[:, 2] < normal_wall_z_max)] = [65535, 0, 0]

# 床（青）
low_colors[(normals[:, 2] > horizontal_threshold) & (low_pts[:, 2] < floor_z_max)] = [0, 0, 65535]

# ビル群（黄） ← 高さだけで分類
high_colors = np.full((len(high_pts), 3), [65535, 65535, 0], dtype=np.uint16)

# === [7] 結合してLAS保存 ===
all_points = np.vstack([low_pts, high_pts])
all_colors = np.vstack([low_colors, high_colors])

print("💾 LAS出力中...")
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(all_points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x = all_points[:, 0]
las_out.y = all_points[:, 1]
las_out.z = all_points[:, 2]
las_out.red = all_colors[:, 0]
las_out.green = all_colors[:, 1]
las_out.blue = all_colors[:, 2]

las_out.write(output_las_path)
print(f"🎉 分類・LAS出力完了: {output_las_path}")
