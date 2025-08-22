# -*- coding: utf-8 -*-
"""
【機能】
補間済み川底LASと統合済みLiDAR LASを読み込み、
法線推定に基づいて「岸壁（赤）・川底（青）・ビル群（黄）」を分類してLAS出力
（ダウンサンプリングなし・統合済みLASを使用）
"""

import os
import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS

# === 入出力設定 ===
floor_las_path = r"/data/matome/0725_suidoubasi_floor_ue.las"
lidar_las_path = r"/data/matome/0821_merged_lidar_ue.las"  # ← 変更済み
output_las_path = r"/output/0821no1_02_500_suidoubasi_ue.las"

# === 分類パラメータ ===
normal_wall_z_max = 3.3
floor_z_max = 1.1
horizontal_threshold = 0.70

# === 法線推定パラメータ ===
search_radius = 0.2      # 検索半径（メートル）
max_neighbors = 500       # 最大近傍点数

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
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(combined_pts)

# === [4] 法線推定と分類 ===
print(f"📐 法線推定中...（radius={search_radius}, max_nn={max_neighbors}）")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=search_radius, max_nn=max_neighbors))

normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)
colors = np.zeros((len(points), 3), dtype=np.uint16)

# 分類マスクと色（16bit: 0–65535）
colors[:] = [65535, 65535, 65535]  # 白: 未分類
colors[(normals[:, 2] < 0.2) & (points[:, 2] < normal_wall_z_max)] = [65535, 0, 0]      # 赤: 壁
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0, 0, 65535]  # 青: 床
colors[(normals[:, 2] < 0.3) & (points[:, 2] >= normal_wall_z_max)] = [65535, 65535, 0]  # 黄: ビル

# === [5] LASとして保存 ===
print("💾 LAS出力中...")
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x = points[:, 0]
las_out.y = points[:, 1]
las_out.z = points[:, 2]
las_out.red = colors[:, 0]
las_out.green = colors[:, 1]
las_out.blue = colors[:, 2]

las_out.write(output_las_path)
print(f"🎉 分類・LAS出力完了: {output_las_path}")
