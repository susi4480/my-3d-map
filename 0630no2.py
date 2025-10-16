# -*- coding: utf-8 -*-
"""
【機能】
補間済み川底LASとLiDARのXYZ点群を統合し、
法線推定に基づいて「岸壁（赤）・川底（青）・ビル群（黄）」を分類してLAS出力
"""

import os
import glob
import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS

# === 設定 ===
input_las_path = r"/workspace/output/0919_floor_only_interp.las"  # 補間済み川底LAS
lidar_xyz_dir = r"/workspace/output/0919_lidar_sita_merged_raw.las"       # LiDAR統合LAS
output_las_path = r"/output/0929_01_500_suidoubasi_ue.las"
voxel_size = 0.1
normal_wall_z_max = 3.2
floor_z_max = 1.1
horizontal_threshold = 0.6

# === [1] LAS 読み込み ===
print("📥 川底LAS読み込み中...")
las = laspy.read(input_las_path)
floor_pts = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 川底LAS点数: {len(floor_pts):,}")

# === [2] LiDAR LAS 読み込み ===
print("📥 LiDAR LAS読み込み中...")
lidar_las = laspy.read(lidar_xyz_dir)
lidar_pts = np.vstack([lidar_las.x, lidar_las.y, lidar_las.z]).T
print(f"✅ LiDAR点数: {len(lidar_pts):,}")

# === [3] 点群統合・ダウンサンプリング ===
combined_pts = np.vstack([floor_pts, lidar_pts])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(combined_pts)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# === [4] 法線推定と分類 ===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=500))
normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)
colors = np.zeros((len(points), 3), dtype=np.uint16)  # 16bit整数で格納

# 分類マスクと色（16bit: 0–65535）
colors[:] = [65535, 65535, 65535]  # 白: 未分類
colors[(normals[:, 2] < 0.6) & (points[:, 2] < normal_wall_z_max)] = [65535, 0, 0]      # 赤: 壁
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0, 0, 65535]  # 青: 床
colors[points[:, 2] >= normal_wall_z_max] = [65535, 65535, 0]  # 黄: ビル（高さのみ判定に変更）

# === [5] LASとして保存 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])  # mm 精度
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
