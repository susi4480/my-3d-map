# -*- coding: utf-8 -*-
"""
floor点群とlidar点群（両方 .las）を統合し、統計的外れ値除去（SOR）して1つのLASとして保存
"""

import laspy
import numpy as np
import open3d as o3d
import os

# === 入力ファイルパス ===
floor_las_path = r"/output/0725_suidoubasi_floor_ue.las"
lidar_las_path = r"/data/0821_merged_lidar_ue.las"
output_las_path = r"/output/0821_merged_lidar_floor_sor.las"

# === LAS読み込み（floor）===
floor_las = laspy.read(floor_las_path)
floor_points = np.vstack([floor_las.x, floor_las.y, floor_las.z]).T

# === LAS読み込み（lidar）===
lidar_las = laspy.read(lidar_las_path)
lidar_points = np.vstack([lidar_las.x, lidar_las.y, lidar_las.z]).T

# === 点群統合 ===
merged_points = np.vstack([floor_points, lidar_points])

# === Open3D形式に変換してSOR適用 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_points)

# SORパラメータ（必要に応じて調整）
NB_NEIGHBORS = 20  # 参照点数
STD_RATIO = 2.0    # 標準偏差の倍率

pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)

# === LASファイルとして出力 ===
filtered_np = np.asarray(pcd_filtered.points)
offset = filtered_np.min(axis=0)
scale = np.array([0.001, 0.001, 0.001])  # 精度

header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = offset
header.scales = scale

las = laspy.LasData(header)
las.x = filtered_np[:, 0]
las.y = filtered_np[:, 1]
las.z = filtered_np[:, 2]

# 出力先フォルダ作成
os.makedirs(os.path.dirname(output_las_path), exist_ok=True)
las.write(output_las_path)

print(f"✅ 出力完了: {output_las_path}")
print(f"✔ 入力点数: {len(merged_points):,}")
print(f"✔ 出力点数（SOR後）: {len(filtered_np):,}")
