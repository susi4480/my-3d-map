# -*- coding: utf-8 -*-
"""
【機能】
補間済み川底LASと統合済みLiDAR LASを読み込み、
法線推定や分類・色付けなしで単純に統合し、1つのLASファイルとして出力
"""

import os
import numpy as np
import laspy
from pyproj import CRS

# === 入出力設定 ===
floor_las_path = r"/output/0821_suidoubasi_floor_ue.las"
lidar_las_path = r"/data/0821_merged_lidar_ue.las"
output_las_path = r"/output/0827_suidoubasi_ue_no_color.las"

# === [1] 川底 LAS 読み込み ===
print("📥 川底LAS読み込み中...")
las_floor = laspy.read(floor_las_path)
floor_pts = np.vstack([las_floor.x, las_floor.y, las_floor.z]).T
print(f"✅ 川底点群数: {len(floor_pts):,}")

# === [2] LiDAR LAS 読み込み ===
print("📥 LiDAR LAS読み込み中...")
las_lidar = laspy.read(lidar_las_path)
lidar_pts = np.vstack([las_lidar.x, las_lidar.y, las_lidar.z]).T
print(f"✅ LiDAR点群数: {len(lidar_pts):,}")

# === [3] 点群統合 ===
merged_pts = np.vstack([floor_pts, lidar_pts])
print(f"🔗 統合点群数: {len(merged_pts):,}")

# === [4] LAS出力（分類・色なし）===
print("💾 LAS出力中...")
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(merged_pts, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])  # mm 精度
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x = merged_pts[:, 0]
las_out.y = merged_pts[:, 1]
las_out.z = merged_pts[:, 2]

las_out.write(output_las_path)
print(f"🎉 統合LAS出力完了（分類なし）: {output_las_path}")
