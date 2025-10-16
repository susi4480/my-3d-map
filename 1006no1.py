# -*- coding: utf-8 -*-
"""
【機能】floorとlidarの統合LAS生成（全点白色・ダウンサンプリング・法線推定なし）
-------------------------------------------------------------------
1. /output/0925_floor_ue_merged.las と /output/0925_lidar_ue_merged.las を読み込み
2. 2つのLASを単純に結合
3. intensityも結合して保持
4. 全点を白(R=G=B=65535)で出力
-------------------------------------------------------------------
出力: /output/0925_ue_merged_white.las
"""

import laspy
import numpy as np
from pyproj import CRS

# === 入出力 ===
input_floor_las = r"/output/0925_floor_ue_merged.las"
input_lidar_las = r"/output/0925_lidar_ue_merged.las"
output_merged_las = r"/output/0925_ue_merged_white.las"

# === LAS読み込み ===
print("📥 floor LAS 読み込み中...")
floor_las = laspy.read(input_floor_las)
floor_points = np.vstack([floor_las.x, floor_las.y, floor_las.z]).T
floor_intensity = np.array(floor_las.intensity, dtype=np.uint16)

print("📥 lidar LAS 読み込み中...")
lidar_las = laspy.read(input_lidar_las)
lidar_points = np.vstack([lidar_las.x, lidar_las.y, lidar_las.z]).T
lidar_intensity = np.array(lidar_las.intensity, dtype=np.uint16)

# === 結合 ===
print("🔗 統合中...")
merged_points = np.vstack([floor_points, lidar_points])
merged_intensity = np.hstack([floor_intensity, lidar_intensity])

# === 全点白色に設定 ===
merged_color = np.full((len(merged_points), 3), 65535, dtype=np.uint16)

# === LAS出力 ===
print("💾 LAS出力中...")
header = laspy.LasHeader(point_format=3, version="1.2")
header.add_crs(CRS.from_epsg(32654))
las_out = laspy.LasData(header)
las_out.x = merged_points[:, 0]
las_out.y = merged_points[:, 1]
las_out.z = merged_points[:, 2]
las_out.intensity = merged_intensity
las_out.red = merged_color[:, 0]
las_out.green = merged_color[:, 1]
las_out.blue = merged_color[:, 2]
las_out.write(output_merged_las)

print(f"🤍 全点白色で統合完了: {output_merged_las} ({len(merged_points):,} 点)")
