# -*- coding: utf-8 -*-
"""
【機能】LiDAR（上部構造物）＋Floor（川底）LAS統合スクリプト（Linux/Docker対応）
---------------------------------------------------------
1. LiDAR LAS と Floor LAS を読み込み
2. XYZ・intensity・color（存在すれば）を統合
3. CRS(EPSG:32654)付きLASとして出力
---------------------------------------------------------
出力例: /output/1009_merged_lidar_floor_ue.las
"""

import laspy
import numpy as np
from pyproj import CRS

# === 入出力設定 ===
lidar_path = "/data/matome/0821_merged_lidar_ue.las"
floor_path = "/data/matome/0910_merged_floor_ue.las"
output_path = "/output/1009_merged_lidar_floor_ue.las"

# === LAS読み込み関数 ===
def load_las(path):
    """LASファイルを読み込み、XYZ・Intensity・RGBを返す"""
    las = laspy.read(path)
    print(f"📥 読み込み完了: {path} ({len(las.x):,} 点)")

    # Intensity（存在しない場合は0で補う）
    if "intensity" in las.point_format.dimension_names:
        intensity = np.array(las.intensity, dtype=np.float32)
    else:
        intensity = np.zeros(len(las.x), dtype=np.float32)

    # RGB（存在しない場合は黒で補う）
    red = getattr(las, "red", np.zeros(len(las.x)))
    green = getattr(las, "green", np.zeros(len(las.x)))
    blue = getattr(las, "blue", np.zeros(len(las.x)))

    data = {
        "xyz": np.vstack([las.x, las.y, las.z]).T,
        "intensity": intensity,
        "rgb": np.vstack([red, green, blue]).T
    }
    return data

# === LAS読み込み ===
lidar = load_las(lidar_path)
floor = load_las(floor_path)

# === 統合 ===
xyz_all = np.vstack([lidar["xyz"], floor["xyz"]])
intensity_all = np.hstack([lidar["intensity"], floor["intensity"]])
rgb_all = np.vstack([lidar["rgb"], floor["rgb"]])

print(f"🧩 統合点数: {len(xyz_all):,} 点")

# === LAS出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.add_crs(CRS.from_epsg(32654))  # UTM Zone 54N（東京エリア）

las_out = laspy.LasData(header)
las_out.x = xyz_all[:, 0]
las_out.y = xyz_all[:, 1]
las_out.z = xyz_all[:, 2]
las_out.intensity = intensity_all.astype(np.uint16)
las_out.red = rgb_all[:, 0].astype(np.uint16)
las_out.green = rgb_all[:, 1].astype(np.uint16)
las_out.blue = rgb_all[:, 2].astype(np.uint16)

las_out.write(output_path)
print(f"✅ 統合LAS出力完了: {output_path}")
