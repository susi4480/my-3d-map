# -*- coding: utf-8 -*-
"""
【機能】
指定ディレクトリ内の .xyz ファイル（緯度・経度・高さ）をすべて読み込み、
UTM座標（Zone 54N）に変換して統合し、1つのLASファイルとして出力
"""

import os
import glob
import numpy as np
import laspy
from pyproj import Transformer, CRS

# === 入出力設定 ===
xyz_dir = r"/data/fulldata/lidar_ue_xyz/"
output_las_path = r"/data/0821_merged_lidar_ue.las"

# === 緯度経度 → UTM（Zone 54N）変換器 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

# === .xyz ファイル読み込みと変換 ===
all_points = []

xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
print(f"📂 読み込むXYZファイル数: {len(xyz_files)}")

for path in xyz_files:
    try:
        data = np.loadtxt(path)
        lon, lat, z = data[:, 1], data[:, 0], data[:, 2]  # [lat, lon, height] → [lon, lat, height]
        x, y = transformer.transform(lon, lat)
        pts = np.vstack([x, y, z]).T
        all_points.append(pts)
        print(f"✅ 読み込み成功: {os.path.basename(path)} ({len(pts):,}点)")
    except Exception as e:
        print(f"⚠ 読み込み失敗: {os.path.basename(path)} → {e}")

if not all_points:
    raise RuntimeError("❌ 有効な.xyzファイルが見つかりませんでした")

# === 点群統合 ===
merged_points = np.vstack(all_points)
print(f"🔗 統合点数: {len(merged_points):,}")

# === LAS書き出し ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(merged_points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])  # mm 精度
header.add_crs(CRS.from_epsg(32654))

las = laspy.LasData(header)
las.x = merged_points[:, 0]
las.y = merged_points[:, 1]
las.z = merged_points[:, 2]
las.red = np.full(len(merged_points), 30000, dtype=np.uint16)
las.green = np.full(len(merged_points), 30000, dtype=np.uint16)
las.blue = np.full(len(merged_points), 30000, dtype=np.uint16)

las.write(output_las_path)
print(f"🎉 LAS出力完了: {output_las_path}")
