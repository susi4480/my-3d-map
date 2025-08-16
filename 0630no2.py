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
from pyproj import Transformer, CRS

# === 設定 ===
input_las_path = r"/output/0725_suidoubasi_floor_sita.las"
lidar_xyz_dir = r"/data/fulldata/lidar_sita_xyz/"
output_las_path = r"/output/0725_suidoubasi_sita.las"
voxel_size = 0.2
normal_wall_z_max = 3.2
floor_z_max = 1.1
horizontal_threshold = 0.90

# === [1] LAS 読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las_path)
floor_pts = np.vstack([las.x, las.y, las.z]).T
print(f"✅ LAS点数: {len(floor_pts):,}")

# === [2] LiDAR XYZ 読み込み & UTM変換 ===
print("📥 LiDAR読み込み中...")
to_utm = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)
lidar_xyz_files = glob.glob(os.path.join(lidar_xyz_dir, "*.xyz"))

lidar_points = []
for path in lidar_xyz_files:
    try:
        data = np.loadtxt(path)
        lon, lat, z = data[:, 1], data[:, 0], data[:, 2]
        x, y = to_utm.transform(lon, lat)
        lidar_points.append(np.vstack([x, y, z]).T)
    except Exception as e:
        print(f"⚠ LiDAR読み込み失敗: {path} → {e}")

if not lidar_points:
    raise RuntimeError("❌ 有効なLiDAR点群が見つかりませんでした")

lidar_pts = np.vstack(lidar_points)
print(f"✅ LiDAR点数: {len(lidar_pts):,}")

# === [3] 点群統合・ダウンサンプリング ===
combined_pts = np.vstack([floor_pts, lidar_pts])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(combined_pts)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# === [4] 法線推定と分類 ===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)
colors = np.zeros((len(points), 3), dtype=np.uint16)  # 16bit整数で格納

# 分類マスクと色（16bit: 0–65535）
colors[:] = [65535, 65535, 65535]  # 白: 未分類
colors[(normals[:, 2] < 0.3) & (points[:, 2] < normal_wall_z_max)] = [65535, 0, 0]      # 赤: 壁
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0, 0, 65535]  # 青: 床
colors[(normals[:, 2] < 0.3) & (points[:, 2] >= normal_wall_z_max)] = [65535, 65535, 0]  # 黄: ビル

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
