# -*- coding: utf-8 -*-
"""
【機能】
floor（補間済み）とLiDARのXYZ点群を統合し、
法線推定により「岸壁（赤）・川底（青）・ビル群（黄）」を分類し、PLY形式で保存
"""

import os
import glob
import numpy as np
import open3d as o3d
from pyproj import Transformer

# === 設定 ===
floor_xyz_dir = r"/data/las2_xyz/floor/"
lidar_xyz_dir = r"/data/las2_xyz/lidar/"
output_ply_path = r"/output/0720_suidoubasi.ply"

voxel_size = 0.2
normal_wall_z_max = 4.5
floor_z_max = 3.2
horizontal_threshold = 0.90

# === 座標変換器（緯度経度 → UTM Zone 54N）===
to_utm = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

# === [1] floor点群読み込み ===
floor_files = glob.glob(os.path.join(floor_xyz_dir, "*.xyz"))
floor_points = []

for path in floor_files:
    try:
        data = np.loadtxt(path)
        lon, lat, z = data[:, 1], data[:, 0], data[:, 2]
        x, y = to_utm.transform(lon, lat)
        floor_points.append(np.vstack([x, y, z]).T)
    except Exception as e:
        print(f"⚠ floor読み込み失敗: {path} → {e}")

if not floor_points:
    raise RuntimeError("❌ floor点群が見つかりません")

floor_pts = np.vstack(floor_points)
print(f"✅ floor点数: {len(floor_pts):,}")

# === [2] LiDAR点群読み込み ===
lidar_files = glob.glob(os.path.join(lidar_xyz_dir, "*.xyz"))
lidar_points = []

for path in lidar_files:
    try:
        data = np.loadtxt(path)
        lon, lat, z = data[:, 1], data[:, 0], data[:, 2]
        x, y = to_utm.transform(lon, lat)
        lidar_points.append(np.vstack([x, y, z]).T)
    except Exception as e:
        print(f"⚠ LiDAR読み込み失敗: {path} → {e}")

if not lidar_points:
    raise RuntimeError("❌ LiDAR点群が見つかりません")

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

# 初期は白（未分類）
colors = np.ones((len(points), 3), dtype=np.float32)

# 壁（赤）
mask_wall = (normals[:, 2] < 0.3) & (points[:, 2] < normal_wall_z_max)
colors[mask_wall] = [1.0, 0.0, 0.0]

# 川底（青）
mask_floor = (normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)
colors[mask_floor] = [0.0, 0.0, 1.0]

# ビル群（黄）
mask_building = (normals[:, 2] < 0.3) & (points[:, 2] >= normal_wall_z_max)
colors[mask_building] = [1.0, 1.0, 0.0]

# === [5] 出力 ===
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(output_ply_path, pcd)
print(f"🎉 分類・PLY出力完了: {output_ply_path}")
