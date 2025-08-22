# -*- coding: utf-8 -*-
"""
【機能】
- floor_las と lidar_las をそれぞれ法線分類（Z成分）して RGB を付与
- 両者を統合して1つのLASに出力（CRSやスケール等を保持）
"""

import laspy
import open3d as o3d
import numpy as np
import os

# ===== 入力ファイル =====
floor_las_path = r"/output/0725_suidoubasi_floor_ue.las"
lidar_las_path = r"/data/0821_merged_lidar_ue.las"
merged_output_path = r"/output/0821_merged_ue_normclassified.las"

# ===== 分類パラメータ =====
radius = 1.0
nz_thresh = 0.9


def classify_las_by_normal(las_path):
    las = laspy.read(las_path)
    coords = np.vstack([las.x, las.y, las.z]).T

    # Open3Dで法線推定
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    normals = np.asarray(pcd.normals)

    # 床/壁分類
    is_floor = normals[:, 2] > nz_thresh
    colors = np.zeros((len(coords), 3), dtype=np.uint16)
    colors[is_floor] = [0, 0, 255]     # 床（青）
    colors[~is_floor] = [255, 0, 0]    # 壁（赤）

    # RGB書き込み（元las保持）
    las.red   = colors[:, 0]
    las.green = colors[:, 1]
    las.blue  = colors[:, 2]

    return las


# ===== 個別分類処理 =====
print("floor LAS の分類処理...")
floor_las = classify_las_by_normal(floor_las_path)

print("lidar LAS の分類処理...")
lidar_las = classify_las_by_normal(lidar_las_path)

# ===== 点群統合 =====
print("分類済み点群を統合中...")
merged_points = np.concatenate([
    np.vstack([floor_las.x, floor_las.y, floor_las.z]).T,
    np.vstack([lidar_las.x, lidar_las.y, lidar_las.z]).T
], axis=0)

merged_rgb = np.concatenate([
    np.vstack([floor_las.red, floor_las.green, floor_las.blue]).T,
    np.vstack([lidar_las.red, lidar_las.green, lidar_las.blue]).T
], axis=0)

# ===== ヘッダーは floor のものをベースに使用 =====
header = floor_las.header
merged_las = laspy.LasData(header)

# ===== 統合データの格納 =====
merged_las.x = merged_points[:, 0]
merged_las.y = merged_points[:, 1]
merged_las.z = merged_points[:, 2]
merged_las.red   = merged_rgb[:, 0]
merged_las.green = merged_rgb[:, 1]
merged_las.blue  = merged_rgb[:, 2]

# ===== 出力 =====
os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)
merged_las.write(merged_output_path)
print(f"✅ 統合出力完了: {merged_output_path}")
print(f"総点数: {len(merged_points)}（floor: {len(floor_las.x)}, lidar: {len(lidar_las.x)}）")
