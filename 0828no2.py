# -*- coding: utf-8 -*-
"""
M5: 3D占有ボクセル接続 + SOR付き
- LAS点群を読み込み
- SORでノイズ除去
- Occupancyボクセルを作成
- 連結成分ラベリングで航行可能空間抽出
- 緑点としてLAS出力
"""

import os
import numpy as np
import laspy
from scipy import ndimage
import open3d as o3d

# === 入出力 ===
input_las = r"/output/0731_suidoubasi_ue.las"
output_las = r"/output/M5_voxel_connected_green.las"

# === パラメータ ===
voxel_size = 0.2
z_limit = 1.9
sor_neighbors = 50
sor_std_ratio = 1.0

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 元の点数: {len(points):,}")

# === [1] SORでノイズ除去 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
points = np.asarray(pcd.points)
print(f"✅ SOR後の点数: {len(points):,}")

# === [2] Z ≤ 1.9m でフィルタリング ===
points = points[points[:, 2] <= z_limit]
print(f"✅ Z制限後の点数: {len(points):,}")

# === [3] Occupancyボクセル生成 ===
min_pt = points.min(axis=0)
coords = np.floor((points - min_pt) / voxel_size).astype(int)
grid_shape = coords.max(axis=0) + 1
grid = np.zeros(grid_shape, dtype=bool)
grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True

# === [4] 3D連結成分ラベリング ===
labeled, num_features = ndimage.label(grid)
print(f"✅ ラベル数: {num_features}")

# === [5] 航行可能空間（最大連結成分を抽出）===
label_counts = np.bincount(labeled.ravel())
largest_label = label_counts[1:].argmax() + 1
navigable_mask = labeled == largest_label
navigable_coords = np.argwhere(navigable_mask)
navigable_points = navigable_coords * voxel_size + min_pt

# === [6] 緑色でLAS出力 ===
colors = np.full((len(navigable_points), 3), [0, 65535, 0], dtype=np.uint16)

header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = navigable_points.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
las_out = laspy.LasData(header)
las_out.x = navigable_points[:, 0]
las_out.y = navigable_points[:, 1]
las_out.z = navigable_points[:, 2]
las_out.red = colors[:, 0]
las_out.green = colors[:, 1]
las_out.blue = colors[:, 2]
las_out.write(output_las)

print(f"💾 LAS出力完了: {output_las}")
print("🎉 M5処理が完了しました！")
