# -*- coding: utf-8 -*-
"""
floor + lidar の統合LASを入力にして処理
1. floorとlidarのLASを読み込み → 統合
2. Z ≤ 2m で制限
3. 2Dグリッド化 & Morphology補間
   - 白 = 元点群
   - 赤 = 補間点
4. 最終LASを1つ出力
"""

import os
import numpy as np
import laspy
import cv2
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力 ===
input_floor_las = "/workspace/output/0910_merged_floor_ue.las"
input_lidar_las = "/workspace/output/0910_merged_lidar_ue.las"
output_final_las = "/workspace/output/0910_ue_floor_lidar_morphfill.las"

# === パラメータ ===
voxel_size = 0.05        # [m] 2Dグリッド解像度
z_upper_limit = 0.0     # [m] Z制限
morph_radius = 100       # [セル] Morphology補間カーネル半径
search_radius_m = 5.0   # [m] 補間点の高さ付与に使う探索半径
max_neighbors = 500     # [点] 近傍最大点数

# === LAS読み込み関数 ===
def load_las_points(path):
    las = laspy.read(path)
    pts = np.vstack([las.x, las.y, las.z]).T
    return pts

# === LAS保存関数 ===
def save_las(points, colors, out_path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_crs(CRS.from_epsg(32654))
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.red, las.green, las.blue = colors[:, 0], colors[:, 1], colors[:, 2]
    las.write(out_path)
    print(f"💾 LAS出力完了: {out_path} ({len(points):,} 点)")

# === [1] floor + lidar 読み込み ===
floor_points = load_las_points(input_floor_las)
lidar_points = load_las_points(input_lidar_las)
print(f"📥 floor 点数: {len(floor_points):,}")
print(f"📥 lidar 点数: {len(lidar_points):,}")

points = np.vstack([floor_points, lidar_points])
print(f"✅ 統合点数: {len(points):,}")

# === [2] Z制限 ===
points = points[points[:, 2] <= z_upper_limit]
print(f"✅ Z制限後の点数: {len(points):,}")

# === [3] 2Dグリッド化 ===
min_x, min_y = points[:, 0].min(), points[:, 1].min()
ix = np.floor((points[:, 0] - min_x) / voxel_size).astype(int)
iy = np.floor((points[:, 1] - min_y) / voxel_size).astype(int)

grid_shape = (ix.max() + 1, iy.max() + 1)
grid = np.zeros(grid_shape, dtype=bool)

cell_to_z = {}
for (iix, iiy, z) in zip(ix, iy, points[:, 2]):
    grid[iix, iiy] = True
    cell_to_z.setdefault((int(iix), int(iiy)), []).append(z)

# === [4] Morphology補間 ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
grid_closed = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

# === [5] 新たに追加されたセルを探索 ===
new_mask = (grid_closed & ~grid)
new_ix, new_iy = np.where(new_mask)
new_xy = np.column_stack([new_ix*voxel_size + min_x, new_iy*voxel_size + min_y])

# === [6] KDTree で近傍Z中央値付与 ===
tree = cKDTree(points[:, :2])
new_z = np.full(len(new_xy), np.nan)

dists, idxs = tree.query(new_xy, k=max_neighbors, distance_upper_bound=search_radius_m)
for i in range(len(new_xy)):
    valid = np.isfinite(dists[i]) & (dists[i] < np.inf)
    if not np.any(valid):
        continue
    neighbor_z = points[idxs[i, valid], 2]
    new_z[i] = np.median(neighbor_z)

valid_points = ~np.isnan(new_z)
new_points = np.column_stack([new_xy[valid_points], new_z[valid_points]]) if np.any(valid_points) else np.empty((0, 3))
print(f"✅ 補間点数: {len(new_points):,}")

# === [7] 統合して出力 ===
all_points_final = np.vstack([points, new_points])
colors = np.zeros((len(all_points_final), 3), dtype=np.uint16)
colors[:len(points)] = [65535, 65535, 65535]  # 白 = 元点群
colors[len(points):] = [65535, 0, 0]          # 赤 = 補間点

save_las(all_points_final, colors, output_final_las)
print("🎉 統合 + Morphology補間 + LAS出力 完了！")
