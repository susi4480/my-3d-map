# -*- coding: utf-8 -*-
"""
0829no4.py
【機能】
LAS点群を2Dグリッド化 → SORノイズ除去 → Morphology補間
→ 補間セルの高さは「近傍セルのZ中央値」で付与
→ 出力LASは1つに統合
  - 白 : SORで残った点
  - 青 : SORで除去された点
  - 赤 : 補間点
"""

import os
import numpy as np
import laspy
import open3d as o3d  # SORに使用
import cv2
from pyproj import CRS

# === 設定 ===
input_las  = r"/output/0827_suidoubasi_floor_ue_ROR_only.las"
output_las = r"/output/0829no4_suidoubasi_floor_ue_SORmorphfill_median_withBlue.las"
voxel_size = 0.05
z_upper_limit = 3.0
morph_radius = 100

# ★SORパラメータ
sor_neighbors = 100
sor_std_ratio = 0.5
neighbor_range = 3  # 補間点のZ推定に使う探索範囲（セル単位）

# === [1] LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 元の点数: {len(points):,}")

# === [2] Z制限 ===
mask = points[:, 2] <= z_upper_limit
points = points[mask]
print(f"✅ Z制限後の点数: {len(points):,}")

# === [3] SORノイズ除去 ===
print("🔹 SORノイズ除去中...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd_clean, ind = pcd.remove_statistical_outlier(
    nb_neighbors=sor_neighbors,
    std_ratio=sor_std_ratio
)
clean_points = np.asarray(pcd_clean.points)   # 白
removed_points = points[~np.asarray(ind)]     # 青
print(f"✅ SOR後の点数: {len(clean_points):,} / {len(points):,} ({len(removed_points)} 点除去)")

# === [4] 2Dグリッド化 + セルごとにZ分布保持 ===
min_x, min_y = clean_points[:, 0].min(), clean_points[:, 1].min()
ix = np.floor((clean_points[:, 0] - min_x) / voxel_size).astype(int)
iy = np.floor((clean_points[:, 1] - min_y) / voxel_size).astype(int)

grid_shape = (ix.max() + 1, iy.max() + 1)
grid = np.zeros(grid_shape, dtype=bool)

cell_to_z = {}
for (iix, iiy, z) in zip(ix, iy, clean_points[:, 2]):
    grid[iix, iiy] = True
    cell_to_z.setdefault((int(iix), int(iiy)), []).append(z)

# === [5] Morphology補間 ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
grid_closed = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
grid_closed = grid_closed.astype(bool)

# === [6] 新たに追加されたセルを探索 ===
new_mask = (grid_closed & ~grid)
new_ix, new_iy = np.where(new_mask)

# === [7] 補間点に高さを付与（近傍セルZの中央値） ===
new_points = []
for iix, iiy in zip(new_ix, new_iy):
    neighbor_z = []
    for dx in range(-neighbor_range, neighbor_range+1):
        for dy in range(-neighbor_range, neighbor_range+1):
            key = (int(iix+dx), int(iiy+dy))
            if key in cell_to_z:
                neighbor_z.extend(cell_to_z[key])
    if neighbor_z:
        z_val = np.median(neighbor_z)
        new_points.append([
            iix*voxel_size + min_x,
            iiy*voxel_size + min_y,
            z_val
        ])
new_points = np.array(new_points)
print(f"✅ 補間点数（中央値付与）: {len(new_points):,}")

# === [8] 統合 ===
all_points = np.vstack([clean_points, removed_points, new_points])
print(f"📦 合計点数: {len(all_points):,}")

# === [9] 色設定 ===
colors = np.zeros((len(all_points), 3), dtype=np.uint16)
colors[:len(clean_points)] = [65535, 65535, 65535]   # 白
colors[len(clean_points):len(clean_points)+len(removed_points)] = [0, 0, 65535]  # 青
colors[len(clean_points)+len(removed_points):] = [65535, 0, 0]  # 赤

# === [10] LAS保存 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = all_points.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x = all_points[:, 0]
las_out.y = all_points[:, 1]
las_out.z = all_points[:, 2]
las_out.red   = colors[:, 0]
las_out.green = colors[:, 1]
las_out.blue  = colors[:, 2]
las_out.write(output_las)
print(f"💾 LAS出力完了: {output_las}")

print("🎉 白＋青＋赤を含めたSOR＋中央値補間LASの出力が完了しました！")
