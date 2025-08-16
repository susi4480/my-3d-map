# -*- coding: utf-8 -*-
"""
method7_sdf_voxel.py
【機能】壁点とのSigned Distance Field（SDF）を計算し、一定距離以上ある空間を航行可能とみなす
"""

import numpy as np
import laspy
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"
output_las = "/output/0629_method7.las"
crs_utm = CRS.from_epsg(32654)

# === パラメータ設定 ===
voxel_size = 0.5     # ボクセルサイズ [m]
safe_margin = 1.0    # 壁からの安全距離 [m]
Z_MAX = 3.5          # 高さ上限 [m]

# === LAS読み込み ===
las = laspy.read(input_las)
points_all = np.vstack([las.x, las.y, las.z])
colors_all = np.vstack([las.red, las.green, las.blue])

# === 壁点抽出（赤: R=255, G=0, B=0）===
wall_mask = (colors_all[0] == 255) & (colors_all[1] == 0) & (colors_all[2] == 0)
wall_pts_all = points_all[:, wall_mask]

if wall_pts_all.size == 0:
    raise ValueError("? 壁点が見つかりません（赤）")

Z_MIN = wall_pts_all[2].min()

# === Zフィルタリング ===
z_mask = (points_all[2] >= Z_MIN) & (points_all[2] <= Z_MAX)
points = points_all[:, z_mask]
colors = colors_all[:, z_mask]
wall_pts = wall_pts_all[:, (wall_pts_all[2] >= Z_MIN) & (wall_pts_all[2] <= Z_MAX)]

if wall_pts.shape[1] == 0:
    raise ValueError("? 指定Z範囲内に壁点が存在しません")

# === ボクセル生成 ===
min_bound = points.min(axis=1)
max_bound = points.max(axis=1)
xg = np.arange(min_bound[0], max_bound[0], voxel_size)
yg = np.arange(min_bound[1], max_bound[1], voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# === SDF距離計算（壁までの最近傍距離）===
tree = cKDTree(wall_pts.T)
distances, _ = tree.query(voxels, k=1)
sdf_mask = distances >= safe_margin
navigable_pts = voxels[sdf_mask]

# === 色は緑で統一 ===
colors_out = np.tile([0, 255, 0], (len(navigable_pts), 1))

# === LAS出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = navigable_pts.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = navigable_pts[:, 0], navigable_pts[:, 1], navigable_pts[:, 2]
las_out.red, las_out.green, las_out.blue = colors_out[:, 0], colors_out[:, 1], colors_out[:, 2]
las_out.write(output_las)

print(f"? 出力完了: {output_las}")
