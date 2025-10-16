# -*- coding: utf-8 -*-
"""
【floor 補間のみ版（補間＋保存だけ）】
1. floor LAS を Morphology補間（平均Z）
2. 補間済み floor を LAS 出力
"""

import os
import numpy as np
import laspy
import cv2
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力 ===
input_floor_las = r"/workspace/output/0919_floor_sita_merged_raw.las"
output_final_las = r"/workspace/output/0919_floor_only_interp.las"

# === パラメータ ===
voxel_size_interp = 0.05
morph_radius = 100
search_radius = 1.0
max_neighbors = 300


# === LAS読み込み ===
def read_las_points(las_path):
    if not os.path.exists(las_path):
        print(f"⚠ LASファイルが存在しません: {las_path}")
        return np.empty((0, 3))
    las = laspy.read(las_path)
    pts = np.vstack([las.x, las.y, las.z]).T
    pts = pts[np.isfinite(pts).all(axis=1)]
    print(f"📥 読み込み: {las_path} ({len(pts):,} 点)")
    return pts


# === Morphology補間（平均Z） ===
def morphology_interpolation_mean(base_points, mask_fn):
    target = base_points[mask_fn(base_points)]
    if target.size == 0:
        print("⚠ 補間対象なし → スキップ")
        return np.empty((0, 3))

    min_x, min_y = target[:, 0].min(), target[:, 1].min()
    ix = np.floor((target[:, 0] - min_x) / voxel_size_interp).astype(int)
    iy = np.floor((target[:, 1] - min_y) / voxel_size_interp).astype(int)

    grid = np.zeros((ix.max()+1, iy.max()+1), dtype=bool)
    grid[ix, iy] = True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*morph_radius+1, 2*morph_radius+1))
    grid_closed = cv2.morphologyEx(grid.astype(np.uint8),
                                   cv2.MORPH_CLOSE, kernel).astype(bool)

    new_ix, new_iy = np.where(grid_closed & ~grid)
    if len(new_ix) == 0:
        print("⚠ 新規セルなし → スキップ")
        return np.empty((0, 3))

    new_xy = np.column_stack([new_ix*voxel_size_interp + min_x,
                              new_iy*voxel_size_interp + min_y])
    tree = cKDTree(target[:, :2])
    dists, idxs = tree.query(new_xy, k=max_neighbors,
                             distance_upper_bound=search_radius)

    new_z = np.full(len(new_xy), np.nan)
    for i in range(len(new_xy)):
        valid = np.isfinite(dists[i]) & (dists[i] < np.inf)
        if not np.any(valid):
            continue
        neighbor_z = target[idxs[i, valid], 2]
        new_z[i] = np.mean(neighbor_z)

    valid = ~np.isnan(new_z)
    return np.column_stack([new_xy[valid], new_z[valid]])


# === [1] floor 補間 ===
floor_points = read_las_points(input_floor_las)
interp_floor = morphology_interpolation_mean(
    floor_points,
    lambda pts: pts[:, 2] <= 3.0
)
floor_completed = np.vstack([floor_points, interp_floor])
print(f"✅ floor補間後点数: {len(floor_completed):,}")


# === [2] LAS保存（補間のみ） ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = floor_completed.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = floor_completed[:, 0], floor_completed[:, 1], floor_completed[:, 2]

# 白色点にする
N = len(floor_completed)
las_out.red = np.full(N, 65535, dtype=np.uint16)
las_out.green = np.full(N, 65535, dtype=np.uint16)
las_out.blue = np.full(N, 65535, dtype=np.uint16)

las_out.write(output_final_las)
print(f"🎉 floor補間LAS出力完了: {output_final_las} ({len(floor_completed):,} 点)")
