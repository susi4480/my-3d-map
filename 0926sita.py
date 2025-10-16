# -*- coding: utf-8 -*-
"""
【統合処理フロー（merged LAS入力版, scales指定なし, floor補間のみ）】
1. /output/0924_floor_merged.las を読み込み → Morphology補間（Zは近傍中央値）
2. /output/0924_lidar_merged.las を読み込み
3. floor+lidar を統合
4. 0.2m ダウンサンプリング
5. 法線推定 & 分類（赤=壁・青=床・黄=ビル）
6. 最終LAS保存（intensity含む）
"""

import numpy as np
import laspy
import cv2
import open3d as o3d
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力 ===
input_floor_las = r"/output/0925_floor_sita_merged.las"
input_lidar_las = r"/output/0925_lidar_sita_merged.las"
output_final_las = r"/output/0925_sita_classified.las"

# === パラメータ ===
voxel_size_interp = 0.05
morph_radius = 100
down_voxel_size = 0.2

normal_wall_z_max = 3.2
floor_z_max = 1.1
horizontal_threshold = 0.6
search_radius_normals = 1.0
max_neighbors_normals = 500

search_radius_z = 5.0
max_neighbors_z = 50

# === LAS読み込み ===
def load_las(path):
    las = laspy.read(path)
    pts = np.vstack([las.x, las.y, las.z]).T
    intensity = np.array(las.intensity, dtype=np.float32)
    return pts, intensity

# === Morphology補間（Zは近傍中央値） ===
def morphology_interpolation_median(base_points, base_intensity, mask_fn):
    target = base_points[mask_fn(base_points)]
    target_int = base_intensity[mask_fn(base_points)]
    if target.size == 0:
        print("⚠ 補間対象なし → スキップ")
        return np.empty((0, 3)), np.empty((0,))
    min_x, min_y = target[:, 0].min(), target[:, 1].min()
    ix = np.floor((target[:, 0] - min_x) / voxel_size_interp).astype(int)
    iy = np.floor((target[:, 1] - min_y) / voxel_size_interp).astype(int)
    grid = np.zeros((ix.max()+1, iy.max()+1), dtype=bool)
    grid[ix, iy] = True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    grid_closed = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    new_ix, new_iy = np.where(grid_closed & ~grid)
    if len(new_ix) == 0:
        print("⚠ 新規セルなし → スキップ")
        return np.empty((0, 3)), np.empty((0,))
    new_xy = np.column_stack([new_ix*voxel_size_interp + min_x,
                              new_iy*voxel_size_interp + min_y])
    tree = cKDTree(target[:, :2])
    dists, idxs = tree.query(new_xy, k=max_neighbors_z, distance_upper_bound=search_radius_z)
    new_z = np.full(len(new_xy), np.nan)
    new_int = np.full(len(new_xy), np.nan)
    for i in range(len(new_xy)):
        valid = np.isfinite(dists[i]) & (dists[i] < np.inf)
        if not np.any(valid):
            continue
        neighbor_z = target[idxs[i, valid], 2]
        neighbor_int = target_int[idxs[i, valid]]
        new_z[i] = np.median(neighbor_z)
        new_int[i] = np.mean(neighbor_int)
    valid = ~np.isnan(new_z)
    return np.column_stack([new_xy[valid], new_z[valid]]), new_int[valid]

# === メイン処理 ===
# floor補間（中央値Z付与）
floor_points, floor_intensity = load_las(input_floor_las)
interp_floor, interp_floor_int = morphology_interpolation_median(
    floor_points, floor_intensity,
    lambda pts: pts[:, 2] <= 3.0
)
floor_completed = np.vstack([floor_points, interp_floor])
floor_int_completed = np.hstack([floor_intensity, interp_floor_int])

# lidar読み込み & 単純統合
lidar_points, lidar_intensity = load_las(input_lidar_las)
all_points_final = np.vstack([floor_completed, lidar_points])
all_intensity_final = np.hstack([floor_int_completed, lidar_intensity])

# ダウンサンプリング
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points_final)
pcd = pcd.voxel_down_sample(voxel_size=down_voxel_size)
points = np.asarray(pcd.points)

# intensityダウンサンプル（最近傍補間）
tree = cKDTree(all_points_final)
_, idx = tree.query(points, k=1)
intensity_ds = all_intensity_final[idx]

# 法線推定 & 分類
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=search_radius_normals, max_nn=max_neighbors_normals))
normals = np.asarray(pcd.normals)

colors = np.zeros((len(points), 3), dtype=np.uint16)
colors[:] = [65535, 65535, 65535]
colors[(normals[:, 2] < 0.6) & (points[:, 2] < normal_wall_z_max)] = [65535, 0, 0]     # 赤=壁
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0, 0, 65535]  # 青=床
colors[points[:, 2] >= normal_wall_z_max] = [65535, 65535, 0]  # 黄=ビル

# 最終LAS保存
header = laspy.LasHeader(point_format=3, version="1.2")
header.add_crs(CRS.from_epsg(32654))
las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = points[:, 0], points[:, 1], points[:, 2]
las_out.red, las_out.green, las_out.blue = colors[:, 0], colors[:, 1], colors[:, 2]
las_out.intensity = intensity_ds.astype(np.uint16)
las_out.write(output_final_las)

print(f"🎉 最終分類LAS出力完了: {output_final_las} ({len(points):,} 点)")
