# -*- coding: utf-8 -*-
"""
【統合処理フロー（統合後補間なし版）】
1. floor LAS を Morphology補間（平均Z）
2. 補間済み floor と lidar LAS を統合
3. 0.2m ダウンサンプリング
4. 法線推定 & 分類（赤=壁・青=床・黄=ビル）
5. 最終LAS保存
"""

import os
import numpy as np
import laspy
import cv2
import open3d as o3d
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力 ===
input_floor_las = r"/workspace/output/0919_floor_sita_merged_raw.las"
input_lidar_las = r"/workspace/output/0919_lidar_sita_merged_raw.las"
output_final_las = r"/workspace/output/0919_sita_classified.las"

# === パラメータ ===
voxel_size_interp = 0.05   # 補間用グリッドサイズ
morph_radius = 100
search_radius = 1.0
max_neighbors = 300
down_voxel_size = 0.2

# 法線推定 & 分類
normal_wall_z_max = 3.2
floor_z_max = 1.1
horizontal_threshold = 0.6
search_radius_normals = 1.0
max_neighbors_normals = 500


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
        new_z[i] = np.mean(neighbor_z)  # 平均Zで補間

    valid = ~np.isnan(new_z)
    return np.column_stack([new_xy[valid], new_z[valid]])


# === [1] floor 補間（平均Z） ===
floor_points = read_las_points(input_floor_las)
interp_floor = morphology_interpolation_mean(
    floor_points,
    lambda pts: pts[:, 2] <= 3.0
)
floor_completed = np.vstack([floor_points, interp_floor])
print(f"✅ floor補間後点数: {len(floor_completed):,}")


# === [2] lidar 読み込み ===
lidar_points = read_las_points(input_lidar_las)


# === [3] floor+lidar 統合 ===
all_points_final = np.vstack([floor_completed, lidar_points])
print(f"✅ 統合点数: {len(all_points_final):,}")


# === [4] ダウンサンプリング ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points_final)
pcd = pcd.voxel_down_sample(voxel_size=down_voxel_size)
print(f"✅ ダウンサンプリング後: {len(pcd.points):,}")


# === [5] 法線推定 ===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=search_radius_normals, max_nn=max_neighbors_normals))
normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)

# === [6] 分類 ===
colors = np.zeros((len(points), 3), dtype=np.uint16)
colors[:] = [65535, 65535, 65535]  # 白=未分類
colors[(normals[:, 2] < 0.6) & (points[:, 2] < normal_wall_z_max)] = [65535, 0, 0]     # 赤=壁
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0, 0, 65535]  # 青=床
colors[points[:, 2] >= normal_wall_z_max] = [65535, 65535, 0]  # 黄=ビル

# === [7] 最終LAS保存 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = points.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = points[:, 0], points[:, 1], points[:, 2]
las_out.red, las_out.green, las_out.blue = colors[:, 0], colors[:, 1], colors[:, 2]
las_out.write(output_final_las)

print(f"🎉 最終分類LAS出力完了: {output_final_las} ({len(points):,} 点)")
