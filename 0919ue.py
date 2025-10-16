# -*- coding: utf-8 -*-
"""
【処理フロー】
1. floor_sita_xyz と lidar_sita_xyz をそれぞれ読み込み → 統合 → LAS出力
   - 出力: raw_floor.las, raw_lidar.las
2. 上記2つを結合して補間処理 & 法線推定 & 分類
   - 出力: classified.las
"""

import os
import glob
import numpy as np
import laspy
import cv2
import open3d as o3d
from pyproj import Transformer, CRS
from scipy.spatial import cKDTree

# === 入出力 ===
floor_dir = r"/workspace/fulldata/floor_ue_xyz/"
lidar_dir = r"/workspace/fulldata/lidar_ue_xyz/"
output_floor_las = r"/workspace/output/0919_floor_ue_merged_raw.las"
output_lidar_las = r"/workspace/output/0919_lidar_ue_merged_raw.las"
output_final_las = r"/workspace/output/0919_ue_classified.las"

# === パラメータ ===
voxel_size = 0.05
morph_radius = 100
max_neighbors = 300

# 補間条件
search_radius_1 = 1.0   # Z<-4m
z_threshold_1 = -4.0
search_radius_2 = 4.0   # -3m ≦ Z ≦ 0m
z_min_2, z_max_2 = -3.0, 0.0

# 法線推定 & 分類
normal_wall_z_max = 3.3
floor_z_max = 1.1
horizontal_threshold = 0.70
search_radius_normals = 0.3
max_neighbors_normals = 200

# === XYZ読み込み（genfromtxt採用、NaN/Inf対策つき）===
def load_xyz_files(directory):
    all_points = []
    files = glob.glob(os.path.join(directory, "*.xyz"))
    if not files:
        print(f"⚠ {directory} に .xyz がありません")
        return np.empty((0, 3))
    for f in files:
        try:
            data = np.genfromtxt(f, dtype=float)  # ← genfromtxtに変更
            if data.ndim == 1 and data.size == 3:
                data = data.reshape(1, 3)
            elif data.ndim != 2 or data.shape[1] != 3:
                print(f"⚠ 無効な形式: {f}")
                continue
            # NaN/Inf 除去
            data = data[~np.isnan(data).any(axis=1)]
            data = data[np.isfinite(data).all(axis=1)]
            if data.size > 0:
                all_points.append(data)
        except Exception as e:
            print(f"⚠ 読み込み失敗: {f} → {e}")
    return np.vstack(all_points) if all_points else np.empty((0, 3))

# === LAS保存 ===
def write_las(points, out_path):
    if points.size == 0:
        print(f"⚠ 出力スキップ: {out_path} (点なし)")
        return
    # NaN/Inf 除去
    points = points[~np.isnan(points).any(axis=1)]
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        print(f"⚠ 出力スキップ（NaN/Inf 除去後 点なし）: {out_path}")
        return
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_crs(CRS.from_epsg(32654))
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    # RGBは白固定
    las.red   = np.full(len(points), 65535, dtype=np.uint16)
    las.green = np.full(len(points), 65535, dtype=np.uint16)
    las.blue  = np.full(len(points), 65535, dtype=np.uint16)
    las.write(out_path)
    print(f"💾 LAS出力: {out_path} ({len(points):,} 点)")

# === Morphology補間 ===
def morphology_interpolation(base_points, mask_fn, search_radius, mode="mean"):
    target = base_points[mask_fn(base_points)]
    target = target[np.isfinite(target).all(axis=1)]
    if target.size == 0:
        print("⚠ 補間対象なし → スキップ")
        return np.empty((0, 3))

    min_x, min_y = target[:, 0].min(), target[:, 1].min()
    ix = np.floor((target[:, 0] - min_x) / voxel_size).astype(int)
    iy = np.floor((target[:, 1] - min_y) / voxel_size).astype(int)
    if ix.size == 0 or iy.size == 0 or ix.max() < 0 or iy.max() < 0:
        print("⚠ ix/iy が不正 → スキップ")
        return np.empty((0, 3))

    grid = np.zeros((ix.max()+1, iy.max()+1), dtype=bool)
    grid[ix, iy] = True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    grid_closed = cv2.morphologyEx(grid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    new_ix, new_iy = np.where(grid_closed & ~grid)
    if len(new_ix) == 0:
        print("⚠ 新規セルなし → スキップ")
        return np.empty((0, 3))

    new_xy = np.column_stack([new_ix*voxel_size + min_x, new_iy*voxel_size + min_y])
    tree = cKDTree(target[:, :2])
    dists, idxs = tree.query(new_xy, k=max_neighbors, distance_upper_bound=search_radius)
    new_z = np.full(len(new_xy), np.nan)
    for i in range(len(new_xy)):
        valid = np.isfinite(dists[i]) & (dists[i] < np.inf)
        if not np.any(valid):
            continue
        neighbor_z = target[idxs[i, valid], 2]
        new_z[i] = np.mean(neighbor_z) if mode == "mean" else np.max(neighbor_z)
    valid = ~np.isnan(new_z)
    return np.column_stack([new_xy[valid], new_z[valid]])

# === [1] floor & lidar を個別にLAS化 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

floor_xyz = load_xyz_files(floor_dir)
lidar_xyz = load_xyz_files(lidar_dir)

floor_points, lidar_points = np.empty((0, 3)), np.empty((0, 3))

if floor_xyz.size > 0:
    xf, yf = transformer.transform(floor_xyz[:, 1], floor_xyz[:, 0])
    floor_points = np.column_stack((xf, yf, floor_xyz[:, 2]))
    write_las(floor_points, output_floor_las)

if lidar_xyz.size > 0:
    xl, yl = transformer.transform(lidar_xyz[:, 1], lidar_xyz[:, 0])
    lidar_points = np.column_stack((xl, yl, lidar_xyz[:, 2]))
    write_las(lidar_points, output_lidar_las)

if floor_points.size == 0 and lidar_points.size == 0:
    raise RuntimeError("❌ floor/lidar の点群が空です")

# === [2] 補間 + 法線推定 + 分類 ===
all_points = np.vstack([p for p in [floor_points, lidar_points] if p.size > 0])
all_points = all_points[np.isfinite(all_points).all(axis=1)]

interp1 = morphology_interpolation(all_points, lambda pts: pts[:, 2] < z_threshold_1,
                                   search_radius_1, mode="mean")
interp2 = morphology_interpolation(all_points, lambda pts: (pts[:, 2] >= z_min_2) & (pts[:, 2] <= z_max_2),
                                   search_radius_2, mode="max")

all_points_final = np.vstack([all_points, interp1, interp2])
all_points_final = all_points_final[np.isfinite(all_points_final).all(axis=1)]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points_final)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=search_radius_normals, max_nn=max_neighbors_normals))

normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)
colors = np.zeros((len(points), 3), dtype=np.uint16)
colors[:] = [65535, 65535, 65535]  # 白=未分類
colors[(normals[:, 2] < 0.2) & (points[:, 2] < normal_wall_z_max)] = [65535, 0, 0]      # 赤=壁
colors[(normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)] = [0, 0, 65535]  # 青=床
colors[(normals[:, 2] < 0.3) & (points[:, 2] >= normal_wall_z_max)] = [65535, 65535, 0]  # 黄=ビル

# NaN/Inf除去
mask = np.isfinite(points).all(axis=1)
points, colors = points[mask], colors[mask]

header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = points.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = points[:, 0], points[:, 1], points[:, 2]
las_out.red, las_out.green, las_out.blue = colors[:, 0], colors[:, 1], colors[:, 2]
las_out.write(output_final_las)

print(f"🎉 補間＋分類LAS出力完了: {output_final_las} ({len(points):,} 点)")
