# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d
import laspy
import os
import glob
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree

# === 設定 ===
floor_dir = "/home/edu3/lab/data/suidoubasi/floor_sita_las"
lidar_dir = "/home/edu3/lab/data/suidoubasi/lidar_sita_las"
output_ply_path = "/home/edu3/lab/output/suidoubasi_classified_combined.ply"

# === 補間パラメータ ===
grid_res = 0.5
search_radius = 12.0
exclude_radius = 1.0
min_neighbors = 3

# === 分類パラメータ ===
voxel_size = 0.2
normal_wall_z_max = 4.5
floor_z_blue_max = 2.7
floor_z_orange_max = 3.2
horizontal_threshold = 0.90

def read_las_files(directory):
    files = glob.glob(os.path.join(directory, "*.las"))
    all_points = []
    for path in files:
        try:
            las = laspy.read(path)
            xyz = np.vstack((las.x, las.y, las.z)).T
            all_points.append(xyz)
        except Exception as e:
            print(f"❌ 読み込みエラー: {path} → {e}")
    return np.vstack(all_points) if all_points else np.empty((0, 3))

# === 川底補間 ===
def interpolate_floor(points):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                         np.arange(y_min, y_max, grid_res))
    grid_points = np.vstack((gx.ravel(), gy.ravel())).T

    tree_exist = cKDTree(points[:, :2])
    distance, _ = tree_exist.query(grid_points, k=1, distance_upper_bound=exclude_radius)
    no_data_mask = ~np.isfinite(distance)
    missing_grid = grid_points[no_data_mask]

    tree_interp = cKDTree(points[:, :2])
    interp_points = []
    interp_colors = []

    for pt in missing_grid:
        idxs = tree_interp.query_ball_point(pt, r=search_radius)
        if len(idxs) >= min_neighbors:
            dists = np.linalg.norm(points[idxs, :2] - pt, axis=1)
            weights = 1 / (dists + 1e-6)
            z_val = np.sum(weights * points[idxs, 2]) / np.sum(weights)
            if np.isfinite(z_val):
                interp_points.append([pt[0], pt[1], z_val])
                interp_colors.append([0.0, 0.0, 1.0])  # 青固定
    return np.array(interp_points), np.array(interp_colors)

# === 点群読み込み ===
floor_points = read_las_files(floor_dir)
lidar_points = read_las_files(lidar_dir)

if floor_points.shape[0] == 0 or lidar_points.shape[0] == 0:
    print("❌ 有効な点群が読み込めませんでした")
    exit()

# === 補間実行
interp_points, interp_colors = interpolate_floor(floor_points)

# === 統合してOpen3Dに変換
all_points = np.vstack([floor_points, lidar_points, interp_points]) if len(interp_points) else np.vstack([floor_points, lidar_points])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)

# === 法線推定
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

points_ds = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
colors = np.ones((len(points_ds), 3))  # 白で初期化

# === 分類（法線とZで色分け）===
wall_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] < normal_wall_z_max)
floor_mask_blue = (normals[:, 2] > horizontal_threshold) & (points_ds[:, 2] < floor_z_blue_max)
floor_mask_orange = (normals[:, 2] > horizontal_threshold) & (points_ds[:, 2] >= floor_z_blue_max) & (points_ds[:, 2] < floor_z_orange_max)
building_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] >= normal_wall_z_max)

colors[wall_mask] = [1.0, 0.0, 0.0]       # 赤：壁
colors[floor_mask_blue] = [0.0, 0.0, 1.0]  # 青：床（低い）
colors[floor_mask_orange] = [1.0, 0.5, 0.0] # オレンジ：床（高め）
colors[building_mask] = [1.0, 1.0, 0.0]    # 黄：ビル等

# === 出力点群構築（補間点も青で追加）===
output_pcd = o3d.geometry.PointCloud()
output_pcd.points = o3d.utility.Vector3dVector(np.vstack([points_ds, interp_points]) if len(interp_points) else points_ds)
output_pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors, interp_colors]) if len(interp_points) else colors)

# === 書き出し ===
o3d.io.write_point_cloud(output_ply_path, output_pcd)
print(f"✅ PLYファイルとして保存完了: {output_ply_path}（補間点数: {len(interp_points)}）")
