# -*- coding: utf-8 -*-
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import open3d as o3d
import os
import glob
from pyproj import Transformer
import laspy

# === 設定 ===
riverbed_dir = "/home/edu3/lab/data/suidoubasi/floor_ue_xyz"
lidar_dir = "/home/edu3/lab/data/suidoubasi/lidar_ue_xyz"
step = 5
grid_res = 0.5
search_radius = 12.0
exclude_radius = 1.0
voxel_size = 0.2
normal_wall_z_max = 4.5
floor_z_max = 3.2
horizontal_threshold = 0.90
output_full_las = "/home/edu3/lab/output/0610suidoubasi_ue_classified_full.las"
output_below_las = "/home/edu3/lab/output/0610suidoubasi_ue_classified_below3.2m.las"

# === UTM変換器 ===
to_utm = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

# === 川底データ読み込みとUTM変換 ===
river_files = glob.glob(os.path.join(riverbed_dir, "*.xyz"))
utm_all = []
for path in river_files:
    try:
        data = np.loadtxt(path)
        lat, lon, z = data[:, 0], data[:, 1], data[:, 2]
        x, y = to_utm.transform(lon, lat)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if np.sum(valid) == 0:
            continue
        utm_all.append(np.vstack([x[valid], y[valid], z[valid]]).T)
    except Exception as e:
        print(f"❌ 川底ファイル読込エラー: {path} → {e}")
if not utm_all:
    print("❌ 有効な川底点群が見つかりません")
    exit()
utm_points = np.vstack(utm_all)

# === Y方向スライスによる上下エッジマスク生成 ===
y_min, y_max = np.min(utm_points[:, 1]), np.max(utm_points[:, 1])
top_edges, bottom_edges = [], []
for y in np.arange(y_min, y_max, step):
    slice_mask = (utm_points[:, 1] >= y) & (utm_points[:, 1] < y + step)
    slice_pts = utm_points[slice_mask]
    if len(slice_pts) > 0:
        bottom = slice_pts[np.argmin(slice_pts[:, 0])]
        top = slice_pts[np.argmax(slice_pts[:, 0])]
        bottom_edges.append(bottom)
        top_edges.append(top)
if len(top_edges) < 1 or len(bottom_edges) < 1:
    print("❌ マスク点が不足しています")
    exit()

mask_polygon = np.array(bottom_edges + top_edges[::-1] + [bottom_edges[0]])
poly = Polygon(mask_polygon[:, :2])

# === 補間グリッド生成
x_min, x_max = np.min(mask_polygon[:, 0]), np.max(mask_polygon[:, 0])
gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                     np.arange(y_min, y_max, grid_res))
grid_points = np.vstack((gx.ravel(), gy.ravel())).T
mask = np.array([poly.contains(Point(p)) for p in grid_points])
masked_grid = grid_points[mask]

# === 川底補間（IDW）
tree_exist = cKDTree(utm_points[:, :2])
distance, _ = tree_exist.query(masked_grid, k=1, distance_upper_bound=exclude_radius)
no_data_mask = ~np.isfinite(distance)
missing_grid = masked_grid[no_data_mask]

tree_interp = cKDTree(utm_points[:, :2])
interp_points = []
for pt in missing_grid:
    idxs = tree_interp.query_ball_point(pt, r=search_radius)
    if len(idxs) >= 3:
        dists = np.linalg.norm(utm_points[idxs, :2] - pt, axis=1)
        weights = 1 / (dists + 1e-6)
        z_val = np.sum(weights * utm_points[idxs, 2]) / np.sum(weights)
        if np.isfinite(z_val):
            interp_points.append([pt[0], pt[1], z_val])

interp_points = np.array(interp_points)

# === LiDARデータ読み込みとUTM変換 ===
lidar_files = glob.glob(os.path.join(lidar_dir, "*.xyz"))
lidar_all = []
for path in lidar_files:
    try:
        data = np.loadtxt(path)
        lat, lon, z = data[:, 0], data[:, 1], data[:, 2]
        x, y = to_utm.transform(lon, lat)
        lidar_all.append(np.vstack([x, y, z]).T)
    except Exception as e:
        print(f"❌ LiDAR読込エラー: {path} → {e}")
if not lidar_all:
    print("❌ 有効なLiDAR点群が見つかりません")
    exit()
lidar_utm = np.vstack(lidar_all)

# === 通常点（補間以外）をOpen3Dで分類
non_interp = np.vstack([utm_points, lidar_utm])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(non_interp)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

points_ds = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
colors = np.ones((len(points_ds), 3))  # 初期色：白

# === 通常点分類（壁・床・ビル）
wall_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] < normal_wall_z_max)
floor_mask = (normals[:, 2] > horizontal_threshold) & (points_ds[:, 2] < floor_z_max)
building_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] >= normal_wall_z_max)

colors[wall_mask] = [1.0, 0.0, 0.0]       # 赤：壁
colors[floor_mask] = [0.0, 0.0, 1.0]      # 青：床
colors[building_mask] = [1.0, 1.0, 0.0]   # 黄：ビル

# === 補間点は無条件で青（床）
if len(interp_points) > 0:
    interp_colors = np.tile([0.0, 0.0, 1.0], (len(interp_points), 1))
    all_points = np.vstack([points_ds, interp_points])
    all_colors = np.vstack([colors, interp_colors])
else:
    all_points = points_ds
    all_colors = colors

# === LASファイル保存関数
def write_las(path, points, colors):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    rgb = (colors * 255).astype(np.uint16)
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(path)

# === 出力（Z 3.2mで分離）
mask_below = all_points[:, 2] <= floor_z_max
write_las(output_full_las, all_points, all_colors)
write_las(output_below_las, all_points[mask_below], all_colors[mask_below])

print(f"✅ 出力完了:\n- 全点: {output_full_las}\n- Z≦3.2m: {output_below_las}")
