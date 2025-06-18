import laspy
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import open3d as o3d
import os
import glob

# === 設定 ===
input_dir = r"C:\Users\user\Documents\lab\data\las_field"
grid_res = 0.5
search_radius = 12.0
exclude_radius = 1.0
z_max_threshold = 4.5  # 補間対象にする最大Z値（高い構造物除去）
output_ply_path = r"C:\Users\user\Documents\lab\output_ply\0523_zslice.ply"

# === LAS読み込みと統合 ===
las_files = glob.glob(os.path.join(input_dir, "*.las"))
all_points = []

for path in las_files:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    all_points.append(xyz)

riverbed_all = np.vstack(all_points)

# === Zスライス（高いZを除外）
riverbed = riverbed_all[riverbed_all[:, 2] < z_max_threshold]

# === XY平面上の外周ポリゴン生成（ConvexHullを代用して簡略化可）
from scipy.spatial import ConvexHull
hull = ConvexHull(riverbed[:, :2])
mask_polygon = riverbed[hull.vertices]
poly = Polygon(mask_polygon[:, :2])

# === グリッド生成（マスク内）
x_min, x_max = np.min(riverbed[:, 0]), np.max(riverbed[:, 0])
y_min, y_max = np.min(riverbed[:, 1]), np.max(riverbed[:, 1])
gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                     np.arange(y_min, y_max, grid_res))
grid_points = np.vstack((gx.ravel(), gy.ravel())).T
mask = np.array([poly.contains(Point(p)) for p in grid_points])
masked_grid = grid_points[mask]

# === 欠損点抽出
tree_exist = cKDTree(riverbed[:, :2])
distance, _ = tree_exist.query(masked_grid, k=1, distance_upper_bound=exclude_radius)
no_data_mask = ~np.isfinite(distance)
missing_grid = masked_grid[no_data_mask]

# === IDW補間
tree_interp = cKDTree(riverbed[:, :2])
interp_points = []
interp_color = []

for pt in missing_grid:
    idxs = tree_interp.query_ball_point(pt, r=search_radius)
    if len(idxs) >= 3:
        dists = np.linalg.norm(riverbed[idxs, :2] - pt, axis=1)
        weights = 1 / (dists + 1e-6)
        z_val = np.sum(weights * riverbed[idxs, 2]) / np.sum(weights)
        interp_points.append([pt[0], pt[1], z_val])
        interp_color.append([1.0, 0.0, 0.0])  # 赤

# === 出力
all_xyz = riverbed_all.tolist()
all_color = [[0.5, 0.5, 0.5]] * len(riverbed_all)
if interp_points:
    all_xyz += interp_points
    all_color += interp_color

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(all_xyz))
pcd.colors = o3d.utility.Vector3dVector(np.array(all_color))
o3d.io.write_point_cloud(output_ply_path, pcd)

print(f"✅ Zスライス補間完了: {output_ply_path} （補間点数: {len(interp_points)}）")
