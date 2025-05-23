import laspy
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import open3d as o3d
import os
import glob

# === 設定 ===
input_dir = r"C:\Users\user\Documents\lab\data\las_field"
step = 5
grid_res = 0.5
search_radius = 12.0
exclude_radius = 1.0
output_ply_path = r"C:\Users\user\Documents\lab\output_ply\0523no3.ply"

# === 全LASファイルを統合 ===
las_files = glob.glob(os.path.join(input_dir, "*.las"))
all_points = []

for path in las_files:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    all_points.append(xyz)

riverbed = np.vstack(all_points)

# === スライスごとの幅と密度を計算 ===
y_min, y_max = np.min(riverbed[:, 1]), np.max(riverbed[:, 1])
slice_y = []
slice_widths = []
slice_densities = []

for y in np.arange(y_min, y_max, step):
    slice_mask = (riverbed[:, 1] >= y) & (riverbed[:, 1] < y + step)
    slice_pts = riverbed[slice_mask]
    if len(slice_pts) > 0:
        x_min = np.min(slice_pts[:, 0])
        x_max = np.max(slice_pts[:, 0])
        width = x_max - x_min
        density = len(slice_pts) / width if width > 0 else 0
        slice_y.append(y)
        slice_widths.append(width)
        slice_densities.append(density)

slice_y = np.array(slice_y)
slice_widths = np.array(slice_widths)
slice_densities = np.array(slice_densities)
width_diff = np.abs(np.diff(slice_widths))

# === 川幅急減・密度増加で橋領域を検出 ===
threshold_width = 10
threshold_density = 2.0
narrow_idxs = np.where(width_diff > threshold_width)[0]

bridge_y_ranges = []
for idx in narrow_idxs:
    if slice_densities[idx + 1] > slice_densities[idx] * threshold_density:
        y_start = slice_y[idx]
        y_end = slice_y[idx + 1]
        bridge_y_ranges.append((y_start, y_end))

# === マスク生成（橋除外）===
left_edges, right_edges = [], []

for y in np.arange(y_min, y_max, step):
    skip = any(start <= y <= end for (start, end) in bridge_y_ranges)
    if skip:
        continue
    slice_mask = (riverbed[:, 1] >= y) & (riverbed[:, 1] < y + step)
    slice_pts = riverbed[slice_mask]
    if len(slice_pts) > 0:
        left = slice_pts[np.argmin(slice_pts[:, 0])]
        right = slice_pts[np.argmax(slice_pts[:, 0])]
        left_edges.append(left)
        right_edges.append(right)

if len(left_edges) < 3 or len(right_edges) < 3:
    print("⚠ マスク点が足りません。処理中止")
    exit()

mask_polygon = np.array(left_edges + right_edges[::-1] + [left_edges[0]])
poly = Polygon(mask_polygon[:, :2])

# === グリッド生成（マスク内） ===
x_min, x_max = np.min(mask_polygon[:, 0]), np.max(mask_polygon[:, 0])
gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                     np.arange(y_min, y_max, grid_res))
grid_points = np.vstack((gx.ravel(), gy.ravel())).T
mask = np.array([poly.contains(Point(p)) for p in grid_points])
masked_grid = grid_points[mask]

# === 欠損グリッド点の抽出（既存点と近すぎる場所を除外） ===
tree_exist = cKDTree(riverbed[:, :2])
distance, _ = tree_exist.query(masked_grid, k=1, distance_upper_bound=exclude_radius)
no_data_mask = ~np.isfinite(distance)
missing_grid = masked_grid[no_data_mask]

# === IDW補間処理 ===
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

# === 元データ（灰）＋補間点（赤）で出力 ===
all_xyz = riverbed.tolist()
all_color = [[0.5, 0.5, 0.5]] * len(riverbed)

if interp_points:
    all_xyz += interp_points
    all_color += interp_color

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(all_xyz))
pcd.colors = o3d.utility.Vector3dVector(np.array(all_color))
o3d.io.write_point_cloud(output_ply_path, pcd)

print(f"✅ 統合補間完了: {output_ply_path} （補間点数: {len(interp_points)}）")
