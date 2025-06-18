import laspy
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import open3d as o3d
import os
import glob

# === 設定 ===
input_dir = r"C:\Users\user\Documents\lab\data\suidoubasi\test_xyz_sita"
step = 5
grid_res = 0.5
search_radius = 12.0
exclude_radius = 1.0
output_ply_path = r"C:\Users\user\Documents\lab\output_ply\suidoubasi_sita.ply"

# === LAS読み込みと統合 ===
las_files = glob.glob(os.path.join(input_dir, "*.las"))
all_points = []

for path in las_files:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    all_points.append(xyz)

riverbed = np.vstack(all_points)

# === スライスごとの高さと密度を計算 ===
x_min, x_max = np.min(riverbed[:, 0]), np.max(riverbed[:, 0])
slice_x = []
slice_heights = []
slice_densities = []

for x in np.arange(x_min, x_max, step):
    slice_mask = (riverbed[:, 0] >= x) & (riverbed[:, 0] < x + step)
    slice_pts = riverbed[slice_mask]
    if len(slice_pts) > 0:
        y_min = np.min(slice_pts[:, 1])
        y_max = np.max(slice_pts[:, 1])
        height = y_max - y_min
        density = len(slice_pts) / height if height > 0 else 0
        slice_x.append(x)
        slice_heights.append(height)
        slice_densities.append(density)

slice_x = np.array(slice_x)
slice_heights = np.array(slice_heights)
slice_densities = np.array(slice_densities)

# === 高さと密度に移動平均を適用 ===
def moving_average(x, window=3):
    return np.convolve(x, np.ones(window)/window, mode='valid')

smoothed_heights = moving_average(slice_heights, window=3)
smoothed_densities = moving_average(slice_densities, window=3)
height_diff = np.abs(np.diff(smoothed_heights))

# === 橋検出：高さ急減・密度増加（スムージング済み）
threshold_height = 10
threshold_density = 2.0
narrow_idxs = np.where(height_diff > threshold_height)[0]

bridge_x_ranges = []
for idx in narrow_idxs:
    if smoothed_densities[idx + 1] > smoothed_densities[idx] * threshold_density:
        x_start = slice_x[idx + 1]  # 移動平均のズレに注意
        x_end = slice_x[idx + 2]
        bridge_x_ranges.append((x_start, x_end))

# === マスク作成（橋除外）
top_edges, bottom_edges = [], []

for x in np.arange(x_min, x_max, step):
    skip = any(start <= x <= end for (start, end) in bridge_x_ranges)
    if skip:
        continue
    slice_mask = (riverbed[:, 0] >= x) & (riverbed[:, 0] < x + step)
    slice_pts = riverbed[slice_mask]
    if len(slice_pts) > 0:
        bottom = slice_pts[np.argmin(slice_pts[:, 1])]
        top = slice_pts[np.argmax(slice_pts[:, 1])]
        bottom_edges.append(bottom)
        top_edges.append(top)

if len(top_edges) < 3 or len(bottom_edges) < 3:
    print("⚠ マスク点が足りません。処理中止")
    exit()

mask_polygon = np.array(bottom_edges + top_edges[::-1] + [bottom_edges[0]])
poly = Polygon(mask_polygon[:, :2])

# === グリッド生成（マスク内）
y_min, y_max = np.min(mask_polygon[:, 1]), np.max(mask_polygon[:, 1])
gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                     np.arange(y_min, y_max, grid_res))
grid_points = np.vstack((gx.ravel(), gy.ravel())).T
mask = np.array([poly.contains(Point(p)) for p in grid_points])
masked_grid = grid_points[mask]

# === 欠損グリッド抽出
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
        interp_color.append([1.0, 0.0, 0.0])  # 赤色

# === 出力
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
