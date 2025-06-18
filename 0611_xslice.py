# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from pyproj import Transformer, CRS
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import laspy

# === 設定 ====================================================================
input_dir = r"C:\Users\user\Documents\lab\data\suidoubasi\test_xyz_sita"
output_las = r"C:\Users\user\Documents\lab\output_ply\suidoubasi_sita_with_crs.las"

step = 5.0
grid_res = 0.5
search_radius = 12.0
exclude_radius = 1.0
threshold_height = 10.0
threshold_density = 2.0

transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

# === [1] XYZ読み込み & UTM変換 ===============================================
xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
all_points = []

for path in xyz_files:
    try:
        data = np.loadtxt(path)
        if data.shape[1] < 3:
            print(f"⚠ 列数不足スキップ: {os.path.basename(path)}")
            continue
        lat, lon, z = data[:, 0], data[:, 1], data[:, 2]
        x, y = transformer.transform(lon, lat)
        all_points.append(np.vstack([x, y, z]).T)
    except Exception as e:
        print(f"⚠ 読み込み失敗スキップ: {path} → {e}")

if not all_points:
    raise RuntimeError("❌ 有効な .xyz ファイルが読み込めませんでした")

riverbed = np.vstack(all_points)
print(f"✅ 入力点数: {len(riverbed):,}")

# === [2] スライスごとの高さ・密度計算 ========================================
x_min, x_max = np.min(riverbed[:, 0]), np.max(riverbed[:, 0])
slice_x, slice_heights, slice_densities = [], [], []

for x in np.arange(x_min, x_max, step):
    mask = (riverbed[:, 0] >= x) & (riverbed[:, 0] < x + step)
    pts = riverbed[mask]
    if pts.size == 0:
        continue
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    height = y_max - y_min
    density = len(pts) / height if height > 0 else 0
    slice_x.append(x)
    slice_heights.append(height)
    slice_densities.append(density)

slice_x = np.array(slice_x)
slice_heights = np.array(slice_heights)
slice_densities = np.array(slice_densities)

def moving_average(arr, window=3):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

sm_h = moving_average(slice_heights, window=3)
sm_d = moving_average(slice_densities, window=3)
height_diff = np.abs(np.diff(sm_h))

# === [3] 橋領域検出 ==========================================================
narrow_idxs = np.where(height_diff > threshold_height)[0]
bridge_x_ranges = []
for idx in narrow_idxs:
    if sm_d[idx + 1] > sm_d[idx] * threshold_density:
        x_start = slice_x[idx + 1]
        x_end = slice_x[idx + 2]
        bridge_x_ranges.append((x_start, x_end))

# === [4] マスクポリゴン生成（橋除外） =========================================
top_edges, bottom_edges = [], []
for x in np.arange(x_min, x_max, step):
    if any(start <= x <= end for (start, end) in bridge_x_ranges):
        continue
    mask = (riverbed[:, 0] >= x) & (riverbed[:, 0] < x + step)
    pts = riverbed[mask]
    if pts.size == 0:
        continue
    bottom_edges.append(pts[np.argmin(pts[:, 1])])
    top_edges.append(pts[np.argmax(pts[:, 1])])

if len(top_edges) < 3 or len(bottom_edges) < 3:
    raise RuntimeError("⚠ マスク点が不足しています。処理中止")

mask_polygon = np.array(bottom_edges + top_edges[::-1] + [bottom_edges[0]])
poly = Polygon(mask_polygon[:, :2])

# === [5] グリッド生成（マスク内） ============================================
y_min, y_max = np.min(mask_polygon[:, 1]), np.max(mask_polygon[:, 1])
gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                     np.arange(y_min, y_max, grid_res))
grid_pts = np.vstack((gx.ravel(), gy.ravel())).T
inside_mask = np.array([poly.contains(Point(p)) for p in grid_pts])
masked_grid = grid_pts[inside_mask]

# === [6] 欠損グリッド → IDW補間 ==============================================
tree_exist = cKDTree(riverbed[:, :2])
dist, _ = tree_exist.query(masked_grid, k=1, distance_upper_bound=exclude_radius)
missing_grid = masked_grid[~np.isfinite(dist)]

tree_interp = cKDTree(riverbed[:, :2])
interp_pts = []

for pt in missing_grid:
    idxs = tree_interp.query_ball_point(pt, r=search_radius)
    if len(idxs) >= 3:
        dists = np.linalg.norm(riverbed[idxs, :2] - pt, axis=1)
        weights = 1 / (dists + 1e-6)
        z_val = np.sum(weights * riverbed[idxs, 2]) / np.sum(weights)
        interp_pts.append([pt[0], pt[1], z_val])

all_points = np.vstack([riverbed, interp_pts]) if interp_pts else riverbed
print(f"✅ 補間点数: {len(interp_pts):,} 点")
print(f"📦 合計点数: {len(all_points):,} 点")

# === [7] LASファイルとして保存（CRS付き） ===================================
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(all_points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])  # mm 精度

# CRSを追加（UTM Zone 54N, WGS84）
header.add_crs(CRS.from_epsg(32654))

las = laspy.LasData(header)
las.x = all_points[:, 0]
las.y = all_points[:, 1]
las.z = all_points[:, 2]

las.write(output_las)
print(f"🎉 LAS出力完了（CRSあり）: {output_las}")

