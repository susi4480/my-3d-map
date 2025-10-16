# -*- coding: utf-8 -*-
"""
M0: Morphology補間 + SOR付き
- XYZ点群を読み込み
- SOR（統計的外れ値除去）でノイズ除去
- 2Dグリッド化 → Morphologyクロージング
- 半径 search_radius 内の最大 max_neighbors 点の平均Zで高さ補完
- 補間点は赤でLAS出力
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
from skimage.morphology import binary_closing, disk
import laspy
from scipy.spatial import cKDTree
import open3d as o3d

# === 入出力設定 ===
input_dir = r"/data/fulldata/floor_ue_xyz/"
output_las_merged = r"/output/M0_floor_ue_merged.las"
output_las_interp_only = r"/output/M0_floor_ue_interp_only.las"

# === パラメータ ===
voxel_size = 0.05          # 2Dグリッドの解像度[m]
z_upper_limit = 3.0        # Z上限
morph_radius = 100         # Morphologyクロージング半径[セル]
search_radius = 1.0        # 補間時の近傍探索半径[m]
max_neighbors = 500        # 補間時の近傍点数上限
sor_neighbors = 50         # SOR近傍点数
sor_std_ratio = 1.0        # SOR除去しきい値

# === XYZ読み込み ===
def load_xyz_files(directory):
    all_points = []
    files = glob.glob(os.path.join(directory, "*.xyz"))
    for f in files:
        try:
            data = np.loadtxt(f, dtype=float)
            if data.ndim == 1 and data.size == 3:
                data = data.reshape(1, 3)
            elif data.ndim != 2 or data.shape[1] != 3:
                print(f"⚠ 無効なファイル形式: {f}")
                continue
            data = data[~np.isnan(data).any(axis=1)]
            all_points.append(data)
        except Exception as e:
            print(f"⚠ 読み込み失敗: {f} → {e}")
    if not all_points:
        raise RuntimeError("❌ 有効な .xyz ファイルがありません")
    return np.vstack(all_points)

# === [1] XYZ点群の読み込み ===
floor_points = load_xyz_files(input_dir)
print(f"✅ 元の点数: {len(floor_points):,}")

# === [2] WGS84 → UTM変換 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)
x_utm, y_utm = transformer.transform(floor_points[:, 1], floor_points[:, 0])
points_utm = np.column_stack((x_utm, y_utm, floor_points[:, 2]))

# === [3] SORによるノイズ除去 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_utm)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
points_utm = np.asarray(pcd.points)
print(f"✅ SOR後の点数: {len(points_utm):,}")

# === [4] Z<3.0 の点を2Dグリッド化 ===
mask = points_utm[:, 2] <= z_upper_limit
grid_points = points_utm[mask]

min_x, min_y = grid_points[:, 0].min(), grid_points[:, 1].min()
ix = np.floor((grid_points[:, 0] - min_x) / voxel_size).astype(int)
iy = np.floor((grid_points[:, 1] - min_y) / voxel_size).astype(int)

grid_shape = (ix.max() + 1, iy.max() + 1)
grid = np.zeros(grid_shape, dtype=bool)
grid[ix, iy] = True

# === [5] Morphology補間 ===
grid_closed = binary_closing(grid, footprint=disk(morph_radius))

# === [6] 補間すべき点を抽出 ===
new_mask = (grid_closed & ~grid)
new_ix, new_iy = np.where(new_mask)
new_x = new_ix * voxel_size + min_x
new_y = new_iy * voxel_size + min_y
new_xy = np.column_stack((new_x, new_y))

# === [7] 近傍500点の平均Zで高さ補間 ===
tree = cKDTree(grid_points[:, :2])
dists, idxs = tree.query(new_xy, k=max_neighbors, distance_upper_bound=search_radius)
new_z = np.full(len(new_xy), np.nan)

for i in range(len(new_xy)):
    valid = np.isfinite(dists[i]) & (dists[i] < np.inf)
    if not np.any(valid):
        continue
    neighbor_z = grid_points[idxs[i, valid], 2]
    new_z[i] = np.mean(neighbor_z)

valid_points = ~np.isnan(new_z)
new_points = np.column_stack((new_xy[valid_points], new_z[valid_points]))
print(f"✅ 補間点数: {len(new_points):,}")

# === [8] 元点群と統合 ===
merged_points = np.vstack([points_utm, new_points])
print(f"📦 合計点数: {len(merged_points):,}")

# === [9] 色設定 ===
merged_colors = np.zeros((len(merged_points), 3), dtype=np.uint16)
merged_colors[:len(points_utm)] = [65535, 65535, 65535]
merged_colors[len(points_utm):] = [65535, 0, 0]
interp_colors = np.full((len(new_points), 3), [65535, 0, 0], dtype=np.uint16)

# === [10] LAS出力 ===
def write_las_with_color(points, colors, out_path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_crs(CRS.from_epsg(32654))

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]
    las.write(out_path)
    print(f"💾 LAS出力完了: {out_path}")

write_las_with_color(merged_points, merged_colors, output_las_merged)
write_las_with_color(new_points, interp_colors, output_las_interp_only)

print("🎉 M0補間処理が完了しました！")
