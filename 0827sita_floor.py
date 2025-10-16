# -*- coding: utf-8 -*-
"""
0630no1.py
【機能】
川底点群を2Dグリッド化 → SORノイズ除去 → Morphology補間
→ 半径1m以内の近傍500点の平均Zで高さ付与
→ 補間点を赤色に設定し、以下2つのLASを出力
  1. 元点群＋補間点統合LAS（補間点は赤）
  2. 補間点のみLAS（赤）
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
from skimage.morphology import binary_closing, disk
import laspy
from scipy.spatial import cKDTree
import open3d as o3d  # ★SORに使用

# === 設定 ===
input_dir = r"/data/fulldata/floor_sita_xyz/"
output_las_merged = r"/output/0827_suidoubasi_floor_sita_merged_SOR.las"
output_las_interp_only = r"/output/0827_suidoubasi_floor_sita_interp_only_SOR.las"
voxel_size = 0.05
z_upper_limit = 3.0
morph_radius = 100
search_radius = 7.0      # 近傍探索半径[m]
max_neighbors = 500      # 近傍最大点数
sor_neighbors = 100       # ★SOR近傍点数
sor_std_ratio = 0.8      # ★SOR除去のしきい値

# === XYZ読み込み（NaN除去あり）===
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
            data = data[~np.isnan(data).any(axis=1)]  # NaN除去
            all_points.append(data)
        except Exception as e:
            print(f"⚠ 読み込み失敗: {f} → {e}")
    if not all_points:
        raise RuntimeError("❌ 有効な .xyz ファイルが読み込めませんでした")
    return np.vstack(all_points)

# === [1] 点群読み込み ===
floor_points = load_xyz_files(input_dir)
print(f"✅ 元の点数: {len(floor_points):,}")

# === [2] 緯度経度 → UTMに変換 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)
x_utm, y_utm = transformer.transform(floor_points[:, 1], floor_points[:, 0])
points_utm = np.column_stack((x_utm, y_utm, floor_points[:, 2]))

# === [3] Z<3.0 の点だけ抽出 ===
mask = points_utm[:, 2] <= z_upper_limit
filtered_points = points_utm[mask]
print(f"✅ Z制限後の点数: {len(filtered_points):,}")

# === [4] SORノイズ除去 ===
print("🔹 SORノイズ除去中...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd_clean, ind = pcd.remove_statistical_outlier(
    nb_neighbors=sor_neighbors,
    std_ratio=sor_std_ratio
)
clean_points = np.asarray(pcd_clean.points)
print(f"✅ SOR後の点数: {len(clean_points):,} / {len(filtered_points):,} ({len(filtered_points)-len(clean_points)} 点除去)")

# === [5] 2Dグリッド化 ===
min_x, min_y = clean_points[:, 0].min(), clean_points[:, 1].min()
ix = np.floor((clean_points[:, 0] - min_x) / voxel_size).astype(int)
iy = np.floor((clean_points[:, 1] - min_y) / voxel_size).astype(int)

grid_shape = (ix.max() + 1, iy.max() + 1)
grid = np.zeros(grid_shape, dtype=bool)
grid[ix, iy] = True

# === [6] Morphology補間（クロージング）===
grid_closed = binary_closing(grid, footprint=disk(morph_radius))

# === [7] 新たに追加された点を抽出 ===
new_mask = (grid_closed & ~grid)
new_ix, new_iy = np.where(new_mask)
new_x = new_ix * voxel_size + min_x
new_y = new_iy * voxel_size + min_y
new_xy = np.column_stack((new_x, new_y))

# === [8] 近傍500点の平均Zで高さ補完 ===
tree = cKDTree(clean_points[:, :2])
dists, idxs = tree.query(new_xy, k=max_neighbors, distance_upper_bound=search_radius)

new_z = np.full(len(new_xy), np.nan)
for i in range(len(new_xy)):
    valid = np.isfinite(dists[i]) & (dists[i] < np.inf)
    if not np.any(valid):
        continue
    neighbor_z = clean_points[idxs[i, valid], 2]
    new_z[i] = np.mean(neighbor_z)

valid_points = ~np.isnan(new_z)
new_points = np.column_stack((new_xy[valid_points], new_z[valid_points]))
print(f"✅ 補間点数: {len(new_points):,}")

# === [9] 元点群と補間点を統合 ===
merged_points = np.vstack([clean_points, new_points])
print(f"📦 合計点数: {len(merged_points):,}")

# === [10] 色設定 ===
merged_colors = np.zeros((len(merged_points), 3), dtype=np.uint16)
merged_colors[:len(clean_points)] = [65535, 65535, 65535]  # 元点群 = 白
merged_colors[len(clean_points):] = [65535, 0, 0]          # 補間点 = 赤

interp_colors = np.full((len(new_points), 3), [65535, 0, 0], dtype=np.uint16)

# === [11] LAS保存関数 ===
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

# === [12] LAS出力 ===
write_las_with_color(merged_points, merged_colors, output_las_merged)
write_las_with_color(new_points, interp_colors, output_las_interp_only)

print("🎉 すべてのLAS出力が完了しました！")
