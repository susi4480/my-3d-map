# -*- coding: utf-8 -*-
"""
0827no1.py
【機能】
川底点群を2Dグリッド化 → RORノイズ除去 → Morphology補間
→ 半径1m以内の近傍500点の平均Zで高さ付与
→ 出力LASは1つに統合
  - 白 : 元点群（ROR後に残った点）
  - 青 : RORで除去された点
  - 赤 : 補間点
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
from skimage.morphology import binary_closing, disk
import laspy
from scipy.spatial import cKDTree
import open3d as o3d  # ★RORに使用

# === 設定 ===
input_dir = r"/data/fulldata/floor_ue_xyz/"
output_las_merged = r"/output/0827_suidoubasi_floor_ue_merged_ROR.las"
voxel_size = 0.05
z_upper_limit = 3.0
morph_radius = 100
search_radius = 6.0      # 近傍探索半径[m]
max_neighbors = 300      # 近傍最大点数

# ★RORパラメータ
ror_radius = 1.0        # 半径[m]
ror_min_points = 500      # この数未満ならノイズ

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

# === [4] RORノイズ除去 ===
print("🔹 RORノイズ除去中...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd_clean, ind = pcd.remove_radius_outlier(nb_points=ror_min_points, radius=ror_radius)

clean_points = np.asarray(pcd_clean.points)         # 残った点
removed_points = filtered_points[~np.asarray(ind)]  # 除去された点
print(f"✅ ROR後の点数: {len(clean_points):,} / {len(filtered_points):,} "
      f"({len(removed_points)} 点除去)")

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

# === [9] 元点群・除去点・補間点を統合 ===
all_points = np.vstack([clean_points, removed_points, new_points])
print(f"📦 合計点数: {len(all_points):,}")

# === [10] 色設定 ===
colors = np.zeros((len(all_points), 3), dtype=np.uint16)
colors[:len(clean_points)] = [65535, 65535, 65535]   # 白（残した点）
colors[len(clean_points):len(clean_points)+len(removed_points)] = [0, 0, 65535]  # 青（除去点）
colors[len(clean_points)+len(removed_points):] = [65535, 0, 0]  # 赤（補間点）

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
write_las_with_color(all_points, colors, output_las_merged)

print("🎉 すべての処理とLAS出力が完了しました！")
