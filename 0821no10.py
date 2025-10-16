# -*- coding: utf-8 -*-
"""
0630no1_sor.py
【機能】
- 川底点群を読み込み（緯度経度 → UTM変換）
- Z ≤ 3.0の点にSORノイズ除去 → Morphology補間 → 最近傍Zで補間点に高さ付与
- 元点群と結合し、CRS付きLASとして保存
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
from skimage.morphology import binary_closing, disk
from scipy.spatial import cKDTree
import open3d as o3d
import laspy

# === 設定 ===
input_dir = r"/data/fulldata/floor_ue_xyz/"
output_las = r"/output/0821no10_suidoubasi_floor_ue.las"
voxel_size = 0.1
z_upper_limit = 3.0
morph_radius = 165
sor_nb_neighbors = 20
sor_std_ratio = 2.0

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
            data = data[~np.isnan(data).any(axis=1)]
            all_points.append(data)
        except Exception as e:
            print(f"⚠ 読み込み失敗: {f} → {e}")
    if not all_points:
        raise RuntimeError("❌ 有効な .xyz ファイルが読み込めませんでした")
    return np.vstack(all_points)

# === [1] 点群読み込み ===
floor_points = load_xyz_files(input_dir)
print(f"✅ 元の点数: {len(floor_points):,}")

# === [2] 緯度経度 → UTM変換 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)
x_utm, y_utm = transformer.transform(floor_points[:, 1], floor_points[:, 0])
points_utm = np.column_stack((x_utm, y_utm, floor_points[:, 2]))

# === [3] Z ≤ z_upper_limit の点だけ抽出 ===
mask = points_utm[:, 2] <= z_upper_limit
filtered_points = points_utm[mask]

# === [4] SORフィルタによるノイズ除去 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio)
clean_points = np.asarray(pcd.points)
print(f"🧹 SOR後点数: {len(clean_points):,}（除去数: {len(filtered_points) - len(clean_points):,}）")

# === [5] Occupancy Grid 作成 ===
min_x, min_y = clean_points[:, 0].min(), clean_points[:, 1].min()
ix = np.floor((clean_points[:, 0] - min_x) / voxel_size).astype(int)
iy = np.floor((clean_points[:, 1] - min_y) / voxel_size).astype(int)
grid_shape = (ix.max() + 1, iy.max() + 1)
grid = np.zeros(grid_shape, dtype=bool)
grid[ix, iy] = True

# === [6] Morphology補間 ===
grid_closed = binary_closing(grid, footprint=disk(morph_radius))

# === [7] 補間されたセルにXY座標を与える ===
new_mask = (grid_closed & ~grid)
new_ix, new_iy = np.where(new_mask)
new_x = new_ix * voxel_size + min_x
new_y = new_iy * voxel_size + min_y
new_xy = np.column_stack((new_x, new_y))

# === [8] 最近傍Z付与 ===
tree = cKDTree(clean_points[:, :2])
_, idxs = tree.query(new_xy, k=1)
new_z = clean_points[idxs, 2]
new_points = np.column_stack((new_x, new_y, new_z))
print(f"✅ 補間点数: {len(new_points):,}")

# === [9] 元の点群と補間点を結合 ===
merged_points = np.vstack([points_utm, new_points])
print(f"📦 合計点数: {len(merged_points):,}")

# === [10] LAS保存（CRS付き）===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = merged_points.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))
las = laspy.LasData(header)
las.x = merged_points[:, 0]
las.y = merged_points[:, 1]
las.z = merged_points[:, 2]
las.write(output_las)

print(f"🎉 LAS出力完了: {output_las}")
