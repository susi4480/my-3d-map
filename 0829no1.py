# -*- coding: utf-8 -*-
"""
0827no2.py
【機能】
川底点群に対して ROR（半径ベース外れ値除去）のみを実施
出力LASは1つに統合
  - 白 : RORで残った点
  - 青 : RORで除去された点
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
import laspy
import open3d as o3d  # RORに使用

# === 設定 ===
input_dir = r"/data/fulldata/floor_ue_xyz/"
output_las = r"/output/0827_suidoubasi_floor_ue_ROR_only.las"
z_upper_limit = 3.0

# ★RORパラメータ
ror_radius = 1.0        # 半径[m]
ror_min_points = 100      # この数未満ならノイズ

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

clean_points = np.asarray(pcd_clean.points)         # 残った点（白）
removed_points = filtered_points[~np.asarray(ind)]  # 除去点（青）

print(f"✅ ROR後の点数: {len(clean_points):,} / {len(filtered_points):,} "
      f"({len(removed_points)} 点を除去)")

# === [5] 元点群と除去点を統合 ===
all_points = np.vstack([clean_points, removed_points])
colors = np.zeros((len(all_points), 3), dtype=np.uint16)
colors[:len(clean_points)] = [65535, 65535, 65535]  # 白（残した点）
colors[len(clean_points):] = [0, 0, 65535]          # 青（除去点）

# === [6] LAS保存関数 ===
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

# === [7] LAS出力 ===
write_las_with_color(all_points, colors, output_las)

print("🎉 補間なしで ROR の結果をLAS出力しました！")
