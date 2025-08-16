# -*- coding: utf-8 -*-
"""
【機能】
1. LASファイルと指定フォルダ内の *.xyz（lat lon z）を統合
2. Z <= 4.5 m の点だけ残す
3. 緯度経度をUTM54Nに変換（LAS側はそのまま）
4. Zを削除して (X, Y) のみを .xyz に保存
"""

import os
import glob
import numpy as np
import laspy
from pyproj import Transformer

# === 設定 ===
las_path    = "/output/0711_suidoubasi_floor_ue_25.las"
xyz_dir     = "/data/suidoubasi/lidar_ue_xyz/"
output_path = "/output/0712_combined_floor_ue_2D.xyz"

z_threshold = 4.5
utm_epsg    = "epsg:32654"
transformer = Transformer.from_crs("epsg:4326", utm_epsg, always_xy=True)

xy_list = []
total_in, total_out = 0, 0

# === [1] LAS読み込み ===
print("📥 LASファイル読み込み中...")
las = laspy.read(las_path)
las_points = np.vstack([las.x, las.y, las.z]).T
total_in += len(las_points)
mask_las = las_points[:, 2] <= z_threshold
xy_list.append(las_points[mask_las][:, :2])
total_out += mask_las.sum()

# === [2] XYZファイル読み込み（緯度経度→UTM）===
print("📥 XYZファイル読み込み中...")
xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))

for path in xyz_files:
    try:
        data = np.loadtxt(path)
        if data.shape[1] < 3:
            print(f"⚠ 列数不足スキップ: {os.path.basename(path)}")
            continue

        lat, lon, z = data[:, 0], data[:, 1], data[:, 2]
        total_in += len(z)
        mask = z <= z_threshold
        if mask.sum() == 0:
            continue

        x, y = transformer.transform(lon[mask], lat[mask])
        xy_list.append(np.vstack([x, y]).T)
        total_out += mask.sum()

    except Exception as e:
        print(f"⚠ 読み込み失敗スキップ: {path} → {e}")

# === [3] 結合 & 保存 ===
if not xy_list:
    raise RuntimeError("❌ 条件を満たす点が 1 点も得られませんでした。")

points_xy = np.vstack(xy_list)
np.savetxt(output_path, points_xy, fmt="%.3f")

print("🎉 完了しました")
print(f"  読み込んだ総点数 : {total_in:,}")
print(f"  出力した点数     : {total_out:,}")
print(f"  出力ファイル     : {output_path}")
