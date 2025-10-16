# -*- coding: utf-8 -*-
"""
floor_ue_xyz と lidar_ue_xyz を統合してそれぞれ LAS 出力
- 入力: 各フォルダの .xyz (lon, lat, z)
- 出力:
    1. floor_ue_xyz → LAS
    2. lidar_ue_xyz → LAS
- 変換: EPSG:4326 → EPSG:32654 (UTM)
"""

import os
import glob
import numpy as np
import laspy
from pyproj import Transformer, CRS

# === 入出力 ===
floor_dir = r"/data/fulldata/floor_ue_xyz/"
lidar_dir = r"/data/fulldata/lidar_ue_xyz/"
output_floor_las = r"/output/0910_floor_merged_raw.las"
output_lidar_las = r"/output/0910_lidar_merged_raw.las"

# === XYZ読み込み関数 ===
def load_xyz_files(directory):
    all_points = []
    files = glob.glob(os.path.join(directory, "*.xyz"))
    if not files:
        raise RuntimeError(f"❌ {directory} に .xyz ファイルが見つかりません")
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
    return np.vstack(all_points)

# === LAS保存関数 ===
def write_las(points, out_path, use_rgb=False):
    if use_rgb:
        header = laspy.LasHeader(point_format=3, version="1.2")  # RGBあり
    else:
        header = laspy.LasHeader(point_format=1, version="1.2")  # RGBなし

    header.offsets = points.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # mm精度
    header.add_crs(CRS.from_epsg(32654))  # UTM Zone54N

    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]

    if use_rgb:
        las.red   = np.full(len(points), 65535, dtype=np.uint16)
        las.green = np.full(len(points), 65535, dtype=np.uint16)
        las.blue  = np.full(len(points), 65535, dtype=np.uint16)

    las.write(out_path)
    print(f"💾 LAS出力完了: {out_path} ({len(points):,} 点)")

# === CRS変換器 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

# === [1] floor_ue_xyz → LAS ===
floor_xyz = load_xyz_files(floor_dir)
print(f"📥 floor 点数: {len(floor_xyz):,}")
x_utm, y_utm = transformer.transform(floor_xyz[:, 0], floor_xyz[:, 1])
floor_points = np.column_stack((x_utm, y_utm, floor_xyz[:, 2]))
write_las(floor_points, output_floor_las, use_rgb=False)  # RGB不要ならFalse

# === [2] lidar_ue_xyz → LAS ===
lidar_xyz = load_xyz_files(lidar_dir)
print(f"📥 lidar 点数: {len(lidar_xyz):,}")
x_utm, y_utm = transformer.transform(lidar_xyz[:, 0], lidar_xyz[:, 1])
lidar_points = np.column_stack((x_utm, y_utm, lidar_xyz[:, 2]))
write_las(lidar_points, output_lidar_las, use_rgb=False)

print("🎉 floor_ue_xyz と lidar_ue_xyz の正しいLAS出力が完了しました！")
