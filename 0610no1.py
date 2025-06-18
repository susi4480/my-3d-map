# -*- coding: utf-8 -*-
import laspy
import numpy as np
import os

# === 入力ファイル ===
las_path1 = r"C:\Users\user\Documents\lab\output_ply\0610suidoubasi_classified_full.las"
las_path2 = r"C:\Users\user\Documents\lab\output_ply\0610suidoubasi_ue_classified_full.las"

# === 出力ファイル ===
output_path = r"C:\Users\user\Documents\lab\output_ply\0610suidoubasi_merged_full.las"

# === LASファイル読み込み ===
las1 = laspy.read(las_path1)
las2 = laspy.read(las_path2)

# === 点群座標とRGB取得 ===
points1 = np.vstack([las1.x, las1.y, las1.z]).T
colors1 = np.vstack([las1.red, las1.green, las1.blue]).T / 255.0

points2 = np.vstack([las2.x, las2.y, las2.z]).T
colors2 = np.vstack([las2.red, las2.green, las2.blue]).T / 255.0

# === 統合 ===
merged_points = np.vstack([points1, points2])
merged_colors = np.vstack([colors1, colors2])

# === LAS保存関数 ===
def write_las(path, points, colors):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    rgb = (colors * 255).astype(np.uint16)
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(path)

# === 書き出し ===
write_las(output_path, merged_points, merged_colors)
print(f"✅ 統合完了: {output_path}")
