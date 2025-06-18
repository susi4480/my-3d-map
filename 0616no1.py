# -*- coding: utf-8 -*-
import numpy as np
import laspy

# === 入力ファイル（元LASと内部空間.xyz）===
input_las = "/home/edu1/miyachi/output/0611_las2_full.las"
internal_xyz = "/home/edu1/miyachi/output/voxel_internal_space.xyz"
output_combined_las = "/home/edu1/miyachi/output/0611_combined_with_internal.las"

# === 1. 元LAS読み込み ===
las = laspy.read(input_las)
base_points = np.vstack([las.x, las.y, las.z]).T
base_colors = np.vstack([las.red, las.green, las.blue]).T

# === 2. 内部空間点群の読み込み ===
internal_points = np.loadtxt(internal_xyz)
internal_colors = np.tile([0, 255, 0], (len(internal_points), 1))  # 航行空間は緑で色付け

# === 3. 結合 ===
all_points = np.vstack([base_points, internal_points])
all_colors = np.vstack([base_colors, internal_colors])

# === 4. 新しいLAS作成 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = all_points.min(axis=0)
header.add_crs(las.header.parse_crs())  # 元と同じCRS

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
las_out.red, las_out.green, las_out.blue = all_colors[:, 0], all_colors[:, 1], all_colors[:, 2]

# === 5. 書き出し ===
las_out.write(output_combined_las)
print(f"✅ 統合LASを書き出しました: {output_combined_las}")
