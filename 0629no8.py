# -*- coding: utf-8 -*-
"""
method8_octomap_like.py
【機能】床ラベルから真上にレーザーを飛ばし、上に壁がない空間を航行可能として抽出
"""

import numpy as np
import laspy
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"
output_las = "/output/0629_method8.las"
crs_utm = CRS.from_epsg(32654)

# === パラメータ ===
voxel_size = 0.5
z_step = 0.5
z_max = 3.5
z_min_offset = 0.2  # 床から少し上から開始

# === 点群読み込み ===
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).T
cols = np.vstack([las.red, las.green, las.blue]).T  # ← 修正ポイント

# === 床点マスク（青）===
floor_mask = (cols[:, 0] == 0) & (cols[:, 1] == 0) & (cols[:, 2] == 255)
floor_pts = pts[floor_mask]

if len(floor_pts) == 0:
    raise RuntimeError("❌ 床ラベルが見つかりません（青）")

print(f"✅ 床点数: {len(floor_pts)} / 全体点数: {len(pts)}")

# === KDTree構築（全点対象）===
tree_all = cKDTree(pts)

# === 床点から上にRayを飛ばして空中点を抽出 ===
navigable = []
for p in floor_pts:
    z_vals = np.arange(p[2] + z_min_offset, z_max, z_step)
    for z in z_vals:
        query = np.array([p[0], p[1], z])
        idx = tree_all.query_ball_point(query, r=voxel_size * 0.5)
        if len(idx) == 0:
            navigable.append(query)
        else:
            break  # 上方向に物体があればそこで停止

navigable = np.array(navigable)

print(f"✅ 抽出された航行可能点数: {len(navigable)}")

if len(navigable) == 0:
    raise RuntimeError("❌ 航行可能点が見つかりませんでした（Rayが全て遮られた可能性）")

# === LAS出力 ===
colors_out = np.tile([0, 255, 0], (len(navigable), 1))  # 緑

header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = navigable.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = navigable[:, 0], navigable[:, 1], navigable[:, 2]
las_out.red, las_out.green, las_out.blue = colors_out[:, 0], colors_out[:, 1], colors_out[:, 2]
las_out.write(output_las)

print(f"✅ 出力完了: {output_las}")
