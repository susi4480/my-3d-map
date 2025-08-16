# -*- coding: utf-8 -*-
"""
【機能】
(1) 上空からRayを下に飛ばし、床までの空間を航行可能空間（緑）として抽出。
(2) もとの点群（壁・ビルなど）と統合して、1つのLASに出力。
"""

import numpy as np
import laspy
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力設定 ===
input_las = "/output/0704_suidoubasi_sita.las"
output_las = "/output/0704_method9_sita.las"
crs_utm = CRS.from_epsg(32654)

# === パラメータ ===
voxel_size = 0.5
z_step = 0.05
Z_MAX = 3.0
Z_MIN_GLOBAL = -6.0

# === 点群読み込み ===
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).T
cols = np.vstack([las.red, las.green, las.blue]).T

# === 床点マスク（青）===
floor_mask = (cols[:, 0] == 0) & (cols[:, 1] == 0) & (cols[:, 2] >= 255)
floor_pts = pts[floor_mask]

if len(floor_pts) == 0:
    print("⚠ 床ラベルが見つかりません（青）→ 航行可能空間を生成せず終了します")
    exit(0)

print(f"✅ 床点数: {len(floor_pts)} / 全体点数: {len(pts)}")

# === KDTree構築 ===
tree_all = cKDTree(pts)
tree_floor = cKDTree(floor_pts)

# === XYグリッド生成（床点に基づく）===
xy_unique = np.unique(floor_pts[:, :2], axis=0)

navigable = []
for xy in xy_unique:
    for z in np.arange(Z_MAX, Z_MIN_GLOBAL, -z_step):
        query = np.array([xy[0], xy[1], z])

        idx_obj = tree_all.query_ball_point(query, r=voxel_size * 0.5)
        if len(idx_obj) > 0:
            break

        idx_floor = tree_floor.query_ball_point(query, r=voxel_size * 0.5)
        if len(idx_floor) > 0:
            navigable.append(query)
            break

        navigable.append(query)

navigable = np.array(navigable)
print(f"✅ 航行可能点数: {len(navigable)}")

if len(navigable) == 0:
    print("⚠ 航行可能点が見つかりませんでした")
    exit(0)

# === 街データと統合 ===
colors_navi = np.tile([0, 255, 0], (len(navigable), 1))  # 航行空間：緑

# Z_MAX以下の元データのみ保持（床・壁・ビルなど）
mask_below = pts[:, 2] <= Z_MAX
pts_below = pts[mask_below]
cols_below = cols[mask_below]

# 統合
pts_combined = np.vstack([pts_below, navigable])
cols_combined = np.vstack([cols_below, colors_navi])

# === LAS出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = pts_combined.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x = pts_combined[:, 0]
las_out.y = pts_combined[:, 1]
las_out.z = pts_combined[:, 2]
las_out.red   = cols_combined[:, 0]
las_out.green = cols_combined[:, 1]
las_out.blue  = cols_combined[:, 2]

las_out.write(output_las)
print(f"🎉 統合出力完了: {output_las}")
