# -*- coding: utf-8 -*-
"""
【機能】
- 床ラベル（青）のXY位置ごとに、Z_MAX（上限）と床Z（終点）で2点の航行可能空間を定義
- 緑ラベルでLAS出力（点数大幅削減）
"""

import numpy as np
import laspy
from pyproj import CRS
from scipy.spatial import cKDTree

# === 入出力設定 ===
input_las = "/output/0704_suidoubasi_ue.las"
output_las = "/output/0707_green_only_ue_simple2pts.las"
crs_utm = CRS.from_epsg(32654)

# === パラメータ ===
Z_MAX = 3.0

# === 点群読み込み ===
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).T
cols = np.vstack([las.red, las.green, las.blue]).T

# === 床点マスク（青）===
floor_mask = (cols[:, 0] == 0) & (cols[:, 1] == 0) & (cols[:, 2] >= 255)
floor_pts = pts[floor_mask]

if len(floor_pts) == 0:
    raise RuntimeError("❌ 床ラベル（青）が見つかりませんでした")

print(f"✅ 床点数: {len(floor_pts)}")

# === XYごとにZ最小（床）を取得 ===
xy_floor = floor_pts[:, :2]
unique_xy, indices = np.unique(xy_floor, axis=0, return_index=True)
floor_pts_unique = floor_pts[indices]

# === 航行可能空間の2点（Z_MAXと床Z）を定義 ===
navi_top = np.column_stack([unique_xy, np.full(len(unique_xy), Z_MAX)])
navi_bottom = floor_pts_unique  # Z: 床点の高さ

navigable = np.vstack([navi_top, navi_bottom])
colors_navi = np.tile([0, 255, 0], (len(navigable), 1))

print(f"✅ 航行可能点数（2点/セル）: {len(navigable)}")

# === LAS出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = navigable.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x = navigable[:, 0]
las_out.y = navigable[:, 1]
las_out.z = navigable[:, 2]
las_out.red   = colors_navi[:, 0]
las_out.green = colors_navi[:, 1]
las_out.blue  = colors_navi[:, 2]

las_out.write(output_las)
print(f"📤 LAS出力完了: {output_las}")
