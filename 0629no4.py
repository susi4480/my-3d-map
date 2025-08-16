# -*- coding: utf-8 -*-
"""
method4_dbscan_outline.py
【機能】DBSCANで壁のクラスタを検出 → 輪郭からポリゴン生成 → 航行可能空間を抽出
"""

import numpy as np
import laspy
from shapely.geometry import MultiPoint, Polygon, Point
from sklearn.cluster import DBSCAN
from pyproj import CRS

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"
output_las = "/output/0629_method4.las"
voxel_size = 0.5
Z_MAX = 3.5
crs_utm = CRS.from_epsg(32654)

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === 床ラベルからZ_MINを決定（青: R=0, G=0, B=255）===
floor_mask = (colors[:, 0] == 0) & (colors[:, 1] == 0) & (colors[:, 2] == 255)
z_floor = points[floor_mask][:, 2]
if len(z_floor) == 0:
    raise ValueError("❌ 床ラベルが見つかりません（青）")
Z_MIN = z_floor.min()

# === Z範囲でフィルタリング ===
z_mask = (points[:, 2] >= Z_MIN) & (points[:, 2] <= Z_MAX)
points = points[z_mask]

# === DBSCANクラスタリングで壁点を輪郭抽出 ===
xy_points = points[:, :2]
clustering = DBSCAN(eps=3.0, min_samples=20).fit(xy_points)
labels = clustering.labels_
valid_mask = labels != -1
core_points = xy_points[valid_mask]

if len(core_points) < 3:
    raise RuntimeError("❌ クラスタリング結果が不十分です")

polygon = MultiPoint(core_points).convex_hull

# === ボクセル生成 & ポリゴン内抽出 ===
xg = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
yg = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

xy_mask = np.array([polygon.contains(Point(p[0], p[1])) for p in voxels])
navigable_pts = voxels[xy_mask]
colors_out = np.tile([0, 255, 0], (len(navigable_pts), 1))  # 緑

# === LAS出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = navigable_pts.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = navigable_pts[:, 0], navigable_pts[:, 1], navigable_pts[:, 2]
las_out.red, las_out.green, las_out.blue = colors_out[:, 0], colors_out[:, 1], colors_out[:, 2]
las_out.write(output_las)

print(f"✅ 出力完了: {output_las}")
