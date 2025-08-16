# -*- coding: utf-8 -*-
import numpy as np
import laspy
from shapely.geometry import Point, Polygon
from pyproj import CRS
import os

# === 設定 ===
input_las = "/home/edu3/lab/data/0611_las2_full.las"
output_las = "/home/edu3/lab/output/0619_navigable_plus_building.las"
voxel_size = 0.5
x_step = 2.0
Z_MIN = -6.0
Z_MAX = 3.5
crs_utm = CRS.from_epsg(32654)

# === LAS読み込み（元データ）
las = laspy.read(input_las)
original_points = np.vstack([las.x, las.y, las.z]).T
colors_original = np.tile([255, 255, 255], (len(original_points), 1))  # 白

# === Z制限内の点のみスライスポリゴン用に使う
z_mask = (original_points[:, 2] > Z_MIN) & (original_points[:, 2] < Z_MAX)
points = original_points[z_mask]

# === スライスポリゴン生成
x_vals = np.arange(points[:, 0].min(), points[:, 0].max(), x_step)
poly_points_top = []
poly_points_bot = []

for x0 in x_vals:
    sl = points[(points[:, 0] >= x0) & (points[:, 0] < x0 + x_step)]
    if len(sl) < 10:
        continue
    poly_points_top.append(sl[np.argmax(sl[:, 1]), :2])
    poly_points_bot.append(sl[np.argmin(sl[:, 1]), :2])

poly_points = np.vstack(poly_points_top + poly_points_bot[::-1])
if len(poly_points) < 3:
    raise RuntimeError("❌ ポリゴン生成に失敗")
mask_poly = Polygon(poly_points)

# === 3Dグリッド点生成（緑ボクセル中心）
xg = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
yg = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# === ポリゴン内のボクセルを抽出
xy_mask = np.array([mask_poly.contains(Point(p[0], p[1])) for p in voxels])
navigable_pts = voxels[xy_mask]
colors_navigable = np.tile([0, 255, 0], (len(navigable_pts), 1))  # 緑

# === 結合（元点群 + 航行可能空間）
combined_pts = np.vstack([original_points, navigable_pts])
combined_colors = np.vstack([colors_original, colors_navigable])

# === LAS出力
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = combined_pts.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x = combined_pts[:, 0]
las_out.y = combined_pts[:, 1]
las_out.z = combined_pts[:, 2]
las_out.red = combined_colors[:, 0]
las_out.green = combined_colors[:, 1]
las_out.blue = combined_colors[:, 2]
las_out.write(output_las)

print(f"✅ 航行可能空間（緑）＋街データ（白）出力完了: {output_las}")
