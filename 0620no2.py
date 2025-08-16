# -*- coding: utf-8 -*-

# ===============================
# 方法②: Zスライスの各層で2D抽出 + 元の街の点群も含めて出力
# ===============================

import numpy as np
import laspy
from shapely.geometry import Point, Polygon
from pyproj import CRS

input_las = "/data/0611_las2_full.las"
output_las = "/output/0620_method2_with_city.las"
voxel_size = 0.5
x_step = 2.0
z_step = 1.0
Z_MIN = -6.0
Z_MAX = 3.5
crs_utm = CRS.from_epsg(32654)

# === 元データ読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors_orig = np.vstack([las.red, las.green, las.blue]).T
z_mask = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
points = points[z_mask]
colors_orig = colors_orig[z_mask]

# === 航行空間ボクセル生成 ===
navigable_all = []
z_vals = np.arange(Z_MIN, Z_MAX, z_step)
for z0 in z_vals:
    slice_z = points[(points[:, 2] >= z0) & (points[:, 2] < z0 + z_step)]
    if len(slice_z) < 10:
        continue
    x_vals = np.arange(slice_z[:, 0].min(), slice_z[:, 0].max(), x_step)
    poly_points_top, poly_points_bot = [], []
    for x0 in x_vals:
        sl = slice_z[(slice_z[:, 0] >= x0) & (slice_z[:, 0] < x0 + x_step)]
        if len(sl) < 5:
            continue
        poly_points_top.append(sl[np.argmax(sl[:, 1]), :2])
        poly_points_bot.append(sl[np.argmin(sl[:, 1]), :2])
    if len(poly_points_top) + len(poly_points_bot) < 3:
        continue
    poly = Polygon(np.vstack(poly_points_top + poly_points_bot[::-1]))
    xg = np.arange(slice_z[:, 0].min(), slice_z[:, 0].max(), voxel_size)
    yg = np.arange(slice_z[:, 1].min(), slice_z[:, 1].max(), voxel_size)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    zz = np.full_like(xx, z0 + z_step / 2)
    voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    xy_mask = np.array([poly.contains(Point(p[0], p[1])) for p in voxels])
    navigable_all.append(voxels[xy_mask])

# === 航行空間と元点群を統合 ===
navigable_pts = np.vstack(navigable_all)
colors_nav = np.tile([0, 255, 0], (len(navigable_pts), 1))

all_points = np.vstack([points, navigable_pts])
all_colors = np.vstack([colors_orig, colors_nav])

# === LASとして書き出し ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = all_points.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
las_out.red, las_out.green, las_out.blue = all_colors[:, 0], all_colors[:, 1], all_colors[:, 2]
las_out.write(output_las)

print(f"✅ 出力完了: {output_las}")
