# -*- coding: utf-8 -*-
"""
method2_z_slice_layers.py
【機能】Z方向にスライスし、各層ごとに左右端点を使ってポリゴンを生成し、航行可能空間を抽出
"""

import numpy as np
import laspy
from shapely.geometry import Point, Polygon
from pyproj import CRS

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"      # 入力: ラベル付きLASファイル（CRSあり）
output_las = "/output/0629_method2.las"     # 出力先
voxel_size = 0.5
x_step = 2.0
z_step = 1.0
Z_MAX = 3.5
crs_utm = CRS.from_epsg(32654)

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === 床ラベル(青)からZ_MINを決定 ===
floor_mask = (colors[:, 0] == 0) & (colors[:, 1] == 0) & (colors[:, 2] == 255)
z_floor = points[floor_mask][:, 2]
if len(z_floor) == 0:
    raise ValueError("❌ 床ラベルが見つかりません（青色が不足）")
Z_MIN = z_floor.min()

# === Z方向スライス処理 ===
navigable_all = []
z_vals = np.arange(Z_MIN, Z_MAX, z_step)
for z0 in z_vals:
    slice_z = points[(points[:, 2] >= z0) & (points[:, 2] < z0 + z_step)]
    if len(slice_z) < 100:
        continue

    x_vals = np.arange(slice_z[:, 0].min(), slice_z[:, 0].max(), x_step)
    poly_top, poly_bot = [], []
    for x0 in x_vals:
        sl = slice_z[(slice_z[:, 0] >= x0) & (slice_z[:, 0] < x0 + x_step)]
        if len(sl) < 5:
            continue
        poly_top.append(sl[np.argmax(sl[:, 1]), :2])
        poly_bot.append(sl[np.argmin(sl[:, 1]), :2])

    if len(poly_top) < 3 or len(poly_bot) < 3:
        continue

    poly_pts = np.vstack(poly_top + poly_bot[::-1])
    polygon = Polygon(poly_pts)

    xg = np.arange(slice_z[:, 0].min(), slice_z[:, 0].max(), voxel_size)
    yg = np.arange(slice_z[:, 1].min(), slice_z[:, 1].max(), voxel_size)
    zg = np.full((len(xg) * len(yg),), z0 + z_step / 2)
    xx, yy = np.meshgrid(xg, yg, indexing="ij")
    grid_pts = np.c_[xx.ravel(), yy.ravel(), zg]

    mask = np.array([polygon.contains(Point(p[:2])) for p in grid_pts])
    navigable_all.append(grid_pts[mask])

# === 出力処理 ===
if len(navigable_all) == 0:
    raise RuntimeError("❌ 航行可能点が抽出できませんでした")

navigable_pts = np.vstack(navigable_all)
colors_out = np.tile([0, 255, 0], (len(navigable_pts), 1))  # 緑色

header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = navigable_pts.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = navigable_pts[:, 0], navigable_pts[:, 1], navigable_pts[:, 2]
las_out.red, las_out.green, las_out.blue = colors_out[:, 0], colors_out[:, 1], colors_out[:, 2]
las_out.write(output_las)

print(f"✅ 出力完了: {output_las}")
