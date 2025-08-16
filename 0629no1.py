# coding: utf-8
"""
method1_median_z_slice.py
【機能】X方向にスライスしてZ中央値の層を抽出し、ポリゴンを構築して航行可能空間を生成
"""

import numpy as np
import laspy
from shapely.geometry import Point, Polygon
from pyproj import CRS

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"
output_las = "/output/0629_method1.las"
voxel_size = 0.5
x_step = 2.0
z_thickness = 1.0
Z_MAX = 3.5  # 固定で設定
crs_utm = CRS.from_epsg(32654)

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === 床ラベル（青：R=0, G=0, B=255）を用いてZ_MINを決定 ===
floor_mask = (colors[:, 0] == 0) & (colors[:, 1] == 0) & (colors[:, 2] == 255)
z_floor = points[floor_mask][:, 2]
Z_MIN = z_floor.min() if len(z_floor) > 0 else points[:, 2].min()

# === Zフィルタリング ===
z_mask = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
points = points[z_mask]

# === スライス処理とポリゴン構築 ===
x_vals = np.arange(points[:, 0].min(), points[:, 0].max(), x_step)
poly_points_top, poly_points_bot = [], []

for x0 in x_vals:
    sl = points[(points[:, 0] >= x0) & (points[:, 0] < x0 + x_step)]
    if len(sl) < 10:
        continue
    z_median = np.median(sl[:, 2])
    z_mask = (sl[:, 2] > z_median - z_thickness / 2) & (sl[:, 2] < z_median + z_thickness / 2)
    sl2d = sl[z_mask]
    if len(sl2d) < 5:
        continue
    poly_points_top.append(sl2d[np.argmax(sl2d[:, 1]), :2])
    poly_points_bot.append(sl2d[np.argmin(sl2d[:, 1]), :2])

if len(poly_points_top) == 0 or len(poly_points_bot) == 0:
    raise RuntimeError("? ポリゴン生成に十分な点がありません")

poly_points = np.vstack(poly_points_top + poly_points_bot[::-1])
polygon = Polygon(poly_points)

# === 航行可能空間のボクセル化 ===
xg = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
yg = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

xy_mask = np.array([polygon.contains(Point(p[:2])) for p in voxels])
navigable_pts = voxels[xy_mask]
colors = np.tile([0, 255, 0], (len(navigable_pts), 1))  # 緑

# === LAS出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001] * 3)
header.offsets = navigable_pts.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = navigable_pts[:, 0], navigable_pts[:, 1], navigable_pts[:, 2]
las_out.red, las_out.green, las_out.blue = colors[:, 0], colors[:, 1], colors[:, 2]
las_out.write(output_las)

print(f"? 出力完了: {output_las}")
