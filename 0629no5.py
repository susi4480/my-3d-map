# -*- coding: utf-8 -*-
"""
method5_morphology_2d.py
【機能】床ラベルを使ってZ範囲決定 → XY平面に2D投影 → 形態学処理 → ポリゴン → 航行空間抽出
"""

import numpy as np
import laspy
from shapely.geometry import Point, Polygon
from skimage.morphology import binary_dilation, binary_erosion, disk
from scipy.ndimage import binary_fill_holes
from pyproj import CRS

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"
output_las = "/output/0629_method5.las"
voxel_size = 0.5
morph_radius = 3  # 膨張/収縮の構造要素サイズ
Z_MAX = 3.5
crs_utm = CRS.from_epsg(32654)

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === 床ラベルからZ_MIN決定（青ラベル: R=0, G=0, B=255）===
floor_mask = (colors[:, 0] == 0) & (colors[:, 1] == 0) & (colors[:, 2] == 255)
z_floor = points[floor_mask][:, 2]
if len(z_floor) == 0:
    raise ValueError("❌ 床ラベルが見つかりません（青）")
Z_MIN = z_floor.min()

# === Zフィルタリング ===
z_mask = (points[:, 2] >= Z_MIN) & (points[:, 2] <= Z_MAX)
points = points[z_mask]

# === XY平面のグリッドマスク生成 ===
x_vals = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
y_vals = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")

grid = np.zeros_like(xx, dtype=bool)
ix = np.floor((points[:, 0] - x_vals[0]) / voxel_size).astype(int)
iy = np.floor((points[:, 1] - y_vals[0]) / voxel_size).astype(int)
valid = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])
grid[ix[valid], iy[valid]] = True

# === 形態学処理（膨張 → 収縮 → 穴埋め）===
grid = binary_dilation(grid, disk(morph_radius))
grid = binary_erosion(grid, disk(morph_radius))
grid = binary_fill_holes(grid)

# === True領域をポリゴンに変換 ===
coords = np.column_stack(np.nonzero(grid))
points_xy = np.c_[x_vals[coords[:, 0]], y_vals[coords[:, 1]]]
poly = Polygon(points_xy).convex_hull

# === ボクセル生成 & ポリゴン内抽出 ===
xg = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
yg = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
xy_mask = np.array([poly.contains(Point(p[0], p[1])) for p in voxels])
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
