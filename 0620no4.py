# -*- coding: utf-8 -*-

# ===============================
# 方法④: DBSCANクラスタリングによる輪郭検出後にポリゴン生成
# ファイル名: method4_dbscan_outline.py
# ===============================

import numpy as np
import laspy
from shapely.geometry import MultiPoint, Polygon, Point
from sklearn.cluster import DBSCAN
from pyproj import CRS

input_las = "/data/0611_las2_full.las"
output_las = "/output/0620_method4.las"
voxel_size = 0.5
Z_MIN = -6.0
Z_MAX = 3.5
crs_utm = CRS.from_epsg(32654)

las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
z_mask = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
points = points[z_mask]

xy_points = points[:, :2]
clustering = DBSCAN(eps=3.0, min_samples=20).fit(xy_points)
labels = clustering.labels_
valid_mask = labels != -1
core_points = xy_points[valid_mask]

polygon = MultiPoint(core_points).convex_hull

xg = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
yg = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
xy_mask = np.array([polygon.contains(Point(p[:2])) for p in voxels])
navigable_pts = voxels[xy_mask]
colors = np.tile([0, 255, 0], (len(navigable_pts), 1))

header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = navigable_pts.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = navigable_pts[:, 0], navigable_pts[:, 1], navigable_pts[:, 2]
las_out.red, las_out.green, las_out.blue = colors[:, 0], colors[:, 1], colors[:, 2]
las_out.write(output_las)

print(f"✅ 出力完了: {output_las}")