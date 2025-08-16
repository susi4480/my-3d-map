# -*- coding: utf-8 -*-

# ===============================
# 方法③: 法線ベクトルのZ成分による地面除去後にスライス
# ファイル名: method3_normal_filter.py
# ===============================

import numpy as np
import laspy
import open3d as o3d
from shapely.geometry import Point, Polygon
from pyproj import CRS

input_las = "/data/0611_las2_full.las"
output_las = "/output/0620_method3.las"
voxel_size = 0.5
x_step = 2.0
Z_MIN = -6.0
Z_MAX = 3.5
crs_utm = CRS.from_epsg(32654)

las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T
z_mask = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
points = points[z_mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
normals = np.asarray(pcd.normals)
non_ground_mask = normals[:, 2] < 0.9
points = np.asarray(pcd.points)[non_ground_mask]

x_vals = np.arange(points[:, 0].min(), points[:, 0].max(), x_step)
poly_points_top, poly_points_bot = [], []
for x0 in x_vals:
    sl = points[(points[:, 0] >= x0) & (points[:, 0] < x0 + x_step)]
    if len(sl) < 10:
        continue
    poly_points_top.append(sl[np.argmax(sl[:, 1]), :2])
    poly_points_bot.append(sl[np.argmin(sl[:, 1]), :2])

poly_points = np.vstack(poly_points_top + poly_points_bot[::-1])
poly = Polygon(poly_points)

xg = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
yg = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
xy_mask = np.array([poly.contains(Point(p[:2])) for p in voxels])
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