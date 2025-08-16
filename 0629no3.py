# -*- coding: utf-8 -*-
"""
method3_xslice_normals.py
【機能】X方向スライス＋法線により壁下端を抽出してポリゴン化し、航行可能空間を抽出
"""

import numpy as np
import laspy
from shapely.geometry import Point, Polygon
from pyproj import CRS
import open3d as o3d

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"
output_las = "/output/0629_method3.las"
voxel_size = 0.5
x_step = 2.0
normal_z_th = 0.3  # 法線Z成分の閾値（壁認識）
Z_MAX = 3.5        # Z上限を固定値で設定
crs_utm = CRS.from_epsg(32654)

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

# === 床ラベルからZ_MINを決定（青ラベル）===
floor_mask = (colors[:, 0] == 0) & (colors[:, 1] == 0) & (colors[:, 2] == 255)
z_floor = points[floor_mask][:, 2]
if len(z_floor) == 0:
    raise ValueError("❌ 床ラベルが見つかりません（青）")
Z_MIN = z_floor.min()

# === 法線推定用にOpen3Dへ変換・推定 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
normals = np.asarray(pcd.normals)
points = np.asarray(pcd.points)

# === X方向スライスごとに壁下端点を検出しポリゴン生成 ===
x_vals = np.arange(points[:, 0].min(), points[:, 0].max(), x_step)
bot_points, top_points = [], []

for x0 in x_vals:
    sl_mask = (points[:, 0] >= x0) & (points[:, 0] < x0 + x_step)
    sl = points[sl_mask]
    sl_normals = normals[sl_mask]
    
    wall_mask = (sl_normals[:, 2] < normal_z_th)
    wall_pts = sl[wall_mask]
    if len(wall_pts) < 5:
        continue

    bot_points.append(wall_pts[np.argmin(wall_pts[:, 1]), :2])
    top_points.append(wall_pts[np.argmax(wall_pts[:, 1]), :2])

if len(bot_points) < 3 or len(top_points) < 3:
    raise RuntimeError("❌ ポリゴン生成に必要な端点が不足しています")

polygon = Polygon(np.vstack(bot_points + top_points[::-1]))

# === ボクセル生成 & ポリゴン内抽出 ===
xg = np.arange(points[:, 0].min(), points[:, 0].max(), voxel_size)
yg = np.arange(points[:, 1].min(), points[:, 1].max(), voxel_size)
zg = np.arange(Z_MIN, Z_MAX, voxel_size)
xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
voxels = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

mask = np.array([polygon.contains(Point(p[:2])) for p in voxels])
navigable_pts = voxels[mask]
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
