# -*- coding: utf-8 -*-
"""
M6 Shellマスクによる点群フィルタリング
-----------------------------------
【機能】
- M6 shell を境界マスクとして利用
- 元データ LAS を読み込み、Shell 内部にある点だけを抽出
- 出力:
  - 内部空間LAS（元データ点群のうち境界内）
-----------------------------------
"""

import os
import numpy as np
import laspy
import open3d as o3d
from shapely.geometry import Point, Polygon

# ===== 入出力 =====
INPUT_LAS   = "/data/0828_01_500_suidoubasi_ue.las"
INPUT_SHELL = "/output/0908M6_shell.ply"
OUTPUT_LAS  = "/output/0909_M6_masked_points.las"

Z_MIN, Z_MAX = -6.0, 1.9   # 航行可能範囲の高さ制限

os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ===== Shell 読み込み =====
pcd_shell = o3d.io.read_point_cloud(INPUT_SHELL)
shell_pts = np.asarray(pcd_shell.points)

# XYポリゴン化（凸包をとる）
from shapely.geometry import MultiPoint
poly = MultiPoint(shell_pts[:, :2]).convex_hull
print(f"✅ Shellポリゴン頂点数: {len(shell_pts)}")

# ===== LAS読み込み =====
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T

# ===== 内部判定 =====
inside_mask = []
for p in points:
    if Z_MIN <= p[2] <= Z_MAX and poly.contains(Point(p[0], p[1])):
        inside_mask.append(True)
    else:
        inside_mask.append(False)
inside_mask = np.array(inside_mask)

masked_points = points[inside_mask]
print(f"✅ 内部点数: {len(masked_points)} / {len(points)}")

# ===== LAS保存 =====
if len(masked_points) > 0:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = masked_points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = masked_points[:,0], masked_points[:,1], masked_points[:,2]
    # 緑固定
    las_out.red   = np.zeros(len(masked_points), dtype=np.uint16)
    las_out.green = np.full(len(masked_points), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(masked_points), dtype=np.uint16)
    las_out.write(OUTPUT_LAS)
    print(f"💾 内部空間LASを保存: {OUTPUT_LAS}")
else:
    print("⚠️ 内部点が見つかりませんでした")
