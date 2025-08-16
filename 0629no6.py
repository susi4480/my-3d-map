# -*- coding: utf-8 -*-
"""
method6_graph_region_growing.py
【機能】床ラベル点から3D近傍を探索し、接続された空間を領域拡張で抽出
"""

import numpy as np
import laspy
from pyproj import CRS
from scipy.spatial import cKDTree
from collections import deque

# === 入出力設定 ===
input_las = "/data/0611_las2_full.las"
output_las = "/output/0629_method6.las"
crs_utm = CRS.from_epsg(32654)

# === パラメータ ===
radius = 0.8            # 隣接判定の半径[m]
max_z_diff = 1.0        # Zの高低差許容[m]
max_points = 1_000_000  # 拡張の上限数
Z_MAX = 3.5             # 上限Z固定

# === LAS読み込み ===
las = laspy.read(input_las)
points_all = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)  # ← 修正済み
colors_all = np.vstack([las.red, las.green, las.blue]).T

# === 床ラベルから開始点抽出（青: R=0, G=0, B=255）===
floor_mask_all = (colors_all[:, 0] == 0) & (colors_all[:, 1] == 0) & (colors_all[:, 2] == 255)
floor_points = points_all[floor_mask_all]
if len(floor_points) == 0:
    raise ValueError("? 床ラベルの点が見つかりません（青）")
Z_MIN = floor_points[:, 2].min()

# === Zフィルタリング ===
z_mask = (points_all[:, 2] >= Z_MIN) & (points_all[:, 2] <= Z_MAX)
points = points_all[z_mask]
colors = colors_all[z_mask]

# === KDTree構築 ===
tree = cKDTree(points)
visited = np.zeros(len(points), dtype=bool)
output_mask = np.zeros(len(points), dtype=bool)

# === 探索キュー初期化（最初の床点から拡張）===
seed_idx = tree.query(points[0], k=1)[1]
queue = deque([seed_idx])
visited[seed_idx] = True
output_mask[seed_idx] = True

# === 領域拡張（Region Growing）===
while queue and output_mask.sum() < max_points:
    idx = queue.popleft()
    center = points[idx]
    neighbors = tree.query_ball_point(center, r=radius)
    for n_idx in neighbors:
        if visited[n_idx]:
            continue
        dz = abs(points[n_idx][2] - center[2])
        if dz < max_z_diff:
            visited[n_idx] = True
            output_mask[n_idx] = True
            queue.append(n_idx)

# === 出力処理 ===
navigable_pts = points[output_mask]
colors_out = np.tile([0, 255, 0], (len(navigable_pts), 1))  # 緑

header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])
header.offsets = navigable_pts.min(axis=0)
header.add_crs(crs_utm)

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = navigable_pts[:, 0], navigable_pts[:, 1], navigable_pts[:, 2]
las_out.red, las_out.green, las_out.blue = colors_out[:, 0], colors_out[:, 1], colors_out[:, 2]
las_out.write(output_las)

print(f"✅ 出力完了: {output_las}")
