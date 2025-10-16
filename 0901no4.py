# -*- coding: utf-8 -*-
"""
【機能】
- M5（3D占有ボクセル接続）により航行可能空間（緑点群）を抽出
- 連結性があるスライスのみを対象にする（M0連携）
- 各点にスライス番号を classification として付与
- LAS形式で保存（緑色点群、classification付き）

入力 : /output/0828no4_ue_M0_connected.las（Z制限済み、スライス分離なし）
出力 : /output/0901no1_M5_voxel_connected_classified.las
"""

import os
import numpy as np
import laspy
import open3d as o3d
from collections import deque
from copy import deepcopy

# ===== 入出力 =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0901no1_M5_voxel_connected_classified.las"
os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

# ===== パラメータ =====
Z_LIMIT = 1.9
GRID_RES = 0.1
MIN_PTS = 30

# ===== LAS読み込み =====
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T if 'red' in las.point_format.dimension_names else None

# ===== Occupancy Grid（3Dボクセル） =====
min_bound = points.min(axis=0)
max_bound = points.max(axis=0)
size = ((max_bound - min_bound) / GRID_RES).astype(int) + 1
voxels = np.zeros(size, dtype=np.uint8)

indices = ((points - min_bound) / GRID_RES).astype(int)
for idx in indices:
    voxels[tuple(idx)] += 1

# ===== 緑点候補（空間領域） =====
green_voxels = np.argwhere(voxels >= MIN_PTS)

# ===== 3D連結成分（6近傍） =====
visited = np.zeros(len(green_voxels), dtype=bool)
kdtree = o3d.geometry.KDTreeFlann(green_voxels.astype(np.float32))

def bfs(start_idx):
    q = deque([start_idx])
    group = [start_idx]
    visited[start_idx] = True
    while q:
        curr = q.popleft()
        _, idxs, _ = kdtree.search_radius_vector_3d(green_voxels[curr].astype(np.float32), 1.5)  # 1.5 cell = √3
        for ni in idxs:
            if not visited[ni]:
                visited[ni] = True
                q.append(ni)
                group.append(ni)
    return group

# ===== 最大連結成分だけ残す =====
components = []
for i in range(len(green_voxels)):
    if not visited[i]:
        comp = bfs(i)
        components.append(comp)

largest_comp = max(components, key=len)
connected_voxels = green_voxels[largest_comp]

# ===== 点群生成（スライス番号を classification に） =====
out_points = []
out_class = []
for voxel in connected_voxels:
    coord = (voxel + 0.5) * GRID_RES + min_bound
    out_points.append(coord)
    out_class.append(int(coord[0] * 1000))  # スライス番号的なID（例：X=387.123 → 387123）

out_points = np.array(out_points)
out_class = np.array(out_class, dtype=np.uint8)  # LAS形式のclassification対応型

# ===== LAS保存 =====
header = deepcopy(las.header)
header.point_format = las.header.point_format
header.offsets = out_points.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
las_out = laspy.LasData(header)
las_out.x = out_points[:,0]
las_out.y = out_points[:,1]
las_out.z = out_points[:,2]
las_out.classification = out_class
las_out.red   = np.zeros(len(out_points), dtype=np.uint16)
las_out.green = np.full (len(out_points), 65535, dtype=np.uint16)
las_out.blue  = np.zeros(len(out_points), dtype=np.uint16)

las_out.write(OUTPUT_LAS)
print(f"✅ 出力完了: {OUTPUT_LAS} 点数: {len(out_points):,}")
