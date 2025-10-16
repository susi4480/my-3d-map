# -*- coding: utf-8 -*-
"""
① 全スライスまとめて α-shape + スライス間つなぎ線
"""

import os, glob
import numpy as np
import laspy, trimesh, alphashape
from scipy.spatial import cKDTree

INPUT_DIR = "/workspace/output/0917no2_7_3_filtered_slices"
OUTPUT_PLY_MESH = "/workspace/output/0929no1_alpha_shape_all.ply"
OUTPUT_PLY_LINES = "/workspace/output/0929no1_alpha_shape_all_lines.ply"
ALPHA = 1.0

# === スライス読み込み ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.las")))
slices = []
for f in files:
    las = laspy.read(f)
    slices.append(np.vstack([las.x, las.y, las.z]).T)

# === α-shape メッシュ ===
print("▶ 全スライスまとめて α-shape ...")
all_points = np.vstack(slices)

mesh = alphashape.alphashape(all_points, ALPHA)
tri_mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices),
                           faces=np.array(mesh.faces))
tri_mesh.export(OUTPUT_PLY_MESH)

# === スライス間つなぎ線 ===
lines = []
for i in range(len(slices)-1):
    tree = cKDTree(slices[i])
    d, idx = tree.query(slices[i+1], k=1)
    for j, p in enumerate(slices[i+1]):
        q = slices[i][idx[j]]
        lines.append([p, q])

line_points = np.array(lines).reshape(-1, 3)
line_colors = np.tile([255, 0, 0], (len(line_points), 1))  # 赤
trimesh.PointCloud(line_points, colors=line_colors).export(OUTPUT_PLY_LINES)

print(f"✅ 一括 α-shape 出力: {OUTPUT_PLY_MESH}")
print(f"✅ つなぎ線 出力: {OUTPUT_PLY_LINES}")
