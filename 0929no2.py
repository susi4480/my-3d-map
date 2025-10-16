# -*- coding: utf-8 -*-
"""
【機能】
- 各スライスの左右端を抽出
- 隣接スライスの端点を結んで「川幅の帯メッシュ」を生成
- ConvexHullや中間線は使わない
"""

import os, glob
import numpy as np
import laspy, trimesh

# === 入出力 ===
INPUT_DIR = "/workspace/output/0917no2_7_3_filtered_slices"
OUTPUT_PLY = "/workspace/output/1001_riverband_mesh.ply"

# === スライス読み込み ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.las")))
slices = []
for f in files:
    las = laspy.read(f)
    pts = np.vstack([las.x, las.y, las.z]).T
    slices.append(pts)

# === 両端抽出関数 ===
def get_left_right(points):
    # Y方向で最小・最大の点を左右端とする（X,Zはそのまま）
    i_left = np.argmin(points[:,1])
    i_right = np.argmax(points[:,1])
    return points[i_left], points[i_right]

# === メッシュ化 ===
verts, faces = [], []
for i in range(len(slices)-1):
    pL1, pR1 = get_left_right(slices[i])
    pL2, pR2 = get_left_right(slices[i+1])

    base_idx = len(verts)
    verts.extend([pL1, pR1, pL2, pR2])

    # 左岸ストリップ
    faces.append([base_idx+0, base_idx+2, base_idx+1])
    # 右岸ストリップ
    faces.append([base_idx+1, base_idx+2, base_idx+3])

mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=False)
mesh.export(OUTPUT_PLY)

print(f"✅ 川幅メッシュ出力完了: {OUTPUT_PLY}")
