# -*- coding: utf-8 -*-

import laspy
import numpy as np
from scipy.spatial import Delaunay
import trimesh
import os

# === 入出力ファイル設定 ===
input_las = "/home/edu1/miyachi/data/pond/MBES_02.las"
output_ply = "/home/edu1/miyachi/output_mesh/MBES_02_delaunay_mesh.ply"

# 出力ディレクトリを作成（存在しない場合のみ）
os.makedirs(os.path.dirname(output_ply), exist_ok=True)

print("🚀 開始: LAS点群 → 補間なし Delaunay メッシュ生成")

# === LASファイル読み込み ===
print("📂 LASファイル読み込み中...")
las = laspy.read(input_las)
points = np.vstack((las.x, las.y, las.z)).T
print(f"✅ 点数: {points.shape[0]:,}")

# === Delaunay三角分割（XY）===
print("🔺 Delaunay三角分割中...")
tri = Delaunay(points[:, :2])
faces = tri.simplices  # 各三角形のインデックス (N, 3)

# === メッシュ構築 ===
print("🧱 メッシュ構築中...")
mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

# === PLYファイルとして出力 ===
print(f"💾 PLY出力中: {output_ply}")
mesh.export(output_ply)

print("🎉 完了: 補間なしメッシュが正常に出力されました")
