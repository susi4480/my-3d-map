# -*- coding: utf-8 -*-
"""
【機能】
/output/0821_suidoubasi_ue.las に対して、Z勾配（高さの局所変化）を用いて
- Z勾配が小さく Z ≤ 1.1 → 青（床）
- Z勾配が大きく Z ≤ 3.2 → 赤（壁）
- その他 → 灰
として分類しPLY出力する。
"""

import laspy
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

# === 入出力 ===
input_las_path = "/output/0821_suidoubasi_sita_no_color.las"
output_ply_path = "/output/0821_sita_zgradient_classified_zlimit.ply"

# === パラメータ ===
search_radius = 1.0        # Z勾配の局所評価半径（メートル）
z_std_threshold = 0.3     # Z標準偏差しきい値
floor_z_max = 1.1
wall_z_max = 3.2

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las_path)
points = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 点数: {len(points):,}")

# === 近傍探索（sklearn）でZ標準偏差を計算 ===
print("📏 Z勾配（局所Z変化）を計算中...")
nbrs = NearestNeighbors(radius=search_radius, algorithm='kd_tree').fit(points)
indices = nbrs.radius_neighbors(return_distance=False)

z_std = np.zeros(len(points))
for i, idx in enumerate(indices):
    if len(idx) > 2:
        z_std[i] = np.std(points[idx, 2])
    else:
        z_std[i] = 999  # 近傍が少なすぎる場合は無効扱い

# === 分類 ===
print("🎨 分類中...")
colors = np.full((len(points), 3), 0.5)  # 初期色: 灰色

flat_mask = (z_std < z_std_threshold) & (points[:, 2] <= floor_z_max)
steep_mask = (z_std >= z_std_threshold) & (points[:, 2] <= wall_z_max)

colors[flat_mask] = [0.0, 0.0, 1.0]  # 青: 平坦かつ低い → 川底・床
colors[steep_mask] = [1.0, 0.0, 0.0] # 赤: 急勾配かつ低め → 壁
unclassified = ~(flat_mask | steep_mask)

print(f"🟦 平坦領域（Z ≤ {floor_z_max}）: {np.sum(flat_mask):,} 点")
print(f"🟥 急勾配領域（Z ≤ {wall_z_max}）: {np.sum(steep_mask):,} 点")
print(f"⚪ 未分類: {np.sum(unclassified):,} 点")

# === 出力（PLY） ===
print(f"💾 出力中... {output_ply_path}")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(output_ply_path, pcd)
print("✅ 出力完了！")
