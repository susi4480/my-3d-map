# -*- coding: utf-8 -*-
"""
M6方式（M5風処理あり）: Occupancyグリッド → Filldown ＋ 境界出力（前後＋底面スライス）
"""

import os
import numpy as np
import laspy

# ===== パラメータ =====
INPUT_LAS = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS_FILLED = "/output/0910no3_M6fill_shellbottom.las"

Z_LIMIT = 1.9
GRID_RES = 0.3
MIN_PTS = 5

# ===== LAS保存関数 =====
def save_las(path, points, classification=None):
    if points is None or len(points) == 0:
        print(f"⚠️ LAS出力なし: {path}")
        return
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = points[:, 0], points[:, 1], points[:, 2]
    if classification is not None:
        las_out.classification = np.asarray(classification, dtype=np.uint16)
    las_out.red   = np.zeros(len(points), dtype=np.uint16)
    las_out.green = np.full(len(points), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(points), dtype=np.uint16)
    las_out.write(path)
    print(f"✅ LAS出力: {path} 点数: {len(points):,}")

# ===== メイン処理 =====
print("📥 LAS読み込み中...")
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

# Occupancyグリッド構築
print("🧱 Occupancyグリッド構築中...")
min_bound = points.min(axis=0)
max_bound = points.max(axis=0)
max_bound[2] = min(max_bound[2], Z_LIMIT)
size = ((max_bound - min_bound) / GRID_RES).astype(int) + 1
voxels = np.zeros(size, dtype=np.uint32)

# Z制限インデックス
indices = ((points - min_bound) / GRID_RES).astype(int)
indices = indices[points[:, 2] <= Z_LIMIT]

for idx in indices:
    if all(0 <= idx[i] < size[i] for i in range(3)):
        voxels[tuple(idx)] += 1

mask = voxels >= MIN_PTS
filled_points = []

print("🔻 Filldown処理中...")
for ix in range(size[0]):
    for iy in range(size[1]):
        z_column = mask[ix, iy, :]
        occ_idx = np.where(z_column)[0]
        if len(occ_idx) >= 2:
            z_min = occ_idx[0]
            z_max = occ_idx[-1]
            for iz in range(z_min, z_max + 1):
                coord = (np.array([ix, iy, iz]) + 0.5) * GRID_RES + min_bound
                filled_points.append(coord)

# ========= 🟢 輪郭追加処理 =========

print("📦 境界（前後スライス＋底面）追加中...")
shell_points = []

# 前後スライス（X方向）
valid_x = np.any(mask, axis=(1, 2))
x_indices = np.where(valid_x)[0]
if len(x_indices) >= 2:
    x_front = x_indices[0]
    x_back = x_indices[-1]
    for iy in range(size[1]):
        for iz in range(size[2]):
            for x_idx in [x_front, x_back]:
                if mask[x_idx, iy, iz]:
                    coord = (np.array([x_idx, iy, iz]) + 0.5) * GRID_RES + min_bound
                    shell_points.append(coord)

# 底面スライス（Z方向）
valid_z = np.any(mask, axis=(0, 1))
z_indices = np.where(valid_z)[0]
if len(z_indices) >= 1:
    z_bottom = z_indices[0]
    for ix in range(size[0]):
        for iy in range(size[1]):
            if mask[ix, iy, z_bottom]:
                coord = (np.array([ix, iy, z_bottom]) + 0.5) * GRID_RES + min_bound
                shell_points.append(coord)

# 結合して出力
print("💾 LAS出力中...")
all_points = np.vstack([filled_points, shell_points])
save_las(OUTPUT_LAS_FILLED, all_points)

print("🎉 M6処理完了（filldown + 川底 + 前後境界）")
