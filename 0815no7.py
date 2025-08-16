# -*- coding: utf-8 -*-
"""
【機能】
- 単一のLASファイルを対象に3Dボクセル化（Z≤Z_LIMIT）
- 3Dモルフォロジー補間（クロージング）で空間の隙間を埋める
- 補間されたボクセルの中心を緑点として抽出
- 元の点群と結合してLAS出力（CRS・RGB情報も継承）
"""

import numpy as np
import laspy
import os
from scipy.ndimage import binary_closing
from tqdm import tqdm

# === 入出力パス ===
INPUT_LAS = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0815no7_3dmorphfill_green.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# === パラメータ ===
Z_LIMIT = 1.9
VOXEL_SIZE = 0.1
MORPH_SIZE = 5  # 構造要素サイズ（voxel単位）

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T
mask = points[:, 2] <= Z_LIMIT
points = points[mask]

# === Occupancy Grid 範囲設定 ===
mins = points.min(axis=0)
maxs = points.max(axis=0)
dims = np.ceil((maxs - mins) / VOXEL_SIZE).astype(int)

# === Occupancy Grid 作成 ===
grid = np.zeros(dims, dtype=bool)
indices = ((points - mins) / VOXEL_SIZE).astype(int)
grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
original_grid = grid.copy()

# === 3D モルフォロジー補間（クロージング）===
structure = np.ones((MORPH_SIZE, MORPH_SIZE, MORPH_SIZE), dtype=bool)
closed_grid = binary_closing(grid, structure=structure)

# === 補間されたセルを抽出（0→1になった部分） ===
filled_mask = (closed_grid == 1) & (original_grid == 0)
filled_indices = np.array(np.nonzero(filled_mask)).T
filled_points = filled_indices * VOXEL_SIZE + mins + VOXEL_SIZE / 2

# === 緑色点の生成 ===
green_points = np.zeros((filled_points.shape[0], 6))
green_points[:, :3] = filled_points
green_points[:, 3:] = [0, 255, 0]  # 緑

# === 元の点群の色（RGB）も考慮（なければ黒） ===
if hasattr(las, "red"):
    orig_rgb = np.vstack([las.red[mask], las.green[mask], las.blue[mask]]).T
else:
    orig_rgb = np.zeros((points.shape[0], 3))

orig_points = np.hstack([points, orig_rgb])
all_points = np.vstack([orig_points, green_points])

# === LASとして保存 ===
header = las.header
new_las = laspy.LasData(header)
new_las.x, new_las.y, new_las.z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
new_las.red  = all_points[:, 3].astype(np.uint16)
new_las.green = all_points[:, 4].astype(np.uint16)
new_las.blue = all_points[:, 5].astype(np.uint16)
new_las.write(OUTPUT_LAS)

print(f"✅ 補間点数: {green_points.shape[0]} 点を統合して保存完了: {OUTPUT_LAS}")
