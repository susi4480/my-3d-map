/data/0828_01_500_suidoubasi_ue.las# -*- coding: utf-8 -*-
"""
M5方式（白点除去付き）: 3D占有ボクセルの最大連結成分を抽出
- 入力LASの白色点群（RGB: 65535,65535,65535）を除去
- Z ≤ Z_LIMIT で Occupancyグリッドを構築
- スライスごとにfree空間を抽出し、filldown処理で埋める
- スライスを3Dで連結し、最大成分の外殻ボクセル中心点を抽出
- 緑色点群としてLAS出力
"""

import os
import numpy as np
import laspy
from collections import deque

# ===== パラメータ =====
INPUT_LAS  = "/output/0731_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0910no2_M5_voxel_only_cleaned.las"

Z_LIMIT = 1.9       # 高さ制限
GRID_RES = 0.1      # ボクセルサイズ
MIN_PTS = 5        # occupancyに使う最小点数
FILLDOWN_DEPTH = 5  # 空中空間のfilldown深さ（Z方向）

# ===== 出力用フォルダ作成 =====
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# ===== LAS保存関数 =====
def save_las(path, points):
    if len(points) == 0:
        print("⚠️ 出力点が0です")
        return
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.red   = np.zeros(len(points), dtype=np.uint16)
    las.green = np.full(len(points), 65535, dtype=np.uint16)
    las.blue  = np.zeros(len(points), dtype=np.uint16)
    las.write(path)
    print(f"✅ LAS出力完了: {path}（点数: {len(points):,}）")

# ===== main処理 =====
def main():
    print("📥 LAS読み込み中...")
    las = laspy.read(INPUT_LAS)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)

    # === 白色点群を除外 ===
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        r = np.asarray(las.red)
        g = np.asarray(las.green)
        b = np.asarray(las.blue)
        is_white = (r == 65535) & (g == 65535) & (b == 65535)
        keep = ~is_white
        print(f"🧹 白点除去: {np.count_nonzero(is_white):,} 点 → {np.count_nonzero(keep):,} 点を使用")
        x, y, z = x[keep], y[keep], z[keep]

    # === Z制限 ===
    mask_z = z <= Z_LIMIT
    x, y, z = x[mask_z], y[mask_z], z[mask_z]
    xyz = np.column_stack([x, y, z])

    # === Occupancyグリッド構築 ===
    print("🧱 Occupancyグリッド構築中...")
    min_bound = xyz.min(axis=0)
    max_bound = xyz.max(axis=0)
    dims = np.ceil((max_bound - min_bound) / GRID_RES).astype(int) + 1

    occ = np.zeros(dims, dtype=np.uint32)
    idx = ((xyz - min_bound) / GRID_RES).astype(int)
    for i in idx:
        occ[tuple(i)] += 1

    occ_mask = occ >= MIN_PTS

    # === スライスごとにfree空間を抽出 ===
    print("📐 スライスごとにfree空間とfilldown処理中...")
    free_mask = np.zeros_like(occ_mask, dtype=bool)
    for i in range(occ_mask.shape[0]):
        for j in range(occ_mask.shape[1]):
            column = occ_mask[i, j, :]
            if not np.any(column): continue
            first_occ = np.argmax(column)
            free_mask[i, j, :first_occ] = True

    # === filldown処理（上の空間を埋める）
    for i in range(occ_mask.shape[0]):
        for j in range(occ_mask.shape[1]):
            col = free_mask[i, j, :]
            ones = np.where(col)[0]
            if len(ones) == 0: continue
            max_z = ones[-1]
            start_z = max(0, max_z - FILLDOWN_DEPTH)
            free_mask[i, j, start_z:max_z+1] = True

    # === 3D最大連結成分（航行可能空間）抽出 ===
    print("🧭 最大連結成分抽出中...")
    visited = np.zeros_like(free_mask, dtype=bool)
    labels = np.zeros_like(free_mask, dtype=np.uint32)
    label = 1
    max_count = 0
    max_indices = []

    directions = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==dy==dz==0)]

    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if not free_mask[i,j,k] or visited[i,j,k]: continue
                queue = deque()
                queue.append((i,j,k))
                visited[i,j,k] = True
                current = []
                while queue:
                    ci,cj,ck = queue.popleft()
                    current.append((ci,cj,ck))
                    for dx,dy,dz in directions:
                        ni,nj,nk = ci+dx, cj+dy, ck+dz
                        if 0<=ni<dims[0] and 0<=nj<dims[1] and 0<=nk<dims[2]:
                            if free_mask[ni,nj,nk] and not visited[ni,nj,nk]:
                                visited[ni,nj,nk] = True
                                queue.append((ni,nj,nk))
                if len(current) > max_count:
                    max_count = len(current)
                    max_indices = current

    print(f"✅ 最大成分ボクセル数: {max_count:,}")

    # === 外殻ボクセル抽出（6近傍にfree以外があれば境界）===
    print("🔍 境界点抽出中...")
    shell = []
    for i,j,k in max_indices:
        for dx,dy,dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            ni,nj,nk = i+dx, j+dy, k+dz
            if not (0<=ni<dims[0] and 0<=nj<dims[1] and 0<=nk<dims[2]) or not free_mask[ni,nj,nk]:
                shell.append((i,j,k))
                break

    # === ボクセル中心を世界座標に変換 ===
    out_points = (np.array(shell) + 0.5) * GRID_RES + min_bound

    # === LAS出力 ===
    save_las(OUTPUT_LAS, out_points)

if __name__ == "__main__":
    main()
