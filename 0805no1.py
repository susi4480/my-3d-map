# -*- coding: utf-8 -*-
"""
【機能】
- X方向に10m幅・2mオーバーラップでブロック分割
- 各ブロックでX方向に投影して1枚のYZビットマップを作成（Xは潰す）
- 各Z行についてY方向の「1に挟まれた0」を航行可能空間として抽出
- 航行可能空間セル(Y,Z)をブロック幅いっぱいにX方向へ押し出し（格子点生成）
- 元点群（Z上限内）＋航行空間点（緑）をLASで出力
"""

import numpy as np
import laspy
import os
from tqdm import tqdm

# === 入出力設定 ===
INPUT_LAS = r"/output/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0805_yz_bitmap_extruded.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# === パラメータ ===
X_WIDTH   = 10.0    # Xスライス幅[m]
X_OVERLAP = 2.0     # Xオーバーラップ[m]
X_RES     = 0.5     # 航行空間のX方向点間隔[m]（押し出し解像度）
Z_RES     = 0.05    # Zスライス厚[m]
Y_RES     = 0.1     # Y方向分解能[m]
Z_LIMIT   = 3.5     # Z上限[m]

# === LAS読み込み & Z制限 ===
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
pts = pts[pts[:, 2] <= Z_LIMIT]
if len(pts) == 0:
    raise RuntimeError("⚠ Z制限内の点が存在しません")

x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
z_min, z_max = pts[:, 2].min(), Z_LIMIT

# 全体のY,Zビンは固定（各ブロックで共有）
y_bins = np.arange(y_min, y_max + Y_RES, Y_RES)
z_bins = np.arange(z_min, z_max + Z_RES, Z_RES)
ny = len(y_bins) - 1
nz = len(z_bins) - 1

green_chunks = []  # 後でまとめてvstack

# === X方向ブロック開始位置 ===
x_starts = np.arange(x_min, x_max, X_WIDTH - X_OVERLAP)

for x0 in tqdm(x_starts, desc="Xブロック処理"):
    x1 = x0 + X_WIDTH
    mask_block = (pts[:, 0] >= x0) & (pts[:, 0] < x1)
    block_pts = pts[mask_block]
    if len(block_pts) == 0:
        continue

    # --- 1) X投影: ブロック全体で1枚のYZビットマップを作成 ---
    # occupancy[zi, yi] = 1 if このブロック内で該当(Y,Z)ビンに点が1つでもあれば1
    occupancy = np.zeros((nz, ny), dtype=np.uint8)
    yi = ((block_pts[:, 1] - y_min) / Y_RES).astype(int)
    zi = ((block_pts[:, 2] - z_min) / Z_RES).astype(int)
    valid = (yi >= 0) & (yi < ny) & (zi >= 0) & (zi < nz)
    yi = yi[valid]; zi = zi[valid]
    occupancy[zi, yi] = 1  # Xは潰しているのでこれでOK

    # --- 2) 航行可能空間の抽出（各Z行でY方向の「1に挟まれた0」）---
    # 押し出し用のX座標（格子）
    x_centers = np.arange(x0 + 0.5 * X_RES, x1, X_RES)
    if len(x_centers) == 0:
        x_centers = np.array([(x0 + x1) * 0.5])

    for zi_row in range(nz):
        z_center = (z_bins[zi_row] + z_bins[zi_row + 1]) * 0.5
        row = occupancy[zi_row, :]

        # 1に挟まれた0区間を走査
        inside = False
        gap_start = None
        for yi_col in range(1, ny - 1):
            if row[yi_col - 1] == 1 and row[yi_col] == 0 and not inside:
                inside = True
                gap_start = yi_col
            if inside and row[yi_col] == 0 and row[yi_col + 1] == 1:
                # gap indices: [gap_start, yi_col]
                y_idx = np.arange(gap_start, yi_col + 1, dtype=int)
                y_centers = y_min + (y_idx + 0.5) * Y_RES

                # --- 3) 押し出し: (X_RES間隔) × (Yビン中心) の格子点を生成 ---
                nx = len(x_centers)
                ny_gap = len(y_centers)
                Xg = np.repeat(x_centers, ny_gap)
                Yg = np.tile(y_centers, nx)
                Zg = np.full(Xg.shape, z_center, dtype=float)

                green_chunks.append(np.column_stack([Xg, Yg, Zg]))
                inside = False

# === 出力（元点群＋緑点） ===
if len(green_chunks) > 0:
    green_pts = np.vstack(green_chunks)
    all_pts = np.vstack([pts, green_pts])

    # RGB16bit
    colors = np.zeros((len(all_pts), 3), dtype=np.uint16)
    colors[:len(pts)] = np.array([0, 0, 0], dtype=np.uint16)        # 元点群：黒
    colors[len(pts):] = np.array([0, 65535, 0], dtype=np.uint16)    # 航行可能空間：緑

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = all_pts.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las_out = laspy.LasData(header)
    las_out.x = all_pts[:, 0]
    las_out.y = all_pts[:, 1]
    las_out.z = all_pts[:, 2]
    las_out.red   = colors[:, 0]
    las_out.green = colors[:, 1]
    las_out.blue  = colors[:, 2]
    las_out.write(OUTPUT_LAS)

    print(f"✅ 出力完了：{OUTPUT_LAS}")
    print(f"   元点数: {len(pts):,d} / 緑点数: {len(green_pts):,d} / 合計: {len(all_pts):,d}")
else:
    print("⚠ 航行可能空間が見つかりませんでした")
