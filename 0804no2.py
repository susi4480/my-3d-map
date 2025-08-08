# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをX方向にスライス（10m幅・9m間隔）
- 各スライス内のYZ断面をビットマップ化（Z×Y）
- モルフォロジー補間（binary_closing）で構造を補完
- 各Y列で「1に挟まれた0」のZ位置を航行空間（緑点）とする
- 元の点群＋緑点をLAS形式で出力
"""

import numpy as np
import laspy
import os
from tqdm import tqdm
from skimage.morphology import binary_closing, disk

# === 入出力設定 ===
INPUT_LAS = r"/output/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0801_no2_morpharogy_xslice.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# === パラメータ ===
Z_LIMIT = 3.5
X_SLICE_WIDTH = 10.0
X_SLICE_STEP = 9.0
Y_RES = 0.1
Z_RES = 0.05
MORPH_RADIUS = 2  # pixel

# === LAS読み込みとZ制限 ===
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
pts = pts[pts[:, 2] <= Z_LIMIT]
if len(pts) == 0:
    raise RuntimeError("⚠ Z制限内の点が存在しません")

x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
z_min, z_max = pts[:, 2].min(), Z_LIMIT

y_bins = np.arange(y_min, y_max + Y_RES, Y_RES)
z_bins = np.arange(z_min, z_max + Z_RES, Z_RES)

green_pts = []

# === X方向スライス処理 ===
x_starts = np.arange(x_min, x_max - X_SLICE_WIDTH + 1e-6, X_SLICE_STEP)
for x0 in tqdm(x_starts, desc="Xスライス処理"):
    x1 = x0 + X_SLICE_WIDTH
    x_center = (x0 + x1) / 2
    mask = (pts[:, 0] >= x0) & (pts[:, 0] < x1)
    slice_pts = pts[mask]
    if len(slice_pts) == 0:
        continue

    # === YZビットマップ化 ===
    bitmap = np.zeros((len(z_bins)-1, len(y_bins)-1), dtype=np.uint8)
    yi = ((slice_pts[:, 1] - y_min) / Y_RES).astype(int)
    zi = ((slice_pts[:, 2] - z_min) / Z_RES).astype(int)
    valid_mask = (yi >= 0) & (yi < bitmap.shape[1]) & (zi >= 0) & (zi < bitmap.shape[0])
    bitmap[zi[valid_mask], yi[valid_mask]] = 1

    # === モルフォロジー補間 ===
    filled = binary_closing(bitmap, footprint=disk(MORPH_RADIUS)).astype(np.uint8)

    # === 各Y列でZ方向に航行空間を探索 ===
    for yi in range(filled.shape[1]):
        col = filled[:, yi]
        inside = False
        gap_start = None
        for zi in range(1, len(col)-1):
            if col[zi-1] == 1 and col[zi] == 0 and not inside:
                inside = True
                gap_start = zi
            if inside and col[zi] == 0 and col[zi+1] == 1:
                for zj in range(gap_start, zi+1):
                    y_center = y_min + (yi + 0.5) * Y_RES
                    z_center = z_min + (zj + 0.5) * Z_RES
                    green_pts.append([x_center, y_center, z_center])
                inside = False

# === 出力（元点群＋緑点）===
if green_pts:
    green_pts = np.array(green_pts)
    all_pts = np.vstack([pts, green_pts])
    colors = np.zeros((len(all_pts), 3), dtype=np.uint16)
    colors[:len(pts)] = np.array([0, 0, 0])        # 元点群：黒
    colors[len(pts):] = np.array([0, 65535, 0])    # 航行空間：緑

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

    print(f"✅ 出力完了：{OUTPUT_LAS}（緑点数: {len(green_pts)}）")
else:
    print("⚠ 航行可能空間が見つかりませんでした")
