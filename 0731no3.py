# -*- coding: utf-8 -*-
"""
【機能】
- 1スライスLASからZ方向をビットマップ化（Yごと）
- 1に挟まれた0を航行可能空間とみなし、補完含めて緑点を抽出
- 航行可能空間をLAS出力（色: 緑）
- 最近傍距離の統計も表示
"""

import numpy as np
import laspy
from scipy.spatial import cKDTree
from tqdm import tqdm

# === 入出力設定 ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_388661.00m.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0731_zslice_gapspace_v2.las"

# === パラメータ ===
Z_RES = 0.1     # Z方向ビットマップ分解能（10cmに修正）
Y_RES = 0.1     # Y方向スライス幅
Z_LIMIT = 3.5   # Z制限（これ以上の点は使用しない）

# === LAS読み込みとZ制限 ===
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
pts = pts[pts[:, 2] <= Z_LIMIT]
if len(pts) == 0:
    raise RuntimeError("⚠ Z制限内の点が存在しません")

x_fixed = np.mean(pts[:, 0])  # このスライスのX座標は全て同一でよい
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
z_min, z_max = pts[:, 2].min(), Z_LIMIT

y_bins = np.arange(y_min, y_max + Y_RES, Y_RES)
z_bins = np.arange(z_min, z_max + Z_RES, Z_RES)

navigable_pts = []

# === 各YスライスでZビット処理 ===
for yi in tqdm(range(len(y_bins) - 1), desc="Yスライス処理"):
    y0, y1 = y_bins[yi], y_bins[yi + 1]
    mask = (pts[:, 1] >= y0) & (pts[:, 1] < y1)
    slice_pts = pts[mask]
    if len(slice_pts) == 0:
        continue

    # Z方向ビットマップ化（Zを縦、Yを横として扱う）
    bitmap = np.zeros(len(z_bins) - 1, dtype=np.uint8)
    zi_indices = ((slice_pts[:, 2] - z_min) / Z_RES).astype(int)
    zi_indices = zi_indices[(zi_indices >= 0) & (zi_indices < len(bitmap))]
    bitmap[zi_indices] = 1

    # 上からの補完（Zが高い方から下方向へ塗りつぶす）
    filled = np.maximum.accumulate(bitmap[::-1])[::-1]

    # 航行可能空間（1に挟まれた0を抽出）
    inside = False
    for zi in range(1, len(filled) - 1):
        prev1 = filled[zi - 1]
        next1 = filled[zi + 1]
        is_gap = (prev1 == 1 and next1 == 1 and bitmap[zi] == 0)
        if is_gap:
            inside = True
        elif inside and bitmap[zi] == 1:
            inside = False
        if inside and bitmap[zi] == 0:
            y_center = (y0 + y1) / 2
            z_center = z_min + (zi + 0.5) * Z_RES
            navigable_pts.append([x_fixed, y_center, z_center])

# === 出力LAS生成（緑）===
if len(navigable_pts) > 0:
    navigable_pts = np.array(navigable_pts)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = navigable_pts.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las_out = laspy.LasData(header)
    las_out.x = navigable_pts[:, 0]
    las_out.y = navigable_pts[:, 1]
    las_out.z = navigable_pts[:, 2]
    las_out.red   = np.zeros(len(navigable_pts), dtype=np.uint16)
    las_out.green = np.full(len(navigable_pts), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(navigable_pts), dtype=np.uint16)
    las_out.write(OUTPUT_LAS)

    # === 距離統計（最近傍）===
    tree = cKDTree(navigable_pts)
    dists, _ = tree.query(navigable_pts, k=2)
    nearest_dists = dists[:, 1]
    print("✅ 最近傍距離の統計:")
    print(f"平均距離     : {np.mean(nearest_dists):.4f} m")
    print(f"中央値距離   : {np.median(nearest_dists):.4f} m")
    print(f"最小距離     : {np.min(nearest_dists):.4f} m")
    print(f"最大距離     : {np.max(nearest_dists):.4f} m")
    print(f"✅ 出力完了：{OUTPUT_LAS}（緑点数: {len(navigable_pts)}）")
else:
    print("⚠ 航行可能空間が見つかりませんでした")
