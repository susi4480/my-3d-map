# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをZ方向にスライス（5cm間隔）
- 各スライス内でY方向ビットマップ化し、1に挟まれた0を航行可能空間に
- Zスライス単位で独立に補間（他スライスに影響しない）
- 元の点群＋航行空間点（緑）をLAS出力
"""

#これからはこれの手法つかう


import numpy as np
import laspy
from scipy.spatial import cKDTree
import os
from tqdm import tqdm

# === 入出力設定 ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_388661.00m.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0731_zslice_gapspace_zslice_based.las"
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

# === パラメータ ===
Z_RES = 0.05     # Zスライス厚み
Y_RES = 0.1      # Y方向分解能
Z_LIMIT = 3.5    # Z制限（上限）

# === LAS読み込みとZ制限 ===
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T
pts = pts[pts[:, 2] <= Z_LIMIT]
if len(pts) == 0:
    raise RuntimeError("⚠ Z制限内の点が存在しません")

x_fixed = np.mean(pts[:, 0])
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
z_min, z_max = pts[:, 2].min(), Z_LIMIT

y_bins = np.arange(y_min, y_max + Y_RES, Y_RES)
z_bins = np.arange(z_min, z_max + Z_RES, Z_RES)
green_pts = []

# === Zスライス処理 ===
for zi in tqdm(range(len(z_bins) - 1), desc="Zスライス処理"):
    z0, z1 = z_bins[zi], z_bins[zi + 1]
    z_center = (z0 + z1) / 2
    mask = (pts[:, 2] >= z0) & (pts[:, 2] < z1)
    slice_pts = pts[mask]
    if len(slice_pts) == 0:
        continue

    # Y方向ビットマップ作成
    bitmap = np.zeros(len(y_bins) - 1, dtype=np.uint8)
    yi_indices = ((slice_pts[:, 1] - y_min) / Y_RES).astype(int)
    yi_indices = yi_indices[(yi_indices >= 0) & (yi_indices < len(bitmap))]
    bitmap[yi_indices] = 1

    # ギャップ抽出：1に挟まれた0（連続長の制限なし）
    inside = False
    gap_start = None
    for yi in range(1, len(bitmap) - 1):
        if bitmap[yi - 1] == 1 and bitmap[yi] == 0 and not inside:
            inside = True
            gap_start = yi
        if inside and bitmap[yi] == 0 and bitmap[yi + 1] == 1:
            # gap: [gap_start, yi]
            for yj in range(gap_start, yi + 1):
                y_center = y_min + (yj + 0.5) * Y_RES
                green_pts.append([x_fixed, y_center, z_center])
            inside = False

# === 出力（元点群＋緑点）===
if green_pts:
    green_pts = np.array(green_pts)
    all_pts = np.vstack([pts, green_pts])
    colors = np.zeros((len(all_pts), 3), dtype=np.uint16)
    colors[:len(pts)] = np.array([0, 0, 0])        # 元点群：黒
    colors[len(pts):] = np.array([0, 65535, 0])    # 航行可能空間：緑

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

    # 最近傍距離
    tree = cKDTree(green_pts)
    dists, _ = tree.query(green_pts, k=2)
    nearest_dists = dists[:, 1]
    print("✅ 最近傍距離の統計:")
    print(f"平均距離     : {np.mean(nearest_dists):.4f} m")
    print(f"中央値距離   : {np.median(nearest_dists):.4f} m")
    print(f"最小距離     : {np.min(nearest_dists):.4f} m")
    print(f"最大距離     : {np.max(nearest_dists):.4f} m")
    print(f"✅ 出力完了：{OUTPUT_LAS}（緑点数: {len(green_pts)}）")
else:
    print("⚠ 航行可能空間が見つかりませんでした")
