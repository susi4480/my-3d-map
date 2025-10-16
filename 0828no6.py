# -*- coding: utf-8 -*-
"""
【機能】2D αシェイプによるスライスごとの輪郭抽出
- LASファイルを読み込み
- X方向に一定間隔でスライス（例：50cm）
- 各スライス内で YZ平面に投影した2D点群からαシェイプ輪郭を抽出
- 輪郭線をPLYで保存（CloudCompareやBlenderで可視化可能）

必要ライブラリ：
pip install laspy numpy shapely alphashape open3d
"""

import os
import numpy as np
import laspy
import alphashape
from shapely.geometry import Polygon, Point
import open3d as o3d

# === 入出力設定 ===
INPUT_LAS = r"/data/0731_suidoubasi_ue.las"
OUTPUT_DIR = r"/output/alpha2d_slices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === パラメータ ===
SLICE_INTERVAL = 0.5  # スライス間隔 [m]
ALPHA = 0.2           # α値（小さいほど詳細に輪郭を追う）
Z_LIMIT = 1.9         # Z上限 [m]

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
points = np.vstack((las.x, las.y, las.z)).T
points = points[points[:, 2] <= Z_LIMIT]  # Z制限

# === X範囲設定 ===
min_x, max_x = points[:, 0].min(), points[:, 0].max()
slice_edges = np.arange(min_x, max_x + SLICE_INTERVAL, SLICE_INTERVAL)

# === スライス処理 ===
for i in range(len(slice_edges) - 1):
    x_min, x_max = slice_edges[i], slice_edges[i + 1]
    mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max)
    slice_points = points[mask]
    if len(slice_points) < 20:
        continue

    # YZ平面に投影
    yz_points = slice_points[:, 1:3]

    # αシェイプで輪郭抽出
    try:
        alpha_shape = alphashape.alphashape(yz_points, ALPHA)
        if alpha_shape.is_empty:
            continue
    except Exception:
        continue

    # 輪郭ポリゴンの座標を取得
    if isinstance(alpha_shape, Polygon):
        contour = np.array(alpha_shape.exterior.coords)
    else:
        # 複数ポリゴンの場合は最大面積のものを採用
        polygons = [poly for poly in alpha_shape.geoms if isinstance(poly, Polygon)]
        if not polygons:
            continue
        largest = max(polygons, key=lambda p: p.area)
        contour = np.array(largest.exterior.coords)

    # 輪郭を3D座標に変換（Xはスライス中央で固定）
    x_center = (x_min + x_max) / 2
    contour_3d = np.column_stack([np.full(len(contour), x_center), contour])

    # Open3DでPLY出力
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(contour_3d)
    pcd.paint_uniform_color([1, 0, 0])  # 赤で可視化
    output_path = os.path.join(OUTPUT_DIR, f"slice_{i:04d}_contour.ply")
    o3d.io.write_point_cloud(output_path, pcd)

    print(f"✅ スライス {i} の輪郭点数: {len(contour_3d)}")
