# -*- coding: utf-8 -*-
"""
【機能】Z=1.5m付近の断面点群から船体を検出し、XY平面上にBBoxを出力
--------------------------------------------------------------------
- 高さ1.5±0.1mの点を抽出
- DBSCANでクラスタリング（船体検出）
- 各クラスタの外接矩形を輪郭点列として出力
- 地図断面（白）＋BBox（緑）を同一LASに保存
--------------------------------------------------------------------
"""

import os
import numpy as np
import laspy
from sklearn.cluster import DBSCAN

# ===== パラメータ =====
INPUT_LAS  = r"/data/0925_ue_classified.las"
OUTPUT_LAS = r"/output/1010_bbox_around1p5m.las"

Z_TARGET   = 1.5      # 中心高さ[m]
Z_TOL      = 0.1      # ±0.1m の範囲を抽出
DBSCAN_EPS = 2.0      # クラスタ距離[m]
MIN_SAMPLES = 30       # クラスタ最小点数
MARGIN     = 0.5       # 外接矩形に余白[m]
RECT_STEP  = 0.2       # 矩形線の点間隔[m]

# ===== 関数 =====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):
        header.srs = src_header.srs
    return header

def make_rectangle_points(xmin, xmax, ymin, ymax, z, step=0.2):
    xs = np.arange(xmin, xmax+step, step)
    ys = np.arange(ymin, ymax+step, step)
    pts = []
    for x in xs: pts.append([x, ymax, z])
    for y in ys[::-1]: pts.append([xmax, y, z])
    for x in xs[::-1]: pts.append([x, ymin, z])
    for y in ys: pts.append([xmin, y, z])
    return np.array(pts)

# ===== メイン処理 =====
def main():
    las = laspy.read(INPUT_LAS)
    X, Y, Z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)

    # === 1. 高さ1.5±0.1mの点を抽出 ===
    m = (Z >= Z_TARGET - Z_TOL) & (Z <= Z_TARGET + Z_TOL)
    if np.count_nonzero(m) == 0:
        print("⚠ 指定高さに点がありません。")
        return
    section_pts = np.column_stack([X[m], Y[m], np.full(np.count_nonzero(m), Z_TARGET)])
    print(f"📏 抽出点数: {len(section_pts)}")

    # === 2. クラスタリング ===
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=MIN_SAMPLES).fit(section_pts[:, :2])
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"🚢 検出クラスタ数: {n_clusters}")

    bbox_points = []
    for cid in range(n_clusters):
        mask = (labels == cid)
        cpts = section_pts[mask]
        x_min, x_max = cpts[:,0].min()-MARGIN, cpts[:,0].max()+MARGIN
        y_min, y_max = cpts[:,1].min()-MARGIN, cpts[:,1].max()+MARGIN
        bbox_pts = make_rectangle_points(x_min, x_max, y_min, y_max, Z_TARGET, RECT_STEP)
        bbox_points.append(bbox_pts)
        print(f"  ↳ クラスタ{cid}: X[{x_min:.2f},{x_max:.2f}], Y[{y_min:.2f},{y_max:.2f}]")

    if len(bbox_points) == 0:
        print("⚠ クラスタが見つかりません。")
        return

    bbox_points = np.vstack(bbox_points)
    combined_pts = np.vstack([section_pts, bbox_points])

    # === 3. LAS出力 ===
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = combined_pts.T

    # 色分け: 地図(白)＋BBox(緑)
    n_map = len(section_pts)
    n_box = len(bbox_points)
    colors = np.zeros((n_map+n_box, 3), dtype=np.uint8)
    colors[:n_map] = [255,255,255]
    colors[n_map:] = [0,255,0]

    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = colors[:,0].astype(np.uint16)*256
        las_out.green = colors[:,1].astype(np.uint16)*256
        las_out.blue  = colors[:,2].astype(np.uint16)*256

    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)
    las_out.write(OUTPUT_LAS)
    print(f"✅ 出力完了: {OUTPUT_LAS}")
    print(f"   地図点数: {n_map:,}, BBox点数: {n_box:,}")

if __name__ == "__main__":
    main()
