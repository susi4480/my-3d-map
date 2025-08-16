# -*- coding: utf-8 -*-
"""
【機能】
- スライスLAS群から以下を適用し、free空間が1点でもあるスライスをLAS出力
    - 白色点群（[65535,65535,65535]）を除外
    - Z ≤ 1.9m に制限
    - 統計的外れ値除去（SOR）
    - occupancy grid → closing → anchor-downfill
    - green点を world座標で出力（v–z断面のu=0）
- 各スライス出力を統合して1つのLASに結合出力

入力 : SLICE_DIR 中の .las（スライス済み）
出力 : OUTPUT_DIR に `_green_voxel.las` を個別出力 + 統合ファイルも出力
"""

import os
import numpy as np
import laspy
import cv2
from sklearn.neighbors import NearestNeighbors

# === 入出力 ===
SLICE_DIR     = r"/output/overlap_band_slices/"
OUTPUT_DIR    = r"/output/0815no3_voxel_slices_green/"
OUTPUT_MERGED_LAS = os.path.join(OUTPUT_DIR, "merged_green_voxel.las")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === パラメータ ===
Z_LIMIT     = 1.9
GRID_RES    = 0.1
MORPH_R     = 15
ANCHOR_Z    = 1.5
ANCHOR_TOL  = 0.5
USE_ANCHOR  = True
SOR_K       = 8
SOR_STD     = 1.0

all_coords = []
ref_header = None

def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    closed_bool = (closed_uint8 > 0)
    gh, gw = closed_bool.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh - 1, i_anchor + pad)
    if i_lo > gh - 1 or i_hi < 0:
        return (closed_bool.astype(np.uint8) * 255)
    out = closed_bool.copy()
    for j in range(gw):
        col = closed_bool[:, j]
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_and_free(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol):
    if len(points_vz) == 0:
        return None, None
    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max - v_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid_raw = np.zeros((gh, gw), dtype=np.uint8)

    vi = ((points_vz[:,0] - v_min) / grid_res).astype(int)
    zi = ((points_vz[:,1] - z_min) / grid_res).astype(int)
    ok = (vi >= 0) & (vi < gw) & (zi >= 0) & (zi < gh)
    grid_raw[zi[ok], vi[ok]] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)
    if use_anchor:
        closed = downfill_on_closed(closed, z_min, grid_res, anchor_z, anchor_tol)

    free_bitmap = ~(closed > 0)  # True=自由
    return free_bitmap, (v_min, z_min, gw, gh)

def sor_filter(xyz, k=8, std_ratio=1.0):
    if len(xyz) < k:
        return np.ones(len(xyz), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(xyz)
    dists, _ = nbrs.kneighbors(xyz)
    avg_dists = dists[:, 1:].mean(axis=1)
    std = avg_dists.std()
    mean = avg_dists.mean()
    return avg_dists <= mean + std_ratio * std

def process_slice(input_path, output_path):
    global all_coords, ref_header

    las = laspy.read(input_path)
    pts = np.vstack([las.x, las.y, las.z]).T

    # 白色点群を除外
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        is_not_white = ~((las.red==65535) & (las.green==65535) & (las.blue==65535))
        pts = pts[is_not_white]

    # Z ≤ Z_LIMIT でフィルタ
    pts = pts[pts[:,2] <= Z_LIMIT]

    # SORノイズ除去
    if len(pts) < SOR_K:
        return
    ok = sor_filter(pts, k=SOR_K, std_ratio=SOR_STD)
    pts = pts[ok]
    if len(pts) == 0:
        return

    # v–z断面に投影（u方向無視）
    c = np.mean(pts[:, :2], axis=0)
    dx = pts[:, 0] - c[0]
    dy = pts[:, 1] - c[1]
    v  = dy
    vz = np.column_stack([v, pts[:,2]])

    # free空間抽出
    free_bitmap, bbox = rectangles_and_free(vz, GRID_RES, MORPH_R, USE_ANCHOR, ANCHOR_Z, ANCHOR_TOL)
    if free_bitmap is None or not np.any(free_bitmap):
        return

    # 緑点生成（world座標へ）
    v_min, z_min, gw, gh = bbox
    coords = []
    for j in range(gw):
        for i in range(gh):
            if not free_bitmap[i, j]:
                continue
            v = v_min + (j + 0.5)*GRID_RES
            z = z_min + (i + 0.5)*GRID_RES
            x = c[0]
            y = c[1] + v
            coords.append([x, y, z])

    if len(coords) == 0:
        return

    # 書き出し
    N = len(coords)
    new_las = laspy.LasData(las.header)
    new_las.x = np.array([p[0] for p in coords])
    new_las.y = np.array([p[1] for p in coords])
    new_las.z = np.array([p[2] for p in coords])
    new_las.red   = np.zeros(N, dtype=np.uint16)
    new_las.green = np.full (N, 65535, dtype=np.uint16)
    new_las.blue  = np.zeros(N, dtype=np.uint16)
    new_las.write(output_path)
    print(f"✅ {os.path.basename(output_path)} 出力 ({N}点)")

    # 統合用に保存
    if ref_header is None:
        ref_header = las.header
    all_coords.append(np.array(coords))

# === 実行 ===
files = sorted([f for f in os.listdir(SLICE_DIR) if f.endswith(".las")])
for fname in files:
    in_path  = os.path.join(SLICE_DIR, fname)
    out_name = fname.replace(".las", "_green_voxel.las")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    process_slice(in_path, out_path)

# === 統合出力 ===
if all_coords and ref_header:
    merged = np.vstack(all_coords)
    N = len(merged)
    merged_las = laspy.LasData(ref_header)
    merged_las.x = merged[:, 0]
    merged_las.y = merged[:, 1]
    merged_las.z = merged[:, 2]
    merged_las.red   = np.zeros(N, dtype=np.uint16)
    merged_las.green = np.full (N, 65535, dtype=np.uint16)
    merged_las.blue  = np.zeros(N, dtype=np.uint16)
    merged_las.write(OUTPUT_MERGED_LAS)
    print(f"\n📦 統合ファイル出力完了: {OUTPUT_MERGED_LAS} ({N}点)")
else:
    print("❌ 緑点が見つかりませんでした")

print("✅ 完了")
