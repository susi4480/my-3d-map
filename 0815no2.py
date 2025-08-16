# -*- coding: utf-8 -*-
"""
【機能】
- 指定ディレクトリ内のスライスLASファイル群を順に処理
- 各スライスで v–z occupancy を作成
- 各vで「点群が存在した最上位Z」より下を使えない領域としてマスク
- 航行可能空間（空きかつ使用可能）セルを緑点として復元
- 統合して1つのLASに保存
"""

import os
import numpy as np
import laspy
import cv2
from glob import glob


# === 入出力パス ===
SLICES_DIR = r"/output/overlap_band_slices/"
OUTPUT_LAS = r"/output/0815no2_navspace_ceiling_mask.las"

# === パラメータ ===
Z_LIMIT = 3.5
GRID_RES = 0.1
MIN_PTS_PER_SLICE = 30

def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):   header.srs  = src_header.srs
    if getattr(src_header, "vlrs", None):  header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def write_las(path, header_src, xyz):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = xyz.shape[0]
    las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    las_out.x = xyz[:,0]
    las_out.y = xyz[:,1]
    las_out.z = xyz[:,2]
    if {"red", "green", "blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full(N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)
    las_out.write(path)
    print(f"✅ 出力完了: {path}  点数: {N}")

def process_slice_las(las_path):
    las = laspy.read(las_path)
    X = np.asarray(las.x)
    Y = np.asarray(las.y)
    Z = np.asarray(las.z)

    mask = Z <= Z_LIMIT
    X = X[mask]; Y = Y[mask]; Z = Z[mask]
    if len(X) < MIN_PTS_PER_SLICE:
        return []

    v_vals = Y
    z_vals = Z

    v_min, v_max = v_vals.min(), v_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()
    gw = int(np.ceil((v_max - v_min)/GRID_RES))
    gh = int(np.ceil((z_max - z_min)/GRID_RES))
    occupancy = np.zeros((gh, gw), dtype=np.uint8)

    # ✅ 修正：グリッドインデックスが範囲外にならないように制限
    vi = ((v_vals - v_min)/GRID_RES).astype(int)
    zi = ((z_vals - z_min)/GRID_RES).astype(int)
    vi = np.clip(vi, 0, gw - 1)
    zi = np.clip(zi, 0, gh - 1)

    occupancy[zi, vi] = 1

    # 天井マスク生成
    mask_ceiling = np.zeros_like(occupancy, dtype=bool)
    for col in range(gw):
        rows = np.where(occupancy[:,col])[0]
        if len(rows) == 0: continue
        top = rows.max()
        mask_ceiling[:top, col] = True

    nav_mask = (~mask_ceiling) & (occupancy == 0)

    green_pts = []
    for zi, vi in zip(*np.where(nav_mask)):
        v = v_min + (vi + 0.5)*GRID_RES
        z = z_min + (zi + 0.5)*GRID_RES
        x_med = np.median(X)  # 仮に X 方向の中央値を使う
        green_pts.append([x_med, v, z])

    return green_pts

def main():
    slice_files = sorted(glob(os.path.join(SLICES_DIR, "*.las")))
    all_green = []

    for path in slice_files:
        green = process_slice_las(path)
        all_green.extend(green)

    if not all_green:
        print("❌ 航行可能空間なし")
        return

    ref_header = laspy.read(slice_files[0]).header
    all_green = np.array(all_green)
    write_las(OUTPUT_LAS, ref_header, all_green)

if __name__ == "__main__":
    main()
