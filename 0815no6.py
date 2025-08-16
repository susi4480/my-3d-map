# -*- coding: utf-8 -*-
"""
【機能】
- 指定フォルダ内のLASスライス群を処理（Z ≤ Z_LIMIT）
- Occupancy Gridで3Dボクセル化
- モルフォロジー閉処理で小さな隙間を補間
- 埋まったボクセルを緑点として抽出し、元点群と統合してLAS出力
"""

import os
import numpy as np
import laspy
from glob import glob
from scipy import ndimage

# === 入出力設定 ===
SLICES_DIR = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0815no6ver2_3dmorphfill_green.las"

# === パラメータ ===
Z_LIMIT = 1.9
GRID_RES = 0.1
MORPH_ITER = 2
MIN_PTS = 30

def read_and_filter_las(las_path):
    las = laspy.read(las_path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    if "red" in las.point_format.dimension_names:
        rgb = np.vstack([las.red, las.green, las.blue]).T
        mask = (xyz[:, 2] <= Z_LIMIT) & ~(np.all(rgb == 65535, axis=1))
    else:
        mask = (xyz[:, 2] <= Z_LIMIT)
    return xyz[mask] if np.sum(mask) >= MIN_PTS else None, las.header

def voxelize(points, mins, dims):
    indices = ((points - mins) / GRID_RES).astype(int)
    indices = np.clip(indices, 0, np.array(dims) - 1)
    occ = np.zeros(dims, dtype=np.uint8)
    occ[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return occ, indices

def process_slice(path):
    points, header = read_and_filter_las(path)
    if points is None:
        return None, None, None

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = np.ceil((maxs - mins) / GRID_RES).astype(int)
    occ, indices = voxelize(points, mins, dims)

    structure = np.ones((3, 3, 3), dtype=bool)
    occ_closed = ndimage.binary_closing(occ, structure=structure, iterations=MORPH_ITER)

    # 埋まった部分 = 補間された部分（1になった場所のうち、元々0だった所）
    filled = (occ_closed == 1) & (occ == 0)
    coords = np.array(np.where(filled)).T
    if len(coords) == 0:
        return points, np.empty((0, 3)), header

    fill_xyz = mins + (coords + 0.5) * GRID_RES
    return points, fill_xyz, header

def write_las(original_pts, filled_pts, header_src, path):
    all_pts = np.vstack([original_pts, filled_pts])
    header = laspy.LasHeader(point_format=header_src.point_format, version=header_src.version)
    header.scales = header_src.scales
    header.offsets = header_src.offsets
    if getattr(header_src, "srs", None): header.srs = header_src.srs
    if getattr(header_src, "vlrs", None): header.vlrs.extend(header_src.vlrs)
    if getattr(header_src, "evlrs", None): header.evlrs.extend(header_src.evlrs)

    las = laspy.LasData(header)
    las.x, las.y, las.z = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]

    N = len(all_pts)
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        red   = np.zeros(N, dtype=np.uint16)
        green = np.zeros(N, dtype=np.uint16)
        blue  = np.zeros(N, dtype=np.uint16)
        green[len(original_pts):] = 65535  # 緑で補間点を可視化
        las.red = red
        las.green = green
        las.blue = blue

    las.write(path)
    print(f"✅ 出力完了: {path} 点数: {N}")

def main():
    slice_paths = sorted(glob(os.path.join(SLICES_DIR, "*.las")))
    all_original = []
    all_filled = []
    ref_header = None

    for path in slice_paths:
        original, filled, header = process_slice(path)
        if original is None:
            continue
        all_original.append(original)
        all_filled.append(filled)
        if ref_header is None:
            ref_header = header

    if not all_original:
        print("❌ データなし")
        return

    all_original = np.vstack(all_original)
    all_filled = np.vstack(all_filled)
    write_las(all_original, all_filled, ref_header, OUTPUT_LAS)

if __name__ == "__main__":
    main()
