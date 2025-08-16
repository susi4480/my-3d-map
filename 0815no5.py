# -*- coding: utf-8 -*-
"""
【機能】
- 指定フォルダ内のLASスライス群を処理（Z ≤ Z_LIMIT）
- Occupancy Gridを構築（3D voxel化）
- Morphological Closingで隙間補間
- 空き空間領域をndimage.label()でラベル分割
- 各ラベル領域ごとに「最小直方体の縁」を緑点として出力
- 全緑点を統合し、LASで保存（RGB=[0,65535,0]）
"""

import os
import numpy as np
import laspy
from glob import glob
from scipy import ndimage

# === 入出力 ===
SLICES_DIR = r"/output/overlap_band_slices/"
OUTPUT_LAS = r"/output/0815no5_m0box_edges_green.las"

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
    return occ

def get_box_edge_voxels(mins_v, maxs_v):
    voxels = []
    for x in range(mins_v[0], maxs_v[0] + 1):
        for y in [mins_v[1], maxs_v[1]]:
            for z in [mins_v[2], maxs_v[2]]:
                voxels.append([x, y, z])
    for y in range(mins_v[1], maxs_v[1] + 1):
        for x in [mins_v[0], maxs_v[0]]:
            for z in [mins_v[2], maxs_v[2]]:
                voxels.append([x, y, z])
    for z in range(mins_v[2], maxs_v[2] + 1):
        for x in [mins_v[0], maxs_v[0]]:
            for y in [mins_v[1], maxs_v[1]]:
                voxels.append([x, y, z])
    return np.unique(voxels, axis=0)

def process_slice(path):
    points, header = read_and_filter_las(path)
    if points is None:
        return [], None

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = np.ceil((maxs - mins) / GRID_RES).astype(int)
    occ = voxelize(points, mins, dims)

    structure = np.ones((3, 3, 3), dtype=bool)
    occ_closed = ndimage.binary_closing(occ, structure=structure, iterations=MORPH_ITER)

    free = (occ_closed == 0)
    labeled, num = ndimage.label(free)
    green_voxels = []

    for label in range(1, num + 1):
        region = (labeled == label)
        coords = np.array(np.where(region)).T
        if coords.size == 0:
            continue
        mins_v = coords.min(axis=0)
        maxs_v = coords.max(axis=0)
        edge_voxels = get_box_edge_voxels(mins_v, maxs_v)
        world_coords = mins + (edge_voxels + 0.5) * GRID_RES
        green_voxels.append(world_coords)

    if not green_voxels:
        return [], None
    return np.vstack(green_voxels), header

def write_las(xyz, header_src, path):
    header = laspy.LasHeader(point_format=header_src.point_format, version=header_src.version)
    header.scales = header_src.scales
    header.offsets = header_src.offsets
    if getattr(header_src, "srs", None): header.srs = header_src.srs
    if getattr(header_src, "vlrs", None): header.vlrs.extend(header_src.vlrs)
    if getattr(header_src, "evlrs", None): header.evlrs.extend(header_src.evlrs)

    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    N = len(xyz)
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        las.red = np.zeros(N, dtype=np.uint16)
        las.green = np.full(N, 65535, dtype=np.uint16)
        las.blue = np.zeros(N, dtype=np.uint16)
    las.write(path)
    print(f"✅ 出力完了: {path} 点数: {N}")

def main():
    slice_paths = sorted(glob(os.path.join(SLICES_DIR, "*.las")))
    all_green = []
    ref_header = None

    for path in slice_paths:
        green, header = process_slice(path)
        if len(green) == 0:
            continue
        all_green.append(green)
        if ref_header is None:
            ref_header = header

    if not all_green:
        print("❌ 航行空間なし")
        return
    all_green = np.vstack(all_green)
    write_las(all_green, ref_header, OUTPUT_LAS)

if __name__ == "__main__":
    main()
