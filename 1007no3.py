# -*- coding: utf-8 -*-
"""
【機能】α-shapeで得た河川ポリゴン領域をマスクとして航行可能空間を抽出（M5方式）
---------------------------------------------------------
1. GeoJSON外郭（河川ポリゴン）を読み込み
2. LAS点群を読み込み、ポリゴン内部の点のみ抽出
3. Occupancyボクセル構築（Z制限あり）
4. 3D最大連結成分を抽出（航行可能空間として緑点出力）
5. 出力: river_masked.las, M5_voxel_connected_green.las
"""

import os
import numpy as np
import laspy
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy import ndimage

# ====== 入出力 ======
INPUT_LAS = "/workspace/fulldata/0925_ue_classified.las"
INPUT_GEOJSON = "/workspace/output/1007_river_outline_alpha_z1.geojson"
MASKED_LAS = "/workspace/output/1007_river_masked.las"
OUTPUT_GREEN_LAS = "/workspace/output/1007_M5_voxel_connected_green.las"

# ====== パラメータ ======
VOXEL_SIZE = 0.10   # 占有ボクセル解像度[m]
Z_MIN, Z_MAX = -6.0, 3.5
CONNECTIVITY = 6    # 3D接続: 6 or 26
MIN_COMPONENT_SIZE = 500  # ノイズ除去

# ------------------------------------------------------------
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    return header

def write_las(path, header_src, xyz, color=(0, 65535, 0)):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(xyz)
    if N == 0: return
    las_out.x, las_out.y, las_out.z = xyz[:,0], xyz[:,1], xyz[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red[:] = color[0]
        las_out.green[:] = color[1]
        las_out.blue[:] = color[2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    las_out.write(path)
    print(f"✅ 出力: {path} 点数={N:,}")

# ------------------------------------------------------------
print("📥 LAS読み込み中...")
las = laspy.read(INPUT_LAS)
pts = np.vstack([las.x, las.y, las.z]).T

print("📥 河川ポリゴン読み込み中...")
gdf = gpd.read_file(INPUT_GEOJSON)
polys = []
for geom in gdf.geometry:
    if geom.type == "Polygon": polys.append(geom)
    elif geom.type == "MultiPolygon": polys.extend(list(geom.geoms))
river_union = MultiPolygon(polys)

print("🧭 河川内部マスク適用中...")
mask = np.array([river_union.contains(Point(x,y)) for x,y in pts[:, :2]])
pts_river = pts[mask]
print(f"✅ 河川内部点数: {len(pts_river):,} / {len(pts):,}")

write_las(MASKED_LAS, las.header, pts_river, color=(0, 0, 65535))  # 青点で確認用

# ------------------------------------------------------------
print("🧱 Occupancyボクセル構築中...")
valid = (pts_river[:,2]>=Z_MIN) & (pts_river[:,2]<=Z_MAX)
pts_valid = pts_river[valid]

mins = pts_valid.min(axis=0)
maxs = pts_valid.max(axis=0)
dims = np.ceil((maxs - mins) / VOXEL_SIZE).astype(int)
occ = np.zeros(dims, dtype=bool)

idx = np.floor((pts_valid - mins) / VOXEL_SIZE).astype(int)
idx = np.clip(idx, 0, dims-1)
occ[idx[:,0], idx[:,1], idx[:,2]] = True

print("🔍 最大連結成分抽出中...")
labeled, n_labels = ndimage.label(occ, structure=ndimage.generate_binary_structure(3, CONNECTIVITY))
sizes = ndimage.sum(occ, labeled, range(1, n_labels+1))
largest = (sizes.argmax() + 1)
mask_main = labeled == largest
print(f"✅ 最大成分: {largest}/{n_labels} ({sizes.max():.0f} voxels)")

# ------------------------------------------------------------
print("📤 航行可能空間点群出力中...")
zz, yy, xx = np.where(mask_main)
pts_green = np.column_stack([xx, yy, zz]) * VOXEL_SIZE + mins
write_las(OUTPUT_GREEN_LAS, las.header, pts_green, color=(0, 65535, 0))
print("🎯 完了: 河川内部の航行可能空間を抽出しました。")
