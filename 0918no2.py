# -*- coding: utf-8 -*-
"""
【機能】スライスごとのLASを統合
- /workspace/output/0917no2_6_3_filtered_slices 内の slice_????_rect.las をすべて読み込み
- 1つのLASファイルに統合して保存
"""

import os
import numpy as np
import laspy
from glob import glob

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/1014no5_final_refined/"
OUTPUT_LAS = "/workspace/output/1014no5_final_refined_merged.las"

def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    if getattr(src_header, "vlrs", None): header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def main():
    files = sorted(glob(os.path.join(INPUT_DIR, "slice_*_rect.las")))
    if not files:
        raise RuntimeError("入力ファイルが見つかりません")

    all_xyz = []
    first_header = None

    for f in files:
        las = laspy.read(f)
        if first_header is None:
            first_header = las.header
        xyz = np.vstack([las.x, las.y, las.z]).T
        all_xyz.append(xyz)
        print(f"✅ 読み込み: {f} 点数 {len(xyz)}")

    all_xyz = np.vstack(all_xyz)

    # 書き出し
    header = copy_header_with_metadata(first_header)
    las_out = laspy.LasData(header)
    N = len(all_xyz)
    las_out.x, las_out.y, las_out.z = all_xyz[:,0], all_xyz[:,1], all_xyz[:,2]

    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full(N, 65535, dtype=np.uint16)
        las_out.blue = np.zeros(N, dtype=np.uint16)

    os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)
    las_out.write(OUTPUT_LAS)
    print(f"🎉 出力: {OUTPUT_LAS} 点数 {N}")

if __name__=="__main__":
    main()
