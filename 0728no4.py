# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルを0.5m間隔でY-Z断面スライス（±0.1m厚）
- 各スライスに対して：
    - Z > 3.2 除去
    - 赤点のみ抽出
    - Y-Z断面でMorphological補間
    - 補間点（白）として追加し、スライス点群と統合して出力
"""

import os
import numpy as np
import laspy
from scipy.spatial import cKDTree
from skimage.morphology import binary_closing, disk

# === 入出力設定 ===
input_las = "/data/0704_suidoubasi_ue.las"
output_dir = "/output/slice_area_morphology/"
os.makedirs(output_dir, exist_ok=True)

# === スライス・補間設定 ===
slice_thickness = 0.2
slice_interval = 0.5
z_cutoff = 3.2
grid_res = 0.05
morph_radius = 2

# === LAS読み込み ===
las = laspy.read(input_las)
pts_all = np.vstack([las.x, las.y, las.z]).T
cols_all = np.vstack([las.red, las.green, las.blue]).T

x_min, x_max = np.floor(pts_all[:,0].min()), np.ceil(pts_all[:,0].max())
x_centers = np.arange(x_min, x_max + slice_interval, slice_interval)

for i, x_center in enumerate(x_centers):
    x_low = x_center - slice_thickness/2
    x_high = x_center + slice_thickness/2
    mask = (pts_all[:,0] >= x_low) & (pts_all[:,0] <= x_high)
    if not np.any(mask): continue

    pts_slice = pts_all[mask]
    cols_slice = cols_all[mask]

    z_mask = pts_slice[:,2] <= z_cutoff
    red_mask = (cols_slice[:,0] > cols_slice[:,1]) & (cols_slice[:,0] > cols_slice[:,2])
    final_mask = z_mask & red_mask

    red_pts = pts_slice[final_mask]
    if len(red_pts) < 5:
        pts_out = pts_slice
        cols_out = cols_slice
    else:
        y, z = red_pts[:,1], red_pts[:,2]
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()
        ny = int(np.ceil((y_max - y_min) / grid_res)) + 1
        nz = int(np.ceil((z_max - z_min) / grid_res)) + 1
        grid = np.zeros((ny, nz), dtype=bool)

        iy = ((y - y_min) / grid_res).astype(int)
        iz = ((z - z_min) / grid_res).astype(int)
        grid[iy, iz] = True

        closed = binary_closing(grid, disk(morph_radius))
        interp_mask = closed & (~grid)
        iy_new, iz_new = np.where(interp_mask)
        y_new = y_min + iy_new * grid_res
        z_new = z_min + iz_new * grid_res
        x_new = np.full_like(y_new, x_center)
        interp_pts = np.stack([x_new, y_new, z_new], axis=1)
        interp_cols = np.tile([65535, 65535, 65535], (len(interp_pts), 1))

        pts_out = np.vstack([pts_slice, interp_pts])
        cols_out = np.vstack([cols_slice, interp_cols])

    out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out_las.header = las.header
    out_las.x, out_las.y, out_las.z = pts_out.T
    out_las.red, out_las.green, out_las.blue = cols_out.T.astype(np.uint16)
    if hasattr(las.header, 'crs') and las.header.crs:
        out_las.header.crs = las.header.crs

    out_path = os.path.join(output_dir, f"slice_x_{x_center:.2f}m_interp.las")
    out_las.write(out_path)
    print(f"✅ [{i+1}/{len(x_centers)}] 出力: {out_path}（点数: {len(pts_out)}）")

print("🎉 Morphological補間処理 完了")
