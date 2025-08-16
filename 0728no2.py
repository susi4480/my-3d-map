# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをX方向に60cm幅スライス（±30cm）
- スライス間隔は50cm（前後10cmオーバーラップ）
- 各スライスを個別LASに出力（Raycastなし）
"""

import os
import numpy as np
import laspy

# === 入出力設定 ===
input_las = "/output/0725_suidoubasi_ue.las"
output_dir = "/output/slice_area_overlap_only/"
os.makedirs(output_dir, exist_ok=True)

# === スライスパラメータ ===
slice_width = 0.6     # スライス幅（60cm）
slice_step = 0.5      # スライス間隔（50cm）

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
pts_all = np.vstack([las.x, las.y, las.z]).T
cols_all = np.vstack([las.red, las.green, las.blue]).T

x_min, x_max = np.floor(pts_all[:, 0].min()), np.ceil(pts_all[:, 0].max())
x_centers = np.arange(x_min, x_max + slice_step, slice_step)

for i, x_center in enumerate(x_centers):
    x_low = x_center - slice_width / 2
    x_high = x_center + slice_width / 2
    mask = (pts_all[:, 0] >= x_low) & (pts_all[:, 0] <= x_high)
    if not np.any(mask):
        continue

    pts_slice = pts_all[mask]
    cols_slice = cols_all[mask]

    out_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    out_las.header = las.header
    out_las.x, out_las.y, out_las.z = pts_slice.T
    out_las.red, out_las.green, out_las.blue = cols_slice.T.astype(np.uint16)
    if hasattr(las.header, 'crs') and las.header.crs:
        out_las.header.crs = las.header.crs

    out_path = os.path.join(output_dir, f"slice_x_{x_center:.2f}m_only.las")
    out_las.write(out_path)
    print(f"✅ 出力: {out_path}（点数: {len(pts_slice)}）")

print("🎉 全スライス（オーバーラップ）の処理が完了しました。")
