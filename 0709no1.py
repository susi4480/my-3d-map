# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルのX範囲を自動検出し、0.5mごとにY-Z断面スライス
- 各スライスをLAS形式で保存
"""

import numpy as np
import laspy
import os

# === 入出力設定 ===
input_las = "/data/0731_suidoubasi_ue.las"
output_dir = "/output/slice_area_ue"
os.makedirs(output_dir, exist_ok=True)

# === スライス厚み設定 ===
slice_thickness = 0.2  # ±0.1m の厚さ
slice_interval = 0.5   # スライスの間隔（50cm）

# === LAS読み込み ===
print("📥 LASファイル読み込み中...")
las = laspy.read(input_las)
x = las.x

# === X範囲に基づいてスライス位置を決定 ===
x_min = np.floor(x.min())
x_max = np.ceil(x.max())
x_slice_positions = np.arange(x_min, x_max + slice_interval, slice_interval)

print(f"📊 X座標の範囲: {x_min:.2f} ～ {x_max:.2f}")
print(f"📏 スライス位置数: {len(x_slice_positions)}（{slice_interval}m 間隔）")

# === スライス処理 ===
print("✂ Y-Z断面スライス処理中...")
for x_center in x_slice_positions:
    x_low = x_center - slice_thickness / 2
    x_high = x_center + slice_thickness / 2

    mask = (x >= x_low) & (x <= x_high)
    if np.count_nonzero(mask) == 0:
        continue

    sliced_points = las.points[mask]
    new_las = laspy.LasData(las.header)
    new_las.points = sliced_points

    filename = f"slice_x_{x_center:.2f}m.las"
    out_path = os.path.join(output_dir, filename)
    new_las.write(out_path)
    print(f"✅ 出力: {filename}（点数: {len(sliced_points)}）")

print("✅ 全ての断面スライスを出力しました。")
