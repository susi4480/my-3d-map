# -*- coding: utf-8 -*-
"""
【機能】
- /home/edu3/lab/output/new/ 内の .xyz をすべて処理
- 緯度経度 → UTM（EPSG:32654）変換
- 同名の .las ファイルに変換し、/output/new_las/ に保存
- 不正行（3列でない、数値変換失敗）はスキップ
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
import laspy

# === 入出力ディレクトリ設定 ===
input_dir = "/output/new/"
output_dir = "/output/new_las/"
os.makedirs(output_dir, exist_ok=True)

# === 緯度経度 → UTM（Zone 54N）変換器 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

# === .xyz ファイルをすべて取得 ===
xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
print(f"📂 入力ファイル数: {len(xyz_files)}")

for xyz_path in xyz_files:
    filename = os.path.basename(xyz_path)
    name_wo_ext = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_wo_ext}.las")

    # === 行単位で安全に読み込み（不正行はスキップ）===
    points = []
    with open(xyz_path, 'r') as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"⚠ スキップ: {filename} 行 {i} → {parts}")
                continue
            try:
                x, y, z = map(float, parts)
                points.append([x, y, z])
            except ValueError:
                print(f"⚠ 数値変換失敗: {filename} 行 {i} → {parts}")
                continue

    if not points:
        print(f"❌ 有効な点がないためスキップ: {filename}")
        continue

    data = np.array(points)

    # === 緯度経度 → UTM変換 ===
    x_utm, y_utm = transformer.transform(data[:, 1], data[:, 0])
    points_utm = np.column_stack((x_utm, y_utm, data[:, 2]))

    # === LAS作成・保存 ===
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points_utm.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_crs(CRS.from_epsg(32654))

    las = laspy.LasData(header)
    las.x = points_utm[:, 0]
    las.y = points_utm[:, 1]
    las.z = points_utm[:, 2]
    las.write(output_path)

    print(f"✅ 変換完了: {filename} → {output_path}")

print("\n🎉 すべてのファイルの変換が完了しました！")
