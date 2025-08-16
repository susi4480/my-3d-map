# -*- coding: utf-8 -*-
"""
【機能】
- floor_sita / lidar_sita の .xyz をすべて読み込み
- 緯度経度 → UTM（EPSG:32654）変換
- 統合して 1 つの .las ファイル（fulldata_sita.las）として出力
- 不正行はスキップ
"""

import os
import glob
import numpy as np
from pyproj import Transformer, CRS
import laspy

# === 入力ディレクトリ ===
input_dirs = [
    "/data/fulldata/floor_sita/",
    "/data/fulldata/lidar_sita/"
]

# === 出力ファイルパス ===
output_path = "/output/fulldata_sita.las"

# === 緯度経度 → UTM（Zone 54N）変換器 ===
transformer = Transformer.from_crs("epsg:4326", "epsg:32654", always_xy=True)

# === 全点群を蓄積 ===
all_points = []

for input_dir in input_dirs:
    xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
    print(f"📂 {input_dir} → {len(xyz_files)} ファイル検出")

    for xyz_path in xyz_files:
        filename = os.path.basename(xyz_path)
        with open(xyz_path, 'r') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 3:
                    print(f"⚠ スキップ: {filename} 行 {i} → {parts}")
                    continue
                try:
                    x, y, z = map(float, parts)
                    all_points.append([x, y, z])
                except ValueError:
                    print(f"⚠ 数値変換失敗: {filename} 行 {i} → {parts}")
                    continue

# === チェック ===
if not all_points:
    raise RuntimeError("❌ 有効な点が見つかりませんでした。")

data = np.array(all_points)
print(f"\n✅ 総点数: {len(data)}")

# === 緯度経度 → UTM変換 ===
x_utm, y_utm = transformer.transform(data[:, 1], data[:, 0])
points_utm = np.column_stack((x_utm, y_utm, data[:, 2]))

# === LASファイル作成 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = points_utm.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
header.add_crs(CRS.from_epsg(32654))

las = laspy.LasData(header)
las.x = points_utm[:, 0]
las.y = points_utm[:, 1]
las.z = points_utm[:, 2]
las.write(output_path)

print(f"\n🎉 統合完了 → {output_path}")
