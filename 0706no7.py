# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルから航行可能空間（緑 [0,255,0]）だけを抽出
- ダウンサンプリングなし
- LAS形式で出力（CRSも保持）
"""

import numpy as np
import laspy
from pyproj import CRS

# === 入出力設定 ===
input_las = "/output/0704_method9_ue.las"
output_las = "/output/0707_green_only_ue.las"
crs_utm = CRS.from_epsg(32654)  # 適切なCRS（東京UTM Zone54N）

# === LAS読み込みと緑点抽出 ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).astype(np.float64).T
colors = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T

# === 緑（航行可能）点の抽出 ===
mask = (colors[:, 0] == 0) & (colors[:, 1] == 255) & (colors[:, 2] == 0)
points_navi = points[mask]
colors_navi = colors[mask]

if len(points_navi) == 0:
    raise RuntimeError("❌ 航行可能空間（緑）が見つかりませんでした")

print(f"✅ 航行可能点数: {len(points_navi):,}")

# === LASヘッダー作成 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.scales = np.array([0.001, 0.001, 0.001])  # 精度
header.offsets = points_navi.min(axis=0)
header.add_crs(crs_utm)

# === LASデータ作成と保存 ===
las_out = laspy.LasData(header)
las_out.x = points_navi[:, 0]
las_out.y = points_navi[:, 1]
las_out.z = points_navi[:, 2]
las_out.red   = colors_navi[:, 0]
las_out.green = colors_navi[:, 1]
las_out.blue  = colors_navi[:, 2]

las_out.write(output_las)
print(f"📤 LAS出力完了: {output_las}")
