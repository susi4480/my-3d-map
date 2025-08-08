# -*- coding: utf-8 -*-
"""
【機能】LASファイルから X Y Z 座標を抽出し、.xyz テキスト形式で保存
"""

import laspy
import numpy as np

# === 入出力パス設定 ===
input_las  = r"C:\Users\user\Documents\lab\output_ply\0711_suidoubasi_floor_ue_25.las"
output_xyz = r"C:\Users\user\Documents\lab\output_ply\0712_suidoubasi_floor_ue_25.xyz"

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T

# === .xyzとして保存（3列）===
np.savetxt(output_xyz, points, fmt="%.3f")

print("✅ 変換完了")
print(f"  入力: {input_las}")
print(f"  出力: {output_xyz}")
print(f"  点数: {len(points):,}")
