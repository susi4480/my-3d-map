# -*- coding: utf-8 -*-
"""
【機能】
- M5で得た航行可能空間（緑点群LAS）を入力
- 3Dアルファシェイプで航行可能空間を「1つの外殻メッシュ」に変換
- PLYまたはOBJ形式で出力
"""

import numpy as np
import laspy
import pygalmesh

# === 入出力 ===
INPUT_LAS = r"/output/M5_voxel_connected_green.las"
OUTPUT_PLY = r"/output/0828no3_M5_alpha_shape_mesh.ply"

# === αパラメータ ===
ALPHA = 2.0   # 単位[m]。大きいと隙間を埋める、小さいと細かく残す

# === LAS読み込み ===
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T
print(f"✅ 緑点数: {len(points):,}")

# === 3Dアルファシェイプでメッシュ生成 ===
print("⏳ 3Dアルファシェイプ計算中...")
mesh = pygalmesh.generate_alpha_shape(points, alpha=ALPHA, do_not_simplify=False)

# === メッシュ保存 ===
mesh.write(OUTPUT_PLY)
print(f"✅ 完了: {OUTPUT_PLY}")
