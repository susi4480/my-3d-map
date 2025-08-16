# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルから航行可能空間（緑）を抽出し、
  上面・底面・側面で構成される閉じた三角形メッシュを生成・出力。
"""

import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS

# === 入出力 ===
input_las = "/output/0707_green_only_ue_simple2pts.las"
output_ply = "/output/0707no9_closed_mesh.ply"
crs_utm = CRS.from_epsg(32654)

# === パラメータ ===
grid_size = 0.5  # XY方向のグリッド分解能

# === LAS読み込みと緑抽出 ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T

mask_green = (colors[:, 0] == 0) & (colors[:, 1] == 255) & (colors[:, 2] == 0)
pts = points[mask_green]

if len(pts) == 0:
    raise RuntimeError("緑点群（航行可能空間）が見つかりません")

# === XYグリッドでZ最小・最大を取得（底面と上面）===
grid_idx = np.floor(pts[:, :2] / grid_size).astype(np.int32)
grid_dict = {}

for i, key in enumerate(map(tuple, grid_idx)):
    z = pts[i, 2]
    if key not in grid_dict:
        grid_dict[key] = {"top": pts[i], "bottom": pts[i]}
    else:
        if z > grid_dict[key]["top"][2]:
            grid_dict[key]["top"] = pts[i]
        if z < grid_dict[key]["bottom"][2]:
            grid_dict[key]["bottom"] = pts[i]

# === 座標配列作成 ===
top_points = []
bot_points = []
key_list = []

for key, val in grid_dict.items():
    top_points.append(val["top"])
    bot_points.append(val["bottom"])
    key_list.append(key)

top_points = np.array(top_points)
bot_points = np.array(bot_points)
key_list = np.array(key_list)

# === グリッドインデックス辞書（座標→index） ===
key_to_index = {tuple(k): i for i, k in enumerate(key_list)}

# === 三角形リスト生成 ===
triangles = []

def add_quad(idx00, idx01, idx10, idx11):
    # 四角形を三角形2つに分割（順序重要）
    triangles.append([idx00, idx10, idx11])
    triangles.append([idx00, idx11, idx01])

# === 上面・底面のメッシュ作成 ===
for i, (gx, gy) in enumerate(key_list):
    for dx, dy in [(1, 0), (0, 1)]:
        neighbor = (gx + dx, gy + dy)
        if neighbor in key_to_index:
            idx00 = i
            idx01 = key_to_index[(gx, gy + dy)]
            idx10 = key_to_index[(gx + dx, gy)]
            idx11 = key_to_index[(gx + dx, gy + dy)]

            # 上面
            if (gx + dx, gy + dy) in key_to_index:
                add_quad(idx00, idx01, idx10, idx11)

# === 側面（三角形2枚）を追加 ===
n = len(top_points)
for i in range(n):
    for j in range(n):
        if i != j and np.allclose(top_points[i][:2], bot_points[j][:2], atol=1e-3):
            # 対応する上下点が見つかったら、側面2枚（三角形）を追加
            top = top_points[i]
            bot = bot_points[j]
            if np.linalg.norm(top - bot) > 0.1:  # 高さ差があるもののみ
                curr_index = len(triangles) + len(top_points) + len(bot_points)
                triangles.append([i, j + n, (j + 1) % n + n])  # 三角形1
                triangles.append([i, (j + 1) % n + n, (i + 1) % n])  # 三角形2

# === メッシュ構築 ===
all_vertices = np.vstack([top_points, bot_points])
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

# 色（緑）
mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (len(all_vertices), 1)))

# === 書き出し ===
o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"✅ 出力完了: {output_ply}")
