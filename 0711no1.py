# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルを読み込み
- 赤色点群から以下の補間線を生成：
  (1) Z=0.0m 付近の左端赤点から垂直線（Z方向）
  (2) Z=0.0m 付近の右端赤点から垂直線（Z方向）
  (3) Z=1.3m 付近の赤点から左の垂直線まで水平線（Y方向）
- 緑で出力し、もとの点群と統合して PLY に保存
"""

import numpy as np
import laspy
import open3d as o3d
import os

# === 入出力設定 ===
input_las  = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_387183.00m.las"
output_ply = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_filled_Lshape.ply"

# === パラメータ ===
Z_VERT  = 0.01  # 垂直線（Z方向）の起点高さ
Z_HORIZ = 1.3   # 水平線（Y方向）の高さ
Z_TOL   = 0.05  # 上記の許容誤差
n_pts   = 50    # 線の点数

# === LAS読み込み ===
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).T
colors = np.column_stack([las.red, las.green, las.blue]).astype(float) / 65535.0


# === 赤点抽出 ===
red_mask = (colors[:, 0] > 0.9) & (colors[:, 1] < 0.1) & (colors[:, 2] < 0.1)
red_pts = points[red_mask]

if len(red_pts) == 0:
    raise RuntimeError("❌ 赤色点群が見つかりません")

# === Z ≈ 0.0 付近の赤点から垂直線を2本 ===
vert_pts = red_pts[np.abs(red_pts[:, 2] - Z_VERT) < Z_TOL]
if len(vert_pts) == 0:
    raise RuntimeError("❌ Z=0.0m付近の赤点が見つかりません")

y_median = np.median(red_pts[:, 1])
left_vert  = vert_pts[vert_pts[:, 1] < y_median]
right_vert = vert_pts[vert_pts[:, 1] >= y_median]

if len(left_vert) == 0:
    raise RuntimeError("❌ 左側の垂直線赤点が見つかりません")
if len(right_vert) == 0:
    raise RuntimeError("❌ 右側の垂直線赤点が見つかりません")

pL = left_vert[np.argmin(left_vert[:, 2])]   # 左下
pR = right_vert[np.argmin(right_vert[:, 2])] # 右下
lineL = np.linspace(pL, pL + [0, 0, 1.5], n_pts)
lineR = np.linspace(pR, pR + [0, 0, 1.5], n_pts)

# === Z ≈ 1.3m 付近の赤点から水平線（Y方向） ===
horiz_pts = red_pts[np.abs(red_pts[:, 2] - Z_HORIZ) < Z_TOL]
if len(horiz_pts) == 0:
    raise RuntimeError("❌ Z=1.3m付近の赤点が見つかりません")

pH = horiz_pts[np.argmax(horiz_pts[:, 1])]  # 最も右の赤点
lineH = np.linspace(pH, [pL[0], pL[1], pH[2]], n_pts)

# === 出力点群の生成 ===
new_pts = np.vstack([points, lineL, lineR, lineH])
green = np.tile([[0.0, 1.0, 0.0]], (len(lineL) + len(lineR) + len(lineH), 1))
new_cols = np.vstack([colors, green])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(new_pts)
pcd.colors = o3d.utility.Vector3dVector(new_cols)
o3d.io.write_point_cloud(output_ply, pcd)

print(f"✅ 緑の補間線を含めた点群を出力しました: {output_ply}")
