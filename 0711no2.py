# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルを読み込み
- 赤色点群から以下の補間線を生成：
  (1) 左下赤点 → 左赤点の高さまで垂直線
  (2) 右下赤点 → 右赤点の高さまで垂直線
  (3) 左赤点 → 左垂直線の上端まで水平線
  (4) 右赤点 → 右垂直線の上端まで水平線
- ※ 垂直線同士や左右の垂直線上端を結ぶ線は生成しない
- 緑で出力し、もとの点群と統合して PLY に保存
"""

import numpy as np
import laspy
import open3d as o3d
import os

# === 入出力設定 ===
input_las  = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_387183.00m.las"
output_ply = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\0712_slice_x_387183_L_only.ply"

# === パラメータ ===
Z_VERT  = 0.01    # 垂直線の基準高さ（下端）
Z_HORIZ = 1.3     # 水平線を引く赤点の高さ
Z_TOL   = 0.35    # 高さ許容誤差
n_pts   = 50      # 線分を構成する点数

# === LAS読み込み ===
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).T
cols = np.column_stack([las.red, las.green, las.blue]).astype(float) / 65535.0

# === 赤点のみ抽出 ===
mask_red = (cols[:,0] > 0.9) & (cols[:,1] < 0.1) & (cols[:,2] < 0.1)
red = pts[mask_red]
if len(red)==0:
    raise RuntimeError("赤点が見つかりません")

# === 垂直線用 下端赤点（Z≈0） ===
vpts = red[np.abs(red[:,2]-Z_VERT)<Z_TOL]
if len(vpts)==0:
    raise RuntimeError("Z≈0mの赤点が見つかりません")
y_med = np.median(red[:,1])
left_v  = vpts[vpts[:,1]< y_med]
right_v = vpts[vpts[:,1]>=y_med]
if len(left_v)==0 or len(right_v)==0:
    raise RuntimeError("左右の垂直線下端赤点が見つかりません")
pL = left_v[np.argmin(left_v[:,2])]    # 左下
pR = right_v[np.argmin(right_v[:,2])]  # 右下

# === 水平線用 赤点（Z≈1.3） ===
hpts = red[np.abs(red[:,2]-Z_HORIZ)<Z_TOL]
if len(hpts)==0:
    raise RuntimeError("Z≈1.3mの赤点が見つかりません")
left_h  = hpts[hpts[:,1]< y_med]
right_h = hpts[hpts[:,1]>=y_med]
if len(left_h)==0 or len(right_h)==0:
    raise RuntimeError("左右の水平用赤点が見つかりません")
pHL = left_h[np.argmin(left_h[:,1])]   # 左端赤点
pHR = right_h[np.argmax(right_h[:,1])] # 右端赤点

# === 垂直線の上端（赤点の高さに合わせる） ===
pL_top = np.array([pL[0], pL[1], pHL[2]])
pR_top = np.array([pR[0], pR[1], pHR[2]])

# === 線の生成 ===
lineL      = np.linspace(pL,     pL_top, n_pts)   # 左垂直線
lineR      = np.linspace(pR,     pR_top, n_pts)   # 右垂直線
lineH_left = np.linspace(pHL,    pL_top, n_pts)   # 左赤点→左垂直線上端
lineH_right= np.linspace(pHR,    pR_top, n_pts)   # 右赤点→右垂直線上端

# === 点群統合＆出力 ===
new_pts = np.vstack([pts, lineL, lineR, lineH_left, lineH_right])
green   = np.tile([[0.0,1.0,0.0]], (new_pts.shape[0]-pts.shape[0],1))
new_cols= np.vstack([cols, green])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(new_pts)
pcd.colors = o3d.utility.Vector3dVector(new_cols)
o3d.io.write_point_cloud(output_ply, pcd)

print(f"✅ L字型補間（垂直線と左右赤点→垂直線上端の水平線）のみを出力しました: {output_ply}")
