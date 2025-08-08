# -*- coding: utf-8 -*-
"""
【機能】
- 補間済みPLYファイルから点群を読み込み
- Z ≤ Z_MAX の点を Y–Z 平面に投影し、占有グリッドを作成 (占有=1, 空=0)
- 連結成分ラベリングで「最大の非占有領域」を抽出
- その領域を find_contours で輪郭トレース
- 輪郭線を 3D (X = x_center 固定) のワイヤーフレーム化し、元点群と合成して PLY 出力
"""

import numpy as np
import open3d as o3d
from scipy.ndimage import label
from skimage.measure import find_contours

# === 入出力設定 ===
input_ply  = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_L_only.ply"
output_ply = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\navigable_region_binary.ply"

# === パラメータ ===
Z_MAX     = 3.0    # 航行可能上限高さ
grid_res  = 0.1    # グリッド解像度(m)
n_line_pts= 50     # 輪郭線を細分化する点数

# --- 1. PLY読み込み ---
pcd   = o3d.io.read_point_cloud(input_ply)
pts   = np.asarray(pcd.points)
cols  = np.asarray(pcd.colors)

# --- 2. Z制限してYZ投影 ---
mask_z      = pts[:,2] <= Z_MAX
pts_limited = pts[mask_z]
if len(pts_limited)==0:
    raise RuntimeError("Z ≤ Z_MAX の点群がありません")
YZ = pts_limited[:,[1,2]]

# --- 3. グリッド生成 ---
y_min, y_max = YZ[:,0].min(), YZ[:,0].max()
z_min, z_max = YZ[:,1].min(), Z_MAX
ny = int(np.ceil((y_max - y_min)/grid_res)) + 1
nz = int(np.ceil((z_max - z_min)/grid_res)) + 1

# --- 4. 占有グリッド (1=点群があるセル) ---
grid = np.zeros((nz,ny), dtype=np.uint8)
iy   = ((YZ[:,0] - y_min)/grid_res).astype(int)
iz   = ((YZ[:,1] - z_min)/grid_res).astype(int)
grid[iz, iy] = 1

# --- 5. 非占有セルを True としたマスク ---
free_space = (grid == 0)

# --- 6. 連結成分ラベリング & 最大成分選択 ---
labeled, n_comp = label(free_space)
if n_comp == 0:
    raise RuntimeError("非占有領域が見つかりません")
counts = np.bincount(labeled.ravel())
counts[0] = 0
best_label = counts.argmax()
region = (labeled == best_label)

# --- 7. 輪郭トレース (skimage) ---
contours = find_contours(region.astype(float), 0.5)
if not contours:
    raise RuntimeError("輪郭が検出できませんでした")
# 最長の輪郭
contour = max(contours, key=len)

# (row, col) → (Y, Z) 座標に戻す
yz_border = np.array([
    [y_min + c * grid_res, z_min + r * grid_res]
    for r, c in contour
])

# --- 8. 3Dワイヤーフレーム化 (X = 平均スライスX) ---
x0     = pts[:,0].mean()
verts3d= np.column_stack([
    np.full(len(yz_border), x0),
    yz_border[:,0],
    yz_border[:,1]
])

wire_pts = []
for i in range(len(verts3d)-1):
    wire_pts.append(np.linspace(verts3d[i], verts3d[i+1], n_line_pts))
# 閉ループ
wire_pts.append(np.linspace(verts3d[-1], verts3d[0], n_line_pts))
wire_pts = np.vstack(wire_pts)

# --- 9. 元点群＋ワイヤー合成 & 出力 ---
pcd_wire = o3d.geometry.PointCloud()
pcd_wire.points = o3d.utility.Vector3dVector(wire_pts)
pcd_wire.colors = o3d.utility.Vector3dVector(
    np.tile([0.0,1.0,0.0], (len(wire_pts),1))
)

pcd_all = pcd + pcd_wire
o3d.io.write_point_cloud(output_ply, pcd_all)
print(f"✅ バイナリマップ＋ラベリングで抽出した航行可能領域を出力: {output_ply}")
