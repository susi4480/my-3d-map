# -*- coding: utf-8 -*-
"""
【機能】地図点群とスキャン点群の2D重ね合わせ可視化（青×赤→重なり紫）
----------------------------------------------------------
入力:
  - map_las_path: 地図点群（raycast_world）
  - scan_las_path: 疑似観測点群（query_world）
出力:
  - overlap_map.png : 青=地図, 赤=スキャン, 重なり=紫
----------------------------------------------------------
"""

import laspy
import numpy as np
import matplotlib.pyplot as plt

# === 入力設定 ===
map_las_path  = "/output/1006_seq_raycast_world/scan_sector_0000_raycast_world.las"
scan_las_path = "/output/1006_seq_query_world/scan_sector_0000_query_world.las"
output_img    = "/output/overlap_map_0000.png"

# === LAS読み込み ===
map_las  = laspy.read(map_las_path)
scan_las = laspy.read(scan_las_path)

map_pts  = np.vstack([map_las.x, map_las.y, map_las.z]).T
scan_pts = np.vstack([scan_las.x, scan_las.y, scan_las.z]).T

# === 2D投影（X-Y） ===
map_xy  = map_pts[:, :2]
scan_xy = scan_pts[:, :2]

# === 範囲合わせ ===
all_xy = np.vstack([map_xy, scan_xy])
xmin, ymin = np.min(all_xy, axis=0)
xmax, ymax = np.max(all_xy, axis=0)

# === ヒートマップ解像度設定 ===
res = 0.1  # 1ピクセル=0.1m
nx = int((xmax - xmin) / res) + 1
ny = int((ymax - ymin) / res) + 1

# === 2Dビン化 ===
def to_grid(xy):
    ix = ((xy[:,0] - xmin) / res).astype(int)
    iy = ((xy[:,1] - ymin) / res).astype(int)
    grid = np.zeros((ny, nx), np.uint8)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    grid[iy[valid], ix[valid]] = 1
    return grid

map_grid  = to_grid(map_xy)
scan_grid = to_grid(scan_xy)

# === 3ch画像（青=map, 赤=scan） ===
img = np.zeros((ny, nx, 3), dtype=np.uint8)
img[..., 2] = map_grid * 255  # Blue
img[..., 0] = scan_grid * 255 # Red

# === 可視化 ===
plt.figure(figsize=(8,8))
plt.imshow(np.flipud(img))  # Y軸を上下反転
plt.title("Map (Blue) × Scan (Red) → Overlap (Purple)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("off")
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()

print(f"💾 保存完了: {output_img}")
