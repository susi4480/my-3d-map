# -*- coding: utf-8 -*-
"""
【機能】
- 1スライスLAS（Y–Z断面）を occupancy grid 化（Z ≤ 3.5m）
- モルフォロジー補間（クロージング）で空間の隙間を補間
- 補間領域を「緑点」として抽出し、元の点群と統合してLAS出力（可視化なし）
"""

import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_387021.00m.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0807_21_slice_x_387021_morphfill_green.las"

# === パラメータ ===
Z_LIMIT = 3.5
GRID_RES = 0.1
X_FIXED = 388661.00
MORPH_RADIUS = 21  # モルフォロジーの構造要素の半径

# === LAS読み込み・Z制限 ===
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0

mask = xyz[:, 2] <= Z_LIMIT
xyz = xyz[mask]
rgb = rgb[mask]

if len(xyz) == 0:
    raise RuntimeError("⚠ Z制限後に点が存在しません")

# === YZ平面ビットマップ化 ===
y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
gw = int(np.ceil((y_max - y_min) / GRID_RES))
gh = int(np.ceil((z_max - z_min) / GRID_RES))
grid = np.zeros((gh, gw), dtype=np.uint8)

for pt in xyz:
    yi = int((pt[1] - y_min) / GRID_RES)
    zi = int((pt[2] - z_min) / GRID_RES)
    if 0 <= zi < gh and 0 <= yi < gw:
        grid[zi, yi] = 255

# === モルフォロジー補間（クロージング）===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MORPH_RADIUS + 1, 2 * MORPH_RADIUS + 1))
closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

# === 補間された緑点を抽出 ===
diff = (closed > 0) & (grid == 0)
green_pts = []

for i, j in zip(*np.where(diff)):
    y = y_min + (j + 0.5) * GRID_RES
    z = z_min + (i + 0.5) * GRID_RES
    green_pts.append([X_FIXED, y, z])

green_pts = np.array(green_pts)
green_rgb = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(green_pts), 1))

# === 点群統合（元点群 + 緑点）===
all_pts = np.vstack([xyz, green_pts])
all_rgb = np.vstack([rgb, green_rgb])
all_rgb16 = (all_rgb * 65535).astype(np.uint16)

# === LAS出力 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = all_pts.min(axis=0)
header.scales = np.array([0.001, 0.001, 0.001])
las_out = laspy.LasData(header)
las_out.x = all_pts[:, 0]
las_out.y = all_pts[:, 1]
las_out.z = all_pts[:, 2]
las_out.red = all_rgb16[:, 0]
las_out.green = all_rgb16[:, 1]
las_out.blue = all_rgb16[:, 2]
las_out.write(OUTPUT_LAS)

print(f"✅ モルフォロジー補間（緑点）出力完了: {OUTPUT_LAS}")
print(f"　　元点数: {len(xyz)} / 緑点数: {len(green_pts)}")
