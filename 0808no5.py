# -*- coding: utf-8 -*-
"""
【機能（長方形のみ抽出版・CSVなし）】
- 1スライスLAS（Y–Z断面）を occupancy grid 化（Z ≤ Z_LIMIT）
- （任意）水面高さ近傍だけ down-fill
- クロージング
- 自由空間に最大内接長方形を複数詰める
- ★長方形の「縁」だけ点群化（緑）して LAS 出力（元点群やモルフォ差分は出さない）
"""

import os
import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS   = r"C:\Users\user\Documents\lab\output_ply\slice_area_ue\slice_x_387021.00m.las"
OUTPUT_LAS  = r"C:\Users\user\Documents\lab\output_ply\0808no5_rectangles_only_slice_x_387021.las"

# === パラメータ ===
Z_LIMIT           = 1.0
GRID_RES          = 0.10
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z          = 1.0
ANCHOR_TOL        = 0.15
MORPH_RADIUS      = 21
MIN_RECT_SIZE     = 5        # [セル]
RECT_DIST_THRESHOLD = 5.0   # [m] 主長方形から遠い矩形は捨てる

# === 長方形検出 ===
def find_max_rectangle(bitmap: np.ndarray):
    h, w = bitmap.shape
    height = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)
    for i in range(h):
        for j in range(w):
            height[j] = height[j] + 1 if bitmap[i, j] else 0
        stack = []; j = 0
        while j <= w:
            cur = height[j] if j < w else 0
            if not stack or cur >= height[stack[-1]]:
                stack.append(j); j += 1
            else:
                top = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = height[top] * width
                if area > max_area:
                    max_area = area
                    max_rect = (i - height[top] + 1,
                                (stack[-1] + 1 if stack else 0),
                                height[top],
                                width)
    return max_rect

def downfill_only_near_anchor(grid_uint8, z_min, grid_res, anchor_z=1.9, tol=0.15):
    occ = (grid_uint8 > 0)
    gh, gw = occ.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh - 1, i_anchor + pad)
    if i_lo > gh - 1 or i_hi < 0:
        return (occ.astype(np.uint8) * 255)
    for j in range(gw):
        col = occ[:, j]
        idx = np.where(col)[0]
        if idx.size == 0:
            continue
        if np.any((idx >= i_lo) & (idx <= i_hi)):
            imax = idx.max()
            col[:imax + 1] = True
            occ[:, j] = col
    return (occ.astype(np.uint8) * 255)

# === 1) 読み込み & Z制限 ===
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)

las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
mask = xyz[:, 2] <= Z_LIMIT
xyz = xyz[mask]
if len(xyz) == 0:
    raise RuntimeError("Z制限後に点が存在しません。")

X_FIXED = float(np.mean(xyz[:, 0]))  # このスライスのX

# === 2) YZビットマップ ===
y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
gw = int(np.ceil((y_max - y_min) / GRID_RES))
gh = int(np.ceil((z_max - z_min) / GRID_RES))
grid = np.zeros((gh, gw), dtype=np.uint8)
for y, z in xyz[:, 1:3]:
    yi = int((y - y_min) / GRID_RES)
    zi = int((z - z_min) / GRID_RES)
    if 0 <= zi < gh and 0 <= yi < gw:
        grid[zi, yi] = 255

# === 2.5) 1.9m近傍だけ down-fill（任意）===
if USE_ANCHOR_DOWNFILL:
    grid = downfill_only_near_anchor(grid, z_min, GRID_RES, ANCHOR_Z, ANCHOR_TOL)

# === 3) クロージング ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MORPH_RADIUS+1, 2*MORPH_RADIUS+1))
closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

# 自由空間（= 0）
free_bitmap = (closed == 0)

# === 4) 長方形を抽出（縁だけ点群化） ===
green_pts = []
main_rect_center = None
free_work = free_bitmap.copy()

while np.any(free_work):
    top, left, h, w = find_max_rectangle(free_work)
    if h < MIN_RECT_SIZE or w < MIN_RECT_SIZE:
        break

    y_center = y_min + (left + w/2.0) * GRID_RES
    z_center = z_min + (top  + h/2.0) * GRID_RES
    if main_rect_center is None:
        main_rect_center = np.array([y_center, z_center])
    else:
        dist = np.linalg.norm(np.array([y_center, z_center]) - main_rect_center)
        if dist > RECT_DIST_THRESHOLD:
            free_work[top:top+h, left:left+w] = False
            continue

    # 縁のみ
    for zi in range(top, top+h):
        for yi in range(left, left+w):
            if zi in (top, top+h-1) or yi in (left, left+w-1):
                y = y_min + (yi + 0.5) * GRID_RES
                z = z_min + (zi + 0.5) * GRID_RES
                green_pts.append([X_FIXED, y, z])

    free_work[top:top+h, left:left+w] = False

green_pts = np.array(green_pts, dtype=float)
green_rgb16 = np.tile(np.array([[0, 65535, 0]], dtype=np.uint16), (len(green_pts), 1))

# === 5) LAS書き出し ===
header = las.header.copy()
las_out = laspy.LasData(header)
n = len(green_pts)
if n == 0:
    raise RuntimeError("長方形が見つかりません。")

las_out.points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
las_out.x = green_pts[:, 0]
las_out.y = green_pts[:, 1]
las_out.z = green_pts[:, 2]
las_out.red   = green_rgb16[:, 0]
las_out.green = green_rgb16[:, 1]
las_out.blue  = green_rgb16[:, 2]
las_out.write(OUTPUT_LAS)

print("✅ 出力完了:", OUTPUT_LAS)
print(f"  出力点数: {len(green_pts)}")
