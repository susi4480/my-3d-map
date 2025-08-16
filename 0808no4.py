# -*- coding: utf-8 -*-
"""
【機能（統合版）】
- 1スライスLAS（Y–Z断面）を occupancy grid 化（Z ≤ Z_LIMIT）
- （任意）水面高さ anchor_z 近傍で見えた列だけ down-fill（船/橋の下を不可化）
- モルフォロジー補間（クロージング）
- 自由空間に「最大内接長方形」を複数詰め、縁を緑点化
- （任意）補間で埋まった差分セルも緑点化
- 緑点＋元点群を統合し、CRS/スケール/オフセット継承でLAS出力
"""

import os
import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS  = r"C:\Users\user\Documents\lab\output_ply\slice_area_sita\slice_x_389177.50m.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0808_integrated_slice_x_389177_rects_and_morph.las"

# === パラメータ ===
Z_LIMIT          = 2.0
GRID_RES         = 0.10
# ▼ 水面高さベースの down-fill
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z        = 1.9      # [m] 水面高さ
ANCHOR_TOL      = 0.15     # [m] 近傍幅（±）
# ▼ モルフォロジー
MORPH_RADIUS     = 21      # 構造要素半径[セル]
USE_MORPH_DIFF   = True    # クロージング差分セルも緑化
# ▼ 長方形
MIN_RECT_SIZE    = 5       # [セル]
RECT_DIST_THRESHOLD = 30.0 # [m] 主長方形から遠い矩形は捨てる

# === ユーティリティ：最大内接長方形（自由空間=True のbitmapに対して） ===
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

# === 水面高さ近傍のみ down-fill（列ごとに最上段の占有まで埋める） ===
def downfill_only_near_anchor(grid_uint8, z_min, grid_res, anchor_z=1.9, tol=0.15):
    """
    grid_uint8: uint8(0/255), shape=(gh,gw) 行=Z(下→上), 列=Y
    z_min: スライスZ最小
    grid_res: [m/セル]
    """
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
        # アンカー近傍に占有がある列だけ down-fill
        if np.any((idx >= i_lo) & (idx <= i_hi)):
            imax = idx.max()
            col[:imax + 1] = True
            occ[:, j] = col
    return (occ.astype(np.uint8) * 255)

# === 1) LAS読み込み & Z制限 ===
os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0
mask = xyz[:, 2] <= Z_LIMIT
xyz = xyz[mask]; rgb = rgb[mask]
if len(xyz) == 0:
    raise RuntimeError("⚠ Z制限後に点が存在しません。")

# Xはスライス厚さ方向と仮定し、固定値（平均）で出力
X_FIXED = float(np.mean(xyz[:, 0]))

# === 2) YZ平面へビットマップ化（占有=255, 空=0） ===
y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
gw = int(np.ceil((y_max - y_min) / GRID_RES))
gh = int(np.ceil((z_max - z_min) / GRID_RES))
grid = np.zeros((gh, gw), dtype=np.uint8)
for y, z in xyz[:, 1:3]:
    yi = int((y - y_min) / GRID_RES)
    zi = int((z - z_min) / GRID_RES)
    if 0 <= zi < gh and 0 <= yi < gw:
        grid[zi, yi] = 255  # 占有

# === 2.5) 水面高さ近傍だけ down-fill（船/橋の下を不可化）===
if USE_ANCHOR_DOWNFILL:
    grid = downfill_only_near_anchor(
        grid_uint8=grid,
        z_min=z_min,
        grid_res=GRID_RES,
        anchor_z=ANCHOR_Z,
        tol=ANCHOR_TOL
    )

# === 3) クロージングで壁の小隙間を補間 ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MORPH_RADIUS + 1, 2 * MORPH_RADIUS + 1))
closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

# 差分セル（クロージングで埋まったセル）
morph_diff_bitmap = (closed > 0) & (grid == 0)

# === 4) 自由空間bitmap（自由=closed==0） ===
free_bitmap = (closed == 0)

# === 5) 自由空間に最大長方形を複数詰める ===
green_pts = []
main_rect_center = None
free_work = free_bitmap.copy()
while np.any(free_work):
    top, left, h, w = find_max_rectangle(free_work)
    if h < MIN_RECT_SIZE or w < MIN_RECT_SIZE:
        break
    # 矩形中心（Y,Z）
    y_center = y_min + (left + w / 2.0) * GRID_RES
    z_center = z_min + (top  + h / 2.0) * GRID_RES
    if main_rect_center is None:
        main_rect_center = np.array([y_center, z_center])
    else:
        dist = np.linalg.norm(np.array([y_center, z_center]) - main_rect_center)
        if dist > RECT_DIST_THRESHOLD:
            free_work[top:top + h, left:left + w] = False
            continue
    # 縁セル→緑点（セル中心）
    for zi in range(top, top + h):
        for yi in range(left, left + w):
            if zi in (top, top + h - 1) or yi in (left, left + w - 1):
                y = y_min + (yi + 0.5) * GRID_RES
                z = z_min + (zi + 0.5) * GRID_RES
                green_pts.append([X_FIXED, y, z])
    free_work[top:top + h, left:left + w] = False

# === 6) （任意）クロージング差分セルも緑点化 ===
if USE_MORPH_DIFF:
    zi_idx, yi_idx = np.where(morph_diff_bitmap)
    for zi, yi in zip(zi_idx, yi_idx):
        y = y_min + (yi + 0.5) * GRID_RES
        z = z_min + (zi + 0.5) * GRID_RES
        green_pts.append([X_FIXED, y, z])

green_pts = np.array(green_pts, dtype=float) if len(green_pts) else np.empty((0, 3), dtype=float)
green_rgb = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(green_pts), 1))

# === 7) 出力点群の統合 ===
all_pts  = np.vstack([xyz, green_pts]) if len(green_pts) else xyz
all_rgb  = np.vstack([rgb,  green_rgb]) if len(green_pts) else rgb
all_rgb16 = (all_rgb * 65535).astype(np.uint16)

# === 8) LAS保存（ヘッダ継承＋ポイント配列を必要数で確保） ===
try:
    header = las.header.copy()
except AttributeError:
    header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.scales  = las.header.scales.copy()
    header.offsets = las.header.offsets.copy()

las_out = laspy.LasData(header)
n = all_pts.shape[0]
try:
    las_out.points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
except AttributeError:
    las_out.points = laspy.PointRecord.zeros(n, header=header)

las_out.x = all_pts[:, 0]
las_out.y = all_pts[:, 1]
las_out.z = all_pts[:, 2]
las_out.red   = all_rgb16[:, 0]
las_out.green = all_rgb16[:, 1]
las_out.blue  = all_rgb16[:, 2]
las_out.write(OUTPUT_LAS)

print("✅ 出力完了:", OUTPUT_LAS)
print(f"　元点数: {len(xyz):,d}")
print(f"　緑点数（長方形縁 + 差分セル）: {len(green_pts):,d}")
print(f"　down-fill: {USE_ANCHOR_DOWNFILL} @ {ANCHOR_Z}±{ANCHOR_TOL} m, GRID_RES={GRID_RES} m")
