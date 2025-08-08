# -*- coding: utf-8 -*-
"""
【機能】
- 1スライスLAS（Y–Z断面）を occupancy grid 化（Z ≤ 3.5m）
- 空間に複数の最大長方形を詰めて、各長方形の縁を緑点として抽出
- 主長方形から遠い長方形はスキップ
- 緑点と元の点群を統合し、LAS出力（CRS情報も継承・可視化なし）
"""

import numpy as np
import laspy

# === 入出力 ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0807_21_slice_x_387021_morphfill_green.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0807_no3_slice_x_387021_with_rects.las"

# === パラメータ ===
Z_LIMIT = 3.5
GRID_RES = 0.1
MIN_RECT_SIZE = 5
RECT_DIST_THRESHOLD = 15.0  # [m] 主長方形からこの距離以上ならスキップ

# === 最大内接長方形を求める関数 ===
def find_max_rectangle(bitmap):
    h, w = bitmap.shape
    height = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)

    for i in range(h):
        for j in range(w):
            height[j] = height[j] + 1 if bitmap[i, j] else 0

        stack = []
        j = 0
        while j <= w:
            cur_height = height[j] if j < w else 0
            if not stack or cur_height >= height[stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = height[top] * width
                if area > max_area:
                    max_area = area
                    max_rect = (i - height[top] + 1, stack[-1] + 1 if stack else 0, height[top], width)
    return max_rect

# === LAS読み込み・Z制限 ===
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0

mask = xyz[:, 2] <= Z_LIMIT
xyz = xyz[mask]
rgb = rgb[mask]

if len(xyz) == 0:
    raise RuntimeError("⚠ Z制限後に点が存在しません")

# ✅ X座標を元点群の平均で固定（固定値から動的に変更）
X_FIXED = np.mean(xyz[:, 0])

# === YZ平面ビットマップ化 ===
y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
gw = int(np.ceil((y_max - y_min) / GRID_RES))
gh = int(np.ceil((z_max - z_min) / GRID_RES))
grid = np.zeros((gh, gw), dtype=bool)

for pt in xyz:
    yi = int((pt[1] - y_min) / GRID_RES)
    zi = int((pt[2] - z_min) / GRID_RES)
    if 0 <= zi < gh and 0 <= yi < gw:
        grid[zi, yi] = True

# === 最大長方形を順に詰めて緑点を生成（主領域から遠いものは除外）===
free_bitmap = ~grid
green_pts = []
main_rect_center = None

while np.any(free_bitmap):
    top, left, h, w = find_max_rectangle(free_bitmap)
    if h < MIN_RECT_SIZE or w < MIN_RECT_SIZE:
        break

    # === 中心座標（Y, Z）を計算 ===
    y_center = y_min + (left + w / 2) * GRID_RES
    z_center = z_min + (top + h / 2) * GRID_RES
    center = np.array([y_center, z_center])

    if main_rect_center is None:
        main_rect_center = center  # 最初の長方形を主領域とする
    else:
        dist = np.linalg.norm(center - main_rect_center)
        if dist > RECT_DIST_THRESHOLD:
            free_bitmap[top:top + h, left:left + w] = False
            continue

    # === 長方形の縁を緑点に変換（セル中心に配置）===
    for i in range(top, top + h):
        for j in range(left, left + w):
            if i == top or i == top + h - 1 or j == left or j == left + w - 1:
                y = y_min + (j + 0.5) * GRID_RES
                z = z_min + (i + 0.5) * GRID_RES
                green_pts.append([X_FIXED, y, z])

    # === この領域は処理済みにマーク ===
    free_bitmap[top:top + h, left:left + w] = False

green_pts = np.array(green_pts)
green_rgb = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(green_pts), 1))

# === 出力用点群統合（緑点＋元点群）===
all_pts = np.vstack([xyz, green_pts])
all_rgb = np.vstack([rgb, green_rgb])
all_rgb16 = (all_rgb * 65535).astype(np.uint16)

# ✅ 元のLASの header を継承（スケール・オフセット・CRS含む）
try:
    header = las.header.copy()
except AttributeError:
    # laspy 2.1系などで .copy() がない場合の互換処理
    header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.scales = las.header.scales.copy()
    header.offsets = las.header.offsets.copy()

# === LAS保存 ===
las_out = laspy.LasData(header)
las_out.x = all_pts[:, 0]
las_out.y = all_pts[:, 1]
las_out.z = all_pts[:, 2]
las_out.red = all_rgb16[:, 0]
las_out.green = all_rgb16[:, 1]
las_out.blue = all_rgb16[:, 2]
las_out.write(OUTPUT_LAS)

print(f"✅ 出力完了: {OUTPUT_LAS}")
print(f"　　元点数: {len(xyz)} / 緑点数: {len(green_pts)}")
