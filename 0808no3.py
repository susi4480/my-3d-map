# -*- coding: utf-8 -*-
"""
【機能（統合版）】
- 1スライスLAS（Y–Z断面）を occupancy grid 化（Z ≤ Z_LIMIT）
- モルフォロジー補間（クロージング）で壁の小隙間を埋め、安定した自由空間を得る
- 自由空間を対象に「最大内接長方形」を複数詰め、各長方形の縁を緑点として生成
- さらに「補間で埋まった差分セル」も緑点に変換（必要に応じてON/OFF可）
- 緑点と元の点群を統合し、CRS/スケール/オフセットを継承してLAS出力
"""

import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS  = r"C:\Users\user\Documents\lab\output_ply\slice_area_sita\slice_x_387426.00m.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0807_integrated_slice_x_387426_rects_and_morph.las"

# === パラメータ ===
Z_LIMIT          = 3.5
GRID_RES         = 0.10
MORPH_RADIUS     = 21                 # モルフォロジー構造要素半径
USE_MORPH_DIFF   = True               # クロージングで埋まった差分セルも緑化するか
MIN_RECT_SIZE    = 5                  # [セル] 長方形の最小高さ・幅
RECT_DIST_THRESHOLD = 15.0            # [m] 主長方形から遠い矩形は捨てる

# === ユーティリティ：最大内接長方形（自由空間=True のbitmapに対して） ===
def find_max_rectangle(bitmap: np.ndarray):
    """与えられたbool配列(True=自由空間)内で面積最大の長方形（True領域）を返す。
    戻り値: (top, left, height, width) ※インデックスは [Z(行), Y(列)] 基準
    """
    h, w = bitmap.shape
    # ヒストグラム方式：Trueを1、Falseを0扱い
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
                top_idx = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = height[top_idx] * width
                if area > max_area:
                    max_area = area
                    # 長方形のtop行は i - height[top_idx] + 1
                    max_rect = (i - height[top_idx] + 1,
                                (stack[-1] + 1 if stack else 0),
                                height[top_idx],
                                width)
    return max_rect

# === 1) LAS読み込み & Z制限 ===
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0

mask = xyz[:, 2] <= Z_LIMIT
xyz = xyz[mask]
rgb = rgb[mask]
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

# === 3) クロージングで壁の小隙間を補間 ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MORPH_RADIUS + 1, 2 * MORPH_RADIUS + 1))
closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

# 差分セル（クロージングで埋まったセル）
morph_diff_bitmap = (closed > 0) & (grid == 0)

# === 4) 自由空間bitmapを作成（壁=占有）→ 自由空間 = (closed==0) ===
#  ※クロージング後の占有を「壁」とみなし、その反転を自由空間とする
free_bitmap = (closed == 0)

# === 5) 自由空間に最大長方形を複数詰める ===
green_pts = []
main_rect_center = None

# 破壊的にfree_bitmapを埋めていくコピー
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
            # 遠すぎる矩形はスキップして、その領域だけ自由空間から除外
            free_work[top:top + h, left:left + w] = False
            continue

    # 矩形の縁を緑点に（セル中心）
    for zi in range(top, top + h):
        for yi in range(left, left + w):
            if zi in (top, top + h - 1) or yi in (left, left + w - 1):
                y = y_min + (yi + 0.5) * GRID_RES
                z = z_min + (zi + 0.5) * GRID_RES
                green_pts.append([X_FIXED, y, z])

    # 使用済みにする
    free_work[top:top + h, left:left + w] = False

# === 6) （オプション）クロージングで埋まった差分セルも緑点にする ===
if USE_MORPH_DIFF:
    zi_idx, yi_idx = np.where(morph_diff_bitmap)
    for zi, yi in zip(zi_idx, yi_idx):
        y = y_min + (yi + 0.5) * GRID_RES
        z = z_min + (zi + 0.5) * GRID_RES
        green_pts.append([X_FIXED, y, z])

green_pts = np.array(green_pts, dtype=float) if len(green_pts) else np.empty((0, 3), dtype=float)
green_rgb = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(green_pts), 1))

# === 7) 出力点群の統合（元点群 + 緑点） ===
all_pts = np.vstack([xyz, green_pts]) if len(green_pts) else xyz
all_rgb = np.vstack([rgb,  green_rgb]) if len(green_pts) else rgb
all_rgb16 = (all_rgb * 65535).astype(np.uint16)

# === 8) LAS保存（元ヘッダ継承） ===
try:
    header = las.header.copy()
except AttributeError:
    header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.scales  = las.header.scales.copy()
    header.offsets = las.header.offsets.copy()

las_out = laspy.LasData(header)
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
