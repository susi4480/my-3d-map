# -*- coding: utf-8 -*-
"""
【機能】
- 1スライスLAS（Y–Z断面）を occupancy grid 化（Z ≤ 3.5m）
- 空間に複数の最大長方形を詰めて、各長方形の縁を緑点として抽出
- 主長方形の「辺」と他の長方形の「辺」の最近傍点距離で除外判定
- 緑点と元の点群を統合し、LAS出力（可視化なし）
"""

import numpy as np
import laspy

# === 入出力 ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0807_21_slice_x_388661_morphfill_green.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\0807_no4_slice_x_388661_with_rects.las"

# === パラメータ ===
Z_LIMIT = 3.5
GRID_RES = 0.1
MIN_RECT_SIZE = 5
X_FIXED = 388661.00
RECT_DIST_THRESHOLD = 1.0  # [m] 最近傍点距離でこれより遠い長方形は除外

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

# === 長方形の辺の点集合を生成 ===
def rect_edge_points(rect, y_min, z_min, res):
    top, left, h, w = rect
    y_start = y_min + left * res
    y_end   = y_min + (left + w) * res
    z_start = z_min + top * res
    z_end   = z_min + (top + h) * res

    y_vals = np.arange(y_start, y_end + res, res)
    z_vals = np.arange(z_start, z_end + res, res)

    edge_points = []

    # 上下辺
    for y in y_vals:
        edge_points.append((y, z_start))  # top
        edge_points.append((y, z_end))    # bottom

    # 左右辺
    for z in z_vals:
        edge_points.append((y_start, z))  # left
        edge_points.append((y_end, z))    # right

    return np.array(edge_points)

# === 2つの長方形の辺上の点集合から最小距離を計算 ===
def rect_min_distance_between_edges(rect1, rect2, y_min, z_min, res):
    pts1 = rect_edge_points(rect1, y_min, z_min, res)
    pts2 = rect_edge_points(rect2, y_min, z_min, res)
    dists = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
    return np.min(dists)

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
grid = np.zeros((gh, gw), dtype=bool)

for pt in xyz:
    yi = int((pt[1] - y_min) / GRID_RES)
    zi = int((pt[2] - z_min) / GRID_RES)
    if 0 <= zi < gh and 0 <= yi < gw:
        grid[zi, yi] = True

# === 最大長方形を順に詰めて緑点を生成（距離判定付き）===
free_bitmap = ~grid
green_pts = []
main_rect = None

while np.any(free_bitmap):
    rect = find_max_rectangle(free_bitmap)
    top, left, h, w = rect

    if h < MIN_RECT_SIZE or w < MIN_RECT_SIZE:
        break

    if main_rect is None:
        main_rect = rect  # 最初の長方形を基準とする
    else:
        dist = rect_min_distance_between_edges(main_rect, rect, y_min, z_min, GRID_RES)
        if dist > RECT_DIST_THRESHOLD:
            free_bitmap[top:top + h, left:left + w] = False
            continue

    # 長方形の縁を緑点に変換
    for i in range(top, top + h):
        for j in range(left, left + w):
            if i == top or i == top + h - 1 or j == left or j == left + w - 1:
                y = y_min + j * GRID_RES
                z = z_min + i * GRID_RES
                green_pts.append([X_FIXED, y, z])

    free_bitmap[top:top + h, left:left + w] = False  # 使用済みマーク

green_pts = np.array(green_pts)
green_rgb = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(green_pts), 1))

# === 出力用点群統合（緑点＋元点群）===
all_pts = np.vstack([xyz, green_pts])
all_rgb = np.vstack([rgb, green_rgb])
all_rgb16 = (all_rgb * 65535).astype(np.uint16)

# === LAS保存 ===
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

print(f"✅ 出力完了: {OUTPUT_LAS}（元点数: {len(xyz)}, 緑点数: {len(green_pts)}）")
