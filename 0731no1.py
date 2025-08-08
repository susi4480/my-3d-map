# -*- coding: utf-8 -*-
"""
【機能】
- 指定スライスLASから Y–Zビットマップを生成
- 壁をモルフォロジー補間し、最大内接長方形を抽出
- その長方形を画像に重ねて可視化（matplotlib）
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.morphology import binary_closing, disk

# === 入力LASファイル ===
INPUT_LAS = r"C:\Users\user\Documents\lab\output_ply\slice_area\slice_x_388661.00m.las"

# === パラメータ ===
GRID_RES = 0.1
MORPH_RADIUS = 2

# === 最大内接長方形を求める関数（DP法）===
def find_max_rectangle(bitmap):
    h, w = bitmap.shape
    height = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)  # top, left, height, width

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

# === LAS読み込みと Y–Z ビットマップ作成 ===
las = laspy.read(INPUT_LAS)
xyz = np.vstack([las.x, las.y, las.z]).T

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

# === モルフォロジー補間と航行空間抽出 ===
closed = binary_closing(grid, disk(MORPH_RADIUS))
free = ~closed

# === 最大長方形検出 ===
top, left, h, w = find_max_rectangle(free)

# === 可視化 ===
fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(free, cmap='gray', origin='lower')
if h > 0 and w > 0:
    rect = patches.Rectangle((left, top), w, h, linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title("航行可能空間（最大内接長方形）")
else:
    ax.set_title("⚠️ 航行空間なし（矩形検出失敗）")

ax.set_xlabel("Y方向インデックス")
ax.set_ylabel("Z方向インデックス")
plt.tight_layout()
plt.show()
