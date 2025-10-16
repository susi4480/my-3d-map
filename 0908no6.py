# -*- coding: utf-8 -*-
"""
M6方式：内部空間抽出（柱除外, LAS 1.4対応版）
-----------------------------------
【機能】
- LAS入力を読み込み
- Occupancyグリッドを構築（Z制限付き）
- 内外判定（最大連結成分）で「内部空間」を抽出
- 小規模な孤立成分（柱など）を除外
- 出力は：
  - 内部空間LAS（緑点, classification=Xインデックス 16bit）
  - 属性LAS（スライス番号, 16bit）
  - シェルPLY（外縁点）
-----------------------------------
"""

import os
import numpy as np
import laspy
import open3d as o3d
from scipy import ndimage

# ===== パラメータ =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS_INTERNAL = "/output/0908M6_internal_space.las"
OUTPUT_LAS_ATTR     = "/output/0908M6_attr_points.las"
OUTPUT_PLY_SHELL    = "/output/0908M6_shell.ply"

Z_LIMIT   = 1.9
GRID_RES  = 0.5
MIN_PTS   = 20
MIN_SIZE  = 500   # 小さい連結成分は柱とみなして除外

for path in [OUTPUT_LAS_INTERNAL, OUTPUT_LAS_ATTR, OUTPUT_PLY_SHELL]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ===== 保存関数 =====
def save_las(path, points, classification=None):
    if points is None or len(points) == 0:
        print(f"⚠️ LAS出力なし: {path}")
        return
    # LAS 1.4 + PointFormat 7 (RGB + uint16 classification)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]

    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = points[:, 0], points[:, 1], points[:, 2]

    if classification is not None:
        las_out.classification = np.asarray(classification, dtype=np.uint16)

    # RGBを緑固定
    las_out.red   = np.zeros(len(points), dtype=np.uint16)
    las_out.green = np.full(len(points), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(points), dtype=np.uint16)

    las_out.write(path)
    print(f"✅ LAS出力: {path} 点数: {len(points)}")

def save_ply_points(path, points):
    if points is None or len(points) == 0:
        print(f"⚠️ 点群なし: {path}")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)
    print(f"✅ PLY出力: {path} 点数: {len(points)}")

# ===== メイン処理 =====
print("📥 LAS読み込み中...")
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

# Occupancy構築
print("🧱 ボクセル化...")
min_bound = points.min(axis=0)
max_bound = points.max(axis=0)
max_bound[2] = min(max_bound[2], Z_LIMIT)
size = ((max_bound - min_bound) / GRID_RES).astype(int) + 1
voxels = np.zeros(size, dtype=np.uint32)

indices = ((points - min_bound) / GRID_RES).astype(int)
indices = indices[points[:, 2] <= Z_LIMIT]
for idx in indices:
    if (0 <= idx[0] < size[0]) and (0 <= idx[1] < size[1]) and (0 <= idx[2] < size[2]):
        voxels[tuple(idx)] += 1

mask = voxels >= MIN_PTS
if not np.any(mask):
    raise RuntimeError("❌ 内部空間候補なし")

# 連結成分抽出（26近傍）
print("🧩 連結成分抽出中...")
structure = np.ones((3, 3, 3), dtype=np.uint8)
labels, ncomp = ndimage.label(mask, structure=structure)
counts = np.bincount(labels.ravel())
counts[0] = 0

# 小さい成分を除外
valid_labels = [i for i, c in enumerate(counts) if c >= MIN_SIZE]

internal_voxels = []
for lbl in valid_labels:
    internal_voxels.append(np.argwhere(labels == lbl))
internal_voxels = np.vstack(internal_voxels)

print(f"✅ 内部成分数: {len(valid_labels)}, 点数: {len(internal_voxels)}")

# ボクセル中心座標に変換
out_points = []
out_class  = []
slice_dict = {}
for vx, vy, vz in internal_voxels:
    coord = (np.array([vx, vy, vz]) + 0.5) * GRID_RES + min_bound
    x_id = int(vx)
    out_points.append(coord)
    out_class.append(x_id)  # 16bitで保存
    slice_dict.setdefault(x_id, []).append(coord)

out_points = np.array(out_points)
out_class  = np.array(out_class, dtype=np.uint16)
save_las(OUTPUT_LAS_INTERNAL, out_points, out_class)

# 属性LAS（スライス番号）
attr_pts = []
attr_cls = []
for u, (x_id, pts) in enumerate(sorted(slice_dict.items())):
    arr = np.array(pts)
    attr_pts.extend(arr)
    attr_cls.extend([u] * len(arr))
if attr_pts:
    save_las(OUTPUT_LAS_ATTR, np.array(attr_pts), np.array(attr_cls, dtype=np.uint16))

# シェル（外縁点のみ）
shell = []
if slice_dict:
    keys = sorted(slice_dict.keys())
    shell.extend(slice_dict[keys[0]])
    shell.extend(slice_dict[keys[-1]])
    for arr in slice_dict.values():
        arr = np.array(arr)
        if len(arr) == 0:
            continue
        shell.append(arr[0])
        shell.append(arr[-1])
if shell:
    save_ply_points(OUTPUT_PLY_SHELL, np.array(shell))

print("🎉 M6処理完了 (LAS 1.4, classification=uint16)")
