# -*- coding: utf-8 -*-
"""
M5方式：3D占有ボクセル＋最大連結成分 (LAS 1.4対応版, RGBあり)
-----------------------------------
【機能】
- LAS入力を読み込み
- 占有ボクセルを構築（Z制限付き）
- ボクセル内点数の閾値でマスク生成
- 26近傍で最大連結成分を抽出
- ボクセル中心座標を航行可能点群とし、classification に Xインデックスをそのまま付与
- 出力は：
  - 緑点LAS（classification=Xインデックス, uint16対応, RGB緑固定）
  - ラインPLY
  - メッシュPLY
  - シェルPLY
  - 属性LAS（スライス番号, uint16対応）
  - ボリューム点LAS
-----------------------------------
"""

import os
import numpy as np
import laspy
import open3d as o3d
from scipy import ndimage

# ===== パラメータ =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0908M5_voxel_green.las"
OUTPUT_PLY_LINES = "/output/0908M5_lines.ply"
OUTPUT_PLY_MESH  = "/output/0908M5_mesh.ply"
OUTPUT_PLY_SHELL = "/output/0908M5_shell.ply"
OUTPUT_LAS_ATTR  = "/output/0908M5_attr_points.las"
OUTPUT_LAS_VOL   = "/output/0908M5_volume_points.las"

Z_LIMIT = 1.9
GRID_RES = 0.5
MIN_PTS = 30

for path in [OUTPUT_LAS, OUTPUT_PLY_LINES, OUTPUT_PLY_MESH, OUTPUT_PLY_SHELL, OUTPUT_LAS_ATTR, OUTPUT_LAS_VOL]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ===== 保存関数 =====
def save_las(path, points, classification=None):
    if points is None or len(points) == 0:
        print(f"⚠️ LAS出力なし: {path}")
        return
    # LAS 1.4 + PointFormat 7 (RGB付き)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]

    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = points[:, 0], points[:, 1], points[:, 2]

    if classification is not None:
        las_out.classification = classification.astype(np.uint16)  # 16bit対応

    # RGBを緑固定
    las_out.red   = np.zeros(len(points), dtype=np.uint16)
    las_out.green = np.full(len(points), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(points), dtype=np.uint16)

    las_out.write(path)
    print(f"✅ LAS出力: {path} 点数: {len(points)}")

def save_ply_points(path, points):
    if points is None or len(points) == 0:
        print(f"⚠️ PLY点群なし: {path}")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)
    print(f"✅ PLY出力: {path} 点数: {len(points)}")

def save_ply_mesh(path, vertices, triangles):
    if len(vertices) == 0 or len(triangles) == 0:
        print(f"⚠️ メッシュなし: {path}")
        return
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)
    print(f"✅ メッシュ出力: {path} 三角形数: {len(triangles)}")

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
    raise RuntimeError("❌ 緑点候補なし")

# 連結成分
print("🧩 連結成分抽出...")
structure = np.ones((3, 3, 3), dtype=np.uint8)
labels, ncomp = ndimage.label(mask, structure=structure)
counts = np.bincount(labels.ravel())
counts[0] = 0
largest_label = np.argmax(counts)
connected_voxels = np.argwhere(labels == largest_label)

print(f"✅ 成分数: {ncomp}, 最大成分: {len(connected_voxels)} voxels")

# ボクセル中心座標に変換
out_points = []
out_class  = []
slice_dict = {}
for vx, vy, vz in connected_voxels:
    coord = (np.array([vx, vy, vz]) + 0.5) * GRID_RES + min_bound
    x_id = int(vx)  # そのまま保存（256以上も可）
    out_points.append(coord)
    out_class.append(x_id)
    slice_dict.setdefault(x_id, []).append(coord)

out_points = np.array(out_points)
out_class  = np.array(out_class, dtype=np.uint16)
save_las(OUTPUT_LAS, out_points, out_class)

# スライスごとの配列
slices = []
for x_id, pts in sorted(slice_dict.items()):
    arr = np.array(pts)
    idx = np.lexsort((arr[:, 2], arr[:, 1]))  # Y→Z順
    slices.append(arr[idx])

# Step1: ライン
lines = []
for u in range(len(slices) - 1):
    A, B = slices[u], slices[u + 1]
    if len(A) == 0 or len(B) == 0:
        continue
    N = min(len(A), len(B))
    paired = np.empty((2 * N, 3))
    paired[0::2] = A[:N]
    paired[1::2] = B[:N]
    lines.append(paired)
if lines:
    save_ply_points(OUTPUT_PLY_LINES, np.vstack(lines))

# Step2: メッシュ
vertices = []
triangles = []
idx = 0
for u in range(len(slices) - 1):
    A, B = slices[u], slices[u + 1]
    if len(A) < 2 or len(B) < 2:
        continue
    N = min(len(A), len(B))
    for i in range(N - 1):
        p0, p1 = A[i], A[i + 1]
        q0, q1 = B[i], B[i + 1]
        vertices.extend([p0, p1, q1, q0])
        triangles.append([idx, idx + 1, idx + 2])
        triangles.append([idx, idx + 2, idx + 3])
        idx += 4
if vertices:
    save_ply_mesh(OUTPUT_PLY_MESH, np.array(vertices), np.array(triangles))

# Step3: シェル
shell = []
if slices:
    if len(slices[0]) > 0:   # ✅ 修正
        shell += slices[0].tolist()
    if len(slices[-1]) > 0:  # ✅ 修正
        shell += slices[-1].tolist()
    for s in slices:
        if len(s) == 0:
            continue
        shell.append(s[0])
        shell.append(s[-1])
if shell:
    save_ply_points(OUTPUT_PLY_SHELL, np.array(shell))

# Step4: 属性LAS
attr_pts = []
attr_cls = []
for u, s in enumerate(slices):
    if len(s) == 0:
        continue
    attr_pts.extend(s)
    attr_cls.extend([u] * len(s))
if attr_pts:
    save_las(OUTPUT_LAS_ATTR, np.array(attr_pts), np.array(attr_cls, dtype=np.uint16))

# Step5: Volume点
vol_pts = []
for u in range(len(slices) - 1):
    A, B = slices[u], slices[u + 1]
    if len(A) == 0 or len(B) == 0:
        continue
    N = min(len(A), len(B))
    mid = 0.5 * (A[:N] + B[:N])
    vol_pts.append(mid)
if vol_pts:
    save_las(OUTPUT_LAS_VOL, np.vstack(vol_pts))

print("🎉 M5処理完了 (LAS 1.4, PointFormat=7, RGB緑固定, classification=uint16)")
