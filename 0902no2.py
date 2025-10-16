# -*- coding: utf-8 -*-
"""
統合処理パイプライン（M5方式・安定版）
- LAS入力から航行可能空間を抽出（3D占有ボクセル）
- scipy.ndimage.label による 26近傍の最大連結成分抽出（Open3D KDTree不使用）
- スライス番号を classification に付与（Xインデックス）
- 各スライスをリスト化し、ライン/メッシュ/シェル/属性LAS/ボリューム点を出力
- 形状不一致の stack エラー回避（配列長に依存しない実装）
"""

import os
import numpy as np
import laspy
import open3d as o3d
from copy import deepcopy
from scipy import ndimage  # ★ 連結成分抽出に使用

# ===== パラメータ =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0901no1_M5_voxel_connected_classified.las"
OUTPUT_PLY_LINES = "/output/0901_M5step1_lines.ply"
OUTPUT_PLY_MESH  = "/output/0901_M5step2_mesh.ply"
OUTPUT_PLY_SHELL = "/output/0901_M5step3_shell.ply"
OUTPUT_LAS_ATTR  = "/output/0901_M5step4_attr_meshpoints.las"
OUTPUT_LAS_VOL   = "/output/0901_M5step5_volume_points.las"

Z_LIMIT = 1.9
GRID_RES = 0.3
MIN_PTS = 30

for path in [OUTPUT_LAS, OUTPUT_PLY_LINES, OUTPUT_PLY_MESH, OUTPUT_PLY_SHELL, OUTPUT_LAS_ATTR, OUTPUT_LAS_VOL]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ===== ユーティリティ =====
def save_ply_points(path, points):
    if points is None or len(points) == 0:
        print(f"⚠️ 空の点群: {path}")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)
    print(f"✅ PLY出力: {path} 点数: {len(points):,}")

def save_ply_mesh(path, vertices, triangles):
    if (vertices is None or len(vertices) == 0) or (triangles is None or len(triangles) == 0):
        print(f"⚠️ 空のメッシュ: {path}")
        return
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)
    print(f"✅ メッシュPLY出力: {path} 三角形数: {len(triangles):,}")

def save_las(path, points, classification=None, rgb=None, scales=(0.001,0.001,0.001)):
    if points is None or len(points) == 0:
        print(f"⚠️ LAS出力対象なし: {path}")
        return
    header = laspy.LasHeader(point_format=3, version="1.2")  # PF=3 (XYZ+RGB)
    header.scales = np.array(scales, dtype=np.float64)
    header.offsets = np.min(points, axis=0).astype(np.float64)
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = points[:,0], points[:,1], points[:,2]

    if classification is not None:
        cls = np.asarray(classification).astype(np.uint8)
        if len(cls) != len(points):
            raise ValueError("classification 長さが点数と一致しません。")
        las_out.classification = cls

    # RGB（オプション）
    if rgb is not None:
        rgb = np.asarray(rgb).astype(np.uint16)
        if rgb.shape != (len(points), 3):
            raise ValueError("RGB 形状が (N,3) ではありません。")
        las_out.red, las_out.green, las_out.blue = rgb[:,0], rgb[:,1], rgb[:,2]

    las_out.write(path)
    print(f"✅ LAS出力: {path} 点数: {len(points):,}")

def sort_slice_points_yz(slice_pts):
    """スライス内の点を Y→Z の順で安定ソート（ライン/メッシュ生成の見栄え安定化）"""
    if len(slice_pts) == 0:
        return slice_pts
    idx = np.lexsort((slice_pts[:,2], slice_pts[:,1]))
    return slice_pts[idx]

# ===== LAS読み込み =====
print("📥 LAS読み込み中...")
las = laspy.read(INPUT_LAS)
points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

# ===== Occupancy 構築（Z制限）=====
print("🧱 占有ボクセル構築中...")
min_bound = points.min(axis=0)
max_bound = points.max(axis=0)
size = ((max_bound - min_bound) / GRID_RES).astype(int) + 1
size = np.maximum(size, 1)  # 念のため

voxels = np.zeros(size, dtype=np.uint32)
# 点→ボクセルインデックス
indices = ((points - min_bound) / GRID_RES).astype(int)
in_z = points[:, 2] <= Z_LIMIT
indices = indices[in_z]

# ボクセル内点数カウント
for idx in indices:
    # インデックス安全確認
    if (0 <= idx[0] < size[0]) and (0 <= idx[1] < size[1]) and (0 <= idx[2] < size[2]):
        voxels[tuple(idx)] += 1

# 閾値以上をマスク
mask = voxels >= MIN_PTS
if not np.any(mask):
    raise RuntimeError("❌ 緑点候補（占有密度が閾値以上のボクセル）が存在しません。")

# ===== 3D 連結成分（26近傍）抽出 =====
print("🧩 3D連結成分（26近傍）抽出中...")
structure = np.ones((3,3,3), dtype=np.uint8)  # 26近傍
labels, ncomp = ndimage.label(mask, structure=structure)
if ncomp == 0:
    raise RuntimeError("❌ 連結成分が見つかりませんでした。")

# 最大成分のラベルを取得
counts = np.bincount(labels.ravel())
counts[0] = 0  # 背景の0は除外
largest_label = np.argmax(counts)
connected_voxels = np.argwhere(labels == largest_label)  # (N,3) int

print(f"✅ 連結成分数: {ncomp}, 最大成分ボクセル数: {len(connected_voxels):,}")

# ===== ボクセル中心座標 → 出力点群＋スライス分類 =====
print("📐 ボクセル中心→座標変換＆スライス分類中...")
out_points = []
out_class  = []
slice_dict = {}  # xインデックス → [点...]

for vx, vy, vz in connected_voxels:
    coord = (np.array([vx, vy, vz], dtype=np.float64) + 0.5) * GRID_RES + min_bound
    x_id = int(vx)  # ★ 既存実装より安定：voxelのXインデックスをそのまま採用
    out_points.append(coord)
    out_class.append(x_id % 256)  # LAS classification は 0-255
    slice_dict.setdefault(x_id, []).append(coord)

out_points = np.asarray(out_points, dtype=np.float64)
out_class  = np.asarray(out_class,  dtype=np.uint8)

# ===== 緑点LAS出力（PF=3, RGB=0,65535,0）=====
rgb_green = np.column_stack([
    np.zeros(len(out_points), dtype=np.uint16),
    np.full (len(out_points), 65535, dtype=np.uint16),
    np.zeros(len(out_points), dtype=np.uint16)
])
save_las(OUTPUT_LAS, out_points, classification=out_class, rgb=rgb_green)

print(f"✅ 出力完了: {OUTPUT_LAS} 点数: {len(out_points):,}")

# ===== スライス配列生成（Y→Zでソートして安定化）=====
slices = []
for x_id, pts in sorted(slice_dict.items()):
    arr = np.asarray(pts, dtype=np.float64)
    arr = sort_slice_points_yz(arr)
    # スライスが単一点だと後段で線/面が張れないため、そのまま保持（出力側で安全に扱う）
    slices.append(arr)

if len(slices) == 0:
    raise RuntimeError("❌ スライスが1つも生成できませんでした。")
if len(slices) == 1:
    print("⚠️ スライスが1枚のみのため、ライン/メッシュ/ボリューム生成はスキップされます。")

# ===== Step1: 横線（隣接スライスの対応点を結ぶ）=====
print("🧵 Step1: ライン生成中...")
lines = []
for u in range(len(slices) - 1):
    A, B = slices[u], slices[u+1]
    if len(A) == 0 or len(B) == 0:
        continue
    N = min(len(A), len(B))
    # 1:1 対応で端まで（長さ差は切り詰め）
    # PLYは線要素非対応のため、両端点を順に点群として保存（ビューで線状に見える）
    paired = np.empty((2*N, 3), dtype=np.float64)
    paired[0::2] = A[:N]
    paired[1::2] = B[:N]
    lines.append(paired)

if len(lines) > 0:
    save_ply_points(OUTPUT_PLY_LINES, np.vstack(lines))
else:
    print("⚠️ ライン出力なし（有効スライスが不足）")

# ===== Step2: メッシュ（隣接スライス間を四辺形→2三角形で接続）=====
print("🔺 Step2: メッシュ生成中...")
vertices = []
triangles = []
idx = 0
for u in range(len(slices) - 1):
    A, B = slices[u], slices[u+1]
    if len(A) < 2 or len(B) < 2:
        continue
    N = min(len(A), len(B))
    for i in range(N - 1):
        p0, p1 = A[i],   A[i+1]
        q0, q1 = B[i],   B[i+1]
        # 四辺形 (p0, p1, q1, q0) を2三角形に分割
        vertices.extend([p0, p1, q1, q0])
        triangles.append([idx,   idx+1, idx+2])
        triangles.append([idx,   idx+2, idx+3])
        idx += 4

if len(vertices) > 0 and len(triangles) > 0:
    save_ply_mesh(OUTPUT_PLY_MESH, np.asarray(vertices), np.asarray(triangles))
else:
    print("⚠️ メッシュ出力なし（対応点不足）")

# ===== Step3: シェル（外縁点の集合）=====
print("🛡️ Step3: シェル点生成中...")
shell_pts = []
if len(slices) >= 1:
    # 端スライス全点
    shell_pts.extend(slices[0].tolist())
    if len(slices) >= 2:
        shell_pts.extend(slices[-1].tolist())
    # 各スライスの端点（最小/最大インデックス）
    for s in slices:
        if len(s) == 0:
            continue
        shell_pts.append(s[0])
        if len(s) >= 2:
            shell_pts.append(s[-1])

if len(shell_pts) > 0:
    save_ply_points(OUTPUT_PLY_SHELL, np.asarray(shell_pts, dtype=np.float64))
else:
    print("⚠️ シェル出力なし（点不足）")

# ===== Step4: 属性点群（スライス番号を classification に格納）=====
print("🏷️ Step4: 属性LAS（スライス番号）出力中...")
attr_pts = []
attr_cls = []
for u, s in enumerate(slices):
    if len(s) == 0:
        continue
    attr_pts.append(s)
    cls_val = np.uint8(u % 256)
    attr_cls.append(np.full(len(s), cls_val, dtype=np.uint8))

if len(attr_pts) > 0:
    attr_pts = np.vstack(attr_pts)
    attr_cls = np.concatenate(attr_cls)
    save_las(OUTPUT_LAS_ATTR, attr_pts, classification=attr_cls)
else:
    print("⚠️ 属性LAS出力なし（点不足）")

# ===== Step5: Volume点（隣接スライスの中点）=====
print("🧊 Step5: Volume点生成中...")
vol_pts = []
for u in range(len(slices) - 1):
    A, B = slices[u], slices[u+1]
    if len(A) == 0 or len(B) == 0:
        continue
    N = min(len(A), len(B))
    mid = 0.5 * (A[:N] + B[:N])
    vol_pts.append(mid)

if len(vol_pts) > 0:
    save_las(OUTPUT_LAS_VOL, np.vstack(vol_pts))
else:
    print("⚠️ Volume点出力なし（対応点不足）")

print("🎉 すべての処理が完了しました。")
