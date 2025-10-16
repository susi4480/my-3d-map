# -*- coding: utf-8 -*-
"""
【機能（手法①：ポリライン→三角形メッシュ化）】
- 入力LAS（M0onM5で出力された外周点群）を読み込み
- スライスごとに外周点をグルーピング（Zスライス or u方向スライス）
- 隣接スライス間の外周点を対応付けて三角形メッシュ化
- PLY形式で保存

依存: numpy, laspy, open3d
"""

import os
import numpy as np
import laspy
import open3d as o3d

# ===== 入出力 =====
INPUT_LAS  = r"/output/0912no1_M0onM5_merged_rects.las"
OUTPUT_PLY = r"/output/0916no3_M0onM5_mesh_method1.ply"

# ===== パラメータ =====
SLICE_BIN = 0.5   # スライスの間隔（Z方向 or 中心線方向に応じて調整）

def load_las_points(path):
    las = laspy.read(path)
    return np.vstack([las.x, las.y, las.z]).T

def group_points_by_slice(points, bin_size=0.5):
    """Zでスライス分け（仮にZ軸を使う）"""
    zmin = points[:,2].min()
    slice_ids = ((points[:,2]-zmin)/bin_size).astype(int)
    slices = {}
    for sid, p in zip(slice_ids, points):
        slices.setdefault(sid, []).append(p)
    # numpy配列に変換
    for sid in slices:
        slices[sid] = np.array(slices[sid])
    return slices

def triangulate_between_slices(slice1, slice2):
    """
    2つのスライス間の点群を対応づけて三角形化
    （単純に最近傍対応をとってストリップ状に連結する）
    """
    tris = []
    verts = []
    verts.extend(slice1.tolist())
    verts.extend(slice2.tolist())
    n1 = len(slice1)
    n2 = len(slice2)

    # インデックスオフセット
    off2 = n1

    # 最も近い点数でループ
    m = min(n1, n2)
    for i in range(m-1):
        # 四辺形を2つの三角形に分割
        tris.append([i, i+1, off2+i])
        tris.append([i+1, off2+i+1, off2+i])

    return verts, tris

def main():
    # === LAS読み込み ===
    points = load_las_points(INPUT_LAS)
    print(f"✅ 入力点数: {len(points)}")

    # === スライス分割 ===
    slices = group_points_by_slice(points, bin_size=SLICE_BIN)
    slice_keys = sorted(slices.keys())
    print(f"✅ スライス数: {len(slice_keys)}")

    all_verts = []
    all_tris = []
    v_offset = 0

    # 隣接スライスを順に三角形化
    for i in range(len(slice_keys)-1):
        s1 = slices[slice_keys[i]]
        s2 = slices[slice_keys[i+1]]
        verts, tris = triangulate_between_slices(s1, s2)
        # インデックスをオフセット補正
        tris = [[a+v_offset, b+v_offset, c+v_offset] for a,b,c in tris]
        all_verts.extend(verts)
        all_tris.extend(tris)
        v_offset += len(verts)

    # === Open3D メッシュ出力 ===
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_tris))
    mesh.compute_vertex_normals()

    os.makedirs(os.path.dirname(OUTPUT_PLY) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(OUTPUT_PLY, mesh)
    print(f"✅ メッシュ出力: {OUTPUT_PLY}")
    print(f" 頂点数: {len(all_verts)}, 三角形数: {len(all_tris)}")

if __name__ == "__main__":
    main()
