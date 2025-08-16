# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
import os

# === 入出力パス ===
input_path = "/home/edu1/miyachi/data/pond/MBES_02_mls_like_-2.5_deduped.ply"
output_path = "/home/edu1/miyachi/output_strategy/MBES_02_mesh_alpha1.0.ply"

# === 入力ファイル存在確認 ===
if not os.path.exists(input_path):
    raise FileNotFoundError(f"入力ファイルが存在しません: {input_path}")

# === 点群読み込み ===
pcd = o3d.io.read_point_cloud(input_path)
points = np.asarray(pcd.points)
print(f"✅ 読み込み完了: {len(points):,} 点")

# === Delaunay三角分割（XY平面）===
print("🔺 Delaunay三角分割中...")
tri = Delaunay(points[:, :2])

# === αフィルタリング（外接円半径で三角形選別）===
alpha = 1.0  # 適宜調整
def circumradius(a, b, c):
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    s = (ab + bc + ca) / 2
    area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
    if area == 0:
        return np.inf
    return (ab * bc * ca) / (4.0 * area)

faces = []
for simplex in tri.simplices:
    a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
    r = circumradius(a[:2], b[:2], c[:2])
    if r < alpha:
        faces.append(simplex)
print(f"✅ 採用された三角形数: {len(faces):,}")

# === メッシュ作成 ===
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(points)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

# === 出力先ディレクトリの作成 ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === メッシュ書き出し ===
success = o3d.io.write_triangle_mesh(output_path, mesh)
if success:
    print(f"🎉 メッシュ保存完了: {output_path}")
else:
    print(f"❌ メッシュ保存に失敗しました: {output_path}")
