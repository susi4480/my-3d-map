# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
import laspy
import os

# === 入出力ファイル設定 ===
input_las_path = "/home/edu1/miyachi/data/pond/Merlin_02.las"
output_dir = "/home/edu1/miyachi/output_mesh"
output_path = os.path.join(output_dir, "Merlin_02_mesh_alpha1.0.ply")

# === フォルダ確認・作成 ===
os.makedirs(output_dir, exist_ok=True)

# === LASファイル読み込み ===
las = laspy.read(input_las_path)
points = np.vstack((las.x, las.y, las.z)).T
print(f"✅ 読み込み完了: {len(points):,} 点")

# === Delaunay三角分割（XY平面で） ===
print("🔺 Delaunay三角分割中...")
tri = Delaunay(points[:, :2])
simplices = tri.simplices
print(f"✅ 採用された三角形数: {len(simplices):,}")

# === メッシュ構築 ===
vertices = o3d.utility.Vector3dVector(points)
triangles = o3d.utility.Vector3iVector(simplices)
mesh = o3d.geometry.TriangleMesh(vertices, triangles)
mesh.compute_vertex_normals()

# === メッシュ保存 ===
success = o3d.io.write_triangle_mesh(output_path, mesh)
if success:
    print(f"🎉 メッシュ保存完了: {output_path}")
else:
    print(f"❌ メッシュ保存失敗: {output_path}")
