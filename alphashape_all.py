# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay
import laspy
import os

# === 入出力ファイル設定 ===
input_las1 = "/home/edu1/miyachi/data/pond/MBES_02.las"
input_las2 = "/home/edu1/miyachi/data/pond/Merlin_02.las"
output_dir = "/home/edu1/miyachi/output_mesh"
output_path = os.path.join(output_dir, "merged_MBES_Merlin_mesh_alpha1.0.ply")

# === フォルダ確認・作成 ===
os.makedirs(output_dir, exist_ok=True)

# === LASファイル読み込みと統合 ===
def read_las_points(path):
    las = laspy.read(path)
    return np.vstack((las.x, las.y, las.z)).T

print("📂 LASファイル読み込み中...")
points1 = read_las_points(input_las1)
points2 = read_las_points(input_las2)

points = np.vstack((points1, points2))
print(f"✅ 統合後の総点数: {len(points):,} 点")

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
