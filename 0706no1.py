# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルから航行可能空間（緑 [0,255,0]）だけを抽出
- Open3DでPoissonメッシュ化
- PLYとして保存
"""

import numpy as np
import laspy
import open3d as o3d
import os

# === 入出力設定 ===
input_las = "/output/0704_method9_ue.las"
output_ply = "/output/0706_mesh_ue.ply"

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).astype(np.float32).T
colors = np.vstack([las.red, las.green, las.blue]).astype(np.uint8).T

# === 緑色点群（航行可能空間）の抽出 ===
mask = (colors[0] == 0) & (colors[1] == 255) & (colors[2] == 0)
points_navi = points[:, mask].T
colors_navi = colors[:, mask].T

if len(points_navi) == 0:
    raise RuntimeError("❌ 航行可能空間（緑色）が見つかりませんでした")

print(f"✅ 航行可能点数: {len(points_navi):,}")

# === Open3Dで点群生成 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_navi)
pcd.colors = o3d.utility.Vector3dVector(colors_navi / 255.0)

# === 法線推定（メッシュ用）===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(20)

# === Poissonメッシュ化 ===
print("🔄 Poissonメッシュ中...")
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# === メッシュをPLY形式で保存 ===
o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 メッシュ出力完了: {output_ply}")
