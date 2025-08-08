# -*- coding: utf-8 -*-
"""
【機能】
- 緑点からPoisson再構成で滑らかなメッシュ（閉じた外郭）を作成
"""

import numpy as np
import laspy
import open3d as o3d

input_las = "/output/0704_method9_ue.las"
output_ply = "/output/0706mesh_poisson_ue.ply"
voxel_size = 0.3

# === 読み込みと抽出 ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).astype(np.float32)
cols_raw = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T
cols = (cols_raw / 256).astype(np.uint8) if np.max(cols_raw) > 255 else cols_raw.astype(np.uint8)
mask = (cols[:, 0] == 0) & (cols[:, 1] == 255) & (cols[:, 2] == 0)
pts_navi = pts[mask]

if len(pts_navi) == 0:
    raise RuntimeError("❌ 緑の航行可能空間が見つかりませんでした")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_navi)
pcd = pcd.voxel_down_sample(voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

# === Poisson再構成 ===
print("🔄 Poissonメッシュ構築中...")
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 Poisson出力完了: {output_ply}")
