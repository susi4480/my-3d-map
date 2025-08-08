# -*- coding: utf-8 -*-
"""
【機能】
- 緑点からConvex Hull（凸包）を構築してPLY保存
"""

import numpy as np
import laspy
import open3d as o3d

input_las = "/output/0704_method9_ue.las"
output_ply = "/output/0706mesh_convex_hull_ue.ply"

# === LAS読み込みと緑点抽出 ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).astype(np.float32)
cols_raw = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T

# 色が16bitなら変換
cols = (cols_raw / 256).astype(np.uint8) if np.max(cols_raw) > 255 else cols_raw.astype(np.uint8)

mask = (cols[:, 0] == 0) & (cols[:, 1] == 255) & (cols[:, 2] == 0)
pts_navi = pts[mask]

if len(pts_navi) == 0:
    raise RuntimeError("❌ 緑の航行可能空間が見つかりませんでした")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_navi)

# === Convex Hull ===
print("🔄 凸包メッシュ化中...")
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)

o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 Convex Hull出力完了: {output_ply}")
