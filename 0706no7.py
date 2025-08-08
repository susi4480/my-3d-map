# -*- coding: utf-8 -*-
"""
【機能】
- 緑点群をボクセル化 → Signed Distance Field → マーチングキューブで外形抽出
"""

import numpy as np
import laspy
import open3d as o3d

input_las = "/output/0704_method9_ue.las"
output_ply = "/output/mesh_marching_cubes.ply"
voxel_size = 0.3  # 解像度（小さいほど高精度）

# === 点群読み込みと緑抽出 ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).astype(np.float32)
cols_raw = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T
cols = (cols_raw / 256).astype(np.uint8) if np.max(cols_raw) > 255 else cols_raw.astype(np.uint8)
mask = (cols[:, 0] == 0) & (cols[:, 1] == 255) & (cols[:, 2] == 0)
pts_navi = pts[mask]

if len(pts_navi) == 0:
    raise RuntimeError("❌ 緑の航行可能空間が見つかりませんでした")

# === 点群をVoxel Gridに変換 → TSDF ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_navi)

print("🔲 Voxel化 + TSDF構築中...")
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.None
)

pose = np.eye(4)
volume.integrate(o3d.geometry.RGBDImage(), o3d.camera.PinholeCameraIntrinsic(), pose)  # 空のRGBDを使う
volume.extract_triangle_mesh().remove_duplicated_vertices()
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 マーチングキューブ出力完了: {output_ply}")

