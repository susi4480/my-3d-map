# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルをZ方向に0.05mごとにスライス
- 各スライスごとに点群処理 → BPAメッシュ化
- 最終的にすべてのメッシュを統合してPLY出力
"""

import numpy as np
import laspy
import open3d as o3d
import os

# === 入出力設定 ===
input_las = "/output/0707_green_only_ue.las"
output_dir = "/output/output_slices"
os.makedirs(output_dir, exist_ok=True)
output_ply = os.path.join(output_dir, "merged_bpa_mesh_005slice.ply")

# === パラメータ ===
z_min = -6.0
z_max = 3.5
z_step = 0.05
voxel_size = 0.05
radii = [0.3, 0.5]

# === LAS読み込み ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).astype(np.float32).T
colors = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T
print(f"✅ 総点数: {len(points):,}")

# === メッシュ統合用オブジェクト ===
merged_mesh = o3d.geometry.TriangleMesh()

# === スライスして順次メッシュ化 ===
for i, z0 in enumerate(np.arange(z_min, z_max, z_step)):
    z1 = z0 + z_step
    mask = (points[:, 2] >= z0) & (points[:, 2] < z1)
    if np.count_nonzero(mask) < 100:
        continue

    sliced_pts = points[mask]
    sliced_cols = colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sliced_pts)
    pcd.colors = o3d.utility.Vector3dVector(sliced_cols / 255.0)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    merged_mesh += mesh
    print(f"✅ Z={z0:.2f}〜{z1:.2f} 完了: 点数={len(pcd.points):,}, 三角形数={len(mesh.triangles):,}")

# === PLY出力 ===
o3d.io.write_triangle_mesh(output_ply, merged_mesh)
print(f"🎉 出力完了: {output_ply}")
