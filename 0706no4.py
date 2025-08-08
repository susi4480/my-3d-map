# -*- coding: utf-8 -*-
"""
【機能】
- LASファイルから航行可能空間（緑 [0,255,0]）だけを抽出
- Open3DでBall Pivoting（BPA）メッシュ化
- PLYとして保存
"""

import numpy as np
import laspy
import open3d as o3d

# === 入出力設定 ===
input_las = "/output/0704_method9_ue.las"
output_ply = "/output/0706_mesh_bpa_ue.ply"

# === LAS読み込みと緑色点抽出 ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
points = np.vstack([las.x, las.y, las.z]).astype(np.float32).T
colors = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T

# 緑の抽出（[0,255,0] スケール）
mask = (colors[:, 0] == 0) & (colors[:, 1] == 255) & (colors[:, 2] == 0)
points_navi = points[mask]
colors_navi = colors[mask]

if len(points_navi) == 0:
    raise RuntimeError("❌ 航行可能空間（緑）が見つかりませんでした")

print(f"✅ 航行可能点数: {len(points_navi):,}")

# === 点群としてOpen3Dに変換 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_navi)
pcd.colors = o3d.utility.Vector3dVector(colors_navi / 255.0)

# === 法線推定（BPAは法線必須）===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(30)

# === BPAのパラメータ設定 ===
print("📏 平均距離計算中...")
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 2.5 * avg_dist  # 調整可（例: 2.0〜3.0倍）

print(f"🔄 BPA実行中（radius={radius:.3f}）...")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector([radius])
)

# === 保存 ===
o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 BPAメッシュ出力完了: {output_ply}")
