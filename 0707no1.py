# -*- coding: utf-8 -*-
"""
【機能】
- ダウンサンプリングせず、緑の点群（PLY）からPoissonメッシュ化
- 法線推定のみ（整列なし）で処理
"""

import open3d as o3d

# === 入出力設定 ===
input_ply = "/output/0707_green_only_ue.ply"
output_ply = "/output/0707no1_mesh_poisson_full.ply"
poisson_depth = 9  # 例：9〜10は高品質、8は中程度、11は重め

# === 点群読み込み ===
print("📥 点群PLY読み込み中...")
pcd = o3d.io.read_point_cloud(input_ply)
print(f"✅ 読み込み完了: {len(pcd.points):,} 点")

# === 法線推定（整列は省略）===
print("📏 法線推定中...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

# === Poissonメッシュ化（整列なし）===
print(f"🔧 Poissonメッシュ化中（depth={poisson_depth}）...")
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)

# === メッシュ保存 ===
o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 Poissonメッシュ出力完了: {output_ply}")
