# -*- coding: utf-8 -*-
"""
【機能】
- 航行可能空間（緑 [0,255,0]）を構造保持で抽出
- Voxelダウンサンプリング（ランダムではない）
- Open3Dでα-Shapeメッシュ化
"""

import numpy as np
import laspy
import open3d as o3d

# === 入出力設定 ===
input_las = "/output/0704_method9_ue.las"
output_ply = "/output/0706_mesh_alpha.ply"

# === α-Shapeパラメータ ===
voxel_size = 1.0     # 点群構造保持のための間引きサイズ
alpha = 2.0          # 形状の細かさ（通常 0.5〜5.0）

# === LAS読み込みと緑点抽出 ===
print("📥 LAS読み込み中...")
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).astype(np.float32)
cols_raw = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T

# === 色スケール変換（16bit → 8bit）===
if np.max(cols_raw) > 255:
    print("⚠️ 色が16bitスケールのため255スケールに変換します")
    cols = (cols_raw / 256).astype(np.uint8)
else:
    cols = cols_raw.astype(np.uint8)

# === 緑点抽出 ===
mask = (cols[:, 0] == 0) & (cols[:, 1] == 255) & (cols[:, 2] == 0)
pts_navi = pts[mask]

if len(pts_navi) == 0:
    raise RuntimeError("❌ 緑の航行可能空間が見つかりませんでした")

print(f"✅ 点数（元の航行可能空間）: {len(pts_navi):,}")

# === Open3D点群作成とダウンサンプリング ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_navi)

pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"✅ 点数（ダウンサンプリング後）: {len(pcd.points):,}")
print(f"🧮 間引き率: {len(pcd.points) / len(pts_navi) * 100:.2f}%")

# === 法線推定 ===
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))

# === α-Shapeメッシュ化 ===
print(f"🔄 α-Shape メッシュ中... (alpha={alpha})")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# === メッシュ保存 ===
o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 α-Shapeメッシュ保存完了: {output_ply}")
