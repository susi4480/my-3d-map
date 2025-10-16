# -*- coding: utf-8 -*-
"""
【機能】floor補間PLY＋LiDAR LAS統合 → 法線推定②版（地形向け, 向き統一なし）＋分類付きPLY出力
-----------------------------------------------------------------------
入力:
  - 床（補間済み）PLY : /workspace/output/1008_floor_interp_only.ply
  - LiDAR（上部構造）LAS : /workspace/output/0925_lidar_sita_merged.las
処理:
  1. 2ファイルを読み込み、統合点群を作成
  2. ダウンサンプリング（Open3D）
  3. 法線推定（地形向け: radius=1.0, max_nn=100）
  4. 向き統一なし（そのままの符号で使用）
  5. 法線Z成分＋高さによる分類（壁=赤, 床=青, ビル=黄）
  6. 法線＋分類色付きPLY出力
-----------------------------------------------------------------------
出力:
  /workspace/output/1010_sita_classified_normals_type2_free.ply
"""

import numpy as np
import open3d as o3d
import laspy

# === 入出力設定 ===
input_floor_ply = r"/workspace/output/1008_floor_interp_only.ply"
input_lidar_las = r"/workspace/output/0925_lidar_sita_merged.las"
output_ply      = r"/workspace/output/1010_sita_classified_normals_type2_free.ply"

# === パラメータ ===
down_voxel_size        = 0.2   # ダウンサンプリング解像度[m]
search_radius_normals  = 1.0   # 法線推定半径[m]（地形向け）
max_neighbors_normals  = 100   # 法線推定近傍点数（地形向け）

normal_wall_z_max      = 3.2   # 壁の高さ上限[m]
floor_z_max            = 1.1   # 床とみなす高さ上限[m]
horizontal_threshold   = 0.6   # 法線Z成分の閾値（水平判定）

# === [1] 床PLY読み込み ===
print("📥 床PLY読み込み中...")
pcd_floor = o3d.io.read_point_cloud(input_floor_ply)
pts_floor = np.asarray(pcd_floor.points)
print(f"✅ 床点数: {len(pts_floor):,}")

# === [2] LiDAR LAS読み込み ===
print("📥 LiDAR LAS読み込み中...")
las = laspy.read(input_lidar_las)
pts_lidar = np.vstack([las.x, las.y, las.z]).T
print(f"✅ LiDAR点数: {len(pts_lidar):,}")

# === [3] 統合 ===
all_points = np.vstack([pts_floor, pts_lidar])
print(f"🧩 統合点数: {len(all_points):,}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)

# === [4] ダウンサンプリング ===
print("📉 ダウンサンプリング中...")
pcd = pcd.voxel_down_sample(voxel_size=down_voxel_size)
points = np.asarray(pcd.points)
print(f"✅ ダウンサンプリング後: {len(points):,}")

# === [5] 法線推定（地形向け, 向き統一なし） ===
print("📐 法線推定中（地形向け, 向き統一なし）...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=search_radius_normals,
        max_nn=max_neighbors_normals
    )
)
normals = np.asarray(pcd.normals)
print("✅ 法線推定完了（向き統一なし）")

# === [6] 分類（壁・床・ビル） ===
print("🎨 分類中...")
colors = np.ones((len(points), 3), dtype=np.float64)  # 初期: 白 (1,1,1)

# 壁（赤）
mask_wall = (normals[:, 2] < 0.6) & (points[:, 2] < normal_wall_z_max)
colors[mask_wall] = (1.0, 0.0, 0.0)

# 床（青）
mask_floor = (normals[:, 2] > horizontal_threshold) & (points[:, 2] < floor_z_max)
colors[mask_floor] = (0.0, 0.0, 1.0)

# ビル（黄）
mask_building = points[:, 2] >= normal_wall_z_max
colors[mask_building] = (1.0, 1.0, 0.0)

pcd.colors = o3d.utility.Vector3dVector(colors)
print(f"✅ 壁={mask_wall.sum():,} 床={mask_floor.sum():,} ビル={mask_building.sum():,}")

# === [7] 出力 ===
# Open3DのPLY出力では normals を自動で nx, ny, nz として保存
ok = o3d.io.write_point_cloud(output_ply, pcd, write_ascii=False, compressed=False)
if not ok:
    raise RuntimeError("PLY出力に失敗しました")

print(f"🎉 出力完了: {output_ply}")
print(f"📊 出力点数: {len(points):,}（法線＋分類色付き, 向き統一なし）")
print(f"🧭 法線含有: {pcd.has_normals()}, 色含有: {pcd.has_colors()}")
