# -*- coding: utf-8 -*-
"""
【機能】
- 床LASをダウンサンプリング
- LiDARと統合
- 法線推定を全体に実施
- Zと法線に基づいて「床（青）」「壁（赤）」「ビル（黄）」に分類
- PLYとして出力
"""

import laspy
import numpy as np
import open3d as o3d

# === 入出力設定 ===
floor_las_path = r"/data/matome/0725_suidoubasi_floor_ue.las"
lidar_las_path = r"/data/matome/0821_merged_lidar_ue.las"
output_ply_path = r"/output/0821no13_floor_lidar_classified_zbased.ply"

# === パラメータ ===
voxel_size = 0.1         # 床のみダウンサンプリング
normal_radius = 0.3       # 法線推定の近傍半径
floor_z_max = 1.1
wall_z_max = 3.2
horizontal_thresh = 0.7
vertical_thresh = 0.3

# === LAS読み込み関数 ===
def las_to_o3d_pointcloud(las_path):
    las = laspy.read(las_path)
    points = np.vstack([las.x, las.y, las.z]).T

    if hasattr(las, 'red'):
        colors = np.vstack([las.red, las.green, las.blue]).T / 65535.0
    else:
        colors = np.zeros_like(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# === [1] 床読み込み＋ダウンサンプリング ===
print("📥 床LAS読み込み＋ダウンサンプリング中...")
pcd_floor = las_to_o3d_pointcloud(floor_las_path)
pcd_floor_down = pcd_floor.voxel_down_sample(voxel_size=voxel_size)

# === [2] LiDAR読み込み（そのまま） ===
print("📥 LiDAR LAS読み込み中...")
pcd_lidar = las_to_o3d_pointcloud(lidar_las_path)

# === [3] 統合 ===
print("🔗 点群を統合中...")
pcd_combined = pcd_floor_down + pcd_lidar

# === [4] 法線推定 ===
print("📏 法線推定中...")
pcd_combined.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
)
pcd_combined.orient_normals_consistent_tangent_plane(30)

# === [5] 分類 ===
print("🎨 分類中...")
points = np.asarray(pcd_combined.points)
normals = np.asarray(pcd_combined.normals)
colors = np.full((len(points), 3), 0.5)  # 初期色：灰色

# ① ビル：Z > 3.2（無条件）
building_mask = points[:, 2] > wall_z_max
colors[building_mask] = [1.0, 1.0, 0.0]

# ② 床：Z ≤ 1.1 かつ 法線Z > 0.7
floor_mask = (points[:, 2] <= floor_z_max) & (normals[:, 2] > horizontal_thresh)
colors[floor_mask] = [0.0, 0.0, 1.0]

# ③ 壁：Z ≤ 3.2 かつ 法線Z < 0.3
wall_mask = (points[:, 2] <= wall_z_max) & (normals[:, 2] < vertical_thresh)
colors[wall_mask] = [1.0, 0.0, 0.0]

print(f"🟨 ビル領域: {np.sum(building_mask):,} 点")
print(f"🟦 床領域: {np.sum(floor_mask):,} 点")
print(f"🟥 壁領域: {np.sum(wall_mask):,} 点")

# === [6] 出力 ===
pcd_combined.colors = o3d.utility.Vector3dVector(colors)
print(f"💾 出力中... {output_ply_path}")
o3d.io.write_point_cloud(output_ply_path, pcd_combined)
print("✅ 完了！")
