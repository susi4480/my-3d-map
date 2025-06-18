import laspy
import numpy as np
import open3d as o3d
import glob
import os

# === 入力設定 ===
ply_path = r"C:\Users\user\Documents\lab\output_ply\0523no3_xslice_smoothed.ply"
lidar_dir = r"C:\Users\user\Documents\lab\data\suidoubasi\lidar"
output_path = r"C:\Users\user\Documents\lab\output_ply\suidoubasi_combined_classified_ver2.ply"

# === 分類パラメータ ===
voxel_size = 0.2
normal_wall_z_max = 4.5
floor_z_max = 4.0

# === ① PLY（川底・補間済）読み込み → 青（床）で分類 ===
ply_pcd = o3d.io.read_point_cloud(ply_path)
ply_points = np.asarray(ply_pcd.points)
ply_colors = np.tile([0.0, 0.0, 1.0], (len(ply_points), 1))  # 全点を青に

# === ② LiDAR（LAS）読み込み → 壁・ビルに分類 ===
las_files = glob.glob(os.path.join(lidar_dir, "*.las"))
lidar_all = []
for path in las_files:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    lidar_all.append(xyz)
lidar_points = np.vstack(lidar_all)

# Open3D 点群化
lidar_pcd = o3d.geometry.PointCloud()
lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points)
lidar_pcd = lidar_pcd.voxel_down_sample(voxel_size=voxel_size)
lidar_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

lidar_ds = np.asarray(lidar_pcd.points)
normals = np.asarray(lidar_pcd.normals)

# マスク定義
wall_mask = (normals[:, 2] < 0.3) & (lidar_ds[:, 2] < normal_wall_z_max)
building_mask = (normals[:, 2] < 0.3) & (lidar_ds[:, 2] >= normal_wall_z_max)

# 色設定
lidar_colors = np.ones((len(lidar_ds), 3))  # 白で初期化
lidar_colors[wall_mask] = [1.0, 0.0, 0.0]     # 赤
lidar_colors[building_mask] = [1.0, 1.0, 0.0] # 黄

# === ③ 統合して出力 ===
all_points = np.vstack((ply_points, lidar_ds))
all_colors = np.vstack((ply_colors, lidar_colors))

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(all_points)
merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)

o3d.io.write_point_cloud(output_path, merged_pcd)
print(f"✅ 分類済点群を統合し出力完了: {output_path}")
