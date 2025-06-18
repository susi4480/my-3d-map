import numpy as np
import open3d as o3d
import glob
import os

# === 入力設定 ===
ply_path = r"C:\Users\user\Documents\lab\output_ply\0523no3_xslice_test2.ply"
xyz_dir = r"C:\Users\user\Documents\lab\data\suidoubasi\lidar_xyz"
output_path = r"C:\Users\user\Documents\lab\output_ply\suidoubasi__idokeido.ply"

# === 分類パラメータ ===
voxel_size = 0.2
normal_wall_z_max = 4.5     # 壁として許容するZ上限
floor_z_max = 3.2           # Z > 3.2m を除外
horizontal_threshold = 0.95 # 法線Z > 0.95 を水平面と判定

# === ① PLY（川底・補間済）読み込み → 青に分類 ===
ply_pcd = o3d.io.read_point_cloud(ply_path)
ply_points = np.asarray(ply_pcd.points)
ply_colors = np.tile([0.0, 0.0, 1.0], (len(ply_points), 1))  # 青（川底）

# === ② XYZファイルを一括読み込み・高さフィルタ ===
xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
if len(xyz_files) == 0:
    raise FileNotFoundError(f"❌ XYZファイルが見つかりません: {xyz_dir}")

lidar_all = []
for path in xyz_files:
    try:
        data = np.loadtxt(path)
        if data.shape[1] >= 3:
            data = data[data[:, 2] <= floor_z_max]  # Z > 3.2m 除外
            lidar_all.append(data[:, :3])
    except Exception as e:
        print(f"❌ 読み込みエラー: {path} → {e}")

lidar_points = np.vstack(lidar_all)

# --- 点群 → Open3D形式 → 法線推定 ---
lidar_pcd = o3d.geometry.PointCloud()
lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points)
lidar_pcd = lidar_pcd.voxel_down_sample(voxel_size=voxel_size)
lidar_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

lidar_ds = np.asarray(lidar_pcd.points)
normals = np.asarray(lidar_pcd.normals)

# --- 分類マスク定義 ---
wall_mask = (normals[:, 2] < 0.3) & (lidar_ds[:, 2] < normal_wall_z_max)  # 壁
horizontal_mask = (normals[:, 2] > horizontal_threshold)                 # 水平面

# --- 色付け ---
lidar_colors = np.ones((len(lidar_ds), 3))  # 初期：白
lidar_colors[wall_mask] = [1.0, 0.0, 0.0]   # 壁 → 赤
lidar_colors[horizontal_mask] = [1.0, 0.5, 0.0]  # 水平面 → オレンジ

# === ③ 川底 + LiDARを統合・出力 ===
all_points = np.vstack((ply_points, lidar_ds))
all_colors = np.vstack((ply_colors, lidar_colors))

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(all_points)
merged_pcd.colors = o3d.utility.Vector3dVector(all_colors)

o3d.io.write_point_cloud(output_path, merged_pcd)
print(f"✅ 分類済点群（川底・岸壁・水平面）を統合し出力完了: {output_path}")
