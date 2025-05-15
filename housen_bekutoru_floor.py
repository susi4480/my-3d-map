import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las2"
voxel_size = 0.2
floor_z_max = 4.0  # 床の最大高さ（河底想定）
floor_normal_threshold = 0.9  # 法線Z成分の下限（床とみなす基準）

# === [1] .lasファイル一覧取得 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
print(f"[1] {len(las_files)} ファイルを検出しました")

# === [2] 点群統合 ===
all_points = []
for path in las_files:
    print(f"  読み込み中: {os.path.basename(path)}")
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    all_points.append(points)

all_points = np.vstack(all_points)
print(f"[2] 統合点群数: {len(all_points)}")

# === [3] ダウンサンプリング + 法線推定 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

points_ds = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

# === [4] 床判定（法線ZとZ座標）===
floor_mask = (normals[:, 2] > floor_normal_threshold) & (points_ds[:, 2] < floor_z_max)
floor_points = points_ds[floor_mask]

# === [5] 出力 ===
floor_pcd = o3d.geometry.PointCloud()
floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
floor_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # 青

output_path = os.path.join(las_dir, "floor_detected_only.ply")
o3d.io.write_point_cloud(output_path, floor_pcd)

print(f"✅ 床のみ出力完了: {output_path}")
