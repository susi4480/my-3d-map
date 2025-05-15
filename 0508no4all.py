import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\全体データ"
voxel_size = 0.2
normal_wall_z_max = 4.5   # 壁のZ上限
floor_z_max = 4.0         # 床のZ上限（誤検出防止）

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

# === [4] 分類マスク作成（壁、ビル、床、未分類）===
colors = np.ones((len(points_ds), 3))  # 初期値＝白

# 各分類条件
wall_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] < normal_wall_z_max)
building_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] >= normal_wall_z_max)
floor_mask = (normals[:, 2] > 0.9) & (points_ds[:, 2] < floor_z_max)

# 色の割り当て
colors[wall_mask] = [1.0, 0.0, 0.0]     # 赤：壁
colors[floor_mask] = [0.0, 0.0, 1.0]    # 青：床（河底）
colors[building_mask] = [1.0, 1.0, 0.0] # 黄：ビル

# === [5] 出力 ===
pcd.colors = o3d.utility.Vector3dVector(colors)
output_path = os.path.join(las_dir, "0508no4all.ply")
o3d.io.write_point_cloud(output_path, pcd)

print("✅ 出力完了：赤=壁、青=床（河底）、黄=ビル、白=未分類")
