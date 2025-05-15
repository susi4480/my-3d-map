import os
import laspy
import numpy as np
import open3d as o3d

# ============================
# 設定
# ============================
folder_path = r"C:\Users\user\Documents\lab\data\las2"
z_limit = 10.0               # 点群の全体高さ上限（除外用）
wall_z_max = 4.5             # 壁とみなす最大高さ（ビルと区別）
normal_z_threshold = 0.3     # 法線Z成分しきい値（垂直に近いものが壁）
output_filename = "wall_detect2.ply"

# ============================
# Step1: .lasファイル読み込み
# ============================
print("\n[Step1] .lasファイルを読み込み中...")
las_files = [f for f in os.listdir(folder_path) if f.endswith(".las")]
all_points = []

for file in las_files:
    path = os.path.join(folder_path, file)
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    all_points.append(points)

all_points = np.vstack(all_points)
print(f"✅ 読み込んだ点数: {all_points.shape[0]}点")

# ============================
# Step2: 高さフィルタリング
# ============================
print("\n[Step2] 高さフィルタリング中...")
mask = all_points[:, 2] < z_limit
filtered_points = all_points[mask]
print(f"✅ 高さ制限後の点数: {filtered_points.shape[0]}点")

# ============================
# Step3: 法線ベクトルを推定
# ============================
print("\n[Step3] 法線ベクトルを推定中...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
normals = np.asarray(pcd.normals)

# ============================
# Step4: 壁候補の抽出（法線Z + Z値）
# ============================
print("\n[Step4] 壁候補点の抽出中...")
wall_mask = (np.abs(normals[:, 2]) < normal_z_threshold) & (filtered_points[:, 2] < wall_z_max)
print(f"✅ 壁候補点数: {np.sum(wall_mask)} / {len(filtered_points)}")

# 色分け
colors = np.zeros((filtered_points.shape[0], 3))
colors[wall_mask] = [1, 0, 0]        # 赤 = 壁候補
colors[~wall_mask] = [0.5, 0.5, 0.5] # 灰色 = その他
pcd.colors = o3d.utility.Vector3dVector(colors)

# ============================
# Step5: 出力
# ============================
o3d.io.write_point_cloud(os.path.join(folder_path, output_filename), pcd)
print(f"\n🎉 出力完了！ファイル名: {output_filename}")
