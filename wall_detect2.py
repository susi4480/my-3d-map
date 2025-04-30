# 必要なライブラリをインポート
import os
import laspy
import numpy as np
import open3d as o3d

# ============================
# 設定（必要に応じて変更）
# ============================
folder_path = r"C:\Users\user\Documents\lab\data\las2"
z_limit = 10.0        # 高さ上限
normal_z_threshold = 0.3  # 法線のZ成分がこの値以下 → 壁とみなす
output_filename = "wall_candidate2.ply"

# ============================
# Step1: .lasファイルを全部読み込み
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
print(f"読み込んだ点数: {all_points.shape[0]}点")

# ============================
# Step2: 高さフィルタリング
# ============================
print("\n[Step2] 高さフィルタリング中...")
mask = all_points[:, 2] < z_limit
filtered_points = all_points[mask]
print(f"フィルタ後の点数: {filtered_points.shape[0]}点")

# ============================
# Step3: 法線ベクトルを推定
# ============================
print("\n[Step3] 法線ベクトルを計算中...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
normals = np.asarray(pcd.normals)

# ============================
# Step4: 法線のZ成分が小さい（= 壁っぽい）点だけ抽出
# ============================
print("\n[Step4] 法線Z成分による壁候補抽出...")
wall_mask = np.abs(normals[:, 2]) < normal_z_threshold
print(f"壁候補点数: {np.sum(wall_mask)}点")

colors = np.zeros((filtered_points.shape[0], 3))
colors[wall_mask] = [1, 0, 0]       # 壁候補→赤
colors[~wall_mask] = [0.5, 0.5, 0.5] # その他→灰色

pcd.colors = o3d.utility.Vector3dVector(colors)

# ============================
# Step5: 書き出し
# ============================
o3d.io.write_point_cloud(output_filename, pcd)
print(f"\n出力完了！ファイル名: {output_filename}")
