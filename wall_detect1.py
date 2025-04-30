# ============================
# 設定
# ============================
folder_path = r"C:\Users\user\Documents\lab\data\las2"
z_limit = 10.0         # ★ 高さ制限10mに拡大
z_diff_threshold = 1.0  # ★ Z差しきい値を0.5mに緩和
output_filename = "wall_candidate1.ply"

# ============================
# Step1～Step4（同じ）
# ============================
import os
import laspy
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

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

print("\n[Step2] 高さフィルタリング中...")
mask = all_points[:, 2] < z_limit
filtered_points = all_points[mask]
print(f"フィルタ後の点数: {filtered_points.shape[0]}点")

print("\n[Step3] Z変化量から壁候補スコア計算中...")
nn = NearestNeighbors(n_neighbors=10, radius=1.5)  # ★ 半径を1.5mに広げた
nn.fit(filtered_points)
distances, indices = nn.kneighbors(filtered_points)

z_vals = filtered_points[:, 2]
z_diff = np.max(np.abs(z_vals[indices] - z_vals[:, np.newaxis]), axis=1)

wall_score = z_diff > z_diff_threshold
print(f"壁候補点数: {np.sum(wall_score)}点")

print("\n[Step4] 壁候補をファイルに書き出し中...")
colors = np.zeros((filtered_points.shape[0], 3))
colors[wall_score] = [1, 0, 0]       # 壁候補→赤
colors[~wall_score] = [0.5, 0.5, 0.5] # それ以外→灰色

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(output_filename, pcd)

print(f"\n出力完了！ファイル名: {output_filename}")
