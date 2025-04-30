import os
import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# --- 設定 ---
las_folder = r"C:\Users\user\Documents\lab\data\las2"
z_limit = 10.0
eps = 1.0
min_samples = 20
z_range_thresh = 1.5      # Z方向の高低差 → 壁らしさ
thickness_thresh = 1.0    # 壁の厚み（XかY方向のどちらかがこれ以下）
output_ply = "wall_candidate3.ply"

# --- Step1: .las読み込み + 高さフィルタ ---
print("[1] .lasファイルを読み込み中...")
las_files = [f for f in os.listdir(las_folder) if f.endswith(".las")]
points_all = []

for file in las_files:
    with laspy.open(os.path.join(las_folder, file)) as f:
        las = f.read()
        pts = np.vstack((las.x, las.y, las.z)).T
        points_all.append(pts)

points = np.vstack(points_all)
points = points[points[:, 2] < z_limit]

# --- Step2: DBSCANクラスタリング ---
print("[2] DBSCANクラスタリング...")
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(points)
unique_labels = np.unique(labels)
print(f"クラスタ数（-1除く）: {len(unique_labels[unique_labels != -1])}")

# --- Step3: 壁候補抽出（Z範囲 + 厚みで判定）---
print("[3] 壁候補を抽出中...")
wall_mask = np.zeros(points.shape[0], dtype=bool)

for lbl in unique_labels:
    if lbl == -1:
        continue
    cluster = points[labels == lbl]
    z_range = cluster[:, 2].max() - cluster[:, 2].min()
    x_range = cluster[:, 0].ptp()
    y_range = cluster[:, 1].ptp()

    # 高さがあり、厚みが薄い（XまたはYが細長い）
    if z_range >= z_range_thresh and (x_range <= thickness_thresh or y_range <= thickness_thresh):
        wall_mask[labels == lbl] = True

print(f"壁候補点数: {np.sum(wall_mask)} / {points.shape[0]}")

# --- Step4: 可視化・出力 ---
colors = np.zeros_like(points)
colors[wall_mask] = [1, 0, 0]
colors[~wall_mask] = [0.5, 0.5, 0.5]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(output_ply, pcd)
print(f"[完了] 出力ファイル: {output_ply}")
