import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# ================================
# 設定
# ================================
las_path = r"C:\Users\user\Documents\lab\data\las2\20211029_Marlin[multibeam]_20240625_TUMSAT LiDAR triai-20240627-121535(1)-R20250425-123306.las" 
voxel_size = 0.3
eps_values = np.arange(0.3, 1.2, 0.2)
min_samples_values = np.arange(5, 30, 5)

# ================================
# LASファイルの読み込み
# ================================
print("[1] LASファイルを読み込み中...")
las = laspy.read(las_path)
points = np.vstack((las.x, las.y, las.z)).T

# ================================
# Open3Dでダウンサンプリング
# ================================
print("[2] Voxelダウンサンプリングを実行...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
points_ds = np.asarray(pcd.points)

# ================================
# DBSCAN パラメータチューニング（ヒートマップ用）
# ================================
print("[3] パラメータチューニング用ヒートマップを作成中...")
cluster_counts = np.zeros((len(eps_values), len(min_samples_values)))

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_ds)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_counts[i, j] = n_clusters

# ヒートマップ表示
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cluster_counts, cmap='viridis', origin='lower')

ax.set_xticks(np.arange(len(min_samples_values)))
ax.set_xticklabels(min_samples_values)
ax.set_yticks(np.arange(len(eps_values)))
ax.set_yticklabels(np.round(eps_values, 2))
ax.set_xlabel("min_samples")
ax.set_ylabel("eps")
ax.set_title("DBSCAN クラスタ数ヒートマップ")
fig.colorbar(im, ax=ax, label="クラスタ数")
plt.tight_layout()
plt.savefig("dbscan_heatmap.png")
plt.show()

# ================================
# 壁抽出の本処理（選んだパラメータをここで指定）
# ================================
print("[4] 最適パラメータで壁クラスタ抽出...")
best_eps = 0.5           # ←ヒートマップ見て調整
best_min_samples = 10    # ←ヒートマップ見て調整

labels = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit_predict(points_ds)
unique_labels = set(labels)
wall_clusters = []

for label in unique_labels:
    if label == -1:
        continue
    cluster = points_ds[labels == label]
    z_range = cluster[:, 2].max() - cluster[:, 2].min()
    x_range = cluster[:, 0].max() - cluster[:, 0].min()
    y_range = cluster[:, 1].max() - cluster[:, 1].min()

    # ★ 壁らしい条件（Z方向の高さ、XYの広がり）
    if 1.5 < z_range < 10.0 and x_range > 0.5 and y_range > 0.5:
        wall_clusters.append(cluster)

# ================================
# PLYファイルとして出力
# ================================
if wall_clusters:
    print("[5] 検出した壁クラスタをPLY出力...")
    wall_points = np.vstack(wall_clusters)
    wall_pcd = o3d.geometry.PointCloud()
    wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
    o3d.io.write_point_cloud("detected_wall.ply", wall_pcd)
    print("出力完了: detected_wall.ply")
else:
    print("壁らしいクラスタは見つかりませんでした。")
