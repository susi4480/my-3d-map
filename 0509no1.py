import laspy
import numpy as np
import open3d as o3d
import os
import glob
from sklearn.cluster import DBSCAN

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las2"
voxel_size = 0.2
height_threshold_pillar = 4.0     # 柱の高さ目安
xy_threshold_pillar = 1.0         # 柱のXYサイズ上限
wall_z_max = 4.5                  # 壁のZ上限（岸壁かどうか）
floor_normal_z = 0.9              # 床とみなす法線Zの下限

# === [1] .lasファイル一覧取得・読み込み ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
all_points = []
for path in las_files:
    print(f"読み込み中: {os.path.basename(path)}")
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    all_points.append(points)
all_points = np.vstack(all_points)

# === [2] Open3D点群化 + ダウンサンプリング + 法線推定 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

# === [3] DBSCANクラスタリング ===
print("クラスタリング中（DBSCAN）...")
dbscan = DBSCAN(eps=1.0, min_samples=10)
labels = dbscan.fit_predict(points)
unique_labels = np.unique(labels)

# === [4] クラスタごとに特徴抽出して分類 ===
colors = np.ones((len(points), 3))  # 初期値は白（未分類）
for lbl in unique_labels:
    if lbl == -1:
        continue  # ノイズ除外
    cluster_points = points[labels == lbl]
    cluster_normals = normals[labels == lbl]

    z_range = np.ptp(cluster_points[:, 2])
    x_range = np.ptp(cluster_points[:, 0])
    y_range = np.ptp(cluster_points[:, 1])

    norm_z_mean = cluster_normals[:, 2].mean()

    # === 分類条件 ===
    if norm_z_mean > floor_normal_z:
        colors[labels == lbl] = [0.0, 0.0, 1.0]  # 青：床
    elif z_range > height_threshold_pillar and max(x_range, y_range) < xy_threshold_pillar:
        colors[labels == lbl] = [0.0, 1.0, 0.0]  # 緑：柱
    elif z_range <= wall_z_max and max(x_range, y_range) < 10:
        colors[labels == lbl] = [1.0, 0.0, 0.0]  # 赤：岸壁
    elif z_range > wall_z_max:
        colors[labels == lbl] = [1.0, 1.0, 0.0]  # 黄：ビル壁
    else:
        colors[labels == lbl] = [1.0, 1.0, 1.0]  # 白：未分類

# === [5] 出力 ===
pcd.colors = o3d.utility.Vector3dVector(colors)
output_path = os.path.join(las_dir, "semantic_colored_segmentation.ply")
o3d.io.write_point_cloud(output_path, pcd)

print(f"✅ 出力完了: {output_path}")
print("赤=岸壁、青=床、緑=柱、黄=ビル、白=未分類")
