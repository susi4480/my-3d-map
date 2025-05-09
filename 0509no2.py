import laspy
import numpy as np
import open3d as o3d
import os
import glob
from sklearn.cluster import DBSCAN

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las2"
voxel_size = 0.2
normal_wall_z_max = 4.5
floor_z_max = 4.0

# === [1] LASファイル読み込みと統合 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
all_points = []
for path in las_files:
    print(f"読み込み中: {os.path.basename(path)}")
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    all_points.append(points)
all_points = np.vstack(all_points)

# === [2] 点群処理 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
colors = np.ones((len(points), 3))  # 初期＝白

# === [3] 点単位分類（壁・床・ビル）===
wall_mask = (normals[:, 2] < 0.3) & (points[:, 2] < normal_wall_z_max)
floor_mask = (normals[:, 2] > 0.9) & (points[:, 2] < floor_z_max)
building_mask = (normals[:, 2] < 0.3) & (points[:, 2] >= normal_wall_z_max)

colors[wall_mask] = [1.0, 0.0, 0.0]     # 赤：壁
colors[floor_mask] = [0.0, 0.0, 1.0]    # 青：床（河底）
colors[building_mask] = [1.0, 1.0, 0.0] # 黄：ビル

# === [4] 未分類点の抽出（柱候補）===
unlabeled_mask = ~(wall_mask | floor_mask | building_mask)
pillar_candidates = points[unlabeled_mask]

# === [5] 柱候補クラスタリング（DBSCAN）===
pillar_labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(pillar_candidates)
unique_labels = np.unique(pillar_labels)

idx_unlabeled = np.where(unlabeled_mask)[0]
for lbl in unique_labels:
    if lbl == -1:
        continue
    idx_local = np.where(pillar_labels == lbl)[0]
    idx_global = idx_unlabeled[idx_local]
    cluster = points[idx_global]
    z_range = np.ptp(cluster[:, 2])
    x_range = np.ptp(cluster[:, 0])
    y_range = np.ptp(cluster[:, 1])
    aspect_ratio = z_range / (max(x_range, y_range) + 1e-6)

    if (
        len(cluster) > 100 and
        z_range > 3.0 and
        max(x_range, y_range) < 1.0 and
        aspect_ratio > 3.5
    ):
        colors[idx_global] = [0.0, 1.0, 0.0]  # 緑：柱

# === [6] 出力 ===
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(las_dir, "pillar_clustered_output.ply"), pcd)

print("✅ 出力完了: pillar_clustered_output.ply")
print("赤＝壁、青＝床、黄＝ビル、緑＝柱（再クラスタ処理）、白＝未分類")
