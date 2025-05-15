import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# === 設定 ===
input_path = r"C:\Users\user\Documents\lab\cloud_model\kai_0508no4.ply"
voxel_size = 0.2
floor_z_limit = 4.0  # ✅ 床と認める最大Z高さ

# === [1] 点群読み込み + ダウンサンプリング ===
pcd = o3d.io.read_point_cloud(input_path)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

# === [2] RANSAC平面抽出 ===
plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
plane_cloud = pcd.select_by_index(inliers)
plane_points = np.asarray(plane_cloud.points)

# ✅ 床の高さ制限を適用（Z < 4.0 のみ残す）
z_filtered = plane_points[:, 2] < floor_z_limit
plane_cloud = plane_cloud.select_by_index(np.where(z_filtered)[0])

# 平面を青で表示・出力
plane_cloud.paint_uniform_color([0.0, 0.0, 1.0])
o3d.io.write_point_cloud("ransac_plane_filtered.ply", plane_cloud)
print("✅ RANSAC + 高さ制限による床抽出完了: ransac_plane_filtered.ply")

# === [3] DBSCANクラスタリング ===
print("▶ DBSCANクラスタリング中...")
labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=20, print_progress=True))
max_label = labels.max()

colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
colors[labels < 0] = [0, 0, 0, 1]  # 外れ値は黒
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.io.write_point_cloud("dbscan_clusters.ply", pcd)
print("✅ DBSCANクラスタリング完了: dbscan_clusters.ply")

