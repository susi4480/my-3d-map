import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# === 設定 ===
input_path = r"C:\Users\user\Documents\lab\cloud_model\0508no4.ply"
voxel_size = 0.2

# === [1] 点群読み込み + ダウンサンプリング ===
pcd = o3d.io.read_point_cloud(input_path)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

# === [2] RANSAC平面抽出（床または壁）===
plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
plane_cloud = pcd.select_by_index(inliers)
plane_cloud.paint_uniform_color([0.0, 0.0, 1.0])  # 青：平面（例：床）
o3d.io.write_point_cloud("ransac_plane.ply", plane_cloud)
print("✅ RANSACによる平面抽出完了: ransac_plane.ply")

# === [3] DBSCANクラスタリング（Region Growingの代替）===
print("▶ DBSCANクラスタリング中...")
labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=20, print_progress=True))
max_label = labels.max()

colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
colors[labels < 0] = [0, 0, 0, 1]  # 外れ値は黒
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.io.write_point_cloud("dbscan_clusters.ply", pcd)
print("✅ DBSCANクラスタリング完了: dbscan_clusters.ply")

