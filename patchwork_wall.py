import laspy
import numpy as np
import open3d as o3d

# === 設定 ===
las_path = r"C:\Users\user\Documents\lab\data\las2\20211029_Marlin[multibeam]_20240625_TUMSAT LiDAR triai-20240627-121535(1)-R20250425-123306.las"
voxel_size = 0.3
ring_width = 2.0
ground_z_threshold = 0.2
verticality_threshold = 0.85
ground_z_limit = 4.0        # 地面として許容する最大Z
wall_z_limit = 4.5          # 壁として許容する最大Z

# === [1] LAS読み込み ===
print("[1] LASファイルを読み込み中...")
las = laspy.read(las_path)
points = np.vstack((las.x, las.y, las.z)).T

# === [2] Voxelダウンサンプリング ===
print("[2] Voxelダウンサンプリング中...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
points_ds = np.asarray(pcd.points)

# === [3] Patchwork++風の地面検出 ===
print("[3] Patchwork++風の地面検出中...")
dists = np.linalg.norm(points_ds[:, :2], axis=1)
max_dist = dists.max()
ground_mask = np.zeros(len(points_ds), dtype=bool)

for r_start in np.arange(0, max_dist, ring_width):
    in_ring = (dists >= r_start) & (dists < r_start + ring_width)
    if np.sum(in_ring) < 10:
        continue
    ring_points = points_ds[in_ring]
    z_min = ring_points[:, 2].min()
    ground_in_ring = in_ring & (points_ds[:, 2] <= z_min + ground_z_threshold)
    ground_mask |= ground_in_ring

# ✅ 地面Z制限を適用（制限を超える床は除外）
ground_mask = ground_mask & (points_ds[:, 2] < ground_z_limit)
non_ground_points = points_ds[~ground_mask]

# === [4] 法線ベクトルに基づく壁検出 ===
print("[4] 法線ベクトルに基づく壁検出中...")
pcd_ng = o3d.geometry.PointCloud()
pcd_ng.points = o3d.utility.Vector3dVector(non_ground_points)
pcd_ng.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

normals = np.asarray(pcd_ng.normals)
vertical_mask = (np.abs(normals[:, 2]) < verticality_threshold)

# ✅ 壁Z制限も同時に適用
z_vals = non_ground_points[:, 2]
wall_mask = vertical_mask & (z_vals < wall_z_limit)
wall_points = non_ground_points[wall_mask]

# === [5] Z値に応じたカラー付与とPLY出力 ===
print("[5] Z値に応じたカラー付与とPLY出力中...")
z_vals = wall_points[:, 2]
z_min, z_max = z_vals.min(), z_vals.max()
z_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)

colors = np.zeros_like(wall_points)
colors[:, 0] = z_norm        # R
colors[:, 1] = 1 - z_norm    # G
colors[:, 2] = 0.2           # B定数

wall_pcd = o3d.geometry.PointCloud()
wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
wall_pcd.colors = o3d.utility.Vector3dVector(colors)

out_path = "wall_patchwork_limited.ply"
o3d.io.write_point_cloud(out_path, wall_pcd)
print(f"✅ 出力完了: {out_path}")
