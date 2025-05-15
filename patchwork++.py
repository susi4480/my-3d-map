import os
import laspy
import numpy as np
import open3d as o3d

# --- 設定 ---
las_folder = r"C:\Users\user\Documents\lab\data\las2"
z_limit = 10.0                    # 一旦Z=10まで読んで、後でZ<4.0で制限をかける
z_ground_limit = 4.0             # ✅ 地面として使う最大高さ
num_sectors = 90                 # 6度刻みの扇形分割
num_rings = 20                   # 距離に基づくリング分割
max_radius = 150.0               # 最大半径
max_z_diff = 1.0                 # 地面とみなすZ差
output_ply = "ground_candidate_patchwork2_zlimit4.0.ply"

# --- Step1: 点群読み込み ---
print("[1] .lasファイルを読み込み中...")
points_all = []

for fname in os.listdir(las_folder):
    if fname.endswith(".las"):
        path = os.path.join(las_folder, fname)
        print(f" - {fname}")
        with laspy.open(path) as f:
            las = f.read()
            pts = np.vstack((las.x, las.y, las.z)).T
            pts = pts[pts[:, 2] < z_limit]  # 一旦広めに読んでおく
            points_all.append(pts)

points = np.vstack(points_all)
print(f"✅ 合計点数（Z<{z_limit}）: {len(points)}点")

# --- Step2: センサ中心を点群重心とみなしてリング分割 ---
center_xy = np.mean(points[:, :2], axis=0)
xy = points[:, :2] - center_xy
dists = np.linalg.norm(xy, axis=1)
angles = (np.arctan2(xy[:, 1], xy[:, 0]) + 2 * np.pi) % (2 * np.pi)

ring_idx = np.minimum((dists / max_radius * num_rings).astype(int), num_rings - 1)
sector_idx = (angles / (2 * np.pi / num_sectors)).astype(int)

# --- Step3: 地面候補抽出 ---
print("[2] 地面候補を抽出中...")
ground_mask = np.zeros(points.shape[0], dtype=bool)

for r in range(num_rings):
    for s in range(num_sectors):
        mask = (ring_idx == r) & (sector_idx == s)
        if np.sum(mask) < 5:
            continue
        zone_points = points[mask]
        z_min = np.min(zone_points[:, 2])
        ground_zone = zone_points[:, 2] - z_min < max_z_diff
        global_idx = np.where(mask)[0][ground_zone]
        ground_mask[global_idx] = True

# ✅ 高さ制限を適用（地面として認めるのは Z < 4.0）
ground_mask = ground_mask & (points[:, 2] < z_ground_limit)

print(f"✅ 地面点数（Z<{z_ground_limit}）: {np.sum(ground_mask)} / {len(points)}")

# --- Step4: 色分け + 出力 ---
colors = np.zeros_like(points)
colors[ground_mask] = [1, 0, 0]         # 地面 → 赤
colors[~ground_mask] = [0.5, 0.5, 0.5]  # その他 → 灰

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(output_ply, pcd)
print(f"[完了] 出力ファイル: {output_ply}")
