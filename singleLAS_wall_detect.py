import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# --- 設定 ---
las_file = r"C:\Users\user\Documents\lab\data\las2\20211029_Marlin[multibeam]_20240625_TUMSAT LiDAR triai-20240627-121535(1)-R20250425-123306.las"
z_limit = 10.0
eps = 1.0
min_samples = 20
z_range_thresh = 1.5      # Z方向の高低差のみを使用（横の厚み制限なし）
output_ply = "wall_candidate_height_only.ply"

# --- Step1: .las 読み込み + 高さフィルタ ---
print("[1] .lasファイル読み込み中...")
with laspy.open(las_file) as f:
    las = f.read()
    points = np.vstack((las.x, las.y, las.z)).T

points = points[points[:, 2] < z_limit]
print(f"使用点数: {len(points)}")

# --- Step2: クラスタリング（DBSCAN）---
print("[2] DBSCANクラスタリング中...")
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(points)
unique_labels = np.unique(labels)
print(f"クラスタ数（-1除く）: {len(unique_labels[unique_labels != -1])}")

# --- Step3: 壁候補抽出（Z範囲のみで判断）---
print("[3] 壁候補抽出中...")
wall_mask = np.zeros(points.shape[0], dtype=bool)

for lbl in unique_labels:
    if lbl == -1:
        continue
    cluster = points[labels == lbl]
    z_range = np.ptp(cluster[:, 2])

    if z_range >= z_range_thresh:
        wall_mask[labels == lbl] = True

print(f"壁候補点数: {np.sum(wall_mask)} / {points.shape[0]}")

# --- Step4: 可視化用に色付け・出力 ---
colors = np.zeros_like(points)
colors[wall_mask] = [1, 0, 0]         # 壁 → 赤
colors[~wall_mask] = [0.5, 0.5, 0.5]  # その他 → 灰

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(output_ply, pcd)
print(f"[完了] 出力ファイル: {output_ply}")
