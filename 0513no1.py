import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import os
import glob

# --- 設定 ---
las_dir = r"C:\Users\user\Documents\lab\data\las2"
z_limit = 10.0                # 地表からの上限フィルタ
wall_z_max = 4.5              # 壁とみなす最大Z値（それ以上はビル扱い）
eps = 1.0                     # DBSCANパラメータ
min_samples = 20
z_range_thresh = 1.5          # クラスタのZ方向高低差しきい値
output_ply = "0513no2.ply"

# --- Step1: 複数LAS読み込み + 統合 + Zフィルタ ---
print("[1] .lasファイルを読み込み中...")
las_files = glob.glob(os.path.join(las_dir, "*.las"))
all_points = []

for path in las_files:
    print(f" - {os.path.basename(path)}")
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    pts = pts[pts[:, 2] < z_limit]  # Zフィルタ適用
    all_points.append(pts)

if not all_points:
    print("❌ 点群が読み込めませんでした。")
    exit()

points = np.vstack(all_points)
print(f"✅ 使用点数: {len(points)}")

# --- Step2: DBSCANクラスタリング ---
print("[2] DBSCANクラスタリング中...")
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(points)
unique_labels = np.unique(labels)
print(f"クラスタ数（-1除く）: {len(unique_labels[unique_labels != -1])}")

# --- Step3: 壁候補抽出（Z範囲 + 高さ制限）---
print("[3] 壁候補抽出中...")
wall_mask = np.zeros(points.shape[0], dtype=bool)

for lbl in unique_labels:
    if lbl == -1:
        continue
    cluster_idx = np.where(labels == lbl)[0]
    cluster = points[cluster_idx]
    z_range = np.ptp(cluster[:, 2])
    z_max = cluster[:, 2].max()

    if z_range >= z_range_thresh and z_max <= wall_z_max:
        wall_mask[cluster_idx] = True

print(f"✅ 壁候補点数: {np.sum(wall_mask)} / {len(points)}")

# --- Step4: 色付けしてPLY出力 ---
colors = np.zeros_like(points)
colors[wall_mask] = [1.0, 0.0, 0.0]         # 壁 → 赤
colors[~wall_mask] = [0.5, 0.5, 0.5]        # その他 → 灰

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(output_ply, pcd)
print(f"[完了] 出力ファイル: {output_ply}")
