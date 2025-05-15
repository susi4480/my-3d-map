import os
import laspy
import numpy as np
import open3d as o3d

# --- 設定 ---
las_folder = r"C:\Users\user\Documents\lab\data\las2"
z_limit = 10.0                   # ← 案①: 高さ制限を緩めて地面を含める
ring_width = 5.0                # 同心円の幅 [m]
max_radius = 150.0              # センサからの最大距離
ransac_dist_thresh = 0.3        # 平面からの許容誤差 [m]
vertical_threshold = 0.90       # 法線Z成分がこの値以上 → 地面とみなす
output_ply = "ground_candidate_CZM_zlimit10.0.ply"

# --- Step1: .lasファイルを読み込み ---
print("[1] .lasファイルを読み込み中...")
points_all = []

for fname in os.listdir(las_folder):
    if fname.endswith(".las"):
        full_path = os.path.join(las_folder, fname)
        print(f" - 読み込み: {fname}")
        with laspy.open(full_path) as f:
            las = f.read()
            pts = np.vstack((las.x, las.y, las.z)).T

            # --- 案②: Z統計出力 ---
            z_min = np.min(pts[:, 2])
            z_max = np.max(pts[:, 2])
            z_mean = np.mean(pts[:, 2])
            print(f"    Z範囲: {z_min:.2f} ～ {z_max:.2f}（平均: {z_mean:.2f}）")

            # Z制限を適用
            pts = pts[pts[:, 2] < z_limit]
            points_all.append(pts)

if not points_all:
    print("❌ 有効な点が見つかりません")
    exit()

points = np.vstack(points_all)
print(f"✅ 高さ制限後の点数: {len(points)}")

# --- Step2: CZM + RANSAC ---
print("[2] 同心円分割 + RANSAC による地面（床）抽出中...")
ground_mask = np.zeros(points.shape[0], dtype=bool)
# データのXY中心をセンサ中心とみなす
center_xy = np.mean(points[:, :2], axis=0)
xy_shifted = points[:, :2] - center_xy
distances = np.linalg.norm(xy_shifted, axis=1)


for r_min in np.arange(0, max_radius, ring_width):
    r_max = r_min + ring_width
    ring_mask = (distances >= r_min) & (distances < r_max)
    ring_points = points[ring_mask]
    ring_indices = np.where(ring_mask)[0]

    print(f"  ◯ Ring {r_min:.1f}–{r_max:.1f}m: {len(ring_points)}点")
    if len(ring_points) < 100:
        print("   → 点数不足でスキップ")
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ring_points)
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_dist_thresh,
                                                 ransac_n=3,
                                                 num_iterations=100)
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        print(f"   → 法線Z: {normal[2]:.3f}, Inliers: {len(inliers)}")

        if abs(normal[2]) > vertical_threshold:
            ground_mask[ring_indices[inliers]] = True
            print("   ✅ 地面として採用")
        else:
            print("   ❌ 傾斜が大きく除外")

    except Exception as e:
        print(f"   ⚠ RANSAC失敗: {e}")
        continue

print(f"\n✅ 地面候補点数: {np.sum(ground_mask)} / {len(points)}")

# --- Step3: 可視化・出力 ---
colors = np.zeros_like(points)
colors[ground_mask] = [1, 0, 0]         # 赤：地面
colors[~ground_mask] = [0.5, 0.5, 0.5]  # 灰：その他

pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(points)
pcd_out.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(output_ply, pcd_out)

print(f"\n🎉 [完了] 出力ファイル: {output_ply}")

