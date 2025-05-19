import laspy
import numpy as np
import open3d as o3d
import os
import glob
from scipy.spatial import cKDTree

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las_field"  # 対象フォルダ
output_dir = r"C:\Users\user\Documents\lab\output_ply"
os.makedirs(output_dir, exist_ok=True)

# === 欠損検出 & 補間関数（再利用） ===
def idw_interpolate(xyz_known, query_xy, k=8, power=2):
    tree = cKDTree(xyz_known[:, :2])
    dists, idxs = tree.query(query_xy, k=k)
    weights = 1 / (dists ** power + 1e-8)
    weights /= weights.sum(axis=1)[:, None]
    interp_z = np.sum(weights * xyz_known[idxs, 2], axis=1)
    return interp_z

# === .lasファイルを取得 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))

if not las_files:
    print("❌ .lasファイルが見つかりませんでした。")
    exit()

# === 各ファイルを処理 ===
for las_path in las_files:
    print(f"\n--- 処理中: {os.path.basename(las_path)} ---")

    # [1] LAS読み込み
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # [2] 欠損除去
    z = points[:, 2]
    mask_nan = np.isnan(z)
    mask_dummy = z < -1000
    mask_zero = z == 0
    valid_mask = ~(mask_nan | mask_dummy | mask_zero)
    known_points = points[valid_mask]

    print(f" - 総点数: {len(points):,}")
    print(f" - 有効点数: {len(known_points):,}")

    # [3] グリッドを使って欠損グリッド抽出
    x_min, x_max = known_points[:, 0].min(), known_points[:, 0].max()
    y_min, y_max = known_points[:, 1].min(), known_points[:, 1].max()
    grid_size = 1.0
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    hist2d, _, _ = np.histogram2d(known_points[:, 0], known_points[:, 1], bins=[x_bins, y_bins])
    low_density_mask = hist2d.T < 3

    missing_coords = []
    for i in range(low_density_mask.shape[0]):
        for j in range(low_density_mask.shape[1]):
            if low_density_mask[i, j]:
                cx = (x_bins[j] + x_bins[j + 1]) / 2
                cy = (y_bins[i] + y_bins[i + 1]) / 2
                missing_coords.append((cx, cy))

    if not missing_coords:
        print(" - 欠損候補なし。スキップします。")
        continue

    missing_coords = np.array(missing_coords)
    print(f" - 欠損候補数: {len(missing_coords)}")

    # [4] IDW補間でZを推定
    interp_z = idw_interpolate(known_points, missing_coords)
    interp_points = np.hstack((missing_coords, interp_z[:, np.newaxis]))

    # [5] 点群＋色を統合（白：既知点、赤：補間点）
    white = np.tile([[1.0, 1.0, 1.0]], (known_points.shape[0], 1))
    red = np.tile([[1.0, 0.0, 0.0]], (interp_points.shape[0], 1))

    all_points = np.vstack((known_points, interp_points))
    all_colors = np.vstack((white, red))

    # [6] Open3Dで出力
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    base_name = os.path.splitext(os.path.basename(las_path))[0]
    output_path = os.path.join(output_dir, base_name + "_idw_filled.ply")
    o3d.io.write_point_cloud(output_path, pcd)

    print(f"✅ 出力完了: {output_path}")
