import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las2"
output_dir = os.path.join(las_dir, "mls_output")
os.makedirs(output_dir, exist_ok=True)

z_limit = 4.5     # 川底の上限
search_radius = 1.0  # MLSの近傍探索範囲

# === ファイル処理 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
for las_path in las_files:
    base = os.path.splitext(os.path.basename(las_path))[0]
    print(f"\n--- 処理中: {base} ---")

    # [1] LAS読み込み
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    z = points[:, 2]
    mask = (~np.isnan(z)) & (z > -1000) & (z < z_limit)
    filtered_points = points[mask]

    if len(filtered_points) < 1000:
        print(" ⚠ 有効点が少なすぎます。スキップ。")
        continue

    # [2] Open3D点群変換
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # [3] 法線推定（MLSの前提）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    pcd.normalize_normals()

    # [4] MLS補間（Open3Dでは近似処理）
    try:
        print(" 🔧 MLS補間中...")
        pcd_smoothed = pcd.compute_moving_least_squares_surface(
            o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30),
            polynomial_order=2
        )
    except Exception as e:
        print(f" ❌ MLS補間失敗: {e}")
        continue

    # [5] 出力
    out_path = os.path.join(output_dir, base + "_mls.ply")
    o3d.io.write_point_cloud(out_path, pcd_smoothed)
    print(f" ✅ 出力完了: {out_path}")
