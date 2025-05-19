import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las_field"  # .lasファイルが入っているフォルダ
output_dir = os.path.join(las_dir, "highlighted_ply")  # 出力フォルダ
os.makedirs(output_dir, exist_ok=True)

# === LASファイル一覧を取得 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))

if not las_files:
    print("LASファイルが見つかりません。")
    exit()

# === 各ファイルを処理 ===
for las_path in las_files:
    print(f"\n--- 処理中: {os.path.basename(las_path)} ---")

    # [1] 点群読み込み
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # [2] XYグリッド生成
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    grid_size = 1.0
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    # [3] 点密度マップ作成
    hist2d, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_bins, y_bins])

    # [4] 欠損グリッドを抽出（点数が少ない場所）
    low_density_mask = hist2d.T < 3
    missing_coords = []
    for i in range(low_density_mask.shape[0]):
        for j in range(low_density_mask.shape[1]):
            if low_density_mask[i, j]:
                cx = (x_bins[j] + x_bins[j+1]) / 2
                cy = (y_bins[i] + y_bins[i+1]) / 2
                missing_coords.append((cx, cy))
    missing_coords = np.array(missing_coords)

    # [5] 欠損近傍点を赤く、他は白
    colors = np.ones((points.shape[0], 3))  # 白 (1,1,1)
    if len(missing_coords) > 0:
        for i, pt in enumerate(points[:, :2]):
            dists = np.linalg.norm(missing_coords - pt, axis=1)
            if np.any(dists < grid_size * 1.5):
                colors[i] = [1.0, 0.0, 0.0]  # 赤

    # [6] Open3D出力
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    output_name = os.path.splitext(os.path.basename(las_path))[0] + "_highlighted.ply"
    output_path = os.path.join(output_dir, output_name)
    o3d.io.write_point_cloud(output_path, pcd)

    print(f" - 出力: {output_path}")
