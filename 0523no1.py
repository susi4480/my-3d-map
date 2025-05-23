import laspy
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import open3d as o3d
import os
import glob

# === 設定 ===
input_dir = r"C:\Users\user\Documents\lab\data\las_field"
step = 5
grid_res = 0.5               # ← グリッド密度を上げる
search_radius = 15.0         # ← 補間できる範囲を広げる
exclude_radius = 1.0         # ← 欠損と判定しやすくする
output_ply_dir = r"C:\Users\user\Documents\lab\output_ply"

os.makedirs(output_ply_dir, exist_ok=True)

las_files = glob.glob(os.path.join(input_dir, "*.las"))
for input_path in las_files:
    print(f"\n▶ 処理中: {os.path.basename(input_path)}")
    try:
        las = laspy.read(input_path)
        xyz = np.vstack((las.x, las.y, las.z)).T
        riverbed = xyz  # 川底点群として処理

        if len(riverbed) < 100:
            print("  ⚠ 川底点が少なすぎるためスキップ")
            continue

        y_min, y_max = np.min(riverbed[:, 1]), np.max(riverbed[:, 1])
        left_edges, right_edges = [], []

        for y in np.arange(y_min, y_max, step):
            slice_mask = (riverbed[:, 1] >= y) & (riverbed[:, 1] < y + step)
            slice_pts = riverbed[slice_mask]
            if len(slice_pts) > 0:
                left = slice_pts[np.argmin(slice_pts[:, 0])]
                right = slice_pts[np.argmax(slice_pts[:, 0])]
                left_edges.append(left)
                right_edges.append(right)

        if len(left_edges) < 3 or len(right_edges) < 3:
            print("  ⚠ マスク点が足りません。スキップ")
            continue

        mask_polygon = np.array(left_edges + right_edges[::-1] + [left_edges[0]])
        poly = Polygon(mask_polygon[:, :2])

        x_min, x_max = np.min(mask_polygon[:, 0]), np.max(mask_polygon[:, 0])
        gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                             np.arange(y_min, y_max, grid_res))
        grid_points = np.vstack((gx.ravel(), gy.ravel())).T

        mask = np.array([poly.contains(Point(p)) for p in grid_points])
        masked_grid = grid_points[mask]

        # === 欠損グリッド点の抽出 ===
        tree_exist = cKDTree(riverbed[:, :2])
        distance, _ = tree_exist.query(masked_grid, k=1, distance_upper_bound=exclude_radius)
        no_data_mask = ~np.isfinite(distance)
        missing_grid = masked_grid[no_data_mask]

        # === 補間処理 ===
        tree_interp = cKDTree(riverbed[:, :2])
        interp_points = []
        interp_color = []

        for pt in missing_grid:
            idxs = tree_interp.query_ball_point(pt, r=search_radius)
            if len(idxs) >= 3:
                dists = np.linalg.norm(riverbed[idxs, :2] - pt, axis=1)
                weights = 1 / (dists + 1e-6)
                z_val = np.sum(weights * riverbed[idxs, 2]) / np.sum(weights)
                interp_points.append([pt[0], pt[1], z_val])
                interp_color.append([1.0, 0.0, 0.0])  # 赤

        # === 点群結合・出力 ===
        all_points = riverbed.tolist()
        all_colors = [[0.5, 0.5, 0.5]] * len(riverbed)

        if interp_points:
            all_points += interp_points
            all_colors += interp_color

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(all_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(all_colors))

        ply_name = os.path.splitext(os.path.basename(input_path))[0] + "_idw_refined.ply"
        ply_path = os.path.join(output_ply_dir, ply_name)
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"  ✅ 出力完了: {ply_path} （補間点数: {len(interp_points)}）")

    except Exception as e:
        print(f"  ❌ エラー: {e}")
