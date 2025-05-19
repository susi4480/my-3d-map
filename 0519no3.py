import laspy
import numpy as np
import open3d as o3d
import os
import glob
from scipy.spatial import cKDTree
import alphashape
from shapely.geometry import Point

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las_field"
output_dir = r"C:\Users\user\Documents\lab\output_ply"
os.makedirs(output_dir, exist_ok=True)

# === IDW補間関数 ===
def idw_interpolate(xyz_known, query_xy, k=8, power=2):
    tree = cKDTree(xyz_known[:, :2])
    dists, idxs = tree.query(query_xy, k=k)
    weights = 1 / (dists ** power + 1e-8)
    weights /= weights.sum(axis=1)[:, None]
    interp_z = np.sum(weights * xyz_known[idxs, 2], axis=1)
    return interp_z

# === .lasファイルを処理 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
if not las_files:
    print("❌ .lasファイルが見つかりません。")
    exit()

for las_path in las_files:
    print(f"\n--- 処理中: {os.path.basename(las_path)} ---")

    # [1] .las読み込み
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # [2] 欠損除去
    z = points[:, 2]
    mask_nan = np.isnan(z)
    mask_dummy = z < -1000
    mask_zero = z == 0
    valid_mask = ~(mask_nan | mask_dummy | mask_zero)
    known_points = points[valid_mask]
    known_xy = known_points[:, :2]

    print(f" - 総点数: {len(points):,}")
    print(f" - 有効点数: {len(known_points):,}")

    # [3] AlphaShape（川底の凹んだ形状にフィット）
    alpha_shape = alphashape.alphashape(known_xy, alpha=1.0)  # α値は調整可能
    if alpha_shape is None or alpha_shape.is_empty or alpha_shape.geom_type != "Polygon":
        print(" - AlphaShape失敗。スキップ。")
        continue

    # [4] XYグリッドを生成
    x_min, x_max = known_xy[:, 0].min(), known_xy[:, 0].max()
    y_min, y_max = known_xy[:, 1].min(), known_xy[:, 1].max()
    grid_size = 0.5  # 解像度を高めに
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, grid_size),
        np.arange(y_min, y_max, grid_size)
    )
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # [5] 補間対象グリッド（α領域内で、既知点の近く）を抽出
    points_in = np.array([
        pt for pt in grid_coords
        if alpha_shape.contains(Point(pt))
    ])

    if len(points_in) == 0:
        print(" - 川内グリッドなし。スキップ。")
        continue

    # [6] さらに「近傍に既知点がある」条件で絞る
    tree = cKDTree(known_xy)
    dists, _ = tree.query(points_in, k=1)
    mask_near = dists < 8.0  # 許容距離
    missing_coords = points_in[mask_near]

    print(f" - 補間対象点数: {len(missing_coords):,}")

    # [7] IDW補間
    interpolated_z = idw_interpolate(known_points, missing_coords)
    interpolated_points = np.hstack((missing_coords, interpolated_z[:, np.newaxis]))

    # [8] 色付けと統合
    white = np.tile([[1.0, 1.0, 1.0]], (known_points.shape[0], 1))
    red = np.tile([[1.0, 0.0, 0.0]], (interpolated_points.shape[0], 1))

    all_points = np.vstack((known_points, interpolated_points))
    all_colors = np.vstack((white, red))

    # [9] 出力
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    base_name = os.path.splitext(os.path.basename(las_path))[0]
    output_path = os.path.join(output_dir, base_name + "_idw_alphashape.ply")
    o3d.io.write_point_cloud(output_path, pcd)

    print(f"✅ 出力完了: {output_path}")
