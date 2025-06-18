import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import open3d as o3d
import os
import glob

# === 設定 ===
input_dir = r"C:\Users\user\Documents\lab\data\suidoubasi\test_xyz"
step = 5
grid_res = 0.5
search_radius = 12.0
exclude_radius = 1.0
output_ply_path = r"C:\Users\user\Documents\lab\output_ply\0605no1_xslice_test2.ply"

# === XYZファイル読み込みと統合 ===
xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
all_points = []

for path in xyz_files:
    try:
        xyz = np.loadtxt(path)
        if xyz.shape[1] >= 3:
            all_points.append(xyz[:, :3])  # XYZの最初の3列のみ使用
    except Exception as e:
        print(f"❌ 読み込みエラー: {path} → {e}")

if not all_points:
    print("❌ 有効な点群データが見つかりません。処理を終了します。")
    exit()

riverbed = np.vstack(all_points)

# === スライスごとの上下端点を取得（橋検出なし） ===
x_min, x_max = np.min(riverbed[:, 0]), np.max(riverbed[:, 0])
top_edges, bottom_edges = [], []

for x in np.arange(x_min, x_max, step):
    slice_mask = (riverbed[:, 0] >= x) & (riverbed[:, 0] < x + step)
    slice_pts = riverbed[slice_mask]
    if len(slice_pts) > 0:
        bottom = slice_pts[np.argmin(slice_pts[:, 1])]
        top = slice_pts[np.argmax(slice_pts[:, 1])]
        bottom_edges.append(bottom)
        top_edges.append(top)

# === マスク点数が少なくても続行（警告表示） ===
if len(top_edges) < 3 or len(bottom_edges) < 3:
    print(f"⚠ マスク点が少ないため、マスク形状が不正確になる可能性があります")
    if len(top_edges) < 1 or len(bottom_edges) < 1:
        print("❌ 十分なマスク点がなく、処理を続行できません")
        exit()

mask_polygon = np.array(bottom_edges + top_edges[::-1] + [bottom_edges[0]])
poly = Polygon(mask_polygon[:, :2])

# === グリッド生成（マスク内） ===
y_min, y_max = np.min(mask_polygon[:, 1]), np.max(mask_polygon[:, 1])
gx, gy = np.meshgrid(np.arange(x_min, x_max, grid_res),
                     np.arange(y_min, y_max, grid_res))
grid_points = np.vstack((gx.ravel(), gy.ravel())).T
mask = np.array([poly.contains(Point(p)) for p in grid_points])
masked_grid = grid_points[mask]

# === 欠損グリッド抽出 ===
tree_exist = cKDTree(riverbed[:, :2])
distance, _ = tree_exist.query(masked_grid, k=1, distance_upper_bound=exclude_radius)
no_data_mask = ~np.isfinite(distance)
missing_grid = masked_grid[no_data_mask]

# === IDW補間 ===
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
        interp_color.append([1.0, 0.0, 0.0])  # 赤色（補間点）

# === 出力 ===
all_xyz = riverbed.tolist()
all_color = [[0.5, 0.5, 0.5]] * len(riverbed)  # 元データは灰色

if interp_points:
    all_xyz += interp_points
    all_color += interp_color

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(all_xyz))
pcd.colors = o3d.utility.Vector3dVector(np.array(all_color))
o3d.io.write_point_cloud(output_ply_path, pcd)

print(f"✅ 統合補間完了: {output_ply_path} （補間点数: {len(interp_points)}）")
