import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las2"
z_threshold_wall = 0.5
z_threshold_floor = 0.05
voxel_size = 0.2

# === [1] .lasファイル一覧を取得 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
print(f"[1] {len(las_files)} ファイルを検出しました")

# === [2] 点群を統合 ===
all_points = []
for path in las_files:
    print(f"  読み込み中: {os.path.basename(path)}")
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    all_points.append(points)

all_points = np.vstack(all_points)
print(f"[2] 統合点群数: {len(all_points)}")

# === [3] Open3Dでダウンサンプリング ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
points_ds = np.asarray(pcd.points)
print(f"[3] ダウンサンプリング後点数: {len(points_ds)}")

# === [4] Z差による壁・床・その他分類と色付け ===
points_sorted = points_ds[np.lexsort((points_ds[:,1], points_ds[:,0]))]
dz = np.abs(np.diff(points_sorted[:, 2], prepend=points_sorted[0, 2]))

colors = np.ones((len(points_sorted), 3))  # 白で初期化
colors[dz > z_threshold_wall] = [1.0, 0.0, 0.0]  # 赤＝壁
colors[dz < z_threshold_floor] = [0.0, 0.0, 1.0]  # 青＝床

pcd_colored = o3d.geometry.PointCloud()
pcd_colored.points = o3d.utility.Vector3dVector(points_sorted)
pcd_colored.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(las_dir, "all_classified_colored.ply"), pcd_colored)

# === [5] 線で囲う直方体（ワイヤーフレーム）作成 ===
wall_points = points_sorted[dz > z_threshold_wall]
floor_points = points_sorted[dz < z_threshold_floor]

z_min = floor_points[:, 2].min()
z_max = z_min + 5.0
x_min = np.percentile(wall_points[:, 0], 5)
x_max = np.percentile(wall_points[:, 0], 95)
y_min = np.percentile(wall_points[:, 1], 5)
y_max = np.percentile(wall_points[:, 1], 95)

bbox = o3d.geometry.AxisAlignedBoundingBox([x_min, y_min, z_min], [x_max, y_max, z_max])
box_points = np.asarray(bbox.get_box_points())

# 線のインデックスと色（12本）
lines = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]
colors_line = [[0.0, 1.0, 0.0] for _ in lines]  # 緑

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(box_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors_line)

o3d.io.write_line_set(os.path.join(las_dir, "all_box_wireframe.ply"), line_set)

print("✅ 壁・床・その他の色分け点群 + 線で囲まれた直方体を出力しました。")
