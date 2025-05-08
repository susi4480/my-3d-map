import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las2"
voxel_size = 0.2
normal_wall_z_max = 10.0  # これ以上高い壁はビルとみなす

# === [1] .lasファイル一覧取得 ===
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

# === [3] ダウンサンプリング + 法線推定 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
points_ds = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)

# === [4] 4クラス分類: 壁・床・ビル・未分類 ===
colors = np.ones((len(points_ds), 3))  # 白で初期化
wall_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] < normal_wall_z_max)
building_mask = (normals[:, 2] < 0.3) & (points_ds[:, 2] >= normal_wall_z_max)
floor_mask = normals[:, 2] > 0.9

colors[wall_mask] = [1.0, 0.0, 0.0]     # 壁 = 赤
colors[floor_mask] = [0.0, 0.0, 1.0]    # 床（河底）= 青
colors[building_mask] = [1.0, 1.0, 0.0] # ビル = 黄

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(las_dir, "classified_wall_floor_building.ply"), pcd)

# === [5] 床と壁から航行空間の直方体定義 ===
floor_points = points_ds[floor_mask]
wall_points = points_ds[wall_mask]

z_min = floor_points[:, 2].min()
z_max = z_min + 5.0
x_min = np.percentile(wall_points[:, 0], 5)
x_max = np.percentile(wall_points[:, 0], 95)
y_min = np.percentile(wall_points[:, 1], 5)
y_max = np.percentile(wall_points[:, 1], 95)

bbox = o3d.geometry.AxisAlignedBoundingBox([x_min, y_min, z_min], [x_max, y_max, z_max])
box_points = np.asarray(bbox.get_box_points())

# === [6] 線を点群化してワイヤーフレーム化 ===
lines = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

interpolated_points = []
for start, end in lines:
    p0 = box_points[start]
    p1 = box_points[end]
    for t in np.linspace(0, 1, 20):
        interpolated_points.append((1 - t) * p0 + t * p1)

wire_pcd = o3d.geometry.PointCloud()
wire_pcd.points = o3d.utility.Vector3dVector(np.array(interpolated_points))
wire_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # 緑

o3d.io.write_point_cloud(os.path.join(las_dir, "box_wireframe_points.ply"), wire_pcd)

print("✅ 出力完了: 赤=壁、青=床（河底）、黄=ビル、白=未分類 + 緑=航行空間直方体")
