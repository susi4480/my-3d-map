import laspy
import numpy as np
import open3d as o3d

# === 入力・出力 ===
input_las_path = r"C:\Users\user\Documents\lab\output_ply\suidoubasi_classified_combined.las"
output_las_path = r"C:\Users\user\Documents\lab\output_ply\0609_suidoubasi_classified_filtered_v2.las"

# === 読み込み ===
las = laspy.read(input_las_path)
points = np.vstack([las.x, las.y, las.z]).T

# === Open3Dで処理 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

normals = np.asarray(pcd.normals)
colors = np.ones((len(points), 3))  # 白で初期化

# === 分類マスクの条件変更 ===
wall_mask = (normals[:, 2] < 0.3) & (points[:, 2] < 4.5)
floor_mask = (normals[:, 2] > 0.90) & (points[:, 2] < 3.2)
orange_mask = (points[:, 2] > 2.0) & (points[:, 2] <= 3.2) & (normals[:, 2] > 0.90)
building_mask = (normals[:, 2] < 0.3) & (points[:, 2] >= 4.5)

# === 色分け ===
colors[wall_mask] = [1.0, 0.0, 0.0]       # 赤：壁
colors[floor_mask] = [0.0, 0.0, 1.0]      # 青：床
colors[orange_mask] = [1.0, 0.5, 0.0]     # オレンジ：高めの床
colors[building_mask] = [1.0, 1.0, 0.0]   # 黄：ビル

# === Z > 3.2 の点を除外 ===
valid_mask = points[:, 2] <= 3.2
points = points[valid_mask]
colors = colors[valid_mask]

# === LASとして保存 ===
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets = np.min(points, axis=0)
header.scales = np.array([0.001, 0.001, 0.001])

las_out = laspy.LasData(header)
las_out.x, las_out.y, las_out.z = points[:, 0], points[:, 1], points[:, 2]
rgb = (colors * 255).astype(np.uint16)
las_out.red, las_out.green, las_out.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]

las_out.write(output_las_path)
print(f"✅ 再分類（法線0.90）＆Z>3.2除外済みLASを保存: {output_las_path}")
