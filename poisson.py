import os
import numpy as np
import open3d as o3d
import laspy

# === 設定 ===
input_file = r"/home/edu3/lab/data/las_output/MBES_02.las"
output_dir = r"/home/edu3/lab/output_strategy"
os.makedirs(output_dir, exist_ok=True)

print("[Poisson] LASファイル読み込み中...")
las = laspy.read(input_file)
points = np.vstack((las.x, las.y, las.z)).T  # numpy配列に変換

# === Open3Dで点群作成と法線推定 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(50)

# === Poissonメッシュ生成 ===
print("[Poisson] 補間処理中...")
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# === 書き出し ===
out_path = os.path.join(output_dir, "poisson_depth8.ply")
o3d.io.write_triangle_mesh(out_path, mesh)
print("[Poisson] 完了 ->", out_path)
