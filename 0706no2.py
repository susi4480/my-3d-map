import numpy as np
import laspy
import open3d as o3d

input_las = "/output/0704_method9_ue.las"
output_ply = "/output/0706_mesh_ashape_ue.ply"

# === 読み込みと緑抽出 ===
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).astype(np.float32).T
cols = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T
mask = (cols[:, 0] == 0) & (cols[:, 1] == 255) & (cols[:, 2] == 0)
pts_navi = pts[mask]

if len(pts_navi) == 0:
    raise RuntimeError("❌ 緑点が見つかりません")

# === 点群生成・法線推定 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_navi)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

# === αシェイプでメッシュ化 ===
print("🔄 α-Shape メッシュ中...")
alpha = 0.5  # 小さすぎると穴、 大きすぎると全体が覆われる
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 α-Shape メッシュ完了: {output_ply}")
