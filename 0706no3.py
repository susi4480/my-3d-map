import numpy as np
import laspy
import open3d as o3d
import matplotlib.tri as mtri

input_las = "/output/0704_method9_ue.las"
output_ply = "/output/0706_mesh_delaunay.ply"

# === 読み込みと緑抽出 ===
las = laspy.read(input_las)
pts = np.vstack([las.x, las.y, las.z]).astype(np.float32).T
cols = np.vstack([las.red, las.green, las.blue]).astype(np.uint16).T
mask = (cols[:, 0] == 0) & (cols[:, 1] == 255) & (cols[:, 2] == 0)
pts_navi = pts[mask]

if len(pts_navi) == 0:
    raise RuntimeError("❌ 緑点が見つかりません")

# === Voxelでダウンサンプリング ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_navi)
pcd = pcd.voxel_down_sample(voxel_size=0.2)
xyz_down = np.asarray(pcd.points)
xy = xyz_down[:, :2]
z = xyz_down[:, 2]

# === Delaunay三角形分割 ===
print(f"🔼 Delaunay分割中（点数: {len(xy)}）...")
tri = mtri.Triangulation(xy[:, 0], xy[:, 1])

# === Open3Dメッシュ化 ===
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(np.column_stack([xy, z]))
mesh.triangles = o3d.utility.Vector3iVector(tri.triangles)

o3d.io.write_triangle_mesh(output_ply, mesh)
print(f"🎉 Delaunayメッシュ完了: {output_ply}")
