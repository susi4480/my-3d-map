import numpy as np
import laspy, open3d as o3d

# === 事前地図読み込み ===
map_las = laspy.read("/data/0925_sita_classified.las")
map_points = np.vstack([map_las.x, map_las.y, map_las.z]).T
pcd_map = o3d.geometry.PointCloud()
pcd_map.points = o3d.utility.Vector3dVector(map_points)
pcd_map = pcd_map.voxel_down_sample(0.3)

# === 擬似スキャン読み込み ===
scan_las = laspy.read("/output/forward_scans/scan_sector_1000.las")  # SELECT_I=1000で作ったやつ
scan_points = np.vstack([scan_las.x, scan_las.y, scan_las.z]).T
pcd_scan = o3d.geometry.PointCloud()
pcd_scan.points = o3d.utility.Vector3dVector(scan_points)

# 初期シフトを仮定（例: 実際は位置ズレがある想定）
init_transform = np.eye(4)
init_transform[:3, 3] = [5, 3, 0]
pcd_scan.transform(init_transform)

# === ICPで位置合わせ ===
threshold = 2.0
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_scan, pcd_map, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("推定された変換行列:\n", reg_p2p.transformation)

# === 結果保存 ===
pcd_scan.transform(reg_p2p.transformation)
o3d.io.write_point_cloud("/output/aligned_scan.ply", pcd_scan)
