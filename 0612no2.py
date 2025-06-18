# -*- coding: utf-8 -*-
"""
.las のみで法線推定し、床(法線Z>0.9)を除外 → .xyzと統合 → Z<-0.7を除外 → XY出力
"""

import os
import glob
import numpy as np
from pyproj import Transformer
import laspy
import open3d as o3d

# === 設定 =========================================================
xyz_dir     = r"C:\Users\user\Documents\lab\data\suidoubasi\lidar_xyz_sita"
las_path    = r"C:\Users\user\Documents\lab\output_ply\suidoubasi_sita_with_crs.las"
output_path = r"C:\Users\user\Documents\lab\output_ply\combined_xy_filtered2.xyz"

z_max = 3.5
z_min_final = -0.7
normal_z_th = 0.9
voxel_size  = 0.2
utm_epsg    = "epsg:32654"
transformer = Transformer.from_crs("epsg:4326", utm_epsg, always_xy=True)

# === 1. .xyz 読み込み（Z ≤ 3.5） =========================================
xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
utm_xyz_list = []

for path in xyz_files:
    try:
        data = np.loadtxt(path)
        if data.shape[1] < 3:
            print(f"⚠ 列不足スキップ: {os.path.basename(path)}")
            continue
        lat, lon, z = data[:, 0], data[:, 1], data[:, 2]
        mask = z <= z_max
        if not np.any(mask):
            continue
        x, y = transformer.transform(lon[mask], lat[mask])
        utm_xyz_list.append(np.vstack([x, y, z[mask]]).T)
    except Exception as e:
        print(f"⚠ 読み込み失敗: {path} → {e}")

utm_xyz = np.vstack(utm_xyz_list) if utm_xyz_list else np.empty((0, 3))

# === 2. .las 読み込み（Z ≤ 3.5） ===========================================
try:
    las = laspy.read(las_path)
    x, y, z = las.x, las.y, las.z
    mask_z = z <= z_max
    las_xyz = np.vstack([x[mask_z], y[mask_z], z[mask_z]]).T
except Exception as e:
    print(f"❌ LAS読み込み失敗: {e}")
    las_xyz = np.empty((0, 3))

# === 3. 法線推定（lasのみ・voxelあり） ======================================
pcd_las = o3d.geometry.PointCloud()
pcd_las.points = o3d.utility.Vector3dVector(las_xyz)
pcd_las = pcd_las.voxel_down_sample(voxel_size=voxel_size)
pcd_las.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

las_points_voxel = np.asarray(pcd_las.points)
normals = np.asarray(pcd_las.normals)

# === 4. 床（法線Z > 0.9）を除外 =============================================
is_floor = normals[:, 2] > normal_z_th
las_points_nofloor = las_points_voxel[~is_floor]

# === 5. .xyz + 床除外後las を統合 ===========================================
combined_xyz = np.vstack([utm_xyz, las_points_nofloor])

# === 6. Z < -0.7 の点を除外 ================================================
mask_z_min = combined_xyz[:, 2] >= z_min_final
filtered_xyz = combined_xyz[mask_z_min]

# === 7. XYのみ保存 ========================================================
xy_output = filtered_xyz[:, :2]
np.savetxt(output_path, xy_output, fmt="%.3f")

# === 完了ログ =============================================================
print("🎉 処理完了")
print(f"📄 出力ファイル : {output_path}")
print(f"📌 XYZ点数     : {len(utm_xyz):,}")
print(f"📌 LAS点数     : {len(las_xyz):,} → 床除去後: {len(las_points_nofloor):,}")
print(f"📌 Z条件通過数 : {len(filtered_xyz):,}") 