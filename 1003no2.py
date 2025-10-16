# -*- coding: utf-8 -*-
"""
【統合処理】
1. ICPでスキャンを地図に整列
2. 推定位置をUTM/WGS84/yawでCSV保存
3. 整列済みPLY出力
4. 図2風のカラー合成画像 (地図=青, スキャン=赤)
"""

import os, glob
import numpy as np
import laspy, open3d as o3d
import matplotlib.pyplot as plt
from pyproj import Transformer

# === 入出力設定 ===
MAP_LAS = "/output/0925_sita_classified.las"
SCAN_DIR = "/output/scan_sector_1000.las"
OUT_DIR  = "/output/1003no2_icp_and_fig2"

ICP_THRESH = 2.0 # ICPの最近傍許容距離[m]
EPSG_MAP   = "EPSG:32654"
EPSG_WGS84 = "EPSG:4326"

GRID_RES = 0.2
IMG_SIZE = 200

os.makedirs(OUT_DIR, exist_ok=True)

# === ユーティリティ ===
def read_las_xyz(path):
    las = laspy.read(path)
    return np.vstack([las.x, las.y, las.z]).T

def save_las(path, points, color):
    header = laspy.LasHeader(point_format=3, version="1.2")
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = points[:,0], points[:,1], points[:,2]
    N = len(points)
    las_out.red   = np.full(N, color[0], dtype=np.uint16)
    las_out.green = np.full(N, color[1], dtype=np.uint16)
    las_out.blue  = np.full(N, color[2], dtype=np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    las_out.write(path)

def rasterize_color(points_xy, grid_res=0.2, size=200, color=(255,255,255)):
    x = points_xy[:,0] - points_xy[:,0].mean()
    y = points_xy[:,1] - points_xy[:,1].mean()
    xi = (x / grid_res + size//2).astype(int)
    yi = (y / grid_res + size//2).astype(int)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    valid = (xi>=0)&(xi<size)&(yi>=0)&(yi<size)
    img[yi[valid], xi[valid]] = color
    return img

def yaw_from_R(R):
    return np.degrees(np.arctan2(R[1,0], R[0,0]))

# === メイン処理 ===
def main():
    # 地図読み込み
    map_xyz = read_las_xyz(MAP_LAS)
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(map_xyz)

    scans = [SCAN_DIR] if SCAN_DIR.endswith(".las") else sorted(glob.glob(os.path.join(SCAN_DIR, "scan_sector_*.las")))
    if not scans:
        raise RuntimeError("⚠ スキャンが見つかりません")

    transformer = Transformer.from_crs(EPSG_MAP, EPSG_WGS84, always_xy=True)
    rows = ["file,Tx,Ty,Tz,yaw_deg,utm_x,utm_y,utm_z,lon,lat,h"]

    for scan_path in scans:
        scan_xyz = read_las_xyz(scan_path)
        if scan_xyz.size == 0:
            print(f"⚠ 空スキャン: {scan_path}")
            continue

        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan_xyz)

        # === ICP ===
        reg = o3d.pipelines.registration.registration_icp(
            pcd_scan, pcd_map, ICP_THRESH, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        T = reg.transformation
        R, t = T[:3,:3], T[:3, 3]
        yaw_deg = yaw_from_R(R)

        utm_x, utm_y, utm_z = t
        lon, lat = transformer.transform(utm_x, utm_y)
        h = utm_z

        # 整列スキャンを保存
        pcd_scan_aligned = pcd_scan.transform(T.copy())
        ply_out = os.path.join(OUT_DIR, os.path.basename(scan_path).replace(".las","_aligned.ply"))
        o3d.io.write_point_cloud(ply_out, pcd_scan_aligned)

        # LAS (赤)
        save_las(os.path.join(OUT_DIR, os.path.basename(scan_path).replace(".las","_aligned.las")),
                 np.asarray(pcd_scan_aligned.points), [65535,0,0])

        # CSV追記
        rows.append(f"{os.path.basename(scan_path)},{t[0]:.3f},{t[1]:.3f},{t[2]:.3f},{yaw_deg:.3f},{utm_x:.3f},{utm_y:.3f},{utm_z:.3f},{lon:.8f},{lat:.8f},{h:.3f}")

        # === 図2風可視化 ===
        img_map  = rasterize_color(map_xyz[:,:2], GRID_RES, IMG_SIZE, (0,0,255))
        img_scan = rasterize_color(np.asarray(pcd_scan_aligned.points)[:,:2], GRID_RES, IMG_SIZE, (255,0,0))
        img_combined = np.maximum(img_map, img_scan)

        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(img_map); axs[0].set_title("事前地図 (青)")
        axs[1].imshow(img_scan); axs[1].set_title("整列スキャン (赤)")
        axs[2].imshow(img_combined); axs[2].set_title("合成表示")
        for ax in axs: ax.axis("off")
        out_png = os.path.join(OUT_DIR, os.path.basename(scan_path).replace(".las","_compare.png"))
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ {os.path.basename(scan_path)} → yaw={yaw_deg:.2f}°, 保存: {ply_out}, {out_png}")

    # CSV保存
    csv_path = os.path.join(OUT_DIR, "poses_icp.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"📝 軌跡CSV: {csv_path}")

if __name__ == "__main__":
    main()
