# -*- coding: utf-8 -*-
"""
【機能】中心線(path.json) × FOV空間 × Scan-to-Map 初期位置合わせ
--------------------------------------------------------------
- path.json（中心線座標）を読み込み、各点を視点として扱う
- 地図点群(PLY)を読み込み、視点ごとにFOV空間画像を生成
- LiDAR(.pcap/.json)から順次フレームを読み出してFOV画像化
- NCCスコアで一致度を比較し、閾値超の時点を初期位置として出力
"""

import os, math, json, numpy as np, open3d as o3d, cv2
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, SensorInfo

# ========= 入出力設定 =========
MAP_PATH   = r"/workspace/data/1016_merged_lidar_uesita.ply"
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
PATH_JSON  = r"/workspace/data/path_resampled.json"
OUTPUT_DIR = r"/workspace/output/fov_path_alignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= パラメータ =========
H_FOV_DEG = 120.0
V_FOV_DEG = 30.0
H_RES_DEG = 0.2
V_RES_DEG = 0.2
SCORE_THRESHOLD = 0.25
MIN_COMMON_PIX = 200
MAX_FRAMES = 1000

# ------------------------------------------------------------
def deg2rad(d): return d * math.pi / 180.0

def rotate_z(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    c, s = math.cos(deg2rad(yaw_deg)), math.sin(deg2rad(yaw_deg))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return points @ R.T

def points_to_fov_depth(points: np.ndarray) -> np.ndarray:
    """点群をFOV空間に投影（±H_FOV/2 × ±V_FOV/2）"""
    h_bins = int(round(H_FOV_DEG / H_RES_DEG))
    v_bins = int(round(V_FOV_DEG / V_RES_DEG))
    depth = np.full((v_bins, h_bins), np.inf, np.float32)
    if points.size == 0: return np.zeros_like(depth)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    yaw = np.degrees(np.arctan2(y, x))
    pitch = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
    mask = (np.abs(yaw) <= H_FOV_DEG/2) & (np.abs(pitch) <= V_FOV_DEG/2) & (r > 0)
    if not np.any(mask): return np.zeros_like(depth)
    yaw, pitch, r = yaw[mask], pitch[mask], r[mask]

    u = ((yaw + H_FOV_DEG/2) / H_RES_DEG).astype(np.int32)
    v = ((V_FOV_DEG/2 - pitch) / V_RES_DEG).astype(np.int32)
    valid = (u >= 0) & (u < h_bins) & (v >= 0) & (v < v_bins)
    u, v, r = u[valid], v[valid], r[valid]
    lin = v * h_bins + u
    buf = depth.reshape(-1)
    np.minimum.at(buf, lin, r.astype(np.float32))
    depth = buf.reshape(v_bins, h_bins)
    depth[np.isinf(depth)] = 0
    return depth

def depth_to_u8(depth):
    d = depth.copy()
    mask = d > 0
    if np.any(mask):
        dmin, dmax = d[mask].min(), d[mask].max()
        d[mask] = 255*(d[mask]-dmin)/(dmax-dmin)
    return d.astype(np.uint8)

def ncc_on_valid(a, b, min_common):
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common: return -1e9
    A, B = a[m].astype(np.float32), b[m].astype(np.float32)
    A -= A.mean(); B -= B.mean()
    denom = np.sqrt((A*A).sum()*(B*B).sum())+1e-6
    return float((A*B).sum()/denom)

def map_to_fov(map_pts, view_xy, yaw_deg):
    vx, vy = view_xy
    pts = map_pts - np.array([vx, vy, 0])
    pts_rot = rotate_z(pts, -yaw_deg)
    return points_to_fov_depth(pts_rot)

def extract_frames(pcap_path, json_path):
    with open(json_path, "r") as f:
        sensor_info = SensorInfo(f.read())
    xyzlut = XYZLut(sensor_info, use_extrinsics=False)
    source = open_source(pcap_path)
    for i, scans in enumerate(source):
        scan = scans if not isinstance(scans, list) else scans[0]
        xyz = xyzlut(scan)
        yield i, xyz.reshape(-1, 3)

# ------------------------------------------------------------
def main():
    print("🗺 地図点群読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    map_pts = np.asarray(map_pcd.points)
    print(f"✅ 地図点数: {len(map_pts):,}")

    print("📄 中心線読み込み中:", PATH_JSON)
    with open(PATH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    path = data["path"]
    print(f"✅ 中心線点数: {len(path)}")

    print("📡 LiDARフレーム探索中...")
    found = False
    for frame_idx, scan_pts in extract_frames(PCAP_PATH, JSON_PATH):
        if frame_idx >= MAX_FRAMES:
            print("🛑 最大フレームに到達"); break

        scan_depth = points_to_fov_depth(scan_pts)
        scan_u8 = depth_to_u8(scan_depth)

        best = dict(score=-1e9, x=None, y=None, yaw=None, ref_u8=None)
        for i, (x, y) in enumerate(path):
            # yaw角は進行方向から推定 (次の点との角度)
            if i < len(path)-1:
                dx, dy = path[i+1][0]-x, path[i+1][1]-y
                yaw = math.degrees(math.atan2(dy, dx))
            else:
                yaw = 0
            ref_depth = map_to_fov(map_pts, (x, y), yaw)
            ref_u8 = depth_to_u8(ref_depth)
            s = ncc_on_valid(ref_u8, scan_u8, MIN_COMMON_PIX)
            if s > best["score"]:
                best.update(score=s, x=x, y=y, yaw=yaw, ref_u8=ref_u8)
        print(f"Frame {frame_idx}: best={best['score']:.3f} ({best['x']:.1f},{best['y']:.1f},{best['yaw']:.1f})")

        if best["score"] > SCORE_THRESHOLD:
            print(f"✅ 初期位置確定: Frame={frame_idx}, Score={best['score']:.3f}")
            ref_c = cv2.applyColorMap(best["ref_u8"], cv2.COLORMAP_JET)
            qry_c = cv2.applyColorMap(scan_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(ref_c, 0.5, qry_c, 0.5, 0)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"compare_F{frame_idx}.png"), overlay)
            np.savetxt(os.path.join(OUTPUT_DIR, f"best_F{frame_idx}.txt"),
                       np.array([[best["x"], best["y"], best["yaw"], best["score"]]]),
                       fmt="%.6f", header="x,y,yaw,score")
            found = True
            break

    if not found:
        print("⚠ 一致する視点が見つかりませんでした。")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
