# -*- coding: utf-8 -*-
"""
IBGAL-on-Path版（path_resampled.json上の視点で初期位置合わせ・反転対応）
---------------------------------------------------------------------
- LiDARの .pcap / .json から1フレームずつ取り出し
- 各フレームをパノラマ画像化して、path_resampled.json の各視点から見た地図画像と比較
- NCCスコアが閾値を超えた時点で「地図に入った」と判断
- CuPy対応（GPUでNCCを高速実行）
- パス方向が逆の場合は REVERSE_PATH_DIRECTION=True でYawを反転
"""

import os, math, json, numpy as np, open3d as o3d, cv2
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, SensorInfo

# ==== GPU自動切替 ====
try:
    import cupy as cp
    _HAS_CUPY = True
    print("🚀 GPU有効 (CuPy)")
except Exception:
    import numpy as cp
    _HAS_CUPY = False
    print("⚠ GPUなし → CPU実行")

# ========= 入出力設定 =========
MAP_PATH   = r"/workspace/data/1016_merged_lidar_uesita.ply"
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
PATH_JSON  = r"/workspace/data/path_resampled.json"
OUTPUT_DIR = r"/workspace/output/icp_no1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= パラメータ =========
YAW_RES_DEG   = 0.5
PITCH_RES_DEG = 1.0
MIN_COMMON_PIX   = 100
SCORE_THRESHOLD  = 0.22
MAX_FRAMES       = 2000
REVERSE_PATH_DIRECTION = True  # ← TrueでYawを180°反転

# ------------------------------------------------------------
def deg2rad(d): return d * math.pi / 180.0
def rotate_z(points: np.ndarray, yaw_deg: float) -> np.ndarray:
    c, s = math.cos(deg2rad(yaw_deg)), math.sin(deg2rad(yaw_deg))
    R = np.array([[ c, -s, 0], [ s, c, 0], [0, 0, 1]], dtype=np.float64)
    return points @ R.T

def to_polar_image(points: np.ndarray) -> np.ndarray:
    """点群 → パノラマ深度画像"""
    yaw_bins   = int(round(360.0 / YAW_RES_DEG))
    pitch_bins = int(round(180.0 / PITCH_RES_DEG))
    if points.size == 0:
        return np.zeros((pitch_bins, yaw_bins), np.uint16)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    yaw   = np.arctan2(y, x)
    pitch = np.arctan2(z, np.sqrt(x*x + y*y))
    yi = ((yaw + np.pi) / deg2rad(YAW_RES_DEG)).astype(np.int32)
    pi = ((pitch + np.pi / 2.0) / deg2rad(PITCH_RES_DEG)).astype(np.int32)
    valid = (yi >= 0) & (yi < yaw_bins) & (pi >= 0) & (pi < pitch_bins)
    if not np.any(valid):
        return np.zeros((pitch_bins, yaw_bins), np.uint16)
    yi, pi, rr = yi[valid], pi[valid], r[valid]
    rr_mm = np.minimum(rr * 1000.0, float(np.iinfo(np.uint16).max)).astype(np.uint16)
    img = np.zeros((pitch_bins, yaw_bins), dtype=np.uint16)
    lin = pi * yaw_bins + yi
    maxv = np.iinfo(np.uint16).max
    buf = np.full(img.size, maxv, dtype=np.uint32)
    np.minimum.at(buf, lin, rr_mm.astype(np.uint32))
    img = buf.reshape(pitch_bins, yaw_bins).astype(np.uint16)
    img[img == maxv] = 0
    return img

def ncc_on_valid(a, b, min_common):
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common: return -1e9
    av, bv = a[m].astype(np.float32), b[m].astype(np.float32)
    av -= av.mean(); bv -= bv.mean()
    denom = np.sqrt((av*av).sum() * (bv*bv).sum()) + 1e-6
    return float((av*bv).sum() / denom)

def pc_to_image_for_path(map_pts, viewpoint_xy, yaw_deg):
    vx, vy = viewpoint_xy
    pts = map_pts - np.array([vx, vy, 0.0], dtype=np.float64)
    pts = rotate_z(pts, -yaw_deg)
    return to_polar_image(pts)

def extract_all_frames(pcap_path, json_path):
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
    print("🗺 地図読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    map_pts = np.asarray(map_pcd.points).astype(np.float64)
    print(f"✅ 地図点数: {len(map_pts):,}")

    print("📄 path_resampled.json 読み込み中...")
    with open(PATH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    path = np.array(data["path"], dtype=np.float64)
    print(f"✅ path点数: {len(path)}")

    # ---- LiDARフレームを順次試行 ----
    for frame_idx, scan_pts in extract_all_frames(PCAP_PATH, JSON_PATH):
        if frame_idx >= MAX_FRAMES:
            print("🛑 最大フレーム数に到達。終了。")
            break

        print(f"\n📡 フレーム {frame_idx} 解析中... (点数={len(scan_pts):,})")
        scan_img = to_polar_image(scan_pts)

        best = dict(score=-1e9, x=None, y=None, yaw=None)
        for i in range(len(path)-1):
            x0, y0 = path[i]
            x1, y1 = path[i+1]
            yaw_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
            if REVERSE_PATH_DIRECTION:
                yaw_deg = (yaw_deg + 180.0) % 360.0  # ← 反転対応
            ref_img = pc_to_image_for_path(map_pts, (x0, y0), yaw_deg)
            s = ncc_on_valid(ref_img, scan_img, MIN_COMMON_PIX)
            if s > best["score"]:
                best.update(score=s, x=x0, y=y0, yaw=yaw_deg)

        print(f"→ ベストスコア={best['score']:.3f} @ (x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f})")

        if best["score"] > SCORE_THRESHOLD:
            print(f"✅ 初期位置発見! Frame={frame_idx}, Score={best['score']:.3f}")
            save_results(map_pts, scan_img, best, frame_idx)
            return

    print("⚠ 有効な一致が見つかりませんでした。")

# ------------------------------------------------------------
def save_results(map_pts, scan_img, best, frame_idx):
    ref_best = pc_to_image_for_path(map_pts, (best["x"], best["y"]), best["yaw"])
    ref_norm = cv2.normalize(ref_best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    qry_norm = cv2.normalize(scan_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    side = np.hstack([ref_norm, qry_norm])
    overlay = cv2.addWeighted(cv2.cvtColor(ref_norm, cv2.COLOR_GRAY2BGR),
                              0.5, cv2.cvtColor(qry_norm, cv2.COLOR_GRAY2BGR), 0.5, 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"compare_side_by_side_F{frame_idx}.png"), side)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"compare_overlay_F{frame_idx}.png"), overlay)
    np.savetxt(os.path.join(OUTPUT_DIR, f"best_transform_F{frame_idx}.txt"),
               np.array([[best["x"], best["y"], 0.0, best["yaw"], best["score"]]], dtype=np.float64),
               fmt="%.6f",
               header="x_m, y_m, z_m(=0), yaw_deg, ncc_score")
    print("📂 出力完了:", OUTPUT_DIR)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
