# -*- coding: utf-8 -*-
"""
IBGAL (Yaw-Shift版, 1フレーム高速化)
- スキャン画像は1度だけ生成し、Yaw変化は列シフト(np.roll)で表現
- XY×Yaw探索を高速化
- 出力: compare_side_by_side / compare_overlay_color / estimated_pose_on_map / alignment_topdown
"""

import os, math, numpy as np, open3d as o3d, cv2, matplotlib.pyplot as plt
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ========= パス =========
PCAP_PATH  = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = "/workspace/output/1013_lidar_map.ply"
OUTPUT_DIR = "/workspace/output/1015no5_ibgal_xyyaw_compare_yawshift"
FRAME_INDEX = 5000

# ========= パノラマ設定 =========
YAW_RES_DEG    = 0.25
PITCH_RES_DEG  = 0.5
PITCH_MIN_DEG  = -22.5
PITCH_MAX_DEG  =  22.5
DEPTH_SCALE    = 100.0
MAX_U16        = np.iinfo(np.uint16).max

# ========= 探索設定 =========
SEARCH_XY_RADIUS = 40.0
SEARCH_XY_STEP   = 2.0
YAW_STEP_DEG     = 1

# ========= スコア/閾値 =========
NCC_MIN_COMMON_PIX   = 3000
FAIL_SCORE_THRESHOLD = 0.25

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def deg2rad(d): return d * math.pi / 180.0

def to_polar_image(points, yaw_res_deg, pitch_res_deg, pitch_min_deg, pitch_max_deg, depth_scale):
    yaw_bins   = int(round(360.0 / yaw_res_deg))
    pitch_bins = int(round((pitch_max_deg - pitch_min_deg) / pitch_res_deg))
    if points.size == 0:
        return np.zeros((pitch_bins, yaw_bins), np.uint16)
    x, y, z = points[:,0], points[:,1], points[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    yaw   = np.arctan2(y, x)
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))
    pmin = deg2rad(pitch_min_deg); pmax = deg2rad(pitch_max_deg)
    valid_pitch = (pitch >= pmin) & (pitch <= pmax)
    yi = ((yaw + np.pi) / deg2rad(yaw_res_deg)).astype(np.int32)
    pi = ((pitch - pmin) / deg2rad(pitch_res_deg)).astype(np.int32)
    valid = valid_pitch & (yi >= 0) & (yi < yaw_bins) & (pi >= 0) & (pi < pitch_bins)
    if not np.any(valid):
        return np.zeros((pitch_bins, yaw_bins), np.uint16)
    yi, pi, rr = yi[valid], pi[valid], r[valid]
    rr_q = (rr * depth_scale)
    rr_q = np.minimum(rr_q, float(MAX_U16)).astype(np.uint16)
    img = np.zeros((pitch_bins, yaw_bins), dtype=np.uint16)
    lin = pi * yaw_bins + yi
    buf = np.full(img.size, MAX_U16, dtype=np.uint32)
    np.minimum.at(buf, lin, rr_q.astype(np.uint32))
    img = buf.reshape(pitch_bins, yaw_bins).astype(np.uint16)
    img[img == MAX_U16] = 0
    return img

def to_edges(img_u16):
    if img_u16.size == 0: return img_u16
    g8 = cv2.normalize(img_u16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.Canny(g8, 30, 150)

def ncc_on_valid(a, b, min_common):
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common: return -1e9
    av = a[m].astype(np.float32); bv = b[m].astype(np.float32)
    av -= av.mean(); bv -= bv.mean()
    denom = np.sqrt((av*av).sum() * (bv*bv).sum()) + 1e-6
    return float((av*bv).sum() / denom)

def pc_to_image_for_viewpoint(map_pts, viewpoint_xy, z_view=0.0):
    vx, vy = viewpoint_xy
    pts = map_pts - np.array([vx, vy, z_view], dtype=np.float64)
    return to_polar_image(pts, YAW_RES_DEG, PITCH_RES_DEG, PITCH_MIN_DEG, PITCH_MAX_DEG, DEPTH_SCALE)

def extract_frame_points_from_pcap(pcap_path, json_path, frame_index):
    with open(json_path, "r") as f:
        sensor_info = SensorInfo(f.read())
    xyzlut = XYZLut(sensor_info, use_extrinsics=False)
    source = open_source(pcap_path)
    for i, scan in enumerate(source):
        if isinstance(scan, list):
            if len(scan) == 0: continue
            scan = scan[0]
        if i == frame_index:
            xyz = xyzlut(scan)
            rng = scan.field(ChanField.RANGE)
            valid = (rng > 0)
            pts = xyz.reshape(-1,3)[valid.reshape(-1)]
            print(f"✅ 抽出成功: frame={i}, 点数={len(pts)}")
            return pts
    raise ValueError("指定フレームが見つかりません")

def visualize_outputs_only_four(map_pts, scan_pts, best, outdir):
    ref_img = pc_to_image_for_viewpoint(map_pts, (best["x"], best["y"]), z_view=0.0)
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG, PITCH_MIN_DEG, PITCH_MAX_DEG, DEPTH_SCALE)
    shift_cols = int(round(best["yaw"] / YAW_RES_DEG))
    scan_rot = np.roll(scan_img, shift_cols, axis=1)
    ref_norm  = cv2.normalize(ref_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scan_norm = cv2.normalize(scan_rot, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    side = np.hstack([ref_norm, scan_norm])
    side_rgb = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
    cv2.putText(side_rgb, "Map render (left) | LiDAR render (right)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir, "compare_side_by_side.png"), side_rgb)
    h, w = ref_norm.shape
    overlay = np.zeros((h, w, 3), np.uint8)
    overlay[...,1] = ref_norm
    overlay[...,2] = scan_norm
    cv2.putText(overlay, "Overlay: Map=Green, LiDAR=Red (gray=match)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir, "compare_overlay_color.png"), overlay)
    plt.figure(figsize=(8,8))
    plt.scatter(map_pts[:,0], map_pts[:,1], s=0.2, c='gray', alpha=0.5)
    plt.arrow(best["x"], best["y"], 5*math.cos(deg2rad(best["yaw"])), 5*math.sin(deg2rad(best["yaw"])),
              color='red', head_width=1.0, length_includes_head=True)
    plt.title(f"Estimated Pose on Map (score={best['score']:.3f})")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.axis("equal")
    plt.savefig(os.path.join(outdir, "estimated_pose_on_map.png"), dpi=300); plt.close()
    yaw = deg2rad(best["yaw"])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0,0,1]])
    scan_tf = (scan_pts @ Rz.T) + np.array([best["x"], best["y"], 0.0])
    plt.figure(figsize=(8,8))
    plt.scatter(map_pts[:,0], map_pts[:,1], s=0.2, c='gray', alpha=0.5, label="Map")
    plt.scatter(scan_tf[:,0], scan_tf[:,1], s=0.5, c='red', alpha=0.6, label="Scan")
    plt.axis("equal"); plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title("Map vs Scan Alignment (Top-Down)")
    plt.savefig(os.path.join(outdir, "alignment_topdown.png"), dpi=300); plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    map_pts = np.asarray(map_pcd.points)
    map_pts[:,2] *= -1
    scan_pts = extract_frame_points_from_pcap(PCAP_PATH, JSON_PATH, FRAME_INDEX)
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG, PITCH_MIN_DEG, PITCH_MAX_DEG, DEPTH_SCALE)
    scan_edge = to_edges(scan_img)

    # --- Yawシフトで総当たり ---
    xs = np.arange(-SEARCH_XY_RADIUS, SEARCH_XY_RADIUS + 1e-9, SEARCH_XY_STEP)
    ys = np.arange(-SEARCH_XY_RADIUS, SEARCH_XY_RADIUS + 1e-9, SEARCH_XY_STEP)
    yaw_candidates = list(np.arange(-180, 181, YAW_STEP_DEG))
    best = dict(score=-1e9, x=None, y=None, yaw=None)
    rolled_cache = {yd: np.roll(scan_edge, int(round(yd / YAW_RES_DEG)), axis=1) for yd in yaw_candidates}

    for y0 in ys:
        for x0 in xs:
            ref = pc_to_image_for_viewpoint(map_pts, (x0, y0), 0.0)
            ref_edge = to_edges(ref)
            for yd in yaw_candidates:
                s = ncc_on_valid(ref_edge, rolled_cache[yd], NCC_MIN_COMMON_PIX)
                if s > best["score"]:
                    best.update(score=s, x=x0, y=y0, yaw=yd)

    if best["score"] < FAIL_SCORE_THRESHOLD:
        print(f"❌ マッチ失敗: score={best['score']:.3f}")
        return
    visualize_outputs_only_four(map_pts, scan_pts, best, OUTPUT_DIR)
    print("✅ 出力4枚 完了")
    print(f"🧭 推定: x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f}°, score={best['score']:.3f}")

if __name__ == "__main__":
    main()
