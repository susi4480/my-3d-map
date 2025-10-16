# -*- coding: utf-8 -*-
"""
IBGAL v2 可視化強化版（高精細パラメータ + 両画像比較 + 重ね合わせ）
--------------------------------------------------------------------
- 単一フレームのスキャン点群を使用して地図上の位置(Yaw+XY)を推定。
- 地図PLYをLiDAR視点からパノラマ画像化してNCCで相関最大化。
- 推定結果を地図上に矢印で描画。
- スキャン画像と地図画像を左右並列表示＆半透明重ね合わせ画像で保存。
"""

import os
import math
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ========= パス設定 =========
PCAP_PATH = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
MAP_PATH  = "/workspace/output/1013_lidar_map.ply"
OUTPUT_DIR = "/workspace/output/ibgal_xyyaw_compare"
FRAME_INDEX = 2000

# ========= 画像化パラメータ（推奨設定）=========
YAW_RES_DEG   = 0.25     # より高精細（0.25°）
PITCH_RES_DEG = 0.5      # より高精細（0.5°）
Z_RANGE       = None     # 高さ制限なし

# ========= 探索パラメータ =========
SEARCH_XY_RADIUS = 20.0
SEARCH_XY_STEP   = 5.0
YAW_STEP_DEG     = 5
REFINE_YAW_LOCAL = True

# ========= 地図の軽量化設定 =========
MAP_RANDOM_DOWNSAMPLE = 1.0   # 無効（全点使用）

# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def deg2rad(d): return d * math.pi / 180.0

def to_polar_image(points: np.ndarray, yaw_res_deg: float, pitch_res_deg: float) -> np.ndarray:
    """点群をパノラマ深度画像に変換（16bit, mm単位）。"""
    if points.size == 0:
        return np.zeros((int(180/pitch_res_deg), int(360/yaw_res_deg)), np.uint16)
    x, y, z = points[:,0], points[:,1], points[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    yaw   = np.arctan2(y, x)
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))
    yaw_bins   = int(round(360.0 / yaw_res_deg))
    pitch_bins = int(round(180.0 / pitch_res_deg))
    img = np.zeros((pitch_bins, yaw_bins), dtype=np.uint16)
    yi = ((yaw + np.pi) / deg2rad(yaw_res_deg)).astype(np.int32)
    pi = ((pitch + np.pi/2) / deg2rad(pitch_res_deg)).astype(np.int32)
    valid = (yi >= 0) & (yi < yaw_bins) & (pi >= 0) & (pi < pitch_bins)
    yi, pi, rr = yi[valid], pi[valid], r[valid]
    rr_mm = (rr * 1000.0).astype(np.uint16)
    lin = pi * yaw_bins + yi
    maxval = np.iinfo(np.uint16).max
    buf = np.full(img.size, maxval, dtype=np.uint32)
    np.minimum.at(buf, lin, rr_mm.astype(np.uint32))
    img = buf.reshape(pitch_bins, yaw_bins).astype(np.uint16)
    img[img == maxval] = 0
    return img

def ncc_on_valid(a: np.ndarray, b: np.ndarray) -> float:
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < 500: return -1e9
    av = a[m].astype(np.float32); bv = b[m].astype(np.float32)
    av -= av.mean(); bv -= bv.mean()
    denom = np.sqrt((av*av).sum() * (bv*bv).sum()) + 1e-6
    return float((av*bv).sum() / denom)

def roll_yaw(img: np.ndarray, yaw_deg: float, yaw_res_deg: float) -> np.ndarray:
    shift_cols = int(round(yaw_deg / yaw_res_deg))
    return np.roll(img, shift_cols, axis=1)

def pc_to_image_for_viewpoint(map_pts: np.ndarray, viewpoint_xy: Tuple[float,float], z_view: float=0.0) -> np.ndarray:
    vx, vy = viewpoint_xy
    pts = map_pts - np.array([vx, vy, z_view])
    return to_polar_image(pts, YAW_RES_DEG, PITCH_RES_DEG)

def extract_frame_points_from_pcap(pcap_path: str, json_path: str, frame_index: int) -> np.ndarray:
    with open(json_path, "r") as f:
        sensor_info = SensorInfo(f.read())
    xyzlut = XYZLut(sensor_info, use_extrinsics=False)
    source = open_source(pcap_path)
    for i, scans in enumerate(source):
        scan = scans if not isinstance(scans, list) else scans[0]
        if i == frame_index:
            xyz = xyzlut(scan)
            rng = scan.field(ChanField.RANGE)
            valid = (rng > 0)
            pts = xyz.reshape(-1, 3)[valid.reshape(-1)]
            return pts
    raise ValueError(f"指定フレーム {frame_index} が見つかりません")

# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    print("🗺 地図読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    if MAP_RANDOM_DOWNSAMPLE < 1.0:
        map_pcd = map_pcd.random_down_sample(MAP_RANDOM_DOWNSAMPLE)
    map_pts = np.asarray(map_pcd.points)
    print(f"✅ 地図点数: {len(map_pts):,}")

    print("📡 スキャン抽出中...")
    scan_pts = extract_frame_points_from_pcap(PCAP_PATH, JSON_PATH, FRAME_INDEX)
    print(f"✅ スキャン点数: {len(scan_pts):,}")

    print("🖼 スキャン画像化...")
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "query_depth.png"),
                cv2.normalize(scan_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    cx, cy, _ = map_pts.mean(axis=0)
    xs = np.arange(cx - SEARCH_XY_RADIUS, cx + SEARCH_XY_RADIUS + 1e-6, SEARCH_XY_STEP)
    ys = np.arange(cy - SEARCH_XY_RADIUS, cy + SEARCH_XY_RADIUS + 1e-6, SEARCH_XY_STEP)
    yaw_candidates = list(range(-180, 181, YAW_STEP_DEG))

    rolled_cache = {yd: roll_yaw(scan_img, yd, YAW_RES_DEG) for yd in yaw_candidates}
    best = dict(score=-1e9, x=None, y=None, yaw=None)

    print("🔎 探索開始 (XY×Yaw)...")
    for y0 in ys:
        for x0 in xs:
            ref_img = pc_to_image_for_viewpoint(map_pts, (x0, y0))
            for yd in yaw_candidates:
                s = ncc_on_valid(ref_img, rolled_cache[yd])
                if s > best["score"]:
                    best.update(score=s, x=x0, y=y0, yaw=yd)
    print(f"✅ 最良: score={best['score']:.3f}, x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f}")

    # ---- 可視化 ----
    print("🖼 可視化出力中...")

    # (1) 地図上で位置と向きを描画
    plt.figure(figsize=(8,8))
    plt.scatter(map_pts[:,0], map_pts[:,1], s=0.2, c='gray', alpha=0.5, label="Map")
    plt.arrow(best["x"], best["y"], 
              5*math.cos(deg2rad(best["yaw"])), 5*math.sin(deg2rad(best["yaw"])),
              color='red', head_width=1.0, length_includes_head=True, label="Estimated Pose")
    plt.title("Estimated Position on Map (Red Arrow = LiDAR view)")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.axis("equal"); plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "estimated_pose_on_map.png"), dpi=300)
    plt.close()

    # (2) 地図 vs スキャン俯瞰重ね合わせ
    yaw = deg2rad(best["yaw"])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0,0,1]])
    scan_transformed = (scan_pts @ Rz.T) + np.array([best["x"], best["y"], 0])
    plt.figure(figsize=(8,8))
    plt.scatter(map_pts[:,0], map_pts[:,1], s=0.2, c='gray', alpha=0.5, label="Map")
    plt.scatter(scan_transformed[:,0], scan_transformed[:,1], s=0.5, c='red', alpha=0.6, label="Scan")
    plt.axis("equal"); plt.legend()
    plt.title("Map vs Scan Alignment (Top-Down)")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.savefig(os.path.join(OUTPUT_DIR, "alignment_topdown.png"), dpi=300)
    plt.close()

    # (3) 深度画像の比較（左右並列 & 半透明重ね）
    ref_best = pc_to_image_for_viewpoint(map_pts, (best["x"], best["y"]))
    qry_best = roll_yaw(scan_img, best["yaw"], YAW_RES_DEG)

    ref_norm = cv2.normalize(ref_best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    qry_norm = cv2.normalize(qry_best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "reference_depth_best.png"), ref_norm)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "query_depth_rotated.png"), qry_norm)

    # 左右並列比較画像
    combined = np.hstack([ref_norm, qry_norm])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "compare_side_by_side.png"), combined)

    # 半透明重ね合わせ画像（緑=地図, 赤=スキャン）
    ref_col = cv2.cvtColor(ref_norm, cv2.COLOR_GRAY2BGR)
    qry_col = cv2.cvtColor(qry_norm, cv2.COLOR_GRAY2BGR)
    overlap = cv2.addWeighted(ref_col, 0.5, qry_col, 0.5, 0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "compare_overlay.png"), overlap)

    print("📸 画像出力完了:")
    print("   - compare_side_by_side.png : 左=地図 / 右=スキャン")
    print("   - compare_overlay.png       : 半透明重ね合わせ")
    print(f"🧭 推定位置: x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f}°, score={best['score']:.3f}")

if __name__ == "__main__":
    main()
