# -*- coding: utf-8 -*-
"""
IBGAL（中心線制約付き × GPU NCC高速化 × 軽量版）
------------------------------------------------------------
- path_resampled.json の中心線 ±R 範囲のみを探索。
- LiDARスキャン点群をフル解像度で使用（range=0 除外のみ）。
- 各グリッド地点(5m刻み)から地図をパノラマ化し、
  スキャンパノラマとの NCC により一致スコアを評価。
- CuPy があれば GPU 並列で NCC を高速実行。
- 出力: 左右比較PNG / 重ねPNG / 俯瞰図 / best_transform.txt
"""

import os
import math
import json
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ========= パス設定 =========
MAP_PATH   = "/workspace/data/1016_merged_lidar_uesita.ply"
PCAP_PATH  = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
PATH_JSON  = "/workspace/data/path_resampled.json"
OUTPUT_DIR = "/workspace/output/icp_no3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INDEX = 1000    # 検証用フレーム番号

# ========= パラメータ =========
GRID_STEP_M   = 5.0     # 中心線沿いのXY間隔 [m]
SEARCH_RADIUS = 50.0    # 中心線から±範囲 [m]
YAW_STEP_DEG  = 5       # 探索Yaw刻み [deg]
MIN_COMMON_PIX = 2000   # NCC有効画素下限
IBGAL_MIN_SCORE = 0.22  # 一致とみなす閾値
Z_VIEW = 0.0            # 仮想視点高さ

# ========= GPU切替 =========
try:
    import cupy as cp
    _HAS_CUPY = True
    print("🚀 CuPy 有効 (GPU NCC)")
except Exception:
    import numpy as cp
    _HAS_CUPY = False
    print("⚠ CuPyなし → CPU NCC")

# ========= パノラマ分解能 =========
YAW_RES_DEG   = 0.5
PITCH_RES_DEG = 1.0


# ------------------------------------------------------------
# 関数群
# ------------------------------------------------------------
def deg2rad(d): return d * math.pi / 180.0

def to_polar_image(points: np.ndarray,
                   yaw_res_deg: float = YAW_RES_DEG,
                   pitch_res_deg: float = PITCH_RES_DEG) -> np.ndarray:
    """点群→パノラマ深度画像 (uint16, 単位mm, 0=未観測)"""
    yaw_bins   = int(round(360.0 / yaw_res_deg))
    pitch_bins = int(round(180.0 / pitch_res_deg))
    if points.size == 0:
        return np.zeros((pitch_bins, yaw_bins), np.uint16)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x*x + y*y + z*z)
    yaw   = np.arctan2(y, x)
    pitch = np.arctan2(z, np.sqrt(x*x + y*y))

    yi = ((yaw + np.pi) / deg2rad(yaw_res_deg)).astype(np.int32)
    pi = ((pitch + np.pi/2.0) / deg2rad(pitch_res_deg)).astype(np.int32)
    valid = (yi >= 0) & (yi < yaw_bins) & (pi >= 0) & (pi < pitch_bins)
    if not np.any(valid): return np.zeros((pitch_bins, yaw_bins), np.uint16)

    yi, pi, rr = yi[valid], pi[valid], r[valid]
    rr_mm = np.minimum(rr*1000.0, float(np.iinfo(np.uint16).max)).astype(np.uint16)
    img = np.zeros((pitch_bins, yaw_bins), dtype=np.uint16)
    lin = pi * yaw_bins + yi
    maxv = np.iinfo(np.uint16).max
    buf = np.full(img.size, maxv, dtype=np.uint32)
    np.minimum.at(buf, lin, rr_mm.astype(np.uint32))
    img = buf.reshape(pitch_bins, yaw_bins).astype(np.uint16)
    img[img == maxv] = 0
    return img

def roll_yaw(img: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Yaw回転＝列シフト"""
    shift_cols = int(round(yaw_deg / YAW_RES_DEG))
    return np.roll(img, shift_cols, axis=1)

def ncc_gpu(a: np.ndarray, b: np.ndarray, min_common: int) -> float:
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common:
        return -1e9
    av = cp.asarray(a[m], dtype=cp.float32)
    bv = cp.asarray(b[m], dtype=cp.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = cp.sqrt((av*av).sum() * (bv*bv).sum()) + 1e-6
    score = (av*bv).sum() / denom
    return float(score.get()) if _HAS_CUPY else float(score)

def ncc(a, b, min_common): return ncc_gpu(a, b, min_common)

def pc_to_image_for_viewpoint(map_pts: np.ndarray, viewpoint_xy: Tuple[float, float]) -> np.ndarray:
    vx, vy = viewpoint_xy
    rel = map_pts - np.array([vx, vy, Z_VIEW], dtype=np.float64)
    return to_polar_image(rel, YAW_RES_DEG, PITCH_RES_DEG)

def extract_frame_points(pcap_path: str, json_path: str, frame_index: int) -> np.ndarray:
    """指定フレームを抽出（欠測除外のみ）"""
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
    raise ValueError(f"Frame {frame_index} not found")


# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
def main():
    # --- 地図読み込み ---
    print("🗺 地図読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    map_pts = np.asarray(map_pcd.points).astype(np.float64)
    print(f"✅ 地図点数: {len(map_pts):,}")

    # --- スキャン抽出 ---
    print("📡 LiDARフレーム抽出...")
    scan_pts = extract_frame_points(PCAP_PATH, JSON_PATH, FRAME_INDEX)
    print(f"✅ スキャン点数: {len(scan_pts):,}")

    scan_img = to_polar_image(scan_pts)
    print("🖼 スキャン画像生成完了")

    # --- 中心線読み込み ---
    print("📄 path_resampled.json 読み込み中...")
    with open(PATH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    path = np.array(data["path"], dtype=np.float64)
    print(f"✅ 中心線点数: {len(path)}")

    # --- 探索候補生成（中心線±R 範囲）---
    print("🧭 探索グリッド生成中...")
    xs, ys = [], []
    for px, py in path:
        for dx in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS+1e-9, GRID_STEP_M):
            for dy in np.arange(-SEARCH_RADIUS, SEARCH_RADIUS+1e-9, GRID_STEP_M):
                xs.append(px + dx)
                ys.append(py + dy)
    xs, ys = np.array(xs), np.array(ys)
    print(f"✅ グリッド候補: {len(xs):,} 点")

    yaw_candidates = list(range(-180, 181, YAW_STEP_DEG))
    rolled_cache = {yd: roll_yaw(scan_img, yd) for yd in yaw_candidates}

    # --- 総当たり探索（中心線限定） ---
    print("🔍 探索開始 (path領域XY × Yaw)...")
    best = dict(score=-1e9, x=None, y=None, yaw=None)
    for x0, y0 in zip(xs, ys):
        ref_img = pc_to_image_for_viewpoint(map_pts, (x0, y0))
        local_best, local_yaw = -1e9, 0
        for yd in yaw_candidates:
            s = ncc(ref_img, rolled_cache[yd], MIN_COMMON_PIX)
            if s > local_best:
                local_best, local_yaw = s, yd
        if local_best > best["score"]:
            best.update(score=local_best, x=float(x0), y=float(y0), yaw=float(local_yaw))

    if best["x"] is None:
        raise RuntimeError("一致が見つかりませんでした。")

    print(f"✅ 最良スコア: {best['score']:.3f}, x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f}")

    # --- 可視化出力 ---
    ref_best = pc_to_image_for_viewpoint(map_pts, (best["x"], best["y"]))
    qry_best = roll_yaw(scan_img, best["yaw"])
    ref_u8 = cv2.normalize(ref_best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    qry_u8 = cv2.normalize(qry_best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    side = np.hstack([ref_u8, qry_u8])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "compare_side.png"), side)

    overlay = np.zeros((*ref_u8.shape, 3), np.uint8)
    overlay[...,1] = ref_u8
    overlay[...,2] = qry_u8
    cv2.imwrite(os.path.join(OUTPUT_DIR, "compare_overlay.png"), overlay)

    yaw_rad = deg2rad(best["yaw"])
    Rz = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                   [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
                   [0, 0, 1]], dtype=np.float64)
    scan_top = (scan_pts @ Rz.T) + np.array([best["x"], best["y"], 0.0])
    plt.figure(figsize=(8,8))
    plt.scatter(map_pts[:,0], map_pts[:,1], s=0.2, c="gray", alpha=0.5)
    plt.scatter(scan_top[:,0], scan_top[:,1], s=0.4, c="red", alpha=0.6)
    plt.axis("equal"); plt.title(f"Top-Down Overlay (score={best['score']:.3f})")
    plt.savefig(os.path.join(OUTPUT_DIR, "overlay_topdown.png"), dpi=300)
    plt.close()

    np.savetxt(os.path.join(OUTPUT_DIR, "best_transform.txt"),
               np.array([[best["x"], best["y"], 0.0, best["yaw"], best["score"]]]),
               fmt="%.6f", header="x_m, y_m, z_m(=0), yaw_deg, ncc_score")

    print("📸 出力完了:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
