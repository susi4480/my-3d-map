# -*- coding: utf-8 -*-
"""
IBGAL v2 (Yaw+XY 初期合わせ, Z=0視点, Ouster PCAP対応)
------------------------------------------------------------
- 地図PLYを LiDAR視点 (x,y, z=0) からパノラマ画像化
- スキャン(1フレーム)のパノラマ画像と NCC で相関最大化
- まず XY を粗探索しつつ、Yaw は「横方向シフト」で超高速探索
  (スキャン画像を1回だけ生成し、列シフト=Yaw に対応)
- ベスト (x,y,yaw) を出力。検証用に画像も保存。

入出力:
  MAP_PATH:  既存の LiDAR 地図 (PLY; UTMなど絶対座標でもOK)
  PCAP/JSON: Ouster .pcap + .json
  OUTPUT_DIR 以下に成果物を保存

必要:
  pip install ouster-sdk open3d numpy opencv-python matplotlib
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

OUTPUT_DIR = "/workspace/output/ibgal_xyyaw"
FRAME_INDEX = 1000   # 使うフレーム (例: 500)

# ========= 画像化パラメータ =========
YAW_RES_DEG   = 0.5   # パノラマの水平方向解像度[deg/px]
PITCH_RES_DEG = 1.0   # パノラマの鉛直方向解像度[deg/px]
Z_RANGE       = (-5.0, 15.0)  # 高さフィルタ(必要に応じて調整/無効可)

# ========= 探索パラメータ =========
# 視点 (x0,y0, z=0) を中心に XY をグリッド探索
SEARCH_XY_RADIUS = 20.0   # ±この距離[m]
SEARCH_XY_STEP   = 5.0    # グリッド間隔[m]   ← 例: 20,15,10,5 の順で粗→細も可
# Yaw は列シフトで探索（0.5°/px のとき、1° = 2px）
YAW_STEP_DEG     = 5      # 5°刻み（列シフトで高速）
REFINE_YAW_LOCAL = True   # 最良近傍 ±10° を 1° 刻みで再探索

# ========= 速度対策 =========
MAP_RANDOM_DOWNSAMPLE = 0.2   # 地図点群の間引き率 (0<r<=1)。重ければ 0.1 など


# ------------------------------------------------------------
#  ユーティリティ
# ------------------------------------------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def deg2rad(d): return d * math.pi / 180.0

def to_polar_image(points: np.ndarray,
                   yaw_res_deg: float=0.5,
                   pitch_res_deg: float=1.0) -> np.ndarray:
    """
    点群をパノラマ深度画像に変換（レンジ最小値を採用）。
    0=未観測（黒）。返り値 dtype=uint16 (mm単位相当で精度↑), ただし表示保存時は8bitへ正規化。
    """
    if points.size == 0:
        return np.zeros((int(180/pitch_res_deg), int(360/yaw_res_deg)), np.uint16)

    x, y, z = points[:,0], points[:,1], points[:,2]
    r = np.sqrt(x**2 + y**2 + z**2)
    yaw   = np.arctan2(y, x)                                # [-pi, pi]
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))             # [-pi/2, pi/2]

    yaw_bins   = int(round(360.0 / yaw_res_deg))
    pitch_bins = int(round(180.0 / pitch_res_deg))
    img = np.zeros((pitch_bins, yaw_bins), dtype=np.uint16) # 0=未観測
    # 角→インデックス
    yi = ((yaw + np.pi) / deg2rad(yaw_res_deg)).astype(np.int32)
    pi = ((pitch + np.pi/2) / deg2rad(pitch_res_deg)).astype(np.int32)
    valid = (yi >= 0) & (yi < yaw_bins) & (pi >= 0) & (pi < pitch_bins)
    yi, pi, rr = yi[valid], pi[valid], r[valid]
    # 同一ピクセルは最小距離（最近点）
    # 16bit固定小数にしておく（mm相当: *1000）
    rr_mm = (rr * 1000.0).astype(np.uint16)
    # 1Dインデックス化 & reduce.min
    lin = pi * yaw_bins + yi
    # 初期は0なので、0を除外するため一旦大値で初期化してminを取る
    maxval = np.iinfo(np.uint16).max
    buf = np.full(img.size, maxval, dtype=np.uint32)
    # 最小値を取る（np.minimum.at を使う）
    np.minimum.at(buf, lin, rr_mm.astype(np.uint32))
    img = buf.reshape(pitch_bins, yaw_bins).astype(np.uint16)
    img[img == maxval] = 0
    return img

def ncc_on_valid(a: np.ndarray, b: np.ndarray) -> float:
    """0を未観測としてマスクし、共通観測画素のみでNCC。"""
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < 500:  # 観測が少なすぎると不安定
        return -1e9
    av = a[m].astype(np.float32)
    bv = b[m].astype(np.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.sqrt((av*av).sum() * (bv*bv).sum())) + 1e-6
    return float((av*bv).sum() / denom)

def roll_yaw(img: np.ndarray, yaw_deg: float, yaw_res_deg: float) -> np.ndarray:
    """Yaw回転は横方向の循環シフトに等価。"""
    shift_cols = int(round(yaw_deg / yaw_res_deg))
    return np.roll(img, shift_cols, axis=1)

def pc_to_image_for_viewpoint(map_pts: np.ndarray,
                              viewpoint_xy: Tuple[float,float],
                              z_view: float=0.0) -> np.ndarray:
    """
    視点 (x,y,z=z_view) から地図を見たときのパノラマ画像を作る:
    すなわち map_pts - [x,y,z_view] を極座標化。
    """
    vx, vy = viewpoint_xy
    pts = map_pts - np.array([vx, vy, z_view], dtype=np.float64)
    if Z_RANGE is not None:
        pts = pts[(pts[:,2] > Z_RANGE[0]) & (pts[:,2] < Z_RANGE[1])]
    return to_polar_image(pts, YAW_RES_DEG, PITCH_RES_DEG)

def extract_frame_points_from_pcap(pcap_path: str, json_path: str, frame_index: int) -> np.ndarray:
    """PCAPから指定フレームの有効点を抽出（range>0）。"""
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
#  メイン
# ------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    # 1) 地図読み込み & 軽量化
    print("🗺 地図読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    if MAP_RANDOM_DOWNSAMPLE < 1.0:
        map_pcd = map_pcd.random_down_sample(MAP_RANDOM_DOWNSAMPLE)
    map_pts = np.asarray(map_pcd.points).astype(np.float64)
    print(f"✅ 地図点数: {len(map_pts):,}")

    # 2) スキャン1フレーム抽出 → パノラマ画像（基準）
    print("📡 スキャン抽出中...")
    scan_pts = extract_frame_points_from_pcap(PCAP_PATH, JSON_PATH, FRAME_INDEX)
    if Z_RANGE is not None:
        scan_pts = scan_pts[(scan_pts[:,2] > Z_RANGE[0]) & (scan_pts[:,2] < Z_RANGE[1])]
    print(f"✅ スキャン点数: {len(scan_pts):,}")

    print("🖼 スキャン画像化（1回だけ）...")
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG)  # 16bit
    # 確認用に8bit保存
    disp = cv2.normalize(scan_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "query_depth.png"), disp)

    # 3) XYグリッドを作る（地図の平均を起点にするのが無難）
    cx, cy, cz = map_pts.mean(axis=0)
    xs = np.arange(cx - SEARCH_XY_RADIUS, cx + SEARCH_XY_RADIUS + 1e-6, SEARCH_XY_STEP)
    ys = np.arange(cy - SEARCH_XY_RADIUS, cy + SEARCH_XY_RADIUS + 1e-6, SEARCH_XY_STEP)

    yaw_candidates = list(range(-180, 181, YAW_STEP_DEG))

    # 4) 探索
    print("🔎 探索開始 (XY × Yaw) ...")
    best = dict(score=-1e9, x=None, y=None, yaw=None)
    score_xy = np.full((len(ys), len(xs)), -1e9, dtype=np.float32)  # Y行, X列

    # yawは「列シフト」で高速評価するため、あらかじめ全候補の画像を作っておく
    rolled_cache = {yd: roll_yaw(scan_img, yd, YAW_RES_DEG) for yd in yaw_candidates}

    for iy, y0 in enumerate(ys):
        for ix, x0 in enumerate(xs):
            # 地図を視点 (x0,y0, z=0) から画像化（1回）
            ref_img = pc_to_image_for_viewpoint(map_pts, (x0, y0), z_view=0.0)

            # 各yaw（=列シフト）でNCC計算
            local_best = -1e9
            local_yaw  = 0
            for yd in yaw_candidates:
                s = ncc_on_valid(ref_img, rolled_cache[yd])
                if s > local_best:
                    local_best = s
                    local_yaw  = yd

            score_xy[iy, ix] = local_best
            if local_best > best["score"]:
                best.update(score=local_best, x=float(x0), y=float(y0), yaw=float(local_yaw))

    print(f"✅ 粗探索ベスト: score={best['score']:.3f}, x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f}")

    # 5) Yawを局所再探索（±10°を1°刻み）
    if REFINE_YAW_LOCAL:
        print("⛏️ Yaw局所再探索中 (±10° / 1°刻み)...")
        ref_img_best_xy = pc_to_image_for_viewpoint(map_pts, (best["x"], best["y"]), z_view=0.0)
        yaw_refine = list(range(int(best["yaw"]) - 10, int(best["yaw"]) + 11, 1))
        refine_best = best["score"]
        refine_yaw  = best["yaw"]
        for yd in yaw_refine:
            rolled = roll_yaw(scan_img, yd, YAW_RES_DEG)
            s = ncc_on_valid(ref_img_best_xy, rolled)
            if s > refine_best:
                refine_best = s
                refine_yaw  = float(yd)
        best["score"] = refine_best
        best["yaw"]   = refine_yaw
        print(f"✅ 再探索後: score={best['score']:.3f}, yaw={best['yaw']:.1f}")

    # 6) ベスト画像を保存
    ref_best = pc_to_image_for_viewpoint(map_pts, (best["x"], best["y"]), z_view=0.0)
    qry_best = roll_yaw(scan_img, best["yaw"], YAW_RES_DEG)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "reference_depth_best.png"),
                cv2.normalize(ref_best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "query_depth_rotated.png"),
                cv2.normalize(qry_best, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # 7) XYヒートマップ保存（Yawは各XYでの最良スコア）
    plt.figure(figsize=(6,5))
    plt.imshow(score_xy, origin='lower',
               extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='auto', cmap='viridis')
    plt.colorbar(label='Best NCC over yaw')
    plt.scatter([best["x"]], [best["y"]], c='r', marker='x', label='best')
    plt.title("XY search heatmap (best NCC over yaw)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap_xy.png"))
    plt.close()

    # 8) ベスト解を保存
    np.savetxt(
        os.path.join(OUTPUT_DIR, "best_transform.txt"),
        np.array([[best["x"], best["y"], 0.0, best["yaw"], best["score"]]], dtype=np.float64),
        fmt="%.6f",
        header="x_m, y_m, z_m(=0), yaw_deg, ncc_score"
    )

    print("📂 出力先:", OUTPUT_DIR)
    print(f"🧭 推定: x={best['x']:.2f} m, y={best['y']:.2f} m, z=0.00 m, yaw={best['yaw']:.1f}°, score={best['score']:.3f}")
    print("👉 この (x,y,yaw) をICPの初期値に使ってください。")

if __name__ == "__main__":
    main()
