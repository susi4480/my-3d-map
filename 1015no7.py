# -*- coding: utf-8 -*-
"""
IBGAL (Path-Limited + Yaw-Shift + CuPy NCC)
---------------------------------------------------------
- 探索は path_resampled.json の中心線各点 ±R のグリッドに限定
- Yawは中心線の進行方向から自動推定（逆向きはフラグで反転）
- スキャンは1回だけパノラマ化。Yawは列シフト(np.roll)で高速探索
- NCCは未観測0を除外し、CuPyがあればGPUで並列化
- Cannyエッジ画像でNCC（幾何に頑健）
- 出力: 比較PNG/重ねPNG/俯瞰オーバーレイPNG/best_transform.txt
"""

import os, math, json
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ================== 入出力パス ==================
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = r"/workspace/data/1016_merged_lidar_uesita.ply"
PATH_JSON  = r"/workspace/data/path_resampled.json"
OUTPUT_DIR = r"/workspace/output/ibgal_path_limited_cupy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_INDEX = 1000  # 検証用フレーム番号

# ================== パラメータ ==================
# パノラマ（スキャン/地図レンダ）
YAW_RES_DEG   = 0.25
PITCH_RES_DEG = 0.5
PITCH_MIN_DEG = -22.5
PITCH_MAX_DEG =  22.5
DEPTH_SCALE   = 100.0  # m→任意量子化(=uint16スケール)
MAX_U16 = np.iinfo(np.uint16).max

# 探索範囲（中心線制約）
GRID_STEP_M     = 5.0     # ±R内のグリッド刻み
SEARCH_RADIUS_M = 30.0    # 中心線からの±半径（例: 20〜50mで調整）

# Yaw設定（中心線方向＋ローカル微調整）
REVERSE_PATH_DIRECTION = True   # 中心線が実際と逆向きなら True
LOCAL_YAW_WIN_DEG      = 8      # 進行方向 ± この範囲で微調整
LOCAL_YAW_STEP_DEG     = 1      # 微調整刻み

# NCC（評価）
NCC_MIN_COMMON_PIX   = 2500     # 有効画素下限
FAIL_SCORE_THRESHOLD = 0.24     # これ未満は不一致扱い

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def deg2rad(d): return d * math.pi / 180.0

# =============== GPU自動切替（CuPy） ===============
try:
    import cupy as cp
    _HAS_CUPY = True
    print("🚀 CuPy(GPU) 有効")
except Exception:
    import numpy as cp
    _HAS_CUPY = False
    print("⚠ CuPy なし → CPU")

# =============== 画像化ユーティリティ ===============
def to_polar_image(points, yaw_res_deg, pitch_res_deg, pitch_min_deg, pitch_max_deg, depth_scale):
    """点群→パノラマ深度 (uint16, 0=未観測)"""
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
    rr_q = np.minimum(rr * depth_scale, float(MAX_U16)).astype(np.uint16)

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
    # エッジ閾値は適宜要調整
    return cv2.Canny(g8, 30, 150)

def roll_yaw(img: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Yaw回転 = 列シフト"""
    shift_cols = int(round(yaw_deg / YAW_RES_DEG))
    return np.roll(img, shift_cols, axis=1)

# =============== NCC ===============
def ncc_on_valid_gpu(a, b, min_common):
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common:
        return -1e9
    av = cp.asarray(a[m], dtype=cp.float32)
    bv = cp.asarray(b[m], dtype=cp.float32)
    av -= av.mean(); bv -= bv.mean()
    denom = cp.sqrt((av*av).sum() * (bv*bv).sum()) + 1e-6
    score = (av*bv).sum() / denom
    return float(score.get()) if _HAS_CUPY else float(score)

def ncc(a, b, min_common):  # ラッパ
    return ncc_on_valid_gpu(a, b, min_common)

# =============== 地図レンダ / PCAP抽出 ===============
def pc_to_image_for_viewpoint(map_pts: np.ndarray, viewpoint_xy, z_view=0.0):
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
            pts = xyz.reshape(-1, 3)[valid.reshape(-1)]
            print(f"✅ 抽出成功: frame={i}, 点数={len(pts):,}")
            return pts
    raise ValueError(f"指定フレーム {frame_index} が見つかりません")

# =============== 可視化 ===============
def visualize_all(map_pts, scan_pts, best, outdir):
    ref_img = pc_to_image_for_viewpoint(map_pts, (best["x"], best["y"]), 0.0)
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG, PITCH_MIN_DEG, PITCH_MAX_DEG, DEPTH_SCALE)
    scan_rot = roll_yaw(scan_img, best["yaw"])
    ref_norm  = cv2.normalize(ref_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scan_norm = cv2.normalize(scan_rot, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 1) 左右比較
    side = np.hstack([ref_norm, scan_norm])
    side_rgb = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
    cv2.putText(side_rgb, "Map render (left) | LiDAR render (right)", (20,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir, "compare_side_by_side.png"), side_rgb)

    # 2) カラー重ね（Map=G, LiDAR=R）
    h,w = ref_norm.shape
    overlay = np.zeros((h,w,3), np.uint8); overlay[...,1]=ref_norm; overlay[...,2]=scan_norm
    cv2.putText(overlay, "Overlay: Map=Green, LiDAR=Red (gray=match)", (20,35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir, "compare_overlay_color.png"), overlay)

    # 3) 俯瞰オーバーレイ（推定姿勢でスキャンを配置）
    yaw = deg2rad(best["yaw"])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0,0,1]])
    scan_tf = (scan_pts @ Rz.T) + np.array([best["x"], best["y"], 0.0])

    plt.figure(figsize=(8,8))
    plt.scatter(map_pts[:,0], map_pts[:,1], s=0.2, c='gray', alpha=0.5, label="Map")
    plt.scatter(scan_tf[:,0], scan_tf[:,1], s=0.5, c='red',  alpha=0.6, label="Scan(placed)")
    plt.axis("equal"); plt.legend()
    plt.title(f"Top-Down Overlay (score={best['score']:.3f})")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_topdown.png"), dpi=300)
    plt.close()

# =============== メイン ===============
def main():
    ensure_dir(OUTPUT_DIR)

    # 地図
    print("🗺 地図読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    map_pts = np.asarray(map_pcd.points).astype(np.float64)
    if map_pts.size == 0:
        raise RuntimeError("地図点群が空です")
    print(f"✅ 地図点数: {len(map_pts):,}")

    # path
    print("📄 path_resampled.json 読み込み中...")
    with open(PATH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    path = np.array(data["path"], dtype=np.float64)
    if len(path) < 2:
        raise RuntimeError("pathの点が不足しています（少なくとも2点必要）")
    print(f"✅ 中心線点数: {len(path)}")

    # スキャン
    print("📡 LiDARフレーム抽出中...")
    scan_pts = extract_frame_points_from_pcap(PCAP_PATH, JSON_PATH, FRAME_INDEX)
    if scan_pts.size == 0:
        raise RuntimeError("スキャン点群が空です")

    # スキャン画像（1回だけ生成）
    print("🖼 スキャン画像生成...")
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG, PITCH_MIN_DEG, PITCH_MAX_DEG, DEPTH_SCALE)
    scan_edge = to_edges(scan_img)

    # ローカルYawスイープ候補を前計算（±LOCAL_YAW_WIN_DEG）
    local_offsets = list(range(-LOCAL_YAW_WIN_DEG, LOCAL_YAW_WIN_DEG + 1, LOCAL_YAW_STEP_DEG))
    # 進行方向に依存しない「相対Yawシフト画像」をキャッシュ（-180〜180は不要、ローカルだけでOK）
    rolled_cache = {off: roll_yaw(scan_edge, off) for off in local_offsets}

    # 探索（path各点の±Rを格子サンプリング。Yawは進行方向±ローカル）
    print("🔍 探索開始（中心線制約）...")
    best = dict(score=-1e9, x=None, y=None, yaw=None)
    # 各path点での進行方向Yawを事前に用意
    path_yaws = []
    for i in range(len(path)-1):
        x0,y0 = path[i]
        x1,y1 = path[i+1]
        yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if REVERSE_PATH_DIRECTION:
            yaw = (yaw + 180.0) % 360.0
        path_yaws.append(yaw)
    path_yaws.append(path_yaws[-1])  # 最終点は前のYawを流用

    # 候補座標を生成して走査
    tried = 0
    for i, (px, py) in enumerate(path):
        base_yaw = path_yaws[i]
        # ±Rの格子（px,py）を中心に GRID_STEP_M でサンプリング
        xs = np.arange(px - SEARCH_RADIUS_M, px + SEARCH_RADIUS_M + 1e-9, GRID_STEP_M)
        ys = np.arange(py - SEARCH_RADIUS_M, py + SEARCH_RADIUS_M + 1e-9, GRID_STEP_M)

        for y0 in ys:
            for x0 in xs:
                ref_img = pc_to_image_for_viewpoint(map_pts, (x0, y0), 0.0)
                ref_edge = to_edges(ref_img)
                # ローカルYaw微調整
                local_best, local_best_yaw = -1e9, base_yaw
                for off in local_offsets:
                    yd = base_yaw + off
                    # 「スキャン画像をydだけ回す」≒「参照側を -yd 回す」だが
                    # 本実装はスキャン側を回す（キャッシュ済み）
                    rolled = rolled_cache[off]
                    s = ncc(ref_edge, rolled, NCC_MIN_COMMON_PIX)
                    if s > local_best:
                        local_best, local_best_yaw = s, yd
                tried += 1
                if local_best > best["score"]:
                    best.update(score=local_best, x=float(x0), y=float(y0), yaw=float(local_best_yaw))

    if best["x"] is None:
        raise RuntimeError("一致が見つかりませんでした（scoreが全て無効）")

    print(f"✅ 最良スコア: score={best['score']:.3f}, x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f}°")
    if best["score"] < FAIL_SCORE_THRESHOLD:
        print("⚠ スコアが閾値未満です。パラメータ（SEARCH_RADIUS_M / NCC_MIN_COMMON_PIX / LOCAL_YAW_WIN）等を調整してください。")

    # 可視化 & 保存
    visualize_all(map_pts, scan_pts, best, OUTPUT_DIR)
    np.savetxt(os.path.join(OUTPUT_DIR, "best_transform.txt"),
               np.array([[best["x"], best["y"], 0.0, best["yaw"], best["score"]]], dtype=np.float64),
               fmt="%.6f",
               header="x_m, y_m, z_m(=0), yaw_deg, ncc_score")

    print("📸 出力完了:", OUTPUT_DIR)
    print(f"🧭 推定: x={best['x']:.2f} m, y={best['y']:.2f} m, yaw={best['yaw']:.1f}°, score={best['score']:.3f}")
    print(f"🧪 試行数: {tried:,}")

if __name__ == "__main__":
    main()
