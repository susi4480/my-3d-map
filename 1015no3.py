# -*- coding: utf-8 -*-
"""
IBGAL v2（Scan-to-Map, z=0固定, 地図Z反転維持, FOV制限, Edge-NCC）
-----------------------------------------------------------------------
入力:
  - MAP_PATH  : 既存地図 .ply
  - PCAP/JSON : Ouster OS-2 の録画（1フレーム抽出）

手順:
  1) 地図PLY読み込み → Z軸反転（map_pts[:,2]*=-1）
  2) PCAP/JSONから指定FRAME_INDEXのスキャン点群抽出（range>0）
  3) 点群をパノラマ化（Yaw×Pitch）。Pitchは実機FOV（±22.5°）に制限
     - 深度は cm量子化（m×100）で uint16 に安全クリップ
  4) 粗探索→細探索（XY×Yaw）。各候補で:
     - 地図側を仮想視点 (x,y,z=0) からレンダ
     - スキャン側はyaw分だけ列シフト
     - Edge-NCC（Cannyエッジ後のNCC, 共通画素閾値あり）でスコア評価
  5) ベスト (x,y,yaw) を可視化

出力（この4枚のみ）:
  - compare_side_by_side.png
  - compare_overlay_color.png
  - estimated_pose_on_map.png
  - alignment_topdown.png
"""

import os
import math
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ========= パス =========
PCAP_PATH  = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = "/workspace/data/1013_lidar_map.ply"
OUTPUT_DIR = "/workspace/output/ibgal_xyyaw_compare"
FRAME_INDEX = 5000

# ========= パノラマ設定 =========
YAW_RES_DEG    = 0.25         # 列解像度（小さいほど高解像度）
PITCH_RES_DEG  = 0.5          # 行解像度
PITCH_MIN_DEG  = -22.5        # 実機FOV（OS-2目安）
PITCH_MAX_DEG  =  22.5
DEPTH_SCALE    = 100.0        # m→cm 量子化 (×100)
MAX_U16        = np.iinfo(np.uint16).max

# ========= 探索設定（粗→細） =========
# 自動中心: 地図の最密セル中心を使う（下の関数で推定）
SEARCH_XY_RADIUS_COARSE = 40.0
SEARCH_XY_STEP_COARSE   = 5.0
YAW_STEP_DEG_COARSE     = 5

SEARCH_XY_RADIUS_FINE   = 10.0
SEARCH_XY_STEP_FINE     = 1.0
YAW_STEP_DEG_FINE       = 1

# ========= スコアリング =========
NCC_MIN_COMMON_PIX      = 3000   # 共通有効画素しきい
FAIL_SCORE_THRESHOLD    = 0.25   # これ未満は失敗扱い

# ========= その他 =========
MAP_RANDOM_DOWNSAMPLE   = 1.0    # 1.0で無効（必要なら 0.5 等）
AUTO_CENTER_BY_DENSITY  = True
COARSE_BIN_M            = 10.0   # 最密セル推定の粗ビン

# ------------------------------------------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def deg2rad(d):   return d * math.pi / 180.0

def to_polar_image(points: np.ndarray,
                   yaw_res_deg: float,
                   pitch_res_deg: float,
                   pitch_min_deg: float,
                   pitch_max_deg: float,
                   depth_scale: float) -> np.ndarray:
    """点群→パノラマ深度画像 (uint16, 単位=1/depth_scale m, 0=未観測)。"""
    yaw_bins   = int(round(360.0 / yaw_res_deg))
    pitch_bins = int(round((pitch_max_deg - pitch_min_deg) / pitch_res_deg))
    if points.size == 0:
        return np.zeros((pitch_bins, yaw_bins), np.uint16)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    yaw   = np.arctan2(y, x)                              # [-pi, pi)
    pitch = np.arctan2(z, np.sqrt(x**2 + y**2))           # [-pi/2, pi/2]

    pmin = deg2rad(pitch_min_deg)
    pmax = deg2rad(pitch_max_deg)
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

def to_edges(img_u16: np.ndarray) -> np.ndarray:
    """Cannyエッジ（0/255）。深度画像から形状境界だけを抽出してNCCを安定化。"""
    if img_u16.size == 0:
        return img_u16
    g = cv2.GaussianBlur(img_u16, (5, 5), 0)
    # Cannyは8bit想定。正規化して使う
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    e = cv2.Canny(g8, 30, 150)
    return e

def ncc_on_valid(a: np.ndarray, b: np.ndarray, min_common: int) -> float:
    """未観測(0)を除外し、共通画素のみでNCC。閾値未満なら極小値。"""
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common:
        return -1e9
    av = a[m].astype(np.float32); bv = b[m].astype(np.float32)
    av -= av.mean(); bv -= bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum()) + 1e-6
    return float((av * bv).sum() / denom)

def roll_yaw(img: np.ndarray, yaw_deg: float, yaw_res_deg: float) -> np.ndarray:
    """Yaw回転は列シフトに等価。"""
    shift_cols = int(round(yaw_deg / yaw_res_deg))
    return np.roll(img, shift_cols, axis=1)

def pc_to_image_for_viewpoint(map_pts: np.ndarray,
                              viewpoint_xy: Tuple[float, float],
                              z_view: float = 0.0) -> np.ndarray:
    """視点 (x,y,z=z_view) から見た地図のパノラマ深度画像。"""
    vx, vy = viewpoint_xy
    pts = map_pts - np.array([vx, vy, z_view], dtype=np.float64)
    return to_polar_image(pts, YAW_RES_DEG, PITCH_RES_DEG,
                          PITCH_MIN_DEG, PITCH_MAX_DEG,
                          DEPTH_SCALE)

def extract_frame_points_from_pcap(pcap_path: str, json_path: str, frame_index: int) -> np.ndarray:
    """Ouster PCAP/JSONから指定フレームの点群を抽出（range>0）。"""
    with open(json_path, "r") as f:
        sensor_info = SensorInfo(f.read())
    xyzlut = XYZLut(sensor_info, use_extrinsics=False)
    source = open_source(pcap_path)

    for i, scan in enumerate(source):
        if i == frame_index:
            xyz = xyzlut(scan)                             # (H, W, 3)
            rng = scan.field(ChanField.RANGE)              # (H, W)
            valid = (rng > 0)
            pts = xyz.reshape(-1, 3)[valid.reshape(-1)]
            return pts
    raise ValueError(f"指定フレーム {frame_index} が見つかりません")

def auto_center_by_density(map_pts: np.ndarray, bin_m: float) -> Tuple[float, float]:
    """地図XYを粗ヒストして最密セル中心を返す。探索起点の安定化。"""
    x_min, x_max = map_pts[:, 0].min(), map_pts[:, 0].max()
    y_min, y_max = map_pts[:, 1].min(), map_pts[:, 1].max()
    nx = max(1, int(np.ceil((x_max - x_min) / bin_m)))
    ny = max(1, int(np.ceil((y_max - y_min) / bin_m)))
    H, xedges, yedges = np.histogram2d(map_pts[:, 0], map_pts[:, 1],
                                       bins=[nx, ny],
                                       range=[[x_min, x_max], [y_min, y_max]])
    iy, ix = np.unravel_index(np.argmax(H), H.shape)
    cx = 0.5 * (xedges[iy] + xedges[iy + 1])
    cy = 0.5 * (yedges[ix] + yedges[ix + 1])
    return float(cx), float(cy)

def search_xy_yaw(map_pts, scan_edge, cx, cy,
                  xy_radius, xy_step, yaw_step_deg):
    """XY×Yaw の探索（Edge-NCC）。scan_edge はすでにエッジ画像。"""
    xs = np.arange(cx - xy_radius, cx + xy_radius + 1e-9, xy_step)
    ys = np.arange(cy - xy_radius, cy + xy_radius + 1e-9, xy_step)
    yaw_candidates = list(range(-180, 181, yaw_step_deg))

    # スキャン側の回転候補エッジをキャッシュ
    rolled_edges = {yd: roll_yaw(scan_edge, yd, YAW_RES_DEG) for yd in yaw_candidates}

    best = dict(score=-1e9, x=None, y=None, yaw=None)
    for y0 in ys:
        for x0 in xs:
            ref_img = pc_to_image_for_viewpoint(map_pts, (x0, y0), z_view=0.0)
            ref_edge = to_edges(ref_img)
            for yd in yaw_candidates:
                s = ncc_on_valid(ref_edge, rolled_edges[yd], NCC_MIN_COMMON_PIX)
                if s > best["score"]:
                    best.update(score=s, x=x0, y=y0, yaw=yd)
    return best

def visualize_outputs_only_four(map_pts, scan_pts, best, outdir):
    """指定の4枚だけを保存。"""
    # 1) 左右比較 / 2) オーバーレイ (エッジではなく深度画像の可視化を使用)
    ref_img = pc_to_image_for_viewpoint(map_pts, (best["x"], best["y"]), z_view=0.0)
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG,
                              PITCH_MIN_DEG, PITCH_MAX_DEG, DEPTH_SCALE)
    scan_rot = roll_yaw(scan_img, best["yaw"], YAW_RES_DEG)

    ref_norm  = cv2.normalize(ref_img,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scan_norm = cv2.normalize(scan_rot, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # (A) compare_side_by_side.png
    side = np.hstack([ref_norm, scan_norm])
    side_rgb = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)
    cv2.putText(side_rgb, "Map render (left)  |  LiDAR render (right)",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir, "compare_side_by_side.png"), side_rgb)

    # (B) compare_overlay_color.png
    h, w = ref_norm.shape
    overlay = np.zeros((h, w, 3), np.uint8)
    overlay[..., 1] = ref_norm     # G: Map
    overlay[..., 2] = scan_norm    # R: Scan
    cv2.putText(overlay, "Overlay: Map=Green, LiDAR=Red (gray=match)",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir, "compare_overlay_color.png"), overlay)

    # (C) estimated_pose_on_map.png
    plt.figure(figsize=(8, 8))
    plt.scatter(map_pts[:, 0], map_pts[:, 1], s=0.2, c='gray', alpha=0.5)
    plt.arrow(best["x"], best["y"],
              5 * math.cos(deg2rad(best["yaw"])),
              5 * math.sin(deg2rad(best["yaw"])),
              color='red', head_width=1.0, length_includes_head=True)
    plt.title(f"Estimated Pose on Map (score={best['score']:.3f})")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "estimated_pose_on_map.png"), dpi=300)
    plt.close()

    # (D) alignment_topdown.png（推定姿勢でスキャンをXYへ重ね）
    yaw = deg2rad(best["yaw"])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0, 0, 1]])
    scan_tf = (scan_pts @ Rz.T) + np.array([best["x"], best["y"], 0.0])
    plt.figure(figsize=(8, 8))
    plt.scatter(map_pts[:, 0], map_pts[:, 1], s=0.2, c='gray', alpha=0.5, label="Map")
    plt.scatter(scan_tf[:, 0], scan_tf[:, 1], s=0.5, c='red',  alpha=0.6, label="Scan")
    plt.axis("equal"); plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.title("Map vs Scan Alignment (Top-Down)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "alignment_topdown.png"), dpi=300)
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)

    # 1) 地図読み込み
    if not os.path.exists(MAP_PATH):
        raise FileNotFoundError(f"地図がありません: {MAP_PATH}")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    if MAP_RANDOM_DOWNSAMPLE < 1.0:
        map_pcd = map_pcd.random_down_sample(MAP_RANDOM_DOWNSAMPLE)
    map_pts = np.asarray(map_pcd.points)
    if map_pts.size == 0:
        raise ValueError("地図点群が空です。PLYを確認してください。")
    # 地図Z軸反転（要求通り維持）
    map_pts[:, 2] *= -1

    # 2) スキャン抽出
    scan_pts = extract_frame_points_from_pcap(PCAP_PATH, JSON_PATH, FRAME_INDEX)
    if scan_pts.size == 0:
        raise ValueError("スキャン点群が空です。PCAP/JSONとFRAME_INDEXを確認してください。")

    # 3) スキャンを画像化 → エッジ化（固定一回）
    scan_img = to_polar_image(scan_pts, YAW_RES_DEG, PITCH_RES_DEG,
                              PITCH_MIN_DEG, PITCH_MAX_DEG, DEPTH_SCALE)
    scan_edge = to_edges(scan_img)

    # 4) 探索中心（自動）
    if AUTO_CENTER_BY_DENSITY:
        cx, cy = auto_center_by_density(map_pts, COARSE_BIN_M)
    else:
        cx, cy, _ = map_pts.mean(axis=0)

    # 5) 粗探索
    best = search_xy_yaw(map_pts, scan_edge, cx, cy,
                         SEARCH_XY_RADIUS_COARSE, SEARCH_XY_STEP_COARSE, YAW_STEP_DEG_COARSE)

    # 6) 細探索（粗ベスト近傍）
    best2 = search_xy_yaw(map_pts, scan_edge, best["x"], best["y"],
                          SEARCH_XY_RADIUS_FINE, SEARCH_XY_STEP_FINE, YAW_STEP_DEG_FINE)
    if best2["score"] > best["score"]:
        best = best2

    # 7) 失敗判定と可視化（4枚のみ）
    if best["score"] < FAIL_SCORE_THRESHOLD:
        print(f"❌ マッチ失敗: score={best['score']:.3f} < {FAIL_SCORE_THRESHOLD}")
        # 失敗でも比較画像は欲しい場合は以下をコメントアウト解除
        # visualize_outputs_only_four(map_pts, scan_pts, best, OUTPUT_DIR)
        return

    visualize_outputs_only_four(map_pts, scan_pts, best, OUTPUT_DIR)
    print("✅ 出力:", ", ".join([
        "compare_side_by_side.png",
        "compare_overlay_color.png",
        "estimated_pose_on_map.png",
        "alignment_topdown.png"
    ]))
    print(f"🧭 推定: x={best['x']:.2f}, y={best['y']:.2f}, yaw={best['yaw']:.1f}°, score={best['score']:.3f}")

if __name__ == "__main__":
    main()
