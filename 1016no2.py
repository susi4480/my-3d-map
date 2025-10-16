# -*- coding: utf-8 -*-
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# === ファイルパス ===
PCAP_PATH = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH = "/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
PATH_JSON = "/workspace/data/path_resampled.json"

# === 仮想スキャン生成 ===
def generate_virtual_scans_along_path(path_list, resolution_deg=1.0, max_range=100.0):
    num_bins = int(360 / resolution_deg)
    scans = []
    for x0, y0 in path_list:
        angles = np.linspace(-math.pi, math.pi, num_bins, endpoint=False)
        distances = np.full_like(angles, fill_value=max_range)
        scans.append((angles, distances))
    return scans

# === 実スキャン読み込み ===
def load_ouster_scan(pcap_path, json_path, frame_index=0):
    with open(json_path, "r") as f:
        sensor_info = SensorInfo(f.read())
    xyzlut = XYZLut(sensor_info, use_extrinsics=False)
    source = open_source(pcap_path)
    for i, scans in enumerate(source):
        scan = scans if not isinstance(scans, list) else scans[0]
        if i == frame_index:
            xyz = xyzlut(scan)
            rng = scan.field(ChanField.RANGE)
            valid = rng > 0
            pts = xyz.reshape(-1, 3)[valid.reshape(-1)]
            return pts
    raise ValueError(f"フレーム {frame_index} が見つかりません")

# === 実スキャンを距離ベクトルに変換 ===
def lidar_scan_to_distance_vector(points, resolution_deg=1.0, max_range=100.0):
    x, y = points[:, 0], points[:, 1]
    angles = np.arctan2(y, x)
    dists = np.sqrt(x**2 + y**2)
    num_bins = int(360 / resolution_deg)
    bin_edges = np.linspace(-math.pi, math.pi, num_bins + 1)
    indices = np.digitize(angles, bin_edges) - 1
    distances = np.full(num_bins, fill_value=max_range)
    for i in range(num_bins):
        mask = indices == i
        if np.any(mask):
            distances[i] = dists[mask].min()
    angles_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    return angles_center, distances

# === 距離ベクトル間のマッチング（L2距離） ===
def match_scan_to_map(scan_vector, virtual_scans):
    scan_angles, scan_distances = scan_vector
    best_index, best_score = -1, float("-inf")
    for i, (v_angles, v_distances) in enumerate(virtual_scans):
        if len(scan_distances) != len(v_distances):
            continue
        score = -np.linalg.norm(scan_distances - v_distances)
        if score > best_score:
            best_score = score
            best_index = i
    return best_index, best_score

# === メイン ===
def main():
    print("[1] path.json 読み込み中...")
    with open(PATH_JSON, "r", encoding="utf-8") as f:
        path_data = json.load(f)
    path_list = path_data["path"]

    print("[2] 仮想スキャン事前生成中...")
    virtual_scans = generate_virtual_scans_along_path(path_list)
    print(f"[3] 仮想スキャン {len(virtual_scans)} 点 生成")

    print("[4] LiDARスキャン読み込み中 (1フレームのみ)")
    scan_pts = load_ouster_scan(PCAP_PATH, JSON_PATH, frame_index=5000)

    # Z軸反転（地図が上下反転している場合に対応）
    scan_pts[:, 2] *= -1

    print("[5] フレーム 0 を処理中...")
    angles, distances = lidar_scan_to_distance_vector(scan_pts)
    best_index, similarity = match_scan_to_map((angles, distances), virtual_scans)

    print(f"    → 最も一致した中心線点 index={best_index}, 類似度={similarity:.3f}")
    plt.plot(angles, distances, label="実スキャン")
    plt.plot(*virtual_scans[best_index], label="仮想スキャン")
    plt.title("スキャンマッチング結果")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
