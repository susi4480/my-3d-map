# -*- coding: utf-8 -*-
"""
【機能】(統合版) ②の4隅定義 × ③の最緩連結ロジック / PLYポリライン出力
-----------------------------------------------------------------------
- 各スライス矩形LASから外周点群を読み込み
- PCAで幅方向(=v軸)を推定し、左右端帯域から Zmin/Zmax で 4隅を抽出
  → [LL=left_low, LU=left_high, RL=right_low, RU=right_high]
- 初期接続：各隅について i → i+1 を接続（水色ファイル）
- 最緩接続：角度が閾値以上なら i → i+LOOKAHEAD で最も“緩い”相手を探索
   ※ タイブレーク：平均角度 → スライス距離(j-i) → XY距離合計
   ※ 採用時は i+1..(best_j-1) をスキップ（飛び越え）
- 出力：PLY(ASCII)のポリライン2種
   (A) 初期接続 only:  /workspace/output/bridges_initial.ply
   (B) 最緩接続 only:  /workspace/output/bridges_relaxed.ply
-----------------------------------------------------------------------
依存:
    pip install laspy numpy
入力:
    /workspace/output/917slices_m0style_rect/slice_????_rect.las
"""

import os
import re
import numpy as np
import laspy
from glob import glob

# ===== 入出力 =====
INPUT_DIR = "/workspace/output/917slices_m0style_rect"
OUTPUT_PLY_INITIAL = "/workspace/output/bridges_initial.ply"   # 初期(i→i+1)
OUTPUT_PLY_RELAXED = "/workspace/output/bridges_relaxed.ply"   # 最緩(i→best_j)

# ===== パラメータ =====
ANGLE_THRESH_DEG = 35.0
LOOKAHEAD_SLICES = 30
EDGE_ORDER = ("angle", "slice_gap", "xy_dist_sum")  # 参考

# ===== 4隅抽出 (②方式: PCA→左右端帯域→Zmin/Zmax) =====
def get_extreme_points_pca(pts_xyz):
    """
    PCAで幅方向を推定し、左右端の帯域からZmin/Zmaxで4点抽出。
    返り値: [LL, LU, RL, RU] 各 shape=(3,)
    """
    if len(pts_xyz) < 4:
        return None

    xy = pts_xyz[:, :2]
    mu = xy.mean(axis=0)
    A = xy - mu
    C = A.T @ A / max(1, len(A) - 1)
    w, V = np.linalg.eigh(C)
    axis = V[:, np.argmax(w)]  # 幅方向(v軸)
    vcoord = A @ axis
    vmin, vmax = vcoord.min(), vcoord.max()
    if vmax - vmin < 1e-9:
        return None

    # 端帯域（全幅の2% or 5cm）
    band = max(0.02 * (vmax - vmin), 0.05)
    left_pts = pts_xyz[vcoord <= vmin + band]
    right_pts = pts_xyz[vcoord >= vmax - band]
    if len(left_pts) == 0 or len(right_pts) == 0:
        return None

    left_low   = left_pts[np.argmin(left_pts[:, 2])]
    left_high  = left_pts[np.argmax(left_pts[:, 2])]
    right_low  = right_pts[np.argmin(right_pts[:, 2])]
    right_high = right_pts[np.argmax(right_pts[:, 2])]
    return [left_low, left_high, right_low, right_high]

# ===== 角度(ターン)計算 =====
def angle_turn_deg(p_prev, p_curr, p_next):
    """XY平面で、ベクトル(p_curr→p_prev) と (p_curr→p_next) の180°からの乖離角"""
    a = np.asarray(p_prev[:2]) - np.asarray(p_curr[:2])
    b = np.asarray(p_next[:2]) - np.asarray(p_curr[:2])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cosv = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    inner = np.degrees(np.arccos(cosv))
    return abs(inner - 180.0)  # 小さいほど直進

# ===== PLY出力（ポリライン） =====
def write_ply_lines(path, vertices, edges):
    """
    頂点とエッジを持つPLY(ASCII)出力（色なし）
    vertices: (N,3) float
    edges   : List[Tuple[int,int]]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")
    print(f"✅ PLY出力: {path} | 頂点:{len(vertices)} 辺:{len(edges)}")

# ===== メイン =====
def main():
    # 1) スライス読み込み & 4隅抽出（②方式）
    slice_files = sorted(
        glob(os.path.join(INPUT_DIR, "slice_*_rect.las")),
        key=lambda f: int(re.search(r"slice_(\d+)_rect\.las", os.path.basename(f)).group(1))
    )
    if not slice_files:
        raise RuntimeError(f"入力がありません: {INPUT_DIR}/slice_*_rect.las")

    extremes_per_slice = []  # list of [4x(3,)] or None
    for f in slice_files:
        las = laspy.read(f)
        pts = np.column_stack([las.x, las.y, las.z])
        extremes = get_extreme_points_pca(pts)
        extremes_per_slice.append(extremes)

    # PLYの頂点配列（各スライスの4隅を順に追加）
    ply_vertices = []
    ply_indices = []  # 各スライスの4隅の頂点index [LL,LU,RL,RU] or None
    for ext in extremes_per_slice:
        if ext is None:
            ply_indices.append(None)
        else:
            base = len(ply_vertices)
            ply_vertices.extend(ext)  # 4点
            ply_indices.append([base + 0, base + 1, base + 2, base + 3])

    # 2) 初期接続（i→i+1）エッジ（各隅ごと）
    init_edges = []
    for i in range(len(ply_indices) - 1):
        idx_a = ply_indices[i]
        idx_b = ply_indices[i + 1]
        if idx_a is None or idx_b is None:
            continue
        # 4本：LL, LU, RL, RU
        for c in range(4):
            init_edges.append((idx_a[c], idx_b[c]))

    # 3) 最緩接続の探索（③ロジック）
    # series[c][i] … c隅のiスライスにおける3D座標
    series = {}
    valid_slices = []
    for c in range(4):
        seq = []
        for ext in extremes_per_slice:
            if ext is None:
                seq.append(None)
            else:
                seq.append(np.asarray(ext[c], float))
        series[c] = seq
    for i, ext in enumerate(extremes_per_slice):
        if ext is not None:
            valid_slices.append(i)

    # 接続先（初期は i→i+1、スキップは per-slice で管理）
    N = len(extremes_per_slice)
    connect_to = np.array([min(i + 1, N - 1) for i in range(N)], dtype=int)
    disabled = np.zeros(N, dtype=bool)  # 最緩採用で i+1..best_j-1 を飛ばす

    for i in range(1, N - 1):
        if disabled[i]:
            continue
        # いずれかの隅が None ならスキップ
        if any(series[c][i] is None for c in range(4)):
            continue
        if any(series[c][i - 1] is None for c in range(4)):
            continue
        if any(series[c][i + 1] is None for c in range(4)):
            continue

        # 角度チェック：4隅平均で閾値超えなら再結合を検討
        angs_now = []
        for c in range(4):
            angs_now.append(angle_turn_deg(series[c][i - 1], series[c][i], series[c][i + 1]))
        if float(np.mean(angs_now)) < ANGLE_THRESH_DEG:
            continue

        last = min(N - 1, i + LOOKAHEAD_SLICES)
        best_j, best_score = i + 1, (1e18, 1e18, 1e18)  # (mean_angle, slice_gap, xy_dist_sum)

        for j in range(i + 2, last + 1):
            if any(series[c][j] is None for c in range(4)):
                continue
            angs, xy_sum = [], 0.0
            for c in range(4):
                angs.append(angle_turn_deg(series[c][i - 1], series[c][i], series[c][j]))
                xy_sum += np.linalg.norm(series[c][j][:2] - series[c][i][:2])
            cand = (float(np.mean(angs)), j - i, float(xy_sum))
            if cand < best_score:
                best_score, best_j = cand, j

        if best_j != i + 1:
            connect_to[i] = best_j
            if best_j - (i + 1) > 0:
                disabled[i + 1:best_j] = True

    # 4) 最緩接続エッジを作成（i→connect_to[i]、disabled[i]は出さない）
    relaxed_edges = []
    for i in range(N - 1):
        if disabled[i]:
            continue
        j = int(connect_to[i])
        if j <= i or j >= N:
            continue
        idx_i = ply_indices[i]
        idx_j = ply_indices[j]
        if idx_i is None or idx_j is None:
            continue
        for c in range(4):
            relaxed_edges.append((idx_i[c], idx_j[c]))

    # 5) PLY出力（初期・最緩を別ファイル）
    vertices = np.asarray(ply_vertices, float)
    write_ply_lines(OUTPUT_PLY_INITIAL, vertices, init_edges)
    write_ply_lines(OUTPUT_PLY_RELAXED, vertices, relaxed_edges)

    print("🏁 完了")
    print(f"  初期接続 edges:  {len(init_edges)}")
    print(f"  最緩接続 edges:  {len(relaxed_edges)}")
    print(f"  4隅抽出(有効スライス): {sum(1 for x in ply_indices if x is not None)} / {len(ply_indices)}")

if __name__ == "__main__":
    main()
