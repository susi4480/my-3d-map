# -*- coding: utf-8 -*-
"""
【機能】LAS点群から川の中心線(X,Y)を自動推定してCSV出力（垂直スライスで1回リファイン）
- Z ≤ Z_LIMIT の点のみ使用（橋・上部ノイズを除去）
- PCAで主方向を推定 → 座標回転（川軸 ≈ x'）
- x' を BIN_SIZE ごとに分け、各ビンの y' の中央値で粗中心線を作成（任意で移動平均）
- 粗中心線の各点で接線ベクトルを計算し、その法線方向に細い帯を切り出して
  法線方向の中央値で中心点を1回リファイン（＝“中心線に垂直”なスライスで再推定）
- 逆回転で元座標に戻して X,Y をCSV出力（ヘッダなし）
"""

import numpy as np
import laspy
import os

# === 入出力 ===
INPUT_LAS  = r"C:\Users\user\Documents\lab\outcome\0731_suidoubasi_ue.las"
OUTPUT_CSV = r"C:\Users\user\Documents\lab\data\centerline_xy.csv"  # ヘッダなし: X,Y

# === パラメータ ===
Z_LIMIT            = 1.5     # [m] これ以下のみ使用
BIN_SIZE           = 2.0     # [m] x'方向のビン幅（粗中心線用）
MIN_PTS_PER_BIN    = 50      # 最低点数
SMOOTH_WINDOW_M    = 10.0    # [m] 粗中心線の移動平均窓（0で無効）

# 局所“垂直”スライスによるリファイン用
CROSS_HALF_WIDTH_M = 20.0    # [m] 法線方向の半幅（±この距離で帯を切り出す）
LONG_HALF_LEN_M    = 5.0     # [m] 接線方向の半長（±この距離で帯の長さを制限）
MIN_PTS_PER_STRIP  = 40      # 帯内の最低点数（少なければ粗中心を採用）
VERBOSE            = True

# === ユーティリティ ===
def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def pca_main_angle_xy(xy):
    xy0 = xy - xy.mean(axis=0, keepdims=True)
    C = np.cov(xy0, rowvar=False)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]         # 最大固有値の固有ベクトル
    angle = np.arctan2(v[1], v[0])       # X軸に対する角度
    return angle

def moving_average_1d(arr, win):
    if win <= 1 or len(arr) < 2:
        return arr
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(arr_pad, kernel, mode='valid')

def compute_tangents(xs, ys):
    """
    粗中心線(x_i, y_i)から接線ベクトル t_i を計算（中央差分・端は片側差分）。
    返り値: (N,2) の単位接線ベクトル
    """
    N = len(xs)
    t = np.zeros((N, 2), dtype=float)
    for i in range(N):
        if i == 0:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
        elif i == N - 1:
            dx = xs[N-1] - xs[N-2]
            dy = ys[N-1] - ys[N-2]
        else:
            dx = xs[i+1] - xs[i-1]
            dy = ys[i+1] - ys[i-1]
        v = np.array([dx, dy], dtype=float)
        n = np.linalg.norm(v) + 1e-12
        t[i] = v / n
    return t

# === メイン ===
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 読み込み＆Z制限
    las = laspy.read(INPUT_LAS)
    pts = np.vstack([las.x, las.y, las.z]).T
    pts = pts[pts[:,2] <= Z_LIMIT]
    if len(pts) == 0:
        raise RuntimeError("Z制限後の点がありません。Z_LIMITを見直してください。")

    # PCAで主方向を推定 → 回転（元→回転後）
    theta = pca_main_angle_xy(pts[:, :2])
    R     = rotation_matrix_z(theta)
    R_inv = rotation_matrix_z(-theta)
    pts_p = (R @ pts.T).T  # (x',y',z)

    # --- 粗中心線（x'ビンで y'中央値） ---
    x_min, x_max = pts_p[:,0].min(), pts_p[:,0].max()
    edges = np.arange(x_min, x_max + BIN_SIZE, BIN_SIZE)
    centers_xp = 0.5 * (edges[:-1] + edges[1:])

    xp_list, yp_list = [], []
    for i in range(len(edges) - 1):
        x0, x1 = edges[i], edges[i+1]
        mask = (pts_p[:,0] >= x0) & (pts_p[:,0] < x1)
        if not np.any(mask): 
            continue
        ys = pts_p[mask, 1]
        if ys.size < MIN_PTS_PER_BIN:
            continue
        xp_list.append(centers_xp[i])
        yp_list.append(np.median(ys))

    if len(xp_list) < 2:
        raise RuntimeError("有効なビンが足りません。BIN_SIZEやMIN_PTS_PER_BINを見直してください。")

    xp = np.array(xp_list)
    yp = np.array(yp_list)

    # 移動平均で平滑化（任意）
    if SMOOTH_WINDOW_M > 0:
        win = max(1, int(round(SMOOTH_WINDOW_M / BIN_SIZE)))
        if win % 2 == 0:
            win += 1
        yp = moving_average_1d(yp, win)

    # --- 局所“垂直”スライスで1回リファイン ---
    # 接線ベクトル（x'y'平面上）
    tangents = compute_tangents(xp, yp)

    # 点群（x'y'のみ）を準備
    XYp = pts_p[:, :2]

    # リファイン結果（x′, y′）
    xp_ref = xp.copy()
    yp_ref = yp.copy()

    for i in range(len(xp)):
        c = np.array([xp[i], yp[i]], dtype=float)  # 粗中心
        t = tangents[i]                            # 単位接線
        n = np.array([-t[1], t[0]])               # 単位法線（左90°）

        # 各点 p について、局所座標 (s, r) = (接線方向, 法線方向) を取る
        # s = (p - c)・t, r = (p - c)・n
        d = XYp - c
        s = d @ t
        r = d @ n

        # 細い帯：|s| <= LONG_HALF_LEN_M かつ |r| <= CROSS_HALF_WIDTH_M
        m = (np.abs(s) <= LONG_HALF_LEN_M) & (np.abs(r) <= CROSS_HALF_WIDTH_M)
        if np.count_nonzero(m) >= MIN_PTS_PER_STRIP:
            r_med = np.median(r[m])   # 法線方向の中央値
            # 中心を法線方向に r_med だけ移動
            c_ref = c + r_med * n
            xp_ref[i], yp_ref[i] = c_ref[0], c_ref[1]
        # それ未満なら、粗中心そのまま（データが薄い領域の保護）

    # --- 逆回転で元座標へ ---
    clp = np.column_stack([xp_ref, yp_ref, np.zeros_like(xp_ref)])
    clw = (R_inv @ clp.T).T   # (N,3)
    centerline_xy = clw[:, :2]

    # 出力
    np.savetxt(OUTPUT_CSV, centerline_xy, fmt="%.6f", delimiter=",")

    if VERBOSE:
        total_bins = len(edges) - 1
        used_bins  = len(xp)
        print("✅ 中心線CSVを出力:", OUTPUT_CSV)
        print(f"  PCA 旋回角θ(rad): {theta:.6f}")
        print(f"  粗中心線 → 垂直リファイン済み点数: {used_bins}")
        print(f"  BIN_SIZE: {BIN_SIZE} m, MIN_PTS_PER_BIN: {MIN_PTS_PER_BIN}, SMOOTH_WINDOW: {SMOOTH_WINDOW_M} m")
        print(f"  垂直スライス帯: 法線±{CROSS_HALF_WIDTH_M} m, 接線±{LONG_HALF_LEN_M} m, 最低点数{MIN_PTS_PER_STRIP}")

if __name__ == "__main__":
    main()
