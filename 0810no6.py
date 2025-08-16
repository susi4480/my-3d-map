# -*- coding: utf-8 -*-
"""
入:  /data/0731_suidoubasi_ue.las
出:  /output/0810_sections_hits_gap50.las

処理:
- LAS読込 → x,y を 0.05m丸めで辞書化
- gap=50m 相当で X 列を間引きしつつ「左右岸の端 (z<=UKC)」から中点を取り through を作成
- through 間を 0.5m刻みで内挿 → 各点で中心線に直交する断面を生成
- 断面上を 0.05m刻みでサンプリング → 丸めた (x,y) で辞書ヒットした元点を収集
- 収集点のみ LAS 出力（RGBがあれば維持、ヘッダ継承）

laspyの x/y/z は ScaledArrayView なので、np.asarray(..., dtype=...) で明示変換。
"""

import os
import math
import numpy as np
import laspy

# ========= 入出力 =========
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0810_sections_hits_gap50.las"

# ========= パラメータ =========
UKC = -1.0                  # [m] 左右岸端の抽出に使う閾値（z<=UKC を水面下とみなす）
gap = 50.0                  # [m] 河川概形取得用：X方向の列の間引きピッチ
line_length = 60.0          # [m] 断面の全長（±line_length/2）
section_interval = 0.5      # [m] through 間の内挿ピッチ（断面中心の間隔）
sample_step = 0.05          # [m] 断面上のサンプリング刻み（辞書ヒット向け）
min_points_per_section = 1  # 各断面の最低ヒット数（満たない断面はスキップ）

# ========= ユーティリティ =========
def round_0_or_5(a: float) -> float:
    """0.01m単位で四捨五入し、下1桁を0/5/繰上げに正規化（0.05mグリッド）"""
    int_a = round(a * 100)
    rem = int_a % 10
    if rem < 3:
        new = int_a - rem
    elif rem < 8:
        new = int_a - rem + 5
    else:
        new = int_a - rem + 10
    return new / 100.0

def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):
        header.srs = src_header.srs
    if getattr(src_header, "vlrs", None):
        header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None):
        header.evlrs.extend(src_header.evlrs)
    return header

def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

# ========= メイン =========
def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    # --- LAS 読込 ---
    las = laspy.read(INPUT_LAS)
    in_dims = set(las.point_format.dimension_names)
    has_rgb = {"red","green","blue"} <= in_dims

    # ScaledArrayView → ndarray に明示変換
    X = np.asarray(las.x, dtype=float)
    Y = np.asarray(las.y, dtype=float)
    Z = np.asarray(las.z, dtype=float)

    if has_rgb:
        R = np.asarray(las.red,   dtype=np.uint16)
        G = np.asarray(las.green, dtype=np.uint16)
        B = np.asarray(las.blue,  dtype=np.uint16)

    n_pts = X.size
    if n_pts == 0:
        raise RuntimeError("入力LASに点がありません。")

    # --- 0.05m丸め（辞書ヒット前提のため）---
    Xq = np.vectorize(round_0_or_5)(X)
    Yq = np.vectorize(round_0_or_5)(Y)

    # --- (x,y)-> index の辞書（同キーは z 最小を優先）---
    index_map = {}
    z_map = {}
    if has_rgb:
        rgb_map = {}

    for idx in range(n_pts):
        key = (Xq[idx], Yq[idx])
        zi = Z[idx]
        if key not in z_map or zi < z_map[key]:
            z_map[key] = zi
            index_map[key] = idx
            if has_rgb:
                rgb_map[key] = (R[idx], G[idx], B[idx])

    # --- X一意値（丸め後）を gap=50m ピッチで選定 ---
    unique_X = np.array(sorted(set(Xq)))
    if unique_X.size == 0:
        raise RuntimeError("丸め後のXが空です。")

    selected_X = []
    c = unique_X.min()
    for xv in unique_X:
        while xv > c:
            c += gap
        if abs(xv - c) < 1e-9:
            selected_X.append(xv)

    # --- 列ごとに z<=UKC の最初/最後 → 中点 = through ---
    through = []
    for xv in selected_X:
        ys = sorted(y for (xx, y) in z_map.keys() if abs(xx - xv) < 1e-9)
        if not ys:
            continue
        left = right = None
        for yv in ys:
            if z_map[(xv, yv)] <= UKC:
                left = (xv, yv); break
        for yv in reversed(ys):
            if z_map[(xv, yv)] <= UKC:
                right = (xv, yv); break
        if left and right:
            cx = round_0_or_5((left[0] + right[0]) * 0.5)
            cy = round_0_or_5((left[1] + right[1]) * 0.5)
            through.append([cx, cy])

    if len(through) < 2:
        raise RuntimeError("through（中心線）が作成できません。UKC/gap/データの丸め間隔を確認してください。")

    # --- through 間を 0.5m刻みで内挿して断面中心 btw_thru を作成 ---
    btw_thru = []
    for i in range(len(through)-1):
        x1, y1 = through[i]
        x2, y2 = through[i+1]
        d = l2((x1,y1), (x2,y2))
        if d < 1e-9:
            continue
        n_steps = int(d / section_interval)
        for si in range(n_steps):
            s = si * section_interval
            t = s / d
            x = round_0_or_5(x1 + t * (x2 - x1))
            y = round_0_or_5(y1 + t * (y2 - y1))
            btw_thru.append([x, y])

    if len(btw_thru) == 0:
        raise RuntimeError("断面中心が0件です。section_intervalの調整が必要かもしれません。")

    # --- 各断面で直交方向にサンプリングし、辞書ヒットした点を集める ---
    hits_idx = set()

    for i in range(len(btw_thru)):
        # 断面方向は近傍のthroughの向きから決定
        if i == 0:
            x1, y1 = through[0]
            x2, y2 = through[1]
        elif i >= len(through) - 1:
            x1, y1 = through[-2]
            x2, y2 = through[-1]
        else:
            x1, y1 = through[i-1]
            x2, y2 = through[i+1]

        dx = x2 - x1
        dy = y2 - y1

        # ゼロ割りガード
        if abs(dx) < 1e-12:
            # 中心線がほぼ縦 → 直交は水平
            m_theta = 90.0
            m2 = 0.0  # y一定で x 掃引
        else:
            m1 = dy / dx
            m2 = -1.0 / m1
            m_theta = abs(math.degrees(math.atan(m1)))

        Xc, Yc = btw_thru[i]
        intercept = Yc - m2 * Xc  # y = m2*x + b

        if abs(m2) < 1e-12:
            # 水平断面: y一定で x を掃引
            x_range = line_length * 0.5
            x = round_0_or_5(Xc - x_range)
            tmp_idx = []
            while x <= Xc + x_range + 1e-9:
                x = round_0_or_5(x)
                y = Yc
                key = (x, y)
                if key in index_map:
                    tmp_idx.append(index_map[key])
                x += sample_step
        else:
            x_range = abs((line_length * 0.5) / m2)
            y_range = abs(m2 * (line_length * 0.5))
            tmp_idx = []

            if 0.0 <= m_theta <= 45.0:
                # xで掃引 → y = m2*x + b
                x = round_0_or_5(Xc - x_range)
                while x <= Xc + x_range + 1e-9:
                    x = round_0_or_5(x)
                    y = round_0_or_5(m2 * x + intercept)
                    key = (x, y)
                    if key in index_map:
                        tmp_idx.append(index_map[key])
                    x += sample_step
            else:
                # yで掃引 → x = (y - b)/m2
                y = round_0_or_5(Yc - y_range)
                while y <= Yc + y_range + 1e-9:
                    y = round_0_or_5(y)
                    x = round_0_or_5((y - intercept) / m2)
                    key = (x, y)
                    if key in index_map:
                        tmp_idx.append(index_map[key])
                    y += sample_step

        if len(tmp_idx) >= min_points_per_section:
            for ii in tmp_idx:
                hits_idx.add(ii)

    if len(hits_idx) == 0:
        raise RuntimeError("ヒット点が0件でした。UKC/gap/line_length/sample_step を調整してください。")

    hits_idx = np.fromiter(hits_idx, dtype=int)
    out_xyz = np.column_stack([X[hits_idx], Y[hits_idx], Z[hits_idx]])

    # --- 出力LAS（RGBがあれば保持、ヘッダ継承）---
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = out_xyz.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = out_xyz[:, 0]
    las_out.y = out_xyz[:, 1]
    las_out.z = out_xyz[:, 2]

    if has_rgb and {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        out_rgb = np.column_stack([R[hits_idx], G[hits_idx], B[hits_idx]])
        las_out.red   = out_rgb[:, 0]
        las_out.green = out_rgb[:, 1]
        las_out.blue  = out_rgb[:, 2]

    las_out.write(OUTPUT_LAS)

    print("✅ 出力完了:", OUTPUT_LAS)
    print(f"  入力点数           : {n_pts:,d}")
    print(f"  through点数        : {len(through):,d}")
    print(f"  断面中心点数       : {len(btw_thru):,d}")
    print(f"  出力ヒット点数     : {N:,d}")

if __name__ == "__main__":
    main()
