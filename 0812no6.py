# -*- coding: utf-8 -*-
"""
【デバッグ専用】中心線バンド・オーバーラップスライス（CSVなし）
- Z ≤ Z_LIMIT で入力LASをフィルタ
- UKC法で中心線推定 → GAP_DISTで間引き → SECTION_INTERVALで内挿（下地ポリライン）
- 弧長等間隔サンプリング（step_s = SLICE_INTERVAL - SLICE_OVERLAP）
- 各サンプル点で接線/法線を計算し、接線直交の薄い帯(|u|≤slice_thickness/2, |v|≤line_length/2)で抽出
- 各スライスをLASにフォルダ出力（連番）
- 隣接スライスの中心間隔Δsと、想定オーバーラップ量(max(0, slice_thickness - Δs))をログ
"""

import os
import math
import numpy as np
import laspy

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
SLICES_DIR = r"/output/overlap_band_slices"
os.makedirs(SLICES_DIR, exist_ok=True)

# ===== 前処理パラメータ =====
Z_LIMIT            = 1.3   # [m] 使用する点の高さ上限

# 中心線（UKC法）
UKC                = -1.0  # [m] 水面下しきい値（左岸/右岸抽出に使用）
BIN_X              = 2.0   # [m] X方向の区間幅（「区間」や「帯」でOK）
MIN_PTS_PER_XBIN   = 50    # 各区間の最低点数
GAP_DIST           = 50.0  # [m] 中心線候補の間引き距離
SECTION_INTERVAL   = 0.5   # [m] 内挿ステップ（下地ポリラインの密度）

# スライス（オーバーラップ）
SLICE_INTERVAL     = 1.0   # [m] スライス中心の基本間隔
SLICE_OVERLAP      = 0.5   # [m] スライス間オーバーラップ（中心間隔から引く）
SLICE_THICKNESS    = 0.20  # [m] 接線方向の帯厚（±半分）
LINE_LENGTH        = 60.0  # [m] 法線方向の幅（±半分）
TOL                = 0.05  # [m] ログでの許容差

VERBOSE = True

# ===== ユーティリティ =====
def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):   header.srs  = src_header.srs
    if getattr(src_header, "vlrs", None):  header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def write_las(path, header_src, xyz, rgb=None):
    if xyz.shape[0] == 0:
        return
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    n = xyz.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(n, header=header)
    las_out.x = xyz[:,0]; las_out.y = xyz[:,1]; las_out.z = xyz[:,2]
    if rgb is not None and {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red, las_out.green, las_out.blue = rgb[:,0], rgb[:,1], rgb[:,2]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)

def resample_polyline_by_arclength(xy, step):
    seg = np.diff(xy, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    L = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(L[-1])
    if total <= 0: return xy.copy()
    targets = np.arange(0.0, total + 1e-9, step)
    out, j = [], 0
    for s in targets:
        while j+1 < len(L) and L[j+1] < s:
            j += 1
        if j+1 >= len(L):
            out.append(xy[-1]); break
        t = (s - L[j]) / max(L[j+1]-L[j], 1e-12)
        p = xy[j]*(1-t) + xy[j+1]*t
        out.append(p)
    return np.asarray(out)

def tangents_normals_from_polyline(xy):
    n = xy.shape[0]
    t = np.zeros((n,2), float)
    if n >= 2:
        t[1:-1] = xy[2:] - xy[:-2]
        t[0]    = xy[1] - xy[0]
        t[-1]   = xy[-1] - xy[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    t /= norm
    nvec = np.stack([-t[:,1], t[:,0]], axis=1)
    return t, nvec

# ===== メイン =====
def main():
    las = laspy.read(INPUT_LAS)
    dims = set(las.point_format.dimension_names)
    has_rgb = {"red","green","blue"} <= dims

    X = np.asarray(las.x, float)
    Y = np.asarray(las.y, float)
    Z = np.asarray(las.z, float)
    if has_rgb:
        RGB = np.vstack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)]).T.astype(np.uint16)

    # Z制限
    mZ = (Z <= Z_LIMIT)
    X = X[mZ]; Y = Y[mZ]; Z = Z[mZ]
    if has_rgb: RGB = RGB[mZ]
    if len(X) == 0:
        raise RuntimeError("Z制限内の点がありません。")

    xy = np.column_stack([X, Y])

    # --- 中心線（UKCで左右岸→中点） ---
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (xy[:,0] >= x0) & (xy[:,0] < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy = xy[m]; slab_z  = Z[m]
        order = np.argsort(slab_xy[:,1])
        slab_xy = slab_xy[order]; slab_z = slab_z[order]
        under = slab_z <= UKC
        if not np.any(under): 
            continue
        idx = np.where(under)[0]
        left  = slab_xy[idx[0]]
        right = slab_xy[idx[-1]]
        c = 0.5*(left + right)
        through.append(c)
    if len(through) < 2:
        raise RuntimeError("中心線が作れません。UKCやBIN_Xを調整してください。")
    through = np.asarray(through, float)

    # --- GAP_DISTで間引き ---
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    # --- 下地ポリライン（SECTION_INTERVALで内挿） ---
    centers = []
    for i in range(len(through)-1):
        p, q = through[i], through[i+1]
        d = l2(p, q)
        if d < 1e-9: 
            continue
        n_steps = int(d / SECTION_INTERVAL)
        for s_i in range(n_steps+1):
            s = min(s_i * SECTION_INTERVAL, d)
            t = s / d
            centers.append((1-t)*p + t*q)
    centers = np.asarray(centers, float)
    if len(centers) < 2:
        raise RuntimeError("内挿後の中心線が短すぎます。")

    # --- オーバーラップ付きサンプリング ---
    step_s = max(1e-6, SLICE_INTERVAL - SLICE_OVERLAP)  # 中心間隔
    cl_samp = resample_polyline_by_arclength(centers, step_s)
    t_hat, n_hat = tangents_normals_from_polyline(cl_samp)

    # --- スライス抽出 & 出力 ---
    XY = xy
    half_th = SLICE_THICKNESS * 0.5
    half_v  = LINE_LENGTH * 0.5

    total_pts = 0
    slice_id  = 0
    print(f"スライス中心数: {len(cl_samp)}  (step_s={step_s:.2f} m, thickness={SLICE_THICKNESS:.2f} m)")
    print(f"想定オーバーラップ(弧長) ≈ max(0, thickness - step_s) = {max(0.0, SLICE_THICKNESS-step_s):.2f} m\n")

    prev_c = None
    for i in range(len(cl_samp)):
        c = cl_samp[i]
        th = t_hat[i]
        nh = n_hat[i]

        dxy = XY - c
        u = dxy @ th   # 接線方向
        v = dxy @ nh   # 法線方向

        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_v)
        idx = np.where(m_band)[0]
        if idx.size == 0:
            # 空スライスはスキップ（出力もしない）
            if prev_c is None:
                print(f"slice {i:04d}: 空 (抽出0点)")
            else:
                ds = l2(prev_c, c)
                print(f"slice {i:04d}: 空 (Δs={ds:.2f} m)")
            continue

        xyz = np.column_stack([X[idx], Y[idx], Z[idx]])
        rgb = RGB[idx] if has_rgb else None

        # ログ（中心間隔と想定オーバーラップ）
        if prev_c is None:
            print(f"slice {i:04d}: pts={idx.size:,d}  (first)")
        else:
            ds = l2(prev_c, c)
            warn = "" if abs(ds - step_s) <= TOL else "  <-- ※中心間隔が期待値とズレ"
            ovl = max(0.0, SLICE_THICKNESS - ds)  # 理論値：帯厚−中心間隔（負なら0）
            print(f"slice {i:04d}: pts={idx.size:,d}  Δs={ds:.2f} m  overlap≈{ovl:.2f} m{warn}")

        out_path = os.path.join(SLICES_DIR, f"slice_{slice_id:04d}.las")
        write_las(out_path, las.header, xyz, rgb)
        total_pts += idx.size
        slice_id  += 1
        prev_c = c

    print("\n==== Summary ====")
    print(f"書き出しスライス数: {slice_id}")
    print(f"スライス出力フォルダ: {SLICES_DIR}")
    print(f"Z≤{Z_LIMIT} の入力点: {len(X):,d}  / スライスに入った点の延べ数: {total_pts:,d}")
    print("(延べ数はオーバーラップのため入力点数より大きくなり得ます)")

if __name__ == "__main__":
    main()
