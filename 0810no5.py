# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import laspy
from scipy.spatial import cKDTree

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0810_sections_hits_gap50.las"

# ===== パラメータ =====
UKC = -1.0                  # [m] z<=UKC を水面下として左右岸抽出に使う
BIN_X = 2.0                 # [m] 中心線作成時の X ビン幅
MIN_PTS_PER_XBIN = 50       # 各 X ビンに必要な最小点数
SECTION_INTERVAL = 0.5      # [m] 断面（中心線内挿）間隔
LINE_LENGTH = 60.0          # [m] 断面の全長
SAMPLE_STEP = 0.05          # [m] 断面上のサンプル間隔
NN_RADIUS = 0.08            # [m] 最近傍許容半径
MIN_POINTS_PER_SECTION = 1  # 各断面の最小ヒット数
INCLUDE_ORIGINAL_POINTS = False
GAP_DIST = 50.0             # [m] gap=50m相当 # gap処理追加

# ===== ユーティリティ =====
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

# ===== メイン =====
def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    las = laspy.read(INPUT_LAS)
    has_rgb_in = {"red","green","blue"} <= set(las.point_format.dimension_names)

    xyz = np.column_stack([las.x, las.y, las.z])
    xy  = xyz[:, :2]
    z   = xyz[:, 2]

    # --- 1) Xビンごとに左右岸→中点 ---
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (xy[:,0] >= x0) & (xy[:,0] < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy = xy[m]
        slab_z  = z[m]
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

    # --- 2) gap=50mで間引き # gap処理追加 ---
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    # --- 3) 中心線を内挿 ---
    centers = []
    for i in range(len(through)-1):
        p, q = through[i], through[i+1]
        d = l2(p, q)
        if d < 1e-9: continue
        n_steps = int(d / SECTION_INTERVAL)
        for s_i in range(n_steps+1):
            s = min(s_i * SECTION_INTERVAL, d)
            t = s / d
            centers.append((1-t)*p + t*q)
    centers = np.asarray(centers, float)

    # --- 4) 断面でヒット収集 ---
    tree = cKDTree(xy)
    half_len = LINE_LENGTH * 0.5
    s_samples = np.arange(-half_len, half_len + 1e-9, SAMPLE_STEP)

    hit_idx_set = set()
    kept_sections = skipped_sections = 0

    for i in range(len(centers)-1):
        c = centers[i]
        c_next = centers[i+1]
        t_vec = c_next - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9:
            continue
        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

        sec_xy = c[None,:] + s_samples[:,None]*n_hat[None,:]
        dists, idxs = tree.query(sec_xy, k=1, workers=-1)
        ok = dists <= NN_RADIUS
        if np.count_nonzero(ok) < MIN_POINTS_PER_SECTION:
            skipped_sections += 1
            continue
        kept_sections += 1
        for ii in np.nonzero(ok)[0]:
            hit_idx_set.add(int(idxs[ii]))

    if not hit_idx_set:
        raise RuntimeError("断面ヒットなし。NN_RADIUSやLINE_LENGTHを調整してください。")

    hit_idx = np.fromiter(hit_idx_set, dtype=int)
    hit_xyz = xyz[hit_idx]

    # 必要なら元点群も含める
    if INCLUDE_ORIGINAL_POINTS:
        out_xyz = np.vstack([xyz, hit_xyz])
        if has_rgb_in:
            rgb_all = np.column_stack([las.red, las.green, las.blue])
            out_rgb = np.vstack([rgb_all, rgb_all[hit_idx]])
    else:
        out_xyz = hit_xyz
        if has_rgb_in:
            out_rgb = np.column_stack([las.red, las.green, las.blue])[hit_idx]

    # --- 5) LAS出力 ---
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = out_xyz.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = out_xyz[:,0]
    las_out.y = out_xyz[:,1]
    las_out.z = out_xyz[:,2]

    if has_rgb_in and {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = out_rgb[:,0]
        las_out.green = out_rgb[:,1]
        las_out.blue  = out_rgb[:,2]

    las_out.write(OUTPUT_LAS)

    print("✅ 出力:", OUTPUT_LAS)
    print(f"  gap=50適用後 中心線点数: {len(through)}")
    print(f"  断面数 (採用/スキップ) : {kept_sections} / {skipped_sections}")
    print(f"  出力点数: {N}")

if __name__ == "__main__":
    main()
