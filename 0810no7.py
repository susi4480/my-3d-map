# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import laspy
from scipy.spatial import cKDTree

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0810no7_sections_hits_with_thickness.las"

# ===== パラメータ =====
UKC = -1.0                  # [m] z<=UKC を水面下として左右岸抽出に使う
BIN_X = 2.0                 # [m] 中心線作成時の X ビン幅
MIN_PTS_PER_XBIN = 50       #    Xビン内の最小点数
SECTION_INTERVAL = 0.5      # [m] 断面中心の間隔
LINE_LENGTH = 60.0          # [m] 断面の全長（±半分）
SLICE_THICKNESS = 0.2       # [m] スライスの厚み（±0.1mなら 0.2）
MIN_POINTS_PER_SECTION = 1  #    断面1本あたりの最小採用点数

INCLUDE_ORIGINAL_POINTS = False  # Trueで元点群も同梱

# ===== ユーティリティ =====
def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):   header.srs = src_header.srs
    if getattr(src_header, "vlrs", None):  header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def l2(p, q):
    return math.hypot(q[0]-p[0], q[1]-p[1])

# ===== メイン =====
def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    las = laspy.read(INPUT_LAS)
    dims = set(las.point_format.dimension_names)
    has_rgb_in = {"red","green","blue"} <= dims

    # 明示的に ndarray 化（ScaledArrayView対策）
    X = np.asarray(las.x, dtype=float)
    Y = np.asarray(las.y, dtype=float)
    Z = np.asarray(las.z, dtype=float)
    if has_rgb_in:
        R = np.asarray(las.red,   dtype=np.uint16)
        G = np.asarray(las.green, dtype=np.uint16)
        B = np.asarray(las.blue,  dtype=np.uint16)

    xyz = np.column_stack([X, Y, Z])
    xy  = xyz[:, :2]
    z   = xyz[:, 2]

    # --- 1) Xビンごとに左右岸（z<=UKC の最初/最後）→ 中点を中心線点に ---
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
        raise RuntimeError("中心線（through）が作れませんでした。UKC/BIN_X/MIN_PTS_PER_XBIN を調整してください。")

    through = np.asarray(through, float)

    # --- 2) 中心線を SECTION_INTERVAL で内挿（断面中心） ---
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

    # --- 3) 厚み付きスライス（帯）で点を収集 ---
    tree = cKDTree(xy)
    half_len = LINE_LENGTH * 0.5
    half_thick = SLICE_THICKNESS * 0.5
    # 半径探索：帯を確実に覆う半径（ちょいマージン）
    search_r = math.hypot(half_len, half_thick) + 1e-6

    hit_idx_set = set()
    kept_sections = skipped_sections = 0

    for i in range(len(centers)-1):
        c = centers[i]
        c_next = centers[i+1]
        t_vec = c_next - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9:
            continue
        t_hat = t_vec / norm              # 接線（中心線方向）
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)  # 法線（断面方向）

        # 候補点を半径探索
        cand_idx = tree.query_ball_point(c, r=search_r)
        if not cand_idx:
            skipped_sections += 1
            continue

        P = xy[cand_idx] - c  # 断面中心基準のベクトル
        # 局所座標 (u:接線, v:法線)
        u = P @ t_hat
        v = P @ n_hat

        mask = (np.abs(u) <= half_thick) & (np.abs(v) <= half_len)
        if not np.any(mask):
            skipped_sections += 1
            continue

        kept_sections += 1
        for idx in np.asarray(cand_idx)[mask]:
            hit_idx_set.add(int(idx))

    if len(hit_idx_set) == 0:
        raise RuntimeError("帯（厚み）付きスライスでヒットなし。SLICE_THICKNESS/LINE_LENGTH/SECTION_INTERVAL を見直してください。")

    hit_idx = np.fromiter(hit_idx_set, dtype=int)
    hit_xyz = xyz[hit_idx]

    # --- 4) 出力点群を構築 ---
    if INCLUDE_ORIGINAL_POINTS:
        out_xyz = np.vstack([xyz, hit_xyz])
        if has_rgb_in:
            out_rgb = np.vstack([
                np.column_stack([R, G, B]),
                np.column_stack([R[hit_idx], G[hit_idx], B[hit_idx]])
            ])
    else:
        out_xyz = hit_xyz
        if has_rgb_in:
            out_rgb = np.column_stack([R[hit_idx], G[hit_idx], B[hit_idx]])

    # --- 5) LAS出力（ヘッダ継承） ---
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
    print(f"  through 点数      : {len(through)}")
    print(f"  断面中心 点数     : {len(centers)}")
    print(f"  採用/スキップ断面 : {kept_sections} / {skipped_sections}")
    print(f"  出力点数          : {N}")

if __name__ == "__main__":
    main()
