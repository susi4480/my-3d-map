# -*- coding: utf-8 -*-
"""
【ワンパス版】中心線CSVを自動生成し、その中心線に沿って“法線方向の薄い帯”で点群を抽出してLAS出力
- 入力LASを Z ≤ Z_LIMIT で制限
- 固定Y軸スライス（幅 BIN_Y）で各スライスの X中央値から中心線を推定 → CSV保存（ヘッダなし: X,Y）
- 中心線を弧長 slice_interval で等間隔サンプリング
- 各サンプル点の接線→法線を計算し、接線に垂直な帯（±slice_thickness/2）で点群抽出
- 抽出点のみを連結して LAS 保存（CRS/VLR/EVLR/SRS 継承、RGBはあればそのまま）
"""

import os
import numpy as np
import laspy

# === 入出力 ===
INPUT_LAS   = "/data/0725_suidoubasi_sita.las"   # 実在パスに変更
OUT_CL_CSV  = "/output/centerline_xy.csv"                  # 生成される中心線CSV（ヘッダなし: X,Y）
OUTPUT_LAS  = "/output/0808no8_slices_only_map.las"           # 抽出点のLAS

# === パラメータ（中心線推定） ===
Z_LIMIT         = 1.0   # [m] これ以下のみ使用
BIN_Y           = 2.0   # [m] Y軸スライス幅
MIN_PTS_PER_BIN = 50    # スライス内の最低点数
SMOOTH_WINDOW_M = 10.0  # [m] Xの移動平均（0で無効）

# === パラメータ（法線スライス抽出） ===
slice_thickness = 0.2   # [m] 接線方向に ±0.1m の薄い帯
slice_interval  = 0.5   # [m] 中心線の弧長等間隔サンプリング
VOXEL_SIZE      = 0.0   # [m] 出力点数を減らしたい時のみ >0（色保持は省略）

VERBOSE = True


# ===== ユーティリティ =====
def moving_average_1d(arr, win_m, bin_m):
    if win_m <= 0 or len(arr) < 2:
        return arr
    win = max(1, int(round(win_m / bin_m)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(arr_pad, kernel, mode='valid')

def make_centerline_ybins(points_xyz, z_limit, bin_y, min_pts, smooth_window_m):
    """固定Y軸スライスで中心線（X中央値, スライス中央Y）を作る"""
    pts = points_xyz[points_xyz[:,2] <= z_limit]
    if len(pts) == 0:
        raise RuntimeError("Z制限後に点がありません。")
    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    edges = np.arange(y_min, y_max + bin_y, bin_y)
    y_centers = 0.5 * (edges[:-1] + edges[1:])
    Xc, Yc = [], []
    for i in range(len(edges)-1):
        y0, y1 = edges[i], edges[i+1]
        m = (pts[:,1] >= y0) & (pts[:,1] < y1)
        if np.count_nonzero(m) < min_pts:
            continue
        slab = pts[m]
        Xc.append(np.median(slab[:,0]))
        Yc.append(y_centers[i])
    if len(Xc) < 2:
        raise RuntimeError("有効なYスライスが不足（中心線を作成できません）。")
    Xc = np.array(Xc, float); Yc = np.array(Yc, float)
    order = np.argsort(Yc)
    Xc, Yc = Xc[order], Yc[order]
    Xc = moving_average_1d(Xc, smooth_window_m, bin_y)
    return np.column_stack([Xc, Yc])

def resample_polyline_by_arclength(xy, step):
    """折れ線xyを弧長stepごとに等間隔サンプリング"""
    seg = np.diff(xy, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    L = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(L[-1])
    if total <= 0:
        return xy.copy()
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
    """前後差分で接線t・法線n（単位）"""
    n = xy.shape[0]
    t = np.zeros((n,2), float)
    if n >= 2:
        t[1:-1] = xy[2:] - xy[:-2]
        t[0]    = xy[1] - xy[0]
        t[-1]   = xy[-1] - xy[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    t /= norm
    nvec = np.stack([-t[:,1], t[:,0]], axis=1)  # 左法線
    return t, nvec

def copy_header_with_metadata(src_header):
    """laspy 2.x: 新規ヘッダに scales/offsets/SRS/VLR/EVLR を継承"""
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if hasattr(src_header, "srs") and src_header.srs is not None:
        header.srs = src_header.srs
    src_vlrs = getattr(src_header, "vlrs", None)
    if src_vlrs:
        header.vlrs.extend(src_vlrs)
    src_evlrs = getattr(src_header, "evlrs", None)
    if src_evlrs:
        header.evlrs.extend(src_evlrs)
    return header

def voxel_downsample(points_xyz, voxel_size):
    if voxel_size <= 0:
        return points_xyz
    keys = np.floor(points_xyz / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points_xyz[np.sort(idx)]


# ===== メイン =====
def main():
    # 入出力の存在確認と作成
    if not os.path.isfile(INPUT_LAS):
        raise FileNotFoundError(f"INPUT_LAS not found: {INPUT_LAS}")
    os.makedirs(os.path.dirname(OUT_CL_CSV) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    # LAS読み込み & Zフィルタ
    las = laspy.read(INPUT_LAS)
    pts_all = np.vstack([las.x, las.y, las.z]).T
    zmask = pts_all[:,2] <= Z_LIMIT
    pts = pts_all[zmask]
    if pts.size == 0:
        raise RuntimeError("Z制限後に点がありません。Z_LIMITを見直してください。")

    # 入力RGB有無
    in_dims = set(las.point_format.dimension_names)
    has_rgb_in = {"red","green","blue"} <= in_dims
    if has_rgb_in:
        rgb_all = np.vstack([las.red, las.green, las.blue]).T
        rgb = rgb_all[zmask]

    # 1) 中心線を自動推定（固定YスライスのX中央値）→ CSV保存
    centerline = make_centerline_ybins(pts_all, Z_LIMIT, BIN_Y, MIN_PTS_PER_BIN, SMOOTH_WINDOW_M)
    np.savetxt(OUT_CL_CSV, centerline, fmt="%.6f", delimiter=",")

    # 2) 中心線を弧長 slice_interval で等間隔サンプリング → 接線/法線
    cl_samp = resample_polyline_by_arclength(centerline, slice_interval)
    t, _ = tangents_normals_from_polyline(cl_samp)

    # 3) 各サンプル点で“接線に垂直な帯”に入る点を抽出（|s| <= thickness/2）
    XY = pts[:, :2]
    half_thick = slice_thickness * 0.5
    keep_xyz = []
    keep_rgb = []

    for i in range(len(cl_samp)):
        c  = cl_samp[i]
        ti = t[i]
        dxy = XY - c
        s = dxy @ ti
        m = np.abs(s) <= half_thick
        if not np.any(m):
            continue
        keep_xyz.append(pts[m])
        if has_rgb_in:
            keep_rgb.append(rgb[m])

    if len(keep_xyz) == 0:
        raise RuntimeError("スライスに入る点がありません。slice_thickness / slice_interval / Z_LIMIT を見直してください。")

    out_xyz = np.vstack(keep_xyz)
    if VOXEL_SIZE > 0:
        out_xyz = voxel_downsample(out_xyz, VOXEL_SIZE)  # ※色は保持しない簡易版
    if has_rgb_in and VOXEL_SIZE == 0:
        out_rgb = np.vstack(keep_rgb)

    # 4) LAS出力（色は変更しない）
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = out_xyz.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    las_out.x = out_xyz[:,0]; las_out.y = out_xyz[:,1]; las_out.z = out_xyz[:,2]

    out_dims = set(las_out.point_format.dimension_names)
    if has_rgb_in and VOXEL_SIZE == 0 and {"red","green","blue"} <= out_dims:
        las_out.red   = out_rgb[:,0]
        las_out.green = out_rgb[:,1]
        las_out.blue  = out_rgb[:,2]

    las_out.write(OUTPUT_LAS)

    if VERBOSE:
        print(f"✅ 中心線CSV: {OUT_CL_CSV}（点数 {len(centerline)}）")
        print(f"✅ 抽出LAS:  {OUTPUT_LAS}")
        print(f"  Z_LIMIT={Z_LIMIT}m, BIN_Y={BIN_Y}m, MIN_PTS_PER_BIN={MIN_PTS_PER_BIN}, SMOOTH={SMOOTH_WINDOW_M}m")
        print(f"  slice_thickness=±{half_thick:.2f}m, slice_interval={slice_interval}m")
        print(f"  抽出点数: {N}（入力Z≤: {np.count_nonzero(zmask)}）")
        print(f"  RGB入出力: in={has_rgb_in}, out_rgb_dims={'yes' if {'red','green','blue'} <= out_dims else 'no'}")

if __name__ == "__main__":
    main()
