# -*- coding: utf-8 -*-
"""
【安定版】中心線に沿った“法線方向”の薄い帯でスライス抽出（色は変更しない）
- Z ≤ Z_LIMIT でフィルタ
- 固定Y軸スライスで中心線を推定（X中央値）→ 平滑化
- 弧長 slice_interval で等間隔サンプリング → さらに平滑化
- 接線ベクトル：
    - 前後差分
    - 連続性強制（前のtと180°超なら反転）
    - 角度をunwrapし移動平均 → 大きく外れたスライスはスキップ
- 各スライスで |s| ≤ slice_thickness/2 の点だけ抽出
- 点数が少ないスライスはスキップ
- LAS出力はCRS/VLR/EVLR/SRS継承（laspy 2.x）
"""

import os
import numpy as np
import laspy

# === 入出力 ===
INPUT_LAS  = "/output/0731_suidoubasi_ue.las"       # 実在ファイルに
OUTPUT_LAS = "/output/0808no9_slices_only_map.las"

# === パラメータ（中心線推定） ===
Z_LIMIT         = 3.5   # [m] これ以下のみ
BIN_Y           = 2.0   # [m] Yスライス幅
MIN_PTS_PER_BIN = 50    # スライス内最低点数
SMOOTH_WINDOW_M = 10.0  # [m] Xの移動平均（0で無効）

# === パラメータ（スライス抽出） ===
slice_thickness      = 0.2   # [m] ±0.1m
slice_interval       = 0.5   # [m] 弧長サンプル間隔
MIN_PTS_PER_SLICE    = 80    # 各スライスで採用する最低点数（少なすぎる断面はスキップ）
ANGLE_SMOOTH_WIN_PTS = 5     # 接線角の平滑化窓（奇数）
ANGLE_OUTLIER_DEG    = 30.0  # [deg] 平滑角からの乖離がこれ以上ならスライスを捨てる

VOXEL_SIZE      = 0.0  # [m] 出力を軽くしたいときのみ >0（色維持は省略）
VERBOSE         = True


# === ユーティリティ ===
def moving_average_1d(arr, win_m, bin_m):
    if win_m <= 0 or len(arr) < 2:
        return arr
    win = max(1, int(round(win_m / bin_m)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(arr_pad, kernel, mode="valid")

def make_centerline_ybins(points_xyz, z_limit, bin_y, min_pts, smooth_window_m):
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

def smooth_polyline_xy(xy, win_pts=5):
    if win_pts <= 1 or len(xy) < 3:
        return xy
    if win_pts % 2 == 0:
        win_pts += 1
    pad = win_pts // 2
    xpad = np.pad(xy[:,0], (pad,pad), mode="edge")
    ypad = np.pad(xy[:,1], (pad,pad), mode="edge")
    ker = np.ones(win_pts)/win_pts
    xs = np.convolve(xpad, ker, mode="valid")
    ys = np.convolve(ypad, ker, mode="valid")
    return np.column_stack([xs, ys])

def tangents_normals_continuous(xy):
    """接線tの連続性を強制しつつ算出。向きが反転したら反転補正。ゼロ長は前のtを引き継ぐ。"""
    n = xy.shape[0]
    t = np.zeros((n,2), float)
    if n >= 3:
        t[1:-1] = xy[2:] - xy[:-2]
    if n >= 2:
        t[0]  = xy[1] - xy[0]
        t[-1] = xy[-1] - xy[-2]
    # 正規化＆ゼロ対策
    for i in range(n):
        norm = np.linalg.norm(t[i])
        if norm < 1e-12:
            t[i] = t[i-1] if i>0 else np.array([1.0,0.0])
        else:
            t[i] /= norm
        if i>0 and np.dot(t[i], t[i-1]) < 0:  # 連続性（180°超えで反転）
            t[i] = -t[i]
    # 法線
    nvec = np.stack([-t[:,1], t[:,0]], axis=1)
    return t, nvec

def stabilize_angles(t, win_pts=5, outlier_deg=30.0):
    """接線角をunwrap→平滑化→大外れはフラグを返す（True=良い/False=外れ）"""
    ang = np.arctan2(t[:,1], t[:,0])
    ang_unw = np.unwrap(ang)
    if win_pts % 2 == 0:
        win_pts += 1
    pad = win_pts // 2
    a_pad = np.pad(ang_unw, (pad,pad), mode="edge")
    ker = np.ones(win_pts)/win_pts
    a_smooth = np.convolve(a_pad, ker, mode="valid")
    diff = np.rad2deg(np.abs(ang_unw - a_smooth))
    ok = diff <= outlier_deg
    # スムージング後の角から単位ベクトル再構成（安定版）
    t_s = np.column_stack([np.cos(a_smooth), np.sin(a_smooth)])
    return t_s, ok

def copy_header_with_metadata(src_header):
    header = laspy.LasHeader(point_format=src_header.point_format, version=src_header.version)
    header.scales  = src_header.scales
    header.offsets = src_header.offsets
    if getattr(src_header, "srs", None):
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


# === メイン ===
def main():
    # 安全な出力ディレクトリ作成
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    # 読み込み
    las = laspy.read(INPUT_LAS)
    pts_all = np.vstack([las.x, las.y, las.z]).T

    # RGB有無
    in_dims = set(las.point_format.dimension_names)
    has_rgb_in = {"red","green","blue"} <= in_dims
    if has_rgb_in:
        rgb_all = np.vstack([las.red, las.green, las.blue]).T
    else:
        rgb_all = None

    # Z≤フィルタ
    zmask = pts_all[:,2] <= Z_LIMIT
    pts = pts_all[zmask]
    if pts.size == 0:
        raise RuntimeError("Z制限後に点がありません。Z_LIMITを見直してください。")
    XY = pts[:, :2]
    if has_rgb_in:
        rgb = rgb_all[zmask]

    # 中心線（固定Yスライス→平滑化）
    centerline = make_centerline_ybins(pts_all, Z_LIMIT, BIN_Y, MIN_PTS_PER_BIN, SMOOTH_WINDOW_M)

    # 弧長等間隔サンプリング ＋ 追加平滑化（点ベース）
    cl_samp = resample_polyline_by_arclength(centerline, slice_interval)
    cl_samp = smooth_polyline_xy(cl_samp, win_pts=5)  # ← 追加で軽く平滑化

    # 接線：連続性強制 → 角度平滑化＆外れスライス判定
    t_raw, _ = tangents_normals_continuous(cl_samp)
    t_stab, ok_ang = stabilize_angles(t_raw, win_pts=ANGLE_SMOOTH_WIN_PTS, outlier_deg=ANGLE_OUTLIER_DEG)

    # 抽出ループ（外れ角のスライスや点数の少ないスライスはスキップ）
    half_thick = slice_thickness * 0.5
    keep_xyz = []
    keep_rgb = []
    kept = 0
    skipped_angle = 0
    skipped_sparse = 0

    for i in range(len(cl_samp)):
        if not ok_ang[i]:
            skipped_angle += 1
            continue
        c  = cl_samp[i]
        ti = t_stab[i]

        dxy = XY - c
        s = dxy @ ti
        m = np.abs(s) <= half_thick
        cnt = np.count_nonzero(m)
        if cnt < MIN_PTS_PER_SLICE:
            skipped_sparse += 1
            continue

        keep_xyz.append(pts[m])
        if has_rgb_in:
            keep_rgb.append(rgb[m])
        kept += 1

    if kept == 0:
        raise RuntimeError("全スライスが外れ角/点不足でスキップされました。ANGLE_OUTLIER_DEGやMIN_PTS_PER_SLICE、slice_thicknessを見直してください。")

    out_xyz = np.vstack(keep_xyz)
    if VOXEL_SIZE > 0:
        out_xyz = voxel_downsample(out_xyz, VOXEL_SIZE)  # ※ 色は維持しない簡易版
    if has_rgb_in and VOXEL_SIZE == 0:
        out_rgb = np.vstack(keep_rgb)

    # 出力
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

    out_dims = set(las_out.point_format.dimension_names)
    if has_rgb_in and VOXEL_SIZE == 0 and {"red","green","blue"} <= out_dims:
        las_out.red   = out_rgb[:,0]
        las_out.green = out_rgb[:,1]
        las_out.blue  = out_rgb[:,2]

    las_out.write(OUTPUT_LAS)

    if VERBOSE:
        print(f"✅ 出力: {OUTPUT_LAS}")
        print(f"  Z_LIMIT={Z_LIMIT}m, BIN_Y={BIN_Y}m, MIN_PTS_PER_BIN={MIN_PTS_PER_BIN}, SMOOTH={SMOOTH_WINDOW_M}m")
        print(f"  slice_thickness=±{half_thick:.2f}m, slice_interval={slice_interval}m")
        print(f"  採用スライス: {kept} / 角度外れスキップ: {skipped_angle} / 点不足スキップ: {skipped_sparse}")
        print(f"  抽出点数: {N}（入力Z≤: {np.count_nonzero(zmask)}）")
        print(f"  RGB入出力: in={has_rgb_in}, out_rgb_dims={'yes' if {'red','green','blue'} <= out_dims else 'no'}")

if __name__ == "__main__":
    main()
