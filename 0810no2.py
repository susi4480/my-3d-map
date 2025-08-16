# -*- coding: utf-8 -*-
"""
【安定版】中心線に沿った“縦/横 切替”の薄い帯でスライス抽出（色は変更しない）
- Z ≤ Z_LIMIT でフィルタ
- 固定Y軸スライスで中心線を推定（X中央値）→ 平滑化
- 弧長 slice_interval で等間隔サンプリング → さらに平滑化
- 接線ベクトルの安定化（符号反転抑止＋unwrap＋移動平均）
- 各サンプルで接線角を評価し、以下を自動切替
    - 0～SWITCH_DEG（既定45°）: 縦スライス（|x - cx| ≤ thickness/2）
    - SWITCH_DEG～90°            : 横スライス（|y - cy| ≤ thickness/2）
- 角度外れ/点数不足のスライスはスキップ
- LAS出力はCRS/VLR/EVLR/SRS継承（laspy 2.x）
"""

import os
import numpy as np
import laspy

# === 入出力 ===
INPUT_LAS  = r"C:\Users\user\Documents\lab\outcome\0725_suidoubasi_sita.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_las\0810no2_slices_only_map_pca.las"

# === パラメータ（中心線推定） ===
Z_LIMIT         = 2.0   # [m]
BIN_Y           = 2.0   # [m]
MIN_PTS_PER_BIN = 50
SMOOTH_WINDOW_M = 10.0  # [m] 中心線Xの移動平均（0で無効）

# === パラメータ（スライス抽出） ===
slice_thickness      = 0.2   # [m] ±0.1m
slice_interval       = 0.5   # [m]
MIN_PTS_PER_SLICE    = 80
ANGLE_SMOOTH_WIN_PTS = 5
ANGLE_OUTLIER_DEG    = 30.0  # [deg] 接線角の外れ許容量

# ★ 縦/横切替の閾値（0～90°のうち）
SWITCH_DEG = 45.0

VOXEL_SIZE = 0.0  # >0 で簡易ダウンサンプル（色維持しない）
VERBOSE    = True

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
    n = xy.shape[0]
    t = np.zeros((n,2), float)
    if n >= 3:
        t[1:-1] = xy[2:] - xy[:-2]
    if n >= 2:
        t[0]  = xy[1] - xy[0]
        t[-1] = xy[-1] - xy[-2]
    for i in range(n):
        norm = np.linalg.norm(t[i])
        if norm < 1e-12:
            t[i] = t[i-1] if i>0 else np.array([1.0,0.0])
        else:
            t[i] /= norm
        if i>0 and np.dot(t[i], t[i-1]) < 0:
            t[i] = -t[i]
    nvec = np.stack([-t[:,1], t[:,0]], axis=1)
    return t, nvec

def stabilize_angles(t, win_pts=5, outlier_deg=30.0):
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

    # 弧長等間隔サンプル ＋ 追加平滑化
    cl_samp = resample_polyline_by_arclength(centerline, slice_interval)
    cl_samp = smooth_polyline_xy(cl_samp, win_pts=5)

    # 接線：連続性強制 → 角度平滑化＆外れスライス判定
    t_raw, _ = tangents_normals_continuous(cl_samp)
    t_stab, ok_ang = stabilize_angles(t_raw, win_pts=ANGLE_SMOOTH_WIN_PTS, outlier_deg=ANGLE_OUTLIER_DEG)

    # 抽出ループ（縦/横 切替）
    half_thick = slice_thickness * 0.5
    keep_xyz = []
    keep_rgb = []
    kept = 0
    skipped_angle = 0
    skipped_sparse = 0
    used_vertical = 0
    used_horizontal = 0

    X = XY[:,0]
    Y = XY[:,1]

    for i in range(len(cl_samp)):
        if not ok_ang[i]:
            skipped_angle += 1
            continue

        c  = cl_samp[i]    # 中心（元座標）
        ti = t_stab[i]     # 接線（単位）

        # 接線角（x軸からの角度）を 0～90°に畳み込む
        ang = np.degrees(np.arctan2(abs(ti[1]), abs(ti[0])))  # 0～90

        if ang <= SWITCH_DEG:
            # 縦スライス（X方向の帯）：|x - cx| <= half_thick
            m = np.abs(X - c[0]) <= half_thick
            used_vertical += 1
        else:
            # 横スライス（Y方向の帯）：|y - cy| <= half_thick
            m = np.abs(Y - c[1]) <= half_thick
            used_horizontal += 1

        cnt = np.count_nonzero(m)
        if cnt < MIN_PTS_PER_SLICE:
            skipped_sparse += 1
            continue

        keep_xyz.append(pts[m])
        if has_rgb_in:
            keep_rgb.append(rgb[m])
        kept += 1

    if kept == 0:
        raise RuntimeError("全スライスが角度外れ/点不足でスキップされました。閾値やslice_thicknessを見直してください。")

    out_xyz = np.vstack(keep_xyz)
    if VOXEL_SIZE > 0:
        out_xyz = voxel_downsample(out_xyz, VOXEL_SIZE)  # ※ 色は省略
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
        print(f"  採用スライス: {kept} / 角度外れ: {skipped_angle} / 点不足: {skipped_sparse}")
        print(f"  切替内訳: 縦={used_vertical} / 横={used_horizontal}（閾値 {SWITCH_DEG}°）")
        print(f"  抽出点数: {N}（入力Z≤: {np.count_nonzero(zmask)}）")
        print(f"  RGB入出力: in={has_rgb_in}, out_rgb_dims={'yes' if {'red','green','blue'} <= out_dims else 'no'}")

if __name__ == "__main__":
    main()
