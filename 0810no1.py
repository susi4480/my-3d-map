# -*- coding: utf-8 -*-
"""
【安定版：PCAで川軸に座標回転→Y'固定ビン中心線→法線帯抽出】
- Z ≤ Z_LIMIT でフィルタ
- XYのPCAで第1主成分を河川軸とみなし、座標を回転（Y'が流向）
- Y'固定ビンで X' 中央値 → 中心線（平滑化）
- 弧長 slice_interval で等間隔サンプル → 接線角の連続化・平滑化
- 各サンプル位置の法線方向 ±slice_thickness/2 の帯で抽出
- 外れ角・点不足スライスはスキップ
- 出力LASはCRS/VLR/EVLR/SRS継承（laspy 2.x）
"""

import os
import numpy as np
import laspy

# === 入出力 ===
INPUT_LAS  = r"C:\Users\user\Documents\lab\outcome\0725_suidoubasi_sita.las"
OUTPUT_LAS = r"C:\Users\user\Documents\lab\output_las\0810no1_slices_only_map_pca.las"

# === パラメータ（中心線推定&抽出）===
Z_LIMIT         = 2.0
BIN_Y           = 2.0
MIN_PTS_PER_BIN = 50
SMOOTH_WINDOW_M = 10.0

slice_thickness      = 0.2
slice_interval       = 0.5
MIN_PTS_PER_SLICE    = 80
ANGLE_SMOOTH_WIN_PTS = 5
ANGLE_OUTLIER_DEG    = 30.0

VOXEL_SIZE = 0.0
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
    ker = np.ones(win, dtype=float) / win
    return np.convolve(arr_pad, ker, mode="valid")

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
    if getattr(src_header, "vlrs", None):
        header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None):
        header.evlrs.extend(src_header.evlrs)
    return header

def voxel_downsample(points_xyz, voxel_size):
    if voxel_size <= 0:
        return points_xyz
    keys = np.floor(points_xyz / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points_xyz[np.sort(idx)]

# === PCAでXYの主軸（川軸）と回転行列を求める ===
def pca_rotation(xy):
    # 平行移動（原点は平均）
    mu = xy.mean(axis=0)
    X = xy - mu
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)  # 固有値昇順
    v1 = vecs[:, -1]  # 第1主成分
    # v1 を y' 方向に合わせたい（川軸=Y'）
    # 回転行列 R: [x'; y']^T = R * [x - mu_x; y - mu_y]
    # v1 = [vx, vy] を (0,1) に回したい ⇒ 角度 theta = atan2(v1_y, v1_x)
    theta = np.arctan2(v1[1], v1[0])
    # x' を川軸に直交させ、y' を川軸に一致させる回転（y'が流向）
    c, s = np.cos(-theta + np.pi/2), np.sin(-theta + np.pi/2)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    return mu, R  # x' y' = R * (xy - mu)

def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    las = laspy.read(INPUT_LAS)
    pts_all = np.vstack([las.x, las.y, las.z]).T

    in_dims = set(las.point_format.dimension_names)
    has_rgb_in = {"red","green","blue"} <= in_dims
    if has_rgb_in:
        rgb_all = np.vstack([las.red, las.green, las.blue]).T
    else:
        rgb_all = None

    # Zフィルタ
    zmask = pts_all[:,2] <= Z_LIMIT
    pts = pts_all[zmask]
    if pts.size == 0:
        raise RuntimeError("Z制限後に点がありません。Z_LIMITを見直してください。")

    # === PCAで川軸に座標回転（XYのみ） ===
    mu, R = pca_rotation(pts[:, :2])
    def to_rot(pxy):
        return (R @ (pxy - mu).T).T
    def from_rot(pxy_):
        return (R.T @ pxy_.T).T + mu

    XY_rot = to_rot(pts[:, :2])   # [x', y']
    pts_rot = np.column_stack([XY_rot, pts[:,2]])
    if has_rgb_in:
        rgb = rgb_all[zmask]

    # === 回転座標系で中心線（固定Y'ビンでX'中央値） ===
    y_min, y_max = XY_rot[:,1].min(), XY_rot[:,1].max()
    edges = np.arange(y_min, y_max + BIN_Y, BIN_Y)
    y_centers = 0.5*(edges[:-1] + edges[1:])
    Xc, Yc = [], []
    for i in range(len(edges)-1):
        y0, y1 = edges[i], edges[i+1]
        m = (XY_rot[:,1] >= y0) & (XY_rot[:,1] < y1)
        if np.count_nonzero(m) < MIN_PTS_PER_BIN:
            continue
        slab = XY_rot[m]
        Xc.append(np.median(slab[:,0]))
        Yc.append(y_centers[i])
    if len(Xc) < 2:
        raise RuntimeError("有効なY'ビンが不足（中心線を作成できません）。")
    Xc = np.array(Xc, float); Yc = np.array(Yc, float)
    order = np.argsort(Yc)
    Xc, Yc = Xc[order], Yc[order]
    Xc = moving_average_1d(Xc, SMOOTH_WINDOW_M, BIN_Y)
    centerline_rot = np.column_stack([Xc, Yc])  # （回転座標）

    # === 弧長等間隔サンプル＋平滑 ===
    cl_samp = resample_polyline_by_arclength(centerline_rot, slice_interval)
    cl_samp = smooth_polyline_xy(cl_samp, win_pts=5)

    # === 接線安定化（回転座標系で計算）===
    t_raw, _ = tangents_normals_continuous(cl_samp)
    t_stab, ok_ang = stabilize_angles(t_raw, win_pts=ANGLE_SMOOTH_WIN_PTS, outlier_deg=ANGLE_OUTLIER_DEG)

    # === 法線帯抽出（回転座標系で |s|<=half_thick ）===
    half_thick = slice_thickness * 0.5
    keep_xyz_rot = []
    keep_rgb = []
    kept = skipped_angle = skipped_sparse = 0

    XYr = XY_rot  # [x', y'] for all points after Z filter

    for i in range(len(cl_samp)):
        if not ok_ang[i]:
            skipped_angle += 1
            continue
        c  = cl_samp[i]      # 中心（回転座標）
        ti = t_stab[i]       # 接線（単位）

        dxy = XYr - c
        s = dxy @ ti         # 接線方向距離（回転座標）
        m = np.abs(s) <= half_thick
        cnt = np.count_nonzero(m)
        if cnt < MIN_PTS_PER_SLICE:
            skipped_sparse += 1
            continue

        keep_xyz_rot.append(pts_rot[m])
        if has_rgb_in:
            keep_rgb.append(rgb[m])
        kept += 1

    if kept == 0:
        raise RuntimeError("全スライスが外れ角/点不足でスキップされました。パラメータを見直してください。")

    out_rot = np.vstack(keep_xyz_rot)         # 回転座標の [x', y', z]
    out_xy  = from_rot(out_rot[:, :2])        # 元座標へ戻す
    out_xyz = np.column_stack([out_xy, out_rot[:,2]])

    if VOXEL_SIZE > 0:
        out_xyz = voxel_downsample(out_xyz, VOXEL_SIZE)
        # 簡易版のため色保持は省略（必要ならボクセル代表インデックスで色同伴）

    if has_rgb_in and VOXEL_SIZE == 0:
        out_rgb = np.vstack(keep_rgb)

    # === 出力 ===
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
        print(f"  PCA回転で川軸に整列 → Y'固定ビン: BIN_Y={BIN_Y}m")
        print(f"  Z_LIMIT={Z_LIMIT}m, MIN_PTS_PER_BIN={MIN_PTS_PER_BIN}, SMOOTH={SMOOTH_WINDOW_M}m")
        print(f"  slice_thickness=±{half_thick:.2f}m, slice_interval={slice_interval}m")
        print(f"  採用スライス: {kept} / 角度外れ: {skipped_angle} / 点不足: {skipped_sparse}")
        print(f"  抽出点数: {N}（入力Z≤: {np.count_nonzero(zmask)}）")
        print(f"  RGB入出力: in={has_rgb_in}, out={'yes' if {'red','green','blue'} <= out_dims else 'no'}")

if __name__ == "__main__":
    main()
