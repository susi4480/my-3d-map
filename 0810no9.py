# -*- coding: utf-8 -*-
"""
【長方形の縁のみ出力版（アンカー＝補間後基準）】
PCAで川軸整列 → 中心線サンプルごとに“法線×高さ(v-z)”断面を生成 →
raw占有 → クロージング（補間） → （補間後を基準に）down-fill → 自由空間 →
最大内接長方形のみ抽出 → 縁セル中心を u=0 上に配置して世界座標へ → LAS出力（緑）。
元点群は出力しない。
"""

import os
import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0810no9ver2_rect_edges_anchor_after_closing.las"
os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

# === パラメータ ===
Z_LIMIT         = 2.0           # 入力をまずこの高さ以下に制限
BIN_Y           = 2.0
MIN_PTS_PER_BIN = 50
SMOOTH_WINDOW_M = 10.0
MIN_RECT_SIZE   = 5             # [セル] 長方形の最小 h/w

slice_thickness      = 0.20     # [m] ±half
slice_interval       = 0.50     # [m]
MIN_PTS_PER_SLICE    = 80
ANGLE_SMOOTH_WIN_PTS = 5
ANGLE_OUTLIER_DEG    = 30.0

GRID_RES         = 0.10         # [m/セル] v,z
MORPH_RADIUS     = 21           # [セル] クロージング
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z         = 1.9          # ★補間後の占有でこの高さ±tolに占有がある列のみ down-fill
ANCHOR_TOL       = 0.45
Z_MAX_FOR_NAV    = 1.9          # ★航行判定に使う z 上限（スライス側で z≤ を適用）
VERBOSE = True

# === ユーティリティ ===
def moving_average_1d(arr, win_m, bin_m):
    if win_m <= 0 or len(arr) < 2: return arr
    win = max(1, int(round(win_m / bin_m)))
    if win % 2 == 0: win += 1
    pad = win // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    ker = np.ones(win)/win
    return np.convolve(arr_pad, ker, mode="valid")

def resample_polyline_by_arclength(xy, step):
    seg = np.diff(xy, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    L = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(L[-1])
    if total <= 0: return xy.copy()
    targets = np.arange(0.0, total + 1e-9, step)
    out, j = [], 0
    for s in targets:
        while j+1 < len(L) and L[j+1] < s: j += 1
        if j+1 >= len(L): out.append(xy[-1]); break
        t = (s - L[j]) / max(L[j+1]-L[j], 1e-12)
        out.append(xy[j]*(1-t) + xy[j+1]*t)
    return np.asarray(out)

def smooth_polyline_xy(xy, win_pts=5):
    if win_pts <= 1 or len(xy) < 3: return xy
    if win_pts % 2 == 0: win_pts += 1
    pad = win_pts // 2
    ker = np.ones(win_pts)/win_pts
    xs = np.convolve(np.pad(xy[:,0], (pad,pad), mode="edge"), ker, mode="valid")
    ys = np.convolve(np.pad(xy[:,1], (pad,pad), mode="edge"), ker, mode="valid")
    return np.column_stack([xs, ys])

def tangents_normals_continuous(xy):
    n = xy.shape[0]
    t = np.zeros((n,2), float)
    if n >= 3: t[1:-1] = xy[2:] - xy[:-2]
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
    if win_pts % 2 == 0: win_pts += 1
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
    if getattr(src_header, "srs", None): header.srs = src_header.srs
    if getattr(src_header, "vlrs", None): header.vlrs.extend(src_header.vlrs)
    if getattr(src_header, "evlrs", None): header.evlrs.extend(src_header.evlrs)
    return header

def pca_rotation(xy):
    mu = xy.mean(axis=0)
    X = xy - mu
    C = np.cov(X.T)
    _, vecs = np.linalg.eigh(C)
    v1 = vecs[:, -1]
    theta = np.arctan2(v1[1], v1[0])
    c, s = np.cos(-theta + np.pi/2), np.sin(-theta + np.pi/2)
    R = np.array([[c, -s],[s, c]], dtype=float)
    return mu, R

def find_max_rectangle(bitmap_bool: np.ndarray):
    h, w = bitmap_bool.shape
    height = [0]*w
    best = (0, 0, 0, 0)
    max_area = 0
    for i in range(h):
        for j in range(w):
            height[j] = height[j] + 1 if bitmap_bool[i, j] else 0
        stack = []; j = 0
        while j <= w:
            cur = height[j] if j < w else 0
            if not stack or cur >= height[stack[-1]]:
                stack.append(j); j += 1
            else:
                top_idx = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area  = height[top_idx]*width
                if area > max_area:
                    max_area = area
                    top  = i - height[top_idx] + 1
                    left = (stack[-1] + 1) if stack else 0
                    best = (top, left, height[top_idx], width)
    return best

# --- 補間後を基準にした down-fill（列ごと） ---
def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    """
    closed_uint8: 0/255（クロージング後）
    補間後の占有でアンカー帯にヒットする列だけ、列の最上段までを下方向に埋める。
    """
    closed_bool = (closed_uint8 > 0)
    gh, gw = closed_bool.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh - 1, i_anchor + pad)

    if i_lo > gh - 1 or i_hi < 0:
        return (closed_bool.astype(np.uint8) * 255)

    out = closed_bool.copy()
    for j in range(gw):
        col = closed_bool[:, j]
        if not np.any(col):  # そもそも占有無し
            continue
        # アンカー帯に占有がある列だけ down-fill
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol,
                        min_rect_size):
    """
    points_vz: (N,2) with [v,z]  ※ここには z≤Z_MAX_FOR_NAV の点のみが渡ってくる
    手順：raw占有 → クロージング → （補間後を基準に）down-fill → 自由空間 → 長方形
    上方チェックも補間後の占有で実施。
    """
    if len(points_vz) == 0:
        return []

    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max - v_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid_raw = np.zeros((gh, gw), dtype=np.uint8)

    # raw占有
    yi = ((points_vz[:,0] - v_min) / grid_res).astype(int)
    zi = ((points_vz[:,1] - z_min) / grid_res).astype(int)
    ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh)
    grid_raw[zi[ok], yi[ok]] = 255

    # クロージング（補間）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed0 = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)

    # （補間後を基準に）アンカー down-fill
    if use_anchor:
        closed = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol)
    else:
        closed = closed0

    closed_bool = (closed > 0)
    free_bitmap = ~closed_bool  # True=自由

    # “上方”チェック：補間後の占有で評価
    def has_points_above_after_interp(top, left, h, w):
        gh_, gw_ = closed_bool.shape
        z_above_start = top + h
        if z_above_start >= gh_: return False
        sub = closed_bool[z_above_start:gh_, left:left+w]
        return np.any(sub)

    # 自由空間に最大長方形を貪欲詰め
    rect_edge_pts_vz = []
    free_work = free_bitmap.copy()
    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break
        if not has_points_above_after_interp(top, left, h, w):
            # 縁セル中心 → (v,z)
            for zi in range(top, top+h):
                for yi_ in range(left, left+w):
                    if zi in (top, top+h-1) or yi_ in (left, left+w-1):
                        v = v_min + (yi_ + 0.5) * grid_res
                        z = z_min + (zi + 0.5) * grid_res
                        rect_edge_pts_vz.append([v, z])
        # 領域除外
        free_work[top:top+h, left:left+w] = False

    return rect_edge_pts_vz

# === メイン ===
def main():
    las = laspy.read(INPUT_LAS)

    # ndarray化（ScaledArrayView対策）
    X = np.asarray(las.x, dtype=float)
    Y = np.asarray(las.y, dtype=float)
    Z = np.asarray(las.z, dtype=float)

    # Z制限
    zmask = (Z <= Z_LIMIT)
    X = X[zmask]; Y = Y[zmask]; Z = Z[zmask]
    pts = np.column_stack([X, Y, Z])
    if len(pts) == 0:
        raise RuntimeError("Z制限後に点がありません。")

    # PCA回転
    mu, R = pca_rotation(pts[:, :2])
    to_rot  = lambda pxy: (R @ (pxy - mu).T).T
    from_rot= lambda pxy_: (R.T @ pxy_.T).T + mu

    XYr = to_rot(pts[:, :2])
    pts_rot = np.column_stack([XYr, Z])

    # 中心線（Y'固定ビンでX'中央値）
    y_min, y_max = XYr[:,1].min(), XYr[:,1].max()
    edges = np.arange(y_min, y_max + BIN_Y, BIN_Y)
    y_centers = 0.5*(edges[:-1] + edges[1:])
    Xc, Yc = [], []
    for i in range(len(edges)-1):
        y0, y1 = edges[i], edges[i+1]
        m = (XYr[:,1] >= y0) & (XYr[:,1] < y1)
        if np.count_nonzero(m) < MIN_PTS_PER_BIN: continue
        slab = XYr[m]
        Xc.append(np.median(slab[:,0]))
        Yc.append(y_centers[i])
    if len(Xc) < 2:
        raise RuntimeError("有効なY'ビンが不足（中心線を作成できません）。")
    Xc = np.array(Xc, float); Yc = np.array(Yc, float)
    order = np.argsort(Yc)
    Xc, Yc = Xc[order], Yc[order]
    Xc = moving_average_1d(Xc, SMOOTH_WINDOW_M, BIN_Y)
    centerline_rot = np.column_stack([Xc, Yc])

    # サンプリング＋接線安定化
    cl_samp = resample_polyline_by_arclength(centerline_rot, slice_interval)
    cl_samp = smooth_polyline_xy(cl_samp, win_pts=5)
    t_raw, _ = tangents_normals_continuous(cl_samp)
    t_stab, ok_ang = stabilize_angles(t_raw, win_pts=ANGLE_SMOOTH_WIN_PTS, outlier_deg=ANGLE_OUTLIER_DEG)

    half_thick = slice_thickness * 0.5
    GREEN_all = []
    kept = skipped_angle = skipped_sparse = 0

    # 各スライスで v-z 矩形の縁のみ抽出（z≤Z_MAX_FOR_NAV を適用）
    for i in range(len(cl_samp)):
        if not ok_ang[i]:
            skipped_angle += 1
            continue

        c  = cl_samp[i]
        ti = t_stab[i]
        ni = np.array([-ti[1], ti[0]])

        dxy = XYr - c
        u = dxy @ ti
        v = dxy @ ni
        m = (np.abs(u) <= half_thick)  # 薄い帯
        if np.count_nonzero(m) < MIN_PTS_PER_SLICE:
            skipped_sparse += 1
            continue

        # ★ 航行判定に使う高さ制限
        z_s = pts_rot[m, 2]
        v_s = v[m]
        m_nav = z_s <= Z_MAX_FOR_NAV
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLICE:
            skipped_sparse += 1
            continue

        points_vz = np.column_stack([v_s[m_nav], z_s[m_nav]])

        rect_edge_vz = rectangles_on_slice(
            points_vz,
            grid_res=GRID_RES,
            morph_radius=MORPH_RADIUS,
            use_anchor=USE_ANCHOR_DOWNFILL,
            anchor_z=ANCHOR_Z,        # ★補間後の占有でアンカー判定
            anchor_tol=ANCHOR_TOL,
            min_rect_size=MIN_RECT_SIZE
        )

        # u=0 上に配置して世界座標へ
        for (v_i, z_i) in rect_edge_vz:
            XYr_edge = c + v_i*ni
            XY_world = from_rot(XYr_edge)
            GREEN_all.append([XY_world[0], XY_world[1], z_i])

        kept += 1

    if len(GREEN_all) == 0:
        raise RuntimeError("長方形の縁点が生成されません（パラメータを再調整）。")

    GREEN_all = np.asarray(GREEN_all, dtype=float)

    # 出力（縁のみ）
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = GREEN_all.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = GREEN_all[:,0]
    las_out.y = GREEN_all[:,1]
    las_out.z = GREEN_all[:,2]

    # RGBが使えるフォーマットなら緑で出力
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full (N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)

    las_out.write(OUTPUT_LAS)

    if VERBOSE:
        print(f"✅ 出力: {OUTPUT_LAS}")
        print(f"  生成点数(縁のみ): {N:,d}")
        print(f"  採用スライス: {kept} / 角度外れ: {skipped_angle} / 点不足: {skipped_sparse}")
        print(f"  params: GRID_RES={GRID_RES}, MORPH_RADIUS={MORPH_RADIUS}, MIN_RECT_SIZE={MIN_RECT_SIZE}")
        print(f"  anchor(after-closing): z={ANCHOR_Z}±{ANCHOR_TOL}m, Z_MAX_FOR_NAV={Z_MAX_FOR_NAV}")

if __name__ == "__main__":
    main()
