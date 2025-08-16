# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import laspy
import cv2

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0810no10ver3_nav_rect_edges_gap50_zle1p9_anchor1p9_AFTER_CLOSING.las"

# ===== パラメータ（中心線・断面）=====
UKC = -1.0                  # [m] 左右岸抽出に使う水面下閾値（中心線用）
BIN_X = 2.0                 # [m] 中心線作成時の X ビン幅
MIN_PTS_PER_XBIN = 50       # 各 X ビンに必要な最小点数
GAP_DIST = 50.0             # [m] 中心線候補の間引き距離 (gap=50m)
SECTION_INTERVAL = 0.5      # [m] 断面（中心線内挿）間隔
LINE_LENGTH = 60.0          # [m] 法線方向の全長（±半分使う）
SLICE_THICKNESS = 0.20      # [m] 接線方向の薄さ（u=±厚/2）
MIN_PTS_PER_SLICE = 80      # [点] 各帯の最低点数

# ===== 航行可能空間に使う高さ制限 =====
Z_MAX_FOR_NAV = 1.9         # [m] ★この高さ以下の点だけで航行空間を判定

# ===== v–z 断面のoccupancy =====
GRID_RES = 0.10             # [m/セル] v,z 解像度
MORPH_RADIUS = 21           # [セル] クロージング構造要素半径
USE_ANCHOR_DOWNFILL = True  # 水面高さ近傍で down-fill を有効化
ANCHOR_Z = 1.90             # [m] ★アンカー高さ=1.9m（補間後の占有で判定）
ANCHOR_TOL = 0.45           # [m] 近傍幅（±）
MIN_RECT_SIZE = 5           # [セル] 長方形の最小 高さ/幅（両方以上）

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

def find_max_rectangle(bitmap_bool: np.ndarray):
    """
    True=自由 の2D配列（行=Z, 列=V）から最大内接長方形を返す。
    戻り：(top, left, h, w)
    """
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

# --- 補間後（クロージング後）を基準にした down-fill ---
def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    """
    closed_uint8: 0/255（クロージング後の占有）
    補間後の占有でアンカー帯[z=anchor±tol]にヒットする列だけ、
    列の最高占有までを下に埋める（不可化）。
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
        if not np.any(col):
            continue
        # アンカー帯に占有があれば down-fill 対象
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol,
                        min_rect_size):
    """
    points_vz: (N,2) with [v,z]  ※ここには z≤Z_MAX_FOR_NAV の点のみが渡ってくる前提
    手順：raw占有 → クロージング（補間） → （補間後を基準に）down-fill → 自由空間 → 最大長方形
    “上方に占有があるか”のチェックも補間後の占有で判定。
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

    # 上方チェック（補間後の占有）
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

# ===== メイン =====
def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    las = laspy.read(INPUT_LAS)

    # ndarrayに明示変換（ScaledArrayView対策）
    X = np.asarray(las.x, float)
    Y = np.asarray(las.y, float)
    Z = np.asarray(las.z, float)
    xyz = np.column_stack([X, Y, Z])
    xy  = xyz[:, :2]

    # --- 1) Xビンごとに左右岸→中点（UKCで水面下を検出） ---
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (xy[:,0] >= x0) & (xy[:,0] < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy = xy[m]
        slab_z  = Z[m]
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

    # --- 2) gap=50mで間引き ---
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    # --- 3) 中心線を内挿（断面中心列） ---
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

    # --- 4) 各断面で“薄い帯”を抽出 → z≤1.9m だけで v–z 矩形化 → 縁のみ作成 ---
    half_len = LINE_LENGTH * 0.5
    half_th  = SLICE_THICKNESS * 0.5
    GREEN = []

    for i in range(len(centers)-1):
        c  = centers[i]
        cn = centers[i+1]
        t_vec = cn - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9:
            continue
        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

        # 帯抽出: |u|<=half_th, |v|<=half_len
        dxy = xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat
        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_len)

        # ★ 高さ制限（z≤1.9m）をここで適用
        m_nav = m_band & (Z <= Z_MAX_FOR_NAV)
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLICE:
            continue

        points_vz = np.column_stack([v[m_nav], Z[m_nav]])

        # 断面occupancy：raw→closing→（closing基準で）down-fill→自由空間→長方形
        rect_edges_vz = rectangles_on_slice(
            points_vz,
            grid_res=GRID_RES,
            morph_radius=MORPH_RADIUS,
            use_anchor=USE_ANCHOR_DOWNFILL,
            anchor_z=ANCHOR_Z,         # ★補間後の占有でアンカー判定
            anchor_tol=ANCHOR_TOL,
            min_rect_size=MIN_RECT_SIZE
        )

        # u=0 上に配置して世界座標へ
        for v_i, z_i in rect_edges_vz:
            p_xy = c + v_i * n_hat  # (u=0, v=v_i)
            GREEN.append([p_xy[0], p_xy[1], z_i])

    if not GREEN:
        raise RuntimeError("航行可能空間の長方形が見つかりませんでした。パラメータを調整してください。")

    GREEN = np.asarray(GREEN, dtype=float)

    # --- 5) LAS出力：長方形の縁のみ（元点群は出さない） ---
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = GREEN.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = GREEN[:,0]
    las_out.y = GREEN[:,1]
    las_out.z = GREEN[:,2]

    # RGBが使えるフォーマットなら緑で出す
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full (N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)

    las_out.write(OUTPUT_LAS)

    print("✅ 出力:", OUTPUT_LAS)
    print(f"  gap=50適用後 中心線点数: {len(through)}")
    print(f"  断面数（内挿）        : {len(centers)}")
    print(f"  出力点（長方形の縁） : {N}")

if __name__ == "__main__":
    main()
