# -*- coding: utf-8 -*-
"""
【機能（M0：断面ごとの最大長方形“縁点”のみ出力）】
- 入力LASを読み込み、中心線に沿った帯（スライス）を生成
- 各スライスで v–z 断面を occupancy 化 → クロージング（morphology close）
- 自由空間から「最大内接長方形」を抽出し、その“縁セル中心”を GREEN 点として世界座標に配置
- すべてのスライスの GREEN 点を 1 つの LAS に保存（M0 のみ）
"""

import os
import math
import numpy as np
import laspy
import cv2

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0817no2_M0_rect_edges_only.las"

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
MORPH_RADIUS = 23           # [セル] クロージング構造要素半径
USE_ANCHOR_DOWNFILL = True  # 水面高さ近傍で down-fill を有効化
ANCHOR_Z = 1.50             # [m]
ANCHOR_TOL = 0.5            # [m]
MIN_RECT_SIZE = 5           # [セル] 長方形の最小 高さ/幅（両方以上）

# ==== 便利関数 ====
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

def resample_polyline(points_vz, ds=0.1):
    if len(points_vz) < 2: return points_vz
    pts = np.array(points_vz, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < 1e-9: return pts.tolist()
    n = max(2, int(np.ceil(L/ds))+1)
    si = np.linspace(0, L, n)
    out = []
    j = 0
    for t in si:
        while j+1 < len(s) and s[j+1] < t:
            j += 1
        if j+1 >= len(s): out.append(pts[-1]); continue
        r = (t - s[j]) / max(1e-9, s[j+1]-s[j])
        out.append((1-r)*pts[j] + r*pts[j+1])
    return [p.tolist() for p in out]

def order_corners_ccw(corners_vz, center_vz):
    c = np.asarray(center_vz)
    rel = np.asarray(corners_vz) - c
    ang = np.arctan2(rel[:,1], rel[:,0])
    idx = np.argsort(ang)
    return [corners_vz[i] for i in idx]

def downfill_on_closed(closed_uint8, z_min, grid_res, anchor_z, tol):
    """補間後の占有に対して、アンカー帯にヒットする列を下に埋める"""
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
        if not np.any(col): continue
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def find_max_rectangle(bitmap_bool: np.ndarray):
    """True=自由 の2D配列（行=Z, 列=V）から最大内接長方形(top, left, h, w)"""
    h, w = bitmap_bool.shape
    height = [0]*w
    best = (0, 0, 0, 0); max_area = 0
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

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol,
                        min_rect_size):
    """
    返り値：
      rect_edge_pts_vz: ビジュアル用縁点（v,z）
      bbox: (v_min, z_min, gw, gh)  ※M0ではメタ最小限
    """
    rect_edge_pts_vz = []

    if len(points_vz) == 0:
        return rect_edge_pts_vz, None

    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max - v_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid_raw = np.zeros((gh, gw), dtype=np.uint8)

    yi = ((points_vz[:,0] - v_min) / grid_res).astype(int)
    zi = ((points_vz[:,1] - z_min) / grid_res).astype(int)
    ok = (yi >= 0) & (yi < gw) & (zi >= 0) & (zi < gh)
    grid_raw[zi[ok], yi[ok]] = 255

    # closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed0 = cv2.morphologyEx(grid_raw, cv2.MORPH_CLOSE, kernel)

    # down-fill
    if use_anchor:
        closed = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol)
    else:
        closed = closed0

    closed_bool = (closed > 0)
    free_bitmap = ~closed_bool  # True=自由

    # 上方チェック：補間で天井が形成されていることを確認（任意の簡易基準）
    def has_points_above_after_interp(top, left, h, w):
        gh_, gw_ = closed_bool.shape
        z_above_start = top + h
        if z_above_start >= gh_: return False
        sub = closed_bool[z_above_start:gh_, left:left+w]
        return np.any(sub)

    free_work = free_bitmap.copy()
    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break
        if not has_points_above_after_interp(top, left, h, w):
            # 縁セルを点化
            for zi_ in range(top, top+h):
                for yi_ in range(left, left+w):
                    if zi_ in (top, top+h-1) or yi_ in (left, left+w-1):
                        v = v_min + (yi_ + 0.5) * grid_res
                        z = z_min + (zi_ + 0.5) * grid_res
                        rect_edge_pts_vz.append([v, z])
        # 充填済み領域を除外
        free_work[top:top+h, left:left+w] = False

    bbox = (v_min, z_min, gw, gh)
    return rect_edge_pts_vz, bbox

def vz_to_world_on_slice(vz, c, n_hat):
    """(v,z) -> 世界座標 (x,y,z)  ※u=0（帯の中心線上）"""
    v, z = vz
    p_xy = c + v * n_hat
    return [p_xy[0], p_xy[1], z]

def write_green_las(path, header_src, pts_xyz):
    header = copy_header_with_metadata(header_src)
    las_out = laspy.LasData(header)
    N = len(pts_xyz)
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)
    pts_xyz = np.asarray(pts_xyz, float)
    las_out.x = pts_xyz[:,0]; las_out.y = pts_xyz[:,1]; las_out.z = pts_xyz[:,2]
    if {"red","green","blue"} <= set(las_out.point_format.dimension_names):
        las_out.red   = np.zeros(N, dtype=np.uint16)
        las_out.green = np.full (N, 65535, dtype=np.uint16)
        las_out.blue  = np.zeros(N, dtype=np.uint16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    las_out.write(path)
    print(f"✅ 出力: {path}  点数: {N}")

# ========= メイン =========
def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)
    las = laspy.read(INPUT_LAS)

    # ndarray化
    X = np.asarray(las.x, float)
    Y = np.asarray(las.y, float)
    Z = np.asarray(las.z, float)
    xy  = np.column_stack([X, Y])

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
        if not np.any(under): continue
        idx = np.where(under)[0]
        left  = slab_xy[idx[0]]
        right = slab_xy[idx[-1]]
        c = 0.5*(left + right)
        through.append(c)
    if len(through) < 2:
        raise RuntimeError("中心線が作れません。UKCやBIN_Xを調整してください。")
    through = np.asarray(through, float)

    

    # --- 中心線を内挿 ---
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

    # --- 断面処理 → GREEN を収集（M0のみ） ---
    half_len = LINE_LENGTH * 0.5
    half_th  = SLICE_THICKNESS * 0.5
    GREEN = []

    for i in range(len(centers)-1):
        c  = centers[i]
        cn = centers[i+1]
        t_vec = cn - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9: continue
        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

        # 帯抽出: |u|<=half_th, |v|<=half_len
        dxy = xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat
        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_len)
        m_nav = m_band & (Z <= Z_MAX_FOR_NAV)
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLICE:
            continue

        points_vz = np.column_stack([v[m_nav], Z[m_nav]])

        # v–z occupancy → 長方形の“縁セル中心”のみ取得
        rect_edges_vz, _bbox = rectangles_on_slice(
            points_vz,
            grid_res=GRID_RES,
            morph_radius=MORPH_RADIUS,
            use_anchor=USE_ANCHOR_DOWNFILL,
            anchor_z=ANCHOR_Z,
            anchor_tol=ANCHOR_TOL,
            min_rect_size=MIN_RECT_SIZE
        )

        # 縁点を世界座標に
        for vv, zz in rect_edges_vz:
            GREEN.append(vz_to_world_on_slice([vv,zz], c, n_hat))

    if not GREEN:
        raise RuntimeError("M0：航行可能空間の長方形縁点が見つかりませんでした。パラメータを調整してください。")

    # === M0 出力（緑枠のみ） ===
    write_green_las(OUTPUT_LAS, las.header, GREEN)
    print("✅ M0 完了")
    print(f"  gap=50適用後 中心線点数: {len(through)}")
    print(f"  断面数（内挿）        : {len(centers)}")
    print(f"  M0基礎出力点         : {len(GREEN)}")

if __name__ == "__main__":
    main()
