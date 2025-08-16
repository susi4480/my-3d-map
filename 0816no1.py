# -*- coding: utf-8 -*-
"""
中心線スライス＋航行可能長方形抽出（複数保持）＋最大長方形のみ隣と4隅接続
- 白色点群除外, Z ≤ 1.9, SORノイズ除去 を実施
- occupancy(closing + anchor-downfill) から自由空間を抽出
- スライス内では複数長方形を抽出して保持（青）
- ただし隣接接続は「最大面積の長方形」だけを4隅で接続（直方体ワイヤ：赤の輪郭＋緑の接続）
- さらに「長方形の下方向（低Z側）に占有点群がある」ことを航行可能条件に追加
- 出力: Open3D LineSet (.ply)
"""

import os
import math
import numpy as np
import laspy
import open3d as o3d
import cv2
from sklearn.neighbors import NearestNeighbors

# ========= 入出力 =========
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_PLY = r"/output/0816no1_maxrect_lineset_with_support.ply"
os.makedirs(os.path.dirname(OUTPUT_PLY) or ".", exist_ok=True)

# ========= スライス（中心線）設定 =========
UKC = -1.0                  # [m] 中心線抽出に使う水面下閾値
BIN_X = 2.0                 # [m] 中心線作成時のXビン幅
MIN_PTS_PER_XBIN = 50       # [点] 各Xビンの最小点数
GAP_DIST = 50.0             # [m] 中心線間引き距離
SECTION_INTERVAL = 0.5      # [m] 断面の間隔
LINE_LENGTH = 60.0          # [m] 法線方向の全長（±半分使用）
SLICE_THICKNESS = 0.20      # [m] 接線方向の厚み（u=±厚/2）
MIN_PTS_PER_SLICE = 80      # [点] スライスで処理する最小点数
Z_MAX_FOR_NAV = 1.9         # [m] 航行高さ上限

# ========= occupancy / 長方形抽出 =========
GRID_RES = 0.10             # [m/セル] v,z グリッド解像度
MORPH_RADIUS = 23           # [セル] closingカーネル半径
USE_ANCHOR_DOWNFILL = True  # アンカーdownfillを使う
ANCHOR_Z = 1.50             # [m]
ANCHOR_TOL = 0.50           # [m]
MIN_RECT_SIZE = 5           # [セル] h,wの最小サイズ（両方>=）

# ========= “下方向に点群がある” 条件 =========
REQUIRE_SUPPORT_BELOW = True     # サポート判定を有効化
SUPPORT_DEPTH_M = 0.5            # [m] 長方形底辺から下方向に見る深さ
SUPPORT_MIN_COL_RATIO = 0.10     # [0-1] 長方形幅方向の何割以上の列でサポートが必要か

# ========= 前処理（白色除外 + SOR） =========
EXCLUDE_WHITE = True
Z_LIMIT = 1.9            # [m] まず高さ制限
SOR_K = 8                # 近傍点数
SOR_STD = 1.0            # 閾値（平均+std*係数）

# ========= 可視化（色） =========
COLOR_MAX_OUTLINE = [1.0, 0.0, 0.0]  # 赤：最大長方形の輪郭
COLOR_CONNECT     = [0.0, 1.0, 0.0]  # 緑：隣スライス接続
COLOR_OTHER_RECT  = [0.0, 0.0, 1.0]  # 青：その他長方形

# ========= ユーティリティ =========
def l2(p, q): return math.hypot(q[0]-p[0], q[1]-p[1])

def order_corners_ccw(corners_vz, center_vz):
    c = np.asarray(center_vz)
    rel = np.asarray(corners_vz) - c
    ang = np.arctan2(rel[:,1], rel[:,0])
    idx = np.argsort(ang)
    return [corners_vz[i] for i in idx]

def vz_to_world_on_slice(vz, c, n_hat):
    v, z = vz
    p_xy = c + v * n_hat
    return [p_xy[0], p_xy[1], z]

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
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol,
                        min_rect_size, require_support_below, support_depth_m, support_min_col_ratio):
    """
    戻り値: rect_models = [
        {"center_vz": [v,z], "size_vw":[W,H], "corners_vz":[4つ], "area":W*H}, ... ]
    ※ require_support_below=True のとき、長方形直下（低z側）に占有が一定割合以上あるものだけ残す
    """
    rect_models = []
    if len(points_vz) == 0:
        return rect_models

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
    closed = downfill_on_closed(closed0, z_min, grid_res, anchor_z, anchor_tol) if use_anchor else closed0

    closed_bool = (closed > 0)          # True=占有
    free_bitmap = ~closed_bool          # True=自由

    support_rows = max(1, int(round(support_depth_m / grid_res)))

    def has_support_below(top, left, h, w):
        """行=Z(下が小z)，列=V。下方向=小z側=行インデックス[0 .. top-1]側。"""
        if not require_support_below:
            return True
        if top <= 0:
            return False
        col_hits = 0
        for j in range(left, left+w):
            lo = max(0, top - support_rows)
            hi = top  # non-inclusive of 'top'
            if lo < hi and np.any(closed_bool[lo:hi, j]):
                col_hits += 1
        # 長方形幅の一定割合以上でサポートあり
        return (col_hits / max(1, w)) >= support_min_col_ratio

    # 上方（=大z側）に占有があるか（航路の上側が閉じているか）のチェック（元コード準拠）
    def has_points_above_after_interp(top, left, h, w):
        z_above_start = top + h
        if z_above_start >= closed_bool.shape[0]: 
            return False
        sub = closed_bool[z_above_start:, left:left+w]
        return np.any(sub)

    # 長方形抽出（自由空間から最大長方形を貪欲に抜いていく）
    free_work = free_bitmap.copy()
    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break

        # 航行条件: 上に占有がある（天井）＆ 下方向に占有がある（サポート）
        if (not has_points_above_after_interp(top, left, h, w)) or (not has_support_below(top, left, h, w)):
            # 充填済み領域として除外（次の候補へ）
            free_work[top:top+h, left:left+w] = False
            continue

        # 長方形モデル化
        v0 = v_min + (left + 0.5) * grid_res
        z0 = z_min + (top  + 0.5) * grid_res
        W  = w * grid_res
        H  = h * grid_res
        corners = [
            [v0,   z0   ],
            [v0+W, z0   ],
            [v0+W, z0+H ],
            [v0,   z0+H ],
        ]
        center = [v0 + 0.5*W, z0 + 0.5*H]
        corners = order_corners_ccw(corners, center)
        rect_models.append({
            "center_vz": np.array(center, dtype=float),
            "size_vw":   np.array([W, H], dtype=float),
            "corners_vz": [np.array(c, dtype=float) for c in corners],
            "area": W * H
        })

        # 充填済み領域を除外して次へ
        free_work[top:top+h, left:left+w] = False

    return rect_models

def sor_filter(xyz, k=8, std_ratio=1.0):
    if len(xyz) < (k+1):
        return np.ones(len(xyz), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(xyz)
    dists, _ = nbrs.kneighbors(xyz)
    avg = dists[:,1:].mean(axis=1)
    mu, sd = avg.mean(), avg.std()
    return avg <= mu + std_ratio*sd

# ========= メイン =========
def main():
    # --- LAS読込 + 白色除外 + Z制限 + SOR ---
    las = laspy.read(INPUT_LAS)
    X = np.asarray(las.x, float)
    Y = np.asarray(las.y, float)
    Z = np.asarray(las.z, float)

    if EXCLUDE_WHITE and {"red","green","blue"} <= set(las.point_format.dimension_names):
        not_white = ~((las.red==65535) & (las.green==65535) & (las.blue==65535))
        X, Y, Z = X[not_white], Y[not_white], Z[not_white]

    mZ = (Z <= Z_LIMIT)
    X, Y, Z = X[mZ], Y[mZ], Z[mZ]

    xyz = np.column_stack([X, Y, Z])
    if len(xyz) == 0:
        raise RuntimeError("入力が空です（Z制限後）。")

    ok = sor_filter(xyz, k=SOR_K, std_ratio=SOR_STD)
    xyz = xyz[ok]
    if len(xyz) == 0:
        raise RuntimeError("SOR後に点が残りません。パラメータを調整してください。")

    X, Y, Z = xyz[:,0], xyz[:,1], xyz[:,2]
    xy = xyz[:, :2]

    # --- 中心線 ---
    x_min, x_max = xy[:,0].min(), xy[:,0].max()
    edges = np.arange(x_min, x_max + BIN_X, BIN_X)
    through = []
    for i in range(len(edges)-1):
        x0, x1 = edges[i], edges[i+1]
        m = (xy[:,0] >= x0) & (xy[:,0] < x1)
        if np.count_nonzero(m) < MIN_PTS_PER_XBIN:
            continue
        slab_xy = xy[m]; slab_z = Z[m]
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

    # 間引き
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

    # 内挿
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

    # --- スライス処理 ---
    half_len = LINE_LENGTH * 0.5
    half_th  = SLICE_THICKNESS * 0.5

    slice_all_rect_edges = []   # 全長方形の輪郭線（青）
    slice_max_rects_xyz  = []   # 最大長方形の4隅（XYZ）

    for i in range(len(centers)-1):
        c  = centers[i]
        cn = centers[i+1]
        t_vec = cn - c
        norm = np.linalg.norm(t_vec)
        if norm < 1e-9:
            continue
        t_hat = t_vec / norm
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)

        dxy = xy - c
        u = dxy @ t_hat
        v = dxy @ n_hat
        m_band = (np.abs(u) <= half_th) & (np.abs(v) <= half_len)
        m_nav  = m_band & (Z <= Z_MAX_FOR_NAV)
        if np.count_nonzero(m_nav) < MIN_PTS_PER_SLICE:
            continue

        points_vz = np.column_stack([v[m_nav], Z[m_nav]])

        rects = rectangles_on_slice(
            points_vz,
            grid_res=GRID_RES,
            morph_radius=MORPH_RADIUS,
            use_anchor=USE_ANCHOR_DOWNFILL,
            anchor_z=ANCHOR_Z,
            anchor_tol=ANCHOR_TOL,
            min_rect_size=MIN_RECT_SIZE,
            require_support_below=REQUIRE_SUPPORT_BELOW,
            support_depth_m=SUPPORT_DEPTH_M,
            support_min_col_ratio=SUPPORT_MIN_COL_RATIO
        )

        if not rects:
            continue

        # 全長方形の輪郭を保持（青）
        for r in rects:
            corners_vz = order_corners_ccw([p.copy() for p in r["corners_vz"]], r["center_vz"])
            corners_xyz = [vz_to_world_on_slice(vz, c, n_hat) for vz in corners_vz]
            for j in range(4):
                a = corners_xyz[j]
                b = corners_xyz[(j+1) % 4]
                slice_all_rect_edges.append((a, b))

        # 最大長方形の4隅（XYZ）
        rmax = max(rects, key=lambda rr: rr["area"])
        rmax_corners_vz = order_corners_ccw([p.copy() for p in rmax["corners_vz"]], rmax["center_vz"])
        rmax_corners_xyz = [vz_to_world_on_slice(vz, c, n_hat) for vz in rmax_corners_vz]
        slice_max_rects_xyz.append(rmax_corners_xyz)

    # --- LineSet 構築（青：その他、赤：最大輪郭、緑：接続） ---
    points = []
    lines  = []
    colors = []

    def add_line(p1, p2, color):
        i0 = len(points)
        points.extend([p1, p2])
        lines.append([i0, i0+1])
        colors.append(color)

    # その他長方形（青）
    for a, b in slice_all_rect_edges:
        add_line(a, b, COLOR_OTHER_RECT)

    # 最大長方形ワイヤと接続（赤＋緑）
    for i in range(len(slice_max_rects_xyz)-1):
        A = slice_max_rects_xyz[i]
        B = slice_max_rects_xyz[i+1]
        # Aの輪郭（赤）
        for j in range(4):
            add_line(A[j], A[(j+1) % 4], COLOR_MAX_OUTLINE)
        # Bの輪郭（赤）
        for j in range(4):
            add_line(B[j], B[(j+1) % 4], COLOR_MAX_OUTLINE)
        # 4隅接続（緑）
        for j in range(4):
            add_line(A[j], B[j], COLOR_CONNECT)

    if not lines:
        raise RuntimeError("出力するラインがありません。パラメータを見直してください。")

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, float)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, int))
    )
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    o3d.io.write_line_set(OUTPUT_PLY, line_set)
    print(f"✅ 出力: {OUTPUT_PLY}")
    print(f"  総ライン数: {len(lines)} / 総頂点数: {len(points)}")
    print(f"  最大長方形接続ペア数: {max(0, len(slice_max_rects_xyz)-1)}")

if __name__ == "__main__":
    main()
