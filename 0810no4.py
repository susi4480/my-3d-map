# -*- coding: utf-8 -*-
"""
【統合版】PCAで川軸整列＋中心線サンプルごとの“法線×高さ”断面で
           occupancy → down-fill → クロージング → 最大内接長方形を抽出。
           長方形より上に元の占有があれば不採用。緑点を集約してLAS出力。

座標系と帯の定義（回転座標で処理）:
- XYをPCAで回転し、Y' ≈ 流向
- 中心線をY'固定ビンのX'中央値で作成
- 中心線の各サンプル c において
    u = 接線方向（t）   … スライスの薄さ方向   → |u| ≤ slice_thickness/2 で抽出
    v = 法線方向（n）   … ビットマップの横軸
    z = 高さ            … ビットマップの縦軸
- 最大長方形は (v,z) 平面上で求め、その縁セル中心を u=0 上に配置して3Dへ戻す
"""

import os
import numpy as np
import laspy
import cv2

# === 入出力 ===
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS = r"/output/0810no4_pca_slice_rects.las"

# === グローバル設定 ===
Z_LIMIT         = 2.0       # [m] 使用する高さ上限
BIN_Y           = 2.0       # [m] 中心線作成のY'ビン幅
MIN_PTS_PER_BIN = 50        #    同上: ビン内の最小点数
SMOOTH_WINDOW_M = 10.0      # [m] 中心線X'移動平均窓（0で無効）

slice_thickness      = 0.20 # [m] 接線方向の帯の厚み（±half）
slice_interval       = 0.50 # [m] 中心線の弧長サンプル間隔
MIN_PTS_PER_SLICE    = 80   #    スライス内の必要最小点数
ANGLE_SMOOTH_WIN_PTS = 5    #    接線角平滑化窓
ANGLE_OUTLIER_DEG    = 30.0 # [deg] 角度外れスキップ閾値

# 断面ビットマップ（v-z）のパラメータ
GRID_RES         = 0.10     # [m/セル]
MORPH_RADIUS     = 21       # [セル] クロージング構造要素半径
USE_ANCHOR_DOWNFILL = True
ANCHOR_Z         = 0.35     # [m] 水面近傍
ANCHOR_TOL       = 0.15     # [m] 近傍幅（±）
USE_MORPH_DIFF   = True     # クロージング差分セルも緑点化
MIN_RECT_SIZE    = 5        # [セル] 長方形の最小高さ/幅（両方満たす）
INCLUDE_ORIGINAL_POINTS = False  # Trueで元点群(Z≤)も一緒に出力

VERBOSE = True

# === ユーティリティ ===
def moving_average_1d(arr, win_m, bin_m):
    if win_m <= 0 or len(arr) < 2:
        return arr
    win = max(1, int(round(win_m / bin_m)))
    if win % 2 == 0:
        win += 1
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
        p = xy[j]*(1-t) + xy[j+1]*t
        out.append(p)
    return np.asarray(out)

def smooth_polyline_xy(xy, win_pts=5):
    if win_pts <= 1 or len(xy) < 3: return xy
    if win_pts % 2 == 0: win_pts += 1
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
        if i>0 and np.dot(t[i], t[i-1]) < 0:  # 連続性（180°超で反転）
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

def voxel_downsample(points_xyz, voxel_size):
    if voxel_size <= 0: return points_xyz
    keys = np.floor(points_xyz / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points_xyz[np.sort(idx)]

def pca_rotation(xy):
    mu = xy.mean(axis=0)
    X = xy - mu
    C = np.cov(X.T)
    vals, vecs = np.linalg.eigh(C)
    v1 = vecs[:, -1]
    theta = np.arctan2(v1[1], v1[0])
    c, s = np.cos(-theta + np.pi/2), np.sin(-theta + np.pi/2)
    R = np.array([[c, -s],[s, c]], dtype=float)
    return mu, R

# === 断面（v-z）での矩形抽出ヘルパ ===
def find_max_rectangle(bitmap_bool: np.ndarray):
    """自由=Trueのbitmapに対する最大長方形（ヒストグラムDP）"""
    h, w = bitmap_bool.shape
    height = [0] * w
    max_area = 0
    max_rect = (0, 0, 0, 0)  # (top, left, h, w)
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
                area = height[top_idx] * width
                if area > max_area:
                    max_area = area
                    max_rect = (i - height[top_idx] + 1,
                                (stack[-1] + 1 if stack else 0),
                                height[top_idx],
                                width)
    return max_rect

def downfill_only_near_anchor(grid_uint8, z_min, grid_res, anchor_z, tol):
    occ = (grid_uint8 > 0)
    gh, gw = occ.shape
    i_anchor = int(round((anchor_z - z_min) / grid_res))
    pad = max(0, int(np.ceil(tol / grid_res)))
    i_lo = max(0, i_anchor - pad)
    i_hi = min(gh - 1, i_anchor + pad)
    if i_lo > gh - 1 or i_hi < 0:
        return (occ.astype(np.uint8) * 255)
    for j in range(gw):
        col = occ[:, j]
        idx = np.where(col)[0]
        if idx.size == 0: continue
        if np.any((idx >= i_lo) & (idx <= i_hi)):
            imax = idx.max()
            col[:imax + 1] = True
            occ[:, j] = col
    return (occ.astype(np.uint8) * 255)

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol,
                        use_morph_diff, min_rect_size):
    """
    points_vz: (N,2) array with columns [v,z]
    return: list of rectangles' edge points in (v,z), and optional diff points
    """
    if len(points_vz) == 0:
        return [], [], None, None, None

    v_min, v_max = points_vz[:,0].min(), points_vz[:,0].max()
    z_min, z_max = points_vz[:,1].min(), points_vz[:,1].max()
    gw = max(1, int(np.ceil((v_max - v_min) / grid_res)))
    gh = max(1, int(np.ceil((z_max - z_min) / grid_res)))
    grid = np.zeros((gh, gw), dtype=np.uint8)

    # 占有セット
    for v, z in points_vz:
        yi = int((v - v_min) / grid_res)  # 横=v
        zi = int((z - z_min) / grid_res)  # 縦=z
        if 0 <= zi < gh and 0 <= yi < gw:
            grid[zi, yi] = 255

    grid_raw = grid.copy()

    # down-fill（任意）
    if use_anchor:
        grid = downfill_only_near_anchor(grid, z_min, grid_res, anchor_z, anchor_tol)

    # クロージング
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*morph_radius+1, 2*morph_radius+1))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

    morph_diff_bitmap = (closed > 0) & (grid == 0)
    free_bitmap = (closed == 0)

    # 長方形詰め（上方チェック＝補間前の占有を使用）
    rect_edge_pts_vz = []
    free_work = free_bitmap.copy()
    grid_raw_bool = (grid_raw > 0)

    def has_raw_points_above(top, left, h, w):
        gh, gw = grid_raw_bool.shape
        z_above_start = top + h
        if z_above_start >= gh: return False
        sub = grid_raw_bool[z_above_start:gh, left:left+w]
        return np.any(sub)

    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break
        if not has_raw_points_above(top, left, h, w):
            # 縁セル→ (v,z) へ
            for zi in range(top, top+h):
                for yi in range(left, left+w):
                    if zi in (top, top+h-1) or yi in (left, left+w-1):
                        v = v_min + (yi + 0.5) * grid_res
                        z = z_min + (zi + 0.5) * grid_res
                        rect_edge_pts_vz.append([v, z])
        # 領域除外（採否に関わらず）
        free_work[top:top+h, left:left+w] = False

    diff_pts_vz = []
    if use_morph_diff:
        zi_idx, yi_idx = np.where(morph_diff_bitmap)
        for zi, yi in zip(zi_idx, yi_idx):
            v = v_min + (yi + 0.5) * grid_res
            z = z_min + (zi + 0.5) * grid_res
            diff_pts_vz.append([v, z])

    return rect_edge_pts_vz, diff_pts_vz, v_min, z_min, grid.shape

def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS) or ".", exist_ok=True)

    las = laspy.read(INPUT_LAS)
    pts_all = np.vstack([las.x, las.y, las.z]).T
    zmask = pts_all[:,2] <= Z_LIMIT
    pts = pts_all[zmask]
    if pts.size == 0:
        raise RuntimeError("Z制限後に点がありません。Z_LIMITを見直してください。")

    in_dims = set(las.point_format.dimension_names)
    has_rgb_in = {"red","green","blue"} <= in_dims
    if has_rgb_in:
        rgb_all = np.vstack([las.red, las.green, las.blue]).T
        rgb = rgb_all[zmask]

    # PCA回転
    mu, R = pca_rotation(pts[:, :2])
    def to_rot(pxy):   return (R @ (pxy - mu).T).T
    def from_rot(pxy_):return (R.T @ pxy_.T).T + mu

    XYr = to_rot(pts[:, :2])                 # [x', y']
    pts_rot = np.column_stack([XYr, pts[:,2]])

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

    # 弧長等間隔＋接線安定化
    cl_samp = resample_polyline_by_arclength(centerline_rot, slice_interval)
    cl_samp = smooth_polyline_xy(cl_samp, win_pts=5)
    t_raw, _ = tangents_normals_continuous(cl_samp)
    t_stab, ok_ang = stabilize_angles(t_raw, win_pts=ANGLE_SMOOTH_WIN_PTS, outlier_deg=ANGLE_OUTLIER_DEG)

    half_thick = slice_thickness * 0.5
    GREEN_all = []

    kept = skipped_angle = skipped_sparse = 0

    # スライスごとに (v,z) 断面を構築して長方形→緑点へ
    for i in range(len(cl_samp)):
        if not ok_ang[i]:
            skipped_angle += 1
            continue

        c  = cl_samp[i]      # 回転座標の中心 [x', y']
        ti = t_stab[i]       # 接線（単位, 回転座標）
        ni = np.array([-ti[1], ti[0]])  # 法線（単位）

        # 接線方向距離 u = (XY' - c)·t
        dxy = XYr - c
        u = dxy @ ti
        m = np.abs(u) <= half_thick
        if np.count_nonzero(m) < MIN_PTS_PER_SLICE:
            skipped_sparse += 1
            continue

        # このスライスの点（v,z）に投影
        XYr_s = XYr[m]
        z_s   = pts_rot[m, 2]
        v = (XYr_s - c) @ ni
        points_vz = np.column_stack([v, z_s])

        # 断面ビットマップ→down-fill→クロージング→最大長方形（上方チェック付き）
        rect_edge_vz, diff_vz, vmin, zmin, shape = rectangles_on_slice(
            points_vz,
            grid_res=GRID_RES,
            morph_radius=MORPH_RADIUS,
            use_anchor=USE_ANCHOR_DOWNFILL,
            anchor_z=ANCHOR_Z,
            anchor_tol=ANCHOR_TOL,
            use_morph_diff=USE_MORPH_DIFF,
            min_rect_size=MIN_RECT_SIZE
        )

        # 矩形縁 + 差分セル を u=0 に配置して回転座標→元座標へ
        for (v_i, z_i) in rect_edge_vz:
            XYr_edge = c + 0.0*ti + v_i*ni  # u=0 上
            XY_world = from_rot(XYr_edge)
            GREEN_all.append([XY_world[0], XY_world[1], z_i])

        for (v_i, z_i) in diff_vz:
            XYr_edge = c + 0.0*ti + v_i*ni
            XY_world = from_rot(XYr_edge)
            GREEN_all.append([XY_world[0], XY_world[1], z_i])

        kept += 1

    if len(GREEN_all) == 0:
        raise RuntimeError("緑点が生成されませんでした。パラメータ（thickness, GRID_RES, MIN_*）を調整してください。")

    GREEN_all = np.asarray(GREEN_all, dtype=float)

    # 出力点群の構築
    if INCLUDE_ORIGINAL_POINTS:
        out_pts = np.vstack([pts, GREEN_all])
        if has_rgb_in:
            green_rgb = np.tile(np.array([[0, 65535, 0]], dtype=np.uint16), (len(GREEN_all),1))
            rgb_out = np.vstack([rgb, green_rgb])
    else:
        out_pts = GREEN_all
        if has_rgb_in:
            rgb_out = np.tile(np.array([[0, 65535, 0]], dtype=np.uint16), (len(GREEN_all),1))

    # LAS保存（ヘッダ継承）
    header = copy_header_with_metadata(las.header)
    las_out = laspy.LasData(header)
    N = out_pts.shape[0]
    try:
        las_out.points = laspy.ScaleAwarePointRecord.zeros(N, header=header)
    except AttributeError:
        las_out.points = laspy.PointRecord.zeros(N, header=header)

    las_out.x = out_pts[:,0]
    las_out.y = out_pts[:,1]
    las_out.z = out_pts[:,2]

    out_dims = set(las_out.point_format.dimension_names)
    if has_rgb_in and {"red","green","blue"} <= out_dims:
        if INCLUDE_ORIGINAL_POINTS:
            las_out.red, las_out.green, las_out.blue = rgb_out[:,0], rgb_out[:,1], rgb_out[:,2]
        else:
            las_out.red   = rgb_out[:,0]
            las_out.green = rgb_out[:,1]
            las_out.blue  = rgb_out[:,2]

    las_out.write(OUTPUT_LAS)

    if VERBOSE:
        print(f"✅ 出力: {OUTPUT_LAS}")
        print(f"  PCA回転: Y'≈流向 / 中心線ビン幅 BIN_Y={BIN_Y} m / Z_LIMIT={Z_LIMIT} m")
        print(f"  スライス: thickness=±{half_thick:.2f} m, interval={slice_interval} m")
        print(f"  採用スライス: {kept} / 角度外れ: {skipped_angle} / 点不足: {skipped_sparse}")
        print(f"  生成緑点: {len(GREEN_all):,d}")
        print(f"  断面設定: GRID_RES={GRID_RES} m, MORPH_RADIUS={MORPH_RADIUS}, MIN_RECT_SIZE={MIN_RECT_SIZE} cells")
        print(f"  down-fill: {USE_ANCHOR_DOWNFILL} @ {ANCHOR_Z}±{ANCHOR_TOL} m")
        if INCLUDE_ORIGINAL_POINTS:
            print("  出力に元点群(Z≤)を含めています。")

if __name__ == "__main__":
    main()
