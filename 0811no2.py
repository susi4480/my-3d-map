# -*- coding: utf-8 -*-
"""
【機能（スライス間接続 6手法フル版）】
1) 既存の中心線→帯抽出→v–zビットマップ→最大長方形（縁点：GREEN）を生成
2) スライスごとに以下を保持
   - free_bitmap（True=自由）
   - union_raster_from_rects（長方形群OR）
   - left/right 壁ポリライン（v–z）
   - 長方形モデル（center、size、4隅）
   - スライス座標系（c, n_hat, v_min, z_min, grid_res）
3) 6手法でスライス間接続を実施し、それぞれ緑点LASで出力
   - M1: 外周ユニオン法（OR→外周→隣接ポリライン対応→ロフト）
   - M2: 左右壁ポリライン法（左↔左・右↔右のみ対応→ロフト）
   - M3: 長方形辺クラスタ法（辺をクラスタ→代表線対応→ロフト）
   - M4: 長方形中心点マッチング法（中心＋サイズ類似→4隅ロフト）
   - M5: 3D占有ボクセル接続（自由空間を3D連結→境界セル中心出力）
   - M6: 緑枠LASを再ラスタ→M1で接続
"""

import os
import math
import numpy as np
import laspy
import cv2

# ===== 入出力 =====
INPUT_LAS  = r"/data/0731_suidoubasi_ue.las"
OUTPUT_LAS_BASE = r"/output/0810no10ver2_nav_rect_edges_gap50_zle1p9_anchor1p9_AFTER_CLOSING"

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

# ===== スライス間接続（マッチング＆ロフトの共通） =====
# M1 外周ユニオン
CONTOUR_RESAMPLE_DS = 0.10     # [m] 外周の再サンプリング間隔
PAIR_MAX_CENTROID_DIST = 1.0   # [m] ポリライン対応：重心距離閾値

# M2 左右壁
WALL_BREAK_GAP = 2             # [セル] 断線とみなす最小ギャップ（現状は未使用）
WALL_MATCH_MAX_D = 0.6         # [m] 左↔左/右↔右の対応距離閾値

# M3 辺クラスタ（今回は未実行）
EDGE_TOL_V = 0.20
EDGE_MIN_LEN = 0.5
EDGE_OVERLAP_RATIO = 0.3
EDGE_MATCH_MAX_D = 0.6

# M4 中心マッチ（今回は未実行）
MATCH_CENTER_WEIGHT = 1.0
MATCH_SIZE_WEIGHT   = 0.5
MATCH_MAX_COST      = 1.5
CORNER_CONNECT_STEPS = 20

# M5 3Dボクセル
VOXEL_RES_V = GRID_RES
VOXEL_RES_S = SECTION_INTERVAL
VOXEL_BORDER_ONLY = True

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

def marching_border_3d(mask):
    """3D bool配列の境界セルをTrueに（6近傍）。"""
    from scipy.ndimage import binary_erosion
    core = binary_erosion(mask, structure=np.array([[[0,0,0],[0,1,0],[0,0,0]],
                                                    [[0,1,0],[1,1,1],[0,1,0]],
                                                    [[0,0,0],[0,1,0],[0,0,0]]], dtype=bool),
                          border_value=False)
    return mask & (~core)

# ========= 断面処理 =========
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
        if not np.any(col): continue
        if np.any(col[i_lo:i_hi+1]):
            imax = np.max(np.where(col)[0])
            out[:imax+1, j] = True
    return (out.astype(np.uint8) * 255)

def slice_left_right_wall(free_bitmap, v_min, z_min, grid_res):
    """各Z行の自由セルから最左/最右のvを取得し、断線で分割したポリライン群を返す"""
    gh, gw = free_bitmap.shape
    vs_left = []; vs_right = []
    for zi in range(gh):
        row = free_bitmap[zi, :]
        xs = np.where(row)[0]
        if xs.size == 0:
            vs_left.append(np.nan); vs_right.append(np.nan); continue
        vL = v_min + (xs.min() + 0.5)*grid_res
        vR = v_min + (xs.max() + 0.5)*grid_res
        vs_left.append(vL); vs_right.append(vR)
    zs = [z_min + (zi + 0.5)*grid_res for zi in range(gh)]

    def split_poly(vs, zs):
        segs = []
        cur = []
        for i,(v,z) in enumerate(zip(vs,zs)):
            if np.isnan(v):  # 断線
                if len(cur) >= 2: segs.append(cur); cur = []
            else:
                cur.append([v, z])
        if len(cur) >= 2: segs.append(cur)
        out = []
        for seg in segs:
            if len(seg) < 2: continue
            out.append(seg)
        return out

    Ls = split_poly(vs_left, zs)
    Rs = split_poly(vs_right, zs)
    return Ls, Rs

def rectangles_on_slice(points_vz, grid_res, morph_radius, use_anchor, anchor_z, anchor_tol,
                        min_rect_size):
    """
    返り値：
      rect_edge_pts_vz: ビジュアル用縁点
      rect_models: [{center_vz, size_vw(=w,h), corners_vz[4]} ...]
      free_bitmap: True=自由（補間後&downfill後の占有の否定）
      union_from_rects: 長方形群をOR合成したラスタ（0/255）
      bbox: (v_min, z_min, gw, gh)
    """
    rect_models = []
    union_from_rects = None
    bbox = None

    if len(points_vz) == 0:
        return [], rect_models, None, None, None

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

    # 上方チェック
    def has_points_above_after_interp(top, left, h, w):
        gh_, gw_ = closed_bool.shape
        z_above_start = top + h
        if z_above_start >= gh_: return False
        sub = closed_bool[z_above_start:gh_, left:left+w]
        return np.any(sub)

    rect_edge_pts_vz = []
    free_work = free_bitmap.copy()
    union_from_rects = np.zeros_like(closed)  # 0/255
    while np.any(free_work):
        top, left, h, w = find_max_rectangle(free_work)
        if h < min_rect_size or w < min_rect_size:
            break
        if not has_points_above_after_interp(top, left, h, w):
            # 座標化
            v0 = v_min + (left + 0.5) * grid_res
            z0 = z_min + (top  + 0.5) * grid_res
            W  = w * grid_res
            H  = h * grid_res
            corners = [
                [v0,     z0     ],
                [v0+W,   z0     ],
                [v0+W,   z0+H   ],
                [v0,     z0+H   ],
            ]
            center = [v0 + 0.5*W, z0 + 0.5*H]
            corners = order_corners_ccw(corners, center)

            rect_models.append({
                "center_vz": np.array(center, dtype=float),
                "size_vw":   np.array([W, H], dtype=float),
                "corners_vz": [np.array(c, dtype=float) for c in corners],
            })

            # 縁セル中心を点に
            for zi_ in range(top, top+h):
                for yi_ in range(left, left+w):
                    if zi_ in (top, top+h-1) or yi_ in (left, left+w-1):
                        v = v_min + (yi_ + 0.5) * grid_res
                        z = z_min + (zi_ + 0.5) * grid_res
                        rect_edge_pts_vz.append([v, z])

            # 長方形ORを更新
            union_from_rects[top:top+h, left:left+w] = 255

        # 充填済み領域を除外
        free_work[top:top+h, left:left+w] = False

    bbox = (v_min, z_min, gw, gh)
    return rect_edge_pts_vz, rect_models, free_bitmap, union_from_rects, bbox

# ========= 接続ユーティリティ =========
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
    las_out.write(path)
    print(f"✅ 出力: {path}  点数: {N}")

# ========= 6手法 =========
def method1_union_loft(slices_meta, header_src, out_path):
    """外周ユニオン法：unionラスタ→外周→隣接対応→ロフト（線を点で打つ）"""
    ALL = []
    for i in range(len(slices_meta)-1):
        a = slices_meta[i]; b = slices_meta[i+1]
        # 外周抽出
        def contours_from_union(union, v_min, z_min, grid_res):
            if union is None: return []
            img = union.copy()
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            out = []
            for cnt in contours:
                pts = []
                for yx in cnt[:,0,:]:  # (x,y)だが列=V、行=Zで保持している
                    x, y = int(yx[0]), int(yx[1])
                    v = v_min + (x + 0.5)*grid_res
                    z = z_min + (y + 0.5)*grid_res
                    pts.append([v, z])
                out.append(resample_polyline(pts, ds=CONTOUR_RESAMPLE_DS))
            return out

        contA = contours_from_union(a["union"], a["v_min"], a["z_min"], a["grid_res"])
        contB = contours_from_union(b["union"], b["v_min"], b["z_min"], b["grid_res"])
        if not contA or not contB: continue

        # 重心で単純対応
        def centroid(poly):
            P = np.asarray(poly); return P.mean(axis=0)
        usedB = set()
        for pa in contA:
            ca = centroid(pa)
            best = None; best_d = 1e9; idxB = -1
            for j, pb in enumerate(contB):
                if j in usedB: continue
                cb = centroid(pb)
                d  = np.linalg.norm(ca - cb)
                if d < best_d:
                    best_d = d; best = pb; idxB = j
            if best is None or best_d > PAIR_MAX_CENTROID_DIST: continue
            usedB.add(idxB)

            # 点数合わせ
            P = np.asarray(resample_polyline(pa, ds=CONTOUR_RESAMPLE_DS))
            Q = np.asarray(resample_polyline(best, ds=CONTOUR_RESAMPLE_DS))
            n = min(len(P), len(Q))
            if n < 2: continue
            P = P[:n]; Q = Q[:n]

            # ロフト（各対応点を結ぶ直線を数点で）
            for k in range(n):
                v1,z1 = P[k]; v2,z2 = Q[k]
                for t in np.linspace(0,1,5):
                    v = (1-t)*v1 + t*v2
                    z = (1-t)*z1 + t*z2
                    ALL.append(vz_to_world_on_slice([v,z], a["c"], a["n_hat"]))

    write_green_las(out_path, header_src, ALL)

def method2_lr_walls(slices_meta, header_src, out_path):
    """左右壁ポリラインを 左↔左／右↔右 で対応→ロフト"""
    ALL = []
    for i in range(len(slices_meta)-1):
        a = slices_meta[i]; b = slices_meta[i+1]
        LsA, RsA = a["walls_L"], a["walls_R"]
        LsB, RsB = b["walls_L"], b["walls_R"]

        def pair_and_loft(listA, listB):
            used = set()
            for pa in listA:
                ca = np.mean(np.asarray(pa), axis=0)
                best=None; best_d=1e9; idx=-1
                for j,pb in enumerate(listB):
                    if j in used: continue
                    cb = np.mean(np.asarray(pb), axis=0)
                    d = np.linalg.norm(ca - cb)
                    if d < best_d: best_d=d; best=pb; idx=j
                if best is None or best_d > WALL_MATCH_MAX_D: continue
                used.add(idx)
                # 同点数へ
                P = np.asarray(resample_polyline(pa, ds=GRID_RES))
                Q = np.asarray(resample_polyline(best, ds=GRID_RES))
                n = min(len(P), len(Q)); 
                if n < 2: return
                P = P[:n]; Q = Q[:n]
                for k in range(n):
                    v1,z1 = P[k]; v2,z2 = Q[k]
                    for t in np.linspace(0,1,3):
                        v = (1-t)*v1 + t*v2
                        z = (1-t)*z1 + t*z2
                        ALL.append(vz_to_world_on_slice([v,z], a["c"], a["n_hat"]))

        pair_and_loft(LsA, LsB)
        pair_and_loft(RsA, RsB)

    write_green_las(out_path, header_src, ALL)

def method3_edge_cluster(slices_meta, header_src, out_path):
    """（未実行）長方形の立辺/横辺をクラスタ→代表線対応→ロフト（簡易版）"""
    pass

def method4_center_track(slices_meta, header_src, out_path):
    """（未実行）長方形中心＋サイズのコストで対応 → 4隅ロフト"""
    pass

def method5_voxel_connect(slices_meta, header_src, out_path):
    """自由空間を3Dグリッド化して連結（境界セル中心を出力）"""
    if not slices_meta: 
        write_green_las(out_path, header_src, []); return

    grid_res = slices_meta[0]["grid_res"]
    v_all_min = min(s["v_min"] for s in slices_meta)
    z_all_min = min(s["z_min"] for s in slices_meta)
    v_all_max = max(s["v_min"] + s["gw"]*grid_res for s in slices_meta)
    z_all_max = max(s["z_min"] + s["gh"]*grid_res for s in slices_meta)
    gw = int(np.ceil((v_all_max - v_all_min)/grid_res))
    gh = int(np.ceil((z_all_max - z_all_min)/grid_res))
    gu = len(slices_meta)

    # mask[u,z,v] = True（自由）
    mask = np.zeros((gu, gh, gw), dtype=bool)
    for u, s in enumerate(slices_meta):
        fb = s["free_bitmap"]
        if fb is None: continue
        off_v = int(round((s["v_min"] - v_all_min)/grid_res))
        off_z = int(round((s["z_min"] - z_all_min)/grid_res))
        h,w = fb.shape
        mask[u, off_z:off_z+h, off_v:off_v+w] = fb

    # 境界抽出
    try:
        from scipy.ndimage import label  # noqa: F401
        border = marching_border_3d(mask)
    except Exception:
        border = np.zeros_like(mask)
        U,Z,V = mask.shape
        for u in range(U):
            for z in range(Z):
                for v in range(V):
                    if not mask[u,z,v]: continue
                    nbr = 0
                    for du,dz,dv in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        uu,zz,vv = u+du, z+dz, v+dv
                        if 0<=uu<U and 0<=zz<Z and 0<=vv<V:
                            if not mask[uu,zz,vv]: nbr += 1
                        else:
                            nbr += 1
                    if nbr>0: border[u,z,v]=True

    ALL=[]
    for u in range(gu):
        c = slices_meta[u]["c"]; n_hat = slices_meta[u]["n_hat"]
        # 修正：np.whereは (z_idx, v_idx) の順に返す
        zz, vv = np.where(border[u])         # 行=Z, 列=V
        for z_i, v_i in zip(zz, vv):
            v = v_all_min + (v_i + 0.5)*grid_res
            z = z_all_min + (z_i + 0.5)*grid_res
            ALL.append(vz_to_world_on_slice([v,z], c, n_hat))
    write_green_las(out_path, header_src, ALL)

def method6_restore_then_union(slices_meta, header_src, green_points_xyz, out_path):
    """既存のGREEN点（世界座標）を各スライスへ逆写像→ラスタ→M1と同様に外周ロフト"""
    rest_slices = []
    for s in slices_meta:
        v_min, z_min, gw, gh = s["v_min"], s["z_min"], s["gw"], s["gh"]
        grid_res = s["grid_res"]

        # 空スライスはスキップ（cv2.dilateの !src.empty() 対策）
        if gw == 0 or gh == 0:
            continue

        img = np.zeros((gh, gw), dtype=np.uint8)
        c = s["c"]; n_hat = s["n_hat"]
        for x,y,z in green_points_xyz:
            dxy = np.array([x - c[0], y - c[1]])
            v = dxy @ n_hat
            vi = int(np.floor((v - v_min)/grid_res))
            zi = int(np.floor((z - z_min)/grid_res))
            if 0 <= zi < gh and 0 <= vi < gw:
                img[zi, vi] = 255

        # 近傍膨張（非空のみ）
        if img.size == 0:
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        img = cv2.dilate(img, kernel, iterations=1)

        rest_slices.append({
            "img": img, "v_min": v_min, "z_min": z_min, "grid_res": grid_res,
            "c": s["c"], "n_hat": s["n_hat"]
        })

    # 外周抽出→隣接対応→ロフト（M1同等）
    ALL=[]
    for i in range(len(rest_slices)-1):
        a = rest_slices[i]; b = rest_slices[i+1]
        def contours(img, v_min, z_min, grid_res):
            if img is None or img.size == 0:
                return []
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            out=[]
            for cnt in contours:
                pts=[]
                for xy in cnt[:,0,:]:
                    x,y=int(xy[0]),int(xy[1])
                    v = v_min + (x+0.5)*grid_res
                    z = z_min + (y+0.5)*grid_res
                    pts.append([v,z])
                out.append(resample_polyline(pts, ds=CONTOUR_RESAMPLE_DS))
            return out
        contA = contours(a["img"], a["v_min"], a["z_min"], a["grid_res"])
        contB = contours(b["img"], b["v_min"], b["z_min"], b["grid_res"])
        if not contA or not contB: continue

        def centroid(poly): return np.mean(np.asarray(poly), axis=0)
        used=set()
        for pa in contA:
            ca = centroid(pa)
            best=None; best_d=1e9; idx=-1
            for j,pb in enumerate(contB):
                if j in used: continue
                cb = centroid(pb)
                d = np.linalg.norm(ca - cb)
                if d<best_d: best_d=d; best=pb; idx=j
            if best is None or best_d>PAIR_MAX_CENTROID_DIST: continue
            used.add(idx)
            P=np.asarray(resample_polyline(pa, ds=CONTOUR_RESAMPLE_DS))
            Q=np.asarray(resample_polyline(best, ds=CONTOUR_RESAMPLE_DS))
            n=min(len(P),len(Q))
            if n<2: continue
            P=P[:n]; Q=Q[:n]
            for k in range(n):
                v1,z1=P[k]; v2,z2=Q[k]
                for t in np.linspace(0,1,5):
                    v=(1-t)*v1 + t*v2
                    z=(1-t)*z1 + t*z2
                    ALL.append(vz_to_world_on_slice([v,z], a["c"], a["n_hat"]))
    write_green_las(out_path, header_src, ALL)

# ========= メイン =========
def main():
    os.makedirs(os.path.dirname(OUTPUT_LAS_BASE) or ".", exist_ok=True)
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

    # --- gap=50mで間引き ---
    thinned = [through[0]]
    for p in through[1:]:
        if l2(thinned[-1], p) >= GAP_DIST:
            thinned.append(p)
    through = np.asarray(thinned, float)

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

    # --- 断面処理 → GREEN と meta を収集 ---
    half_len = LINE_LENGTH * 0.5
    half_th  = SLICE_THICKNESS * 0.5
    GREEN = []
    slices_meta = []  # per slice

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
            # empty slice meta
            slices_meta.append({
                "c": c, "n_hat": n_hat, "grid_res": GRID_RES,
                "v_min": 0, "z_min": 0, "gw": 0, "gh": 0,
                "free_bitmap": None, "union": None,
                "rects": [], "walls_L": [], "walls_R": []
            })
            continue

        points_vz = np.column_stack([v[m_nav], Z[m_nav]])

        # v–z occupancy → 長方形＆自由空間
        rect_edges_vz, rect_models, free_bitmap, union_rects, bbox = rectangles_on_slice(
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

        # 左右壁ポリライン（自由空間から）
        walls_L, walls_R = slice_left_right_wall(free_bitmap, bbox[0], bbox[1], GRID_RES)

        slices_meta.append({
            "c": c, "n_hat": n_hat, "grid_res": GRID_RES,
            "v_min": bbox[0], "z_min": bbox[1], "gw": bbox[2], "gh": bbox[3],
            "free_bitmap": free_bitmap, "union": union_rects,
            "rects": rect_models, "walls_L": walls_L, "walls_R": walls_R
        })

    if not GREEN:
        raise RuntimeError("航行可能空間の長方形が見つかりませんでした。パラメータを調整してください。")

    # === ベース（緑枠のみ） ===
    write_green_las(OUTPUT_LAS_BASE + "_M0_rect_edges.las", las.header, GREEN)

    # === 手法1,2,5,6を実行（M3/M4はオフ） ===
    method1_union_loft(slices_meta, las.header, OUTPUT_LAS_BASE + "_M1_union.las")
    method2_lr_walls(slices_meta, las.header, OUTPUT_LAS_BASE + "_M2_lrwall.las")
    # method3_edge_cluster(slices_meta, las.header, OUTPUT_LAS_BASE + "_M3_edgecluster.las")
    # method4_center_track(slices_meta, las.header, OUTPUT_LAS_BASE + "_M4_centertrack.las")
    method5_voxel_connect(slices_meta, las.header, OUTPUT_LAS_BASE + "_M5_voxel.las")
    method6_restore_then_union(slices_meta, las.header, GREEN, OUTPUT_LAS_BASE + "_M6_restore_union.las")

    print("✅ 全手法完了")
    print(f"  gap=50適用後 中心線点数: {len(through)}")
    print(f"  断面数（内挿）        : {len(centers)}")
    print(f"  M0基礎出力点         : {len(GREEN)}")

if __name__ == "__main__":
    main()
