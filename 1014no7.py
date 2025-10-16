# -*- coding: utf-8 -*-
"""
【中心線制約IBGAL → 逐次ICP → 航跡描画（黒影対応・法線自動・GPU対応）】

- 地図に入るまで：path_resampled.json の各点±RのみXY探索（IBGAL, 画像NCC）
- 一致後：逐次ICPで追従（CupochがあればGPU、無ければOpen3D）
- 欠測(range=0)除外 & 距離フィルタ
- 地図法線は無ければ推定
- 航跡CSV・俯瞰重ねPNGを保存

必要 pip:
  ouster-sdk open3d numpy opencv-python cupy-cuda11x cupoch matplotlib
"""

import os, math, csv, json
import numpy as np
import open3d as o3d

from ouster.sdk.open_source import open_source
from ouster.sdk.core import XYZLut, ChanField, SensorInfo

# ==========================================================
# 🔧 入出力パス
# ==========================================================
PCAP_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
JSON_PATH  = r"/workspace/data/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
MAP_PATH   = r"/workspace/data/1016_merged_lidar_uesita.ply"
PATH_JSON  = r"/workspace/data/path_resampled.json"
OUTPUT_DIR = r"/workspace/output/ibgal_path_init_icp_route"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# ⚙️ スキャン前処理
# ==========================================================
MIN_RANGE = 1.0         # [m]
MAX_RANGE = 120.0       # [m] 0で無効
MIN_POINTS_PER_SCAN = 2000

# ==========================================================
# 🧭 IBGAL（画像ベース粗探索, path制約）
# ==========================================================
# パノラマ分解能と視野
YAW_RES_DEG   = 0.5
PITCH_RES_DEG = 1.0
FOV_PITCH_MIN = -20.0
FOV_PITCH_MAX =  20.0
MAX_DEPTH     = 120.0   # [m] 画像のクリップ最大距離

# 地図を画像レンダするZ帯域（視認したい高さ）
Z_RENDER_MIN = -5.0
Z_RENDER_MAX =  15.0

# path制約：各path点の周囲だけ探索
SEARCH_RADIUS_M   = 30.0   # [m] 中心線からの±半径
GRID_STEP_M       = 5.0    # [m] XYグリッド刻み

# path進行方向 → 初期Yaw, 局所Yaw微調整
REVERSE_PATH_DIRECTION = True
LOCAL_YAW_WIN    = 10.0    # [deg]
LOCAL_YAW_STEP   = 1.0     # [deg]

# 一致採用のNCCしきい値
IBGAL_MIN_SCORE  = 0.22
NCC_MIN_COMMON   = 2500    # 有効共通画素の下限

# ==========================================================
# 🔁 逐次ICP
# ==========================================================
VOXEL_SIZE     = 0.15
MAX_CORR_DIST  = 0.5
MAX_ITER       = 60
MIN_FITNESS    = 0.30
MAX_RMSE       = 0.25
SAVE_EVERY     = 25

MAP_NORMAL_RAD = 1.0
MAP_NORMAL_NN  = 50

# ==========================================================
# ⚡ GPU切替: Cupoch (ICP) / CuPy (NCC)
# ==========================================================
_HAS_CUPOCH = False
try:
    import cupoch as cph
    _HAS_CUPOCH = True
    print("🚀 Cupoch(GPU ICP) 有効")
except Exception:
    print("⚠ Cupoch なし → CPU ICP")

_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
    print("🚀 CuPy(GPU NCC) 有効")
except Exception:
    import numpy as cp  # API互換のため
    print("⚠ CuPy なし → CPU NCC")

# ==========================================================
# 🧩 基本ユーティリティ
# ==========================================================
def yaw_to_Rz(yaw_deg: float) -> np.ndarray:
    y = math.radians(yaw_deg)
    c, s = math.cos(y), math.sin(y)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)

def filter_scan_points(xyz_raw, rng):
    valid = (rng > 0)
    pts = xyz_raw[valid].reshape(-1, 3)
    if pts.size == 0:
        return pts
    d = np.linalg.norm(pts, axis=1)
    ok = (d > MIN_RANGE)
    if MAX_RANGE > 0:
        ok &= (d < MAX_RANGE)
    return pts[ok]

# ==========================================================
# 🖼 パノラマ深度画像（スキャン/地図共通）
# ==========================================================
def points_to_panorama_depth(
    pts, origin_xyz, yaw_deg,
    yaw_res=YAW_RES_DEG, pitch_res=PITCH_RES_DEG,
    pitch_min=FOV_PITCH_MIN, pitch_max=FOV_PITCH_MAX,
    z_band=(Z_RENDER_MIN, Z_RENDER_MAX),
    max_depth=MAX_DEPTH
):
    H = int((pitch_max - pitch_min) / pitch_res) + 1
    W = int(360.0 / yaw_res) + 1
    if pts.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    # 視点座標系へ（ワールド→視点）
    Rz = yaw_to_Rz(yaw_deg)
    rel = (pts - origin_xyz[None, :]) @ Rz

    # Z帯域内に限定
    zmin, zmax = z_band
    rel = rel[(rel[:, 2] >= zmin) & (rel[:, 2] <= zmax)]
    if rel.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    x, y, z = rel[:,0], rel[:,1], rel[:,2]
    r = np.linalg.norm(rel, axis=1)
    r = np.clip(r, 1e-6, max_depth)

    yaw   = np.degrees(np.arctan2(y, x))                        # [-180,180)
    pitch = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))       # [-90,90]

    mask_p = (pitch >= pitch_min) & (pitch <= pitch_max)
    if not np.any(mask_p):
        return np.zeros((H, W), dtype=np.float32)
    yaw, pitch, r = yaw[mask_p], pitch[mask_p], r[mask_p]

    xi = np.round((yaw + 180.0) / yaw_res).astype(int)
    yi = np.round((pitch - pitch_min) / pitch_res).astype(int)
    xi = np.clip(xi, 0, W-1)
    yi = np.clip(yi, 0, H-1)

    img = np.zeros((H, W), dtype=np.float32)
    # 同一画素は最小距離（手前）を保持
    for u, v, d in zip(xi, yi, r):
        px = img[v, u]
        if px == 0 or d < px:
            img[v, u] = d
    # 0..1 正規化
    img = np.clip(img / max_depth, 0.0, 1.0)
    return img

# ==========================================================
# 🔗 NCC（未観測0を除外した相互相関, GPU/CPU）
# ==========================================================
def ncc_masked_gpu(a: np.ndarray, b: np.ndarray, min_common: int) -> float:
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common:
        return -1e9
    av = cp.asarray(a[m], dtype=cp.float32)
    bv = cp.asarray(b[m], dtype=cp.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = cp.sqrt((av*av).sum() * (bv*bv).sum()) + 1e-6
    score = (av*bv).sum() / denom
    return float(score.get()) if _HAS_CUPY else float(score)

def ncc_masked_cpu(a: np.ndarray, b: np.ndarray, min_common: int) -> float:
    m = (a > 0) & (b > 0)
    n = int(m.sum())
    if n < min_common:
        return -1e9
    A = a[m].astype(np.float32); A -= A.mean()
    B = b[m].astype(np.float32); B -= B.mean()
    denom = np.sqrt((A*A).sum() * (B*B).sum()) + 1e-6
    return float((A*B).sum() / denom)

def ncc(a, b, min_common):  # ラッパ
    return ncc_masked_gpu(a, b, min_common) if _HAS_CUPY else ncc_masked_cpu(a, b, min_common)

# ==========================================================
# 🧭 IBGAL（path制約）：このフレームで初期姿勢を探す
# ==========================================================
def compute_path_yaws(path_xy: np.ndarray, reverse: bool) -> np.ndarray:
    yaws = []
    for i in range(len(path_xy) - 1):
        x0,y0 = path_xy[i]
        x1,y1 = path_xy[i+1]
        yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
        if reverse:
            yaw = (yaw + 180.0) % 360.0
        yaws.append(yaw)
    if len(path_xy) >= 2:
        yaws.append(yaws[-1])  # 末尾は最後のyawを使う
    else:
        yaws.append(0.0)
    return np.array(yaws, dtype=np.float32)

def ibgal_search_path_limited(scan_pts, map_pts, path_xy):
    """中心線各点±RのXYのみ探索。Yawは進行方向±局所微調整。"""
    # スキャンを画像化（視点はLiDAR原点, yaw=0）
    scan_img = points_to_panorama_depth(
        scan_pts, origin_xyz=np.array([0.0,0.0,0.0]),
        yaw_deg=0.0,
        yaw_res=YAW_RES_DEG, pitch_res=PITCH_RES_DEG,
        pitch_min=FOV_PITCH_MIN, pitch_max=FOV_PITCH_MAX,
        z_band=(Z_RENDER_MIN, Z_RENDER_MAX),
        max_depth=MAX_DEPTH
    )

    base_yaws = compute_path_yaws(path_xy, REVERSE_PATH_DIRECTION)
    local_offsets = np.arange(-LOCAL_YAW_WIN, LOCAL_YAW_WIN + 1e-6, LOCAL_YAW_STEP)

    best = {"score": -1e9, "x":0.0, "y":0.0, "yaw":0.0}

    for (px, py), base_yaw in zip(path_xy, base_yaws):
        xs = np.arange(px - SEARCH_RADIUS_M, px + SEARCH_RADIUS_M + 1e-9, GRID_STEP_M)
        ys = np.arange(py - SEARCH_RADIUS_M, py + SEARCH_RADIUS_M + 1e-9, GRID_STEP_M)
        for y0 in ys:
            for x0 in xs:
                # まず基準yawで地図レンダ
                ref_img = points_to_panorama_depth(
                    map_pts, origin_xyz=np.array([x0,y0,0.0]),
                    yaw_deg=float(base_yaw),
                    yaw_res=YAW_RES_DEG, pitch_res=PITCH_RES_DEG,
                    pitch_min=FOV_PITCH_MIN, pitch_max=FOV_PITCH_MAX,
                    z_band=(Z_RENDER_MIN, Z_RENDER_MAX),
                    max_depth=MAX_DEPTH
                )
                s = ncc(scan_img, ref_img, NCC_MIN_COMMON)
                best_local = s
                best_yaw_local = float(base_yaw)

                # 局所Yaw微調整（±LOCAL_YAW_WIN）
                if LOCAL_YAW_WIN > 0:
                    for off in local_offsets:
                        if off == 0:
                            continue
                        ref2 = points_to_panorama_depth(
                            map_pts, origin_xyz=np.array([x0,y0,0.0]),
                            yaw_deg=float(base_yaw + off),
                            yaw_res=YAW_RES_DEG, pitch_res=PITCH_RES_DEG,
                            pitch_min=FOV_PITCH_MIN, pitch_max=FOV_PITCH_MAX,
                            z_band=(Z_RENDER_MIN, Z_RENDER_MAX),
                            max_depth=MAX_DEPTH
                        )
                        s2 = ncc(scan_img, ref2, NCC_MIN_COMMON)
                        if s2 > best_local:
                            best_local = s2
                            best_yaw_local = float(base_yaw + off)

                if best_local > best["score"]:
                    best.update({"score": best_local, "x": float(x0), "y": float(y0), "yaw": best_yaw_local})

    return best  # {"score","x","y","yaw"}

# ==========================================================
# 💻/⚡ ICP
# ==========================================================
def cpu_icp(scan_ds, map_ds, init_T):
    res = o3d.pipelines.registration.registration_icp(
        scan_ds, map_ds, MAX_CORR_DIST, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )
    return res.transformation, float(res.fitness), float(res.inlier_rmse)

def gpu_icp(scan_ds, map_ds, init_T):
    s = cph.geometry.PointCloud(); m = cph.geometry.PointCloud()
    s.points = cph.utility.Vector3fVector(np.asarray(scan_ds.points, dtype=np.float32))
    m.points = cph.utility.Vector3fVector(np.asarray(map_ds.points, dtype=np.float32))
    if not map_ds.has_normals():
        m.estimate_normals(cph.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD, max_nn=MAP_NORMAL_NN))
    reg = cph.registration.registration_icp(
        s, m, MAX_CORR_DIST, init_T.astype(np.float32),
        cph.registration.TransformationEstimationPointToPlane(),
        cph.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    )
    return reg.transformation.astype(np.float64), float(reg.fitness), float(reg.inlier_rmse)

# ==========================================================
# 🖼 俯瞰重ねPNG
# ==========================================================
def save_overlay_png(map_pcd, scan_pts_world, out_png, lookat_xyz):
    m = o3d.geometry.PointCloud(map_pcd)  # copy
    s = o3d.geometry.PointCloud()
    s.points = o3d.utility.Vector3dVector(scan_pts_world)
    m.paint_uniform_color([0.6,0.6,0.6])
    s.paint_uniform_color([1.0,0.0,0.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    vis.add_geometry(m); vis.add_geometry(s)
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])  # top-down
    ctr.set_up([0, 1, 0])
    ctr.set_lookat([float(lookat_xyz[0]), float(lookat_xyz[1]), float(lookat_xyz[2])])
    ctr.set_zoom(0.05)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(out_png)
    vis.destroy_window()

# ==========================================================
# 🚀 メイン
# ==========================================================
def main():
    # 地図
    print("🗺 地図読み込み中...")
    map_pcd = o3d.io.read_point_cloud(MAP_PATH)
    if not map_pcd.has_normals():
        print("⚠ 地図に法線なし → 推定")
        map_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=MAP_NORMAL_RAD, max_nn=MAP_NORMAL_NN))
    map_pts = np.asarray(map_pcd.points)
    print(f"✅ 地図点数: {len(map_pts):,}")

    # path
    print("📄 path_resampled.json 読み込み中...")
    with open(PATH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    path_xy = np.array(data["path"], dtype=np.float64)
    if len(path_xy) < 2:
        raise RuntimeError("pathの点が不足（>=2点必要）")
    print(f"✅ path点数: {len(path_xy)}")

    # LiDARメタ
    with open(JSON_PATH, "r") as f:
        sensor_info = SensorInfo(f.read())
    xyzlut = XYZLut(sensor_info, use_extrinsics=False)
    source = open_source(PCAP_PATH)

    # 出力CSV
    csv_path = os.path.join(OUTPUT_DIR, "trajectory.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["frame","status","x","y","z","yaw_deg","fitness","rmse","ibgal_score"])

    icp_started = False
    T_global = np.eye(4)
    frame_idx = 0
    traj_xy = []

    print("📡 ストリーム処理開始（path制約IBGAL → 逐次ICP）...")
    for scans in source:
        for scan in (scans if isinstance(scans, list) else [scans]):
            xyz_raw = xyzlut(scan)                 # (H,W,3)
            rng     = scan.field(ChanField.RANGE)  # (H,W)
            scan_pts = filter_scan_points(xyz_raw, rng)
            if scan_pts.shape[0] < MIN_POINTS_PER_SCAN:
                frame_idx += 1
                continue

            if not icp_started:
                print(f"Frame {frame_idx:04d}: 初期合わせ（path±{SEARCH_RADIUS_M}m, {GRID_STEP_M}m刻み, yaw±{LOCAL_YAW_WIN}°）...")
                best = ibgal_search_path_limited(scan_pts, map_pts, path_xy)
                if best["score"] >= IBGAL_MIN_SCORE:
                    yaw = best["yaw"]; x0 = best["x"]; y0 = best["y"]; z0 = 0.0
                    T_global = np.eye(4)
                    T_global[:3,:3] = yaw_to_Rz(yaw)
                    T_global[:3, 3] = [x0, y0, z0]
                    icp_started = True
                    print(f"✅ 初期合わせ成功: score={best['score']:.3f}, x={x0:.2f}, y={y0:.2f}, yaw={yaw:.1f}")
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([frame_idx, "INIT_OK", x0, y0, z0, yaw, "", "", best["score"]])
                    scan_world = (scan_pts @ T_global[:3,:3].T) + T_global[:3,3]
                    save_overlay_png(map_pcd, scan_world,
                                     os.path.join(OUTPUT_DIR, f"overlay_init_{frame_idx:05d}.png"),
                                     lookat_xyz=T_global[:3,3])
                else:
                    print(f"❌ 初期合わせ失敗: score={best['score']:.3f} < {IBGAL_MIN_SCORE}")
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([frame_idx, "INIT_FAIL", "", "", "", "", "", "", best["score"]])
                frame_idx += 1
                continue

            # --- 地図内：ICP ---
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(scan_pts)
            scan_ds = scan_pcd.voxel_down_sample(VOXEL_SIZE)

            if _HAS_CUPOCH:
                try:
                    T_new, fit, rmse = gpu_icp(scan_ds, map_pcd, T_global)
                except Exception as e:
                    print(f"⚠ Cupoch失敗→CPU: {e}")
                    T_new, fit, rmse = cpu_icp(scan_ds, map_pcd, T_global)
            else:
                T_new, fit, rmse = cpu_icp(scan_ds, map_pcd, T_global)

            if fit < MIN_FITNESS or rmse > MAX_RMSE:
                print(f"Frame {frame_idx:04d}: REJECT fit={fit:.3f}, rmse={rmse:.3f}")
                with open(csv_path, "a", newline="") as f:
                    yaw_deg = math.degrees(math.atan2(T_global[1,0], T_global[0,0]))
                    csv.writer(f).writerow([frame_idx, "REJECT", *T_global[:3,3], yaw_deg, fit, rmse, ""])
                frame_idx += 1
                continue

            T_global = T_new
            x, y, z = T_global[:3,3]
            yaw_deg = math.degrees(math.atan2(T_global[1,0], T_global[0,0]))
            traj_xy.append([x, y])

            print(f"Frame {frame_idx:04d}: OK fit={fit:.3f}, rmse={rmse:.3f}")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([frame_idx, "OK", x, y, z, yaw_deg, fit, rmse, ""])

            if frame_idx % SAVE_EVERY == 0:
                scan_world = (scan_pts @ T_global[:3,:3].T) + T_global[:3,3]
                outpng = os.path.join(OUTPUT_DIR, f"overlay_track_{frame_idx:05d}.png")
                save_overlay_png(map_pcd, scan_world, outpng, lookat_xyz=T_global[:3,3])

            frame_idx += 1

    print("✅ 全フレーム処理完了")
    print(f"📄 Trajectory CSV: {csv_path}")

    # 航跡画像
    if len(traj_xy) > 1:
        import matplotlib.pyplot as plt
        traj_xy_np = np.asarray(traj_xy)
        map_xy = np.asarray(map_pts)[:, :2]
        plt.figure(figsize=(10,10))
        plt.scatter(map_xy[:,0], map_xy[:,1], s=0.1, c='gray', alpha=0.4)
        plt.plot(traj_xy_np[:,0], traj_xy_np[:,1], c='red', linewidth=2)
        plt.axis('equal'); plt.title("Route on Map")
        out_route = os.path.join(OUTPUT_DIR, "map_route.png")
        plt.savefig(out_route, dpi=300)
        print(f"🖼 航跡画像保存: {out_route}")
    else:
        print("（航跡点が少なく map_route.png を生成しませんでした）")

if __name__ == "__main__":
    main()
