# -*- coding: utf-8 -*-
"""
【機能】
- PLYファイルを読み込み、ノイズ除去（Statistical Outlier Removal）でレイキャスト用点群を生成
- 水面水位 (water_level) + 船高クリアランス (clearance) を上限 z_limit として Z ≤ z_limit の点群を抽出
- 2D凸包の重心から放射状にレイキャストして垂直・斜面の外殻点を抽出
- Y軸方向にビン分割して水平床 (床面) の高さを検出し、安全距離内側にオフセットした床外殻点を抽出（水平面も内側寄せ）
- 衝突点／床外殻点ともにボクセルグリッドにマッピングし、各セル中心を緑色外殻点とする
- オリジナルの点群（ノイズ除去前）＋全ての外殻点を統合して PLY出力
"""

import os
import math
import argparse
import logging

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, ConvexHull


def run_raycast(yz, valid_pts, origin, n_rays, steps, dist_thresh, safety_dist):
    tree   = cKDTree(yz)
    angles = np.linspace(0, 2*math.pi, n_rays, endpoint=False)
    dirs   = np.vstack((np.cos(angles), np.sin(angles))).T

    grid   = origin + steps[:, None, None] * dirs[None, :, :]
    flat   = grid.reshape(-1, 2)
    dists, idxs = tree.query(flat)
    dists = dists.reshape(grid.shape[:2])
    idxs  = idxs.reshape(grid.shape[:2])

    hit_pts, missing = [], []
    for j in range(n_rays):
        col  = dists[:, j]
        hits = np.where(col < dist_thresh)[0]
        if hits.size == 0:
            missing.append(j)
            continue
        i       = hits[0]
        d_hit   = steps[i]
        d_safe  = max(d_hit - safety_dist, 0.0)
        dy, dz  = dirs[j]
        y_s, z_s = origin + np.array([dy, dz]) * d_safe
        _, idx3 = tree.query([y_s, z_s])
        x_s     = valid_pts[idx3, 0]
        hit_pts.append([x_s, y_s, z_s])
    return np.array(hit_pts), missing


def detect_floor_points(valid_pts, bin_size, clearance, horizontal_offset):
    # Y軸方向にビン分割して水平床面の高さ+安全距離を求め、水平面も内側寄せ
    ys = valid_pts[:, 1]
    y_min, y_max = ys.min(), ys.max()
    bins = np.arange(y_min, y_max + bin_size, bin_size)
    center2d = np.mean(valid_pts[:, :2], axis=0)
    floor_pts = []
    for i in range(len(bins)-1):
        y_low, y_high = bins[i], bins[i+1]
        mask = (ys >= y_low) & (ys < y_high)
        seg = valid_pts[mask]
        if seg.size == 0:
            continue
        floor_z = seg[:, 2].max()
        y_c = (y_low + y_high) / 2
        x_c = seg[:, 0].mean()
        # 内側方向ベクトル (XY平面)
        vec = center2d - np.array([x_c, y_c])
        norm = np.linalg.norm(vec)
        if norm > 0:
            dir2d = vec / norm
        else:
            dir2d = np.zeros(2)
        # 水平面の内側寄せ
        x_s, y_s = np.array([x_c, y_c]) + dir2d * horizontal_offset
        floor_pts.append([x_s, y_s, floor_z + clearance])
    return np.array(floor_pts)


def parse_args():
    p = argparse.ArgumentParser(description="Extract navigable boundary including floors and walls with inward offset")
    p.add_argument("--input_ply", default=r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_L_only.ply",
                   help="入力 PLY ファイルパス")
    p.add_argument("--output_ply", default=r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\navigable_full_h_offset.ply",
                   help="出力 PLY ファイルパス")
    p.add_argument("--water_level", type=float, default=3.5,
                   help="水面高さ h (m)")
    p.add_argument("--clearance", type=float, default=1.0,
                   help="船高クリアランス (m)")
    p.add_argument("--voxel_size", type=float, default=0.2,
                   help="ボクセルサイズ (m)")
    p.add_argument("--n_rays", type=int, default=720,
                   help="初期放射レイ本数")
    p.add_argument("--ray_length", type=float, default=20.0,
                   help="レイ最大長さ (m)")
    p.add_argument("--step", type=float, default=0.05,
                   help="レイサンプリング間隔 (m)")
    p.add_argument("--dist_thresh", type=float, default=0.1,
                   help="衝突判定距離閾値 (m)")
    p.add_argument("--bin_size", type=float, default=0.2,
                   help="床検出時の Y ビン幅 (m)")
    p.add_argument("--safety_dist", type=float, default=1.0,
                   help="垂直／水平の安全距離 (m)")
    p.add_argument("--outlier", action="store_true",
                   help="Statistical Outlier Removal を適用する")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger()

    # 1) オリジナル点群保持
    pcd_orig = o3d.io.read_point_cloud(args.input_ply)
    pts_all  = np.asarray(pcd_orig.points)
    cols_all = np.asarray(pcd_orig.colors)

    # 2) ノイズ除去
    valid_pts = pts_all
    if args.outlier:
        logger.info("Statistical Outlier Removal 適用")
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_pts))
        tmp, _ = tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        valid_pts = np.asarray(tmp.points)

    # 3) Z制限 (water_level+clearance)
    z_limit = args.water_level + args.clearance
    logger.info(f"z_limit = {z_limit} m")
    mask = valid_pts[:, 2] <= z_limit
    valid = valid_pts[mask]
    if valid.size == 0:
        logger.error("Z ≤ z_limit の点がありません")
        return

    # 4) 2D凸包原点
    yz = valid[:, 1:3]
    hull = ConvexHull(yz)
    origin = yz[hull.vertices].mean(axis=0)
    logger.info(f"Ray origin (Y,Z) = {origin}")

    # 5) レイキャスト (壁面)
    steps = np.arange(0, args.ray_length + args.step, args.step)
    wall_pts, missing = run_raycast(
        yz, valid, origin,
        args.n_rays, steps,
        args.dist_thresh, args.safety_dist
    )
    logger.info(f"Wall hits={len(wall_pts)}, missing={len(missing)}")

    # 6) 床面検出 (水平) + 内側寄せ
    floor_pts = detect_floor_points(valid, args.bin_size, args.clearance, args.safety_dist)
    logger.info(f"Floor points detected={len(floor_pts)}")

    # 7) ユニーク化してボクセル中心化
    all_shell = np.vstack([wall_pts, floor_pts])
    minb = valid.min(axis=0)
    ijk = np.floor((all_shell - minb) / args.voxel_size).astype(int)
    uniq, _ = np.unique(ijk, axis=0, return_index=True)
    centers = minb + (uniq + 0.5) * args.voxel_size

    # 8) 緑外殻点を統合出力
    green = np.tile([0.0, 1.0, 0.0], (len(centers), 1))
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack([pts_all, centers]))
    merged.colors = o3d.utility.Vector3dVector(np.vstack([cols_all, green]))
    os.makedirs(os.path.dirname(args.output_ply), exist_ok=True)
    o3d.io.write_point_cloud(args.output_ply, merged)
    logger.info(f"Output complete: {args.output_ply}")

if __name__ == "__main__":
    main()
