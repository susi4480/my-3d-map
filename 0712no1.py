#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【機能】
- PLYファイルを読み込み、ノイズ除去（Statistical Outlier Removal）で「レイキャスト用点群」を生成
- レイキャスト用点群には Z 制限をせず、全体の Y–Z 座標から 2D 凸包の重心を起点として放射状レイキャスト
- レイキャストのヒット位置を安全距離だけ内側にオフセット
- 衝突点をボクセルグリッドにマッピングし、各セル中心を「航行可能外殻点」として抽出
- **抽出した緑点を高さ上限 z_limit（= water_level + clearance）でフィルタ**
- 最終出力として、「オリジナルの点群（ノイズ除去前／高さ制限前）」＋「フィルタ済み外殻点」を統合して PLY保存
"""

import os, math, argparse, logging
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, ConvexHull

def parse_args():
    p = argparse.ArgumentParser(description="Raycast + height‐filter for green points")
    p.add_argument("--input_ply",
                   default=r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_L_only.ply",
                   help="入力PLYファイルパス")
    p.add_argument("--output_ply",
                   default=r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\0712no1_navigable_filtered.ply",
                   help="出力PLYファイルパス")
    p.add_argument("--water_level", type=float, default=3.5,
                   help="水面高さ h [m]")
    p.add_argument("--clearance",   type=float, default=1.0,
                   help="クリアランス（船高余裕）[m]")
    p.add_argument("--voxel_size",  type=float, default=0.2,
                   help="ボクセルサイズ [m]")
    p.add_argument("--n_rays",      type=int,   default=720,
                   help="放射レイ本数")
    p.add_argument("--ray_length",  type=float, default=20.0,
                   help="レイ最大長さ [m]")
    p.add_argument("--step",        type=float, default=0.05,
                   help="レイサンプリング間隔 [m]")
    p.add_argument("--dist_thresh", type=float, default=0.1,
                   help="衝突判定距離閾値 [m]")
    p.add_argument("--safety_dist", type=float, default=1.0,
                   help="安全距離 [m]")
    p.add_argument("--outlier",     action="store_true",
                   help="Statistical Outlier Removal を適用")
    return p.parse_args()

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
        i      = hits[0]
        d_hit  = steps[i]
        d_safe = max(d_hit - safety_dist, 0.0)
        dy, dz = dirs[j]
        y_s, z_s = origin + np.array([dy, dz]) * d_safe
        _, idx3 = tree.query([y_s, z_s])
        x_s     = valid_pts[idx3, 0]
        hit_pts.append([x_s, y_s, z_s])
    return np.array(hit_pts), missing

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger()

    # 1) オリジナル点群保持
    pcd_orig = o3d.io.read_point_cloud(args.input_ply)
    pts_all  = np.asarray(pcd_orig.points)
    cols_all = np.asarray(pcd_orig.colors)

    # 2) レイキャスト用にノイズ除去
    valid_pts = pts_all.copy()
    if args.outlier:
        logger.info("Statistical Outlier Removal 適用")
        tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(valid_pts))
        tmp, _ = tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        valid_pts = np.asarray(tmp.points)

    # ※ Z制限は行わずにレイキャスト用点群を作成

    # 3) Y–Z断面とレイ原点
    yz     = valid_pts[:, 1:3]
    hull   = ConvexHull(yz)
    origin = yz[hull.vertices].mean(axis=0)
    logger.info(f"Ray origin (Y,Z): {origin}")

    # 4) レイキャスト（壁・斜面）
    steps   = np.arange(0, args.ray_length + args.step, args.step)
    hit_pts, missing = run_raycast(
        yz, valid_pts, origin,
        args.n_rays, steps,
        args.dist_thresh, args.safety_dist
    )
    logger.info(f"ヒット点数: {len(hit_pts)}, 欠損レイ数: {len(missing)}")

    # 5) ボクセル中心にマッピング
    minb   = valid_pts.min(axis=0)
    ijk    = np.floor((hit_pts - minb) / args.voxel_size).astype(int)
    uniq, _= np.unique(ijk, axis=0, return_index=True)
    centers = minb + (uniq + 0.5) * args.voxel_size

    # 6) 高さ上限フィルタ
    z_limit = args.water_level + args.clearance
    before  = centers.shape[0]
    centers = centers[centers[:, 2] <= z_limit]
    logger.info(f"Removed {before - len(centers)} green points above z_limit={z_limit}")

    # 7) 緑点生成＆PLY出力
    green    = np.tile([0.0, 1.0, 0.0], (len(centers), 1))
    merged_pc = o3d.geometry.PointCloud()
    merged_pc.points = o3d.utility.Vector3dVector(np.vstack([pts_all, centers]))
    merged_pc.colors = o3d.utility.Vector3dVector(np.vstack([cols_all, green]))
    os.makedirs(os.path.dirname(args.output_ply), exist_ok=True)
    o3d.io.write_point_cloud(args.output_ply, merged_pc)
    logger.info(f"出力完了: {args.output_ply}")

if __name__ == "__main__":
    main()