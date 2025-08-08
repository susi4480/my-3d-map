# -*- coding: utf-8 -*-
"""
【機能】
- 補間済みPLYファイルから点群を読み込み
- ノイズ除去（統計的外れ値除去＋ボクセルダウンサンプル）
- Z ≤ 3.5m の点だけを抽出し、Y–Z平面に投影
- Gift Wrap（Jarvis March）で Convex Hull を計算
- Convex Hull を 3D ワイヤーフレーム化し、元点群と合成して出力
"""

import numpy as np
import open3d as o3d

def orientation(p, q, r):
    # 2D クロスプロダクトの符号
    return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])

def jarvis_march(points2d):
    n = len(points2d)
    if n < 3:
        return list(range(n))
    hull = []
    leftmost = np.argmin(points2d[:,0])
    p = leftmost
    while True:
        hull.append(p)
        q = (p + 1) % n
        for r in range(n):
            if orientation(points2d[p], points2d[q], points2d[r]) < 0:
                q = r
        p = q
        if p == leftmost:
            break
    return hull

# === 入出力設定 ===
input_ply  = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\slice_x_387183_L_only.ply"
output_ply = r"C:\Users\user\Documents\lab\output_ply\slice_area_navigation\convex_hull_below_3_5m.ply"

# === パラメータ ===
Z_LIMIT       = 3.5    # Convex Hull をかける上限Z
nb_neighbors  = 20     # 統計的外れ値除去の近傍点数
std_ratio     = 2.0    # 外れ値閾値（標準偏差倍数）
voxel_size    = 0.1    # ボクセルダウンサンプルサイズ
n_wire_points = 50     # ワイヤーフレーム各辺の分割数

# 1. 点群読み込み
pcd = o3d.io.read_point_cloud(input_ply)

# 2. ノイズ除去
pcd, _ = pcd.remove_statistical_outlier(
    nb_neighbors=nb_neighbors,
    std_ratio=std_ratio
)
# 3. 密度均一化
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

# 4. Z ≤ Z_LIMIT の点を抽出
pts  = np.asarray(pcd.points)
cols = np.asarray(pcd.colors)
mask = pts[:,2] <= Z_LIMIT
pts_f = pts[mask]
cols_f= cols[mask]
if len(pts_f) < 3:
    raise RuntimeError("❌ Z ≤ 3.5m の点が十分にありません")

# 5. Y–Z平面に投影
yz = pts_f[:, [1,2]]

# 6. Convex Hull via Jarvis March
hull_idx = jarvis_march(yz)
hull2d   = yz[hull_idx]

# 7. 3Dワイヤーフレーム化 (X はスライスの平均X)
x0     = pts_f[:,0].mean()
hull3d = np.column_stack([
    np.full(len(hull2d), x0),
    hull2d[:,0],
    hull2d[:,1]
])

wire_pts = []
for i in range(len(hull3d)):
    p = hull3d[i]
    q = hull3d[(i+1) % len(hull3d)]
    wire_pts.append(np.linspace(p, q, n_wire_points))
wire_pts = np.vstack(wire_pts)

# 8. 点群合成＆出力
pcd_clean = o3d.geometry.PointCloud()
pcd_clean.points = o3d.utility.Vector3dVector(pts_f)
pcd_clean.colors = o3d.utility.Vector3dVector(cols_f)

pcd_wire = o3d.geometry.PointCloud()
pcd_wire.points = o3d.utility.Vector3dVector(wire_pts)
pcd_wire.colors = o3d.utility.Vector3dVector(
    np.tile([1.0, 0.0, 0.0], (len(wire_pts), 1))
)

pcd_all = pcd_clean + pcd_wire
o3d.io.write_point_cloud(output_ply, pcd_all)

print(f"✅ Z ≤ {Z_LIMIT}m の点群に対する Convex Hull を出力しました: {output_ply}")
