# -*- coding: utf-8 -*-
"""
M1方式：スライス間の横線生成（LAS 1.4対応）
-----------------------------------
【機能】
- LAS入力を読み込み
- X方向に一定間隔でスライスを作成
- 各スライスから航行可能長方形の縁点を抽出
- 隣接スライスの対応点を線で接続
- 出力は：
  - 緑点LAS（classification=スライス番号, 16bit対応）
  - 横線PLY
-----------------------------------
"""

import os
import numpy as np
import laspy
import cv2
import open3d as o3d

# ===== パラメータ =====
INPUT_LAS  = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0908M1_rect_edges_green.las"
OUTPUT_PLY_LINES = "/output/0908M1_lines.ply"

BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
SECTION_INTERVAL = 0.5
SLICE_THICKNESS = 0.20
Z_LIMIT = 1.9
LINE_LENGTH = 60.0
VOXEL_SIZE = 0.05
MIN_RECT_SIDE = 5

os.makedirs(os.path.dirname(OUTPUT_LAS), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PLY_LINES), exist_ok=True)

# ===== ユーティリティ関数 =====
def make_slices(XYZ):
    centers, slices = [], []
    for x in np.arange(XYZ[:,0].min(), XYZ[:,0].max(), SECTION_INTERVAL):
        pts = XYZ[(XYZ[:,0] >= x-BIN_X/2) & (XYZ[:,0] <= x+BIN_X/2) & (XYZ[:,2] <= Z_LIMIT)]
        if len(pts) < MIN_PTS_PER_XBIN:
            centers.append(None); slices.append(None)
            continue
        cx, y_mean = x, np.mean(pts[:,1])
        centers.append(np.array([cx, y_mean, 0]))
        mask = np.abs(XYZ[:,0] - cx) <= SLICE_THICKNESS/2
        slices.append(XYZ[mask])
    return centers, slices

def project_to_vz(points, c):
    bin_v = int(np.ceil(LINE_LENGTH/VOXEL_SIZE))
    bin_z = int(np.ceil((Z_LIMIT+2.0)/VOXEL_SIZE))
    grid = np.zeros((bin_z, bin_v), dtype=np.uint8)
    for p in points:
        r = p - c
        v = int(np.floor((r[1]+LINE_LENGTH/2)/VOXEL_SIZE))
        z = int(np.floor((p[2]+2.0)/VOXEL_SIZE))
        if 0 <= v < bin_v and 0 <= z < bin_z:
            grid[z,v] = 1
    return grid

def downfill(free):
    filled = np.copy(free)
    for v in range(free.shape[1]):
        ones = np.where(free[:,v])[0]
        if len(ones) >= 2:
            filled[ones[0]:ones[-1]+1, v] = True
    return filled

def rectangles_on_slice(free_bitmap):
    h, w = free_bitmap.shape
    max_area, best = 0, []
    for top in range(h):
        for left in range(w):
            if not free_bitmap[top,left]: continue
            for bottom in range(top+MIN_RECT_SIDE, h):
                for right in range(left+MIN_RECT_SIDE, w):
                    if np.all(free_bitmap[top:bottom, left:right]):
                        area = (bottom-top)*(right-left)
                        if area > max_area:
                            max_area = area
                            best = [(top,left,bottom,right)]
    if not best:
        return []
    edges=[]
    for top,left,bottom,right in best:
        for v in range(left,right):
            edges.append((v,top)); edges.append((v,bottom-1))
        for z in range(top,bottom):
            edges.append((left,z)); edges.append((right-1,z))
    return edges

def vz_to_world(vz, c):
    v, z = vz
    z = z*VOXEL_SIZE - 2.0
    y = v*VOXEL_SIZE - LINE_LENGTH/2
    x = c[0]
    return np.array([x,y,z])

def save_las(points, cls, path):
    """LAS保存（緑点＋classification, 16bit対応）"""
    header = laspy.LasHeader(point_format=6, version="1.4")  # ★ 1.4対応
    header.offsets = np.min(points,axis=0); header.scales = [0.001,0.001,0.001]
    las_out = laspy.LasData(header)
    las_out.x, las_out.y, las_out.z = points[:,0],points[:,1],points[:,2]
    las_out.classification = np.array(cls, dtype=np.uint16)  # ★ 16bit
    las_out.red   = np.zeros(len(points), dtype=np.uint16)
    las_out.green = np.full(len(points), 65535, dtype=np.uint16)
    las_out.blue  = np.zeros(len(points), dtype=np.uint16)
    las_out.write(path)
    print(f"✅ LAS出力: {path} 点数: {len(points)}")

def save_ply_points(path, points):
    if len(points)==0: 
        print("⚠️ 空の点群"); return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)
    print(f"✅ PLY出力: {path} 点数: {len(points)}")

# ===== メイン処理 =====
print("📥 LAS読み込み中...")
las = laspy.read(INPUT_LAS)
XYZ = np.vstack([las.x, las.y, las.z]).T

centers, slices = make_slices(XYZ)

GREEN=[]; CLS=[]; slices_pts=[]
for u,(c,pts) in enumerate(zip(centers,slices)):
    if c is None or pts is None:
        slices_pts.append([]); continue
    raw = project_to_vz(pts,c)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    free = np.logical_and(~closed.astype(bool), raw.astype(bool))
    filled = downfill(free)
    edges = rectangles_on_slice(filled)
    if not edges:
        slices_pts.append([]); continue
    one_slice=[]
    for e in edges:
        pt = vz_to_world(e,c)
        GREEN.append(pt)
        CLS.append(u)   # ← スライス番号そのまま (0〜65535)
        one_slice.append(pt)
    slices_pts.append(one_slice)

if GREEN:
    save_las(np.array(GREEN), np.array(CLS), OUTPUT_LAS)
else:
    print("⚠️ 緑点なし")

# Step1: 横線生成
lines=[]
for u in range(len(slices_pts)-1):
    A,B=slices_pts[u],slices_pts[u+1]
    if not A or not B: continue
    N=min(len(A),len(B))
    for i in range(N):
        lines.append(A[i]); lines.append(B[i])
if lines:
    save_ply_points(OUTPUT_PLY_LINES, np.array(lines))
else:
    print("⚠️ ライン生成なし")

print("🎉 M1処理完了")
