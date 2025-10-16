# -*- coding: utf-8 -*-
"""
【機能】
M0スライス → 長方形縁点抽出 → 線 → メッシュ → シェル → 属性点群 → Volume出力
エラー対策済み：stackエラー対応、None対策、空スライス回避、出力前チェック、make_slicesの初期化修正
"""

import os
import numpy as np
import laspy
import cv2
from scipy.ndimage import label
import open3d as o3d

# === パラメータ ===
INPUT_LAS = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_LAS = "/output/0901no3_M0_connected_rect_edges.las"
BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
SECTION_INTERVAL = 0.5
SLICE_THICKNESS = 0.20
Z_LIMIT = 1.9
MIN_RECT_SIDE = 5
VOXEL_SIZE = 0.05
LINE_LENGTH = 60.0

OUTPUT_PLY_LINES = "/output/0901_M0step1_lines.ply"
OUTPUT_PLY_MESH = "/output/0901_M0step2_mesh.ply"
OUTPUT_PLY_SHELL = "/output/0901_M0step3_shell.ply"
OUTPUT_LAS_ATTR  = "/output/0901_M0step4_attr_meshpoints.las"
OUTPUT_LAS_VOL   = "/output/0901_M0step5_volume_points.las"

for path in [OUTPUT_LAS, OUTPUT_PLY_LINES, OUTPUT_PLY_MESH, OUTPUT_PLY_SHELL, OUTPUT_LAS_ATTR, OUTPUT_LAS_VOL]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# === 関数群 ===
def make_slices(XYZ):
    centers, slices = [], []
    for x in np.arange(XYZ[:, 0].min(), XYZ[:, 0].max(), SECTION_INTERVAL):
        pts = XYZ[(XYZ[:, 0] >= x - BIN_X / 2) & (XYZ[:, 0] <= x + BIN_X / 2) & (XYZ[:, 2] <= Z_LIMIT)]
        if len(pts) < MIN_PTS_PER_XBIN:
            centers.append(None); slices.append(None)
            continue
        cx, y_mean = x, np.mean(pts[:, 1])
        centers.append(np.array([cx, y_mean, 0]))
        mask = np.abs(XYZ[:, 0] - cx) <= SLICE_THICKNESS / 2
        slices.append(XYZ[mask])
    return centers, slices

def project_to_vz(points, c):
    bin_v = int(np.ceil(LINE_LENGTH / VOXEL_SIZE))
    bin_z = int(np.ceil((Z_LIMIT + 2.0) / VOXEL_SIZE))
    grid = np.zeros((bin_z, bin_v), dtype=np.uint8)
    for p in points:
        r = p - c
        v = int(np.floor((r[1] + LINE_LENGTH / 2) / VOXEL_SIZE))
        z = int(np.floor((p[2] + 2.0) / VOXEL_SIZE))
        if 0 <= v < bin_v and 0 <= z < bin_z:
            grid[z, v] = 1
    return grid

def downfill(free):
    filled = np.copy(free)
    for v in range(free.shape[1]):
        ones = np.where(free[:, v])[0]
        if len(ones) >= 2:
            filled[ones[0]:ones[-1]+1, v] = True
    return filled

def rectangles_on_slice(free_bitmap):
    h, w = free_bitmap.shape
    max_area, best = 0, []
    for top in range(h):
        for left in range(w):
            if not free_bitmap[top, left]: continue
            for bottom in range(top + MIN_RECT_SIDE, h):
                for right in range(left + MIN_RECT_SIDE, w):
                    if np.all(free_bitmap[top:bottom, left:right]):
                        area = (bottom - top) * (right - left)
                        if area > max_area:
                            max_area = area; best = [(top, left, bottom, right)]
    edges = []
    for top, left, bottom, right in best:
        for v in range(left, right):
            edges.append((v, top)); edges.append((v, bottom-1))
        for z in range(top, bottom):
            edges.append((left, z)); edges.append((right-1, z))
    return edges

def vz_to_world_on_slice(vz, c):
    v, z = vz
    z = z * VOXEL_SIZE - 2.0
    y = v * VOXEL_SIZE - LINE_LENGTH / 2
    x = c[0]
    return np.array([x, y, z])

def write_green_las_with_classification(points_with_class, output_las):
    if len(points_with_class) == 0:
        print("⚠️ No green points to write."); return
    xyz = np.array([p for p,_ in points_with_class])
    cls = np.array([c for _,c in points_with_class], dtype=np.uint8)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0); header.scales = np.array([0.001,0.001,0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:,0], xyz[:,1], xyz[:,2]
    las.red = np.zeros(len(xyz), dtype=np.uint16)
    las.green = np.full(len(xyz), 65535, dtype=np.uint16)
    las.blue = np.zeros(len(xyz), dtype=np.uint16)
    las.classification = cls
    las.write(output_las)

def save_ply_points(path, points):
    if points is None or len(points)==0:
        print(f"⚠️ Empty point list for {path}"); return
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

def save_ply_mesh(path, vertices, triangles):
    if len(vertices)==0 or len(triangles)==0:
        print(f"⚠️ Empty mesh data for {path}"); return
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)

def save_las(path, points, attr=None):
    if points is None or len(points)==0:
        print(f"⚠️ No points to write for {path}"); return
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0); header.scales = np.array([0.001,0.001,0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:,0], points[:,1], points[:,2]
    if attr is not None: las.classification = attr.astype(np.uint8)
    las.write(path)

# === メイン処理 ===
las = laspy.read(INPUT_LAS)
XYZ = np.vstack([las.x, las.y, las.z]).T
centers, slices = make_slices(XYZ)

bitmap_stack=[]
for c, pts in zip(centers, slices):
    if c is None or pts is None:
        bitmap_stack.append(None); continue
    raw = project_to_vz(pts, c)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    free = np.logical_and(~closed.astype(bool), raw.astype(bool))
    filled = downfill(free)
    bitmap_stack.append(filled)

valid_bitmap = next((b for b in bitmap_stack if b is not None), None)
if valid_bitmap is None: raise RuntimeError("❌ 全てのスライスが空です。")
h,w = valid_bitmap.shape
empty_bitmap = np.zeros((h,w), dtype=bool)
stack = np.stack([b if b is not None else empty_bitmap for b in bitmap_stack])

labels,num = label(stack)
max_label = np.argmax(np.bincount(labels.flatten())[1:])+1 if num>0 else 0

GREEN=[]; slices_pts=[]
for u,(c,bmap) in enumerate(zip(centers,bitmap_stack)):
    if bmap is None or np.all(labels[u]!=max_label):
        slices_pts.append([]); continue
    edges = rectangles_on_slice(bmap)
    one_slice=[]
    for e in edges:
        pt = vz_to_world_on_slice(e,c)
        GREEN.append((pt,u)); one_slice.append(pt)
    slices_pts.append(one_slice)

write_green_las_with_classification(GREEN, OUTPUT_LAS)

# === 各種出力 ===
def step1_generate_lines(slices):
    pts=[]
    for u in range(len(slices)-1):
        A,B = slices[u],slices[u+1]
        if not A or not B: continue
        N=min(len(A),len(B))
        for i in range(N):
            pts.append(A[i]); pts.append(B[i])
    save_ply_points(OUTPUT_PLY_LINES, np.array(pts))

def step2_generate_mesh(slices):
    vertices=[]; triangles=[]; idx=0
    for u in range(len(slices)-1):
        A,B=slices[u],slices[u+1]
        if len(A)<2 or len(B)<2: continue
        N=min(len(A),len(B))
        for i in range(N-1):
            p0,p1=A[i],A[i+1]; q0,q1=B[i],B[i+1]
            vertices.extend([p0,p1,q1,q0])
            triangles.append([idx,idx+1,idx+2])
            triangles.append([idx,idx+2,idx+3])
            idx+=4
    if vertices and triangles:
        save_ply_mesh(OUTPUT_PLY_MESH,np.array(vertices),np.array(triangles))
    else:
        print("⚠️ No mesh generated.")

def step3_generate_shell(slices):
    shell=[]
    if slices:
        if slices[0]: shell+=slices[0]
        if slices[-1]: shell+=slices[-1]
        for s in slices:
            if s: shell.append(s[0]); shell.append(s[-1])
    save_ply_points(OUTPUT_PLY_SHELL,np.array(shell))

def step4_export_attr_points(slices):
    pts=[]; cls=[]
    for u,s in enumerate(slices):
        if not s: continue
        pts.extend(s); cls.extend([u]*len(s))
    if pts:
        save_las(OUTPUT_LAS_ATTR,np.array(pts),np.array(cls))

def step5_export_volume_points(slices):
    pts=[]
    for u in range(len(slices)-1):
        A,B=slices[u],slices[u+1]
        if not A or not B: continue
        N=min(len(A),len(B))
        for i in range(N):
            pts.append(0.5*(A[i]+B[i]))
    if pts:
        save_las(OUTPUT_LAS_VOL,np.array(pts))

step1_generate_lines(slices_pts)
step2_generate_mesh(slices_pts)
step3_generate_shell(slices_pts)
step4_export_attr_points(slices_pts)
step5_export_volume_points(slices_pts)
