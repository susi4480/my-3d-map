# -*- coding: utf-8 -*-
"""
M4方式：属性点群出力（LAS 1.4対応）
-----------------------------------
【機能】
- LAS入力を読み込み
- X方向に一定間隔でスライスを作成
- 各スライスから航行可能長方形の縁点を抽出
- 隣接スライスの対応点を結んでメッシュ化
- 各スライスの点群に「スライス番号」を classification として付与
- 出力は：
  - メッシュPLY
  - 属性LAS（スライス番号属性付き）
-----------------------------------
"""

import os
import numpy as np
import laspy
import cv2
import open3d as o3d

# ===== パラメータ =====
INPUT_LAS      = "/data/0828_01_500_suidoubasi_ue.las"
OUTPUT_PLY_MESH = "/output/0908M4_mesh.ply"
OUTPUT_LAS_ATTR = "/output/0908M4_attr_points.las"

BIN_X = 2.0
MIN_PTS_PER_XBIN = 50
SECTION_INTERVAL = 0.5
SLICE_THICKNESS = 0.20
Z_LIMIT = 1.9
LINE_LENGTH = 60.0
VOXEL_SIZE = 0.05
MIN_RECT_SIDE = 5

os.makedirs(os.path.dirname(OUTPUT_PLY_MESH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_LAS_ATTR), exist_ok=True)

# ===== 関数群 =====
def make_slices(XYZ):
    centers, slices = [], []
    for x in np.arange(XYZ[:,0].min(), XYZ[:,0].max(), SECTION_INTERVAL):
        pts = XYZ[(XYZ[:,0]>=x-BIN_X/2)&(XYZ[:,0]<=x+BIN_X/2)&(XYZ[:,2]<=Z_LIMIT)]
        if len(pts)<MIN_PTS_PER_XBIN:
            centers.append(None); slices.append(None)
            continue
        cx, y_mean = x, np.mean(pts[:,1])
        centers.append(np.array([cx,y_mean,0]))
        mask = np.abs(XYZ[:,0]-cx) <= SLICE_THICKNESS/2
        slices.append(XYZ[mask])
    return centers,slices

def project_to_vz(points,c):
    bin_v=int(np.ceil(LINE_LENGTH/VOXEL_SIZE))
    bin_z=int(np.ceil((Z_LIMIT+2.0)/VOXEL_SIZE))
    grid=np.zeros((bin_z,bin_v),dtype=np.uint8)
    for p in points:
        r=p-c
        v=int(np.floor((r[1]+LINE_LENGTH/2)/VOXEL_SIZE))
        z=int(np.floor((p[2]+2.0)/VOXEL_SIZE))
        if 0<=v<bin_v and 0<=z<bin_z:
            grid[z,v]=1
    return grid

def downfill(free):
    filled=np.copy(free)
    for v in range(free.shape[1]):
        ones=np.where(free[:,v])[0]
        if len(ones)>=2: filled[ones[0]:ones[-1]+1,v]=True
    return filled

def rectangles_on_slice(free_bitmap):
    h,w=free_bitmap.shape
    max_area,best=0,[]
    for top in range(h):
        for left in range(w):
            if not free_bitmap[top,left]: continue
            for bottom in range(top+MIN_RECT_SIDE,h):
                for right in range(left+MIN_RECT_SIDE,w):
                    if np.all(free_bitmap[top:bottom,left:right]):
                        area=(bottom-top)*(right-left)
                        if area>max_area:
                            max_area=area; best=[(top,left,bottom,right)]
    if not best:
        return []
    edges=[]
    for top,left,bottom,right in best:
        for v in range(left,right):
            edges.append((v,top)); edges.append((v,bottom-1))
        for z in range(top,bottom):
            edges.append((left,z)); edges.append((right-1,z))
    return edges

def vz_to_world(vz,c):
    v,z=vz
    z=z*VOXEL_SIZE-2.0
    y=v*VOXEL_SIZE-LINE_LENGTH/2
    x=c[0]
    return np.array([x,y,z])

def save_las_attr(points, cls, path):
    header=laspy.LasHeader(point_format=6, version="1.4")  # LAS 1.4, PF=6
    header.offsets=points.min(axis=0); header.scales=[0.001,0.001,0.001]
    las_out=laspy.LasData(header)
    las_out.x,las_out.y,las_out.z=points[:,0],points[:,1],points[:,2]
    las_out.classification=cls.astype(np.uint16)  # 16bit対応
    las_out.write(path)
    print(f"✅ 属性LAS出力: {path} 点数: {len(points)}")

def save_ply_mesh(path,vertices,triangles):
    if len(vertices)==0 or len(triangles)==0:
        print("⚠️ メッシュなし"); return
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(vertices)
    mesh.triangles=o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path,mesh)
    print(f"✅ メッシュ出力: {path} 三角形数: {len(triangles)}")

# ===== メイン処理 =====
print("📥 LAS読み込み中...")
las=laspy.read(INPUT_LAS)
XYZ=np.vstack([las.x,las.y,las.z]).T

centers,slices=make_slices(XYZ)

slices_pts=[]
for u,(c,pts) in enumerate(zip(centers,slices)):
    if c is None or pts is None:
        slices_pts.append([]); continue
    raw=project_to_vz(pts,c)
    closed=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    free=np.logical_and(~closed.astype(bool),raw.astype(bool))
    filled=downfill(free)
    edges=rectangles_on_slice(filled)
    one_slice=[]
    for e in edges:
        pt=vz_to_world(e,c)
        one_slice.append(pt)
    slices_pts.append(one_slice)

# Step2: メッシュ生成
vertices=[]; triangles=[]; idx=0
for u in range(len(slices_pts)-1):
    A,B=slices_pts[u],slices_pts[u+1]
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
    print("⚠️ メッシュ生成なし")

# Step4: 属性点群（スライス番号を classification に格納）
attr_pts=[]; attr_cls=[]
for u,s in enumerate(slices_pts):
    if not s: continue
    attr_pts.extend(s)
    attr_cls.extend([u]*len(s))
if attr_pts:
    save_las_attr(np.array(attr_pts), np.array(attr_cls), OUTPUT_LAS_ATTR)
else:
    print("⚠️ 属性LASなし")

print("🎉 M4処理完了")
