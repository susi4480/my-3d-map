# -*- coding: utf-8 -*-
"""
統合処理パイプライン：M0スライス縁点から
① 横線生成
② メッシュ化（三角形）
③ 3Dシェル構築
④ 属性付きメッシュ出力
⑤ Volume（占有点）化

出力：PLY (可視化用), LAS (属性・体積点群)
Poisson補間は使用しない
"""

import numpy as np
import open3d as o3d
import laspy
import os

# ===== 入出力 =====
INPUT_GREEN_NPY = "/output/0901no3_M0_connected_rect_edges.las"  # 各スライスごとの縁点（順序あり）: List[np.ndarray(N,3)]
OUTPUT_PLY_LINES = "/output/0901_M0step1_lines.ply"
OUTPUT_PLY_MESH = "/output/0901_M0step2_mesh.ply"
OUTPUT_PLY_SHELL = "/output/0901_M0step3_shell.ply"
OUTPUT_LAS_ATTR  = "/output/0901_M0step4_attr_meshpoints.las"
OUTPUT_LAS_VOL   = "/output/0901_M0step5_volume_points.las"

# ===== ユーティリティ =====
def save_ply_points(path, points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)
    print(f"✅ PLY出力: {path} 点数: {len(points)}")

def save_ply_mesh(path, vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)
    print(f"✅ メッシュPLY出力: {path} 三角形数: {len(triangles)}")

def save_las(path, points, attr=None):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:,0], points[:,1], points[:,2]
    if attr is not None:
        las.classification = attr.astype(np.uint8)
    las.write(path)
    print(f"✅ LAS出力: {path} 点数: {len(points)}")

# ===== ステップ1: 横線生成 =====
def step1_generate_lines(slices):
    lines = []
    for u in range(len(slices)-1):
        A = slices[u]
        B = slices[u+1]
        N = min(len(A), len(B))
        for i in range(N):
            lines.append(A[i])
            lines.append(B[i])
    save_ply_points(OUTPUT_PLY_LINES, np.array(lines))

# ===== ステップ2: メッシュ化（三角形） =====
def step2_generate_mesh(slices):
    vertices = []
    triangles = []
    idx = 0
    for u in range(len(slices)-1):
        A = slices[u]
        B = slices[u+1]
        N = min(len(A), len(B))
        for i in range(N-1):
            p0 = A[i];   p1 = A[i+1]
            q0 = B[i];   q1 = B[i+1]
            vertices.extend([p0, p1, q1, q0])
            triangles.append([idx, idx+1, idx+2])
            triangles.append([idx, idx+2, idx+3])
            idx += 4
    save_ply_mesh(OUTPUT_PLY_MESH, np.array(vertices), np.array(triangles))

# ===== ステップ3: シェル構築（スライス全体を囲う） =====
def step3_generate_shell(slices):
    shell_pts = []
    shell_pts.extend(slices[0])            # 前面
    shell_pts.extend(slices[-1])           # 背面
    for s in slices:                        # 側面
        shell_pts.append(s[0])             # 左端
        shell_pts.append(s[-1])            # 右端
    save_ply_points(OUTPUT_PLY_SHELL, np.array(shell_pts))

# ===== ステップ4: 属性付きメッシュ点群出力 =====
def step4_export_attr_points(slices):
    pts = []
    cls = []
    for u, s in enumerate(slices):
        pts.extend(s)
        cls.extend([u]*len(s))  # u = スライス番号をクラス属性に
    save_las(OUTPUT_LAS_ATTR, np.array(pts), np.array(cls))

# ===== ステップ5: Volume化（占有点） =====
def step5_export_volume_points(slices):
    vol_pts = []
    for u in range(len(slices)-1):
        A = slices[u]
        B = slices[u+1]
        N = min(len(A), len(B))
        for i in range(N):
            mid = 0.5 * (A[i] + B[i])
            vol_pts.append(mid)
    save_las(OUTPUT_LAS_VOL, np.array(vol_pts))

# ===== メイン =====
def main():
    if not os.path.exists(INPUT_GREEN_NPY):
        raise FileNotFoundError(INPUT_GREEN_NPY)
    slices = np.load(INPUT_GREEN_NPY, allow_pickle=True)  # List of np.ndarray(N,3)
    step1_generate_lines(slices)
    step2_generate_mesh(slices)
    step3_generate_shell(slices)
    step4_export_attr_points(slices)
    step5_export_volume_points(slices)

if __name__ == '__main__':
    main()
