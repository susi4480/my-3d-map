# -*- coding: utf-8 -*-
"""
M0の航行可能空間(緑点) → Voxel Occupancy → 3Dモルフォロジーで充填 → Marching Cubes で外殻メッシュ化
- 入力: M0出力LAS（緑点）
- 出力: 厚みある3D外殻メッシュ（PLY）

【背景】
3D α-shape は点群がほぼ平面/線状だと数値的に不安定になりやすい。
そのため、本スクリプトは点群を一旦ボクセル占有（3D）に変換し、
3Dのモルフォロジー（閉処理・膨張・穴埋め）で「中身のある体積」に整形した後、
Marching Cubes で安定的に外殻メッシュを抽出する。

【必要ライブラリ】
pip install numpy laspy open3d scikit-image scipy

【パラメータのコツ】
- VOXEL_SIZE_M: 小さいほど精細だがメモリ使用量が増える。0.25〜0.8mあたりから調整。
- MORPH_CLOSE_RAD_M: 小さな隙間を塞ぐ半径（m）。点群が疎らなら少し大きめに。
- DILATE_RAD_M: 航行空間の“厚み”付与。板状/縁だけの点群でも体積化できる。
- FILL_HOLES: 立体内部の空洞を埋めて、より“殻”らしくする。

【出力】
- PLYメッシュ（ワイヤーフレームではなく厚みのある外殻）
"""

import os
import math
import numpy as np
import laspy
import open3d as o3d
from skimage import measure
from scipy import ndimage as ndi

# ========= 入出力 =========
INPUT_LAS  = r"/data/0908_M0onM5_voxel_style.las"   # M0出力（緑点）
OUTPUT_PLY = r"/output/0910_M0_voxel_marchingcubes_mesh.ply"

# ========= 基本パラメータ =========
VOXEL_SIZE_M        = 0.05     # 1ボクセルの一辺[m]（自動調整の下限候補にもなる）
TARGET_MAX_CELLS    = 60_000_000  # 3D配列の最大セル数目安（超えるとVOXEL_SIZEを自動で粗く）
BBOX_PADDING_VOX    = 2       # バウンディングボックスの外側に余白（ボクセル単位）

# 3Dモルフォロジー（半径[m]で指定）※データに応じて調整
MORPH_CLOSE_RAD_M   = 0.2     # 3Dクロージング半径（小さな隙間を塞ぐ）
DILATE_RAD_M        = 0.1     # 3D膨張（厚みの付与・線/縁を体積化）
FILL_HOLES          = False    # 3D穴埋め（内部空洞の充填）

# ========= 便利関数 =========
def sphere_struct(radius_vox: int) -> np.ndarray:
    """半径[ボクセル]の球状構造要素（bool 3D）"""
    if radius_vox <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    r = radius_vox
    zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    se = (xx*xx + yy*yy + zz*zz) <= (r*r)
    return se

def autoscale_voxel_size(mins, maxs, voxel_size, target_cells):
    """BBOXと希望セル数から、過大メモリを避けるためのボクセルサイズを自動調整"""
    Lx, Ly, Lz = (maxs - mins).tolist()
    Lx = max(Lx, 1e-9); Ly = max(Ly, 1e-9); Lz = max(Lz, 1e-9)
    # 体積/目標セル数 の 立方根 ≈ 必要最小ボクセルサイズの推定
    s_min = ( (Lx*Ly*Lz) / max(target_cells,1) ) ** (1.0/3.0)
    s = max(voxel_size, s_min * 1.2)  # 少し余裕を見て拡大
    return float(s)

def main():
    # === 1) LAS読み込み ===
    print("📥 LAS読み込み中...")
    las = laspy.read(INPUT_LAS)
    pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    print(f"✅ 入力点数: {len(pts):,}")

    # === 2) BBOXとボクセルサイズ自動調整 ===
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    voxel = autoscale_voxel_size(mins, maxs, VOXEL_SIZE_M, TARGET_MAX_CELLS)
    print(f"🧮 自動調整後のボクセルサイズ: {voxel:.3f} m")

    # === 3) 3Dグリッド（Z,Y,Xの順）を確保 ===
    # skimage.measure.marching_cubes は (z,y,x) 形状を前提とするため、ボリュームは (Nz,Ny,Nx)
    nx = int(math.floor((maxs[0] - mins[0]) / voxel)) + 1 + 2*BBOX_PADDING_VOX
    ny = int(math.floor((maxs[1] - mins[1]) / voxel)) + 1 + 2*BBOX_PADDING_VOX
    nz = int(math.floor((maxs[2] - mins[2]) / voxel)) + 1 + 2*BBOX_PADDING_VOX

    # メモリ目安の表示（boolで約1byte/セル）
    est_mem_mb = (nx*ny*nz) / (1024**2)
    print(f"📦 3Dボリューム形状: (Nz,Ny,Nx)=({nz:,}, {ny:,}, {nx:,}) ≈ {est_mem_mb:.1f} MB (bool)")

    vol = np.zeros((nz, ny, nx), dtype=bool)

    # === 4) 点群 → 占有ボクセル化 ===
    # インデックス（X→ix, Y→iy, Z→iz）→ (z,y,x) の vol[iz,iy,ix] を True
    ox, oy, oz = mins[0] - BBOX_PADDING_VOX*voxel, mins[1] - BBOX_PADDING_VOX*voxel, mins[2] - BBOX_PADDING_VOX*voxel
    ix = np.floor((pts[:,0] - ox) / voxel).astype(np.int64)
    iy = np.floor((pts[:,1] - oy) / voxel).astype(np.int64)
    iz = np.floor((pts[:,2] - oz) / voxel).astype(np.int64)

    # 範囲外安全化
    mask = (ix>=0)&(ix<nx)&(iy>=0)&(iy<ny)&(iz>=0)&(iz<nz)
    ix, iy, iz = ix[mask], iy[mask], iz[mask]
    vol[iz, iy, ix] = True
    print(f"✅ 占有ボクセル数（元点由来）: {vol.sum():,}")

    # === 5) 3Dモルフォロジー（閉処理→膨張→穴埋め）===
    #   目的: 線/縁だけの点群でも“中身の詰まった体積”へ変換し、外殻を安定抽出
    close_rad_vox  = max(1, int(round(MORPH_CLOSE_RAD_M / voxel)))
    dilate_rad_vox = max(0, int(round(DILATE_RAD_M     / voxel)))
    print(f"🧱 3Dクロージング半径: {close_rad_vox} vox, 3D膨張半径: {dilate_rad_vox} vox")

    if close_rad_vox > 0:
        se_close = sphere_struct(close_rad_vox)
        vol = ndi.binary_closing(vol, structure=se_close)

    if dilate_rad_vox > 0:
        se_dil = sphere_struct(dilate_rad_vox)
        vol = ndi.binary_dilation(vol, structure=se_dil)

    if FILL_HOLES:
        # N次元対応の穴埋め
        vol = ndi.binary_fill_holes(vol)

    print(f"✅ 体積ボクセル数（整形後）: {vol.sum():,}")

    # === 6) Marching Cubes（外殻抽出）===
    # skimage は (z,y,x) 入力を前提。spacing は各軸の物理スケール[m]
    print("🔺 Marching Cubes 実行中...")
    verts_zyx, faces, normals_zyx, _ = measure.marching_cubes(vol.astype(np.float32),
                                                               level=0.5,
                                                               spacing=(voxel, voxel, voxel),
                                                               allow_degenerate=False)
    # (z,y,x) → (x,y,z) に並べ替え & オフセットを付加
    verts_xyz = np.column_stack([
        verts_zyx[:, 2] + ox,
        verts_zyx[:, 1] + oy,
        verts_zyx[:, 0] + oz,
    ])
    print(f"✅ メッシュ: 頂点 {len(verts_xyz):,}, 面 {len(faces):,}")

    # === 7) Open3Dメッシュ化 & クリーンアップ ===
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts_xyz.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    # クリーニング（縮退/重複/非多様体）
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # === 8) 書き出し ===
    ok = o3d.io.write_triangle_mesh(OUTPUT_PLY, mesh)
    if not ok:
        raise RuntimeError("PLY書き出しに失敗しました。パス/書込権限をご確認ください。")
    print(f"💾 出力完了: {OUTPUT_PLY}")
    print("🎉 Voxel + Marching Cubes による厚みある3D外殻メッシュ生成 完了！")

if __name__ == "__main__":
    main()
