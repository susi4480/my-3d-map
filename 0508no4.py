import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\suidoubasi\floor"
output_dir = os.path.join(las_dir, "poisson_final")
os.makedirs(output_dir, exist_ok=True)

z_limit = 4.5           # 川底とみなすZ上限
depth = 10              # Poissonの解像度
density_thresh = 0.01   # 下位X%を補間領域と判断（赤色に）

# === 処理 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
for las_path in las_files:
    base = os.path.splitext(os.path.basename(las_path))[0]
    print(f"\n--- 処理中: {base} ---")

    # [1] LAS読み込み & フィルタ
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    z = points[:, 2]
    mask = (~np.isnan(z)) & (z > -1000) & (z < z_limit)
    points_floor = points[mask]
    if len(points_floor) < 1000:
        print(" ⚠ 有効点が少ないためスキップ")
        continue

    # [2] 点群変換と法線推定
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_floor)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
    pcd.normalize_normals()

    # [3] 上向き法線だけ抽出
    normals = np.asarray(pcd.normals)
    up_mask = normals[:, 2] > 0.85
    pcd = pcd.select_by_index(np.where(up_mask)[0])
    if len(pcd.points) < 1000:
        print(" ⚠ 上向き法線が少ないためスキップ")
        continue

    # [4] Poisson補間
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        print(" ✅ Poissonメッシュ生成完了")
    except Exception as e:
        print(f" ❌ Poisson補間失敗: {e}")
        continue

    # [5] 色付け（赤：補間っぽい、白：信頼できる）
    densities = np.asarray(densities)
    red_mask = densities < np.quantile(densities, density_thresh)
    colors = np.tile([1.0, 1.0, 1.0], (len(densities), 1))  # 白
    colors[red_mask] = [1.0, 0.0, 0.0]  # 赤

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # [6] 保存
    out_path = os.path.join(output_dir, base + "_poisson_final.ply")
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f" ✅ 出力完了: {out_path}")


