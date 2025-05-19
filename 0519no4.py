import laspy
import numpy as np
import open3d as o3d
import os
import glob

# === 入力ディレクトリ（.las）===
las_dir = r"C:\Users\user\Documents\lab\data\las_field"
output_dir = os.path.join(las_dir, "poisson_colored")
os.makedirs(output_dir, exist_ok=True)

# === Poissonパラメータ ===
z_limit = 4.5      # 川底とみなす最大Z
depth = 10         # Poisson補間の解像度
density_thresh = 0.01  # 補間（低密度）とみなす割合（下位1%を赤）

# === .lasファイルを処理 ===
las_files = glob.glob(os.path.join(las_dir, "*.las"))
if not las_files:
    print("❌ .lasファイルが見つかりません。")
    exit()

for las_path in las_files:
    base = os.path.splitext(os.path.basename(las_path))[0]
    print(f"\n--- 処理中: {base} ---")

    # [1] .las読み込み
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # [2] 川底だけ抽出
    z = points[:, 2]
    mask = (~np.isnan(z)) & (z > -1000) & (z < z_limit)
    points_floor = points[mask]
    if len(points_floor) == 0:
        print(" ⚠ 川底データなし。スキップ")
        continue

    # [3] 点群 → Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_floor)

    # [4] 法線推定
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
    pcd.normalize_normals()

    # [5] Poisson補間
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        print(" ✅ Poissonメッシュ生成完了")
    except Exception as e:
        print(f" ❌ 補間失敗: {e}")
        continue

    # [6] 補間領域の色分け（密度ベース）
    densities = np.asarray(densities)
    red_mask = densities < np.quantile(densities, density_thresh)

    colors = np.tile([1.0, 1.0, 1.0], (len(densities), 1))  # 初期は白
    colors[red_mask] = [1.0, 0.0, 0.0]  # 赤に置換

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # [7] 保存
    out_path = os.path.join(output_dir, base + "_poisson_colored.ply")
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f" ✅ 出力完了: {out_path}")

