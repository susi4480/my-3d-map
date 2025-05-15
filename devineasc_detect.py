import os
import laspy
import numpy as np
import pandas as pd
import open3d as o3d

# === 設定 ===
las_dir = r"C:\Users\user\Documents\lab\data\las2"
z_threshold_wall = 0.2     # 勾配 > これ → 壁
z_threshold_floor = 0.05   # 勾配 < これ → 床
output_wall_ply = "las_detected_wall.ply"
output_floor_ply = "las_detected_floor.ply"

# === [1] .lasファイル読み込み ===
las_files = [f for f in os.listdir(las_dir) if f.endswith(".las")]
all_points = []

for file in las_files:
    path = os.path.join(las_dir, file)
    las = laspy.read(path)
    points = np.vstack((las.x, las.y, las.z)).T
    all_points.append(points)

all_points = np.vstack(all_points)
print(f"✅ 読み込んだ点数: {len(all_points)} 点")

# === [2] DataFrame化と勾配（dZ）計算 ===
df = pd.DataFrame(all_points, columns=["X", "Y", "Z"])
df_sorted = df.sort_values(by=["X", "Y"]).reset_index(drop=True)
df_sorted["dZ"] = df_sorted["Z"].diff().abs().fillna(0)

# === [3] 壁・床に分類 ===
wall_df = df_sorted[df_sorted["dZ"] > z_threshold_wall]
floor_df = df_sorted[df_sorted["dZ"] < z_threshold_floor]

wall_points = wall_df[["X", "Y", "Z"]].to_numpy()
floor_points = floor_df[["X", "Y", "Z"]].to_numpy()

print(f"✅ 壁点数: {len(wall_points)}")
print(f"✅ 床点数: {len(floor_points)}")

# === [4] Open3DでPLY出力（色付き） ===
wall_pcd = o3d.geometry.PointCloud()
wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
wall_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 赤
o3d.io.write_point_cloud(output_wall_ply, wall_pcd)

floor_pcd = o3d.geometry.PointCloud()
floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
floor_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # 青
o3d.io.write_point_cloud(output_floor_ply, floor_pcd)

print("🎉 完了：壁と床をそれぞれ色付きで出力しました。")


