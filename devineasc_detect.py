import numpy as np
import pandas as pd
import open3d as o3d

# === 設定：読み込む .asc ファイルのパス ===
asc_path = r"C:\Users\user\Documents\lab\devine_data\decvine_data1.asc"
z_threshold_wall = 0.2   # 勾配 > これ → 壁
z_threshold_floor = 0.05  # 勾配 < これ → 底面

# === 安全なエンコーディング読み込み（数値行だけ取り出す） ===
valid_lines = []
with open(asc_path, "rb") as f:
    for line in f:
        try:
            decoded = line.decode("utf-8").strip()  # まずはUTF-8で試みる
            parts = decoded.split(",")
            if len(parts) == 3:
                nums = [float(p) for p in parts]
                valid_lines.append(nums)
        except:
            continue  # デコード失敗や数値でない行はスキップ

data = np.array(valid_lines)

# === データをDataFrameに変換 ===
df = pd.DataFrame(data, columns=["X", "Y", "Z"])

# === Zの勾配（差分の絶対値）を計算 ===
df_sorted = df.sort_values(by=["X", "Y"]).reset_index(drop=True)
df_sorted["dZ"] = df_sorted["Z"].diff().abs().fillna(0)

# === 壁と底面を抽出 ===
wall_df = df_sorted[df_sorted["dZ"] > z_threshold_wall]
floor_df = df_sorted[df_sorted["dZ"] < z_threshold_floor]

# === 点群形式へ変換
wall_points = wall_df[["X", "Y", "Z"]].to_numpy()
floor_points = floor_df[["X", "Y", "Z"]].to_numpy()

# === Open3DでPLYファイル出力（壁）
wall_pcd = o3d.geometry.PointCloud()
wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
o3d.io.write_point_cloud("asc_detected_wall.ply", wall_pcd)

# === Open3DでPLYファイル出力（底面）
floor_pcd = o3d.geometry.PointCloud()
floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
o3d.io.write_point_cloud("asc_detected_floor.ply", floor_pcd)

# === 壁＋底面の点群で直方体（AxisAlignedBoundingBox）作成
combined = np.vstack((wall_points, floor_points))
pcd_combined = o3d.geometry.PointCloud()
pcd_combined.points = o3d.utility.Vector3dVector(combined)

bbox = pcd_combined.get_axis_aligned_bounding_box()
bbox.color = (1, 0, 0)  # 赤い直方体

# === 直方体の点群出力（視覚化用）
box_points = bbox.get_box_points()
box_pcd = o3d.geometry.PointCloud()
box_pcd.points = o3d.utility.Vector3dVector(box_points)
o3d.io.write_point_cloud("asc_box_wall_floor.ply", box_pcd)

print("✅ 壁・床・直方体のPLYファイルを出力しました。")
print("壁の点数:", len(wall_points))
print("床の点数:", len(floor_points))

