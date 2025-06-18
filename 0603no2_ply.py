import open3d as o3d
import numpy as np

# PLY読み込み
pcd = o3d.io.read_point_cloud(r"C:\Users\user\Documents\lab\data\pond\las_output\MBES_02_mls_like_-2.5.ply")
points = np.asarray(pcd.points)
print(f"🔢 元の点数: {len(points):,}")

# 重複削除
_, idx = np.unique(points, axis=0, return_index=True)
deduped_pcd = pcd.select_by_index(idx)
print(f"✅ 重複除去後の点数: {len(idx):,}")
print(f"🗑️ 重複していた点数: {len(points) - len(idx):,}")

# 保存
o3d.io.write_point_cloud(
    r"C:\Users\user\Documents\lab\data\pond\las_output\MBES_02_mls_like_-2.5_deduped.ply",
    deduped_pcd
)
