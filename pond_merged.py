import numpy as np
import os

# === ファイルパス設定 ===
mbes_path = r"C:\Users\user\Documents\lab\data\pond\MBES_02.xyz"
merlin_path = r"C:\Users\user\Documents\lab\data\pond\Merlin_02.xyz"
output_path = r"C:\Users\user\Documents\lab\data\pond\merged_pond.xyz"

# === ファイル読み込み ===
try:
    mbes = np.loadtxt(mbes_path)
    merlin = np.loadtxt(merlin_path)
    print(f"✅ MBES 点数: {len(mbes):,}")
    print(f"✅ Merlin 点数: {len(merlin):,}")
except Exception as e:
    print(f"❌ 読み込みエラー: {e}")
    exit()

# === 結合 ===
merged = np.vstack((mbes, merlin))
print(f"🔗 結合完了: 合計 {len(merged):,} 点")

# === 保存 ===
try:
    np.savetxt(output_path, merged, fmt="%.8f")
    print(f"✅ 保存完了: {output_path}")
except Exception as e:
    print(f"❌ 保存エラー: {e}")
