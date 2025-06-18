import pandas as pd
from pyproj import Transformer
import os
import glob

# === 入力と出力フォルダー ===
input_dir = r"C:\Users\user\Documents\lab\data\ido"
output_dir = r"C:\Users\user\Documents\lab\data\ido\converted"
os.makedirs(output_dir, exist_ok=True)

# === 緯度・経度 → 直交座標変換用Transformer（UTM Zone 54N） ===
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32654", always_xy=True)

# === .xyzファイルの一括処理 ===
xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))

for file_path in xyz_files:
    try:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        # === ファイル読み込み ===
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["lat", "lon", "alt"])

        # === 座標変換 ===
        x, y = transformer.transform(df["lon"].values, df["lat"].values)
        df_converted = pd.DataFrame({
            "X": x,
            "Y": y,
            "Z": df["alt"]
        })

        # === ファイル保存 ===
        df_converted.to_csv(output_path, sep=" ", header=False, index=False)
        print(f"✅ 変換完了: {filename}")

    except Exception as e:
        print(f"❌ エラー（{filename}）: {e}")
