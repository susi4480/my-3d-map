import pandas as pd

# === ファイルパス ===
file_path = r"C:\Users\user\Documents\lab\data\ido\20211029_Marlin[multibeam]_20240625_TUMSAT LiDAR triai-20240627-121535(1)-R20250519-164056.xyz"

try:
    # === ファイル読み込み（空白またはタブ区切りに対応）===
    df = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # === 列数に応じてカラム名をつける（一般的なXYZ形式：X Y Z）===
    col_count = df.shape[1]
    if col_count >= 3:
        df.columns = ['X', 'Y', 'Z'] + [f'col{i}' for i in range(4, col_count+1)]
    else:
        df.columns = [f'col{i}' for i in range(1, col_count+1)]

    # === 最初の10行を表示して確認 ===
    print("=== ファイルの先頭10行 ===")
    print(df.head(10))

    # === 数値範囲も確認 ===
    if 'X' in df.columns and 'Y' in df.columns:
        print(f"\n📍 Xの範囲: {df['X'].min()} ～ {df['X'].max()}")
        print(f"📍 Yの範囲: {df['Y'].min()} ～ {df['Y'].max()}")
    if 'Z' in df.columns:
        print(f"📍 Zの範囲: {df['Z'].min()} ～ {df['Z'].max()}")

except Exception as e:
    print(f"❌ 読み込みエラー: {e}")
