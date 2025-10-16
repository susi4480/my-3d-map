import ouster.sdk as ouster
from ouster.sdk import pcap
from ouster.sdk import client

PCAP_PATH = r"C:\Users\user\Downloads\realdata\2022-07-06-17-32-45_OS-2-128-992048000507-1024x10-002.pcap"
JSON_PATH = r"C:\Users\user\Downloads\realdata\2022-07-06-17-32-45_OS-2-128-992048000507-1024x10.json"
# センサー情報の読み込み
with open(JSON_PATH, 'r') as f:
    metadata = client.SensorInfo(f.read())

# パケットソースを開く
source = pcap.Pcap(metadata, PCAP_PATH)

# 1フレーム分のスキャンを取得
for scan in ouster.client.Scans(source):
    xyz = client.XYZLut(metadata)(scan)  # ← 点群変換
    points = xyz.reshape(-1, 3)
    print(points.shape)  # (N,3)
    break  # 最初の1スキャンだけ
