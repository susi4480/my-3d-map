# -*- coding: utf-8 -*-
"""
【機能】
Ouster OS-2 の .pcap + .json を読み込み、
各フレームの距離画像(Range)・反射強度画像(Reflectivity)をPNGとして出力。

出力先:
    C:/Users/user/Documents/lab/flame
"""

import os
import numpy as np
import cv2
from ouster.sdk.open_source import open_source
from ouster.sdk.core import SensorInfo, destagger, ChanField


def main():
    # === 入出力設定 ===
    INPUT_PCAP = "C:/Users/user/Downloads/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
    META_JSON  = "C:/Users/user/Downloads/realdata/2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
    OUTPUT_DIR = "C:/Users/user/Documents/lab/flame1"

    # --- 出力フォルダ作成 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- メタデータ読み込み ---
    print("📄 メタデータ読込中...")
    with open(META_JSON, "r", encoding="utf-8") as f:
        meta_json = f.read()
    info = SensorInfo(meta_json)

    # --- PCAPソース読み込み ---
    print("📥 PCAP読込中...")
    source = open_source(INPUT_PCAP, meta=[META_JSON])

    print("🌀 フレーム処理開始...")
    counter = 0

    for scans in source:
        for scan in scans:
            try:
                # === 距離画像 ===
                rng = destagger(info, scan.field(ChanField.RANGE))
                rng_norm = np.uint8(np.clip(rng / np.max(rng) * 255, 0, 255))

                # === 反射強度画像 ===
                refl = destagger(info, scan.field(ChanField.REFLECTIVITY))
                refl_norm = np.uint8(np.clip(refl / np.max(refl) * 255, 0, 255))

                # === 保存 ===
                rng_path = os.path.join(OUTPUT_DIR, f"range_{counter:04d}.png")
                refl_path = os.path.join(OUTPUT_DIR, f"reflectivity_{counter:04d}.png")

                cv2.imwrite(rng_path, rng_norm)
                cv2.imwrite(refl_path, refl_norm)

                print(f"✅ Frame {counter:04d} saved")
                counter += 1

            except Exception as e:
                print(f"⚠️ Frame {counter} skipped due to error: {e}")
                counter += 1
                continue

    cv2.destroyAllWindows()
    print("🎉 完了しました！ 出力先:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
