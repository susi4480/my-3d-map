# -*- coding: utf-8 -*-
"""
Ouster OS-2 の .pcap と .json メタデータを読み込み、
各フレームの Range（距離）と Reflectivity（反射強度）画像を横並びにして
MP4 動画として出力する。
"""

import os
import numpy as np
import cv2
from ouster.sdk.open_source import open_source
from ouster.sdk.core import Scans, SensorInfo, destagger, ChanField

def main():
    INPUT_PCAP = r"C:\Users\user\Downloads\realdata\2022-07-06-18-33-15_OS-2-128-992048000507-1024x10-001.pcap"
    META_JSON  = r"C:\Users\user\Downloads\realdata\2022-07-06-18-33-15_OS-2-128-992048000507-1024x10.json"
    OUTPUT_DIR = r"C:\Users\user\Documents\lab\flame_video"
    VIDEO_PATH = os.path.join(OUTPUT_DIR, "ouster_output.mp4")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(META_JSON, 'r') as f:
        meta_json = f.read()
    info = SensorInfo(meta_json)

    source = open_source(INPUT_PCAP, meta=[META_JSON])
    scans = Scans(source)

    print("🎬 Processing frames and building video...")

    video_writer = None
    counter = 0

    for scans_batch in source:
        for scan in scans_batch:
            rng = destagger(info, scan.field(ChanField.RANGE))
            rng_norm = np.uint8(np.clip(rng / np.max(rng) * 255, 0, 255))
            rng_color = cv2.applyColorMap(rng_norm, cv2.COLORMAP_JET)

            refl = destagger(info, scan.field(ChanField.REFLECTIVITY))
            refl_norm = np.uint8(np.clip(refl / np.max(refl) * 255, 0, 255))
            refl_color = cv2.cvtColor(refl_norm, cv2.COLOR_GRAY2BGR)

            combined = np.hstack((rng_color, refl_color))

            scale_y = 1.0  # ← 見やすくするなら 2.0 などに変更可
            if scale_y != 1.0:
                new_h = int(combined.shape[0] * scale_y)
                new_w = combined.shape[1]
                combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if video_writer is None:
                h, w, _ = combined.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = 10
                video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (w, h))

            video_writer.write(combined)
            cv2.imshow("Ouster Range + Reflectivity", combined)
            if cv2.waitKey(1) == 27:
                print("⏹ ESC pressed. Stopping early...")
                video_writer.release()
                cv2.destroyAllWindows()
                return
            counter += 1

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    print(f"✅ Done! {counter} frames processed.")
    print(f"📹 Saved video: {VIDEO_PATH}")

if __name__ == "__main__":
    main()
