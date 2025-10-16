# lidar/loader.py
from ouster.sdk.client import Scans, LidarScan, XYZLut, Destination, SensorInfo
from ouster.sdk.pcap import Pcap


def load_ouster_scans(pcap_path, meta_path, max_frames=1):
    with open(meta_path, 'r') as f:
        metadata = SensorInfo(f.read())
    scans = []
    with Pcap(pcap_path, metadata) as source:
        for idx, scan in enumerate(Scans(source)):
            if idx >= max_frames:
                break
            scans.append(scan)
    return scans
