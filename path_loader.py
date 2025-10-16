# map/path_loader.py
import json

def load_path_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
