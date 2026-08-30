"""JSON checkpoint helper for long, resumable experiment runs.

Each per-seed result is written to disk as it completes; on restart the driver
loads what is already there and skips it, so an interrupted sweep resumes.
"""
import json
import os


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)
