from __future__ import annotations
import csv,json,os, sys, tempfile
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__)))
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


def _to_py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x

@dataclass
class EpisodeMetrics:
    episode: int
    reward: float
    length: int
    extras: Dict[str, Any]

class MetricsLogger:
    def __init__(self, out_dir: str, run_name: str):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, f"{run_name}.csv")
        self.json_path = os.path.join(out_dir, f"{run_name}.json")
        self.buffer: List[EpisodeMetrics] = []

    def add(self, ep: int, reward: float, length: int, **extras):
        self.buffer.append(EpisodeMetrics(ep, reward, length, extras))

    def flush(self):
        if not self.buffer: return
# CSV
        keys = ["episode","reward","length"]
        extra_keys = sorted({k for m in self.buffer for k in m.extras.keys()})
        wrote_header = False
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not os.path.exists(self.csv_path) or os.stat(self.csv_path).st_size == 0:
                w.writerow(keys + extra_keys)
                wrote_header = True
            for m in self.buffer:
                row = [m.episode, m.reward, m.length] + [m.extras.get(k, "") for k in extra_keys]
                w.writerow(row)
# JSON (append list)
        data = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
            except json.JSONDecodeError:
                # keep a backup of the corrupted file and start fresh
                os.replace(self.json_path, self.json_path + ".bak")

        for m in self.buffer:
            row = {"episode": int(m.episode), "reward": float(m.reward), "length": int(m.length)}
            for k, v in m.extras.items():
                row[k] = _to_py(v)
            data.append(row)

        # atomic write to avoid partial JSON
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="metrics_", suffix=".json", dir=os.path.dirname(self.json_path))
        os.close(tmp_fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.json_path)

        self.buffer.clear()

