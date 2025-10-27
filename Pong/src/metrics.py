# src/metrics.py
import os, csv, json, time
from typing import Dict, Any

class MetricsLogger:
    """
    Logs per-episode metrics to CSV and summary JSON.
    """
    def __init__(self, outdir='logs', run_name='run'):
        self.outdir = outdir
        self.run_name = run_name
        os.makedirs(self.outdir, exist_ok=True)
        self.csv_path = os.path.join(self.outdir, f"{run_name}_episodes.csv")
        self.summary_path = os.path.join(self.outdir, f"{run_name}_summary.json")
        self.rows = []
        # create CSV with header we will update dynamically
        self._write_header(['episode','episode_reward','episode_length','app','algo','persona','seed','timestamp'])

    def _write_header(self, header):
        self.header = header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def add(self, episode:int, episode_reward:float, episode_length:int, app:str, algo:str, persona:str, seed:int, extras:Dict[str,Any]|None=None):
        if extras is None:
            extras = {}
        row = {
            'episode': episode,
            'episode_reward': float(episode_reward),
            'episode_length': int(episode_length),
            'app': app,
            'algo': algo,
            'persona': persona,
            'seed': int(seed),
            'timestamp': int(time.time())
        }
        row.update(extras)
        self.rows.append(row)
        self._append_row_csv(row)

    def _append_row_csv(self, row:Dict[str,Any]):
        # Ensure header contains all keys
        for k in row.keys():
            if k not in self.header:
                self.header.append(k)
        # re-write header if expanded
        with open(self.csv_path, 'r', newline='') as f:
            old = f.read()
        # rewrite entire file with new header and existing rows (naive but simple)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            # write previous rows from memory
            for r in self.rows:
                writer.writerow([r.get(k,'') for k in self.header])

    def flush(self):
        # write summary JSON
        summary = {
            'run_name': self.run_name,
            'episodes': len(self.rows),
            'sample_rows': self.rows[:5]
        }
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return self.csv_path, self.summary_path
