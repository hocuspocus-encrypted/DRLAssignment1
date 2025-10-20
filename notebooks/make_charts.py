
from pathlib import Path
import json
import csv
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "logs" / "eval"
OUT_DIR = ROOT / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_runs() -> Dict[str, List[Dict[str, Any]]]:
    runs = {}
    for p in sorted(EVAL_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                runs[p.stem] = data
        except Exception as e:
            print(f"[WARN] Could not read {p.name}: {e}")
    return runs

def to_array(seq, key, default_val=np.nan):
    arr = []
    for d in seq:
        v = d.get(key, default_val)
        try:
            v = float(v)
        except Exception:
            v = default_val
        arr.append(v)
    return np.array(arr, dtype=float)

def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")

def plot_series(x_list, y_list, labels, title, ylabel, outfile):
    plt.figure()
    for x, y, label in zip(y_list, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_hist(data, title, xlabel, outfile):
    plt.figure()
    plt.hist(data, bins=20)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def main():
    runs = load_runs()
    if not runs:
        print("[INFO] No JSON files found in logs/eval. Run src.eval first.")
        return

    # Summary CSV
    summary_path = ROOT / "logs" / "eval_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run","episodes","reward_mean","reward_std","length_mean","length_std","lines_mean","lines_std"])
        for name, data in runs.items():
            ep = to_array(data, "episode")
            rew = to_array(data, "reward")
            leng = to_array(data, "length")
            lines = to_array(data, "lines")
            w.writerow([
                name,
                int(np.nanmax(ep)+1) if ep.size else 0,
                safe_mean(rew), float(np.nanstd(rew)),
                safe_mean(leng), float(np.nanstd(leng)),
                safe_mean(lines), float(np.nanstd(lines)),
            ])
    print(f"[OK] Wrote summary CSV -> {summary_path}")

    # Combined plots: Reward vs Episode
    xs = []
    ysets = []
    labels = []
    for name, data in runs.items():
        ep = to_array(data, "episode")
        rew = to_array(data, "reward")
        if ep.size and rew.size:
            xs = ep if ep.size > len(xs) else xs
            ysets.append(rew)
            labels.append(name)
    if ysets:
        plot_series(xs, ysets, labels, "Reward vs Episode (All Runs)", "Reward", OUT_DIR / "reward_all_runs.png")
        print("[OK] reward_all_runs.png")

    # Combined plots: Episode Length vs Episode
    xs = []
    ysets = []
    labels = []
    for name, data in runs.items():
        ep = to_array(data, "episode")
        leng = to_array(data, "length")
        if ep.size and leng.size:
            xs = ep if ep.size > len(xs) else xs
            ysets.append(leng)
            labels.append(name)
    if ysets:
        plot_series(xs, ysets, labels, "Episode Length vs Episode (All Runs)", "Episode Length", OUT_DIR / "length_all_runs.png")
        print("[OK] length_all_runs.png")

    # Per-run histograms for 'lines' if present
    for name, data in runs.items():
        lines = to_array(data, "lines")
        if np.isfinite(lines).any():
            plot_hist(lines[np.isfinite(lines)], f"Lines Cleared Distribution — {name}", "Lines per Episode", OUT_DIR / f"{name}_lines_hist.png")
            print(f"[OK] {name}_lines_hist.png")

    # Markdown snippet for README
    md_path = OUT_DIR / "README_charts_snippet.md"
    parts = ["## 📊 Results (Auto‑Generated Charts)\n"]
    if (OUT_DIR / "reward_all_runs.png").exists():
        parts.append("![Reward vs Episode](docs/plots/reward_all_runs.png)\n")
    if (OUT_DIR / "length_all_runs.png").exists():
        parts.append("![Episode Length vs Episode](docs/plots/length_all_runs.png)\n")
    for name in runs.keys():
        p = OUT_DIR / f"{name}_lines_hist.png"
        if p.exists():
            parts.append(f"![Lines Distribution — {name}](docs/plots/{name}_lines_hist.png)\n")
    md_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] Wrote README snippet -> {md_path}")

if __name__ == "__main__":
    main()
