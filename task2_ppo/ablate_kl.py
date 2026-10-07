"""Task 2 Step 3: KL-pressure study. Matched short forks from the identical supplied midpoint,
beta_KL in kl_values, clip epsilon at the config default; only beta_KL changes."""
from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, repo_path
from task2_ppo._runner import fork_row, run_fork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    eps = float(cfg["clip_epsilon"])
    rows = []
    for b in cfg["kl_values"]:
        name = run_fork(args.config, cfg, eps, b)
        rows.append(fork_row(cfg, name, eps, float(b)))
    out = repo_path(cfg["results_dir"]) / "kl_study_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("saved", out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()