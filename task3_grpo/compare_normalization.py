"""Task 3 Step 3: canonical GRPO (1/T_k) vs Dr.-GRPO-style (1/L_max) sequence normalization.

Both forks start from the identical supplied midpoint, consume the same seeded prompt prefix and
seed, and keep reward, beta, epsilon, K and the generation cap fixed; only loss_type changes.
Length-conditioned statistic: the gradient mass each completion carries under its normalization
(sequence_weight = sum of per-token weights = |A| for grpo, |A|*T/L_max for dr_grpo), split by
completion length (below / above the median of all completions in both forks).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr


def run(args):
    cmd = [sys.executable, "-m", *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    rdir = repo_path(cfg["results_dir"])
    forks = {"fork_grpo": "grpo", "fork_dr_grpo": "dr_grpo"}
    for name, lt in forks.items():
        out = f"outputs/task3_grpo/{name}"
        if not (repo_path(out) / "adapter_model.safetensors").exists():
            run(["task3_grpo.continue_train", "--config", args.config, "--run-name", name, "--output", out,
                 "--updates", str(cfg["fork_updates"]), "--loss-type", lt])
        if not (rdir / f"eval_{name}.json").exists():
            run(["task3_grpo.evaluate", "--config", args.config, "--adapter", out, "--name", name])

    comps = {n: read_jsonl(rdir / f"{n}_completions.jsonl") for n in forks}
    all_len = [c["length"] for cs in comps.values() for c in cs if c["in_loss"]]
    med = float(np.median(all_len))
    rows = []
    for name, lt in forks.items():
        cs = [c for c in comps[name] if c["in_loss"]]
        short = [c for c in cs if c["length"] <= med]
        long_ = [c for c in cs if c["length"] > med]
        e = load_json(rdir / f"eval_{name}.json")
        log = read_jsonl(rdir / f"{name}_train_log.jsonl")
        s = load_json(rdir / f"{name}_train_summary.json")
        tot = sum(c["sequence_weight"] for c in cs) or 1.0
        rows.append({
            "fork": name, "loss_type": lt, "updates": len(log), "tokens_generated": s["tokens_generated"],
            "heldout_reward": e["reward_mean"], "heldout_kl_token": e["kl_token_mean"],
            "heldout_len_mean": e["length"]["mean"], "heldout_len_std": e["length"]["std"],
            "heldout_truncated": e["truncated_rate"], "heldout_entropy": e["entropy_token_mean"],
            "train_len_mean": float(np.mean([c["length"] for c in comps[name]])),
            "train_reward_mean": float(np.mean([r["reward"] for r in log])),
            "grad_norm_mean": float(np.mean([r["grad_norm"] for r in log])),
            "median_len_split": med,
            "short_token_weight_mean": float(np.mean([c["token_weight"] for c in short])) if short else float("nan"),
            "long_token_weight_mean": float(np.mean([c["token_weight"] for c in long_])) if long_ else float("nan"),
            "short_share_of_gradient_mass": sum(c["sequence_weight"] for c in short) / tot,
            "long_share_of_gradient_mass": sum(c["sequence_weight"] for c in long_) / tot,
            "corr_length_reward": safe_corr([c["length"] for c in cs], [c["reward"] for c in cs]),
            "corr_length_advantage": safe_corr([c["length"] for c in cs], [c["advantage"] for c in cs]),
            "signed_weight_long_minus_short": float(sum(np.sign(c["advantage"]) * c["sequence_weight"] for c in long_)
                                                    - sum(np.sign(c["advantage"]) * c["sequence_weight"] for c in short)),
        })
    with (rdir / "normalization_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    save_json(rdir / "normalization_summary.json", {"rows": rows})
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()