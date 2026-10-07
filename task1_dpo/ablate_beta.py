"""Task 1 Step 2: short-run DPO forks for beta in {0.03, 0.10, 0.30}.

Every fork starts from the original Qwen2.5-1.5B-Instruct (fresh LoRA), uses the same first
`short_ablation_examples` filtered pairs in the same seeded order, same optimizer/LoRA/seed.
Only beta changes. Re-running skips finished forks.
"""
from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from task1_dpo._runner import eval_if_missing, train_if_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    n = str(cfg["short_ablation_examples"])
    rows = []
    for b in cfg["betas"]:
        tag = f"beta_{float(b):.2f}"
        out = f"outputs/task1_dpo/{tag}"
        train_if_missing(args.config, tag, out, ["--beta", str(b), "--max-examples", n])
        eval_if_missing(args.config, tag, out, ["--beta", str(b), "--parts", "pairs,gen,wl"])
        e = load_json(repo_path(cfg["results_dir"]) / f"eval_{tag}.json")
        rows.append({"condition": tag, "beta": b, "train_pairs": n,
                     "heldout_dpo_loss": e["heldout"]["dpo_loss"],
                     "heldout_pref_acc": e["heldout"]["preference_accuracy"],
                     "kl_token": e["generation"]["kl_token_mean"],
                     "kl_seq": e["generation"]["kl_sequence_mean"],
                     "reward_mean": e["generation"]["reward_mean"],
                     "len_mean": e["generation"]["length"]["mean"],
                     "len_std": e["generation"]["length"]["std"],
                     "wordlimit_compliance": e["word_limit"]["compliance"]})
    out = repo_path(cfg["results_dir"]) / "beta_sweep_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("saved", out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()