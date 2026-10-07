"""Task 1 Step 3: length-confounding study.

1. Train the length-balanced DPO model (same full-epoch budget and settings as standard DPO).
2. Evaluate it under the same protocol as standard (held-out, stratified, generation, word-limit).
3. Report dataset-level length structure of both training files (property of the DATA),
   next to generated length (property of the POLICY).
Requires the standard DPO model and its eval (eval_standard.json) to exist first.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import load_tokenizer
from task1_dpo._runner import eval_if_missing, train_if_missing


def data_length_profile(path, tokenizer):
    diffs = []
    for r in read_jsonl(path):
        yc, yr = preference_responses(r)
        diffs.append(len(tokenizer(yc, add_special_tokens=False)["input_ids"])
                     - len(tokenizer(yr, add_special_tokens=False)["input_ids"]))
    d = np.asarray(diffs)
    return {"n": int(d.size), "frac_chosen_longer": float((d > 0).mean()),
            "mean_len_diff_chosen_minus_rejected": float(d.mean()), "median_len_diff": float(np.median(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"])

    train_if_missing(args.config, "length_balanced", cfg["length_output"],
                     ["--dataset", cfg["paths"]["dpo_length_train"]])
    eval_if_missing(args.config, "length_balanced", cfg["length_output"])

    tok = load_tokenizer(cfg["base_model"])
    profile = {"standard_train": data_length_profile(cfg["paths"]["dpo_standard_train"], tok),
               "length_balanced_train": data_length_profile(cfg["paths"]["dpo_length_train"], tok)}

    rows = []
    for name in ["sft", "standard", "length_balanced"]:
        e = load_json(rdir / f"eval_{name}.json")
        row = {"model": name,
               "gen_len_mean": e["generation"]["length"]["mean"],
               "gen_len_std": e["generation"]["length"]["std"],
               "gen_len_iqr": e["generation"]["length"]["iqr"],
               "reward_mean": e["generation"]["reward_mean"],
               "wordlimit_compliance": e["word_limit"]["compliance"],
               "wordlimit_words_mean": e["word_limit"]["words"]["mean"]}
        if "stratified" in e:
            for st, v in e["stratified"]["by_stratum"].items():
                row[f"acc_{st}"] = v["preference_accuracy"]
                row[f"margin_{st}"] = v["mean_margin"]
        rows.append(row)

    save_json(rdir / "length_study_summary.json", {"data_profile": profile, "models": rows})
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
    with (rdir / "length_study_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(profile)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()