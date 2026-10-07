"""Task 3 Step 2: group-size study on the supplied K=8 cache (no training, CPU only).

Budget: the cache holds 24 prompts x 8 completions = 192 generations. Every K uses all 192:
  K=8 -> 24 groups (one per prompt), K=4 -> 48 groups (each prompt's 8 split into 2 disjoint
  groups of 4), K=2 -> 96 groups (4 disjoint pairs per prompt). Canonical partition follows
  generation_index order; we also average over random partitions (R resamples) to remove the
  dependence on one arbitrary split.

Metrics per K (overall and per prompt-difficulty bin):
  informative_rate      : groups with reward std > 1e-6 (tolerance of the released advantage helper)
  mean_within_group_std : mean reward std inside a group
  centered_signal_var   : variance of (r - group mean), the raw relative signal before scaling
  advantage_resample_var: for a fixed completion, variance of its group-relative advantage across
                          random group assignments (0 for K=8: only one possible group) - how noisy
                          the signal a completion receives is, purely from who it is grouped with
  sign_agreement_with_K8: share of completions whose advantage sign matches their K=8 sign
Difficulty bins (defined once): prompts split into tertiles of their mean reward over all 8
completions -> low / mid / high reward (hard / medium / easy for the policy).
Binarized variant: reward -> 1[reward > median of all 192 rewards], a low-resolution reward like a
verifier, where very easy/hard prompts can produce all-equal (uninformative) groups.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

STD_TOL = 1e-6
RESAMPLES = 200


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Partition rule: within each prompt, the first 8 completions (generation_index order) are
    split into 8/K consecutive disjoint groups of size K, so every K uses all 8*n_prompts
    generations exactly once and no group mixes prompts.
    """
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:8]
        for s in range(0, 8, k):
            groups.append({"prompt": pid, "rows": rows[s:s + k]})
    return groups


def advantages(r):
    r = np.asarray(r, dtype=float)
    return (r - r.mean()) / max(r.std(), STD_TOL)


def group_metrics(groups, reward_key):
    stds, centered = [], []
    for g in groups:
        r = np.asarray([x[reward_key] for x in g["rows"]], dtype=float)
        stds.append(r.std())
        centered += list(r - r.mean())
    stds = np.asarray(stds)
    return {"n_groups": len(groups), "informative_rate": float((stds > STD_TOL).mean()),
            "uninformative_fraction": float((stds <= STD_TOL).mean()),
            "mean_within_group_std": float(stds.mean()), "centered_signal_var": float(np.var(centered))}


def resample_stats(by_prompt, k, reward_key, rng):
    """Advantage variance per completion across random K-partitions, and sign agreement with K=8."""
    per_completion = defaultdict(list)
    agree = []
    for pid, rows in by_prompt.items():
        r = np.asarray([x[reward_key] for x in rows[:8]], dtype=float)
        a8 = advantages(r)
        for _ in range(RESAMPLES if k < 8 else 1):
            perm = rng.permutation(8)
            for s in range(0, 8, k):
                idx = perm[s:s + k]
                a = advantages(r[idx])
                for i, ai in zip(idx, a):
                    per_completion[(pid, int(i))].append(ai)
                    if a8[i] != 0 and ai != 0:
                        agree.append(np.sign(ai) == np.sign(a8[i]))
    var = np.mean([np.var(v) for v in per_completion.values()])
    return {"advantage_resample_var": float(var),
            "sign_agreement_with_K8": float(np.mean(agree)) if agree else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    rng = np.random.default_rng(int(cfg["seed"]))

    all_r = np.asarray([x["reward"] for rows in by_prompt.values() for x in rows[:8]])
    med = float(np.median(all_r))
    for rows in by_prompt.values():
        for x in rows:
            x["reward_binary"] = float(x["reward"] > med)

    prompt_mean = {pid: float(np.mean([x["reward"] for x in rows[:8]])) for pid, rows in by_prompt.items()}
    q1, q2 = np.quantile(list(prompt_mean.values()), [1 / 3, 2 / 3])
    bin_of = {pid: ("low_reward" if m <= q1 else "high_reward" if m > q2 else "mid_reward")
              for pid, m in prompt_mean.items()}

    out_rows, detail = [], {"n_prompts": len(by_prompt), "total_generations": int(all_r.size),
                            "binarization_threshold": med, "difficulty_tertile_cuts": [float(q1), float(q2)],
                            "prompt_bins": bin_of, "results": {}}
    for reward_key in ("reward", "reward_binary"):
        for k in cfg["group_sizes"]:
            k = int(k)
            for bin_name in ("all", "low_reward", "mid_reward", "high_reward"):
                sub = {p: r for p, r in by_prompt.items() if bin_name == "all" or bin_of[p] == bin_name}
                groups = regroup_equal_generation_budget(sub, k)
                row = {"reward_type": "continuous_RM" if reward_key == "reward" else "binarized",
                       "K": k, "difficulty_bin": bin_name, "n_prompts": len(sub),
                       **group_metrics(groups, reward_key), **resample_stats(sub, k, reward_key, rng)}
                out_rows.append(row)
    rdir = repo_path(cfg["results_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    with (rdir / "group_size_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    detail["results"] = out_rows
    save_json(rdir / "group_size_summary.json", detail)
    for r in out_rows:
        if r["difficulty_bin"] == "all":
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("saved", rdir / "group_size_summary.csv")


if __name__ == "__main__":
    main()