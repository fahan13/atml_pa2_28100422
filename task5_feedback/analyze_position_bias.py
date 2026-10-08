"""Task 5 supplementary: position consistency of the pairwise judge on the diagnostic pairs (CPU only).

score_perturbations queries each (clean, perturbed) pair twice: compare(q, clean, pert) and
compare(q, pert, clean). The released PairwiseAIJudge additionally swaps A/B internally using a
hash of (model, problem, a, b), so in roughly half of the pairs the two calls end up showing the
judge the SAME physical order. Those calls are deterministic duplicates and are trivially
consistent. This script recomputes the released swap bit for both calls (no model needed) and
reports consistency separately for pairs whose physical presentation order really flipped,
which is the quantity that measures position bias.
Output: results/task5_feedback/position_consistency.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json

from common.data import load_yaml, read_jsonl, repo_path


def released_swap(model, problem, a, b):
    """Identical to PairwiseAIJudge._key + its swap rule in task5_feedback/rlaif.py."""
    payload = json.dumps({"model": model, "problem": problem, "a": a, "b": b}, sort_keys=True)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % 2 == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    model = cfg["ai_judge_model"]
    by = collections.defaultdict(dict)
    for r in read_jsonl(cfg["paths"]["task5_diagnostics"]):
        by[str(r["problem_id"])][r["variant_type"]] = r
    pairs = read_jsonl(repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostic_pairs.jsonl")
    out = {}
    for pert in sorted({p["perturbation"] for p in pairs}):
        rows = {"flipped": [], "same_order": []}
        for p in (x for x in pairs if x["perturbation"] == pert):
            v = by[str(p["problem_id"])]
            q, clean, worse = str(v["clean_correct"]["question"]), v["clean_correct"]["response"], v[pert]["response"]
            flipped = released_swap(model, q, clean, worse) == released_swap(model, q, worse, clean)
            rows["flipped" if flipped else "same_order"].append(p["judge_order_consistent"])
        out[pert] = {k: {"n": len(x), "consistency": (sum(x) / len(x)) if x else None} for k, x in rows.items()}
        print(pert, out[pert])
    dest = repo_path(cfg["results_dir"]) / "task5_feedback" / "position_consistency.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", dest)


if __name__ == "__main__":
    main()
