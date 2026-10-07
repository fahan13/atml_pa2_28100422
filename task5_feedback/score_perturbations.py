"""Task 5 Step 2: controlled reward diagnostics.

Every perturbation is compared against the clean response of the SAME problem, with the clean
response as the diagnostically better one. For each feedback mechanism we report the fraction of
pairs where it prefers the better response, ties, or prefers the worse one.
  RLVR : released exact verifier, R = exact_reward(response, gold)
  RLAIF: released pairwise AI judge. Primary call = compare(clean, perturbed); we also query the
         reversed order to measure position consistency of the judge.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
# perturbation -> what it isolates
PAIR_KIND = {
    "corrupt_reasoning_correct_final": "reasoning",   # final held correct, reasoning degraded
    "good_reasoning_wrong_final": "outcome",          # reasoning ~fixed, designated final changed
    "gold_distractor_wrong_final": "outcome",         # gold number present only as a distractor
    "persuasive_filler_correct": "filler",            # correct, plus irrelevant persuasive filler
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def outcome_label(sign):
    return {1: "better", 0: "tie", -1: "wrong"}[sign]


def rates(labels):
    n = len(labels)
    return {"n": n, "better_rate": labels.count("better") / n, "tie_rate": labels.count("tie") / n,
            "wrong_rate": labels.count("wrong") / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    rdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    rdir.mkdir(parents=True, exist_ok=True)
    judge = PairwiseAIJudge(cfg, rdir / "judge_cache_diagnostics.json")

    # absolute verifier reward for every response (all five categories)
    verifier_by_variant = defaultdict(list)
    records = []
    for pid, v in groups.items():
        gold = str(v["clean_correct"]["gold_final"])
        R = {k: exact_reward(v[k]["response"], gold) for k in EXPECTED_VARIANTS}
        for k in EXPECTED_VARIANTS:
            verifier_by_variant[k].append(R[k])
        q = str(v["clean_correct"]["question"])
        for pert, kind in PAIR_KIND.items():
            clean, worse = v["clean_correct"]["response"], v[pert]["response"]
            ver = outcome_label(int(np.sign(R["clean_correct"] - R[pert])))
            j1 = judge.compare(q, clean, worse)                  # A = clean
            j2 = judge.compare(q, worse, clean)                  # A = perturbed (reversed order)
            jud = {"A": "better", "B": "wrong", "TIE": "tie"}[j1]
            jud_rev = {"A": "wrong", "B": "better", "TIE": "tie"}[j2]
            records.append({"problem_id": pid, "perturbation": pert, "kind": kind,
                            "verifier_clean": R["clean_correct"], "verifier_perturbed": R[pert],
                            "verifier": ver, "judge": jud, "judge_reversed": jud_rev,
                            "judge_order_consistent": jud == jud_rev})
        print(f"problem {pid} done", flush=True)

    write_jsonl(rdir / "diagnostic_pairs.jsonl", records)
    res = {"verifier_pass_rate_by_variant": {k: float(np.mean(x)) for k, x in sorted(verifier_by_variant.items())},
           "by_perturbation": {}, "sensitivity": {}}
    for pert in PAIR_KIND:
        rs = [r for r in records if r["perturbation"] == pert]
        res["by_perturbation"][pert] = {
            "kind": PAIR_KIND[pert],
            "verifier": rates([r["verifier"] for r in rs]),
            "judge": rates([r["judge"] for r in rs]),
            "judge_reversed_order": rates([r["judge_reversed"] for r in rs]),
            "judge_order_consistency": float(np.mean([r["judge_order_consistent"] for r in rs])),
        }
    for kind in ("reasoning", "outcome", "filler"):
        rs = [r for r in records if r["kind"] == kind]
        for mech in ("verifier", "judge"):
            res["sensitivity"][f"S_{kind}_{mech}"] = rates([r[mech] for r in rs])
    res["S_reason"] = {m: res["sensitivity"][f"S_reasoning_{m}"]["better_rate"] for m in ("verifier", "judge")}
    res["S_outcome"] = {m: res["sensitivity"][f"S_outcome_{m}"]["better_rate"] for m in ("verifier", "judge")}
    save_json(rdir / "diagnostic_summary.json", res)
    print("S_reason:", res["S_reason"])
    print("S_outcome:", res["S_outcome"])
    for pert, v in res["by_perturbation"].items():
        print(pert, "verifier", v["verifier"], "judge", v["judge"], "consistency", v["judge_order_consistency"])


if __name__ == "__main__":
    main()