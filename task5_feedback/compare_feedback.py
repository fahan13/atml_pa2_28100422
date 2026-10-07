"""Task 5: final RLVR-vs-RLAIF table from the saved in-domain, transfer and diagnostic results."""
from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm = load_json(rdir / "math_gsm_summary.json")
    tr = load_json(rdir / "math_transfer_summary.json")
    diag = load_json(rdir / "diagnostic_summary.json")

    rows = []
    for name in ("sft", "rlvr", "rlaif"):
        g, t = gsm["policies"][name], tr["policies"][name]
        row = {"policy": name,
               "gsm_acc": g["exact_accuracy"], "svamp_acc": t["exact_accuracy"],
               "acc_drop": g["exact_accuracy"] - t["exact_accuracy"],
               "gsm_format": g["format_compliance"], "svamp_format": t["format_compliance"],
               "gsm_len": g["length_mean"], "svamp_len": t["length_mean"],
               "gsm_truncated": g["truncated_rate"], "svamp_truncated": t["truncated_rate"]}
        if name != "sft":
            pg, pt = gsm["pairwise_vs_sft"][name], tr["pairwise_vs_sft"][name]
            row.update({"gsm_winrate_vs_sft": pg["win_rate_vs_sft"], "gsm_judge_ties": pg["ties"],
                        "svamp_winrate_vs_sft": pt["win_rate_vs_sft"], "svamp_judge_ties": pt["ties"],
                        "winrate_drop": pg["win_rate_vs_sft"] - pt["win_rate_vs_sft"],
                        "gsm_verifier_judge_agreement": pg["verifier_judge_agreement_on_strict_pairs"]})
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (rdir / "feedback_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    save_json(rdir / "feedback_comparison.json", {
        "policies": rows, "S_reason": diag["S_reason"], "S_outcome": diag["S_outcome"],
        "by_perturbation": diag["by_perturbation"], "devices": [gsm.get("device"), tr.get("device")]})
    for r in rows:
        print(r)
    print("S_reason", diag["S_reason"], "S_outcome", diag["S_outcome"])


if __name__ == "__main__":
    main()