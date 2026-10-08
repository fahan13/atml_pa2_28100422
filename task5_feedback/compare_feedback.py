"""Task 5: final RLVR-vs-RLAIF table from the saved in-domain, transfer and diagnostic results.

Primary metrics use the released verifier (designated `#### <number>` only). Two clearly-labelled
supplementary diagnostics are computed from the saved generations (CPU only):
  lenient_accuracy : the response's final stated answer (last \\boxed{...} or, failing that, the last
                     number in the text) equals gold - separates format failure from math failure;
  judge results restricted to non-identical pairs: identical greedy responses are a trivial TIE.
"""
from __future__ import annotations

import argparse
import csv
import re

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.rlvr import extract_designated_final, numerically_equal

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def lenient_final(text):
    d = extract_designated_final(text)
    if d is not None:
        return d
    boxed = _BOXED.findall(text)
    src = boxed[-1] if boxed else text
    nums = _NUM.findall(src)
    return nums[-1].replace(",", "") if nums else None


def supplementary(rdir, dataset):
    gens = {p: read_jsonl(rdir / f"generations_{dataset}_{p}.jsonl") for p in ("sft", "rlvr", "rlaif")}
    comps = read_jsonl(rdir / f"judge_comparisons_{dataset}.jsonl")
    out = {}
    for p, rs in gens.items():
        out[p] = {"lenient_accuracy": sum(numerically_equal(lenient_final(r["response"]), r["gold"]) for r in rs) / len(rs)}
    for p in ("rlvr", "rlaif"):
        same = [a["response"] == b["response"] for a, b in zip(gens[p], gens["sft"])]
        cs = [c for c, s in zip([c for c in comps if c["policy"] == p], same) if not s]
        out[p].update({
            "identical_to_sft": sum(same), "non_identical_pairs": len(cs),
            "winrate_non_identical": (sum(c["score"] for c in cs) / len(cs)) if cs else None,
            "judge_ties_non_identical": sum(c["judge"] == "TIE" for c in cs),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm = load_json(rdir / "math_gsm_summary.json")
    tr = load_json(rdir / "math_transfer_summary.json")
    diag = load_json(rdir / "diagnostic_summary.json")
    sup = {"gsm": supplementary(rdir, "gsm"), "transfer": supplementary(rdir, "transfer")}

    rows = []
    for name in ("sft", "rlvr", "rlaif"):
        g, t = gsm["policies"][name], tr["policies"][name]
        row = {"policy": name,
               "gsm_acc": g["exact_accuracy"], "svamp_acc": t["exact_accuracy"],
               "acc_drop": g["exact_accuracy"] - t["exact_accuracy"],
               "gsm_format": g["format_compliance"], "svamp_format": t["format_compliance"],
               "gsm_lenient_acc": sup["gsm"][name]["lenient_accuracy"],
               "svamp_lenient_acc": sup["transfer"][name]["lenient_accuracy"],
               "gsm_len": g["length_mean"], "svamp_len": t["length_mean"],
               "gsm_truncated": g["truncated_rate"], "svamp_truncated": t["truncated_rate"]}
        if name != "sft":
            pg, pt = gsm["pairwise_vs_sft"][name], tr["pairwise_vs_sft"][name]
            row.update({"gsm_winrate_vs_sft": pg["win_rate_vs_sft"], "gsm_judge_ties": pg["ties"],
                        "svamp_winrate_vs_sft": pt["win_rate_vs_sft"], "svamp_judge_ties": pt["ties"],
                        "winrate_drop": pg["win_rate_vs_sft"] - pt["win_rate_vs_sft"],
                        "gsm_verifier_judge_agreement": pg["verifier_judge_agreement_on_strict_pairs"],
                        "gsm_identical_to_sft": sup["gsm"][name]["identical_to_sft"],
                        "gsm_winrate_non_identical": sup["gsm"][name]["winrate_non_identical"],
                        "svamp_identical_to_sft": sup["transfer"][name]["identical_to_sft"],
                        "svamp_winrate_non_identical": sup["transfer"][name]["winrate_non_identical"]})
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (rdir / "feedback_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    save_json(rdir / "feedback_comparison.json", {
        "policies": rows, "supplementary": sup, "S_reason": diag["S_reason"], "S_outcome": diag["S_outcome"],
        "by_perturbation": diag["by_perturbation"], "devices": [gsm.get("device"), tr.get("device")]})
    for r in rows:
        print(r)
    print("S_reason", diag["S_reason"], "S_outcome", diag["S_outcome"])


if __name__ == "__main__":
    main()