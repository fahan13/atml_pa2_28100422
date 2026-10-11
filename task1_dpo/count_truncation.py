"""Task 1: how much the 768-token DPO budget truncates responses (TA announcement, 3 Oct).

For every DPO file, after the same prompt filter used in training/evaluation, count pairs whose
chosen and/or rejected response loses tail tokens, and compare the chosen-minus-rejected length
difference before and after truncation (what the data says vs what DPO actually sees).

    python -m task1_dpo.count_truncation
Output: results/task1_dpo/truncation_profile.json   (tokenizer only, runs on CPU in seconds)
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import (load_yaml, preference_responses, prompt_messages_from_preference,
                         read_jsonl, repo_path)
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.train import filter_rows


def profile(rows, tok, max_len):
    out = defaultdict(list)
    for r in rows:
        p = len(tok.apply_chat_template(prompt_messages_from_preference(r), tokenize=True, add_generation_prompt=True))
        budget = max_len - p - 1                       # one token reserved for EOS, as in encode_prompt_response
        yc, yr = preference_responses(r)
        lc = len(tok(yc, add_special_tokens=False)["input_ids"])
        lr = len(tok(yr, add_special_tokens=False)["input_ids"])
        out["stratum"].append(r.get("length_stratum"))
        out["c_trunc"].append(lc > budget); out["r_trunc"].append(lr > budget)
        out["diff_raw"].append(lc - lr); out["diff_seen"].append(min(lc, budget) - min(lr, budget))
    return out


def summarize(o, idx=None):
    a = {k: np.asarray(v) for k, v in o.items()}
    if idx is not None:
        a = {k: v[idx] for k, v in a.items()}
    n = len(a["diff_raw"])
    return {"n_pairs": n,
            "chosen_truncated": int(a["c_trunc"].sum()), "rejected_truncated": int(a["r_trunc"].sum()),
            "either_truncated": int((a["c_trunc"] | a["r_trunc"]).sum()),
            "both_truncated": int((a["c_trunc"] & a["r_trunc"]).sum()),
            "frac_either_truncated": float((a["c_trunc"] | a["r_trunc"]).mean()) if n else 0.0,
            "mean_diff_raw": float(a["diff_raw"].mean()), "mean_diff_seen": float(a["diff_seen"].mean()),
            "frac_chosen_longer_raw": float((a["diff_raw"] > 0).mean()),
            "frac_chosen_longer_seen": float((a["diff_seen"] > 0).mean()),
            "frac_equal_length_seen": float((a["diff_seen"] == 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    res = {"max_sequence_length": max_len, "files": {}}
    for key in ("dpo_standard_train", "dpo_length_train", "dpo_standard_eval", "dpo_length_eval"):
        rows, dropped = filter_rows(read_jsonl(cfg["paths"][key]), tok, max_len)
        o = profile(rows, tok, max_len)
        entry = {"dropped_by_prompt_filter": len(dropped), **summarize(o)}
        strata = [s for s in set(o["stratum"]) if s is not None]
        if strata:
            st = np.asarray(o["stratum"], dtype=object)
            entry["by_stratum"] = {s: summarize(o, st == s) for s in sorted(strata)}
        res["files"][key] = entry
        print(key, {k: v for k, v in entry.items() if k != "by_stratum"}, flush=True)
    out = repo_path(cfg["results_dir"]) / "truncation_profile.json"
    save_json(out, res)
    print("saved", out)


if __name__ == "__main__":
    main()
