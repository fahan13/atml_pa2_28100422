"""Task 5 supplementary check: are the judge's TIE verdicts real, or parse failures?

The released PairwiseAIJudge maps any unparseable output to "TIE". This probe sends the judge
the 40 clean-vs-perturbed diagnostic pairs whose perturbation should be detectable
(good_reasoning_wrong_final, corrupt_reasoning_correct_final) with the released rubric and
decoding, WITHOUT the cache or the A/B swap, and counts the exact raw text it generates.
Output: results/task5_feedback/judge_raw_output_probe.json
"""
from __future__ import annotations

import argparse
import collections
import json

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge

PERTURBATIONS = ["good_reasoning_wrong_final", "corrupt_reasoning_correct_final"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    judge = PairwiseAIJudge(cfg, repo_path("results/task5_feedback/_probe_unused_cache.json"))
    by = collections.defaultdict(dict)
    for r in read_jsonl(cfg["paths"]["task5_diagnostics"]):
        by[r["problem_id"]][r["variant_type"]] = r
    raw = collections.Counter()
    for v in by.values():
        for pert in PERTURBATIONS:
            text = PAIRWISE_RUBRIC.format(problem=v["clean_correct"]["question"],
                                          a=v["clean_correct"]["response"], b=v[pert]["response"])
            ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                                      add_generation_prompt=True).to(judge.model.device)
            out = judge.model.generate(ids, max_new_tokens=4, do_sample=False,
                                       pad_token_id=judge.tokenizer.eos_token_id)
            raw[judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()] += 1
    print(raw.most_common(15))
    out = repo_path("results/task5_feedback/judge_raw_output_probe.json")
    out.write_text(json.dumps(dict(raw), indent=2), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    main()
