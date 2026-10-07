"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on GSM8K (in-domain) or SVAMP (transfer).

Fixed protocol for all three policies: the dataset's own chat messages (which carry the
"end with exactly `#### <number>`" instruction), greedy decoding, math_max_new_tokens cap,
released verifier for accuracy/format, released pairwise judge for policy-vs-SFT win rate.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

GEN_BATCH = 16
MAX_PROMPT = 1024          # math prompts are short; large cap guarantees no truncation
INSTRUCTION = "\n\nShow your reasoning and end your response with exactly `#### <number>`."


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def messages_for(row):
    if isinstance(row.get("messages"), list):
        return prompt_messages(row)
    return [{"role": "user", "content": str(row["question"]) + INSTRUCTION}]


def row_id(row, i):
    return str(row.get("prompt_id", row.get("source_index", i)))


def failure_type(rec):
    if rec["correct"]:
        return "correct"
    if rec["truncated"]:
        return "truncated_no_final" if rec["pred"] is None else "truncated_wrong"
    return "no_designated_final" if rec["pred"] is None else "wrong_final"


def generate_all(cfg, tok, rows, name):
    policy = load_frozen_policy(cfg, name)
    set_seed(int(cfg["seed"]))
    msgs = [messages_for(r) for r in rows]
    out = []
    for s in range(0, len(rows), GEN_BATCH):
        g = batch_generate(policy, tok, msgs[s:s + GEN_BATCH], MAX_PROMPT, int(cfg["math_max_new_tokens"]),
                           do_sample=False)
        for j, text in enumerate(g["responses"]):
            r = rows[s + j]
            pred = extract_designated_final(text)
            rec = {"id": row_id(r, s + j), "policy": name, "response": text,
                   "length": g["response_lengths"][j],
                   "truncated": g["truncated"][j], "gold": str(r["gold_final"]), "pred": pred,
                   "format_ok": pred is not None, "correct": bool(exact_reward(text, str(r["gold_final"])))}
            rec["failure_type"] = failure_type(rec)
            out.append(rec)
        print(f"[{name}] {len(out)}/{len(rows)}", flush=True)
    clear_gpu(policy)
    del policy
    torch.cuda.empty_cache()
    return out


def stats(recs):
    L = np.asarray([r["length"] for r in recs], dtype=float)
    q1, q3 = np.percentile(L, [25, 75])
    ft = {}
    for r in recs:
        ft[r["failure_type"]] = ft.get(r["failure_type"], 0) + 1
    return {"n": len(recs), "exact_accuracy": float(np.mean([r["correct"] for r in recs])),
            "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
            "truncated_rate": float(np.mean([r["truncated"] for r in recs])),
            "length_mean": float(L.mean()), "length_std": float(L.std()), "length_iqr": float(q3 - q1),
            "failure_types": ft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    rdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    rdir.mkdir(parents=True, exist_ok=True)
    timer = wall_timer()

    gens = {}
    for name in policy_specs(cfg):
        path = rdir / f"generations_{args.dataset}_{name}.jsonl"
        if path.exists():
            gens[name] = read_jsonl(path)
            print(f"loaded cached generations for {name}")
            continue
        gens[name] = generate_all(cfg, tok, rows, name)
        write_jsonl(path, gens[name])

    # ---- pairwise AI judge: each RL policy vs SFT on the same problem
    judge = PairwiseAIJudge(cfg, rdir / f"judge_cache_{args.dataset}.json")
    pairwise, comparisons = {}, []
    for name in ("rlvr", "rlaif"):
        wins = ties = losses = 0
        agree = disagree = judge_tie_when_verifier_strict = judge_strict_when_verifier_tie = verifier_strict = 0
        for i, row in enumerate(rows):
            a, b = gens[name][i], gens["sft"][i]
            pref = judge.compare(str(row["question"]), a["response"], b["response"])
            score = {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref]
            wins += pref == "A"; losses += pref == "B"; ties += pref == "TIE"
            v = int(a["correct"]) - int(b["correct"])            # verifier preference: +1 policy, -1 SFT, 0 tie
            j = {"A": 1, "B": -1, "TIE": 0}[pref]
            if v != 0:
                verifier_strict += 1
                if j == 0:
                    judge_tie_when_verifier_strict += 1
                elif j == v:
                    agree += 1
                else:
                    disagree += 1
            elif j != 0:
                judge_strict_when_verifier_tie += 1
            comparisons.append({"id": a["id"], "policy": name, "judge": pref, "score": score,
                                "policy_correct": a["correct"], "sft_correct": b["correct"]})
        n = len(rows)
        pairwise[name] = {
            "win_rate_vs_sft": (wins + 0.5 * ties) / n, "wins": wins, "ties": ties, "losses": losses,
            "verifier_strict_pairs": verifier_strict,
            "verifier_judge_agreement_on_strict_pairs": agree / max(verifier_strict, 1),
            "judge_opposes_verifier": disagree, "judge_ties_when_verifier_strict": judge_tie_when_verifier_strict,
            "judge_strict_when_verifier_ties": judge_strict_when_verifier_tie,
            "verifier_ties": n - verifier_strict,
        }
        print(name, pairwise[name], flush=True)

    write_jsonl(rdir / f"judge_comparisons_{args.dataset}.jsonl", comparisons)
    summary = {"dataset": args.dataset, "n": len(rows),
               "decoding": {"greedy": True, "max_new_tokens": int(cfg["math_max_new_tokens"])},
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "policies": {k: stats(v) for k, v in gens.items()}, "pairwise_vs_sft": pairwise,
               "wall_clock_s": round(timer(), 1)}
    save_json(rdir / f"math_{args.dataset}_summary.json", summary)
    for k, v in summary["policies"].items():
        print(k, {x: v[x] for x in ("exact_accuracy", "format_compliance", "length_mean", "truncated_rate")})


if __name__ == "__main__":
    main()