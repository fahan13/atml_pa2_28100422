"""Task 1 evaluation under one fixed protocol for every DPO condition.

  pairs : held-out DPO loss + preference accuracy (manual definition: reference-adjusted margin > 0)
  gen   : sampled responses on a fixed prompt set -> KL to reference, reward-model score, length
  wl    : the 10 common word-limit prompts -> word count + compliance
Use --adapter none for the untouched SFT baseline.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.logprobs import sequence_logprobs
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.logging_utils import append_jsonl
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import MIN_RESPONSE_ROOM, filter_rows, make_collate, to_device

PAIR_BATCH = 4
GEN_BATCH = 16
N_GEN = 200            # fixed generation prompts: first 200 filtered rows of dpo_standard_eval
GEN_PROMPT_MAX = 512   # generation prompts longer than this are excluded (rule fixed once)
RM_BATCH = 8


def length_stats(lengths):
    a = np.asarray(lengths, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "iqr": float(q3 - q1), "n": int(a.size)}


@torch.no_grad()
def eval_pairs(model, tokenizer, cfg, rows, beta, out_jsonl):
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    margins, losses, strata = [], [], []
    if out_jsonl.exists():
        out_jsonl.unlink()
    for s in range(0, len(rows), PAIR_BATCH):
        chunk = rows[s:s + PAIR_BATCH]
        ch, rj = collate(chunk)
        ch, rj = to_device(ch, device), to_device(rj, device)
        pol_c, _, _ = sequence_logprobs(model, **ch)
        pol_r, _, _ = sequence_logprobs(model, **rj)
        with reference_mode(model):
            ref_c, _, _ = sequence_logprobs(model, **ch)
            ref_r, _, _ = sequence_logprobs(model, **rj)
        m = (pol_c - ref_c) - (pol_r - ref_r)
        for j, row in enumerate(chunk):
            l, _ = dpo_loss(pol_c[j:j+1], pol_r[j:j+1], ref_c[j:j+1], ref_r[j:j+1], beta)
            rec = {"prompt_id": str(row.get("prompt_id")), "stratum": row.get("length_stratum"),
                   "margin": float(m[j]), "dpo_loss": float(l),
                   "policy_chosen_logp": float(pol_c[j]), "policy_rejected_logp": float(pol_r[j]),
                   "ref_chosen_logp": float(ref_c[j]), "ref_rejected_logp": float(ref_r[j])}
            append_jsonl(out_jsonl, rec)
            margins.append(rec["margin"]); losses.append(rec["dpo_loss"]); strata.append(rec["stratum"])
    margins = np.asarray(margins)
    res = {"n_pairs": int(margins.size), "dpo_loss": float(np.mean(losses)),
           "preference_accuracy": float((margins > 0).mean()), "mean_margin": float(margins.mean())}
    if any(s is not None for s in strata):
        res["by_stratum"] = {}
        for st in sorted({s for s in strata if s is not None}):
            idx = np.array([s == st for s in strata])
            res["by_stratum"][st] = {"n": int(idx.sum()),
                                     "preference_accuracy": float((margins[idx] > 0).mean()),
                                     "dpo_loss": float(np.mean(np.asarray(losses)[idx])),
                                     "mean_margin": float(margins[idx].mean())}
    return res


@torch.no_grad()
def generate_and_score_kl(model, tokenizer, cfg, prompts, ids, max_prompt):
    gcfg = cfg["generation"]
    set_seed(int(cfg["seed"]))
    out, kl_sum, kl_tokens = [], 0.0, 0.0
    for s in range(0, len(prompts), GEN_BATCH):
        g = batch_generate(model, tokenizer, prompts[s:s + GEN_BATCH], max_prompt,
                           int(cfg["max_generation_tokens"]), float(gcfg["temperature"]),
                           float(gcfg["top_p"]), bool(gcfg["do_sample"]))
        pw = g["prompt_width"]
        full_mask = torch.zeros_like(g["sequences"], dtype=torch.float32)
        full_mask[:, pw:] = g["response_mask"]
        diffs = []
        for c in range(0, full_mask.shape[0], 4):          # 4 rows at a time keeps logits small
            sl = slice(c, c + 4)
            _, pol_tok, mask = sequence_logprobs(model, g["sequences"][sl], g["attention_mask"][sl], full_mask[sl])
            with reference_mode(model):
                _, ref_tok, _ = sequence_logprobs(model, g["sequences"][sl], g["attention_mask"][sl], full_mask[sl])
            d = (pol_tok - ref_tok) * mask
            kl_sum += float(d.sum()); kl_tokens += float(mask.sum())
            diffs += [float(x) for x in d.sum(-1)]
        for j in range(len(g["responses"])):
            out.append({"prompt_id": ids[s + j], "response": g["responses"][j],
                        "length": g["response_lengths"][j], "truncated": g["truncated"][j],
                        "seq_kl": diffs[j]})
    return out, kl_sum / max(kl_tokens, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the SFT baseline")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used to report held-out DPO loss (default: config beta)")
    ap.add_argument("--parts", default="pairs,stratified,gen,wl")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    beta = float(args.beta if args.beta is not None else cfg["beta"])
    parts = set(args.parts.split(","))
    rdir = repo_path(cfg["results_dir"])
    timer = wall_timer()

    tokenizer = load_tokenizer(cfg["base_model"])
    adapter = None if args.adapter.lower() == "none" else args.adapter
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    summary = {"name": args.name, "adapter": args.adapter, "beta_for_loss": beta}
    max_len = int(cfg["max_sequence_length"])

    if adapter is not None and "pairs" in parts:
        rows, dropped = filter_rows(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tokenizer, max_len)
        summary["heldout"] = eval_pairs(model, tokenizer, cfg, rows, beta, rdir / f"pairs_{args.name}_standard.jsonl")
        summary["heldout"]["dropped_prompt_ids"] = dropped
        print("held-out:", {k: v for k, v in summary["heldout"].items() if k != "dropped_prompt_ids"})
    if adapter is not None and "stratified" in parts:
        rows, dropped = filter_rows(read_jsonl(cfg["paths"]["dpo_length_eval"]), tokenizer, max_len)
        summary["stratified"] = eval_pairs(model, tokenizer, cfg, rows, beta, rdir / f"pairs_{args.name}_stratified.jsonl")
        summary["stratified"]["dropped_prompt_ids"] = dropped
        print("stratified:", summary["stratified"].get("by_stratum"))

    gens, wl = None, None
    if "gen" in parts:
        rows, _ = filter_rows(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tokenizer, GEN_PROMPT_MAX + MIN_RESPONSE_ROOM)
        rows = rows[:N_GEN]
        prompts = [prompt_messages_from_preference(r) for r in rows]
        ids = [str(r.get("prompt_id")) for r in rows]
        gens, kl_tok = generate_and_score_kl(model, tokenizer, cfg, prompts, ids, GEN_PROMPT_MAX)
        summary["generation"] = {"n_prompts": len(gens), "kl_token_mean": kl_tok,
                                 "kl_sequence_mean": float(np.mean([g["seq_kl"] for g in gens])),
                                 "length": length_stats([g["length"] for g in gens]),
                                 "truncated_rate": float(np.mean([g["truncated"] for g in gens]))}
        gen_prompts = prompts
    if "wl" in parts:
        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wl_prompts = [prompt_messages(r) for r in wl_rows]
        wl, _ = generate_and_score_kl(model, tokenizer, cfg, wl_prompts,
                                      [str(r["prompt_id"]) for r in wl_rows], GEN_PROMPT_MAX)
        for r, p in zip(wl, wl_prompts):
            r["words"] = word_count(r["response"])
            r["compliant"] = word_limit_compliance(p[-1]["content"], r["response"])
        comp = [r["compliant"] for r in wl if r["compliant"] is not None]
        summary["word_limit"] = {"compliance": float(np.mean(comp)), "n": len(comp),
                                 "words": length_stats([r["words"] for r in wl]),
                                 "tokens": length_stats([r["length"] for r in wl])}

    clear_gpu(model)
    del model
    torch.cuda.empty_cache()

    if gens is not None:
        rm, rm_tok = load_reward_model(cfg)
        scores = []
        for s in range(0, len(gens), RM_BATCH):
            scores += score_reward_pairs(rm, rm_tok, gen_prompts[s:s + RM_BATCH],
                                         [g["response"] for g in gens[s:s + RM_BATCH]]).tolist()
        for g, sc in zip(gens, scores):
            g["reward"] = sc
        summary["generation"]["reward_mean"] = float(np.mean(scores))
        summary["generation"]["reward_std"] = float(np.std(scores))
        out = rdir / f"generations_{args.name}.jsonl"
        if out.exists():
            out.unlink()
        for g in gens:
            append_jsonl(out, g)
    if wl is not None:
        out = rdir / f"wordlimit_{args.name}.jsonl"
        if out.exists():
            out.unlink()
        for r in wl:
            append_jsonl(out, r)

    summary["wall_clock_s"] = round(timer(), 1)
    save_json(rdir / f"eval_{args.name}.json", summary)
    print({k: v for k, v in summary.items() if k not in ("heldout", "stratified")})


if __name__ == "__main__":
    main()