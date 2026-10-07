"""Common held-out protocol for online-RL policies (PPO now, reused by GRPO).

Fixed: rl_prompt_eval prompts that fit max_prompt_length (no truncation), seed, sampled decoding
from the config, cap = eval_max_response_length. Reports learned reward, sampled KL to the frozen
reference (base model), sampled entropy, response length and truncation; saves every response.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.logprobs import sequence_logprobs
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.continue_train import filter_prompts, generate, response_masks

GEN_BATCH = 16
LP_CHUNK = 4
RM_BATCH = 8


def length_stats(x):
    a = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(np.median(a)),
            "iqr": float(q3 - q1), "n": int(a.size)}


def evaluate_policy(config_path: str, adapter: str, name: str):
    cfg = load_yaml(config_path)
    rdir = repo_path(cfg["results_dir"])
    timer = wall_timer()
    tok = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_prompts(read_jsonl(cfg["paths"]["rl_prompt_eval"]), tok, int(cfg["max_prompt_length"]))
    prompts = [prompt_messages(r) for r in rows]
    ids = [str(r.get("prompt_id")) for r in rows]
    cap = int(cfg.get("eval_max_response_length", cfg["max_response_length"]))

    policy = load_policy(cfg, adapter_path=None if adapter.lower() == "none" else adapter, trainable=False)
    set_seed(int(cfg["seed"]))
    gens = []
    kl_sum = ent_sum = n_tok = 0.0
    with torch.no_grad():
        for s in range(0, len(prompts), GEN_BATCH):
            g = generate(policy, tok, prompts[s:s + GEN_BATCH], cfg, cap)
            full = response_masks(g)
            for c in range(0, full.shape[0], LP_CHUNK):
                sl = slice(c, c + LP_CHUNK)
                _, lp, mask = sequence_logprobs(policy, g["sequences"][sl], g["attention_mask"][sl], full[sl])
                with reference_mode(policy):
                    _, ref, _ = sequence_logprobs(policy, g["sequences"][sl], g["attention_mask"][sl], full[sl])
                d = (lp - ref) * mask
                kl_sum += float(d.sum()); ent_sum += float(-(lp * mask).sum()); n_tok += float(mask.sum())
                for j in range(mask.shape[0]):
                    i = c + j
                    gens.append({"prompt_id": ids[s + i], "response": g["responses"][i],
                                 "length": g["response_lengths"][i], "truncated": g["truncated"][i],
                                 "seq_kl": float(d[j].sum()),
                                 "mean_token_entropy": float(-(lp[j] * mask[j]).sum() / mask[j].sum())})
            print(f"[eval {name}] generated {len(gens)}/{len(prompts)}", flush=True)
    clear_gpu(policy)
    del policy
    torch.cuda.empty_cache()

    rm, rm_tok = load_reward_model(cfg)
    scores = []
    for s in range(0, len(gens), RM_BATCH):
        scores += score_reward_pairs(rm, rm_tok, prompts[s:s + RM_BATCH], [x["response"] for x in gens[s:s + RM_BATCH]],
                                     max_length=int(cfg["reward_max_length"])).tolist()
    for x, sc in zip(gens, scores):
        x["reward"] = sc

    out = rdir / f"generations_{name}.jsonl"
    if out.exists():
        out.unlink()
    for x in gens:
        append_jsonl(out, x)
    summary = {
        "name": name, "adapter": adapter, "n_prompts": len(gens), "n_eval_prompts_dropped_by_filter": len(dropped),
        "max_new_tokens": cap,
        "reward_mean": float(np.mean(scores)), "reward_std": float(np.std(scores)),
        "kl_token_mean": kl_sum / max(n_tok, 1.0),
        "kl_sequence_mean": float(np.mean([x["seq_kl"] for x in gens])),
        "entropy_token_mean": ent_sum / max(n_tok, 1.0),
        "length": length_stats([x["length"] for x in gens]),
        "truncated_rate": float(np.mean([x["truncated"] for x in gens])),
        "wall_clock_s": round(timer(), 1),
    }
    save_json(rdir / f"eval_{name}.json", summary)
    print({k: v for k, v in summary.items() if k != "length"}, flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_policy(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()