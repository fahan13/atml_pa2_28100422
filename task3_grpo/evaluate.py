"""GRPO held-out evaluation: the identical protocol used for PPO (task2_ppo/evaluate.py) - same
eval prompts, seed, sampled decoding, 768-token cap (grpo.yaml: cache_generation_cap, equal to the
PPO eval cap), same reward-model truncation - so PPO and GRPO numbers are directly comparable."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.logprobs import sequence_logprobs
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.continue_train import filter_prompts, generate, response_masks
from task2_ppo.evaluate import GEN_BATCH, LP_CHUNK, RM_BATCH, length_stats
from task3_grpo.continue_train import REWARD_MAX_LENGTH


def evaluate_grpo(config_path: str, adapter: str, name: str):
    cfg = load_yaml(config_path)
    rdir = repo_path(cfg["results_dir"])
    timer = wall_timer()
    tok = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_prompts(read_jsonl(cfg["paths"]["rl_prompt_eval"]), tok, int(cfg["max_prompt_length"]))
    prompts = [prompt_messages(r) for r in rows]
    ids = [str(r.get("prompt_id")) for r in rows]
    cap = int(cfg["cache_generation_cap"])

    policy = load_policy(cfg, adapter_path=None if adapter.lower() == "none" else adapter, trainable=False)
    set_seed(int(cfg["seed"]))
    gens, kl_sum, ent_sum, n_tok = [], 0.0, 0.0, 0.0
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
                                     max_length=REWARD_MAX_LENGTH).tolist()
    for x, sc in zip(gens, scores):
        x["reward"] = sc
    write_jsonl(rdir / f"generations_{name}.jsonl", gens)
    summary = {
        "name": name, "adapter": adapter, "n_prompts": len(gens), "n_eval_prompts_dropped_by_filter": len(dropped),
        "max_new_tokens": cap, "reward_mean": float(np.mean(scores)), "reward_std": float(np.std(scores)),
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
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_grpo(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()