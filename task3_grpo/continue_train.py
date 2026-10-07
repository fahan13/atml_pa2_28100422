from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.logprobs import sequence_logprobs
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task2_ppo.continue_train import disable_dropout, filter_prompts, generate, response_masks, upcast_trainable
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

# A group is uninformative when its reward std is within the tolerance used by the released
# group_relative_advantages helper (its eps); such groups give all-zero advantages.
STD_TOL = 1e-6
REWARD_MAX_LENGTH = 1280   # grpo.yaml has no RM length key; same RM truncation as the PPO config


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)
    upcast_trainable(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def prompt_schedule(cfg, tokenizer, n):
    rows, dropped = filter_prompts(read_jsonl(cfg["paths"]["rl_prompt_train"]), tokenizer, int(cfg["max_prompt_length"]))
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    order = torch.randperm(len(rows), generator=g).tolist()
    return [rows[i] for i in order[:n]], dropped


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy, rm, rm_tok, opt = (bundle["tokenizer"], bundle["policy"], bundle["reward_model"],
                                     bundle["reward_tokenizer"], bundle["optimizer"])
    n_updates, ppu, K = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["num_generations"])
    eps, beta, Lmax = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["max_completion_length"])
    schedule, dropped = prompt_schedule(cfg, tok, n_updates * ppu)
    gen_cfg = dict(cfg, max_prompt_length=cfg["max_prompt_length"])

    rdir = repo_path(cfg["results_dir"])
    log_path = rdir / f"{run_name}_train_log.jsonl"
    comp_path = rdir / f"{run_name}_completions.jsonl"
    for p in (log_path, comp_path):
        if p.exists():
            p.unlink()
    scaler = torch.amp.GradScaler("cuda")
    device = next(policy.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    tokens_generated = 0
    print(f"[{run_name}] loss_type={loss_type} updates={n_updates} K={K} eps={eps} beta={beta}", flush=True)

    for u in range(n_updates):
        rows = schedule[u * ppu:(u + 1) * ppu]
        prompts = [prompt_messages(r) for r in rows for _ in range(K)]      # K completions per prompt
        group_ids = torch.tensor([i for i in range(len(rows)) for _ in range(K)], device=device)

        g = generate(policy, tok, prompts, gen_cfg, Lmax)
        seq, attn = g["sequences"], g["attention_mask"]
        full_mask = response_masks(g)
        tokens_generated += int(sum(g["response_lengths"]))

        with torch.no_grad():
            _, old_lp, mask = sequence_logprobs(policy, seq, attn, full_mask)
            with reference_mode(policy):
                _, ref_lp, _ = sequence_logprobs(policy, seq, attn, full_mask)
            rewards = score_reward_pairs(rm, rm_tok, prompts, g["responses"], max_length=REWARD_MAX_LENGTH).float()
            adv = group_relative_advantages(rewards, group_ids)
        loss_mask = mask_truncated_sequences(mask, g["truncated"]) if cfg.get("mask_truncated_completions", True) else mask

        group_std, uninformative = [], 0
        for gid in range(len(rows)):
            r = rewards[group_ids == gid]
            s = float(r.std(unbiased=False))
            group_std.append(s)
            uninformative += s <= STD_TOL

        stats = []
        for _ in range(int(cfg["policy_epochs"])):
            _, new_lp, _ = sequence_logprobs(policy, seq, attn, full_mask)
            loss, diag = grpo_policy_loss(new_lp, old_lp, adv, loss_mask, ref_lp, eps, beta,
                                          loss_type=loss_type, max_completion_length=Lmax)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"])))
            scaler.step(opt)
            scaler.update()
            stats.append({"loss": loss.item(), "grad_norm": gn, **{k: float(v) for k, v in diag.items()}})

        lengths = mask.sum(-1).tolist()
        # Length-conditioned weights implied by the normalization (per completion):
        #   grpo    : each token weighted |A|/T_k   -> whole completion carries |A|
        #   dr_grpo : each token weighted |A|/Lmax  -> whole completion carries |A|*T_k/Lmax
        kept = loss_mask.sum(-1) > 0
        for i in range(len(prompts)):
            T = lengths[i]
            tw = (abs(float(adv[i])) / (T if loss_type == "grpo" else Lmax)) if kept[i] else 0.0
            append_jsonl(comp_path, {
                "update": u + 1, "prompt_id": str(rows[int(group_ids[i])].get("prompt_id")), "k": i % K,
                "length": T, "truncated": g["truncated"][i], "in_loss": bool(kept[i]),
                "reward": float(rewards[i]), "advantage": float(adv[i]),
                "token_weight": tw, "sequence_weight": tw * T,
                "kl_seq": float(((old_lp[i] - ref_lp[i]) * mask[i]).sum()),
                "response": g["responses"][i]})

        s0 = stats[-1]
        rec = {
            "update": u + 1,
            "prompt_ids": [str(r.get("prompt_id")) for r in rows],
            "reward": float(rewards.mean()),
            "reward_std_within_group": float(np.mean(group_std)),
            "uninformative_group_fraction": uninformative / len(rows),
            "kl": float(((old_lp - ref_lp) * mask).sum() / mask.sum()),        # sampled estimator, as in Tasks 1-2
            "kl_k3_in_loss": s0["sampled_kl"],
            "policy_term": s0["policy_term"],
            "loss": s0["loss"],
            "grad_norm": s0["grad_norm"],
            "clip_fraction": s0["clip_fraction"],
            "entropy": float(-(old_lp * mask).sum() / mask.sum()),
            "response_length": float(np.mean(lengths)),
            "truncated_fraction": float(np.mean(g["truncated"])),
            "masked_completions": int((~kept).sum()),
            "tokens_generated": tokens_generated,
            "grad_scale": scaler.get_scale(),
            "elapsed_s": round(elapsed(), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        append_jsonl(log_path, rec)
        print(f"[{run_name}] upd {u + 1:2d}/{n_updates} R {rec['reward']:+.3f} std {rec['reward_std_within_group']:.3f} "
              f"KL {rec['kl']:+.4f} len {rec['response_length']:.0f} masked {rec['masked_completions']} "
              f"gn {rec['grad_norm']:.3f} {rec['elapsed_s']:.0f}s vram {rec['peak_vram_gb']}GB", flush=True)

    policy.save_pretrained(str(out))
    save_json(rdir / f"{run_name}_train_summary.json", {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "num_generations": K,
        "clip_epsilon": eps, "kl_beta": beta, "max_completion_length": Lmax,
        "prompt_ids": [str(r.get("prompt_id")) for r in schedule],
        "n_train_prompts_dropped_by_filter": len(dropped),
        "uninformative_std_tolerance": STD_TOL, "reward_max_length": REWARD_MAX_LENGTH,
        "tokens_generated": tokens_generated,
        "wall_clock_s": round(elapsed(), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapter": str(out),
        "config": {k: cfg[k] for k in ["seed", "policy_epochs", "learning_rate", "mask_truncated_completions",
                                       "max_prompt_length", "max_grad_norm", "generation"]},
    })
    print(f"[{run_name}] saved policy adapter to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()