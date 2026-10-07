from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.logprobs import sequence_logprobs
from common.metrics import sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


# ----------------------------------------------------------------------------- shared helpers
def disable_dropout(model):
    """Dropout off so old/new log-probs differ only through parameter updates (TRL default)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def upcast_trainable(model):
    """fp16 base, fp32 trainable weights (same mixed-precision scheme as Task 1)."""
    for p in trainable_parameters(model):
        p.data = p.data.float()


def filter_prompts(rows, tokenizer, max_prompt_length):
    """Keep prompts whose rendered chat prompt fits max_prompt_length, so batch_generate never
    truncates away the assistant header. Same rule for training and evaluation prompts."""
    kept, dropped = [], []
    for r in rows:
        n = len(tokenizer.apply_chat_template(prompt_messages(r), tokenize=True, add_generation_prompt=True))
        (kept if n <= max_prompt_length else dropped).append(r)
    return kept, [str(r.get("prompt_id")) for r in dropped]


def prompt_schedule(cfg, tokenizer, n):
    """Fixed, seeded prompt order; every fork consumes a prefix of the same order."""
    rows, dropped = filter_prompts(read_jsonl(cfg["paths"]["rl_prompt_train"]), tokenizer,
                                   int(cfg["max_prompt_length"]))
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    order = torch.randperm(len(rows), generator=g).tolist()
    return [rows[i] for i in order[:n]], dropped


def value_forward(value_model, input_ids, attention_mask):
    """Per-position values. Same computation as common.models.token_values (last normed hidden
    state -> scalar head), but runs the fp32 head on fp32 hidden states."""
    cls = value_model.base_model.model                 # Qwen2ForSequenceClassification (LoRA inside)
    backbone = getattr(cls, cls.base_model_prefix)     # Qwen2Model
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                      return_dict=True).last_hidden_state
    return value_model.score(hidden.float()).squeeze(-1)


def response_masks(gen):
    full = torch.zeros_like(gen["sequences"], dtype=torch.float32)
    full[:, gen["prompt_width"]:] = gen["response_mask"]
    return full


def generate(policy, tokenizer, prompts, cfg, max_new_tokens):
    policy.config.use_cache = True                      # KV cache for generation only
    g = batch_generate(policy, tokenizer, prompts, int(cfg["max_prompt_length"]), int(max_new_tokens),
                       float(cfg["generation"]["temperature"]), float(cfg["generation"]["top_p"]),
                       bool(cfg["generation"]["do_sample"]))
    policy.config.use_cache = False
    return g


# ----------------------------------------------------------------------------- setup (starter)
def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    for m in (policy, value_model):
        disable_dropout(m)
        upcast_trainable(m)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


# ----------------------------------------------------------------------------- PPO loop
def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy, value = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    opt_p, opt_v = bundle["policy_optimizer"], bundle["value_optimizer"]
    n_updates, ppu = int(cfg["updates"]), int(cfg["prompts_per_update"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    max_gn = float(cfg["max_grad_norm"])
    schedule, dropped = prompt_schedule(cfg, tok, n_updates * ppu)

    rdir = repo_path(cfg["results_dir"])
    log_path = rdir / f"{run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    scaler_p, scaler_v = torch.amp.GradScaler("cuda"), torch.amp.GradScaler("cuda")
    device = next(policy.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    tokens_generated = 0
    print(f"[{run_name}] updates={n_updates} eps={eps} kl_beta={beta} prompts/update={ppu}", flush=True)

    for u in range(n_updates):
        rows = schedule[u * ppu:(u + 1) * ppu]
        prompts = [prompt_messages(r) for r in rows]

        # ---- 1. rollout from the current (= old) policy
        g = generate(policy, tok, prompts, cfg, cfg["max_response_length"])
        seq, attn, pw = g["sequences"], g["attention_mask"], g["prompt_width"]
        full_mask = response_masks(g)
        tokens_generated += int(sum(g["response_lengths"]))

        # ---- 2. old / reference log-probs, values, reward, shaped reward, GAE
        with torch.no_grad():
            _, old_lp, mask = sequence_logprobs(policy, seq, attn, full_mask)
            with reference_mode(policy):
                _, ref_lp, _ = sequence_logprobs(policy, seq, attn, full_mask)
            values = value_forward(value, seq, attn)[:, pw - 1:-1].float()
            rm_score = score_reward_pairs(rm, rm_tok, prompts, g["responses"],
                                          max_length=int(cfg["reward_max_length"])).float().to(device)
            penalty = torch.tensor([float(cfg["missing_eos_penalty"]) if t else 0.0 for t in g["truncated"]],
                                   device=device)
            task_reward = rm_score - penalty
            rewards = shaped_rewards(task_reward, old_lp, ref_lp, mask, beta)
            adv, returns = compute_gae(rewards, values, mask, gamma, lam)
            adv_n = normalize_advantages(adv, mask)
            m = mask.bool()
            resid = (returns - values)[m]
            explained_var = float(1.0 - resid.var(unbiased=False) / returns[m].var(unbiased=False).clamp_min(1e-8))

        # ---- 3. PPO epochs on this rollout
        st = defaultdict(list)
        for _ in range(int(cfg["ppo_epochs"])):
            _, new_lp, _ = sequence_logprobs(policy, seq, attn, full_mask)
            ploss, ratio, clipfrac = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            opt_p.zero_grad(set_to_none=True)
            scaler_p.scale(ploss).backward()
            scaler_p.unscale_(opt_p)
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_gn)
            scaler_p.step(opt_p)
            scaler_p.update()

            v_new = value_forward(value, seq, attn)[:, pw - 1:-1]
            vloss = float(cfg["value_coef"]) * value_mse_loss(v_new, returns, mask)
            opt_v.zero_grad(set_to_none=True)
            scaler_v.scale(vloss).backward()
            scaler_v.unscale_(opt_v)
            vgn = torch.nn.utils.clip_grad_norm_(trainable_parameters(value), max_gn)
            scaler_v.step(opt_v)
            scaler_v.update()

            log_r = (new_lp.detach() - old_lp)[m]
            st["policy_loss"].append(ploss.item())
            st["value_loss"].append(vloss.item())
            st["grad_norm"].append(float(gn))
            st["value_grad_norm"].append(float(vgn))
            st["clip_fraction"].append(clipfrac.item())
            st["approx_kl_old_new"].append(float(((ratio[m] - 1) - log_r).mean()))   # k3 estimator
            st["max_ratio"].append(float(ratio[m].max()))
            st["min_ratio"].append(float(ratio[m].min()))

        rec = {k: sum(v) / len(v) for k, v in st.items()}
        rec.update({
            "update": u + 1,
            "prompt_ids": [str(r.get("prompt_id")) for r in rows],
            "reward": float(rm_score.mean()),
            "task_reward": float(task_reward.mean()),
            "kl": float(sampled_kl(old_lp, ref_lp, mask)),
            "kl_seq": float(((old_lp - ref_lp) * mask).sum(-1).mean()),
            "entropy": float(sample_entropy(old_lp, mask)),
            "response_length": float(mask.sum(-1).mean()),
            "truncated": float(sum(g["truncated"]) / len(g["truncated"])),
            "clip_fraction_last_epoch": st["clip_fraction"][-1],
            "max_ratio_max": max(st["max_ratio"]),
            "value_mean": float(values[m].mean()),
            "return_mean": float(returns[m].mean()),
            "explained_variance": explained_var,
            "tokens_generated": tokens_generated,
            "grad_scale_policy": scaler_p.get_scale(),
            "elapsed_s": round(elapsed(), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        })
        append_jsonl(log_path, rec)
        print(f"[{run_name}] upd {u + 1:2d}/{n_updates} R {rec['reward']:+.3f} KL {rec['kl']:+.4f} "
              f"len {rec['response_length']:.0f} clip {rec['clip_fraction_last_epoch']:.3f} "
              f"vloss {rec['value_loss']:.3f} EV {explained_var:+.2f} {rec['elapsed_s']:.0f}s "
              f"vram {rec['peak_vram_gb']}GB", flush=True)

    policy.save_pretrained(str(out))
    save_json(rdir / f"{run_name}_train_summary.json", {
        "run_name": run_name, "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta,
        "prompt_ids": [str(r.get("prompt_id")) for r in schedule],
        "prompt_filter": f"rendered prompt tokens <= max_prompt_length ({cfg['max_prompt_length']})",
        "n_train_prompts_dropped_by_filter": len(dropped),
        "tokens_generated": tokens_generated,
        "wall_clock_s": round(elapsed(), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapter": str(out),
        "config": {k: cfg[k] for k in ["seed", "ppo_epochs", "policy_learning_rate", "value_lora_learning_rate",
                                       "value_head_learning_rate", "value_train_mode", "gamma", "gae_lambda",
                                       "value_coef", "missing_eos_penalty", "max_prompt_length",
                                       "max_response_length", "reward_max_length", "max_grad_norm", "generation"]},
    })
    print(f"[{run_name}] saved policy adapter to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()