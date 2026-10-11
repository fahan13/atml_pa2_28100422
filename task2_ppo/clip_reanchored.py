"""Task 2 Step 2, re-anchored cached-batch clipping study.

The original cached study measured the ratio between our fresh forward pass of the supplied
midpoint and the cache's stored old log-probs. Before any update that ratio should be exactly 1
(the cache was produced by the midpoint), but it was not: the re-computation differs from the
stored values, so the step-0 "clip fraction" measured a re-computation mismatch, not clipping.

This script
  1. diagnoses that mismatch: our midpoint log-probs vs cached old log-probs, and our reference
     (adapter disabled) log-probs vs cached ref log-probs, with the most extreme tokens listed;
  2. re-anchors pi_old := our own midpoint log-probs (so step 0 is exactly 0 clipping, the
     baseline the TAs describe), keeps the cached advantages, and takes PPO steps on the fixed
     batch for each epsilon with the release learning rate, logging clip/affected fractions per step.

    python -m task2_ppo.clip_reanchored --steps 10
Output: results/task2_ppo/clip_cached_reanchored.json
"""
from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from common.data import load_yaml, repo_path
from common.logging_utils import save_json, set_seed
from common.logprobs import sequence_logprobs
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from task2_ppo.analyze_clipping import encode, geometry, load_cached_rollouts, new_logprobs, rebuild_batch
from task2_ppo.continue_train import disable_dropout, upcast_trainable
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def mismatch(mine, cached, mask, items, tok, top=12):
    m = mask.bool()
    d = (mine - cached)[m]
    out = {"mean_abs_log_ratio": float(d.abs().mean()), "max_log_ratio": float(d.max()),
           "min_log_ratio": float(d.min())}
    for e in (0.05, 0.2, 0.5):
        out[f"frac_outside_{e}"] = float(((d.exp() > 1 + e) | (d.exp() < 1 - e)).float().mean())
    diff = (mine - cached).masked_fill(~m, 0.0)
    flat = diff.abs().flatten().topk(top).indices
    worst = []
    for idx in flat.tolist():
        i, t = divmod(idx, diff.shape[1])
        ids = items[i]["ids"]
        worst.append({"row": i, "pos": t, "len": len(ids), "token_id": int(ids[t]),
                      "token": tok.decode([ids[t]]), "is_last": t == len(ids) - 1,
                      "cached": float(cached[i, t]), "ours": float(mine[i, t])})
    out["worst_tokens"] = worst
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    items, old, ref, val, mask, reward, mismatched = rebuild_batch(cfg, tok, load_cached_rollouts(cfg["cached_rollouts"]))
    rewards = shaped_rewards(reward, old, ref, mask, float(cfg["kl_beta"]))
    adv, _ = compute_gae(rewards, val, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = normalize_advantages(adv, mask)
    Tm = mask.shape[1]
    res = {"n_used": len(items), "n_tokens": int(mask.sum()), "retokenization_mismatches": mismatched,
           "steps": args.steps, "learning_rate": float(cfg["policy_learning_rate"]), "per_epsilon": {}}

    for k, eps in enumerate(cfg["clip_values"]):
        eps = float(eps)
        set_seed(int(cfg["seed"]))
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
        disable_dropout(policy); upcast_trainable(policy)
        device = next(policy.parameters()).device
        a, mk = adv.to(device), mask.to(device)
        anchor = new_logprobs(policy, tok, items, Tm, device).detach()
        if k == 0:
            with policy.disable_adapter():
                my_ref = new_logprobs(policy, tok, items, Tm, device).detach()
            res["diagnostic"] = {
                "midpoint_vs_cached_old": mismatch(anchor.cpu(), old, mask, items, tok),
                "reference_vs_cached_ref": mismatch(my_ref.cpu(), ref, mask, items, tok),
            }
            print("diagnostic:", {kk: {x: v for x, v in dd.items() if x != "worst_tokens"}
                                  for kk, dd in res["diagnostic"].items()}, flush=True)
        traj = [dict(step=0, **geometry(anchor, anchor, a, mk, eps))]
        opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda")
        for step in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)
            total = mk.sum()
            for i, it in enumerate(items):
                seq, attn, rm = encode(tok, it["prompt"], it["ids"], device)
                _, lp, _ = sequence_logprobs(policy, seq, attn, rm)
                T = lp.shape[1]
                loss_i, _, _ = ppo_policy_loss(lp, anchor[i:i + 1, :T], a[i:i + 1, :T], mk[i:i + 1, :T], eps)
                scaler.scale(loss_i * mk[i, :T].sum() / total).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"])))
            scaler.step(opt); scaler.update()
            g = geometry(new_logprobs(policy, tok, items, Tm, device), anchor, a, mk, eps)
            traj.append(dict(step=step, grad_norm=gn, **g))
            print(f"eps={eps:.2f} step {step}: clip {g['clip_fraction']:.4f} affected {g['affected_fraction']:.4f} "
                  f"ratio [{g['ratio_min']:.3f}, {g['ratio_max']:.3f}]", flush=True)
        res["per_epsilon"][f"{eps:.2f}"] = traj
        clear_gpu(policy, opt)
        del policy, opt
        torch.cuda.empty_cache()
        save_json(repo_path(cfg["results_dir"]) / "clip_cached_reanchored.json", res)
    print("saved", repo_path(cfg["results_dir"]) / "clip_cached_reanchored.json")


if __name__ == "__main__":
    main()
