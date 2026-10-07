"""Task 2 Step 2: clipping study.

Part A (cached batch, no rollouts): rebuild the fixed 32-rollout batch, recompute GAE advantages
from the cached old/ref log-probs, values and terminal rewards, score the responses under the
supplied midpoint policy, and evaluate the clipped surrogate for each epsilon:
  clip_fraction     = tokens whose ratio lies outside [1-eps, 1+eps]          (manual definition)
  affected_fraction = tokens where the clipped branch is SELECTED by the min, i.e. clipping
                      actually removes the gradient: (A>0 & rho>1+eps) | (A<0 & rho<1-eps)
Because the midpoint may be very close to the policy that produced the cache, we also take
`--probe-steps` PPO steps on this fixed batch for each epsilon (fresh midpoint each time) and
re-measure, which exposes how epsilon constrains movement on identical data.

Part B: matched short continuation forks for each epsilon (beta_KL fixed at config default).
"""
from __future__ import annotations

import argparse
import csv

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.logprobs import sequence_logprobs
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from task2_ppo._runner import fork_row, run_fork
from task2_ppo.continue_train import disable_dropout, upcast_trainable
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def rebuild_batch(cfg, tok, rows):
    """Token ids for each cached response (re-tokenized; EOS appended when the rollout terminated),
    padded tensors for old/ref log-probs, values and mask, and the cached terminal reward."""
    by_id = {}
    for key in ("rl_prompt_train", "rl_prompt_eval"):
        for r in read_jsonl(cfg["paths"][key]):
            by_id.setdefault(str(r.get("prompt_id")), r)
    train_rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    items, mismatched = [], []
    for row in rows:
        prow = by_id.get(str(row.get("prompt_id"))) or train_rows[int(row["source_index"])]
        ids = tok(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos", False):
            ids = ids + [tok.eos_token_id]
        T = len(row["old_logprobs"])
        if len(ids) != T:
            mismatched.append({"prompt_id": row.get("prompt_id"), "retokenized": len(ids), "cached": T})
            continue
        items.append({"row": row, "prompt": prompt_messages(prow), "ids": ids})
    B, Tm = len(items), max(len(it["ids"]) for it in items)
    old = torch.zeros(B, Tm); ref = torch.zeros(B, Tm); val = torch.zeros(B, Tm); mask = torch.zeros(B, Tm)
    reward = torch.zeros(B)
    for i, it in enumerate(items):
        r, T = it["row"], len(it["ids"])
        old[i, :T] = r["old_logprobs"].float(); ref[i, :T] = r["ref_logprobs"].float()
        val[i, :T] = r["values"].float(); mask[i, :T] = 1.0
        reward[i] = float(r.get("effective_terminal_reward", r.get("raw_terminal_reward")))
    return items, old, ref, val, mask, reward, mismatched


def encode(tok, prompt, ids, device):
    p = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    seq = torch.tensor([p + ids], device=device)
    rmask = torch.zeros_like(seq, dtype=torch.float32)
    rmask[0, len(p):] = 1.0
    return seq, torch.ones_like(seq), rmask


def new_logprobs(policy, tok, items, Tm, device, grad=False):
    out = torch.zeros(len(items), Tm, device=device)
    for i, it in enumerate(items):
        seq, attn, rm = encode(tok, it["prompt"], it["ids"], device)
        with torch.set_grad_enabled(grad):
            _, lp, _ = sequence_logprobs(policy, seq, attn, rm)
        out[i, :lp.shape[1]] = lp[0]
    return out


def geometry(new_lp, old, adv, mask, eps):
    ratio = torch.exp(new_lp - old)
    m = mask.bool()
    loss, _, clip_frac = ppo_policy_loss(new_lp, old, adv, mask, eps)
    unclipped = -(ratio * adv * mask).sum() / mask.sum()
    affected = (((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))) & m
    lr = (new_lp - old)[m]
    return {"clipped_surrogate_loss": float(loss), "unclipped_surrogate_loss": float(unclipped),
            "clip_fraction": float(clip_frac), "affected_fraction": float(affected.sum() / m.sum()),
            "ratio_mean": float(ratio[m].mean()), "ratio_min": float(ratio[m].min()),
            "ratio_max": float(ratio[m].max()), "abs_log_ratio_mean": float(lr.abs().mean())}


def cached_study(cfg, args):
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    items, old, ref, val, mask, reward, mismatched = rebuild_batch(cfg, tok, rows)
    print(f"cached rollouts: {len(rows)}, usable: {len(items)}, re-tokenization mismatches: {len(mismatched)}")
    rewards = shaped_rewards(reward, old, ref, mask, float(cfg["kl_beta"]))
    adv, _ = compute_gae(rewards, val, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = normalize_advantages(adv, mask)

    res = {"n_cached": len(rows), "n_used": len(items), "retokenization_mismatches": mismatched,
           "n_tokens": int(mask.sum()), "kl_beta_for_shaping": float(cfg["kl_beta"]),
           "frac_tokens_positive_advantage": float((adv[mask.bool()] > 0).float().mean()),
           "probe_steps": args.probe_steps, "per_epsilon": {}}
    for eps in cfg["clip_values"]:
        eps = float(eps)
        set_seed(int(cfg["seed"]))
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
        disable_dropout(policy); upcast_trainable(policy)
        device = next(policy.parameters()).device
        o, a, mk = old.to(device), adv.to(device), mask.to(device)
        Tm = mask.shape[1]
        traj = [dict(step=0, **geometry(new_logprobs(policy, tok, items, Tm, device), o, a, mk, eps))]
        opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda")
        for step in range(1, args.probe_steps + 1):
            opt.zero_grad(set_to_none=True)
            total = mk.sum()
            for i, it in enumerate(items):                       # token-weighted mean over the batch
                seq, attn, rm = encode(tok, it["prompt"], it["ids"], device)
                _, lp, _ = sequence_logprobs(policy, seq, attn, rm)
                T = lp.shape[1]
                loss_i, _, _ = ppo_policy_loss(lp, o[i:i + 1, :T], a[i:i + 1, :T], mk[i:i + 1, :T], eps)
                scaler.scale(loss_i * mk[i, :T].sum() / total).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"])))
            scaler.step(opt); scaler.update()
            g = geometry(new_logprobs(policy, tok, items, Tm, device), o, a, mk, eps)
            traj.append(dict(step=step, grad_norm=gn, **g))
        res["per_epsilon"][f"{eps:.2f}"] = traj
        print(f"eps={eps:.2f}: " + " | ".join(
            f"step {t['step']}: clip {t['clip_fraction']:.3f} affected {t['affected_fraction']:.3f} "
            f"L_clip {t['clipped_surrogate_loss']:+.4f}" for t in traj), flush=True)
        clear_gpu(policy, opt)
        del policy, opt
        torch.cuda.empty_cache()
    save_json(repo_path(cfg["results_dir"]) / "clip_cached_analysis.json", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--probe-steps", type=int, default=2)
    ap.add_argument("--skip-forks", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required epsilon values:", cfg["clip_values"])

    if not (repo_path(cfg["results_dir"]) / "clip_cached_analysis.json").exists():
        cached_study(cfg, args)
    else:
        print("skip cached study: clip_cached_analysis.json exists")

    if args.skip_forks:
        return
    beta = float(cfg["kl_beta"])
    rows = []
    for eps in cfg["clip_values"]:
        name = run_fork(args.config, cfg, float(eps), beta)
        rows.append(fork_row(cfg, name, float(eps), beta))
    out = repo_path(cfg["results_dir"]) / "clip_study_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("saved", out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()