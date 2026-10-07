"""Shared fork orchestration for Task 2: every fork starts from the same supplied midpoint
(policy + value) and consumes the same seeded prompt prefix. Re-running skips finished work."""
from __future__ import annotations

import subprocess
import sys

from common.data import repo_path
from common.logging_utils import load_json


def run(args):
    cmd = [sys.executable, "-m", *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fork_name(eps, beta):
    return f"fork_eps{float(eps):.2f}_kl{float(beta):.2f}"


def run_fork(config, cfg, eps, beta):
    name = fork_name(eps, beta)
    out = f"outputs/task2_ppo/{name}"
    if (repo_path(out) / "adapter_model.safetensors").exists():
        print(f"skip training {name}")
    else:
        run(["task2_ppo.continue_train", "--config", config, "--run-name", name, "--output", out,
             "--updates", str(cfg["fork_updates"]), "--clip-epsilon", str(eps), "--kl-beta", str(beta)])
    if (repo_path(cfg["results_dir"]) / f"eval_{name}.json").exists():
        print(f"skip eval {name}")
    else:
        run(["task2_ppo.evaluate", "--config", config, "--adapter", out, "--name", name])
    return name


def fork_row(cfg, name, eps, beta):
    rdir = repo_path(cfg["results_dir"])
    e = load_json(rdir / f"eval_{name}.json")
    log = [__import__("json").loads(l) for l in (rdir / f"{name}_train_log.jsonl").read_text().splitlines() if l.strip()]
    s = load_json(rdir / f"{name}_train_summary.json")
    return {
        "fork": name, "clip_epsilon": eps, "kl_beta": beta, "updates": len(log),
        "tokens_generated": s["tokens_generated"],
        # held-out (frozen evaluation protocol)
        "heldout_reward": e["reward_mean"], "heldout_kl_token": e["kl_token_mean"],
        "heldout_kl_seq": e["kl_sequence_mean"], "heldout_entropy": e["entropy_token_mean"],
        "heldout_len_mean": e["length"]["mean"], "heldout_len_std": e["length"]["std"],
        "heldout_truncated": e["truncated_rate"],
        # training-time diagnostics / stability statistics
        "train_reward_mean": sum(r["reward"] for r in log) / len(log),
        "train_kl_last": log[-1]["kl"],
        "clip_fraction_mean_last_epoch": sum(r["clip_fraction_last_epoch"] for r in log) / len(log),
        "approx_kl_old_new_mean": sum(r["approx_kl_old_new"] for r in log) / len(log),
        "approx_kl_old_new_max": max(r["approx_kl_old_new"] for r in log),
        "max_ratio_max": max(r["max_ratio_max"] for r in log),
        "grad_norm_max": max(r["grad_norm"] for r in log),
        "value_loss_mean": sum(r["value_loss"] for r in log) / len(log),
    }