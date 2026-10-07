"""Validate the three course objective implementations against hand-computed values.

Run from the repo root:
    python -m tests.test_objectives --tag starter   # before fixing (expect FAILs)
    python -m tests.test_objectives --tag fixed     # after fixing (expect all PASS)

Writes results/objective_validation_<tag>.json so the before/after evidence is saved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import ppo_policy_loss
from task3_grpo.grpo import group_relative_advantages

TOL = 1e-4
checks = []


def check(name, got, expected, note=""):
    ok = abs(float(got) - float(expected)) < TOL
    checks.append({"check": name, "got": float(got), "expected": float(expected), "pass": ok, "note": note})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {float(got):.4f}, expected {float(expected):.4f}  {note}")


# ---------------- DPO ----------------
# Test 1: at initialization policy == reference, so every logit is 0 and the loss is ln 2.
torch.manual_seed(6304)
pc, pr = torch.randn(8) * 50, torch.randn(8) * 50
loss, diag = dpo_loss(pc, pr, pc.clone(), pr.clone(), beta=0.1)
check("DPO loss at policy==ref", loss, math.log(2), "must be ln2 for every pair")
check("DPO logit mean at policy==ref", diag["logit_mean"], 0.0)

# Test 2: policy margin 2, reference margin 1, beta 0.1 -> logit = 0.1*(2-1) = 0.1.
loss, _ = dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]), beta=0.1)
check("DPO loss, margins (2,1), beta 0.1", loss, -math.log(1 / (1 + math.exp(-0.1))))

# ---------------- PPO ----------------
# Four sign/ratio cases with eps = 0.2. Objective = min(rho*A, clip(rho)*A); loss = -objective.
# 'grad_zero' = whether the clipped (flat) branch should be selected, i.e. no gradient.
ppo_cases = [
    ("A=+1, rho=1.5", 1.5, +1.0, 1.2, True),
    ("A=+1, rho=0.5", 0.5, +1.0, 0.5, False),
    ("A=-1, rho=0.5", 0.5, -1.0, -0.8, True),
    ("A=-1, rho=1.5", 1.5, -1.0, -1.5, False),
]
for name, rho, adv, expected_obj, grad_zero in ppo_cases:
    new_logp = torch.tensor([[math.log(rho)]], requires_grad=True)
    old_logp = torch.zeros(1, 1)
    loss, _, _ = ppo_policy_loss(new_logp, old_logp, torch.tensor([[adv]]), torch.ones(1, 1), eps=0.2)
    loss.backward()
    check(f"PPO objective {name}", -loss.item(), expected_obj)
    grad_is_zero = abs(new_logp.grad.item()) < 1e-8
    check(f"PPO grad-is-zero {name}", float(grad_is_zero), float(grad_zero),
          "brake on" if grad_zero else "gradient must flow")

# ---------------- GRPO ----------------
# Group 0: identical rewards -> uninformative -> all advantages must be 0.
# Group 1: rewards 0,1,0,1 -> mean 0.5, std 0.5 -> advantages -1,+1,-1,+1.
rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
adv = group_relative_advantages(rewards, group_ids)
expected = [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0]
for i, (g, e) in enumerate(zip(adv.tolist(), expected)):
    check(f"GRPO advantage[{i}] (group {int(group_ids[i])})", g, e)

# ---------------- save ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="fixed")
args = ap.parse_args()
n_pass = sum(c["pass"] for c in checks)
print(f"\n{n_pass}/{len(checks)} checks passed")
out = Path("results") / f"objective_validation_{args.tag}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"tag": args.tag, "passed": n_pass, "total": len(checks), "checks": checks}, indent=2))
print("saved", out)