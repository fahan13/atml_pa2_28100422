"""Shortlist candidate qualitative examples for the report from saved results (no GPU, no models).

    python report/shortlist_examples.py
Output: results/qualitative_shortlist.md. Candidates only: the author reads them and picks.
"""
from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

R = Path("results"); D = Path("data")
out = []


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def cut(s, n=420):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " …"


def user_text(row):
    m = row.get("messages")
    if isinstance(m, str):
        try: m = ast.literal_eval(m)
        except Exception: m = None
    if isinstance(m, list):
        return next((x.get("content", "") for x in m if x.get("role") == "user"), "")
    return row.get("prompt", row.get("question", ""))


def h(t): out.append(f"\n## {t}\n")
def p(t=""): out.append(t)


# ---------------- Task 1 ----------------
h("Task 1 (DPO)")
prompts = {r["prompt_id"]: r.get("prompt", "") for r in jl(D / "dpo_standard_eval.jsonl")}
sft = {g["prompt_id"]: g for g in jl(R / "task1_dpo/generations_sft.jsonl")}
std = {g["prompt_id"]: g for g in jl(R / "task1_dpo/generations_standard.jsonl")}
diff = [(std[k]["reward"] - sft[k]["reward"], k) for k in std if k in sft and std[k]["response"] != sft[k]["response"]]
diff.sort(reverse=True)
p(f"Generations that differ from SFT: {len(diff)}/200. Top reward gains (standard DPO vs SFT); check whether the higher-reward answer is actually better (truncated? padded? less correct?).\n")
for d, k in diff[:5]:
    p(f"**Prompt:** {cut(prompts.get(k, k), 200)}  ")
    p(f"- SFT (reward {sft[k]['reward']:.2f}, {sft[k]['length']} tok, truncated={sft[k]['truncated']}): {cut(sft[k]['response'])}")
    p(f"- DPO (reward {std[k]['reward']:.2f}, {std[k]['length']} tok, truncated={std[k]['truncated']}): {cut(std[k]['response'])}\n")
p("Largest reward *drops* (DPO worse by the RM):\n")
for d, k in diff[-2:]:
    p(f"**Prompt:** {cut(prompts.get(k, k), 200)}  ")
    p(f"- SFT (reward {sft[k]['reward']:.2f}, {sft[k]['length']} tok): {cut(sft[k]['response'])}")
    p(f"- DPO (reward {std[k]['reward']:.2f}, {std[k]['length']} tok): {cut(std[k]['response'])}\n")

p("### Word-limit prompts (words / compliant) for SFT, standard, length-balanced\n")
wl = {r["prompt_id"]: r for r in jl(D / "word_limit_prompts.jsonl")}
gens = {m: {g["prompt_id"]: g for g in jl(R / f"task1_dpo/wordlimit_{m}.jsonl")} for m in ("sft", "standard", "length_balanced")}
for pid, r in wl.items():
    cells = " | ".join(f"{m}: {int(gens[m][pid]['words'])}w {'✓' if gens[m][pid]['compliant'] else '✗'}" for m in gens)
    p(f"- `{pid}` {cut(user_text(r), 120)} → {cells}")
bad = [pid for pid in wl if not gens["standard"][pid]["compliant"]]
if bad:
    pid = bad[0]
    p(f"\nExample non-compliant standard-DPO answer (`{pid}`): {cut(gens['standard'][pid]['response'], 600)}")

# ---------------- Task 2 ----------------
h("Task 2 (PPO)")
ev = {r["prompt_id"]: user_text(r) for r in jl(D / "rl_prompt_pool_eval.jsonl")}
mid = {g["prompt_id"]: g for g in jl(R / "task2_ppo/generations_midpoint.jsonl")}
ppo = {g["prompt_id"]: g for g in jl(R / "task2_ppo/generations_standard.jsonl")}
p("### Very short responses with high reward (any PPO run): reward and quality disagree?\n")
short = []
for f in sorted((R / "task2_ppo").glob("generations_*.jsonl")):
    for g in jl(f):
        if g["length"] <= 12 and g["reward"] > 0.5:
            short.append((g["reward"], f.stem.replace("generations_", ""), g))
seen = set()
for rw, run, g in sorted(short, key=lambda x: -x[0]):
    if g["prompt_id"] in seen: continue
    seen.add(g["prompt_id"])
    p(f"- [{run}] reward {rw:.2f}, {g['length']} tok. Prompt: {cut(ev.get(g['prompt_id'], ''), 220)} → Response: \"{cut(g['response'], 160)}\"")
    if len(seen) >= 5: break
p("\n### Reward gains midpoint → standard where the response changed (reward and quality may agree)\n")
gain = sorted(((ppo[k]["reward"] - mid[k]["reward"], k) for k in ppo if k in mid and ppo[k]["response"] != mid[k]["response"]
               and not ppo[k]["truncated"]), reverse=True)
for d, k in gain[:3]:
    p(f"**Prompt:** {cut(ev.get(k, ''), 200)}  ")
    p(f"- midpoint (reward {mid[k]['reward']:.2f}, {mid[k]['length']} tok): {cut(mid[k]['response'])}")
    p(f"- standard (reward {ppo[k]['reward']:.2f}, {ppo[k]['length']} tok): {cut(ppo[k]['response'])}\n")

# ---------------- Task 3 ----------------
h("Task 3 (GRPO)")
tr = {r["prompt_id"]: user_text(r) for r in jl(D / "rl_prompt_pool_train.jsonl")}
comp = defaultdict(list)
for c in jl(R / "task3_grpo/standard_completions.jsonl"):
    comp[int(c["update"])].append(c)
p("### Updates with no learning signal (standard run)\n")
for u, cs in sorted(comp.items()):
    rs = [c["reward"] for c in cs]
    if max(rs) - min(rs) < 1e-6 or not any(c["in_loss"] for c in cs):
        why = "all truncated → masked" if not any(c["in_loss"] for c in cs) else "identical rewards (std 0)"
        p(f"- update {u} ({why}): prompt {cut(tr.get(cs[0]['prompt_id'], cs[0]['prompt_id']), 160)}; lengths {[int(c['length']) for c in cs]}; "
          f"responses: {[cut(c['response'], 40) for c in cs]}")
p("\n### Canonical vs Dr. GRPO weights on the same update (first update of each fork)\n")
for fork in ("fork_grpo", "fork_dr_grpo"):
    cs = [c for c in jl(R / f"task3_grpo/{fork}_completions.jsonl") if int(c["update"]) == 1]
    p(f"- {fork}: " + "; ".join(f"len {int(c['length'])}, A {c['advantage']:+.2f}, per-token weight {c['token_weight']:.4f}, "
                                f"in_loss {c['in_loss']}" for c in cs))

# ---------------- Task 4 ----------------
h("Task 4 (Safety): manual vs judge disagreements")
rows = list(csv.DictReader(open(R / "task4_safety/audit_disagreements.csv", encoding="utf-8-sig")))
by = defaultdict(list)
for r in rows:
    by[(r["manual"], r["judge"])].append(r)
for key, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    p(f"### manual {key[0]} vs judge {key[1]}: {len(rs)} responses")
    shown = set()
    for r in rs:
        if r["prompt"] in shown: continue
        shown.add(r["prompt"])
        p(f"- [{r['policy']}, {r['benchmark_class']}/{r['type']}] **{r['prompt']}** → {cut(r['response'], 260)}")
        if len(shown) >= 3: break

# ---------------- Task 5 ----------------
h("Task 5 (diagnostic set): judge decisions vs verifier")
diag = defaultdict(dict)
for r in jl(D / "task5_controlled_reward_diagnostics.jsonl"):
    diag[str(r["problem_id"])][r["variant_type"]] = r
for pr in jl(R / "task5_feedback/diagnostic_pairs.jsonl"):
    n_cr = globals().get("_n_cr", 0)
    interesting = pr["judge"] in ("wrong", "better") or (pr["perturbation"] == "corrupt_reasoning_correct_final" and n_cr < 2)
    if not interesting: continue
    d = diag[str(pr["problem_id"])]
    p(f"### {pr['perturbation']} (problem {pr['problem_id']}): verifier {pr['verifier']}, judge {pr['judge']}, "
      f"judge with order reversed {pr['judge_reversed']}")
    p(f"- Q: {cut(d['clean_correct']['question'], 200)} (gold {d['clean_correct']['gold_final']})")
    p(f"- clean: {cut(d['clean_correct']['response'], 300)}")
    p(f"- perturbed: {cut(d[pr['perturbation']]['response'], 400)}\n")
    if pr["perturbation"] == "corrupt_reasoning_correct_final":
        globals().setdefault("_n_cr", 0)
        globals()["_n_cr"] += 1

text = "# Qualitative example shortlist (candidates — pick and quote minimally)\n" + "\n".join(out)
(R / "qualitative_shortlist.md").write_text(text, encoding="utf-8")
print("wrote", R / "qualitative_shortlist.md", len(text), "chars")
