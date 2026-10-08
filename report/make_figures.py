"""Regenerate every report figure from the saved result files (CPU only, no models).

    python report/make_figures.py            # all figures
    python report/make_figures.py t1_beta    # one figure by name
    python report/make_figures.py --list

Each figure is one function below; each reads only files under results/ and writes
report/figures/<name>.png. Nothing here re-runs an experiment.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "figures"

# Categorical slots in fixed order (validated reference palette, light mode).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
GOOD, TIE, BAD = "#2a78d6", "#c9c8c3", "#e34948"   # better / tie / wrong preference

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------- io helpers
def jl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", (OUT / f"{name}.png").relative_to(ROOT))


def panel_label(ax, s):
    ax.set_title(s, loc="left", color=INK)


# ============================================================================ Task 1
def t1_training():
    """DPO training loss and smoothed training preference accuracy vs pairs seen."""
    d = R / "task1_dpo"
    runs = [("standard", "standard (1 epoch, 1430 pairs)", C[0]), ("beta_0.03", "β=0.03 fork (600)", C[1]),
            ("beta_0.10", "β=0.10 fork (600)", C[2]), ("beta_0.30", "β=0.30 fork (600)", C[3]),
            ("length_balanced", "length-balanced (1 epoch)", C[6])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for name, lab, col in runs:
        log = jl(d / f"{name}_train_log.jsonl")
        x = [r["pairs_seen"] for r in log]
        k = 5
        loss = np.convolve([r["loss"] for r in log], np.ones(k) / k, mode="valid")
        axes[0].plot(x[k - 1:], loss, color=col, label=lab, lw=1.3)
        acc = np.convolve([r["acc"] for r in log], np.ones(k) / k, mode="valid")
        axes[1].plot(x[k - 1:], acc, color=col, lw=1.3)
    axes[0].axhline(math.log(2), color=GREY, ls=":", lw=1)
    axes[0].text(axes[0].get_xlim()[1], math.log(2), " ln 2", va="center", color=MUTED, fontsize=7)
    axes[1].axhline(0.5, color=GREY, ls=":", lw=1)
    panel_label(axes[0], "(a) training DPO loss (5-step mean)")
    panel_label(axes[1], "(b) training preference accuracy (5-step mean)")
    for ax in axes:
        ax.set_xlabel("preference pairs seen")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, "t1_training")


def t1_beta():
    """β sweep (matched 600-pair budget) + standard run shown separately (different budget)."""
    d = R / "task1_dpo"
    betas = [0.03, 0.10, 0.30]
    ev = [js(d / f"eval_beta_{b:.2f}.json") for b in betas]
    std, sft = js(d / "eval_standard.json"), js(d / "eval_sft.json")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw={"wspace": 0.45})
    specs = [("(a) held-out pref. acc.", lambda e: e["heldout"]["preference_accuracy"],
              lambda e: se(e["heldout"]["preference_accuracy"], e["heldout"]["n_pairs"])),
             ("(b) KL to ref. (per seq.)", lambda e: e["generation"]["kl_sequence_mean"], None),
             ("(c) RM reward", lambda e: e["generation"]["reward_mean"],
              lambda e: e["generation"]["reward_std"] / math.sqrt(e["generation"]["n_prompts"])),
             ("(d) length (tokens)", lambda e: e["generation"]["length"]["mean"],
              lambda e: e["generation"]["length"]["std"] / math.sqrt(e["generation"]["length"]["n"]))]
    xs = np.arange(len(betas))
    for ax, (title, f, err) in zip(axes, specs):
        y = [f(e) for e in ev]
        yerr = [err(e) for e in ev] if err else None
        ax.errorbar(xs, y, yerr=yerr, color=C[0], marker="o", ms=4, capsize=2, lw=1.3, label="β fork (600 pairs)")
        ax.errorbar([3.2], [f(std)], yerr=[err(std)] if err else None, color=C[1], marker="D", ms=4, capsize=2,
                    ls="none", label="standard β=0.1 (1430 pairs)")
        if not title.startswith("(a)"):
            ax.axhline(f(sft), color=GREY, ls=":", lw=1, label="SFT")
        ax.set_xticks(list(xs) + [3.2], ["0.03", "0.10", "0.30", "std"])
        ax.set_xlim(-0.5, 3.7)
        panel_label(ax, title)
    axes[0].axhline(0.5, color=GREY, ls=":", lw=1)
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               title="x-axis: β of the matched 600-pair forks; std = standard one-epoch run (different budget)",
               title_fontsize=6.5)
    save(fig, "t1_beta")


def t1_length():
    """Per-stratum held-out accuracy + how chosen/rejected log-probs moved vs the reference."""
    d = R / "task1_dpo"
    strata = [("preferred_longer", "chosen\nlonger"), ("length_matched", "matched"), ("rejected_longer", "rejected\nlonger")]
    models = [("standard", "standard DPO", C[0]), ("length_balanced", "length-balanced DPO", C[6])]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), gridspec_kw={"width_ratios": [1.1, 1, 1], "wspace": 0.42})
    w = 0.38
    for i, (m, lab, col) in enumerate(models):
        st = js(d / f"eval_{m}.json")["stratified"]["by_stratum"]
        y = [st[s]["preference_accuracy"] for s, _ in strata]
        e = [se(st[s]["preference_accuracy"], st[s]["n"]) for s, _ in strata]
        axes[0].bar(np.arange(3) + (i - 0.5) * w, y, w * 0.92, yerr=e, color=col, label=lab, capsize=2,
                    error_kw={"lw": 0.8, "ecolor": INK})
    axes[0].axhline(0.5, color=GREY, ls=":", lw=1)
    axes[0].set_xticks(range(3), [s[1] for s in strata])
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "(a) pref. accuracy by length stratum")
    for ax, (m, lab, _) in zip(axes[1:], models):
        rows = jl(d / f"pairs_{m}_stratified.jsonl")
        dc = [np.mean([r["policy_chosen_logp"] - r["ref_chosen_logp"] for r in rows if r["stratum"] == s]) for s, _ in strata]
        dr = [np.mean([r["policy_rejected_logp"] - r["ref_rejected_logp"] for r in rows if r["stratum"] == s]) for s, _ in strata]
        ax.bar(np.arange(3) - w / 2, dc, w * 0.92, color=C[2], label="chosen")
        ax.bar(np.arange(3) + w / 2, dr, w * 0.92, color=C[7], label="rejected")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(3), [s[1] for s in strata])
        panel_label(ax, f"({'b' if ax is axes[1] else 'c'}) {lab.split()[0]}: Δ log-prob vs ref.")
    axes[1].set_ylabel("mean log π_θ − log π_ref (summed)")
    axes[1].legend(loc="lower left")
    save(fig, "t1_length")


# ============================================================================ Task 2
def _grid(log, xkey, panels, name, title_prefix):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.7), sharex=True, gridspec_kw={"wspace": 0.5, "hspace": 0.45})
    x = [r[xkey] for r in log]
    for ax, (key, title, col) in zip(axes.flat, panels):
        y = [r.get(key, float("nan")) for r in log]
        y = [float("nan") if v is None else v for v in y]
        ax.plot(x, y, color=col, marker="o", ms=2.5, lw=1.2)
        panel_label(ax, f"({'abcdefgh'[list(axes.flat).index(ax)]}) {title}")
    for ax in axes[1]:
        ax.set_xlabel("update")
    save(fig, name)


def t2_trajectories():
    log = jl(R / "task2_ppo" / "standard_train_log.jsonl")
    _grid(log, "update", [
        ("reward", "rollout reward", C[0]), ("kl", "KL to ref. (token)", C[1]),
        ("policy_loss", "policy loss", C[2]), ("value_loss", "value loss", C[3]),
        ("entropy", "entropy (−log π)", C[6]), ("clip_fraction_last_epoch", "clip frac. (epoch 2)", C[7]),
        ("grad_norm", "grad norm", C[4]), ("response_length", "response length", C[5])],
        "t2_trajectories", "PPO")


def t2_clipping():
    """Cached-batch clip geometry per ε, and the ratio range actually reached in the forks."""
    c = js(R / "task2_ppo" / "clip_cached_analysis.json")
    eps = sorted(c["per_epsilon"], key=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    x = np.arange(len(eps))
    w = 0.38
    cf = [c["per_epsilon"][e][0]["clip_fraction"] for e in eps]
    af = [c["per_epsilon"][e][0]["affected_fraction"] for e in eps]
    axes[0].bar(x - w / 2, cf, w * 0.92, color=C[0], label="outside [1−ε, 1+ε]")
    axes[0].bar(x + w / 2, af, w * 0.92, color=C[1], label="clipped branch selected (no gradient)")
    for i, (a, b) in enumerate(zip(cf, af)):
        axes[0].text(i - w / 2, a, f"{a:.1%}", ha="center", va="bottom", fontsize=6.5, color=INK)
        axes[0].text(i + w / 2, b, f"{b:.1%}", ha="center", va="bottom", fontsize=6.5, color=INK)
    axes[0].set_xticks(x, [f"ε={float(e):.2f}" for e in eps])
    axes[0].set_ylabel(f"fraction of {c['n_tokens']} cached tokens")
    axes[0].set_ylim(0, max(cf) * 1.45)
    axes[0].legend(loc="upper right")
    panel_label(axes[0], "(a) cached batch: clipping geometry")
    for i, e in enumerate(eps):
        log = jl(R / "task2_ppo" / f"fork_eps{float(e):.2f}_kl0.10_train_log.jsonl")
        u = [r["update"] for r in log]
        axes[1].plot(u, [r["max_ratio"] - 1 for r in log], color=C[i], marker="o", ms=2.5, lw=1.1, label=f"ε={float(e):.2f} fork")
        axes[1].plot(u, [r["min_ratio"] - 1 for r in log], color=C[i], marker="o", ms=2.5, lw=1.1, ls="--")
    axes[1].axhspan(-0.05, 0.05, color="#eeeeea", zorder=0)
    axes[1].text(1, 0.047, "±5% band (smallest ε)", fontsize=6.5, color=MUTED, va="top")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("max (solid) / min (dashed) ρ − 1")
    axes[1].legend(loc="lower right")
    panel_label(axes[1], "(b) training forks: ratio range reached")
    save(fig, "t2_clipping")


def t2_forks():
    """Held-out effect of ε and β_KL forks vs the shared midpoint, with run-to-run noise visible."""
    d = R / "task2_ppo"
    mid = js(d / "eval_midpoint.json")
    groups = [("clip ε (β=0.1)", [(f"ε={e}", f"fork_eps{e}_kl0.10") for e in ("0.05", "0.20", "0.50")]),
              ("KL β (ε=0.2)", [(f"β={b}", f"fork_eps0.20_kl{b}") for b in ("0.00", "0.10", "0.20")])]
    specs = [("held-out RM reward", lambda e: e["reward_mean"], lambda e: e["reward_std"] / math.sqrt(e["n_prompts"])),
             ("KL to ref. (token, ×10⁻⁵)", lambda e: e["kl_token_mean"] * 1e5, None),
             ("response length", lambda e: e["length"]["mean"], lambda e: e["length"]["std"] / math.sqrt(e["length"]["n"]))]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw={"wspace": 0.38})
    for pi, (ax, (title, f, err)) in enumerate(zip(axes, specs)):
        xt, xl, pos = [], [], 0
        for gi, (gname, items) in enumerate(groups):
            for lab, name in items:
                e = js(d / f"eval_{name}.json")
                ax.errorbar([pos], [f(e)], yerr=[err(e)] if err else None, color=C[gi], marker="o", ms=4, capsize=2,
                            label=gname if (pi == 0 and lab == items[0][0]) else None)
                xt.append(pos); xl.append(lab); pos += 1
            pos += 0.6
        ax.axhline(f(mid), color=GREY, ls=":", lw=1, label="supplied midpoint" if pi == 0 else None)
        ax.set_xticks(xt, xl, rotation=35)
        panel_label(ax, f"({'abc'[pi]}) {title}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08),
               title="8-update forks from the identical midpoint; bars = ±1 SE over 165 held-out prompts", title_fontsize=6.5)
    save(fig, "t2_forks")


# ============================================================================ Task 3
def t3_trajectories():
    log = jl(R / "task3_grpo" / "standard_train_log.jsonl")
    _grid(log, "update", [
        ("reward", "mean group reward", C[0]), ("reward_std_within_group", "in-group reward std", C[1]),
        ("uninformative_group_fraction", "uninformative group", C[7]), ("kl", "KL to ref. (token)", C[2]),
        ("grad_norm", "grad norm", C[4]), ("entropy", "entropy (−log π)", C[6]),
        ("response_length", "completion length", C[5]), ("masked_completions", "masked (of 4)", C[3])],
        "t3_trajectories", "GRPO")


def t3_group_size():
    rows = js(R / "task3_grpo" / "group_size_summary.json")["results"]
    bins = [("all", "all prompts", INK), ("low_reward", "low-reward (hard)", C[7]),
            ("mid_reward", "mid", C[0]), ("high_reward", "high-reward (easy)", C[2])]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3), gridspec_kw={"wspace": 0.5})
    for ax, rt, title in [(axes[0], "continuous_RM", "(a) informative, RM reward"),
                          (axes[1], "binarized", "(b) informative, binarized")]:
        for b, lab, col in bins:
            rr = sorted([r for r in rows if r["reward_type"] == rt and r["difficulty_bin"] == b], key=lambda r: r["K"])
            ax.plot([r["K"] for r in rr], [r["informative_rate"] for r in rr], color=col, marker="o", ms=3.5,
                    lw=1.8 if b == "all" else 1.1, ls="-" if b == "all" else "--", label=lab)
        ax.set_xticks([2, 4, 8])
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("group size K")
        panel_label(ax, title)
    axes[0].set_ylabel("groups with std > 1e-6")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               title="equal budget: the same 192 cached generations regrouped for every K; bins = prompt mean-reward tertiles",
               title_fontsize=6.5)
    for rt, col, lab in [("continuous_RM", C[0], "RM reward"), ("binarized", C[1], "binarized")]:
        rr = sorted([r for r in rows if r["reward_type"] == rt and r["difficulty_bin"] == "all"], key=lambda r: r["K"])
        axes[2].plot([r["K"] for r in rr], [r["advantage_resample_var"] for r in rr], color=col, marker="o", ms=3.5, label=lab)
    axes[2].set_xticks([2, 4, 8])
    axes[2].set_xlabel("group size K")
    axes[2].set_ylabel("advantage variance\nacross random groupings")
    axes[2].legend()
    panel_label(axes[2], "(c) grouping noise")
    save(fig, "t3_group_size")


def t3_normalization():
    d = R / "task3_grpo"
    forks = [("fork_grpo", "canonical GRPO (1/T)", C[0]), ("fork_dr_grpo", "Dr. GRPO-style (1/L_max)", C[1])]
    summ = {r["fork"]: r for r in js(d / "normalization_summary.json")["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.6, 1]})
    for name, lab, col in forks:
        cs = [c for c in jl(d / f"{name}_completions.jsonl") if c["in_loss"] and c["token_weight"] > 0]
        axes[0].scatter([c["length"] for c in cs], [c["token_weight"] for c in cs], s=9, color=col, alpha=0.75,
                        edgecolors="white", linewidths=0.4, label=lab)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("completion length T (tokens)")
    axes[0].set_ylabel("per-token gradient weight |A|/denominator")
    axes[0].legend(loc="lower left")
    panel_label(axes[0], "(a) per-token weight vs completion length")
    x = np.arange(2)
    short = [summ[n]["short_share_of_gradient_mass"] for n, _, _ in forks]
    long_ = [summ[n]["long_share_of_gradient_mass"] for n, _, _ in forks]
    axes[1].bar(x, short, 0.6, color=C[2], label=f"T ≤ {summ['fork_grpo']['median_len_split']:.0f}")
    axes[1].bar(x, long_, 0.6, bottom=short, color=C[6], label="longer")
    for i in range(2):
        axes[1].text(i, short[i] / 2, f"{short[i]:.0%}", ha="center", va="center", color="white", fontsize=7)
        axes[1].text(i, short[i] + long_[i] / 2, f"{long_[i]:.0%}", ha="center", va="center", color="white", fontsize=7)
    axes[1].set_xticks(x, ["canonical", "Dr. GRPO"])
    axes[1].set_ylabel("share of total gradient mass")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    panel_label(axes[1], "(b) gradient mass by length")
    save(fig, "t3_normalization")


# ============================================================================ Task 4
T4_POL = ["sft", "dpo", "ppo", "grpo"]
LABS = ["SAFE_ANSWER", "OVER_REFUSAL", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "AMBIGUOUS"]
LAB_COL = {"SAFE_ANSWER": C[2], "OVER_REFUSAL": C[3], "JUSTIFIED_REFUSAL": C[0], "UNSAFE_COMPLIANCE": C[7], "AMBIGUOUS": GREY}
SHORT = {"SA": "SAFE_ANSWER", "JR": "JUSTIFIED_REFUSAL", "UC": "UNSAFE_COMPLIANCE", "OR": "OVER_REFUSAL", "AM": "AMBIGUOUS"}


def t4_judge_labels():
    d = R / "task4_safety"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, cls in zip(axes, ["SAFE", "UNSAFE"]):
        bottoms = np.zeros(len(T4_POL))
        fr = {}
        for p in T4_POL:
            g = {r["xstest_id"]: r["benchmark_class"] for r in jl(d / f"generated_{p}.jsonl")}
            labs = [j["judge_label"] for j in jl(d / f"judged_{p}.jsonl") if g[j["xstest_id"]] == cls]
            cnt = Counter(labs)
            fr[p] = {k: cnt.get(k, 0) / len(labs) for k in LABS}
        for lab in LABS:
            v = np.array([fr[p][lab] for p in T4_POL])
            if v.sum() == 0:
                continue
            ax.bar(range(len(T4_POL)), v, 0.62, bottom=bottoms, color=LAB_COL[lab], label=lab.replace("_", " ").lower(),
                   edgecolor="white", linewidth=1)
            for i, (b, h) in enumerate(zip(bottoms, v)):
                if h > 0.06:
                    ax.text(i, b + h / 2, f"{h:.0%}", ha="center", va="center", color="white", fontsize=6.5)
            bottoms += v
        ax.set_xticks(range(len(T4_POL)), [p.upper() for p in T4_POL])
        panel_label(ax, f"AI-judge labels on {cls} prompts (n={sum(1 for _ in jl(d / 'generated_sft.jsonl') if _['benchmark_class'] == cls)})")
    axes[0].set_ylabel("fraction of responses")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    save(fig, "t4_judge_labels")


def t4_audit_confusion():
    d = R / "task4_safety"
    sheet, key = d / "manual_audit_sheet.csv", d / "manual_audit_key.csv"
    if not sheet.exists():
        print("skip t4_audit_confusion: no audit sheet"); return
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    pol = {r["audit_row"]: r["policies"].split(";") for r in csv.DictReader(open(key, encoding="utf-8-sig"))}
    judge = {p: {j["xstest_id"]: j["judge_label"] for j in jl(d / f"judged_{p}.jsonl")} for p in T4_POL}
    pairs = []
    for r in rows:
        m = SHORT.get(r["manual_label"].strip().upper(), r["manual_label"].strip().upper())
        if m not in LABS:
            continue
        for p in pol[r["audit_row"]]:
            pairs.append((m, judge[p][int(r["xstest_id"])]))
    if not pairs:
        print("skip t4_audit_confusion: audit sheet has no labels yet"); return
    M = np.zeros((5, 5), int)
    for m, j in pairs:
        M[LABS.index(m), LABS.index(j)] += 1
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.imshow(M, cmap="Blues", vmin=0)
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > M.max() * 0.55 else INK)
    ticks = ["SA", "OR", "JR", "UC", "AM"]
    ax.set_xticks(range(5), ticks); ax.set_yticks(range(5), ticks)
    ax.set_xlabel("AI judge"); ax.set_ylabel("manual (ours)")
    agree = np.trace(M) / M.sum()
    panel_label(ax, f"manual vs judge, {M.sum()} responses (agree {agree:.0%})")
    save(fig, "t4_audit_confusion")


# ============================================================================ Task 5
def t5_accuracy():
    d = R / "task5_feedback"
    comp = {r["policy"]: r for r in js(d / "feedback_comparison.json")["policies"]}
    pols = ["sft", "rlvr", "rlaif"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, (ds, title) in zip(axes, [("gsm", "GSM8K (in-domain, n=300)"), ("svamp", "SVAMP transfer (n=100)")]):
        n = 300 if ds == "gsm" else 100
        metrics = [(f"{ds}_acc", "exact (verifier)", C[0]), (f"{ds}_format", "has `####` line", C[3]),
                   (f"{ds}_lenient_acc", "lenient final answer*", C[2])]
        w = 0.26
        for k, (key, lab, col) in enumerate(metrics):
            y = [comp[p][key] for p in pols]
            ax.bar(np.arange(3) + (k - 1) * w, y, w * 0.92, color=col, label=lab,
                   yerr=[se(v, n) for v in y], capsize=1.5, error_kw={"lw": 0.7, "ecolor": INK})
        ax.set_xticks(range(3), [p.upper() for p in pols])
        ax.set_ylim(0, 1)
        panel_label(ax, f"({'ab'[list(axes).index(ax)]}) {title}")
    axes[0].set_ylabel("fraction of problems")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               title="*supplementary: `####` value, else last \\boxed{}, else last number; bars = ±1 SE", title_fontsize=6.5)
    save(fig, "t5_accuracy")


def t5_diagnostics():
    d = R / "task5_feedback"
    bp = js(d / "diagnostic_summary.json")["by_perturbation"]
    order = [("corrupt_reasoning_correct_final", "corrupt\nreasoning"),
             ("good_reasoning_wrong_final", "wrong\nfinal"),
             ("gold_distractor_wrong_final", "gold\ndistractor"),
             ("persuasive_filler_correct", "persuasive\nfiller")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for ax, mech, title in [(axes[0], "verifier", "(a) exact verifier (RLVR)"), (axes[1], "judge", "(b) pairwise AI judge (RLAIF)")]:
        for i, (k, lab) in enumerate(order):
            r = bp[k][mech]
            b = 0
            for val, col, nm in [(r["better_rate"], GOOD, "prefers clean"), (r["tie_rate"], TIE, "tie"),
                                 (r["wrong_rate"], BAD, "prefers perturbed")]:
                ax.bar(i, val, 0.62, bottom=b, color=col, edgecolor="white", linewidth=1, label=nm if i == 0 else None)
                if val > 0.07:
                    ax.text(i, b + val / 2, f"{val:.0%}", ha="center", va="center", fontsize=6.5,
                            color="white" if col != TIE else INK)
                b += val
        ax.set_xticks(range(4), [o[1] for o in order])
        panel_label(ax, title)
    axes[0].set_ylabel("fraction of 20 pairs (clean vs perturbed)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, "t5_diagnostics")


FIGS = {f.__name__: f for f in [t1_training, t1_beta, t1_length, t2_trajectories, t2_clipping, t2_forks,
                                t3_trajectories, t3_group_size, t3_normalization, t4_judge_labels,
                                t4_audit_confusion, t5_accuracy, t5_diagnostics]}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list" in args:
        for k, f in FIGS.items():
            print(f"{k:20s} {(f.__doc__ or '').strip().splitlines()[0] if f.__doc__ else ''}")
        sys.exit()
    for name in (args or FIGS):
        try:
            FIGS[name]()
        except FileNotFoundError as e:
            print(f"skip {name}: missing {Path(e.filename).relative_to(ROOT) if e.filename else e}")
