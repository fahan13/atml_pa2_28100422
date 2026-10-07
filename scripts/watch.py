"""Live, read-only progress dashboard for PA2 runs.

It only READS log/result files that the training scripts already write; it never imports a
model, never touches the GPU memory of the run, and never writes into results/. Safe to start,
stop (Ctrl+C) and restart at any time while training runs in another terminal.

    python -m scripts.watch --task 1            # live dashboard, refresh every 5 s
    python -m scripts.watch --task 1 --plot     # also save logs/task1_curves.png each refresh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G, Y, R, D, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
SPARK = "▁▂▃▄▅▆▇█"


# ----------------------------------------------------------------------------- run specs
def task1_runs():
    """(label, train_log, train_summary, eval_json, expected_optimizer_steps)."""
    cache = ROOT / "logs" / ".watch_task1_totals.json"
    if cache.exists():
        totals = json.loads(cache.read_text())
    else:
        print("first start: counting training pairs (one-off, ~10 s)...")
        from common.data import load_yaml, read_jsonl
        from common.models import load_tokenizer
        from task1_dpo.train import filter_rows
        cfg = load_yaml("configs/dpo.yaml")
        tok = load_tokenizer(cfg["base_model"])
        L, per_step = int(cfg["max_sequence_length"]), int(cfg["batch_size"]) * int(cfg["grad_accum_steps"])
        n_std = len(filter_rows(read_jsonl(cfg["paths"]["dpo_standard_train"]), tok, L)[0])
        n_len = len(filter_rows(read_jsonl(cfg["paths"]["dpo_length_train"]), tok, L)[0])
        n_short = min(int(cfg["short_ablation_examples"]), n_std)
        totals = {"standard": math.ceil(n_std / per_step), "short": math.ceil(n_short / per_step),
                  "length_balanced": math.ceil(n_len / per_step),
                  "betas": [f"beta_{float(b):.2f}" for b in cfg["betas"]]}
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(totals))
    rd = ROOT / "results" / "task1_dpo"
    runs = [("standard", totals["standard"])] + [(b, totals["short"]) for b in totals["betas"]] \
        + [("length_balanced", totals["length_balanced"])]
    cols = {"step": "step", "m1": ("loss", "{:.4f}"), "m2": ("acc", "{:.2f}", 10), "trend": "loss"}
    return [(name, rd / f"{name}_train_log.jsonl", rd / f"{name}_train_summary.json",
             rd / f"eval_{name}.json", total, cols) for name, total in runs], ROOT / "logs" / "task1.log"


def task2_runs():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "ppo.yaml").read_text())
    rd = ROOT / "results" / "task2_ppo"
    eps0, b0 = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    forks = []
    for b in cfg["kl_values"]:
        forks.append((eps0, float(b)))
    for e in cfg["clip_values"]:
        if (float(e), b0) not in forks:
            forks.append((float(e), b0))
    names = [("standard", int(cfg["updates"]))] + [
        (f"fork_eps{e:.2f}_kl{b:.2f}", int(cfg["fork_updates"])) for e, b in forks]
    cols = {"step": "update", "m1": ("reward", "{:+.3f}"), "m2": ("kl", "{:+.4f}", 1), "trend": "reward"}
    runs = [(n, rd / f"{n}_train_log.jsonl", rd / f"{n}_train_summary.json", rd / f"eval_{n}.json", t, cols)
            for n, t in names]
    runs.append(("midpoint (eval)", rd / "_none_", rd / "_none_", rd / "eval_midpoint.json", 0, cols))
    return runs, ROOT / "logs" / "task2.log"


TASKS = {1: ("Task 1 - DPO", task1_runs), 2: ("Task 2 - PPO", task2_runs)}


# ----------------------------------------------------------------------------- helpers
def read_log(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass                      # line being written right now
    return rows


def spark(values, width=24):
    if len(values) < 2:
        return ""
    v = values[-width:]
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1.0
    return "".join(SPARK[int((x - lo) / span * (len(SPARK) - 1))] for x in v)


def bar(frac, width=20):
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def fmt_t(s):
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def gpu_line():
    if not shutil.which("nvidia-smi"):
        return "GPU: nvidia-smi not found"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        util, used, total, temp = [x.strip() for x in out.split(",")]
        colour = R if int(temp) >= 87 else (Y if int(temp) >= 80 else G)
        return f"GPU {util:>3}% util | VRAM {int(used) / 1024:.1f}/{int(total) / 1024:.1f} GB | {colour}{temp}°C{X}"
    except Exception as exc:  # never crash the dashboard
        return f"GPU: unavailable ({exc.__class__.__name__})"


def last_error(main_log):
    if not main_log.exists():
        return None
    text = main_log.read_text(encoding="utf-8", errors="ignore")
    if "Traceback (most recent call last)" not in text:
        return None
    tail = text[text.rfind("Traceback (most recent call last)"):].strip().splitlines()
    return [l for l in tail if l.strip()][-3:]


# ----------------------------------------------------------------------------- render
def render(title, runs, main_log):
    lines = [f"{B}{title}{X}   {D}{time.strftime('%H:%M:%S')}  (Ctrl+C to quit; training is unaffected){X}",
             gpu_line(), ""]
    c0 = runs[0][5]
    h1, h2, ht = c0["m1"][0], c0["m2"][0] + ("~" if c0["m2"][2] > 1 else ""), c0["trend"] + " trend"
    lines.append(f"{'run':<22}{'train':<9}{'progress':<30}{h1:>9}{h2:>9}  {ht:<26}{'ETA':>8}  eval")
    now = time.time()
    for name, log, summ, ev, total, cols in runs:
        if total == 0:              # evaluation-only row
            evs = f"{G}done{X}" if ev.exists() else f"{D}-{X}"
            lines.append(f"{name:<22}{D}{'(eval only)':<9}{X}{'':<30}{'':>9}{'':>9}  {'':<26}{'':>8}  {evs}")
            continue
        rows = read_log(log)
        step = rows[-1][cols["step"]] if rows else 0
        if summ.exists():
            status = f"{G}done{X}     "
        elif rows and now - log.stat().st_mtime < 180:
            status = f"{Y}running{X}  "
        elif rows:
            status = f"{R}stalled?{X} "
        else:
            status = f"{D}pending{X}  "
        frac = min(step / total, 1.0) if total else 0
        prog = f"{bar(frac)} {step:>3}/{total}"
        k1, f1 = cols["m1"]
        k2, f2, win = cols["m2"]
        loss = f1.format(rows[-1][k1]) if rows else "-"
        w = [r[k2] for r in rows[-win:]]
        acc = f2.format(sum(w) / len(w)) if w else "-"
        if rows and not summ.exists():
            per_step = rows[-1]["elapsed_s"] / max(step, 1)
            left = (total - step) * per_step
            eta = fmt_t(left)
        else:
            eta = "-" if not summ.exists() else ""
        evs = f"{G}done{X}" if ev.exists() else f"{D}-{X}"
        lines.append(f"{name:<22}{status}{prog:<30}{loss:>9}{acc:>9}  {spark([r[cols['trend']] for r in rows]):<26}{eta:>8}  {evs}")
    lines.append("")
    if c0["m2"][2] > 1:
        lines.append(f"{D}{h2} = mean over the last {c0['m2'][2]} steps (noisy by design).{X}")
    err = last_error(main_log)
    if err:
        lines.append(f"{R}{B}A traceback is in {main_log.name}:{X}")
        lines += [f"{R}  {l}{X}" for l in err]
    return "\n".join(lines)


def save_plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, log, summ, ev, total, cols in runs:
        rows = read_log(log) if total else []
        if rows:
            s = [r[cols["step"]] for r in rows]
            axes[0].plot(s, [r[cols["trend"]] for r in rows], label=name)
            axes[1].plot(s, [r[cols["m2"][0]] if cols["m2"][0] != "acc" else r["grad_norm"] for r in rows], label=name)
    c0 = runs[0][5]
    if c0["trend"] == "loss":
        axes[0].axhline(math.log(2), ls=":", c="grey", lw=1)
        axes[0].set(title="training DPO loss (dotted = ln 2)", xlabel="optimizer step")
        axes[1].set(title="gradient norm (pre-clip)", xlabel="optimizer step")
    else:
        axes[0].set(title="rollout reward (learned RM)", xlabel="update")
        axes[1].set(title="sampled KL to reference (token mean)", xlabel="update")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=1, choices=sorted(TASKS))
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    title, spec = TASKS[args.task]
    runs, main_log = spec()
    os.system("")                     # enables ANSI colours in Windows consoles
    try:
        while True:
            frame = render(title, runs, main_log)
            os.system("cls" if os.name == "nt" else "clear")
            print(frame, flush=True)
            if args.plot:
                save_plot(runs, ROOT / "logs" / f"task{args.task}_curves.png")
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nwatcher stopped (training keeps running).")


if __name__ == "__main__":
    main()