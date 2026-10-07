from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.logprobs import sequence_logprobs
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss

# Pairs whose prompt leaves fewer than this many tokens for the response (inside the 768-token
# budget) are filtered out. Applied identically to every DPO training and evaluation file.
MIN_RESPONSE_ROOM = 128


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_rows(rows, tokenizer, max_length):
    """Drop pairs whose prompt alone exceeds max_length - MIN_RESPONSE_ROOM tokens."""
    kept, dropped = [], []
    for r in rows:
        n = len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(r), tokenize=True, add_generation_prompt=True))
        (kept if n <= max_length - MIN_RESPONSE_ROOM else dropped).append(r)
    return kept, [str(r.get("prompt_id")) for r in dropped]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_rows(read_jsonl(path), tokenizer, int(cfg["max_sequence_length"]))
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    # Explicit seeded permutation so the exact training order can be saved.
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    order = torch.randperm(len(rows), generator=g).tolist()
    rows = [rows[i] for i in order]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    # Mixed precision as in HF Trainer fp16: frozen base stays fp16, trainable LoRA weights in fp32.
    for p in trainable_parameters(model):
        p.data = p.data.float()

    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped": dropped,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, loader, opt, beta = bundle["model"], bundle["loader"], bundle["optimizer"], bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"{run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = next(model.parameters()).device
    params = trainable_parameters(model)
    accum = int(cfg["grad_accum_steps"])
    n_micro = len(loader)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    print(f"[{run_name}] beta={beta} pairs={len(bundle['rows'])} dropped={len(bundle['dropped'])} "
          f"micro-batches={n_micro} optimizer steps={-(-n_micro // accum)}")

    step, window = 0, defaultdict(list)
    opt.zero_grad(set_to_none=True)
    for i, (ch, rj) in enumerate(loader):
        ch, rj = to_device(ch, device), to_device(rj, device)
        with torch.no_grad(), reference_mode(model):
            ref_c, _, _ = sequence_logprobs(model, **ch)
            ref_r, _, _ = sequence_logprobs(model, **rj)
        pol_c, _, _ = sequence_logprobs(model, **ch)
        pol_r, _, _ = sequence_logprobs(model, **rj)
        loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)

        window_start = (i // accum) * accum
        group = min(accum, n_micro - window_start)       # last window may be shorter
        scaler.scale(loss / group).backward()

        window["loss"].append(loss.item())
        window["acc"].append(diag["preference_accuracy"].item())
        window["logit"].append(diag["logit_mean"].item())
        window["chosen_reward"].append((beta * (pol_c - ref_c)).mean().item())
        window["rejected_reward"].append((beta * (pol_r - ref_r)).mean().item())
        if i == 0:
            print(f"[{run_name}] first micro-batch loss {loss.item():.4f} (≈0.693 expected at init)")

        if (i + 1) % accum == 0 or i + 1 == n_micro:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {k: sum(v) / len(v) for k, v in window.items()}
            rec.update({
                "step": step,
                "pairs_seen": min((i + 1) * int(cfg["batch_size"]), len(bundle["rows"])),
                "grad_norm": grad_norm,
                "grad_scale": scaler.get_scale(),
                "elapsed_s": round(elapsed(), 1),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            })
            append_jsonl(log_path, rec)
            print(f"[{run_name}] step {step:3d} loss {rec['loss']:.4f} acc {rec['acc']:.2f} "
                  f"gnorm {grad_norm:.3f} {rec['elapsed_s']:.0f}s vram {rec['peak_vram_gb']}GB")
            window = defaultdict(list)

    model.save_pretrained(str(output))
    save_json(results_dir / f"{run_name}_train_summary.json", {
        "run_name": run_name,
        "dataset": bundle["dataset_path"],
        "beta": beta,
        "n_pairs": len(bundle["rows"]),
        "optimizer_steps": step,
        "filter_rule": f"drop prompt tokens > max_sequence_length - {MIN_RESPONSE_ROOM}",
        "dropped_prompt_ids": bundle["dropped"],
        "training_order_prompt_ids": [str(r.get("prompt_id")) for r in bundle["rows"]],
        "wall_clock_s": round(elapsed(), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapter": str(output),
        "config": {k: cfg[k] for k in ["learning_rate", "batch_size", "grad_accum_steps", "epochs",
                                       "max_sequence_length", "max_grad_norm", "seed", "lora", "dtype"]},
    })
    print(f"[{run_name}] saved adapter to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()