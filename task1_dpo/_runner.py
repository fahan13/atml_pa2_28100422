"""Small helpers so each train/eval runs in its own process (frees GPU memory between runs)."""
from __future__ import annotations

import subprocess
import sys

from common.data import repo_path


def run(args: list[str]):
    cmd = [sys.executable, "-m", *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def train_if_missing(config, run_name, output, extra=()):
    if (repo_path(output) / "adapter_model.safetensors").exists():
        print(f"skip training {run_name}: adapter exists at {output}")
        return
    run(["task1_dpo.train", "--config", config, "--run-name", run_name, "--output", output, *extra])


def eval_if_missing(config, name, adapter, extra=()):
    if (repo_path("results/task1_dpo") / f"eval_{name}.json").exists():
        print(f"skip eval {name}: results exist")
        return
    run(["task1_dpo.evaluate", "--config", config, "--name", name, "--adapter", adapter, *extra])