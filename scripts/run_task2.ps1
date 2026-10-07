# Task 2 (PPO) chain. Usage from repo root with the conda env active:
#   powershell -ExecutionPolicy Bypass -File scripts\run_task2.ps1
# Re-running is safe: finished trainings/evaluations are skipped.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs/task2.log"
"=== Task 2 chain started $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
if (-not (Test-Path outputs/task2_ppo/standard/adapter_model.safetensors)) {
    python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard *>> $log
}
if (-not (Test-Path results/task2_ppo/eval_standard.json)) {
    python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard *>> $log
}
if (-not (Test-Path results/task2_ppo/eval_midpoint.json)) {
    python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint *>> $log
}
python -m task2_ppo.ablate_kl --config configs/ppo.yaml *>> $log
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml *>> $log
"=== Task 2 chain finished $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log