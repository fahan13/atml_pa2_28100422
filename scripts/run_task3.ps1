# Task 3 (GRPO) chain. Usage from repo root with the conda env active:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_task3.ps1
# Re-running is safe: finished trainings/evaluations are skipped.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs/task3.log"
"=== Task 3 chain started $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
if (-not (Test-Path outputs/task3_grpo/standard/adapter_model.safetensors)) {
    python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard *>> $log
}
if (-not (Test-Path results/task3_grpo/eval_standard.json)) {
    python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard *>> $log
}
if (-not (Test-Path results/task3_grpo/eval_midpoint.json)) {
    python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint *>> $log
}
python -m task3_grpo.compare_normalization --config configs/grpo.yaml *>> $log
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml *>> $log
"=== Task 3 chain finished $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log