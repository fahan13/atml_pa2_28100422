# Runs the full Task 1 chain with the right environment, logging everything to logs/task1.log.
# Usage (from the repo root, with the conda env active):
#   powershell -ExecutionPolicy Bypass -File scripts\run_task1.ps1
# Re-running is safe: finished trainings/evaluations are skipped.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs/task1.log"
"=== Task 1 chain started $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
if (-not (Test-Path outputs/task1_dpo/standard/adapter_model.safetensors)) {
    python -m task1_dpo.train --run-name standard *>> $log
}
if (-not (Test-Path results/task1_dpo/eval_standard.json)) {
    python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard *>> $log
}
python -m task1_dpo.ablate_beta *>> $log
python -m task1_dpo.analyze_length *>> $log
"=== Task 1 chain finished $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log