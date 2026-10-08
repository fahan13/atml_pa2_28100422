# Task 4 (safety) chain on the local GPU. Usage from repo root with the conda env active:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_task4.ps1
# Re-running is safe: finished generations are skipped and judging resumes where it stopped.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs/task4.log"
"=== Task 4 chain started $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log
python -m task4_safety.generate_responses --config configs/feedback.yaml *>> $log
python -m task4_safety.judge_responses --config configs/feedback.yaml *>> $log
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml *>> $log
python -m task4_safety.evaluate_safety --config configs/feedback.yaml *>> $log
"=== Task 4 chain finished $(Get-Date) ===" | Out-File -Append -Encoding utf8 $log