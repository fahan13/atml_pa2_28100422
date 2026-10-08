#!/usr/bin/env bash
# Task 5 (RLVR vs RLAIF) exactly as run for the report: a Kaggle notebook with 2x Tesla T4,
# Internet on. Every step only uses the supplied frozen RLVR/RLAIF checkpoints, so it does not
# depend on Tasks 1-4. From the repo root:
#   bash scripts/run_task5.sh
# Generations and judge calls are cached under results/task5_feedback/, so a re-run resumes.
set -e
mkdir -p logs results/task5_feedback
python -m scripts.check_environment | tee results/task5_feedback/environment_kaggle.txt

# Step 1 (GSM8K, in-domain) and Step 3 (SVAMP transfer) in parallel, one per GPU.
CUDA_VISIBLE_DEVICES=0 python -m task5_feedback.evaluate_math --dataset gsm      > logs/task5_gsm.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m task5_feedback.evaluate_math --dataset transfer > logs/task5_transfer.log 2>&1 &
wait

# Step 2: controlled reward diagnostics (released verifier and released pairwise judge).
CUDA_VISIBLE_DEVICES=0 python -m task5_feedback.score_perturbations > logs/task5_diag.log 2>&1
# Supplementary: raw judge outputs on the 40 detectable-perturbation pairs (are TIEs real?).
CUDA_VISIBLE_DEVICES=0 python -m task5_feedback.probe_judge_raw

# Final comparison table (CPU only; was run on the laptop after copying the results back).
python -m task5_feedback.compare_feedback
