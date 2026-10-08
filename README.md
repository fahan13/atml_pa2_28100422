# LLM Post-Training: DPO, PPO, GRPO, Safety Calibration and Verifiable vs AI Feedback

Programming Assignment 2 — EE-5102 / CS-6304, Advanced Topics in Machine Learning
Fahan Hassan Khan (28100422), LUMS

Built on the course starter repository
(https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining, release commit `1d64ac6`).
This repository contains all code, configurations, logs and per-run result files for the
five tasks of PA2:

| Task | Topic | Policy / data |
|------|-------|---------------|
| 1 | Direct Preference Optimization | Qwen2.5-1.5B-Instruct + LoRA, fixed UltraFeedback-style preference pairs |
| 2 | PPO continuation | supplied PPO midpoint policy + critic, learned reward model |
| 3 | GRPO continuation | supplied GRPO midpoint policy, learned reward model, K=8 cache |
| 4 | Safety calibration | SFT / DPO / PPO / GRPO on XSTest, fixed AI judge + manual audit |
| 5 | RLVR vs RLAIF | supplied frozen RLVR and RLAIF policies, GSM8K and SVAMP |

**Seed 6304 everywhere. All hyperparameters are the released `configs/*.yaml`, unchanged.**
Every number in the report is read from a file under `results/` (see §9).

---

## 1. Repository layout

```
common/                 released helpers (unchanged) + logprobs.py (see §6)
configs/                released configs (byte-identical to the release)
task1_dpo/              dpo.py (objective, fixed), train.py, evaluate.py, ablate_beta.py,
                        analyze_length.py, _runner.py
task2_ppo/              ppo.py (objectives, fixed), continue_train.py, evaluate.py,
                        analyze_clipping.py, ablate_kl.py, _runner.py
task3_grpo/             grpo.py (objectives, fixed), continue_train.py, evaluate.py,
                        analyze_group_size.py, compare_normalization.py
task4_safety/           generate_responses.py, judge_responses.py (released judge, unchanged),
                        make_audit_sheet.py, evaluate_safety.py
task5_feedback/         rlvr.py, rlaif.py (released verifier/judge, unchanged), evaluate_math.py,
                        score_perturbations.py, compare_feedback.py, probe_judge_raw.py
tests/                  test_objectives.py (the three defects), test_logprobs.py
scripts/                released asset scripts + run_task{1..4}.ps1, run_task5.sh, watch.py
report/                 make_figures.py and figures/ (every report figure)
results/                one folder per task: JSON/JSONL/CSV results, train logs, generations
logs/                   console logs of every run
```

Logic lives in modules; each `python -m taskN_*.<step>` is one step of the manual. Downloaded
assets (`data/`, `cached/`, `checkpoints/`, `manifests/`) and trained adapters (`outputs/`) are
gitignored, as the manual requires.

---

## 2. Environment and hardware

| | Tasks 1–4 | Task 5 |
|---|---|---|
| Machine | Windows 11 laptop, NVIDIA RTX 4060 Laptop GPU (8 GB) | Kaggle notebook, 2× Tesla T4 (15 GB) |
| Python / PyTorch | 3.12.15 / 2.14.1+cu130 | 3.13.15 / 2.11.0+cu128 |
| Transformers / TRL / PEFT / Tokenizers | 4.57.1 / 0.27.2 / 0.17.1 / 0.22.1 | same (pinned) |
| bitsandbytes | 0.50.2 | from `requirements.txt` |

The library versions that matter are pinned and identical on both machines; Kaggle's
environment is recorded in `results/task5_feedback/environment_kaggle.txt`. **No comparison
mixes machines:** every condition of Tasks 1–4 ran on the laptop, and all three Task 5 policies
ran on the same T4. Task 5 was moved to Kaggle only for time; it depends on no adapter trained
in Tasks 1–3.

```powershell
conda create -n atml_pa2 python=3.12 -y; conda activate atml_pa2
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
```

Base, reward and judge models are downloaded from Hugging Face on first use (~10 GB).

**Measured cost of the main runs (laptop):**

| Run | Wall-clock | Peak VRAM | Budget |
|---|---|---|---|
| Standard DPO, 1 epoch | 1431 s | 4.8 GB | 1430 pairs, 90 optimizer steps |
| Standard PPO continuation | 267 s | 6.35 GB | 20 updates, 4005 generated tokens |
| Standard GRPO continuation | 254 s | 6.58 GB | 20 updates × K=4, 17,112 generated tokens |

---

## 3. The three deliberate defects (Tasks 1–3)

Each objective was validated against hand-computed values before any experiment.
`python -m tests.test_objectives --tag starter` fails 19/19 checks on the released code
(`results/objective_validation_starter.json`); after the fixes, `--tag fixed` passes 19/19
(`results/objective_validation_fixed.json`). The fixes are commit `ccd82c8`.

| File | Defect | Fix |
|---|---|---|
| `task1_dpo/dpo.py` | logit = β(policy margin **+** reference margin) | β(policy margin **−** reference margin). Check: at initialization the loss is exactly ln 2 for every pair. |
| `task2_ppo/ppo.py` | clipped surrogate used `torch.maximum` | `torch.minimum`. Check: the four sign/ratio cases give the correct value and zero gradient exactly where clipping should bind. |
| `task3_grpo/grpo.py` | advantages normalized over the whole batch, `group_ids` ignored | mean/std computed per prompt group. Check: a group of identical rewards gets all-zero advantages. |

All other released objective functions (`compute_gae`, `shaped_rewards`, `value_mse_loss`,
`normalize_advantages`, `grpo_policy_loss`, `mask_truncated_sequences`) were checked against the
manual's equations and left unchanged.

---

## 4. Reproducing each task

Each PowerShell script runs its task end to end, appends to `logs/taskN.log`, and **skips any
training or evaluation whose output already exists**, so it is safe to re-run after an
interruption. Use `-NoProfile` so the child shell keeps the active conda environment.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_task1.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_task2.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_task3.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_task4.ps1   # needs Tasks 1–3 adapters
bash scripts/run_task5.sh                                                   # Kaggle, 2 GPUs
python report/make_figures.py                                               # all figures, CPU only
```

`python -m scripts.watch --task N --plot` shows a live, read-only progress table for Tasks 1–3.

### Task 1 — DPO
| Step | Command | Output |
|---|---|---|
| SFT baseline | `task1_dpo.evaluate --adapter none --name sft` | `eval_sft.json` |
| 1. Standard DPO (β=0.1, 1 epoch) | `task1_dpo.train --run-name standard`, then `task1_dpo.evaluate` | `standard_train_log.jsonl`, `eval_standard.json` |
| 2. β ∈ {0.03, 0.10, 0.30}, 600 pairs each | `task1_dpo.ablate_beta` | `beta_sweep_summary.csv` |
| 3. Length-balanced DPO + strata + word limits | `task1_dpo.analyze_length` | `length_study_summary.{json,csv}` |

### Task 2 — PPO
| Step | Command | Output |
|---|---|---|
| 1. 20-update continuation | `task2_ppo.continue_train --run-name standard` | `standard_train_log.jsonl` |
| Held-out eval (standard and midpoint) | `task2_ppo.evaluate` | `eval_standard.json`, `eval_midpoint.json` |
| 2. Clipping study (cached batch + ε forks) | `task2_ppo.analyze_clipping` | `clip_cached_analysis.json`, `clip_study_summary.csv` |
| 3. KL study (β_KL forks) | `task2_ppo.ablate_kl` | `kl_study_summary.csv` |

### Task 3 — GRPO
| Step | Command | Output |
|---|---|---|
| 1. 20-update continuation, K=4 | `task3_grpo.continue_train --run-name standard` | `standard_train_log.jsonl`, `standard_completions.jsonl` |
| Held-out eval (standard and midpoint) | `task3_grpo.evaluate` | `eval_standard.json`, `eval_midpoint.json` |
| 2. Group size K ∈ {2, 4, 8} on the cache (CPU) | `task3_grpo.analyze_group_size` | `group_size_summary.{json,csv}` |
| 3. Canonical vs Dr.-GRPO-style forks | `task3_grpo.compare_normalization` | `normalization_summary.{json,csv}` |

### Task 4 — Safety calibration
| Step | Command | Output |
|---|---|---|
| 1. Greedy responses, 4 policies × 450 prompts | `task4_safety.generate_responses` | `generated_{sft,dpo,ppo,grpo}.jsonl` |
| 2. Released AI judge | `task4_safety.judge_responses` | `judged_*.jsonl` |
| 3. Blind audit sheet (60 fixed IDs) | `task4_safety.make_audit_sheet` | `manual_audit_sheet.csv` (labels), `manual_audit_key.csv` |
| Aggregation + audit agreement | `task4_safety.evaluate_safety` | `safety_summary.json`, `safety_by_category.csv`, `audit_confusion_matrix.csv`, `audit_disagreements.csv` |

### Task 5 — RLVR vs RLAIF
| Step | Command | Output |
|---|---|---|
| 1. GSM8K, SFT/RLVR/RLAIF | `task5_feedback.evaluate_math --dataset gsm` | `math_gsm_summary.json`, `generations_gsm_*.jsonl` |
| 2. Controlled diagnostics | `task5_feedback.score_perturbations` | `diagnostic_summary.json`, `diagnostic_pairs.jsonl` |
| 3. SVAMP transfer | `task5_feedback.evaluate_math --dataset transfer` | `math_transfer_summary.json` |
| Comparison table | `task5_feedback.compare_feedback` | `feedback_comparison.{json,csv}` |

---

## 5. Evaluation protocols (fixed once, identical for every condition)

- **Task 1.** Held-out DPO loss and preference accuracy on `dpo_standard_eval` (285 pairs after
  the filter in §6) and per stratum on `dpo_length_stratified_eval` (233 pairs). Preference
  accuracy uses the manual's definition: reference-adjusted margin > 0. Generation metrics on the
  first 200 eval prompts that fit 512 prompt tokens, sampled with the config's decoding
  (T=0.7, top-p 0.9, seed 6304), 256-token cap: sampled KL to the frozen reference (base model,
  adapter disabled), reward-model score, length. Word-limit compliance on the 10 fixed prompts
  with the released helper.
- **Tasks 2–3.** The same 165 held-out RL prompts (the 35 of 200 whose rendered prompt exceeds
  `max_prompt_length`=256 are excluded rather than truncated), seed 6304, config decoding,
  768-token cap, reward model truncated at 1280 tokens. PPO and GRPO numbers are therefore directly
  comparable. Midpoint checkpoints are evaluated under the same protocol as the "before" point.
- **Task 4.** Greedy decoding, 256-token cap, fixed CSV order, the released judge prompt and parser.
- **Task 5.** The datasets' own chat messages (the training prompt format, with the
  "end with exactly `#### <number>`" instruction), greedy decoding, 512-token cap.
- **KL** everywhere is the released `sampled_kl` estimator (mean of log π − log π_ref over sampled
  tokens), accumulated over the whole evaluation set; the per-sequence mean is saved alongside.

---

## 6. Design choices and deviations

All are stated in the report; they are repeated here so the code can be read without it.

**Shared**

- *Deviation (implementation only):* `common/logprobs.py` computes summed response-token
  log-probabilities with `cross_entropy` and `logits_to_keep` instead of the released full-vocabulary
  `log_softmax`, which does not fit an 8 GB GPU during training (~0.9 GB per call at 768 tokens).
  It computes the same quantity: on two held-out pairs it matches the released helper to within
  0.007 on sums of −656.1 and −243.6 (`results/logprob_equivalence.json`, `tests/test_logprobs.py`).
  It is used for every condition of every task, so no comparison mixes implementations.
- Mixed precision: frozen base in fp16, trainable LoRA (and critic) weights in fp32, with dynamic
  loss scaling. Overflowed steps are skipped, not applied: over the standard PPO continuation this
  skipped 1 of 40 policy steps and at least 4 of 40 critic steps during the scaler's warm-up
  (`grad_norm` / `value_grad_norm` logged as NaN on those updates). Every run starts with a fresh
  scaler, so all forks are treated identically.
- Training order is an explicit seeded permutation; the exact prompt IDs used by every run are
  saved in its `*_train_summary.json`.

**Task 1**

- *Over-length pairs (TA announcement of 3 Oct):* the starter's prompt-preserving truncation is
  kept, and pairs whose prompt leaves fewer than 128 tokens of the 768-token budget are filtered.
  The same rule is applied to every DPO training and evaluation file: 70/1500 standard-train,
  83/1500 length-balanced-train, 15/300 eval and 13/246 stratified pairs are dropped (IDs saved).
  Responses longer than the remaining budget lose their tail tokens, which compresses the length
  difference DPO sees on the longest pairs.
- The standard one-epoch run (1430 pairs) and the β forks (first 600 pairs of the same order) use
  different budgets and are labelled separately everywhere.
- 47.5% of SFT generations hit the released 256-token cap, so generated-length statistics are
  censored; truncation rate is reported next to length.

**Task 2**

- Dropout is disabled in policy and critic (TRL's default), so the probability ratio moves only
  through parameter updates. Epoch 1 of each update therefore has ρ ≡ 1 and clip fraction 0 by
  construction; the epoch-2 clip fraction is reported.
- The task reward is the learned reward minus the released `missing_eos_penalty` (1.0) for responses
  that hit the 512-token cap without EOS.
- Training prompts are filtered by the same `max_prompt_length` rule (256 of 1200 excluded), and
  every fork consumes the same seeded prefix of that order from the identical supplied policy and
  critic. The ε=0.20 / β=0.10 configuration is shared by the clipping and KL studies (5 forks).
- Forks are matched on updates, prompts and seed; their generated-token counts are logged
  (`tokens_generated`) because response length is chosen by the policy.
- Cached clipping study: responses are re-tokenized (32/32 reproduce the cached token count
  exactly), advantages are recomputed with the released GAE from the cached log-probs, values and
  terminal rewards, and two extra PPO steps per ε on the fixed batch probe how ε constrains
  movement. "Affected fraction" counts tokens where the clipped branch is selected by the min (no
  gradient); "clip fraction" is the manual's out-of-band fraction.
- Explained variance is logged but is ill-conditioned with one prompt per update (near-constant
  returns); value loss is the primary critic diagnostic.

**Task 3**

- One prompt per update × K=4 completions; completions that hit the 512-token cap are masked from
  the loss (released `mask_truncated_sequences`) but kept in their group's reward statistics.
- A group is uninformative when its reward std ≤ 1e-6, the tolerance of the released advantage
  helper. The reward model truncates at 1280 tokens, as in Task 2, since `grpo.yaml` sets no length.
- Group-size study: all 192 cached generations are used for every K (24/48/96 groups of 8/4/2,
  partitioned in `generation_index` order), plus the variance of a completion's advantage over
  200 random partitions. Difficulty bins are tertiles of each prompt's mean reward over its 8
  completions. A **binarized** reward (above/below the median of all 192 rewards) is reported as an
  added, clearly labelled illustration of a verifier-like reward.
- The supplied Dr.-GRPO option changes only the length normalization (1/L_max instead of 1/T); it
  keeps the division by σ. The length-conditioned statistic is the gradient weight each completion
  carries under its normalization, computed exactly from the loss, split at the median length.

**Task 4**

- Manual audit: the 60 IDs come from the released seeded rule (30 SAFE + 30 UNSAFE). Each ID is
  joined to all four policies' responses, as the starter instructs; identical texts for the same ID
  are merged into one row (240 responses → 117 rows) and the label applies to every policy that
  produced it. The sheet hides policy names and AI labels; the mapping is in
  `manual_audit_key.csv`. The author saw the AI label of one audited response (xstest_id 0) before
  labelling; it is marked in the sheet's notes.
- *Supplementary metric:* the released judge never outputs OVER_REFUSAL or AMBIGUOUS, assigns
  confidence 1.0 to every item, and labels refusals of benign prompts as JUSTIFIED_REFUSAL. Next to
  the manual's rates, `evaluate_safety` therefore also reports class-remapped rates (refused =
  {JR, OR}, answered = {SA, UC}, with the benchmark class choosing the definition). These are
  labelled `remapped_*` and are not substitutes for the manual's definitions.

**Task 5**

- Primary metrics use only the released verifier (designated `#### <number>` field) and the released
  pairwise judge (win = 1, tie = 0.5, loss = 0 against SFT; ties reported separately).
- *Supplementary, clearly labelled:* lenient accuracy (the `####` value, else the last `\boxed{}`,
  else the last number) separates format failures from arithmetic failures; win rates are also
  reported on non-identical pairs only, because 242/300 RLVR and 256/300 RLAIF greedy GSM8K answers
  are word-for-word identical to SFT's and trivially tie.
- Each diagnostic perturbation is compared against the clean response of the same problem; the
  judge is also queried with the order reversed to measure position consistency.
- `probe_judge_raw.py` confirms the judge's TIEs are genuine outputs, not parse failures
  (40/40 raw outputs are literally "TIE").

---

## 7. Integrity notes

- No course data, cached rollout or supplied checkpoint was modified. `configs/`, `common/*.py`
  (except the added `logprobs.py`), `task5_feedback/rlvr.py`, `task5_feedback/rlaif.py`, and the
  judge loader/prompt/parser in `task4_safety/judge_responses.py` are byte-identical to the release.
- Every PPO and GRPO fork starts from the identical supplied midpoint; every DPO run starts from
  the original Qwen2.5-1.5B-Instruct with a fresh LoRA adapter.
- The manual audit was labelled blind to AI labels and policy identity, before agreement was
  computed.
- Run-to-run noise is visible in the results: three PPO ε forks that made mathematically identical
  updates (no ratio ever left ±4%, so clipping never bound) still differ by up to 0.09 in held-out
  reward through GPU non-determinism in sampling. Differences of that size are treated as noise.

---

## 8. Figures

`python report/make_figures.py` regenerates every figure in `report/figures/` from `results/`
only (no model is loaded); `--list` shows them, and a name regenerates one.

| Figure | Content |
|---|---|
| `t1_training` | DPO training loss and accuracy, all five runs |
| `t1_beta` | β sweep: held-out accuracy, KL, reward, length (+ standard run, labelled) |
| `t1_length` | per-stratum accuracy; how chosen/rejected log-probs moved vs the reference |
| `t2_trajectories` | PPO: reward, KL, policy/value loss, entropy, clip fraction, grad norm, length |
| `t2_clipping` | cached clip/affected fraction per ε; ratio range reached in the forks |
| `t2_forks` | held-out reward/KL/length of the ε and β forks vs the midpoint |
| `t3_trajectories` | GRPO: reward, group std, uninformative fraction, KL, grad norm, entropy, length, masking |
| `t3_group_size` | informative-group rate vs K by difficulty; grouping noise |
| `t3_normalization` | per-token weight vs length; gradient mass short vs long |
| `t4_judge_labels` | AI-judge label distribution per policy and class |
| `t4_audit_confusion` | manual vs AI-judge confusion matrix |
| `t5_accuracy` | exact accuracy, format compliance, lenient accuracy; GSM8K and SVAMP |
| `t5_diagnostics` | verifier vs judge preference on the controlled perturbations |

---

## 9. Where the reported numbers live

| Report element | File |
|---|---|
| Defect validation | `results/objective_validation_{starter,fixed}.json` |
| Log-prob equivalence | `results/logprob_equivalence.json` |
| DPO standard / β sweep | `results/task1_dpo/eval_{sft,standard,beta_*}.json`, `beta_sweep_summary.csv` |
| DPO length study, strata, word limits | `results/task1_dpo/length_study_summary.json`, `eval_length_balanced.json`, `pairs_*_stratified.jsonl`, `wordlimit_*.jsonl` |
| PPO trajectory, cost | `results/task2_ppo/standard_train_log.jsonl`, `standard_train_summary.json` |
| PPO clipping / KL studies | `results/task2_ppo/clip_cached_analysis.json`, `clip_study_summary.csv`, `kl_study_summary.csv`, `eval_*.json` |
| GRPO trajectory, cost | `results/task3_grpo/standard_train_log.jsonl`, `standard_train_summary.json` |
| GRPO group size / normalization | `results/task3_grpo/group_size_summary.json`, `normalization_summary.json`, `fork_*_completions.jsonl` |
| Safety rates, categories, length | `results/task4_safety/safety_summary.json`, `safety_by_category.csv` |
| Manual audit | `results/task4_safety/manual_audit_sheet.csv`, `audit_confusion_matrix.csv`, `audit_disagreements.csv` |
| RLVR vs RLAIF | `results/task5_feedback/feedback_comparison.json`, `math_*_summary.json`, `diagnostic_summary.json` |
| Qualitative examples | the `generations_*.jsonl` files of each task (every response is saved with its prompt ID) |

---

## 10. Attribution

Course-provided (used as released): the starter repository and its helpers, configs, datasets,
cached rollouts, the PPO/GRPO midpoint checkpoints, the RLVR and RLAIF policies, the exact
verifier, the pairwise AI judge and the Task 4 safety judge (course TAs, ATML Fall 2026).

Models: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct (judges) and Qwen2.5-0.5B-Instruct (critic
initialization) — Qwen Team, 2024; reward model `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`.

Methods: DPO — Rafailov et al., NeurIPS 2023; PPO — Schulman et al., 2017; GAE — Schulman et al.,
ICLR 2016; GRPO — Shao et al., *DeepSeekMath*, 2024; Dr. GRPO — Liu et al., *Understanding
R1-Zero-Like Training*, 2025; LoRA — Hu et al., ICLR 2022.

Data: UltraFeedback (Cui et al., 2023), XSTest (Röttger et al., NAACL 2024), GSM8K (Cobbe et al.,
2021), SVAMP (Patel et al., NAACL 2021).

Libraries: PyTorch, Hugging Face Transformers, PEFT, TRL, bitsandbytes, NumPy, pandas, matplotlib.

---

## 11. Use of AI assistance

An AI assistant was used for writing and debugging code, for explanation of the methods, and for
review of the experimental design. All experiments were run by the author, the manual audit labels
are the author's own, and the report was written entirely by the author.
