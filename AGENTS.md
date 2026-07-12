# Project Instructions

## Language And Naming

- Use Chinese for experiment planning, status updates, result interpretation, and handoff notes.
- Do not use unexplained experiment codes as the primary name. On first use, write a plain Chinese name followed by the code in parentheses, for example: `普通 DataB 续训模型（M0）`.
- Prefer a one-sentence plain Chinese explanation of what an experiment tests before listing implementation details or metrics.

## Persistent Experiment Log

The single source of truth for this project's experiment history is:

`docs/camera_conditioned_experiment_log.md`

Maintain it for all training, inference, evaluation, ablation, probe, and benchmark work in this project.

Update the log only when an experiment is actually committed to execution or when new experimental results are available. Do not update it for ordinary conversation, command explanations, path or file checks, environment setup, troubleshooting, tentative ideas, or repeated discussion of an already recorded experiment.

### When Starting Or Changing An Experiment

Before giving the final execution instructions for a new experiment, create or update its section in the experiment log. Record:

- date and status;
- a descriptive Chinese experiment name;
- the concrete question being tested;
- model lineage and starting checkpoint;
- training and evaluation datasets, including exact paths when known;
- the single changed factor and the control condition;
- important training or inference settings;
- pass/fail criteria;
- known leakage, distribution mismatch, or train-test mismatch;
- the immediate next action.

If information is unavailable, write `待补充`; do not guess.

### When Results Are Provided

When the user supplies terminal output, metrics, JSON summaries, CSV files, screenshots, or pasted evaluation results:

1. Update the existing experiment section before giving the final interpretation.
2. Add the result source path when known.
3. Preserve the core raw metrics in a compact Markdown table.
4. Explain in one to three Chinese sentences what was actually tested and what the result does and does not establish.
5. Mark the result as `通过`, `未通过`, or `结论不足`, with the reason.
6. Update the experiment index at the top of the log.
7. Append corrections with a date and reason. Do not silently replace historical results.

Do not paste large raw logs into the experiment log when a compact metric table and source path are sufficient.

## Experiment Integrity

- Explicitly distinguish a true held-out test from a diagnostic set previously seen by an inherited checkpoint.
- Explicitly flag when training receives camera context but inference does not.
- Treat camera-conditioned, shuffled-camera, null-camera, and no-camera inputs as different experimental conditions.
- Do not describe a missing-camera stress test as a camera-conditioned method result.
- Do not recommend a clean retraining run until the current low-cost gate has been evaluated, unless the user explicitly requests it.

## Protected File

- Do not modify, delete, rename, or overwrite `docs/final_experiment_plan_20260708.md` unless the user explicitly reverses this instruction.

## Server Storage Policy

- Treat `/tmp` on the training server as fast but ephemeral container storage. Files there may disappear when the container exits or is recreated.
- Treat `/input/...` as persistent NAS storage. NAS capacity is limited, so do not place large videos, frame directories, NPZ feature sets, checkpoints, or similarly large artifacts there by default.
- Disposable validation and smoke-test outputs may stay only in `/tmp`; do not ask the user to copy them to NAS or OSS unless they become reusable inputs for a later experiment.
- Store small, formal, reusable artifacts on NAS under the project directory when practical. This includes JSON/JSONL manifests, split definitions, YAML configs, CSV metrics, audit summaries, Markdown records, and compact logs.
- Produce large formal artifacts in `/tmp` for speed. Once an expensive artifact passes its audit and will be reused, explicitly remind the user to upload it to OSS before the container can be lost. Do not assume the OSS upload has happened; record its OSS location in a small NAS manifest when the user provides it.
- Before giving execution commands, classify each important output as disposable validation output, persistent small metadata, or reusable large artifact, and choose `/tmp`, NAS, or `/tmp` plus an OSS reminder accordingly.

## Server Compute Policy

- The normal full-scale server allocation is 16 GPUs with 96 GB memory per GPU and 96 physical CPU cores / 192 hardware threads.
- For formal extraction, inference, and other case-shardable workloads, use all 16 GPUs by default unless the method or reproducibility requirements make that unsafe. Small smoke tests may use one GPU.
- Judge utilization by end-to-end throughput and bottlenecks, not by trying to fill all GPU memory. Increase inference batch sizes only after a representative timing run shows a worthwhile gain without OOM or output-contract changes.
- Avoid CPU oversubscription across 16 worker processes. Budget CPU threads against 96 physical cores, measure decoding and preprocessing throughput, and do not assume that using all 192 hardware threads is faster.
