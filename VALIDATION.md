# Executed validation

Date: 2026-10-23. Windows 11, Python 3.12.5. CPU-only machine.

Executed from this repository with `PYTHONPATH=src`:

- `python -m unittest discover -s tests -v`: **12 tests passed**. Includes path boundaries, exact edit matching and atomic failure, tool schema validation, bounded invalid rollouts, finish requirement, held-out family/ID/path disjointness, training-split rejection, deterministic generation and privileged oracle correctness.
- `python -m compileall -q src tests`: passed.
- `python -m toollearn.cli demo`: 16/16 synthetic test tasks completed by the privileged scripted oracle; zero invalid calls, four calls per task. This validates plumbing only. Exact run summary is `evidence/oracle-environment-check.json`.
- `python -m toollearn.cli data --count 16`: generated 16 train-split oracle trajectories, explicitly labeled `scripted_privileged_oracle`.

## Learning evidence

HF/PEFT adapter smoke and the end-to-end train/reload/evaluate integration are implemented but dependency installation is still in progress. This document will be updated with actual results when executable. No pretrained-model adaptation, teacher-generated dataset, generalization improvement or GPU performance result is claimed.

## Reproduce the optional integration

```powershell
python -m pip install -e '.[train]'
python -m toollearn.cli smoke
python tests/integration_hf.py
```

The integration initializes a random local GPT-2 with a local tokenizer, trains LoRA using the actual Trainer route, reloads the resulting checkpoint and executes a bounded model rollout. It does not download weights or establish coding competence.
