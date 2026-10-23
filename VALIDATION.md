# Executed validation

Date: 2026-10-23. Windows 11, Python 3.12.5. CPU-only machine.

Executed from this repository with `PYTHONPATH=src`:

- `python -m unittest discover -s tests -v`: **12 tests passed**. Includes path boundaries, exact edit matching and atomic failure, tool schema validation, bounded invalid rollouts, finish requirement, held-out family/ID/path disjointness, training-split rejection, deterministic generation and privileged oracle correctness.
- `python -m compileall -q src tests`: passed.
- `python -m toollearn.cli demo`: 16/16 synthetic test tasks completed by the privileged scripted oracle; zero invalid calls, four calls per task. This validates plumbing only. Exact run summary is `evidence/oracle-environment-check.json`.
- `python -m toollearn.cli data --count 16`: generated 16 train-split oracle trajectories, explicitly labeled `scripted_privileged_oracle`.

## Learning evidence

Executed locally with Torch 2.14.1+cpu, Transformers 4.57.6 and PEFT 0.21.2:

- `python -m toollearn.cli smoke --output artifacts/smoke`: four optimizer steps on a random local one-layer GPT-2. Both LoRA A/B tensors changed. Full output is `evidence/lora-smoke.json`. This is evidence of adapter optimization, not task accuracy.
- `python tests/integration_hf.py`: the actual completion-masked Trainer completed two updates, saved the adapter, reloaded it, and generated 16 tokens across two bounded tool calls. The random model did **not** solve the evaluation task. See `evidence/hf-integration.json`.
- GitHub CPU CI also completed successfully after initial publication. New commits may be checked separately; see the Actions page for the current status.

No pretrained-model adaptation, accepted model-teacher dataset, generalization improvement or GPU performance result is claimed. The default teacher and pretrained training commands remain executable experiments to run, not completed quality results.

## Reproduce the optional integration

```powershell
python -m pip install -e '.[train]'
python -m toollearn.cli smoke
python tests/integration_hf.py
```

The integration initializes a random local GPT-2 with a local tokenizer, trains LoRA using the actual Trainer route, reloads the resulting checkpoint and executes a bounded model rollout. It does not download weights or establish coding competence.
