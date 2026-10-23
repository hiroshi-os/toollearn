# toollearn

A small, inspectable research harness for adapting language models to coding tools. It includes a virtual editing environment, deterministic data generation, real Hugging Face teacher collection, completion-only LoRA training, and baseline/adapter evaluation.

**Evidence status:** the standard-library demo tests a privileged scripted oracle, not a learned model. The tiny random-model smoke proves adapter optimization only. No pretrained task improvement or teacher-distillation result is claimed without running and reporting those experiments.

## Run without downloads

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m toollearn.cli demo
python -m toollearn.cli data --count 32
```

`demo` uses an explicitly labeled scripted oracle with access to generated solutions. Its success rate validates the environment and evaluator; it is **not** evidence of agent intelligence. The oracle reports zero token counts with `tokens_measured: false`.

## Real training and evaluation

```powershell
python -m pip install -e '.[train]'
python -m toollearn.cli smoke --output artifacts/smoke
python -m toollearn.cli data --count 32 --output artifacts/oracle-train.jsonl
python -m toollearn.cli evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --split test --output artifacts/baseline.json
python -m toollearn.cli train --model HuggingFaceTB/SmolLM2-135M-Instruct --data artifacts/oracle-train.jsonl --steps 40 --rank 8 --output checkpoints/oracle-lora
python -m toollearn.cli evaluate --model HuggingFaceTB/SmolLM2-135M-Instruct --adapter checkpoints/oracle-lora --split test --output artifacts/adapted.json
```

The pretrained commands download weights and require enough CPU RAM or a GPU. The tiny-model smoke needs Torch, Transformers and PEFT but **no download**. It constructs a random one-layer GPT-2 and checks that trainable adapter tensors actually change. It does not evaluate coding ability.

The model protocol is deliberately the same plain role-delimited text during teacher collection, training, and evaluation. It does not use model-specific chat templates. That is reproducible but can disadvantage instruction models; report this limitation with results. Generation is greedy, bounded to 160 tokens per call and eight calls per task. Malformed responses are counted as invalid calls and returned to the model as errors.

## Actual teacher distillation

```powershell
python -m toollearn.cli distill --teacher HuggingFaceTB/SmolLM2-1.7B-Instruct --count 128 --output artifacts/teacher-train.jsonl
python -m toollearn.cli train --model HuggingFaceTB/SmolLM2-135M-Instruct --data artifacts/teacher-train.jsonl --output checkpoints/distilled-lora
```

`distill` runs the teacher model in the environment and keeps only successful complete trajectories. The adjacent report records attempted and accepted counts. If the teacher solves nothing, the data file is empty and training rejects it. Synthetic oracle imitation and model-teacher distillation are separate label sources. Successful filtering introduces selection bias; disclose acceptance rate.

## What is measured

Each evaluation report records exact task IDs, traces, final exact-match success plus explicit finish, invalid calls, total calls, input/output token counts, and full-episode wall time. Model loading is outside episode latency; on CPU the timer covers tokenization, inference and environment calls. Input tokens include repeated conversation context. Output tokens include all generated tokens, even if protocol delimiters truncate the text passed to the parser. This is an end-to-end CPU harness, not a GPU-serving benchmark.

Train uses rename and constant-literal edits, dev uses arithmetic-operator repair, test uses return-value repair. Splits have disjoint task IDs, file paths and **edit categories**, checked by tests. All categories share the same explicit replace instruction, so this is only a narrow compositional generalization test; it does not establish real repository generalization. The model only receives instructions and tool observations, never expected outputs or oracle metadata. The evaluator checks every file, including absence of unintended edits.

## Hypothesis and experiment status

Hypothesis: completion-only LoRA training on valid multi-turn tool traces improves held-out edit-category completion compared with the same model and inference budget before adaptation. A teacher-trained adapter may require fewer labels than oracle imitation, but that is an open experimental question. Neither hypothesis is established by the random-model smoke or oracle checks. Pretrained baseline, adaptation comparison and teacher-distillation experiments have not been executed in the checked-in evidence.

Training writes a manifest with input SHA-256, label sources, resolved model revision where available, seed, rank, learning rate, platform, package versions and actual metrics. Evaluation records resolved revision, local adapter SHA-256, seed, task split and call/token budgets. Remote model names can change; use a pinned local model snapshot for strict reproduction. A null revision means a local model rather than a verified Hub commit.

## Experiment plan

1. Fix seed 17 and dataset sizes; record package versions, model revisions and machine details.
2. Tune only on dev; run baseline and adapter with identical budgets on test once choices are fixed.
3. Compare oracle SFT, teacher distillation, adapter rank 4/8/16, and dataset sizes 32/128/512.
4. Repeat training with at least three seeds before claiming an effect. Pass `--seed` to data, training and evaluation commands for seed sweeps; use identical evaluation seeds across conditions.
5. Report success and failure examples, teacher acceptance, latency/tokens, and uncertainty. Do not present synthetic task accuracy as SWE-bench performance.

There is no host-shell tool: all reads, searches and edits operate on an in-memory dictionary. Absolute paths, traversal and Windows-drive syntax are rejected. This environment is suitable for generated string-edit tasks, not execution of untrusted repository code.

## References

- [PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

GitHub Actions includes a CPU learning integration job that uploads generated adapter-smoke and train/reload/evaluate reports. A workflow definition alone is not execution evidence; inspect the actual run and artifacts.

MIT licensed. See `VALIDATION.md` for executed checks and the boundary between implementation and experimental evidence.
