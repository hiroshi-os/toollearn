"""Reproducible pretrained baseline, oracle SFT and accepted-teacher comparison.

Run: python -m toollearn.studies --output artifacts/study
Teacher collection is real inference; an empty accepted set is a reported outcome.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from .experiment import HFPolicy, oracle_policy, run, summarize
from .tasks import generate
from .training import train


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_records(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect(tasks, policy, max_steps, teacher=None):
    results, accepted = [], []
    for task in tasks:
        result = run(task, policy if teacher else oracle_policy(task), max_steps)
        results.append(result)
        if result["success"]:
            accepted.append({"id": task["id"], "split": "train", "family": task["family"],
                             "label_source": "model_teacher" if teacher else "scripted_privileged_oracle",
                             "teacher": teacher, "trace": result["trace"]})
    return accepted, results


def reusable_teacher_data(path, tasks, expected):
    """Never silently reuse a dataset produced under a different collection config."""
    path = Path(path)
    report = json.loads(Path(str(path) + ".report.json").read_text(encoding="utf-8"))
    if report["collection_config"] != expected:
        raise ValueError("existing teacher dataset configuration differs; choose another output directory")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["sha256"]:
        raise ValueError("teacher dataset digest differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    allowed = {t["id"] for t in tasks}
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or any(row["id"] not in allowed or row["split"] != "train" or
           row["label_source"] != "model_teacher" for row in rows):
        raise ValueError("teacher dataset contains duplicate, held-out or non-teacher records")
    if len(rows) != report["accepted"]:
        raise ValueError("teacher accepted count differs")
    return rows, report


def evaluate(model, adapter, tasks, args, revision):
    policy = HFPolicy(model, adapter, args.max_new_tokens, revision, args.seed, args.chat_template)
    results = [run(task, policy, args.max_steps) for task in tasks]
    report = {"runtime": policy.metadata, "summary": summarize(results), "results": results,
              "task_ids": [t["id"] for t in tasks], "max_steps": args.max_steps,
              "max_new_tokens": args.max_new_tokens, "split": "test"}
    del policy
    release()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--teacher-revision", default="main")
    parser.add_argument("--output", default="artifacts/study")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--chat-template", action="store_true")
    args = parser.parse_args()
    if min(args.count, args.test_count, args.steps, args.rank, args.max_steps, args.max_new_tokens) < 1:
        parser.error("counts and budgets must be positive")
    # Resolve Hub references once, before any condition loads the weights.
    from huggingface_hub import HfApi
    api = HfApi()
    revision = api.model_info(args.model, revision=args.revision).sha
    teacher_revision = api.model_info(args.teacher, revision=args.teacher_revision).sha
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    tasks, heldout = generate("train", args.count, args.seed), generate("test", args.test_count, args.seed)
    config = dict(vars(args), resolved_revision=revision, resolved_teacher_revision=teacher_revision)
    write_json(root / "config.json", config)
    write_json(root / "heldout.json", heldout)
    baseline = evaluate(args.model, None, heldout, args, revision)
    write_json(root / "baseline.json", baseline)
    oracle, _ = collect(tasks, None, 8)
    write_records(root / "oracle.jsonl", oracle)
    train(root / "oracle.jsonl", args.model, str(root / "oracle-adapter"), args.steps,
          args.rank, args.seed, revision, args.chat_template)
    release()
    oracle_report = evaluate(args.model, str(root / "oracle-adapter"), heldout, args, revision)
    write_json(root / "oracle-adapted.json", oracle_report)
    data = root / "teacher.jsonl"
    collection_config = {"model": args.teacher, "revision": teacher_revision, "seed": args.seed,
                         "count": args.count, "max_steps": args.max_steps,
                         "tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                         "max_new_tokens": args.max_new_tokens, "chat_template": args.chat_template}
    if data.exists():
        accepted, teacher_report = reusable_teacher_data(data, tasks, collection_config)
    else:
        policy = HFPolicy(args.teacher, max_new_tokens=args.max_new_tokens,
                          revision=teacher_revision, seed=args.seed, chat_template=args.chat_template)
        accepted, attempted = collect(tasks, policy, args.max_steps, args.teacher)
        write_records(data, accepted)
        teacher_report = {"collection_config": collection_config, "runtime": policy.metadata,
                          "attempted": len(attempted), "accepted": len(accepted),
                          "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                          "summary": summarize(attempted), "results": attempted}
        write_json(str(data) + ".report.json", teacher_report)
        del policy
        release()
    distilled = None
    if accepted:
        train(data, args.model, str(root / "teacher-adapter"), args.steps,
              args.rank, args.seed, revision, args.chat_template)
        release()
        distilled = evaluate(args.model, str(root / "teacher-adapter"), heldout, args, revision)
        write_json(root / "teacher-adapted.json", distilled)
    write_json(root / "comparison.json", {
        "config": config, "baseline": baseline["summary"], "oracle_sft": oracle_report["summary"],
        "teacher_attempted": teacher_report["attempted"], "teacher_accepted": len(accepted),
        "teacher_sft": distilled["summary"] if distilled else None,
        "distillation_status": "trained_and_evaluated" if accepted else "skipped_zero_accepted",
        "limitations": "Synthetic held-out edit category; single seed; no general coding competence claim."})


if __name__ == "__main__":
    main()
