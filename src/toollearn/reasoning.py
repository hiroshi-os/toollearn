"""Generated final-answer evaluation; no grading or inference of hidden reasoning."""
import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from .studies import write_json, release


def generate_tasks(split="test", count=24, seed=17):
    if split not in {"dev", "test"} or not 0 < count <= 1000:
        raise ValueError("split must be dev/test and count must be in 1..1000")
    rng = random.Random(f"reasoning-v1:{seed}:{split}")
    # Disjoint numeric ranges prohibit identical dev/test instances.
    low, high = (10, 99) if split == "dev" else (100, 999)
    tasks = []
    for i in range(count):
        a, b, c = [rng.randint(low, high) for _ in range(3)]
        family = ("arithmetic", "modular", "linear")[i % 3]
        if family == "arithmetic":
            question, answer = f"Compute ({a} - {b}) * {c}.", (a - b) * c
        elif family == "modular":
            question, answer = f"Find the nonnegative remainder of {a} * {b} + {c} divided by 7.", (a * b + c) % 7
        else:
            question, answer = f"Solve for integer x: {a} * x + {b} = {a*c+b}.", c
        tasks.append({"id": f"{split}-{family}-{i}", "split": split, "family": family,
                      "question": question, "answer": str(answer), "operands": [a, b, c]})
    return tasks


def parse_final(text, require_think_end=False):
    """Strict FINAL integer, after a completed think block if one was emitted.

    A truncated thinking response cannot receive credit for an intermediate number.
    Multiple final markers, decimal approximations and trailing prose are invalid.
    """
    if (require_think_end or "<think>" in text) and "</think>" not in text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    matches = re.findall(r"(?m)^FINAL:\s*([+-]?\d+)\s*$", text)
    if len(matches) != 1 or not re.search(r"FINAL:\s*[+-]?\d+\s*\Z", text):
        return None
    return str(int(matches[0]))


def evaluate_final(policy, tasks, mode):
    results = []
    for task in tasks:
        messages = [{"role": "user", "content": task["question"] +
                     "\nEnd your response with exactly FINAL: <integer> on its own line."}]
        started = time.perf_counter()
        raw, metrics = policy(messages, 0)
        predicted = parse_final(raw, require_think_end=mode == "thinking")
        results.append({"id": task["id"], "family": task["family"], "expected": task["answer"],
                        "predicted": predicted, "correct": predicted == task["answer"],
                        "raw": raw, "latency_seconds": time.perf_counter() - started, **metrics})
    return {"kind": "generated_math_final_answer_exact_match", "mode": mode,
            "runtime": getattr(policy, "metadata", {}), "tasks": len(tasks),
            "accuracy": sum(row["correct"] for row in results) / len(results),
            "invalid_final_answers": sum(row["predicted"] is None for row in results),
            "results": results,
            "limitations": "Final-answer behavior only; not a measurement of internal reasoning or a standard benchmark."}


def main():
    from .experiment import HFPolicy
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--mode", choices=["default", "thinking", "nonthinking", "both"], default="both")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="artifacts/reasoning")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("token budget must be positive")
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    tasks = generate_tasks(args.split, args.count, args.seed)
    root = Path(args.output)
    write_json(root / "tasks.json", tasks)
    config = dict(vars(args), resolved_revision=revision,
                  tasks_sha256=hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest())
    write_json(root / "config.json", config)
    modes = ["thinking", "nonthinking"] if args.mode == "both" else [args.mode]
    for mode in modes:
        thinking = None if mode == "default" else mode == "thinking"
        policy = HFPolicy(args.model, max_new_tokens=args.max_new_tokens, revision=revision,
                          seed=args.seed, chat_template=True, thinking=thinking, final_answer=True)
        report = evaluate_final(policy, tasks, mode)
        report["config"] = config
        report["budget_hit_count"] = sum(r["output_tokens"] >= args.max_new_tokens for r in report["results"])
        write_json(root / f"{mode}.json", report)
        del policy
        release()


if __name__ == "__main__":
    main()
