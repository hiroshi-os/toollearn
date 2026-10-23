"""Disjoint templates AND instances; test family never appears in train/dev."""
import random

FAMILIES = {"train": ("rename", "literal"), "dev": ("operator",), "test": ("return",)}

def generate(split, count=32, seed=17):
    if split not in FAMILIES or count < 1:
        raise ValueError("valid split and positive count required")
    rng = random.Random(f"{seed}:{split}")
    tasks = []
    for i in range(count):
        family = FAMILIES[split][i % len(FAMILIES[split])]
        a, b = rng.randrange(10, 99), rng.randrange(100, 999)
        path = f"{split}/module_{i}.py"
        if family == "rename":
            old, new = f"value_{a}", f"result_{b}"
            source = f"{old} = 7\n"
        elif family == "literal":
            old, new = str(a), str(b)
            source = f"TIMEOUT = {a}\n"
        elif family == "operator":
            old, new = "a - b", "a + b"
            source = f"def combine_{i}(a, b):\n    return a - b\n"
        else:
            old, new = "return None", f"return {b}"
            source = f"def answer_{i}():\n    return None\n"
        tasks.append({"id": f"{split}-{family}-{i}", "family": family, "split": split,
            "instruction": f"In {path}, replace exactly {old!r} with {new!r}. Preserve all other text.",
            "files": {path: source}, "expected": {path: source.replace(old,new)},
            "oracle_edit": {"path": path, "old": old, "new": new}})
    return tasks

def oracle_actions(task):
    """Privileged synthetic labeler, never a trained/model baseline."""
    edit = task["oracle_edit"]
    return [{"tool":"read", "arguments":{"path":edit["path"]}},
            {"tool":"search", "arguments":{"query":edit["old"]}},
            {"tool":"edit", "arguments":edit}, {"tool":"finish", "arguments":{}}]
