import json
import time
import importlib.metadata
import hashlib
from pathlib import Path
from .env import Workspace, ToolError
from .tasks import oracle_actions

SYSTEM = """Edit a virtual workspace. Reply with exactly one JSON object, no markdown.
Schema: {"tool": "read|search|edit|finish", "arguments": {...}}.
read: path; search: query; edit: path, old, new (old must match once); finish: no arguments.
Paths are workspace-relative. Use tools to inspect and edit; call finish when done."""

def prompt_text(messages):
    return "\n".join(m["role"].upper()+": "+m["content"] for m in messages)+"\nASSISTANT: "

def run(task, policy, max_steps=8):
    env = Workspace(task["files"])
    messages = [{"role":"system", "content":SYSTEM}, {"role":"user", "content":task["instruction"]}]
    trace, invalid, output_tokens, input_tokens, finished = [], 0, 0, 0, False
    start = time.perf_counter()
    for step in range(max_steps):
        raw, metrics = policy(messages, step)
        output_tokens += metrics.get("output_tokens",0)
        input_tokens += metrics.get("input_tokens",0)
        before = list(messages)
        messages.append({"role":"assistant", "content":raw})
        try:
            action = json.loads(raw)
            result = env.call(action)
            finished = action["tool"] == "finish"
        except (ValueError, TypeError, KeyError) as exc:
            invalid += 1
            result = "ERROR: "+str(exc)
        trace.append({"messages":before, "completion":raw, "result":result})
        messages.append({"role":"tool", "content":result})
        if finished:
            break
    return {"id":task["id"], "family":task["family"], "success":env.files == task["expected"] and finished,
        "finished":finished, "invalid_calls":invalid, "calls":len(trace), "input_tokens":input_tokens,
        "output_tokens":output_tokens, "latency_seconds":time.perf_counter()-start, "trace":trace}

def oracle_policy(task):
    actions = oracle_actions(task)
    return lambda messages, step: (json.dumps(actions[min(step,len(actions)-1)]), {})


def decode_generation(tokenizer, generated, final_answer=False):
    if not final_answer:
        return tokenizer.decode(generated, skip_special_tokens=True).split("\nTOOL:")[0].split("\nUSER:")[0].strip()
    # Some tokenizers classify thinking markers as special. Preserve these while
    # removing transport/end tokens, so parsing cannot credit unfinished thinking.
    text = tokenizer.decode(generated, skip_special_tokens=False)
    for token in tokenizer.all_special_tokens:
        if token not in {"<think>", "</think>"}:
            text = text.replace(token, "")
    return text.strip()

class HFPolicy:
    def __init__(self, model_name, adapter=None, max_new_tokens=160, revision=None,
                 seed=17, chat_template=False, thinking=None, final_answer=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        from .runtime import load_model, seed_all
        seed_all(seed)
        self.chat_template, self.thinking = chat_template, thinking
        self.final_answer = final_answer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = load_model(model_name, revision)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.metadata = {"model": model_name, "adapter": adapter,
            "model_revision": getattr(self.model.config, "_commit_hash", None),
            "tokenizer_revision": self.tokenizer.init_kwargs.get("_commit_hash"),
            "requested_revision": revision,
            "seed": seed, "chat_template": chat_template, "thinking": thinking, "final_answer": final_answer,
            "adapter_sha256": hashlib.sha256(Path(adapter, "adapter_model.safetensors").read_bytes()).hexdigest() if adapter and Path(adapter, "adapter_model.safetensors").exists() else None,
            "device": str(next(self.model.parameters()).device),
            "versions": {name:importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]}}

    def __call__(self, messages, step):
        from .runtime import render
        text = render(self.tokenizer, messages, self.chat_template, self.thinking)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        generated = result[0, inputs["input_ids"].shape[1]:]
        raw = decode_generation(self.tokenizer, generated, self.final_answer)
        return raw, {"input_tokens":inputs["input_ids"].shape[1], "output_tokens":len(generated)}

def summarize(results):
    n=len(results)
    return {"tasks":n, "success_rate":sum(r["success"] for r in results)/n,
        "invalid_call_rate":sum(r["invalid_calls"] for r in results)/max(1,sum(r["calls"] for r in results)),
        "mean_calls":sum(r["calls"] for r in results)/n,
        "mean_latency_seconds":sum(r["latency_seconds"] for r in results)/n,
        "input_tokens":sum(r["input_tokens"] for r in results),
        "output_tokens":sum(r["output_tokens"] for r in results)}
