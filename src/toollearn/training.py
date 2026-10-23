"""Real causal LM LoRA adaptation, with prompt tokens excluded from loss."""
import json
import hashlib
import platform
import importlib.metadata
from pathlib import Path
from .experiment import prompt_text

def examples_from_records(records):
    examples=[]
    for record in records:
        if record.get("split") != "train":
            raise ValueError("training accepts only train-split records")
        for turn in record["trace"]:
            examples.append((prompt_text(turn["messages"]), turn["completion"]))
    return examples

def train(data_path, model_name, output, steps=40, rank=8, seed=17):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    records=[json.loads(line) for line in Path(data_path).read_text(encoding="utf-8").splitlines() if line]
    examples=examples_from_records(records)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model=get_peft_model(AutoModelForCausalLM.from_pretrained(model_name),
        LoraConfig(r=rank, lora_alpha=rank*2, target_modules="all-linear", task_type="CAUSAL_LM"))
    rows=[]
    for prompt, completion in examples:
        prefix=tokenizer.encode(prompt, add_special_tokens=False)
        tail=tokenizer.encode(completion+tokenizer.eos_token, add_special_tokens=False)
        ids=prefix+tail
        if len(ids)>2048:
            raise ValueError("example exceeds 2048 tokens; shorten traces explicitly")
        rows.append({"input_ids":ids,"attention_mask":[1]*len(ids),"labels":[-100]*len(prefix)+tail})
    if not rows:
        raise ValueError("no training examples")
    def collate(batch):
        width=max(len(row["input_ids"]) for row in batch)
        return {k:torch.tensor([row[k]+[(-100 if k=="labels" else tokenizer.pad_token_id if k=="input_ids" else 0)]*(width-len(row[k])) for row in batch]) for k in batch[0]}
    args=TrainingArguments(output_dir=output, max_steps=steps, per_device_train_batch_size=1,
        learning_rate=2e-4, logging_steps=1, save_strategy="no", report_to=[], seed=seed,
        use_cpu=not torch.cuda.is_available())
    trainer=Trainer(model=model,args=args,train_dataset=rows,data_collator=collate)
    result=trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    Path(output,"training_metrics.json").write_text(json.dumps(result.metrics,indent=2),encoding="utf-8")
    manifest={"model":model_name,"model_revision":getattr(model.config,"_commit_hash",None),
        "data_sha256":hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "label_sources":sorted({r.get("label_source","unspecified") for r in records}),
        "seed":seed,"steps":steps,"rank":rank,"learning_rate":2e-4,"examples":len(rows),
        "python":platform.python_version(),"platform":platform.platform(),"cuda_available":torch.cuda.is_available(),
        "versions":{name:importlib.metadata.version(name) for name in ["torch","transformers","peft","accelerate"]},
        "metrics":result.metrics}
    Path(output,"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return result.metrics

def smoke(output, steps=4):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(17)
    model=get_peft_model(GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,n_embd=32,n_layer=1,n_head=2)),
        LoraConfig(r=4,lora_alpha=8,target_modules=["c_attn"],task_type="CAUSAL_LM"))
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.01)
    ids=torch.randint(0,32,(2,16)); losses=[]
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(); loss=model(input_ids=ids,labels=ids).loss
        loss.backward(); optimizer.step(); losses.append(loss.item())
    changed=[n for n,p in model.named_parameters() if n in before and not torch.equal(before[n],p.detach())]
    if not changed:
        raise RuntimeError("adapter weights did not update")
    Path(output).mkdir(parents=True,exist_ok=True)
    model.save_pretrained(output)
    report={"kind":"random_tiny_model_adapter_smoke_only","steps":steps,"losses":losses,"changed_adapter_tensors":changed,"task_accuracy_measured":False}
    Path(output,"smoke.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
