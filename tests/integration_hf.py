"""Download-free end-to-end trainer/checkpoint/evaluator integration check.
Run with the train extra installed: python tests/integration_hf.py
This random model is not a competent coding agent.
"""
import json
import argparse
import tempfile
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from toollearn.tasks import generate
from toollearn.experiment import run, oracle_policy, HFPolicy, prompt_text
from toollearn.training import train

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="artifacts/integration.json")
    args=parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="toollearn-integration-") as directory:
        root=Path(directory); base=root/"base"; base.mkdir()
        records=[]
        for task in generate("train",2):
            result=run(task,oracle_policy(task))
            records.append({"split":"train","trace":result["trace"]})
        corpus=[prompt_text(turn["messages"])+turn["completion"] for record in records for turn in record["trace"]]
        raw=Tokenizer(WordLevel(unk_token="[UNK]")); raw.pre_tokenizer=Whitespace()
        raw.train_from_iterator(corpus,WordLevelTrainer(special_tokens=["[UNK]","[EOS]","[PAD]"]))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token="[UNK]",eos_token="[EOS]",pad_token="[PAD]")
        tokenizer.save_pretrained(base)
        GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=2048,n_embd=32,n_layer=1,n_head=2,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)).save_pretrained(base)
        data=root/"train.jsonl"; data.write_text("".join(json.dumps(row)+"\n" for row in records),encoding="utf-8")
        metrics=train(str(data),str(base),str(root/"adapter"),steps=2,rank=4)
        assert (root/"adapter"/"adapter_model.safetensors").exists()
        policy=HFPolicy(str(base),str(root/"adapter"),max_new_tokens=8)
        result=run(generate("test",1)[0],policy,max_steps=2)
        assert result["calls"]<=2 and result["output_tokens"]>0
        report={"kind":"random_model_train_reload_evaluate_integration","train_steps":2,"train_loss":metrics["train_loss"],"evaluation_calls":result["calls"],"output_tokens":result["output_tokens"],"task_success":result["success"],"runtime":policy.metadata}
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report))

if __name__=="__main__": main()
