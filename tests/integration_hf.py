"""Download-free end-to-end trainer/checkpoint/evaluator integration check.
Run with the train extra installed: python tests/integration_hf.py
This random model is not a competent coding agent.
"""
import json
import argparse
import tempfile
import sys
from types import SimpleNamespace
from unittest.mock import patch
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
from toollearn.reasoning import generate_tasks, evaluate_final
from toollearn.studies import main as study_main
from toollearn.reasoning import main as reasoning_main

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="artifacts/integration.json")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
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
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
        tokenizer.save_pretrained(base)
        GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=2048,n_embd=32,n_layer=1,n_head=2,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)).save_pretrained(base)
        data=root/"train.jsonl"; data.write_text("".join(json.dumps(row)+"\n" for row in records),encoding="utf-8")
        baseline=HFPolicy(str(base),max_new_tokens=8,seed=17)
        baseline_result=run(generate("test",1)[0],baseline,max_steps=2)
        del baseline
        metrics=train(str(data),str(base),str(root/"adapter"),steps=2,rank=4)
        assert (root/"adapter"/"adapter_model.safetensors").exists()
        policy=HFPolicy(str(base),str(root/"adapter"),max_new_tokens=8)
        result=run(generate("test",1)[0],policy,max_steps=2)
        assert result["calls"]<=2 and result["output_tokens"]>0
        chat_metrics=train(str(data),str(base),str(root/"chat-adapter"),steps=2,rank=4,chat_template=True)
        chat_policy=HFPolicy(str(base),str(root/"chat-adapter"),max_new_tokens=8,chat_template=True,final_answer=True)
        final_report=evaluate_final(chat_policy,generate_tasks("test",3),"default")
        assert final_report["tasks"]==3 and all(r["output_tokens"]>0 for r in final_report["results"])
        # Exercise orchestration offline with only Hub reference resolution mocked.
        # Actual tokenizer/model generation, teacher filtering and LoRA training run.
        study=root/"study"
        with patch("huggingface_hub.HfApi.model_info",return_value=SimpleNamespace(sha=None)):
            with patch.object(sys,"argv",["study","--model",str(base),"--teacher",str(base),
                    "--output",str(study),"--count","2","--test-count","1","--steps","2",
                    "--max-steps","1","--max-new-tokens","8"]):
                study_main()
            with patch.object(sys,"argv",["reasoning","--model",str(base),"--mode","default",
                    "--output",str(root/"reasoning"),"--count","3","--max-new-tokens","8"]):
                reasoning_main()
        comparison=json.loads((study/"comparison.json").read_text())
        assert comparison["distillation_status"]=="skipped_zero_accepted"
        assert comparison["teacher_sft"] is None and not (study/"teacher-adapter").exists()
        report={"kind":"random_model_train_reload_evaluate_integration","train_steps":2,"train_loss":metrics["train_loss"],"chat_train_loss":chat_metrics["train_loss"],"baseline_success":baseline_result["success"],"evaluation_calls":result["calls"],"output_tokens":result["output_tokens"],"task_success":result["success"],"final_answer_tasks":final_report["tasks"],"final_answer_accuracy":final_report["accuracy"],"study_zero_accept_status":comparison["distillation_status"],"offline_hub_resolution_mocked":True,"runtime":policy.metadata}
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report))

if __name__=="__main__": main()
