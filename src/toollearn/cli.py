import argparse
import json
import platform
from pathlib import Path
from .tasks import generate
from .experiment import run, oracle_policy, HFPolicy, summarize

def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")

def main():
    parser=argparse.ArgumentParser(description="Safe tool-use research harness")
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo"); demo.add_argument("--output",default="artifacts/demo.json")
    data=sub.add_parser("data"); data.add_argument("--output",default="artifacts/oracle-train.jsonl"); data.add_argument("--count",type=int,default=32)
    ev=sub.add_parser("evaluate"); ev.add_argument("--model",required=True); ev.add_argument("--adapter"); ev.add_argument("--split",choices=["train","dev","test"],default="test"); ev.add_argument("--count",type=int,default=16); ev.add_argument("--output",default="artifacts/evaluation.json")
    dist=sub.add_parser("distill"); dist.add_argument("--teacher",required=True); dist.add_argument("--count",type=int,default=32); dist.add_argument("--output",default="artifacts/teacher-train.jsonl")
    tr=sub.add_parser("train"); tr.add_argument("--data",required=True); tr.add_argument("--model",required=True); tr.add_argument("--output",default="checkpoints/lora"); tr.add_argument("--steps",type=int,default=40); tr.add_argument("--rank",type=int,default=8)
    sm=sub.add_parser("smoke"); sm.add_argument("--output",default="artifacts/smoke"); sm.add_argument("--steps",type=int,default=4)
    for command in [demo, data, ev, dist, tr]:
        command.add_argument("--seed", type=int, default=17)
    for command in [ev, dist]:
        command.add_argument("--max-steps", type=int, default=8)
        command.add_argument("--max-new-tokens", type=int, default=160)
    args=parser.parse_args()
    if args.command=="data":
        rows=[]
        for task in generate("train",args.count,args.seed):
            result=run(task,oracle_policy(task)); rows.append({"split":"train","id":task["id"],"family":task["family"],"label_source":"scripted_privileged_oracle","trace":result["trace"]})
        write_jsonl(args.output,rows); print(json.dumps({"records":len(rows),"label_source":"scripted_privileged_oracle"})); return
    if args.command=="train":
        from .training import train
        print(json.dumps(train(args.data,args.model,args.output,args.steps,args.rank,args.seed))); return
    if args.command=="smoke":
        from .training import smoke
        print(json.dumps(smoke(args.output,args.steps))); return
    if args.command=="distill":
        policy=HFPolicy(args.teacher,max_new_tokens=args.max_new_tokens); accepted=[]; attempted=[]
        for task in generate("train",args.count,args.seed):
            result=run(task,policy,max_steps=args.max_steps); attempted.append(result)
            if result["success"]:
                accepted.append({"split":"train","id":task["id"],"family":task["family"],"label_source":"model_teacher","teacher":args.teacher,"trace":result["trace"]})
        write_jsonl(args.output,accepted)
        report={"teacher":args.teacher,"runtime":policy.metadata,"seed":args.seed,"max_steps":args.max_steps,"max_new_tokens":args.max_new_tokens,"attempted":len(attempted),"accepted":len(accepted),"summary":summarize(attempted)}
        Path(args.output+".report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report)); return
    if args.command=="demo":
        tasks=generate("test",16,args.seed); results=[run(t,oracle_policy(t)) for t in tasks]
        kind="scripted_oracle_environment_check"; token_measured=False
    else:
        policy=HFPolicy(args.model,args.adapter,args.max_new_tokens); tasks=generate(args.split,args.count,args.seed)
        results=[run(t,policy,max_steps=args.max_steps) for t in tasks]; kind="model_evaluation"; token_measured=True
    report={"kind":kind,"model":getattr(args,"model",None),"adapter":getattr(args,"adapter",None),"python":platform.python_version(),"platform":platform.platform(),"seed":args.seed,"split":getattr(args,"split","test"),"max_steps":getattr(args,"max_steps",8),"max_new_tokens":getattr(args,"max_new_tokens",None),"runtime":getattr(locals().get("policy"),"metadata",None),"tokens_measured":token_measured,"summary":summarize(results),"results":results}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))

if __name__=="__main__": main()
