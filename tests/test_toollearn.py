import json
import unittest
from toollearn.env import Workspace, ToolError, safe_path
from toollearn.tasks import generate, FAMILIES, oracle_actions
from toollearn.experiment import run, oracle_policy, summarize
from toollearn.training import examples_from_records

class WorkspaceTests(unittest.TestCase):
    def test_boundaries(self):
        for path in ["../secret", "/etc/passwd", "C:/secret", "a/../../b", "a\\b", ""]:
            with self.subTest(path=path), self.assertRaises(ToolError): safe_path(path)
    def test_strict_schema(self):
        env=Workspace({"a":"hello"})
        for action in [[], {}, {"tool":"shell","arguments":{"command":"whoami"}},
            {"tool":"read","arguments":{"path":"a","extra":1}},
            {"tool":"edit","arguments":{"path":"a","old":"","new":"x"}},
            {"tool":"read","arguments":{"path":3}}]:
            with self.subTest(action=action), self.assertRaises(ToolError): env.call(action)
        self.assertEqual(env.files,{"a":"hello"})
    def test_edit_atomic(self):
        env=Workspace({"a":"x x"})
        with self.assertRaises(ToolError): env.call({"tool":"edit","arguments":{"path":"a","old":"x","new":"z"}})
        self.assertEqual(env.files["a"],"x x")
    def test_search_and_read(self):
        env=Workspace({"a":"foo\nbar\nfoo"})
        self.assertEqual(len(json.loads(env.call({"tool":"search","arguments":{"query":"foo"}}))),2)
        self.assertEqual(env.call({"tool":"read","arguments":{"path":"a"}}),"foo\nbar\nfoo")

class DatasetTests(unittest.TestCase):
    def test_deterministic(self): self.assertEqual(generate("train"),generate("train"))
    def test_disjoint(self):
        sets=[generate(split) for split in FAMILIES]
        for i,left in enumerate(sets):
            for right in sets[i+1:]:
                self.assertFalse({t["family"] for t in left}&{t["family"] for t in right})
                self.assertFalse({t["id"] for t in left}&{t["id"] for t in right})
                self.assertFalse({next(iter(t["files"])) for t in left}&{next(iter(t["files"])) for t in right})
    def test_oracle_all_families(self):
        for split in FAMILIES:
            for task in generate(split,8):
                result=run(task,oracle_policy(task))
                self.assertTrue(result["success"]); self.assertEqual(result["invalid_calls"],0)
    def test_training_rejects_test(self):
        with self.assertRaises(ValueError): examples_from_records([{"split":"test","trace":[]}])
    def test_prompt_excludes_answers(self):
        t=generate("test",1)[0]
        seen=[]
        def policy(messages,step):
            seen.extend(messages)
            return '{"tool":"finish","arguments":{}}',{}
        run(t,policy)
        self.assertEqual(len(seen),2)
        self.assertNotIn("expected",seen[1]["content"])

class EvaluationTests(unittest.TestCase):
    def test_invalid_bounded(self):
        r=run(generate("test",1)[0],lambda m,s:("invalid",{}),max_steps=3)
        self.assertEqual(r["invalid_calls"],3); self.assertEqual(r["calls"],3); self.assertFalse(r["success"])
    def test_finish_required(self):
        task=generate("train",1)[0]
        actions=oracle_actions(task)
        r=run(task,lambda m,s:(json.dumps(actions[s]),{}),max_steps=3)
        self.assertFalse(r["success"])
    def test_metrics(self):
        r=run(generate("test",1)[0],oracle_policy(generate("test",1)[0]))
        metrics=summarize([r]); self.assertEqual(metrics["success_rate"],1); self.assertEqual(metrics["mean_calls"],4)

if __name__=="__main__": unittest.main()
