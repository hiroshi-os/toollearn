import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from toollearn.reasoning import generate_tasks, parse_final, evaluate_final
from toollearn.runtime import render
from toollearn.studies import collect, reusable_teacher_data, write_json, write_records
from toollearn.tasks import generate
from toollearn.experiment import oracle_policy, decode_generation


class ReasoningTests(unittest.TestCase):
    def test_math_and_disjoint_holdouts(self):
        dev, test = generate_tasks("dev", 99), generate_tasks("test", 99)
        self.assertEqual(test, generate_tasks("test", 99))
        self.assertFalse({t["question"] for t in dev} & {t["question"] for t in test})
        for task in dev + test:
            a, b, c = task["operands"]
            expected = {"arithmetic": (a-b)*c, "modular": (a*b+c)%7, "linear": c}[task["family"]]
            self.assertEqual(int(task["answer"]), expected)

    def test_parser(self):
        self.assertEqual(parse_final("details\nFINAL: -002"), "-2")
        self.assertEqual(parse_final("<think>FINAL: 9</think>\nFINAL: +7"), "7")
        self.assertEqual(parse_final("details</think>\nFINAL: 7", True), "7")
        for text in ["42", "FINAL: 1.0", "FINAL: 1\nprose", "FINAL: 1\nFINAL: 2",
                     "<think>\nFINAL: 5"]:
            self.assertIsNone(parse_final(text))
        self.assertIsNone(parse_final("FINAL: 7", True))

    def test_actual_final_answer_scoring(self):
        tasks = generate_tasks("test", 3)
        answers = iter([f"FINAL: {t['answer']}" for t in tasks])
        report = evaluate_final(lambda m, s: (next(answers), {"output_tokens": 3}), tasks, "default")
        self.assertEqual(report["accuracy"], 1)

    def test_thinking_support_rejected(self):
        class Tokenizer:
            chat_template = "ordinary template"
        with self.assertRaises(ValueError):
            render(Tokenizer(), [], True, True)
        with self.assertRaises(ValueError):
            render(Tokenizer(), [], False, True)

    def test_special_thinking_delimiters_survive_decode(self):
        class Tokenizer:
            all_special_tokens = ["<think>", "</think>", "[EOS]", "[PAD]"]
            def decode(self, tokens, skip_special_tokens):
                self.assert_preserved = not skip_special_tokens
                return tokens
        tokenizer = Tokenizer()
        text = decode_generation(tokenizer, "<think>intermediate</think>\nFINAL: 7[EOS][PAD]", True)
        self.assertTrue(tokenizer.assert_preserved)
        self.assertEqual(parse_final(text, True), "7")
        self.assertIsNone(parse_final(decode_generation(tokenizer, "<think>\nFINAL: 7[EOS]", True), True))


class DistillationTests(unittest.TestCase):
    def test_zero_accept_and_dataset_reuse(self):
        tasks = generate("train", 2)
        accepted, results = collect(tasks, lambda m,s: ("bad", {}), 2, "test_teacher")
        self.assertEqual(accepted, [])
        self.assertEqual(len(results), 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"teacher.jsonl"
            write_records(path, accepted)
            config = {"seed": 17}
            write_json(str(path)+".report.json", {"collection_config": config, "accepted": 0,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            self.assertEqual(reusable_teacher_data(path, tasks, config)[0], [])
            with self.assertRaises(ValueError):
                reusable_teacher_data(path, tasks, {"seed": 18})
            path.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                reusable_teacher_data(path, tasks, config)

    def test_accepted_records_are_model_teacher_only(self):
        task = generate("train", 1)[0]
        rows, results = collect([task], oracle_policy(task), 8, "fake_test_teacher")
        self.assertTrue(results[0]["success"])
        self.assertEqual(rows[0]["label_source"], "model_teacher")
        self.assertEqual(rows[0]["split"], "train")


if __name__ == "__main__":
    unittest.main()
