"""Подготовка обучения: экспорт, метки, оценка готовых ответов (docs/stage2_plan.md).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, evaluate, hf_runtime, mc12run, score, training  # noqa: E402

V2 = ROOT / "data" / "dataset" / "v2"
WORK = ROOT / "out" / "test_training"


class FakeTokenizer:
    """Токены - символы: хватает, чтобы проверить разметку меток."""
    eos_token_id = 0
    system_ok = True

    drop_system = False
    end = None

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.kw = kw
        if not self.system_ok and msgs[0]["role"] == "system":
            raise ValueError("System role not supported")
        if self.drop_system:
            msgs = [m for m in msgs if m["role"] != "system"]
        if self.end is not None:
            return "".join(f"<{m['role']}>{m['content']}" + (self.end if m["role"] == "assistant" else "") for m in msgs)                 + ("<assistant>" if add_generation_prompt else "")
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class Export(unittest.TestCase):
    def test_subset_and_answers(self):
        shutil.rmtree(WORK, ignore_errors=True)
        rep = training.export(V2, WORK / "n", "netlist", 2000)
        self.assertEqual(rep["train"]["records"], 2000)
        self.assertEqual(rep["dev"]["records"], 150)
        rows = training.read_jsonl(WORK / "n" / "train.jsonl")
        self.assertEqual([m["role"] for m in rows[0]["messages"]], ["system", "user", "assistant"])

    def test_limit(self):
        rep = training.export(V2, WORK / "l", "netlist", 1000, 200)
        self.assertEqual(rep["train"]["records"], 200)
        self.assertEqual(rep["dev"]["records"], 150)

    def test_every_answer_parses(self):
        for fmt in training.FORMATS:
            for r in training.read_jsonl(V2 / "dev.jsonl"):
                a = training.answer_for(r, fmt)
                if r.get("refuse"):
                    self.assertEqual(score.parse_refusal(a, fmt)[0], r["refuse"])
                else:
                    el, err = score.parse_output(a, fmt)
                    self.assertEqual(err, [], r["id"])
                    self.assertEqual(canon.to_netlist(canon.canonicalize(el)), r["netlist"], r["id"])

    def test_no_circuit_shared_with_test(self):
        train = {r["circuit_id"] for r in training.read_jsonl(V2 / "train.jsonl") if r.get("circuit_id")}
        test = {r["circuit_id"] for r in training.read_jsonl(V2 / "test.jsonl") if r.get("circuit_id")}
        self.assertFalse(train & test)


class Labels(unittest.TestCase):
    MSGS = training.messages("RC 1к", "R1 IN OUT 1k")

    def test_only_answer_is_learned(self):
        tok = FakeTokenizer()
        e = training.encode(tok, self.MSGS, 10000)
        n = sum(l != -100 for l in e["labels"])
        self.assertEqual(n, len("R1 IN OUT 1k") + 1)
        self.assertEqual(e["input_ids"][-1], tok.eos_token_id)
        self.assertEqual(len(e["input_ids"]), len(e["labels"]))

    def test_too_long_is_dropped(self):
        self.assertIsNone(training.encode(FakeTokenizer(), self.MSGS, 10))

    def test_template_without_system_role(self):
        tok = FakeTokenizer()
        tok.system_ok = False
        text = training.chat_text(tok, self.MSGS)
        self.assertIn("RC 1к", text)
        self.assertIn(training.SYSTEM["netlist"][:30], text)

    def test_thinking_off(self):
        tok = FakeTokenizer()
        training.chat_text(tok, self.MSGS)
        self.assertIs(tok.kw.get("enable_thinking"), False)

    def test_system_silently_dropped(self):
        tok = FakeTokenizer()
        tok.drop_system = True  # как шаблон YandexGPT 5: system не попадает в текст, ошибки нет
        self.assertIn(training.SYSTEM["netlist"][:30], training.chat_text(tok, self.MSGS))

    def test_end_of_turn_from_template(self):
        tok = FakeTokenizer()
        tok.end = "|"  # шаблон закрывает ответ своим знаком, как <|im_end|> или <turn|>
        self.assertEqual(training.end_of_turn(tok), [ord("|")])
        e = training.encode(tok, self.MSGS, 10000)
        self.assertEqual(e["input_ids"][-1], ord("|"))
        self.assertEqual(training.end_of_turn(FakeTokenizer()), [FakeTokenizer.eos_token_id])


class Instructions(unittest.TestCase):
    """Инструкция ответа зависит от того, на чём обучена модель, а не от выборки (колаб: predict --instructions)."""

    def test_v3_model_on_any_split(self):
        for rec in ({"id": "llm"}, {"gen": {"version": "gen-2"}}, {"mode": 1, "gen": {"version": "gen-3"}}):
            self.assertEqual(training.instruction(rec, "netlist", "v3"), training.SYSTEM_V3["netlist"], rec)
        self.assertEqual(training.instruction({"mode": 2}, "netlist", "v3"), training.SYSTEM_V3["mode2"])

    def test_v2_model_unchanged(self):
        for rec in ({"id": "llm"}, {"mode": 2}, {"gen": {"version": "gen-3"}}):
            self.assertEqual(training.instruction(rec, "netlist", "v2"), training.SYSTEM["netlist"])

    def test_training_and_answer_agree_on_v3(self):
        """Чему модель v3 училась (system_for по записи v3) - с тем она и отвечает."""
        recs = training.read_jsonl(ROOT / "data/dataset/v3/dev.jsonl")
        for r in recs:
            self.assertEqual(training.system_for(r), training.instruction(r, "netlist", "v3"), r["id"])

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            training.instruction({}, "netlist", "v4")


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Predictions(unittest.TestCase):
    def test_gold_answers_score_full(self):
        recs = training.read_jsonl(V2 / "dev.jsonl")[:40]
        preds = [{"id": r["id"], "model": "gold", "format": "ir", "output": training.answer_for(r, "ir")} for r in recs]
        s = evaluate.evaluate_predictions(preds, recs, WORK / "gold")["gold/ir"]["all"]
        self.assertEqual(s["S3"]["k"], s["n"])
        self.assertEqual(s["S4c"]["k"], s["n"])
        self.assertEqual(s.get("refusals", {"missed": 0})["missed"], 0)

    def test_id_mismatch_refused(self):
        recs = training.read_jsonl(V2 / "dev.jsonl")[:3]
        with self.assertRaises(ValueError):
            evaluate.evaluate_predictions([{"id": "x", "model": "m", "output": ""}], recs, WORK / "bad")


class AdapterPath(unittest.TestCase):
    def test_falls_back_to_last_checkpoint(self):
        run = WORK / "adapter_run"
        shutil.rmtree(run, ignore_errors=True)
        (run / "adapter").mkdir(parents=True)
        (run / "adapter" / "adapter_model.safetensors").write_bytes(b"")
        for n in (100, 125):
            (run / "checkpoints" / f"checkpoint-{n}").mkdir(parents=True)
            (run / "checkpoints" / f"checkpoint-{n}" / "adapter_config.json").write_text("{}")
        self.assertEqual(Path(hf_runtime.adapter_path(run / "adapter")).name, "checkpoint-125")
        (run / "adapter" / "adapter_config.json").write_text("{}")
        self.assertEqual(Path(hf_runtime.adapter_path(run / "adapter")).name, "adapter")

    def test_nothing_found(self):
        with self.assertRaises(FileNotFoundError):
            hf_runtime.adapter_path(WORK / "no_such_run" / "adapter")


if __name__ == "__main__":
    unittest.main()
