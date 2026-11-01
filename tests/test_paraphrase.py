"""Проверка парафраз train v2 (docs/generator_v2_spec.md, раздел 4).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import hashlib
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import paraphrase  # noqa: E402

WORK = ROOT / "out" / "test_paraphrase"


class Frozen(unittest.TestCase):
    def test_v2_hashes(self):
        d = ROOT / "data" / "dataset" / "v2"
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for name, info in m["files"].items():
            b = (d / name).read_bytes()
            self.assertEqual(hashlib.sha256(b).hexdigest(), info["sha256"], name)
            self.assertEqual(len(b.splitlines()), info["records"], name)

    def test_v2_no_frozen_circuits(self):
        d = ROOT / "data" / "dataset"
        frozen = {r["circuit_id"] for r in paraphrase.read_jsonl(d / "llm_test_v1" / "llm_test.jsonl") if r.get("circuit_id")}
        for name in ("train", "dev", "test"):
            ids = {r["circuit_id"] for r in paraphrase.read_jsonl(d / "v2" / f"{name}.jsonl")}
            self.assertFalse(ids & frozen, name)


class Check(unittest.TestCase):
    def test_same_values_other_words(self):
        self.assertEqual(paraphrase.check("R1 10к вх-т1; C1 0,1мк т1-0; сигнал на вх",
                                          "поставь резистор на 10 кОм, а за ним конденсатор 100 нФ на землю"), [])

    def test_changed_value(self):
        self.assertTrue(paraphrase.check("RC ФНЧ 1к 100н", "RC-фильтр: 1,5 кОм и 100 нФ"))

    def test_lost_value(self):
        self.assertTrue(paraphrase.check("RC ФНЧ 1к 100н", "RC-фильтр на 1 кОм"))

    def test_source_voltage(self):
        self.assertTrue(paraphrase.check("RC 1к 100н, на вх 5 В", "RC-фильтр 1 кОм и 100 нФ"))
        self.assertEqual(paraphrase.check("RC 1к 100н, на вх 5 В", "RC-фильтр 1 кОм, 100 нФ, вход 5 В"), [])

    def test_designators_are_not_values(self):
        self.assertEqual(paraphrase.values("R1 1к, C2 100н, т3"), paraphrase.values("1 кОм и 100 нФ"))

    def test_inductance(self):
        self.assertEqual(paraphrase.check("L 1,5мГн", "дроссель 1500 мкГн"), [])


class Apply(unittest.TestCase):
    def test_apply(self):
        shutil.rmtree(WORK, ignore_errors=True)
        src = WORK / "src"
        src.mkdir(parents=True)
        recs = [{"id": "a", "prompt": "RC 1к 100н", "family": "terse", "paraphrase": True},
                {"id": "b", "prompt": "RC 2к 100н", "family": "terse", "paraphrase": True},
                {"id": "c", "prompt": "RC 3к 100н", "family": "terse", "paraphrase": False}]
        (src / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        (src / "manifest.json").write_text(json.dumps({"files": {"train.jsonl": {}}}), encoding="utf-8")
        para = WORK / "p.jsonl"
        para.write_text('{"id": "a", "prompt": "фильтр на 1 кОм и 100 нФ"}\n{"id": "b", "prompt": "фильтр на 5 кОм и 100 нФ"}\n', encoding="utf-8")
        applied, errors = paraphrase.apply(src, para, WORK / "out")
        self.assertEqual((applied, [e["id"] for e in errors]), (1, ["b"]))
        out = paraphrase.read_jsonl(WORK / "out" / "train.jsonl")
        self.assertEqual((out[0]["family"], out[0]["prompt_template"]), ("paraphrase", "RC 1к 100н"))
        self.assertEqual(out[1]["prompt"], "RC 2к 100н")


if __name__ == "__main__":
    unittest.main()
