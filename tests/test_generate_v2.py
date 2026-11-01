"""Сборка датасета v2 (docs/generator_v2_spec.md): пробная партия с MC12, без него пропускается.

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import random
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import generate_v2, mc12run, score  # noqa: E402

WORK = ROOT / "out" / "test_gen2"


class Plan(unittest.TestCase):
    def test_shares(self):
        c = Counter(generate_v2.plan(1000, random.Random(1)))
        self.assertEqual(sum(c.values()), 1000)
        self.assertEqual(c["refuse"], 50)
        self.assertEqual(c["strange"], 100)
        self.assertEqual(c["compose"], 300)

    def test_held_out(self):
        self.assertEqual(set(generate_v2.named_pool("gen_topology")), generate_v2.HELD_OUT)
        self.assertFalse(set(generate_v2.named_pool("train")) & generate_v2.HELD_OUT)

    def test_common_filters_dominate(self):
        pool = generate_v2.named_pool("train")
        rare = sum(w for t, w in pool.items() if t in generate_v2.RARE)
        self.assertLess(rare / sum(pool.values()), 0.2)

    def test_frozen_circuits_known(self):
        self.assertGreater(len(generate_v2.excluded_circuits()), 100)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Trial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)
        cls.m = generate_v2.generate(WORK / "ds", seed=3, scale=0.03, workdir=WORK / "runs")
        cls.recs = {s: [json.loads(l) for l in (WORK / "ds" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
                    for s in generate_v2.SPLITS}

    def test_splits_filled(self):
        for s, n in generate_v2.SPLITS.items():
            self.assertGreaterEqual(len(self.recs[s]), max(1, round(n * 0.03)) - 2, s)

    def test_circuits_do_not_repeat(self):
        ids = [r["circuit_id"] for rs in self.recs.values() for r in rs if r["circuit_id"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & generate_v2.excluded_circuits())

    def test_refusal_records(self):
        for r in self.recs["train"]:
            if r.get("refuse"):
                self.assertEqual(score.parse_refusal(r["output"])[0], r["refuse"])

    def test_held_out_family(self):
        self.assertEqual({r["family"] for r in self.recs["gen_family"] if r.get("kind") in ("named_a1", "named_a")}, {"path"})

    def test_subset_nested(self):
        self.assertEqual({r["subset"] for r in self.recs["train"]} - set(generate_v2.SUBSETS), set())


if __name__ == "__main__":
    unittest.main()
