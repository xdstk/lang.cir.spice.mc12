"""Тесты сборки датасета (docs/generator_spec.md, разделы 1, 5, 6).

Без MC12 проверяются выборка схем и формулировки; маленькая партия целиком - с MC12 (иначе пропуск).
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import collections
import json
import random
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import generate, mc12run  # noqa: E402


class Parts(unittest.TestCase):
    def test_level_plan_exact(self):
        plan = generate.level_plan(["L1", "L2", "L3"], 40, random.Random(1))
        self.assertEqual(collections.Counter(plan), {"L1": 12, "L2": 18, "L3": 10})

    def test_pools_hold_out(self):
        for split in ("train", "dev", "test", "gen_family", "gen_range"):
            pool = generate._pool(split)
            names = {t for ts in pool.values() for t in ts}
            with self.subTest(split=split):
                self.assertFalse(names & generate.HELD_OUT_TOPOLOGIES)
                self.assertFalse(names & generate.HELD_OUT_CASCADES)
        held = {t for ts in generate._pool("gen_topology").values() for t in ts}
        self.assertEqual(held, generate.HELD_OUT_TOPOLOGIES | generate.HELD_OUT_CASCADES)

    def test_sample_circuits_unique_and_range(self):
        seen = set()
        a = generate.sample_circuits("train", 30, random.Random(2), seen)
        b = generate.sample_circuits("gen_range", 10, random.Random(3), seen)
        ids = [r["circuit_id"] for r in a + b]
        self.assertEqual(len(ids), len(set(ids)))
        for r in b:
            self.assertTrue(any(isinstance(v, float) and k.startswith("R") and v > 1e6 for k, v in r["params"].items()))

    def test_prompts(self):
        refs = generate.sample_circuits("gen_family", 5, random.Random(4), set())
        for r in refs:
            ps = generate.prompts_for(r, 1, random.Random(5), "gen_family")
            self.assertEqual([(t, f) for t, f, _ in ps], [("A", "path")])
        r = generate.sample_circuits("train", 1, random.Random(6), set())[0]
        ps = generate.prompts_for(r, 2, random.Random(7), "train")
        self.assertEqual(len({p for _, _, p in ps}), 2)
        self.assertTrue(all(f != "path" for _, f, _ in ps))
        fams = [f for _ in range(60) for _, f, _ in generate.prompts_for(r, 1, random.Random(_), "train")]
        self.assertIn("brief", fams)
        self.assertIn("roles_brief", fams)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class SmallBatch(unittest.TestCase):
    def test_batch(self):
        out = ROOT / "out" / "test_generate"
        shutil.rmtree(out, ignore_errors=True)
        m = generate.generate(out, seed=3, scale=0.02, workdir=out / "runs")
        self.assertEqual(set(m["files"]), {f"{s}.jsonl" for s in generate.SPLITS})
        recs = [json.loads(l) for f in m["files"] for l in (out / f).read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(r["admission"]["admitted"] for r in recs))
        by_cid = collections.defaultdict(set)
        for r in recs:
            by_cid[r["circuit_id"]].add(r["split"])
        self.assertTrue(all(len(s) == 1 for s in by_cid.values()))
        self.assertEqual(m["files"]["train.jsonl"]["records"], 2 * m["files"]["train.jsonl"]["circuits"])


if __name__ == "__main__":
    unittest.main()
