"""Проверка самой метрики на golden-наборе (docs/eval_spec.md, раздел 8).

Эталоны прогоняются в MC12 через bench, их измерения становятся measured; затем score.py оценивает
изменённые ответы, и уровни/класс ошибки сверяются с ожидаемыми из tests/golden/golden.json.
Требует Micro-Cap 12 (уровни S1 и S4); без него пропускается - офлайн-часть покрыта test_iso.
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import bench, canon, mc12run, measure, score  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "golden" / "golden.json").read_text(encoding="utf-8"))
WORK = ROOT / "out" / "test_golden"


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)
        refs = {r["id"]: dict(r) for r in GOLDEN["refs"]}
        runs = bench.run([(rid, canon.parse_netlist(r["netlist"])[0], r["f_char"]) for rid, r in refs.items()],
                         WORK / "refs")
        for rid, r in refs.items():
            assert runs[rid]["ok"], runs[rid]
            r["measured"] = measure.from_output(runs[rid]["output"])
        cls.refs = refs
        preds = [{"id": c["ref"], "model": "golden", "format": c.get("format", "netlist"), "output": c["output"]}
                 for c in GOLDEN["cases"]]
        cls.rows = dict(zip((c["name"] for c in GOLDEN["cases"]), score.score_all(refs, preds, WORK / "preds")))

    def test_reference_measurements(self):
        m = self.refs["g_rc_lp"]["measured"]
        self.assertEqual(m["type"], "LP")
        self.assertAlmostEqual(m["fc"] / 1591.55, 1, delta=1e-3)
        m = self.refs["g_inv_lp"]["measured"]
        self.assertEqual(m["type"], "LP")
        self.assertAlmostEqual(m["k_pass_db"], 40.0, delta=0.01)  # |K| = 1MEG/10k = 100
        self.assertAlmostEqual(m["fc"] / 159.155, 1, delta=1e-3)

    def test_cases(self):
        for c in GOLDEN["cases"]:
            row = self.rows[c["name"]]
            for k, v in c["expect"].items():
                with self.subTest(case=c["name"], field=k):
                    self.assertEqual(row[k], v, {x: row[x] for x in ("S0", "S1", "S2", "S3", "S4",
                                                                      "functional", "error", "parse_errors",
                                                                      "behavior_detail")})

    def test_strict_levels(self):
        for name, row in self.rows.items():
            for lo, hi in zip(score.LEVELS, score.LEVELS[1:]):
                self.assertTrue(row[lo] or not row[hi], (name, lo, hi))

    def test_summary(self):
        s = score.summarize(list(self.rows.values()))["golden/netlist"]
        self.assertEqual(s["all"]["n"], sum(1 for c in GOLDEN["cases"] if c.get("format", "netlist") == "netlist"))
        lo, hi = s["all"]["S3"]["ci95"]
        self.assertTrue(0 <= lo <= s["all"]["S3"]["rate"] <= hi <= 1)
        self.assertEqual(set(s["by_level"]), {"L1", "L2"})


class Wilson(unittest.TestCase):
    def test_known_values(self):
        lo, hi = score.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(score.wilson(0, 0), (None, None))
        self.assertEqual(score.wilson(10, 10)[1], 1.0)


if __name__ == "__main__":
    unittest.main()
