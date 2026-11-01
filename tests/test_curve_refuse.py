"""Сравнение кривых и отказы модели (docs/generator_v2_spec.md, разделы 2.4 и 6).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import app, canon, mc12run, measure, score, translators  # noqa: E402

WORK = ROOT / "out" / "test_curve"
RC = "V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n"


class Curve(unittest.TestCase):
    F = [10.0, 100.0, 1000.0, 10000.0]

    def test_same_curve(self):
        c = (self.F, [0.0, -1.0, -20.0, -80.0], [0.0, -10.0, -80.0, -90.0])
        self.assertEqual(measure.curve_match(c, c)[0], True)

    def test_level_and_phase_limits(self):
        a = (self.F, [0.0, -1.0, -20.0, -30.0], [0.0, -10.0, -80.0, -90.0])
        self.assertFalse(measure.curve_match(a, (self.F, [0.0, -1.0, -21.5, -30.0], a[2]))[0])
        self.assertFalse(measure.curve_match(a, (self.F, a[1], [0.0, -10.0, -90.0, -90.0]))[0])

    def test_below_floor_ignored(self):
        a = (self.F, [0.0, -1.0, -20.0, -80.0], [0.0, 0.0, 0.0, 0.0])
        b = (self.F, [0.0, -1.0, -20.0, -70.0], [0.0, 0.0, 0.0, 120.0])
        self.assertTrue(measure.curve_match(a, b)[0])

    def test_phase_wraps(self):
        a = (self.F[:1], [0.0], [179.0])
        b = (self.F[:1], [0.0], [-179.0])
        self.assertAlmostEqual(measure.curve_diff(a, b)[1], 2.0)

    def test_other_grid(self):
        a = (self.F, [0.0] * 4, [0.0] * 4)
        b = ([2 * x for x in self.F], [0.0] * 4, [0.0] * 4)
        self.assertEqual(measure.curve_match(a, b)[1]["max_db"], None)
        self.assertFalse(measure.curve_match(a, b)[0])


class Refusal(unittest.TestCase):
    def test_netlist_format(self):
        self.assertEqual(score.parse_refusal("REFUSE ambiguous: не указано, куда подключён C2"),
                         ("ambiguous", "не указано, куда подключён C2"))
        self.assertEqual(score.parse_refusal("REFUSE needs_design")[0], "needs_design")
        self.assertIsNone(score.parse_refusal("REFUSE maybe: не знаю"))
        self.assertIsNone(score.parse_refusal(RC))
        self.assertIsNone(score.parse_refusal(None))

    def test_ir_format(self):
        self.assertEqual(score.parse_refusal('{"refuse": "out_of_class", "reason": "диод"}', "ir"),
                         ("out_of_class", "диод"))
        self.assertIsNone(score.parse_refusal('{"format": "mcnl.ir/1", "elements": []}', "ir"))
        self.assertIsNone(score.parse_refusal("[1, 2]", "ir"))
        self.assertIsNone(score.parse_refusal("не JSON", "ir"))

    def test_app_reports_refusal(self):
        class Refuser:
            output_format = "netlist"

            def translate(self, prompt):
                return "REFUSE out_of_class: транзистор"
        rep = app.run("усилитель на КТ315", Refuser(), mc12=False, outdir=WORK)
        self.assertEqual(rep["refused"]["code"], "out_of_class")
        self.assertNotIn("netlist", rep)

    def test_summary_without_mc12(self):
        rows = [{"model": "m", "format": "netlist", "class": "A", "level": "L1", "refuse_ref": "ambiguous",
                 "refuse_pred": "ambiguous"},
                {"model": "m", "format": "netlist", "class": "A", "level": "L1", "refuse_ref": "out_of_class",
                 "refuse_pred": None}]
        s = score.summarize(rows)["m/netlist"]["all"]
        self.assertEqual(s["refusals"], {"n": 2, "caught": 1, "code_correct": 1, "missed": 1})
        self.assertEqual(s["n"], 0)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class ScoreWithMC12(unittest.TestCase):
    def test_levels(self):
        shutil.rmtree(WORK, ignore_errors=True)
        elems = canon.canonicalize(canon.parse_netlist(RC)[0])
        q = ["type", "k_pass_db", "fc"]
        m = {"type": "LP", "k_pass_db": 0.0, "fc": 1591.5}
        refs = {"a": {"netlist": canon.to_netlist(elems), "f_char": 1591.5, "measured": m, "quantities": q},
                "b": {"refuse": "ambiguous"}}
        preds = [{"id": "a", "model": "m", "output": "R7 IN X 1k\nC2 X 0 100n\nV1 IN 0 DC 0\n"},  # другой узел OUT
                 {"id": "a", "model": "m", "output": RC},
                 {"id": "a", "model": "m", "output": RC.replace("1k", "1.5k")},
                 {"id": "a", "model": "m", "output": "REFUSE ambiguous: неясно"},
                 {"id": "b", "model": "m", "output": "REFUSE ambiguous: неясно"},
                 {"id": "b", "model": "m", "output": RC}]
        rows = score.score_all(refs, preds, WORK)
        by = {i: r for i, r in enumerate(rows)}
        good = next(r for r in rows if r.get("S3"))
        self.assertTrue(good["curve"] and good["S4c"] and good["functional_c"])
        self.assertLess(good["curve_detail"]["max_db"], 0.02)
        wrong = rows[2]
        self.assertTrue(wrong["S1"] and not wrong["curve"])
        self.assertFalse(wrong["S3"])
        self.assertEqual(rows[3]["error"], "false_refusal")
        self.assertFalse(rows[3]["S0"])
        s = score.summarize(rows)["m/netlist"]["all"]
        self.assertEqual(s["refusals"], {"n": 2, "caught": 1, "code_correct": 1, "missed": 1})
        self.assertEqual(s["false_refusals"], 1)
        self.assertEqual(s["n"], 4)
        self.assertEqual(len(by), 6)


if __name__ == "__main__":
    unittest.main()
