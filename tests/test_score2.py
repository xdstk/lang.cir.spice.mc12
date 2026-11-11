"""Оценка режима 2 (docs/mode2_spec.md, раздел 7): схема с `?`, поля .SPEC, поведение через synth и MC12.

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import random
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import evaluate, mc12run, mode2_data, score2  # noqa: E402

WORK = ROOT / "out" / "test_score2"
SK = ("V1 IN 0 DC 0\nR1 IN N1 ?\nR2 N1 N2 ?\nC1 N1 OUT 22n\nC2 N2 0 10n\nX1 N2 OUT OUT OPAMP_IDEAL\n"
      ".SPEC type=LP fc=1k")


def ref(output, rid="a", kind="named"):
    rep = mode2_data.solve(output)
    spec = {k: v for k, v in (tok.split("=") for tok in output.splitlines()[-1].split()[1:])}
    return {"id": rid, "mode": 2, "kind": kind, "family": "x", "output": output, "spec": spec,
            "solution": {"wanted": rep["wanted"], "route": rep["route"]}}


class Parts(unittest.TestCase):
    def test_parse_marks_unknown(self):
        elems, spec, errors = score2.parse(SK)
        self.assertEqual(errors, [])
        self.assertEqual([e["value"] for e in elems if e["name"] in ("R1", "R2")], [score2.UNKNOWN] * 2)
        self.assertEqual(spec, {"type": "LP", "fc": "1k"})
        self.assertTrue(score2.parse("R1 IN OUT 1k\nC1 OUT 0 100n")[2])  # нет строки .SPEC
        self.assertTrue(score2.parse("привет")[2])

    def test_same_spec(self):
        a = {"type": "LP", "fc": "1k", "Q": "0.707"}
        self.assertEqual(score2.same_spec(a, {"type": "lp", "fc": "1000", "Q": "707m"}), (True, []))
        self.assertEqual(score2.same_spec(a, {"type": "LP", "fc": "1.1k", "Q": "0.707"}), (False, ["fc"]))
        self.assertEqual(score2.same_spec(a, {"type": "LP", "fc": "1k"}), (False, ["Q"]))
        self.assertEqual(score2.same_spec(a, dict(a, active="yes")), (False, ["active"]))
        self.assertEqual(score2.same_spec(a, dict(a, type="HP")), (False, ["type"]))

    def test_circuit_error(self):
        r = score2.parse(SK)[0]
        self.assertEqual(score2.circuit_error(r, score2.parse(SK.replace("R2 N1 N2 ?", "R2 N1 N2 10k"))[0]), "wrong_unknowns")
        self.assertEqual(score2.circuit_error(r, score2.parse(SK.replace("22n", "47n"))[0]), "wrong_value")
        self.assertEqual(score2.circuit_error(r, score2.parse(SK.replace("C1 N1 OUT", "C1 N1 0"))[0]), "wrong_circuit")


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)

    def score(self, reference, outputs, name):
        refs = {reference["id"]: reference}
        preds = [{"id": reference["id"], "model": f"m{i}", "format": "netlist", "output": o} for i, o in enumerate(outputs)]
        return score2.score_all(refs, preds, WORK / name)

    def test_circuit_answers(self):
        r = ref(SK)
        rows = self.score(r, [
            SK,                                                                  # 0 эталон
            SK.replace("R1 IN N1 ?", "R1 IN A ?").replace("R2 N1 N2", "R2 A N2").replace("C1 N1", "C1 A"),  # 1 другое имя узла
            SK.replace("R2 N1 N2 ?", "R2 N1 N2 10k"),                             # 2 `?` не там
            SK.replace("fc=1k", "fc=2k"),                                         # 3 другая цель
            SK.replace("fc=1k", "fc=1000"),                                       # 4 та же цель в другой записи
            ".SPEC type=LP fc=1k order=2 active=yes",                             # 5 без схемы, но тот же фильтр
            "REFUSE ambiguous: не задана частота",                               # 6 ложный отказ
            "R1 IN OUT ?",                                                        # 7 нет .SPEC
            SK.replace("22n", "10n"),                                             # 8 C1 = C2: Q 0,707 недостижима
        ], "circuit")
        exact = [x["exact"] for x in rows]
        self.assertEqual(exact, [True, True, False, False, True, False, False, False, False])
        self.assertEqual([x["error"] for x in rows], [None, None, "wrong_unknowns", "wrong_spec", None, "missing_circuit",
                                                      "false_refusal", "syntax", "wrong_value"])
        self.assertTrue(rows[0]["behavior"] and rows[1]["behavior"] and rows[4]["behavior"])
        self.assertFalse(rows[3]["behavior"])        # фильтр на 2 кГц вместо 1 кГц
        self.assertTrue(rows[5]["behavior"])         # другая запись ответа, но фильтр с теми же характеристиками
        self.assertFalse(rows[6]["parse"] or rows[7]["parse"])
        self.assertEqual(rows[6]["refuse_pred"], "ambiguous")

    def test_spec_answers(self):
        r = ref(".SPEC type=BP f0=5k Q=5 active=yes", kind="spec")
        rows = self.score(r, [".SPEC type=BP f0=5k Q=5 active=yes", ".SPEC type=BP f0=5k active=yes Q=5",
                              ".SPEC type=BP f0=5k Q=5", ".SPEC type=BS f0=5k Q=5 active=yes",
                              "V1 IN 0 DC 0\nR1 IN OUT ?\nC1 OUT 0 ?\n.SPEC type=BP f0=5k Q=5 active=yes"], "spec")
        self.assertEqual([x["exact"] for x in rows], [True, True, False, False, False])
        self.assertEqual([x["error"] for x in rows], [None, None, "wrong_spec", "wrong_spec", "extra_circuit"])
        self.assertTrue(rows[2]["behavior"])   # пассивный полосовой с теми же f0 и Q - поведение то же
        self.assertFalse(rows[3]["behavior"])

    def test_generated_references_score_full(self):
        """Эталоны mode2_data на собственных ответах: все уровни 100%, отказы пойманы."""
        rng = random.Random(1)
        recs = [dict(mode2_data.sample(rng), id=f"g{i:02d}") for i in range(30)]
        preds = [{"id": r["id"], "model": "gold", "format": "netlist", "output": r["output"]} for r in recs]
        rows = score2.score_all({r["id"]: r for r in recs}, preds, WORK / "gold")
        s = score2.summarize(rows)["all"]
        for lvl in score2.LEVELS:
            self.assertEqual(s[lvl]["k"], s["n"], (lvl, [x["id"] for x in rows if not x.get("refuse_ref") and not x[lvl]]))
        if "refusals" in s:
            self.assertEqual(s["refusals"]["code_correct"], s["refusals"]["n"])

    def test_evaluate_mixes_modes(self):
        """evaluate: записи режима 1 - score, режима 2 - score2, сводка содержит оба блока."""
        net = "V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n"
        m1 = {"id": "r1", "mode": 1, "task": "A1", "family": "terse", "topology": "rc_lp", "level": "L1", "prompt": "фнч 1к 100н",
              "netlist": net, "f_char": 1591.5, "quantities": ["type", "k_pass_db", "fc"],
              "measured": {"type": "LP", "k_pass_db": 0.0, "fc": 1591.5}}
        m2 = dict(ref(SK, rid="r2"), prompt="Саллен-Ки ФНЧ на 1 кГц, C1 22н, C2 10н", task="B", topology="sallen_key_lp", level=None)
        preds = [{"id": "r1", "model": "gold", "format": "netlist", "output": net},
                 {"id": "r2", "model": "gold", "format": "netlist", "output": SK}]
        summary = evaluate.evaluate_predictions(preds, [m1, m2], WORK / "mixed")["gold/netlist"]
        self.assertEqual(summary["all"]["n"], 1)
        self.assertEqual(summary["all"]["S3"]["k"], 1)
        self.assertEqual(summary["mode2"]["all"]["n"], 1)
        self.assertEqual(summary["mode2"]["all"]["exact"]["k"], 1)
        self.assertEqual(summary["mode2"]["all"]["behavior"]["k"], 1)


if __name__ == "__main__":
    unittest.main()
