"""Тесты сменных переводчиков («батареек»), baseline на правилах, программы app.py и evaluate.py.

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import math
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import app, canon, evaluate, iso, mc12run, translators  # noqa: E402

TEST = [json.loads(l) for l in (ROOT / "data" / "dataset" / "v1" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
RC = "V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n"


class Plugins(unittest.TestCase):
    def test_registry_and_module_spec(self):
        self.assertEqual(translators.load("baseline").name, "baseline")
        self.assertEqual(translators.load("mcnl.translators.echo:Echo").translate(RC), RC)

    def test_file_spec(self):
        # «батарейка» из отдельного файла - так подключается обученная модель
        d = ROOT / "out" / "test_app"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "my_model.py"
        f.write_text("class Model:\n    name = 'my'\n    output_format = 'ir'\n"
                     "    def translate(self, prompt):\n        return '{}'\n", encoding="utf-8")
        m = translators.load(f"{f}:Model")
        self.assertEqual((m.name, m.output_format), ("my", "ir"))

    def test_bad_specs(self):
        with self.assertRaises(ValueError):
            translators.load("nosuch")
        d = ROOT / "out" / "test_app"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "broken.py"
        f.write_text("class Model:\n    pass\n", encoding="utf-8")
        with self.assertRaises(TypeError):
            translators.load(f"{f}:Model")


class Baseline(unittest.TestCase):
    def test_all_tasks_and_families(self):
        tr = translators.load("baseline")
        seen = {}
        for r in TEST:
            key = (r["task"], r["family"])
            if seen.get(key, 0) >= 15:
                continue
            seen[key] = seen.get(key, 0) + 1
            out = tr.translate(r["prompt"])
            with self.subTest(id=r["id"], key=key):
                self.assertTrue(out, r["prompt"])
                self.assertTrue(iso.match(canon.parse_netlist(r["netlist"])[0], canon.parse_netlist(out)[0],
                                          "values")["match"])
        self.assertGreaterEqual(len(seen), 6)

    def test_tolerance_and_failure(self):
        tr = translators.load("baseline")
        self.assertEqual(tr.translate("rc фнч, r=1 кОм, c=100 нФ"), RC)
        self.assertEqual(tr.translate("нарисуй что-нибудь красивое"), "")

    def test_role_inside_topology_name(self):
        # «интегратор с Rос: Rвх …» - название содержит обозначение роли Rос
        out = translators.load("baseline").translate("интегратор с Rос: Rвх 1.5к, Rос 150к, C 39н; сигнал на вх")
        self.assertIn("150k", out)


class App(unittest.TestCase):
    def test_offline_pipeline(self):
        rep = app.run("rc фнч, r=1 кОм, c=100 нФ", translators.load("baseline"), outdir=ROOT / "out" / "test_app",
                      name="rc_off", mc12=False)
        self.assertEqual(rep["netlist"], RC)
        self.assertAlmostEqual(rep["f_char"] / (1 / (2 * math.pi * 1e-4)), 1, delta=5e-3)
        self.assertTrue(Path(rep["cir"]).exists())

    def test_unparsed_answer(self):
        rep = app.run("непонятно", translators.load("baseline"), outdir=ROOT / "out" / "test_app", mc12=False)
        self.assertNotIn("netlist", rep)
        self.assertTrue(rep["messages"])

    def test_flat_circuit_has_no_fchar(self):
        rep = app.run(netlist="V1 IN 0 DC 5\nR1 IN OUT 10k\nR2 OUT 0 10k\n", outdir=ROOT / "out" / "test_app",
                      name="div", mc12=False)
        self.assertIsNone(rep["f_char"])


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class WithMC12(unittest.TestCase):
    def test_pipeline_measures(self):
        rep = app.run(netlist=RC, outdir=ROOT / "out" / "test_app", name="rc_mc12")
        self.assertTrue(rep["run"]["ok"], rep)
        self.assertEqual(rep["measured"]["type"], "LP")
        self.assertAlmostEqual(rep["measured"]["fc"] / (1 / (2 * math.pi * 1e-4)), 1, delta=3e-3)

    def test_evaluate_subset(self):
        out = ROOT / "out" / "test_app" / "eval"
        shutil.rmtree(out, ignore_errors=True)
        s = evaluate.evaluate(translators.load("baseline"), TEST[:12], out)["baseline/netlist"]
        self.assertEqual(s["all"]["n"], 12)
        self.assertEqual(s["all"]["S4"]["k"], 12)
        self.assertTrue((out / "predictions.jsonl").exists())
        self.assertIn("by_task", s)


if __name__ == "__main__":
    unittest.main()
