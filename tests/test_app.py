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

    def test_aliases(self):
        # синонимы названий и ролей из data/roles_ru.json (aliases)
        tr = translators.load("baseline")
        self.assertEqual(tr.translate("RC-фильтр низких частот: резистор 1 кОм, конденсатор 100 нФ"), RC)
        out = tr.translate("инвертирующее включение ОУ: Rin=4,7к, Rf=47к, вход 1 В")
        self.assertEqual(out, "V1 IN 0 DC 1\nR1 IN N1 4.7k\nR2 N1 OUT 47k\nX1 0 N1 OUT OPAMP_IDEAL\n")
        self.assertIn("C1 N1 OUT 100n", tr.translate(
            "последовательный колебательный контур: катушка 10 мГн, конденсатор 0,1 мкФ, резистор 100 Ом"))
        self.assertIn("X1 N2 OUT OUT OPAMP_IDEAL", tr.translate(  # «–» в названии и «ё/е» нормализуются
            "ФНЧ Саллена–Ки на повторителе: R1 10 кОм, R2 10 кОм, C1(ос) 22 нФ, C2(земля) 10 нФ"))

    def test_span_used_once(self):
        # «резистор обратной связи» не должен отдать номинал ещё и роли «резистор»/«R1»
        out = translators.load("baseline").translate("интегратор на ОУ: резистор на входе 1к, "
                                                      "резистор обратной связи 100к, конденсатор 10н")
        self.assertEqual(out, "V1 IN 0 DC 0\nR1 IN N1 1k\nR2 N1 OUT 100k\nC1 N1 OUT 10n\nX1 0 N1 OUT OPAMP_IDEAL\n")

    def test_aliases_belong_to_known_roles(self):
        from mcnl import describe_ru
        for topo, r in describe_ru.ROLES.items():
            with self.subTest(topo=topo):
                self.assertIn("aliases", r)
                self.assertTrue(set(r["aliases"]["roles"]) <= set(r["roles"]))

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
        self.assertIn(app.INPUT_HINT, rep["messages"])

    def test_named_type(self):
        self.assertEqual(app.named_type("RC ФНЧ: R 1к, C 100н"), "LP")
        self.assertEqual(app.named_type("Фильтр высоких частот Саллена-Ки"), "HP")
        self.assertEqual(app.named_type("полосовой MFB: R1 11к"), "BP")
        self.assertEqual(app.named_type("Двойной Т-мост на 50 Гц"), "BS")
        self.assertEqual(app.named_type("полосно-заграждающий фильтр"), "BS")
        self.assertEqual(app.named_type("notch filter, R 10k"), "BS")
        self.assertIsNone(app.named_type("инвертирующий усилитель: Rвх 10к, Rос 100к"))
        self.assertIsNone(app.named_type("каскад: [Саллен-Ки ФВЧ] → [активный ФНЧ]"))  # два разных типа

    def test_name_mismatch(self):
        self.assertIsNone(app.name_mismatch("RC ФНЧ: R 1к, C 100н", "LP"))
        self.assertIsNone(app.name_mismatch("R1 1к вх-вых; C1 100н вых-0", "LP"))  # тип не назван
        note = app.name_mismatch("RL ФВЧ: L от входа к выходу, R на землю", "LP")
        self.assertEqual(note, "в запросе назван ФВЧ, а по расчёту MC12 получился ФНЧ: схема построена "
                               "по описанным соединениям - проверьте их и добавленные элементы, если они есть")
        self.assertIn("без частотной зависимости", app.name_mismatch("ФНЧ из двух резисторов", "flat"))

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

    def test_name_mismatch_note(self):
        """Название «ФВЧ», а соединения дают ФНЧ (как у запросов Kimi): схема строится по соединениям, замечание есть."""
        class Fixed:
            output_format = "netlist"

            def translate(self, prompt):
                return "V1 IN 0 DC 0\nL1 IN OUT 10m\nR1 OUT 0 1k\n"
        rep = app.run("RL ФВЧ: катушка 10 мГн от входа к выходу, резистор 1 кОм с выхода на землю", Fixed(),
                      outdir=ROOT / "out" / "test_app", name="rl_name")
        self.assertEqual(rep["measured"]["type"], "LP")
        self.assertTrue(any("в запросе назван ФВЧ" in n for n in rep["notes"]), rep["notes"])
        rep = app.run("RL ФНЧ: катушка 10 мГн от входа к выходу, резистор 1 кОм с выхода на землю", Fixed(),
                      outdir=ROOT / "out" / "test_app", name="rl_ok")
        self.assertFalse(any("в запросе назван" in n for n in rep["notes"]), rep["notes"])

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
