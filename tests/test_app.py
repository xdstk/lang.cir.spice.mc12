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

from mcnl import app, canon, evaluate, iso, mc12run, measure, training, translators  # noqa: E402

TEST = [json.loads(l) for l in (ROOT / "data" / "dataset" / "v1" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
RC = "V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n"
SK_HP = "V1 IN 0 DC 0\nC1 IN N1 10n\nC2 N1 N2 10n\nR1 N1 OUT 1.1k\nR2 N2 0 2.2k\nX1 N2 OUT OUT OPAMP_IDEAL\n"


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

    def test_fchar_of_integrator_and_differentiator(self):
        # полосы нет: рабочая точка - частота единичного усиления 1/(2 pi R C), как в датасете (раньше - край сетки)
        f_u = 1 / (2 * math.pi * 10e3 * 100e-9)
        for net in ("V1 IN 0 DC 0\nR1 IN N1 10k\nC1 N1 OUT 100n\nX1 0 N1 OUT OPAMP_IDEAL\n",
                    "V1 IN 0 DC 0\nC1 IN N1 100n\nR1 N1 OUT 10k\nX1 0 N1 OUT OPAMP_IDEAL\n"):
            with self.subTest(net=net):
                self.assertAlmostEqual(app.characteristic_frequency(canon.parse_netlist(net)[0]) / f_u, 1, delta=0.02)

    def test_card_band(self):
        self.assertEqual(app.card_band({"type": "HP", "fc": 1e3}), 1e4)
        self.assertAlmostEqual(app.card_band({"type": "BP", "f0": 1e3, "Q": 1.0}), 1e4 * (math.sqrt(1.25) + 0.5))
        self.assertIsNone(app.card_band({"type": "flat"}))
        self.assertIsNone(app.card_band(None))

    def test_mode2_answer_detection(self):
        self.assertTrue(app.is_mode2_answer(".SPEC type=LP fc=1k"))
        self.assertTrue(app.is_mode2_answer("V1 IN 0 DC 0\nR1 IN OUT ?\nC1 OUT 0 10n\n.SPEC type=LP fc=1k"))
        self.assertFalse(app.is_mode2_answer(RC))

    def test_mode2_offline(self):
        """Режим 2 на эталонных ответах test v3 (как от модели): все виды дают схему с номиналами, отказы - отказ,
        ПФ выше 2-го порядка - сообщение о дизайнере MC12 (в программу не подключён)."""
        recs = [json.loads(l) for l in (ROOT / "data" / "dataset" / "v3" / "test.jsonl").read_text(encoding="utf-8")
                .splitlines()]
        m2 = [r for r in recs if r.get("mode") == 2]
        echo = translators.load("echo", answers={r["prompt"]: training.answer_for(r) for r in m2})
        got = {}
        for r in m2:
            rep = app.run(r["prompt"], echo, outdir=ROOT / "out" / "test_app", name="m2", mc12=False, mode=2)
            kind = "refused" if rep.get("refused") else "netlist" if rep.get("netlist") else "message"
            got.setdefault((r["kind"], kind), 0)
            got[(r["kind"], kind)] += 1
            if kind == "netlist":
                self.assertNotIn("?", rep["netlist"])
                self.assertEqual(rep["output"], training.answer_for(r))  # в окне правится ответ модели
                self.assertTrue(rep["notes"][0].startswith(("подбор номиналов", "схему выбрала программа")), rep["notes"])
                self.assertIn("задано -> получено (узловой расчёт)", rep["notes"][1])
                if rep["notes"][0].startswith("схему выбрала программа"):  # имена - как на чертеже (17.11.2026)
                    body = "; ".join(l for l in rep["netlist"].splitlines() if l.strip() and not l.startswith("V"))
                    self.assertTrue(rep["notes"][0].endswith(body), (rep["notes"][0], rep["netlist"]))
            if kind == "message":
                self.assertIn("дизайнер", rep["messages"][0])
        self.assertEqual(got, {("circuit", "netlist"): 24, ("named", "netlist"): 25, ("spec", "netlist"): 30,
                               ("spec", "message"): 1, ("refuse", "refused"): 10})

    def test_mode2_messages(self):
        out = ROOT / "out" / "test_app"
        rep = app.run("ФНЧ на 1 кГц", translators.load("baseline"), outdir=out, mc12=False, mode=2)
        self.assertIn("не умеет режим 2", rep["messages"][0])
        rep = app.run(netlist=".SPEC type=LP", outdir=out, name="m2_err", mc12=False, mode=2)
        self.assertTrue(rep["messages"][0].startswith("номиналы не подобраны"), rep["messages"])
        # пересчёт правленого ответа: .SPEC с другой частотой - другие номиналы
        rep = app.run(netlist=".SPEC type=LP fc=2k", outdir=out, name="m2_edit", mc12=False, mode=2)
        self.assertEqual(rep["synth"]["topology"], "rc_lp")
        self.assertIn("fc 2 кГц -> 2", rep["notes"][1])
        self.assertFalse(any("схема из ответа модели" in n for n in rep["notes"]), rep["notes"])
        # ответ m2 v3.1 на «RL-фильтр верхних частот на 16 кГц» (program-009): RL ФНЧ при type=HP - замечание
        rep = app.run(netlist="V1 IN 0 DC 0\nR1 OUT 0 ?\nL1 IN OUT 10m\n.SPEC type=HP fc=16k", outdir=out,
                      name="m2_type", mc12=False, mode=2)
        self.assertTrue(any("задан ФВЧ, а схема из ответа модели - ФНЧ" in n for n in rep["notes"]), rep["notes"])
        # нетлист без ? и .SPEC в режиме 2 - обычная схема
        rep = app.run(netlist=RC, outdir=out, name="m2_plain", mc12=False, mode=2)
        self.assertNotIn("synth", rep)
        self.assertEqual(rep["netlist"], RC)
        # режим 1, номиналы не заданы: подсказка про режим 2
        class Refuse:
            output_format = "netlist"

            def translate(self, prompt):
                return "REFUSE needs_design: номиналы не заданы"
        rep = app.run("ФНЧ на 1 кГц", Refuse(), outdir=out, mc12=False)
        self.assertIn("режима 2", rep["messages"][-1])

    def test_hf_adapter_slots(self):
        # без torch: какой адаптер и какая инструкция у каждого режима
        from mcnl.translators import hf_model
        self.assertEqual(hf_model.slots("a1"), {1: ("default", "v2")})  # модель на v2: режима 2 нет
        self.assertEqual(hf_model.slots("a1", "v3"), {1: ("default", "v3"), 2: ("default", "v3")})  # общая v3
        self.assertEqual(hf_model.slots("a1", "v2", "a2"), {1: ("default", "v2"), 2: ("mode2", "v3")})
        self.assertEqual(hf_model.slots("a1", "v3", "a1"), {1: ("default", "v3"), 2: ("default", "v3")})

    def test_real_opamp_offline(self):
        out = ROOT / "out" / "test_app"
        rep = app.run(netlist=SK_HP, outdir=out, name="sk_real_off", mc12=False, opamp=app.OPAMP_REAL)
        self.assertEqual(rep["opamp"], "TL071C")
        self.assertIn("V=TL071C", Path(rep["cir"]).read_text(encoding="ascii"))
        # усиление 20 при DC 1 В: выход по постоянному току +20 В - чертёж с идеальным ОУ и замечание
        rep = app.run(netlist="V1 IN 0 DC 1\nR1 N1 0 1k\nR2 N1 OUT 19k\nX1 IN N1 OUT OPAMP_IDEAL\n", outdir=out,
                      name="sat_off", mc12=False, opamp=app.OPAMP_REAL)
        self.assertIsNone(rep["opamp"])
        self.assertNotIn("TL071C", Path(rep["cir"]).read_text(encoding="ascii"))
        self.assertTrue(any("около +20 В" in n for n in rep["notes"]), rep["notes"])
        rep = app.run(netlist="V1 IN 0 DC 0\nR1 IN N1 10k\nC1 N1 OUT 100n\nX1 0 N1 OUT OPAMP_IDEAL\n", outdir=out,
                      name="int_off", mc12=False, opamp=app.OPAMP_REAL)
        self.assertTrue(any("нет обратной связи по постоянному току" in n for n in rep["notes"]), rep["notes"])
        rep = app.run(netlist=RC, outdir=out, name="rc_real_off", mc12=False, opamp=app.OPAMP_REAL)
        self.assertEqual((rep["opamp"], rep["notes"]), (None, []))  # без ОУ - ни питания, ни замечаний


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

    def test_real_opamp_pipeline(self):
        """Чертёж с TL071C: один запуск MC12 считает обвязку (идеальный ОУ) и сам чертёж; карточки - по чертежу до
        10 fc, где ФВЧ ещё не упирается в полосу ОУ (без границы fc мерился бы по краю диапазона: 2,75 кГц)."""
        rep = app.run(netlist=SK_HP, outdir=ROOT / "out" / "test_app", name="sk_real", opamp=app.OPAMP_REAL)
        self.assertTrue(rep["run"]["ok"] and rep["run_real"]["ok"], rep)
        fc = 1 / (2 * math.pi * math.sqrt(1.1e3 * 2.2e3) * 10e-9)
        self.assertEqual((rep["measured"]["type"], rep["measured_ideal"]["type"]), ("HP", "HP"))
        self.assertAlmostEqual(rep["measured"]["fc"] / fc, 1, delta=0.01)
        self.assertAlmostEqual(rep["measured"]["Q"], 0.707, delta=0.01)
        self.assertEqual(rep["notes"], [])
        f, db, _ = measure.read_curve(rep["run_real"]["output"])
        self.assertLess(db[-1], -10)  # у 10 МГц полоса TL071 уже кончилась - это видно на графике
        # дифференциатор с реальным ОУ звенит (подъём около sqrt(f_u * GBW)): другой вид АЧХ - замечание
        rep = app.run(netlist="V1 IN 0 DC 0\nC1 IN N1 100n\nR1 N1 OUT 10k\nX1 0 N1 OUT OPAMP_IDEAL\n",
                      outdir=ROOT / "out" / "test_app", name="diff_real", opamp=app.OPAMP_REAL)
        self.assertEqual(rep["measured"]["type"], "BP")
        self.assertTrue(any("сказывается конечная полоса" in n for n in rep["notes"]), rep["notes"])

    def test_mode2_pipeline(self):
        """Режим 2 целиком: подобранная схема считается в MC12 (обвязка и чертёж с TL071C одним запуском) и попадает
        в цели с допуском оценки режима 2 (score2: частота и Q - 10%, усиление - 1 дБ)."""
        recs = [json.loads(l) for l in (ROOT / "data" / "dataset" / "v3" / "test.jsonl").read_text(encoding="utf-8")
                .splitlines()]
        picked, seen = [], set()
        for r in recs:  # по одной записи каждого вида и с ОУ, и без
            if r.get("mode") == 2 and not r.get("refuse"):
                key = (r["kind"], "OPAMP" in training.answer_for(r) or "active=yes" in training.answer_for(r))
                if key not in seen:
                    seen.add(key)
                    picked.append(r)
        self.assertGreaterEqual(len(picked), 4)
        for i, r in enumerate(picked):
            with self.subTest(rec=r["id"]):
                rep = app.run(netlist=training.answer_for(r), outdir=ROOT / "out" / "test_app", name=f"m2_{i}",
                              opamp=app.OPAMP_REAL, mode=2)
                if rep["synth"].get("route") == "designer":
                    continue
                self.assertTrue(rep["run"]["ok"], rep)
                got = rep.get("measured_ideal") or rep["measured"]
                for k, want in rep["synth"]["wanted"].items():
                    if k == "type" or want is None:
                        continue
                    if k == "k_pass_db":
                        self.assertLess(abs(got[k] - want), 1.0, (k, got[k], want))
                    else:
                        self.assertLess(abs(got[k] / want - 1), 0.1, (k, got[k], want))
                self.assertIn("получено (MC12)", rep["notes"][1])

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
