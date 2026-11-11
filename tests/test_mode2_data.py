"""Примеры режима 2 для датасета v3 и инструкции модели (docs/mode2_spec.md, разделы 5 и 7).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import bench, canon, iso, mc12run, measure, mode2_data, synth, training  # noqa: E402


def elements(net):
    return canon.parse_netlist(net)[0]


def batch(kind, n, seed):
    rng = random.Random(seed)
    return [mode2_data.sample(rng, kind) for _ in range(n)]


class SpecLine(unittest.TestCase):
    def test_order_and_numbers(self):
        spec = {"K": 10, "active": "yes", "Q": 0.707, "order": 4, "fc": 1500.0, "type": "LP", "response": "Butterworth"}
        self.assertEqual(mode2_data.spec_line(spec), ".SPEC type=LP fc=1.5k order=4 response=Butterworth Q=0.707 active=yes K=10")
        self.assertEqual(mode2_data.spec_line({"type": "BP", "f0": 50.0, "bw": 10.0, "series": "exact"}),
                         ".SPEC type=BP f0=50 bw=10 series=exact")

    def test_examples_of_spec(self):
        """Строки из docs/mode2_spec.md (раздел 7а) записываются так же и решаются."""
        for spec, line in (({"type": "LP", "fc": 1e3, "order": 4, "active": "yes"}, ".SPEC type=LP fc=1k order=4 active=yes"),
                           ({"type": "BP", "f0": 5e3, "Q": 5, "active": "yes"}, ".SPEC type=BP f0=5k Q=5 active=yes"),
                           ({"type": "LP", "fc": 1e4, "order": 5, "response": "Chebyshev", "ripple": 0.5},
                            ".SPEC type=LP fc=10k order=5 response=Chebyshev ripple=0.5")):
            self.assertEqual(mode2_data.spec_line(spec), line)
            self.assertIsNotNone(mode2_data.solve(line))


class HideInText(unittest.TestCase):
    def test_every_notation(self):
        for text in ("R1 = 7,7717 кОм между", "резистор 7.7717 килоом к", "C 7771,7 пФ на", "7,7717к от", "7.7717н: вход",
                     "L1 7,7717мГн", "катушка 7771.7 мкГн к точке", "ёмкостью 7,7717 нФ."):
            out = mode2_data._hide_in_text(text, 1, "?")
            self.assertIn("?", out)
            self.assertNotRegex(out, r"7[.,]?77")
        self.assertEqual(mode2_data._hide_in_text("7,7717к от входа к узлу A", 1, "?"), "? от входа к узлу A")
        self.assertEqual(mode2_data._hide_in_text("R1 7,7717 кОм к точке 1", 1, "подобрать"), "R1 подобрать к точке 1")

    def test_missing_or_repeated_fails(self):
        with self.assertRaises(ValueError):
            mode2_data._hide_in_text("R1 10к", 1, "?")
        with self.assertRaises(ValueError):
            mode2_data._hide_in_text("7,7717к и 7,7717к", 1, "?")


class Samples(unittest.TestCase):
    def check_solved(self, rec):
        """Ответ решается через synth.run и попадает в названные цели."""
        rep = mode2_data.solve(rec["output"])
        self.assertIsNotNone(rep, rec["output"])
        if rep.get("route") != "designer":
            for k, d in rep["deviation"].items():
                self.assertLessEqual(abs(d), mode2_data.TOL["db"] if k == "k_pass_db" else mode2_data.TOL["rel"], rec["output"])
            self.assertNotIn("?", rep["netlist"])
        return rep

    def test_spec(self):
        routes = set()
        for rec in batch("spec", 80, 1):
            self.assertRegex(rec["output"], r"^\.SPEC type=(LP|HP|BP|BS) (fc|f0)=\S+")
            self.assertEqual(rec["output"].count("\n"), 0)
            elems, unknown, raw, errors = synth.parse(rec["output"])
            self.assertEqual((elems, unknown, errors), ([], [], []))
            self.assertEqual(set(raw), set(rec["spec"]))  # в ответе только названное в запросе
            self.assertRegex(rec["prompt"], r"\d ?(Гц|кГц|МГц)")
            self.assertEqual("active" in rec["spec"], bool(re.search(r"активн|на ОУ|операционн", rec["prompt"])), rec["prompt"])
            self.assertEqual("order" in rec["spec"], "порядка" in rec["prompt"], rec["prompt"])
            routes.add(self.check_solved(rec)["route"])
        self.assertLessEqual({"single", "cascade"}, routes)

    def test_named(self):
        families = set()
        for rec in batch("named", 60, 2):
            families.add(rec["family"])
            lines = rec["output"].splitlines()
            self.assertTrue(lines[-1].startswith(".SPEC type="))
            hidden = [l.split()[0] for l in lines[:-1] if l.endswith(" ?")]
            self.assertEqual(sorted(hidden), sorted(rec["unknown"]))
            self.assertTrue(hidden)
            self.assertNotRegex(rec["prompt"], r"7[.,]77\d7|777\d[.,]7")  # контрольные номиналы в запрос не попали
            rep = self.check_solved(rec)
            self.assertEqual(rep["route"], "formula")
            self.assertEqual(rep["topology"], rec["topology"])
            # схема та же, заданные номиналы - как у исходной схемы
            ref, sol = elements(rec["reference"]), elements(rep["netlist"])
            self.assertTrue(iso.match(ref, sol, "topology")["match"])
            given = sorted(l.split()[-1] for l in lines[1:-1] if l[0] in "RCL" and not l.endswith(" ?"))
            for v in given:
                self.assertIn(v, [e["value"] for e in ref])
            if rec["family"] == "named_all":
                self.assertFalse(given)
                self.assertNotRegex(rec["prompt"].split(";")[0], r"\d (кОм|нФ|мГн)")
        self.assertEqual(families, {"named_all", "named_brief", "named_roles"})

    def test_circuit(self):
        routes, families = set(), set()
        for rec in batch("circuit", 30, 3):
            families.add(rec["family"])
            lines = rec["output"].splitlines()
            self.assertEqual(sum(l.endswith(" ?") for l in lines), len(rec["hidden"]))
            self.assertIn(len(rec["hidden"]), (1, 2))
            self.assertNotRegex(rec["prompt"], r"7[.,]77\d7|777\d[.,]7")
            self.assertTrue(any(w in rec["prompt"] for w in mode2_data.TYPE_WORDS[rec["spec"]["type"]]), rec["prompt"])  # тип назван
            if rec["spec"]["type"] in ("BP", "BS"):
                self.assertIn("Q", rec["spec"])
            rep = self.check_solved(rec)
            routes.add(rep["route"])
            # остальные номиналы и соединения - как у исходной схемы
            self.assertTrue(iso.match(elements(rec["reference"]), elements(rep["netlist"]), "topology")["match"])
        self.assertIn("numeric", routes)
        self.assertGreaterEqual(len(families), 4)

    def test_refuse(self):
        codes = set()
        for rec in batch("refuse", 40, 4):
            self.assertRegex(rec["output"], r"^REFUSE (ambiguous|out_of_class): \S")
            codes.add(rec["refuse"])
            if rec["refuse"] == "ambiguous":
                self.assertEqual(rec["output"], "REFUSE ambiguous: не задана частота")
                self.assertNotRegex(rec["prompt"], r"Гц")
        self.assertEqual(codes, {"ambiguous", "out_of_class"})

    def test_mix(self):
        rng = random.Random(5)
        kinds = [mode2_data.sample(rng)["kind"] for _ in range(120)]
        for kind, share in mode2_data.MIX.items():
            self.assertAlmostEqual(kinds.count(kind) / 120, share, delta=0.12, msg=kind)


class Instructions(unittest.TestCase):
    def test_v2_instruction_unchanged(self):
        self.assertEqual(training.messages("RC")[0]["content"], training.SYSTEM["netlist"])
        self.assertIn("N1, N2, ... - внутренние.", training.SYSTEM["netlist"])
        self.assertEqual(training.system_for({"gen": {"version": "gen-2"}}), training.SYSTEM["netlist"])
        self.assertEqual(training.system_for({}, "ir"), training.SYSTEM["ir"])

    def test_v3_and_mode2(self):
        v3 = training.system_for({"gen": {"version": "gen-3"}})
        self.assertIn("узел 3 - N3", v3)
        self.assertNotEqual(v3, training.SYSTEM["netlist"])
        m2 = training.system_for({"mode": 2, "gen": {"version": "gen-3"}})
        self.assertIn(".SPEC type=", m2)
        self.assertIn("?", m2)
        with self.assertRaises(ValueError):
            training.system_for({"mode": 2}, "ir")
        rec = batch("named", 1, 6)[0]
        msgs = training.messages(rec["prompt"], training.answer_for(rec), system=training.system_for(rec))
        self.assertEqual(msgs[0]["content"], m2)
        self.assertEqual(msgs[2]["content"], rec["output"].strip())
        ref = batch("refuse", 1, 7)[0]
        self.assertTrue(training.answer_for(ref).startswith("REFUSE "))

    def test_instruction_names_every_field(self):
        for field in mode2_data.FIELDS:
            self.assertIn(f"{field}=", training.SYSTEM_V3["mode2"], field)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Mc12(unittest.TestCase):
    def test_solutions_run(self):
        """Подобранные схемы считаются в MC12; срез и f0 совпадают с узловым расчётом synth."""
        recs = batch("spec", 8, 8) + batch("named", 8, 9) + batch("circuit", 8, 10)
        cases = []
        for i, rec in enumerate(recs):
            sol = rec["solution"]
            if sol["route"] == "designer":
                continue
            f = rec["spec"].get("fc") or rec["spec"].get("f0")
            cases.append((f"m{i:02d}", elements(sol["netlist"]), f, sol))
        runs = bench.run([c[:3] for c in cases], ROOT / "out" / "test_mode2_data")
        for name, _el, _f, sol in cases:
            with self.subTest(name, route=sol["route"]):
                self.assertTrue(runs[name]["ok"], runs[name]["messages"])
                got = measure.from_output(runs[name]["output"])
                self.assertEqual(got["type"], sol["achieved"]["type"])
                for k in ("fc", "f0"):
                    if sol["achieved"].get(k) and k in sol["wanted"]:
                        self.assertAlmostEqual(got[k] / sol["achieved"][k], 1, delta=0.01)


if __name__ == "__main__":
    unittest.main()
