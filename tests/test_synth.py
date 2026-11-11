"""Режим 2: подбор номиналов по характеристикам (src/mcnl/synth.py, docs/mode2_spec.md)."""
import cmath
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import bench, canon, mc12run, measure, mna, synth  # noqa: E402

# спецификация -> ожидаемая схема (раздел 3 docs/mode2_spec.md)
DEFAULTS = [
    (".SPEC type=LP fc=1k", "rc_lp"),
    (".SPEC type=HP fc=1k", "rc_hp"),
    (".SPEC type=LP fc=1k active=yes", "active_lp_inv"),
    (".SPEC type=HP fc=500 active=yes K=3", "active_hp_inv"),
    (".SPEC type=LP fc=1k response=Butterworth", "rlc_series_lp"),
    (".SPEC type=HP fc=2k order=2", "rlc_series_hp"),
    (".SPEC type=LP fc=1k order=2 active=yes", "sallen_key_lp"),
    (".SPEC type=HP fc=2k order=2 active=yes", "sallen_key_hp"),
    (".SPEC type=LP fc=1k order=2 active=yes K=10", "mfb_lp"),
    (".SPEC type=HP fc=100 order=2 active=yes K=5", "mfb_hp"),
    (".SPEC type=BP f0=5k", "rlc_series_bp"),
    (".SPEC type=BP f0=5k Q=5 active=yes", "mfb_bp"),
    (".SPEC type=BS f0=1k", "rlc_series_bs"),
    (".SPEC type=BS f0=1k Q=2 active=yes", "twin_t_active"),
]
CASCADES = [".SPEC type=LP fc=1k order=4 active=yes", ".SPEC type=HP fc=300 order=3 active=yes",
            ".SPEC type=LP fc=2k order=5 response=Chebyshev active=yes"]
LADDERS = [".SPEC type=LP fc=10k order=3", ".SPEC type=LP fc=10k order=5 response=Chebyshev ripple=0.5"]


def check_dev(test, rep, f_tol, q_tol, k_tol):
    for k, v in rep["deviation"].items():
        tol = {"fc": f_tol, "f0": f_tol, "Q": q_tol, "k_pass_db": k_tol}[k]
        test.assertLessEqual(abs(v), tol, f"{k}: {rep}")


class Spec(unittest.TestCase):
    def test_defaults(self):
        s, err = synth.resolve_spec({"type": "lp", "fc": "1k"})
        self.assertEqual(err, [])
        self.assertEqual((s["type"], s["f"], s["order"], s["active"], s["K"]), ("LP", 1000.0, 1, False, 1.0))
        s, _ = synth.resolve_spec({"type": "LP", "fc": "1k", "response": "butterworth"})
        self.assertEqual(s["order"], 2)  # названа аппроксимация - 2-й порядок
        s, _ = synth.resolve_spec({"type": "BP", "f0": "5k"})
        self.assertEqual((s["order"], s["Q"]), (2, 1.0))  # Q = 1 по умолчанию (решение 11.11.2026)
        s, _ = synth.resolve_spec({"type": "BS", "f0": "1k", "bw": "250"})
        self.assertAlmostEqual(s["Q"], 4.0)

    def test_errors(self):
        self.assertTrue(synth.resolve_spec({"fc": "1k"})[1])
        self.assertTrue(synth.resolve_spec({"type": "LP"})[1])
        self.assertIn("error", synth.run(".SPEC fc=1k"))
        self.assertIn("error", synth.run(".SPEC type=HP fc=1k order=3"))  # пассивный ФВЧ 3-го порядка не поддержан
        self.assertEqual(synth.run(".SPEC type=BP f0=1k order=4")["route"], "designer")

    def test_parse_unknowns(self):
        elems, unknown, spec, err = synth.parse("V1 IN 0 DC 0\nR1 IN OUT ?\nC1 OUT 0 10n\n.SPEC type=LP fc=1k")
        self.assertEqual((unknown, spec, err), (["R1"], {"type": "LP", "fc": "1k"}, []))
        self.assertEqual(len(elems), 3)


class Sections(unittest.TestCase):
    def test_butterworth(self):
        pairs, real = synth.sections({"order": 4, "f": 1.0, "type": "LP", "response": "Butterworth"})
        self.assertEqual([round(q, 3) for _, q in pairs], [1.307, 0.541])
        self.assertIsNone(real)
        pairs, real = synth.sections({"order": 3, "f": 1.0, "type": "LP", "response": "Butterworth"})
        self.assertEqual((round(pairs[0][1], 3), real), (1.0, 1.0))

    def test_chebyshev_2nd_order_1db(self):
        (f0, q), = synth.sections({"order": 2, "f": 1.0, "type": "LP", "response": "Chebyshev", "ripple": 1.0})[0]
        self.assertAlmostEqual(q, 0.9565, places=3)   # табличные значения: Q 0,957, f0 1,050
        self.assertAlmostEqual(f0, 1.0500, places=3)


class Designs(unittest.TestCase):
    def test_default_topologies(self):
        for text, top in DEFAULTS:
            rep = synth.run(text)
            self.assertEqual(rep.get("topology"), top, f"{text}: {rep}")

    def test_exact_values_hit_targets(self):
        for text, _ in DEFAULTS:
            rep = synth.run(text + " series=exact")
            check_dev(self, rep, 0.002, 0.005, 0.02)

    def test_series_values(self):
        for text, _ in DEFAULTS + [(c, None) for c in CASCADES + LADDERS]:
            rep = synth.run(text)
            check_dev(self, rep, 0.03, 0.05, 0.5)
            for e in canon.parse_netlist(rep["netlist"])[0]:
                if e["type"] in "RCL" and not rep["topology"].startswith("twin_t"):  # у Т-моста R/2, 2C
                    v = canon.parse_value(e["value"])
                    self.assertIn(round(v, 15), [round(x, 15) for x in synth.series(e["type"])], f"{text}: {e}")
                    lo, hi = synth.RANGES[e["type"]]
                    self.assertTrue(lo <= v <= hi)

    def test_twin_t_active_q_formula(self):
        """Q = 1/(4(1 - k)) активного двойного Т-моста - сверка с узловым расчётом."""
        for rk2 in (10e3, 30e3, 90e3):
            net, exp = synth.twin_t_active(10e3, 10e-9, 10e3, rk2)
            got = synth.achieved(net, exp["f0"])
            self.assertAlmostEqual(got["f0"] / exp["f0"], 1, delta=0.002)
            self.assertAlmostEqual(got["Q"] / exp["Q"], 1, delta=0.01)

    def test_chebyshev_ripple_edge(self):
        """Чебышёв: на заданной частоте |H| = -ripple (нечётный порядок) или 0 дБ (чётный) от уровня на 0 Гц."""
        for order, expect in ((5, -1.0), (4, 0.0)):
            rep = synth.run(f".SPEC type=LP fc=1k order={order} response=Chebyshev active=yes series=exact")
            elems = canon.parse_netlist(rep["netlist"])[0]
            h0, hc = abs(mna.response(elems, 1.0)), abs(mna.response(elems, 1000.0))
            self.assertAlmostEqual(20 * math.log10(hc / h0), expect, delta=0.02)


class UserNetlist(unittest.TestCase):
    SK = "V1 IN 0 DC 0\nR1 IN N1 ?\nR2 N1 N2 ?\nC1 N1 OUT {c1}\nC2 N2 0 {c2}\nX1 N2 OUT OUT OPAMP_IDEAL\n"

    def test_known_topology_with_given_caps(self):
        rep = synth.run(self.SK.format(c1="22n", c2="10n") + ".SPEC type=LP fc=1k Q=0.707")
        self.assertEqual((rep["route"], rep["topology"]), ("formula", "sallen_key_lp"))
        net = canon.parse_netlist(rep["netlist"])[0]
        self.assertEqual({e["name"]: e["value"] for e in net if e["type"] == "C"}, {"C1": "22n", "C2": "10n"})
        check_dev(self, rep, 0.03, 0.05, 0.5)

    def test_infeasible_given_values(self):
        rep = synth.run(self.SK.format(c1="10n", c2="10n") + ".SPEC type=LP fc=1k Q=0.707")  # при C1 = C2 Q <= 0,5
        self.assertIn("заданными номиналами", rep["error"])

    def test_twin_t_fills_derived_elements(self):
        rep = synth.run("V1 IN 0 DC 0\nR1 IN N1 ?\nR2 N1 OUT ?\nC3 N1 0 ?\nC1 IN N2 ?\nC2 N2 OUT ?\nR3 N2 0 ?\n"
                        ".SPEC type=BS f0=50")
        self.assertEqual(rep["topology"], "twin_t")
        v = {e["name"]: canon.parse_value(e["value"]) for e in canon.parse_netlist(rep["netlist"])[0] if e["type"] in "RC"}
        self.assertAlmostEqual(v["R3"], v["R1"] / 2)
        self.assertAlmostEqual(v["C3"], 2 * v["C1"])
        self.assertLess(abs(rep["deviation"]["f0"]), 0.03)
        self.assertNotIn("Q", rep["wanted"])

    def test_default_targets_dropped_when_values_given(self):
        """Заданные R1, R3, C определяют Q и K полосового МОС: подбирается только R2 под f0."""
        rep = synth.run("V1 IN 0 DC 0\nR1 IN N1 620\nR2 N1 0 ?\nR3 N2 OUT 18k\nC1 N1 OUT 47n\nC2 N1 N2 47n\n"
                        "X1 0 N2 OUT OPAMP_IDEAL\n.SPEC type=BP f0=2k")
        self.assertEqual(set(rep["wanted"]), {"type", "f0"})
        self.assertLess(abs(rep["deviation"]["f0"]), 0.03)

    def test_numeric_outside_table(self):
        rep = synth.run("V1 IN 0 DC 0\nR1 IN N1 1k\nC1 N1 0 ?\nR2 N1 OUT 10k\nC2 OUT 0 ?\n.SPEC type=LP fc=1k")
        self.assertEqual(rep["route"], "numeric")
        self.assertLess(abs(rep["deviation"]["fc"]), 0.06)

    def test_check_only(self):
        rep = synth.run("V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n.SPEC type=LP fc=1.6k")
        self.assertEqual(rep["route"], "check")
        self.assertLess(abs(rep["deviation"]["fc"]), 0.01)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class InMC12(unittest.TestCase):
    """Подобранные схемы (с номиналами из ряда) в MC12 дают то же, что узловой расчёт."""

    @classmethod
    def setUpClass(cls):
        cls.reps = {}
        cases = []
        for i, text in enumerate([t for t, _ in DEFAULTS] + CASCADES + LADDERS):
            rep = synth.run(text)
            name = f"s{i:02d}"
            cls.reps[name] = rep
            cases.append((name, canon.parse_netlist(rep["netlist"])[0], rep["spec"]["f"]))
        cls.res = bench.run(cases, ROOT / "out" / "test_synth")

    def test_mc12_matches_nodal(self):
        for name, rep in self.reps.items():
            r = self.res[name]
            self.assertTrue(r["ok"], r)
            got = measure.from_output(r["output"])
            for k in ("fc", "f0", "Q"):
                a, b = got.get(k), rep["achieved"].get(k)
                if a and b and not (k != "fc" and rep["spec"]["type"] in ("LP", "HP")):
                    self.assertAlmostEqual(a / b, 1, delta=0.005, msg=f"{name} {k}: {rep['netlist']}")


if __name__ == "__main__":
    unittest.main()
