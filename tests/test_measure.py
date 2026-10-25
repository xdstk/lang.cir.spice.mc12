"""Тесты измерений раздела 6 eval_spec на аналитических АЧХ/ФЧХ.

Отклик считается по передаточной функции на сетке обвязки (100 точек на декаду, f_хар/1000 ...
f_хар*1000) и округляется до 4 значащих цифр, как в выводе MC12. Ответ известен аналитически.
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import cmath
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import measure  # noqa: E402

OUT = ROOT / "tests" / "data" / "mc12_out"


def sig4(x):
    return float(f"{x:.4g}")


def response(H, f_char):
    f = [f_char / 1000 * 10 ** (k / 100) for k in range(601)]
    db = [sig4(20 * math.log10(abs(H(2j * math.pi * x)))) for x in f]
    ph = [sig4(math.degrees(cmath.phase(H(2j * math.pi * x)))) for x in f]
    return [sig4(x) for x in f], db, ph


def fc_second_order(f0, Q):
    """Частота |H| = k/sqrt(2) для ФНЧ 2-го порядка (для ФВЧ - f0^2 / это значение)."""
    a = 1 - 1 / (2 * Q * Q)
    return f0 * math.sqrt(a + math.sqrt(a * a + 1))


class Analytic(unittest.TestCase):
    def assertRel(self, a, b, tol=1e-3, msg=None):
        self.assertIsNotNone(a, msg)
        self.assertAlmostEqual(a / b, 1, delta=tol, msg=msg)

    def test_lp1(self):
        fc, K = 1591.55, 3.0
        m = measure.measure(*response(lambda s: K / (1 + s / (2 * math.pi * fc)), fc))
        self.assertEqual(m["type"], "LP")
        self.assertRel(m["fc"], fc)
        self.assertAlmostEqual(m["k_pass_db"], 20 * math.log10(K), delta=0.01)
        self.assertIsNone(m["f0"])  # 1-й порядок: фаза не доходит до -90
        self.assertIsNone(m["Q"])

    def test_hp1_inverting(self):
        fc, K = 200.0, -10.0
        m = measure.measure(*response(lambda s: K * (s / (2 * math.pi * fc)) / (1 + s / (2 * math.pi * fc)), fc))
        self.assertEqual(m["type"], "HP")
        self.assertRel(m["fc"], fc)
        self.assertAlmostEqual(m["k_pass_db"], 20.0, delta=0.01)

    def test_lp2_peaked_and_flat(self):
        for Q in (0.5, 0.7071, 2.0, 5.0):
            f0 = 1e4
            w0 = 2 * math.pi * f0
            m = measure.measure(*response(lambda s: w0 ** 2 / (s * s + s * w0 / Q + w0 ** 2), f0))
            self.assertEqual(m["type"], "LP", Q)
            self.assertRel(m["f0"], f0, msg=Q)
            self.assertRel(m["Q"], Q, tol=5e-3, msg=Q)
            self.assertRel(m["fc"], fc_second_order(f0, Q), msg=Q)  # от уровня полосы, не от максимума

    def test_hp2_gain2(self):
        f0, Q, K = 9895.3, 0.6, 2.0
        w0 = 2 * math.pi * f0
        m = measure.measure(*response(lambda s: K * s * s / (s * s + s * w0 / Q + w0 ** 2), f0))
        self.assertEqual(m["type"], "HP")
        self.assertRel(m["f0"], f0)
        self.assertRel(m["Q"], Q, tol=5e-3)
        self.assertRel(m["fc"], f0 ** 2 / fc_second_order(f0, Q))
        self.assertAlmostEqual(m["k_pass_db"], 20 * math.log10(K), delta=0.01)

    def test_bp(self):
        for Q in (0.5, 2.0, 10.0):
            f0, K = 1000.0, 4.0
            w0 = 2 * math.pi * f0
            m = measure.measure(*response(lambda s: K * s * w0 / Q / (s * s + s * w0 / Q + w0 ** 2), f0))
            self.assertEqual(m["type"], "BP", Q)
            self.assertRel(m["f0"], f0, msg=Q)
            self.assertRel(m["Q"], Q, tol=5e-3, msg=Q)
            self.assertAlmostEqual(m["k_pass_db"], 20 * math.log10(K), delta=0.01)

    def test_asymmetric_band(self):
        # ФВЧ 2-го порядка на 100 Гц и ФНЧ 1-го порядка на 10 кГц: края затухают на ~80 и ~40 дБ,
        # правило «разница краёв > 10 дБ» дало бы HP - полосовой должен определяться раньше
        wh, wl = 2 * math.pi * 100, 2 * math.pi * 1e4
        H = lambda s: (s * s / (s * s + s * wh / 0.7071 + wh * wh)) / (1 + s / wl)
        m = measure.measure(*response(H, 1e3))
        self.assertEqual(m["type"], "BP")
        self.assertIsNotNone(m["Q"])

    def test_bs(self):
        f0, Q = 50.0, 3.0
        w0 = 2 * math.pi * f0
        # сетка сдвинута относительно f0: точный ноль в узле сетки дал бы log(0), MC12 так не выводит
        m = measure.measure(*response(lambda s: (s * s + w0 ** 2) / (s * s + s * w0 / Q + w0 ** 2), f0 * 1.003))
        self.assertEqual(m["type"], "BS")
        self.assertRel(m["f0"], f0)
        self.assertRel(m["Q"], Q, tol=5e-3)

    def test_integrator_with_rf(self):
        R1, Rf, C = 1e3, 100e3, 10e-9
        fc = 1 / (2 * math.pi * Rf * C)
        m = measure.measure(*response(lambda s: -(Rf / R1) / (1 + s * Rf * C), fc))
        self.assertEqual(m["type"], "LP")
        self.assertAlmostEqual(m["k_dc_db"], 40.0, delta=0.01)
        self.assertRel(m["f_u"], fc * math.sqrt((Rf / R1) ** 2 - 1))

    def test_flat_amplifier(self):
        m = measure.measure(*response(lambda s: -4.7 + 0 * s, 1e3))
        self.assertEqual(m["type"], "flat")
        self.assertAlmostEqual(m["k_pass_db"], 20 * math.log10(4.7), delta=0.01)
        self.assertIsNone(m["fc"])

    def test_phase_unwrap(self):
        self.assertEqual(measure.unwrap([170, 179, -179, -170, None, 175]), [170, 179, 181, 190, None, 175 - 360 + 360])


class FromMC12Output(unittest.TestCase):
    def test_rc_probe(self):
        m = measure.from_output(OUT / "rc_probe.ANO")
        self.assertEqual(m["type"], "LP")
        self.assertAlmostEqual(m["fc"] / (1 / (2 * math.pi * 1e3 * 100e-9)), 1, delta=1e-3)


class Tolerance(unittest.TestCase):
    def test_within(self):
        ref = {"type": "LP", "fc": 1000.0, "k_pass_db": 0.0, "Q": 0.707}
        ok, _ = measure.within_tolerance(ref, {"type": "LP", "fc": 1004.0, "k_pass_db": 0.09, "Q": 0.713},
                                         ["type", "fc", "k_pass_db", "Q"])
        self.assertTrue(ok)
        ok, d = measure.within_tolerance(ref, {"type": "LP", "fc": 1006.0, "k_pass_db": 0.0, "Q": 0.707},
                                         ["fc"])
        self.assertFalse(ok)
        self.assertFalse(measure.within_tolerance(ref, dict(ref, type="HP"), ["type"])[0])
        self.assertFalse(measure.within_tolerance(ref, dict(ref, Q=None), ["Q"])[0])


if __name__ == "__main__":
    unittest.main()
