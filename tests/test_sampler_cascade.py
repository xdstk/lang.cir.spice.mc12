"""Тесты выборки номиналов и каскадов L3 (docs/generator_spec.md, разделы 3 и 5).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, cascade, formulas, mna, sampler  # noqa: E402


def in_series(v, kind):
    base = formulas.E24 if kind == "R" else formulas.E12
    m = v / 10 ** math.floor(math.log10(v) + 1e-9)
    return any(abs(m - s) < 1e-6 for s in base)


class Sampler(unittest.TestCase):
    def test_series_values(self):
        r = sampler.series_values("R", 100, 1e6)
        self.assertEqual((r[0], r[-1], len(r)), (100.0, 1e6, 97))  # 4 декады * 24 + 1 МОм
        self.assertEqual(len(sampler.series_values("C", 100e-12, 10e-6)), 61)

    def test_deterministic(self):
        a = sampler.sample("sallen_key_lp", random.Random(5))[0]
        b = sampler.sample("sallen_key_lp", random.Random(5))[0]
        self.assertEqual(a["netlist"], b["netlist"])

    def test_constraints_hold(self):
        rng = random.Random(1)
        for topo in formulas.TOPOLOGIES:
            for _ in range(15):
                ref, _ = sampler.sample(topo, rng)
                with self.subTest(topo=topo):
                    self.assertTrue(sampler.acceptable(ref)[0])
                    for k, v in ref["params"].items():
                        if k != "dc":
                            self.assertTrue(in_series(v, k[0]), (k, v))
                            lo, hi = sampler.RANGES[k[0]]
                            self.assertTrue(lo * 0.999 <= v <= hi * 1.001, (k, v))
                    if "Q" in ref["quantities"]:
                        self.assertTrue(0.5 <= ref["expected"]["Q"] <= 10)

    def test_gen_range(self):
        rng = random.Random(3)
        refs = [sampler.sample("rc_lp", rng, sampler.RANGES_GEN)[0] for _ in range(40)]
        self.assertTrue(any(r["params"]["R"] > 1e6 for r in refs))
        self.assertTrue(all(r["params"]["R"] <= 10e6 * 1.001 for r in refs))

    def test_rejects_unstable_sallen_key(self):
        ref = formulas.reference("sallen_key_lp", {"R1": 10e3, "R2": 10e3, "C1": 10e-9, "C2": 10e-9,
                                                   "Ra": 1e3, "Rb": 2.2e3})  # K = 3.2 -> Q < 0
        self.assertFalse(sampler.acceptable(ref)[0])


class Cascade(unittest.TestCase):
    def test_product_of_stages(self):
        # выход ОУ - идеальный источник: отклик каскада = произведение откликов звеньев
        s1 = formulas.reference("active_lp_inv", {"R1": 10e3, "R2": 22e3, "C": 10e-9})
        s2 = formulas.reference("sallen_key_lp", {"R1": 15e3, "R2": 15e3, "C1": 10e-9, "C2": 4.7e-9})
        net = cascade.compose([s1, s2])
        e, e1, e2 = (canon.parse_netlist(x)[0] for x in (net, s1["netlist"], s2["netlist"]))
        for f in (10, 300, 1e3, 3e3, 1e5):
            self.assertAlmostEqual(abs(mna.response(e, f)) / abs(mna.response(e1, f) * mna.response(e2, f)), 1,
                                   places=5)
        self.assertEqual(net.count("OPAMP_IDEAL"), 2)
        self.assertEqual(net.count("V1 "), 1)

    def test_kinds(self):
        rng = random.Random(2)
        want = {"LP+LP": "LP", "HP+HP": "HP", "AMP+LP": "LP", "AMP+HP": "HP", "HP+LP": "BP",
                "LP+LP+LP": "LP", "AMP+LP+LP": "LP"}
        for kind in cascade.KINDS:
            r = cascade.sample(kind, rng)
            with self.subTest(kind=kind):
                self.assertEqual(r["level"], "L3")
                self.assertEqual(r["expected"]["type"], want[kind])
                self.assertEqual(canon.canonical_netlist(r["netlist"]), r["netlist"])
                self.assertTrue(10 <= r["f_char"] <= 1e6)


if __name__ == "__main__":
    unittest.main()
