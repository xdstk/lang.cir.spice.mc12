"""Композиции v2: цепочки и добавки (docs/generator_v2_spec.md, раздел 2.2).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, compose, filters_v2, iso, mc12run, pilot  # noqa: E402


def elements(net):
    return canon.parse_netlist(net)[0]


class Chain(unittest.TestCase):
    def test_netlist(self):
        net = compose.chain_netlist([("ser", "R", 1e3), ("shunt", "C", 1e-7), ("ser", "L", 1e-3), ("shunt_rc", 100.0, 1e-9)])
        self.assertEqual(net, "V1 IN 0 DC 0\nR1 IN N1 1k\nC1 N1 0 100n\nL1 N1 OUT 1m\nR2 OUT M1 100\nC2 M1 0 1n\n")

    def test_prompt_round_trip(self):
        rng = random.Random(1)
        for _ in range(300):
            ref = compose.sample_chain(rng)
            prompt = compose.describe_chain(ref["params"]["sections"], rng)
            back = elements(compose.parse_chain(prompt))
            self.assertTrue(iso.match(elements(ref["netlist"]), back, "values")["match"], prompt)

    def test_limits(self):
        rng = random.Random(2)
        for _ in range(100):
            ref = compose.sample_chain(rng)
            el = elements(ref["netlist"])
            self.assertLessEqual(len(el) - 1, compose.MAX_ELEMENTS)
            self.assertIn("OUT", {n for e in el for n in e["nodes"]})


class Addon(unittest.TestCase):
    def test_operations_keep_anchors(self):
        rng = random.Random(3)
        base = ["R1 IN OUT 1k", "C1 OUT 0 100n"]
        lines, _vals = compose._op(list(base), "out_rc", rng)
        el = elements("V1 IN 0 DC 0\n" + "\n".join(lines) + "\n")
        self.assertEqual(len(el), 5)
        self.assertEqual(sum("OUT" in e["nodes"] for e in el), 2)  # новый выход: R и C после добавки
        lines, _vals = compose._op(list(base), "side", rng)
        self.assertTrue(any(l.startswith("C2 ") and l.split()[2] == "0" for l in lines))
        self.assertEqual(sum(" OUT " in l + " " for l in lines), 3)  # выход на месте

    def test_feedback_only_with_opamp(self):
        rng = random.Random(4)
        self.assertIsNone(compose._op(["R1 IN OUT 1k", "C1 OUT 0 100n"], "par_fb", rng))
        lines, _ = compose._op(["R1 IN M 1k", "R2 M OUT 10k", "X1 0 M OUT OPAMP_IDEAL"], "par_fb", rng)
        self.assertTrue(any(l.startswith("C1 M OUT") for l in lines))

    def test_prompts_name_every_value(self):
        rng = random.Random(5)
        for _ in range(100):
            ref = compose.sample_addon(rng)
            prompt = compose.describe_addon(ref, rng)
            self.assertIn(";", prompt)
            for op, vals in ref["params"]["ops"]:
                self.assertTrue(op in compose.OPS and all(isinstance(v, float) for v in vals))
            self.assertLessEqual(len(elements(ref["netlist"])) - 1, compose.MAX_ELEMENTS)
            self.assertTrue(re.search(r"\d", prompt))
            names = [op for op, _ in ref["params"]["ops"]]
            self.assertEqual(names, sorted(names, key=compose.OPS.index))
            if "bridge" in names:  # «между входом и выходом» - итоговые узлы
                self.assertTrue(any(e["type"] == "C" and set(e["nodes"]) == {"IN", "OUT"} for e in elements(ref["netlist"])))


class SampleV2(unittest.TestCase):
    def test_all_topologies(self):
        rng = random.Random(6)
        for topo in filters_v2.TOPOLOGIES_V2:
            with self.subTest(topo):
                self.assertEqual(filters_v2.sample(topo, rng)[0]["class"], topo)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Admission(unittest.TestCase):
    def test_batch(self):
        rng = random.Random(7)
        refs = []
        for _ in range(10):
            for fn in (compose.sample_chain, compose.sample_addon):
                r = fn(rng)
                r["id"] = f"c{len(refs):02d}"
                refs.append(r)
        bad = [r for r in pilot.admit(refs, ROOT / "out" / "test_compose") if not r["admission"]["admitted"]]
        self.assertLessEqual(len(bad), 2, [(r["class"], r["netlist"]) for r in bad])


if __name__ == "__main__":
    unittest.main()
