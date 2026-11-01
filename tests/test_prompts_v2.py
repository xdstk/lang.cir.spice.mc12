"""Запросы v2: точные номиналы, freeform, terse, правки текста (docs/generator_v2_spec.md, разделы 3-4).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, compose, filters_v2, iso, prompts_v2, sampler  # noqa: E402


def elements(net):
    return canon.parse_netlist(net)[0]


def numbers(prompt):
    """Числа запроса с единой десятичной точкой: правки текста их не меняют."""
    return sorted(m.replace(",", ".") for m in re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"[RCLСXт]\d+", " ", prompt)))


class Values(unittest.TestCase):
    def test_share_of_exact_values(self):
        rng = random.Random(1)
        series = set(sampler.series_values("R", *sampler.RANGES["R"]))
        vals = [filters_v2.pick("R", rng) for _ in range(2000)]
        share = sum(v not in series for v in vals) / len(vals)
        self.assertAlmostEqual(share, filters_v2.EXACT_SHARE, delta=0.04)

    def test_exact_value_survives_format(self):
        rng = random.Random(2)
        for _ in range(500):
            v = filters_v2.pick("C", rng)
            self.assertAlmostEqual(canon.parse_value(canon.format_value(v)) / v, 1, delta=1e-5)

    def test_v1_pick_unchanged(self):
        rng = random.Random(3)
        series = set(sampler.series_values("R", *sampler.RANGES["R"]))
        self.assertTrue(all(sampler.pick("R", rng) in series for _ in range(200)))


class Freeform(unittest.TestCase):
    def test_round_trip(self):
        rng = random.Random(4)
        refs = [compose.sample_chain(rng) for _ in range(40)] + [compose.sample_addon(rng) for _ in range(40)]
        refs += [filters_v2.sample(t, rng)[0] for t in filters_v2.TOPOLOGIES_V2]
        for ref in refs:
            prompt = prompts_v2.describe_freeform(ref["netlist"], rng)
            back = prompts_v2.parse_freeform(prompt)
            self.assertTrue(iso.match(elements(ref["netlist"]), back, "values")["match"], prompt)


class Terse(unittest.TestCase):
    def test_named(self):
        rng = random.Random(5)
        ref = filters_v2.sample("mfb_lp", rng)[0]
        prompt = prompts_v2.describe_terse(ref, rng)
        self.assertEqual(len(prompt.split()) - len(prompt.split()[:-5]), 5)

    def test_every_topology(self):
        rng = random.Random(6)
        for topo in filters_v2.TOPOLOGIES_V2:
            self.assertTrue(prompts_v2.describe_terse(filters_v2.sample(topo, rng)[0], rng), topo)


class Mutate(unittest.TestCase):
    def test_values_survive(self):
        rng = random.Random(7)
        seen = set()
        for _ in range(300):
            ref = compose.sample_chain(rng)
            base = compose.describe_chain(ref["params"]["sections"], rng)
            new, used = prompts_v2.mutate(base, rng)
            seen |= set(used)
            if "suffix" not in used:  # «питание ±12 В» добавляет числа сознательно
                self.assertEqual(numbers(new), numbers(base), new)
        self.assertGreaterEqual(len(seen), 6)

    def test_lower_keeps_units(self):
        self.assertEqual(prompts_v2._lower("Резистор 1 МОм"), "резистор 1 МОм")
        self.assertEqual(prompts_v2._lower("10 нФ: Вход"), "10 нФ: Вход")

    def test_cyrillic_c(self):
        self.assertEqual(prompts_v2._cyrillic_c("C1 100н, R1 1к"), "С1 100н, R1 1к")

    def test_typo_keeps_digits(self):
        rng = random.Random(8)
        s = prompts_v2._typo("конденсатор 100 нФ", rng)
        self.assertIn("100 нФ", s)
        self.assertNotEqual(s, "конденсатор 100 нФ")


if __name__ == "__main__":
    unittest.main()
