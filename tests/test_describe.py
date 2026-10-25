"""Тесты русских формулировок и их обратного разбора (docs/generator_spec.md, раздел 4).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import inspect
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, cascade, describe_ru, formulas, iso, sampler  # noqa: E402


def roundtrip(netlist, family, seed):
    prompt = describe_ru.describe_a(netlist, family, random.Random(seed))
    back = describe_ru.parse_a(prompt)
    return iso.match(canon.parse_netlist(netlist)[0], back, "values")["match"], prompt


class TaskA(unittest.TestCase):
    def test_back_parse_all_topologies(self):
        rng = random.Random(9)
        for topo in formulas.TOPOLOGIES:
            for k in range(6):
                ref, _ = sampler.sample(topo, rng)
                for fam in describe_ru.FAMILIES:
                    ok, prompt = roundtrip(ref["netlist"], fam, k)
                    with self.subTest(topo=topo, family=fam, seed=k):
                        self.assertTrue(ok, prompt)

    def test_back_parse_cascades(self):
        rng = random.Random(4)
        for kind in cascade.KINDS:
            ref = cascade.sample(kind, rng)
            for fam in describe_ru.FAMILIES:
                ok, prompt = roundtrip(ref["netlist"], fam, 1)
                with self.subTest(kind=kind, family=fam):
                    self.assertTrue(ok, prompt)

    def test_source_level(self):
        ok, prompt = roundtrip("V1 IN 0 DC -5\nR1 IN OUT 2.2k\nR2 OUT 0 4.7MEG\n", "narrative", 2)
        self.assertTrue(ok, prompt)
        self.assertIn("-5", prompt.replace(",", "."))

    def test_unit_words(self):
        st = describe_ru.Style(random.Random(0))
        st.comma = True
        self.assertIn(st.value("R", "4.7k"), {"4,7 кОм", "4,7 килоом", "4700 Ом"})
        self.assertIn(st.value("R", "1MEG"), {"1 МОм", "1 мегаом"})
        self.assertIn(st.value("C", "100n"), {"100 нФ", "0,1 мкФ"})
        self.assertEqual(st.value("L", "100u"), "100 мкГн")

    def test_ambiguous_prompt_rejected(self):
        with self.assertRaises(ValueError):
            describe_ru.parse_a("На вход схемы подаётся сигнал. Резистор 10 кОм между входом схемы и точкой 1 и выходом схемы.")


class TaskA1(unittest.TestCase):
    def test_roles_cover_all_parameters(self):
        for topo, (fn, *_rest) in formulas.TOPOLOGIES.items():
            params = [p for p in inspect.signature(fn).parameters if p != "dc"]
            with self.subTest(topo=topo):
                self.assertIn(topo, describe_ru.ROLES)
                self.assertEqual(set(describe_ru.ROLES[topo]["roles"]), set(params))

    def test_every_value_mentioned(self):
        rng = random.Random(6)
        for topo in formulas.TOPOLOGIES:
            ref, _ = sampler.sample(topo, rng)
            text = describe_ru.describe_a1(ref, random.Random(1))
            n_values = sum(1 for k in ref["params"] if k != "dc")
            with self.subTest(topo=topo):
                self.assertEqual(len(describe_ru.VALUE_RE.findall(text)), n_values, text)

    def test_cascade(self):
        ref = cascade.sample("AMP+LP", random.Random(8))
        text = describe_ru.describe_a1(ref, random.Random(2))
        self.assertTrue(text.startswith("Каскад из 2 звеньев"), text)
        self.assertIn("1) ", text)
        self.assertIn("2) ", text)


if __name__ == "__main__":
    unittest.main()
