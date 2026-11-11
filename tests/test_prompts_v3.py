"""Запросы v3: имена узлов из запроса (П1) и описание не по порядку (П6), docs/generator_v3_spec.md.

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, cascade, compose, describe_ru, filters_v2, formulas, iso, mna, prompts_v2, prompts_v3  # noqa: E402

TWIN_T = ("V1 IN 0 DC 0\nR1 IN N1 26.7k\nR2 N3 0 95.3k\nR3 N1 N4 26.7k\nR4 N2 N5 13.3k\nR5 N3 OUT 4.99k\n"
          "C1 IN N2 100n\nC2 N1 N5 200n\nC3 N2 N4 100n\nX1 N3 N5 N5 OPAMP_IDEAL\nX2 N4 OUT OUT OPAMP_IDEAL\n")


def elements(net):
    return canon.parse_netlist(net)[0]


def circuits(seed, n=25):
    rng = random.Random(seed)
    refs = [compose.sample_chain(rng) for _ in range(n)] + [compose.sample_addon(rng) for _ in range(n)]
    refs += [filters_v2.sample_any(t, rng) for t in list(formulas.TOPOLOGIES) + list(filters_v2.TOPOLOGIES_V2)]
    refs += [cascade.sample(k, rng) for k in cascade.KINDS]
    return [r["netlist"] for r in refs]


class NodeNames(unittest.TestCase):
    def test_rule(self):
        for token, name in (("A", "A"), ("a", "A"), ("P1", "P1"), ("n2", "N2"), ("3", "N3"), ("т3", "N3"),
                            ("т.3", "N3"), ("у12", "N12"), ("E,", "E"), ("C", "C")):
            self.assertEqual(prompts_v3.node_name(token), name, token)

    def test_names_are_valid(self):
        """Имена не повторяются, не совпадают с якорями и с именами элементов (предупреждение MC12)."""
        rng = random.Random(1)
        inner = [f"N{k}" for k in range(1, 11)]
        seen = set()
        for _ in range(400):
            n = rng.randint(1, 10)
            scheme, label = prompts_v3.pick_names(inner[:n], rng)
            seen.add(scheme)
            out = [prompts_v3.node_name(x) for x in label.values()]
            self.assertEqual(len(set(out)), n)
            for x in out:
                self.assertNotIn(x, prompts_v3.RESERVED)
                self.assertNotRegex(x, r"^[RCLVIX]\d+$")
                self.assertRegex(x, r"^[A-Z0-9_]+$")
        self.assertEqual(seen, set(prompts_v3.SCHEMES))

    def test_seq_is_v2_numbering(self):
        self.assertEqual(prompts_v3.pick_names(["N2", "N1", "N10"], random.Random(1), "seq")[1],
                         {"N1": "1", "N2": "2", "N10": "10"})

    def test_too_many_nodes_fall_back_to_indexed(self):
        scheme, label = prompts_v3.pick_names([f"N{k}" for k in range(1, 15)], random.Random(1), "xyz")
        self.assertEqual(scheme, "indexed")
        self.assertEqual(len(label), 14)


class Named(unittest.TestCase):
    def test_round_trip_every_family(self):
        """Обратный разбор возвращает эталон с теми же именами узлов; эталон - та же схема, что канон."""
        rng = random.Random(2)
        nets = circuits(3)
        done = {f: 0 for f in prompts_v3.FAMILIES}
        for net in nets:
            for fam in prompts_v3.FAMILIES:
                try:
                    d = prompts_v3.verified(net, fam, rng)
                except (AssertionError, ValueError):
                    # разбор v1 не рассчитан на элемент на источнике и т.п.; freeform обязан проходить всегда
                    self.assertNotEqual(fam, "freeform", net)
                    continue
                done[fam] += 1
                self.assertEqual(canon.parse_netlist(d["netlist"])[1], [])
        for fam, k in done.items():
            self.assertGreater(k, 0.9 * len(nets), fam)
        self.assertEqual(done["freeform"], len(nets))

    def test_reference_keeps_element_lines(self):
        """В эталоне заменены только узлы: имена, порядок строк и номиналы - как в каноне."""
        rng = random.Random(4)
        for net in circuits(5, 10):
            d = prompts_v3.describe_named(net, "freeform", rng)
            for a, b in zip(net.splitlines(), d["netlist"].splitlines()):
                ta, tb = a.split(), b.split()
                self.assertEqual(ta[0], tb[0])
                self.assertEqual(ta[-1], tb[-1])
            self.assertEqual(len(net.splitlines()), len(d["netlist"].splitlines()))

    def test_names_from_prompt_are_in_reference(self):
        rng = random.Random(6)
        for scheme in prompts_v3.SCHEMES:
            for fam in prompts_v3.FAMILIES:
                d = prompts_v3.verified(TWIN_T, fam, rng, scheme=scheme)
                nodes = {n for e in elements(d["netlist"]) for n in e["nodes"]} - set(canon.ANCHORS)
                self.assertEqual(len(nodes), 5)
                named = {prompts_v3.node_name(m) for m in re.findall(r"(?:точк\w+|узл\w+|узел) ([A-Za-z0-9_]+)", d["prompt"])}
                if named:  # freeform может называть узлы без слова «узел»
                    self.assertEqual(named, nodes, d["prompt"])
                self.assertEqual(d["names"], scheme)

    def test_numbers_become_n(self):
        d = prompts_v3.verified(TWIN_T, "list", random.Random(7), scheme="numbers", shuffle=0)
        nodes = {n for e in elements(d["netlist"]) for n in e["nodes"]} - set(canon.ANCHORS)
        self.assertTrue(all(re.fullmatch(r"N\d+", n) for n in nodes))
        self.assertNotEqual(d["netlist"], TWIN_T)  # номера из запроса, не из обхода канона

    def test_seq_without_shuffle_is_v2_text(self):
        """Номера канона и без перестановки: эталон - канон, текст семейств describe_ru - тот же, что в gen 3."""
        for i, net in enumerate(circuits(8, 5)):
            for fam in ("list", "table", "narrative", "path"):
                d = prompts_v3.describe_named(net, fam, random.Random(i), scheme="seq", shuffle=0)
                self.assertEqual(d["netlist"], net)
                self.assertEqual(d["prompt"], describe_ru.describe_a(net, fam, random.Random(i), gen=3))
                self.assertFalse(d["shuffled"])

    def test_no_inner_nodes(self):
        d = prompts_v3.verified("V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n", "freeform", random.Random(9))
        self.assertIsNone(d["names"])
        self.assertEqual(d["netlist"], "V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n")

    def test_rejects_non_canonical_nodes(self):
        with self.assertRaises(ValueError):
            prompts_v3.describe_named("V1 IN 0 DC 0\nR1 IN A 1k\nC1 A OUT 100n\n", "list", random.Random(1))

    def test_behaviour_unchanged(self):
        """Имена узлов не меняют схему: узловой расчёт эталона совпадает с расчётом канона."""
        d = prompts_v3.verified(TWIN_T, "freeform", random.Random(10), scheme="abc")
        a = mna.sweep(elements(TWIN_T), 10, 1e4, per_decade=5)
        b = mna.sweep(elements(d["netlist"]), 10, 1e4, per_decade=5)
        for x, y in zip(a[1], b[1]):
            self.assertAlmostEqual(x, y, places=9)


class Shuffle(unittest.TestCase):
    def test_shuffle_changes_order_not_circuit(self):
        changed = 0
        for fam in prompts_v3.FAMILIES:
            for i in range(20):
                a = prompts_v3.verified(TWIN_T, fam, random.Random(i), scheme="seq", shuffle=0)
                b = prompts_v3.verified(TWIN_T, fam, random.Random(i), scheme="seq", shuffle=1)
                self.assertTrue(b["shuffled"])
                self.assertEqual(a["netlist"], b["netlist"])  # эталон от порядка описания не зависит
                changed += a["prompt"] != b["prompt"]
        self.assertGreater(changed, 90)

    def test_share(self):
        rng = random.Random(11)
        n = sum(prompts_v3.describe_named(TWIN_T, "freeform", rng)["shuffled"] for _ in range(1000))
        self.assertAlmostEqual(n / 1000, prompts_v3.SHUFFLE, delta=0.05)

    def test_opamps_between_parts(self):
        """freeform: ОУ описан не только в конце."""
        early = 0
        for i in range(40):
            parts = prompts_v3.describe_named(TWIN_T, "freeform", random.Random(i), shuffle=1)["prompt"].split("; ")
            early += any(p.startswith("ОУ") for p in parts[:-3])
        self.assertGreater(early, 20)


class Freeform(unittest.TestCase):
    def test_bare_names_and_designators(self):
        bare = des = 0
        for i in range(200):
            d = prompts_v3.verified(TWIN_T, "freeform", random.Random(i), scheme="abc")
            bare += not re.search(r"узл|узел|точк", d["prompt"])
            des += bool(re.search(r"\bR1\b", d["prompt"]))
        self.assertGreater(bare, 40)
        self.assertGreater(des, 40)

    def test_numbers_are_never_bare(self):
        for i in range(100):
            d = prompts_v3.verified(TWIN_T, "freeform", random.Random(i), scheme="mixed")
            if re.search(r"\d", "".join(re.findall(r"(?:узл\w*|узел|точк\w+) (\S+)", d["prompt"]))):
                self.assertRegex(d["prompt"], r"узл|узел|точк")

    def test_parse_real_style(self):
        """Запись ближе к запросам mc12_real: обозначения, имена без слова «узел», «на землю»."""
        prompt = ("R1 26,7к от входа к A; C1 100н от входа к B; R3 26,7к между A и C; C2 200н между A и D; "
                  "R4 13,3к между B и D; C3 100н между B и C; R2 95,3к от E на землю; R5 4,99к от E к выходу; "
                  "ОУ1: неинв. вход - E, инв. вход - D, выход ОУ - D; ОУ2: плюс - C, минус - выход, выход ОУ - выход; "
                  "сигнал на вх")
        back = prompts_v3.parse_freeform(prompt)
        self.assertTrue(iso.match(elements(TWIN_T), back, "values")["match"])
        self.assertEqual({n for e in back for n in e["nodes"]}, {"IN", "OUT", "0", "A", "B", "C", "D", "E"})

    def test_text_mutations_keep_numbers(self):
        """Правки текста v2 применимы к запросам v3: номиналы не меняются."""
        rng = random.Random(12)
        for net in circuits(13, 8):
            d = prompts_v3.describe_named(net, "freeform", rng, scheme="abc")
            new, used = prompts_v2.mutate(d["prompt"], rng)
            if "suffix" not in used:
                strip = lambda s: sorted(m.replace(",", ".") for m in re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"[RCLСXУ]\d+", " ", s)))  # noqa: E731
                self.assertEqual(strip(new), strip(d["prompt"]), new)


class SameCircuit(unittest.TestCase):
    def test_node_names_matter(self):
        a = elements("V1 IN 0 DC 0\nR1 IN A 1k\nC1 A OUT 100n\n")
        self.assertTrue(prompts_v3.same_circuit(a, elements("V1 IN 0 DC 0\nC7 OUT A 100n\nR9 A IN 1k\n")))
        self.assertFalse(prompts_v3.same_circuit(a, elements("V1 IN 0 DC 0\nR1 IN B 1k\nC1 B OUT 100n\n")))
        self.assertFalse(prompts_v3.same_circuit(a, elements("V1 IN 0 DC 0\nR1 IN A 1k\nC1 A OUT 10n\n")))

    def test_opamp_pins_ordered(self):
        a = elements("V1 IN 0 DC 0\nX1 IN A OUT OPAMP_IDEAL\n")
        self.assertFalse(prompts_v3.same_circuit(a, elements("V1 IN 0 DC 0\nX1 A IN OUT OPAMP_IDEAL\n")))


if __name__ == "__main__":
    unittest.main()
