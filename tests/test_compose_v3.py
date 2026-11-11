"""Композиции v3: цепочки с ветвями, добавки без указания выхода, схемы на 2-4 ОУ (docs/generator_v3_spec.md, П2-П6).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, compose, compose_v3, iso, mc12run, pilot, prompts_v3, spice2cir  # noqa: E402


def elements(net):
    return canon.parse_netlist(net)[0]


class Chain(unittest.TestCase):
    def test_branch_netlist(self):
        """Первый элемент ветви - от линии к внутреннему узлу, второй - от него на землю."""
        net = compose_v3.chain_netlist([("ser", "R", 1e3), ("branch", "C", 1e-7, "R", 100.0), ("ser", "L", 1e-3),
                                        ("branch", "R", 220.0, "L", 1e-3), ("shunt", "C", 1e-9)])
        self.assertEqual(net, "V1 IN 0 DC 0\nR1 IN N1 1k\nC1 N1 M1 100n\nR2 M1 0 100\nL1 N1 OUT 1m\n"
                              "R3 OUT M2 220\nL2 M2 0 1m\nC2 OUT 0 1n\n")

    def test_shares_and_limits(self):
        rng = random.Random(1)
        refs = [compose_v3.sample_chain(rng) for _ in range(200)]
        sections = [r["params"]["sections"] for r in refs]
        share = sum(any(s[0] == "branch" for s in sec) for sec in sections) / len(refs)
        self.assertAlmostEqual(share, compose_v3.BRANCH_SHARE, delta=0.1)
        self.assertTrue(any(a[0] == b[0] == "branch" for sec in sections for a, b in zip(sec, sec[1:])))  # две ветви подряд
        self.assertTrue(any(len(sec) > compose.CHAIN_SECTIONS[1] for sec in sections))                    # длиннее, чем в v2
        kinds = {(s[1], s[3]) for sec in sections for s in sec if s[0] == "branch"}
        self.assertEqual(kinds, set(compose_v3.BRANCH_KINDS))
        for r in refs:
            self.assertLessEqual(len(elements(r["netlist"])) - 1, compose.MAX_ELEMENTS)

    def test_branch_flag(self):
        rng = random.Random(2)
        self.assertTrue(any(s[0] == "branch" for s in compose_v3.random_chain(rng, branch=True)))
        self.assertFalse(any(s[0] == "branch" for s in compose_v3.random_chain(rng, branch=False)))

    def test_prompt_round_trip(self):
        rng = random.Random(3)
        seen = {True: 0, False: 0}
        for _ in range(250):
            ref = compose_v3.sample_chain(rng)
            d = compose_v3.describe_chain(ref["params"]["sections"], rng)
            seen[d["reordered"]] += 1
            back = elements(compose_v3.parse_chain(d["prompt"]))
            self.assertTrue(iso.match(elements(ref["netlist"]), back, "values")["match"], d["prompt"])
        self.assertAlmostEqual(seen[True] / 250, compose_v3.REORDER, delta=0.08)

    def test_reordered_form(self):
        sections = [("ser", "R", 1e3), ("shunt", "C", 1e-7), ("ser", "R", 2.2e3), ("branch", "R", 100.0, "C", 1e-8)]
        for i in range(30):
            d = compose_v3.describe_chain(sections, random.Random(i), reorder=True)
            self.assertRegex(d["prompt"], r"после (1-го|первого) - ")
            self.assertRegex(d["prompt"], r"после (2-го|второго|последнего) - .* и .* последовательно")
            self.assertEqual(compose_v3.parse_chain(d["prompt"]), compose_v3.chain_netlist(sections))

    def test_v2_chain_prompts_still_parse(self):
        """Запросы цепочек v2 (ветвь - только RC) разбираются и новым разбором."""
        rng = random.Random(4)
        for _ in range(100):
            ref = compose.sample_chain(rng)
            prompt = compose.describe_chain(ref["params"]["sections"], rng)
            self.assertTrue(iso.match(elements(ref["netlist"]), elements(compose_v3.parse_chain(prompt)), "values")["match"], prompt)


class Addon(unittest.TestCase):
    def test_all_bases(self):
        """П5: основой может быть любая названная топология, кроме интегратора без резистора в ОС."""
        rng = random.Random(5)
        self.assertEqual(len(compose_v3.BASES), 29)
        for base in compose_v3.BASES:
            with self.subTest(base):
                self.assertEqual(compose_v3.sample_addon(rng, [base], max_tries=200)["params"]["base_topology"], base)

    def test_band_pass_bases_admitted(self):
        """Полосовая основа проходит отбор по Q; в v2 её отсекал подъём над краями развёртки."""
        rng = random.Random(6)
        types = [compose_v3.sample_addon(rng, ["mfb_bp", "rlc_series_bp"])["expected"]["type"] for _ in range(12)]
        self.assertGreater(types.count("BP"), 3)

    def test_output_clause(self):
        rng = random.Random(7)
        out_words = r"выход за ним|снимаем после него|выход с конденсатора|выход с него|выход с буфера|выход остаётся на месте"
        n = {True: 0, False: 0}
        for _ in range(150):
            ref = compose_v3.sample_addon(rng)
            if not {op for op, _ in ref["params"]["ops"]} & {"out_r", "out_rc", "buffer", "side"}:
                continue
            for explicit in (True, False):
                d = compose_v3.describe_addon(ref, rng, explicit_out=explicit)
                self.assertEqual(bool(re.search(out_words, d["prompt"])), explicit, d["prompt"])
                n[explicit] += 1
        self.assertGreater(n[False], 40)

    def test_prompt_names_every_added_value(self):
        rng = random.Random(8)
        shares = {"explicit_out": 0, "reordered": 0}
        for _ in range(200):
            ref = compose_v3.sample_addon(rng)
            d = compose_v3.describe_addon(ref, rng)
            for k in shares:
                shares[k] += d[k]
            numbers = {float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", d["prompt"])}
            for _op, vals in ref["params"]["ops"]:
                for v in vals:
                    mant = float(re.match(r"[\d.]+", canon.format_value(v)).group())
                    self.assertTrue(any(abs(x - mant) < 1e-9 * max(1, mant) or abs(x - v) < 1e-9 * v for x in numbers),
                                    (v, d["prompt"]))
        self.assertAlmostEqual(shares["explicit_out"] / 200, 1 - compose_v3.NO_OUT_SHARE, delta=0.1)
        self.assertAlmostEqual(shares["reordered"] / 200, compose_v3.REORDER, delta=0.1)

    def test_reordered_keeps_all_parts(self):
        rng = random.Random(9)
        ref = compose_v3.sample_addon(rng)
        a = compose_v3.describe_addon(ref, random.Random(1), explicit_out=True, reorder=False)["prompt"]
        b = compose_v3.describe_addon(ref, random.Random(1), explicit_out=True, reorder=True)["prompt"]
        self.assertTrue(any(h in b for h in compose_v3.HEAD_LAST))
        self.assertFalse(b.startswith(a.split(";")[0]))
        self.assertEqual(len(b.split("; ")), len(a.split("; ")))


class Multi(unittest.TestCase):
    def test_structure(self):
        rng = random.Random(10)
        kinds, opamps = set(), set()
        for _ in range(40):
            ref = compose_v3.sample_multi(rng)
            el = elements(ref["netlist"])
            n_op = sum(e["type"] == "OPAMP" for e in el)
            kinds.add(ref["params"]["kind"])
            opamps.add(n_op)
            self.assertEqual(ref["class"], "compose:multi")
            self.assertEqual(ref["level"], "L3")
            self.assertTrue(2 <= n_op <= compose_v3.MULTI_OPAMPS)
            self.assertLessEqual(len(el) - 1, compose_v3.MULTI_ELEMENTS)
            self.assertEqual(ref["netlist"], canon.canonical_netlist(ref["netlist"]))
            prompts_v3.verified(ref["netlist"], "freeform", rng)  # запрос - описание соединений
        self.assertEqual(kinds, {"cascade", "loop", "loop+stage"})
        self.assertEqual(opamps, {2, 3, 4})

    def test_output_on_any_opamp_of_loop(self):
        rng = random.Random(11)
        outs = set()
        for _ in range(60):
            net, k = compose_v3._move_out(compose_v3.filters_v2.sample_any("tow_thomas", rng)["netlist"], rng)
            outs.add(k)
            self.assertEqual(sum(e["type"] == "OPAMP" and e["nodes"][2] == "OUT" for e in elements(net)), 1)
        self.assertEqual(len(outs), 3)


class V2Unchanged(unittest.TestCase):
    def test_reference_defaults(self):
        """Пределы и отсев v2 не изменились: полосовая схема с глубокими краями по-прежнему отсекается."""
        net = "V1 IN 0 DC 0\nR1 IN N1 1k\nL1 N1 N2 10m\nC1 N2 OUT 100n\nR2 OUT 0 100\n"
        self.assertEqual(compose._reference(net, "x", {})[1], "resonance")
        ref, why = compose._reference(net, "x", {}, band=True)
        self.assertIsNone(why)
        self.assertEqual(ref["quantities"], ["type", "k_pass_db", "f0"])
        big = "V1 IN 0 DC 0\n" + "".join(f"R{i} IN OUT {i}k\n" for i in range(1, 15))
        self.assertEqual(compose._reference(big, "x", {})[1], "size")


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Admission(unittest.TestCase):
    def test_batch(self):
        """Допуск, как в датасете: MC12 против узлового расчёта; каждая схема строится в .CIR."""
        rng = random.Random(12)
        refs = []
        for _ in range(8):
            for fn in (compose_v3.sample_chain, compose_v3.sample_addon, compose_v3.sample_multi):
                r = fn(rng)
                r["id"] = f"c{len(refs):02d}"
                refs.append(r)
        bad = [r for r in pilot.admit(refs, ROOT / "out" / "test_compose_v3") if not r["admission"]["admitted"]]
        self.assertLessEqual(len(bad), 2, [(r["class"], r["netlist"]) for r in bad])
        for r in refs:
            self.assertTrue(spice2cir.build(r["netlist"], r["f_char"]))


if __name__ == "__main__":
    unittest.main()
