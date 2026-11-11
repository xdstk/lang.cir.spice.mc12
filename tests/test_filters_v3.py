"""Названные схемы и состав v3: активный двойной Т-мост, каскады Саллен-Ки, роли KHN (docs/generator_v3_spec.md).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import inspect
import json
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, cascade, describe_ru, filters_v2, filters_v3, formulas, iso, mc12run, measure, mna, pilot, spice2cir, synth  # noqa: E402

ROLES = json.loads((ROOT / "data" / "roles_ru.json").read_text(encoding="utf-8"))
KHN = dict(R1=18e3, R2=18e3, R3=24e3, R4=18e3, R5=13e3, R6=18e3, C1=8.2e-9, C2=8.2e-9)


def elements(net):
    return canon.parse_netlist(net)[0]


class KhnRoles(unittest.TestCase):
    """Короткие роли v3 называют, за что отвечает элемент; соответствие проверяется по схеме и расчётом."""

    def band_pass(self, p):
        """f0 и Q по полосовому выходу (выход первого интегратора) и усиление ФНЧ-выхода."""
        net, an = filters_v2.khn(**p)
        el = elements(net)
        swap = {"OUT": "T0", "N4": "OUT"}
        bp = [dict(e, nodes=[swap.get(n, n) for n in e["nodes"]]) for e in el]
        m = measure.measure(*mna.sweep(bp, an["f0"] / 1000, an["f0"] * 1000, per_decade=400))
        k = measure.measure(*mna.sweep(el, an["f0"] / 1000, an["f0"] * 1000, per_decade=100))["k_pass_db"]
        return m["f0"], m["Q"], k

    def test_structure(self):
        """Rвх и Rq - на неинвертирующем входе сумматора; Rос и Rсум - на инвертирующем; Rи1, Cи1 и Rи2, Cи2 -
        интеграторы; выход - второй интегратор."""
        sr = describe_ru.short_roles("khn", 3)
        el = {e["name"]: e for e in elements(filters_v2.khn(**KHN)[0])}
        summer = next(e for e in el.values() if e["type"] == "OPAMP" and e["nodes"][0] != "0")
        int1 = next(e for e in el.values() if e["type"] == "OPAMP" and summer["nodes"][2] in
                    {n for r in el.values() if r["type"] == "R" and e["nodes"][1] in r["nodes"] for n in r["nodes"]})
        int2 = next(e for e in el.values() if e["type"] == "OPAMP" and e not in (summer, int1))
        self.assertEqual(int2["nodes"][2], "OUT")
        plus, minus, hp = summer["nodes"]
        self.assertEqual(set(el["R1"]["nodes"]), {"IN", plus})
        self.assertEqual(set(el["R2"]["nodes"]), {int1["nodes"][2], plus})
        self.assertEqual(set(el["R4"]["nodes"]), {"OUT", minus})
        self.assertEqual(set(el["R6"]["nodes"]), {hp, minus})
        self.assertEqual(set(el["R5"]["nodes"]), {hp, int1["nodes"][1]})
        self.assertEqual(set(el["C2"]["nodes"]), {int1["nodes"][1], int1["nodes"][2]})
        self.assertEqual(set(el["R3"]["nodes"]), {int1["nodes"][2], int2["nodes"][1]})
        self.assertEqual(set(el["C1"]["nodes"]), {int2["nodes"][1], "OUT"})
        self.assertEqual(sr, {"R1": "Rвх", "R2": "Rq", "R3": "Rи2", "R4": "Rос", "R5": "Rи1", "R6": "Rсум",
                              "C1": "Cи2", "C2": "Cи1"})

    def test_what_each_resistor_sets(self):
        f0, q, k = self.band_pass(KHN)
        for name in ("R1", "R2"):  # вход и добротность: частоту не меняют
            f, qq, _ = self.band_pass(dict(KHN, **{name: 2 * KHN[name]}))
            self.assertAlmostEqual(f / f0, 1, delta=0.01)
            self.assertGreater(abs(qq / q - 1), 0.2)
        for name in ("R3", "R5", "C1", "C2"):  # интеграторы: частота в корень из двух ниже, усиление то же
            f, _, kk = self.band_pass(dict(KHN, **{name: 2 * KHN[name]}))
            self.assertAlmostEqual(f / f0, 2 ** -0.5, delta=0.01)
            self.assertAlmostEqual(kk, k, delta=0.05)
        for name, sign in (("R4", 1), ("R6", -1)):  # обратные связи сумматора: и частота, и усиление
            f, _, kk = self.band_pass(dict(KHN, **{name: 2 * KHN[name]}))
            self.assertAlmostEqual(f / f0, 2 ** (-0.5 * sign), delta=0.01)
            self.assertGreater(sign * (kk - k), 2)

    def test_labels(self):
        sr = describe_ru.short_roles("khn", 3)
        self.assertEqual(len(set(sr.values())), 8)
        self.assertEqual(describe_ru.short_roles("khn", 2), {k: k for k in sr})  # gen 2 - прежние R1-R6
        ref = {"class": "khn", "params": KHN}
        self.assertIn("R1 18", describe_ru.describe_a1_brief(ref, random.Random(1)))
        text = describe_ru.describe_a1_brief(ref, random.Random(1), 3)
        self.assertNotRegex(text, r"\bR[1-6]\b")
        for label in sr.values():
            self.assertIn(label + " ", text)


class TwinTActive(unittest.TestCase):
    def test_roles(self):
        roles = ROLES["topologies_v3"]
        self.assertEqual(set(roles), set(filters_v3.TOPOLOGIES_V3))
        for topo, (fn, *_rest) in filters_v3.TOPOLOGIES_V3.items():
            params = {p for p in inspect.signature(fn).parameters if p != "dc"}
            self.assertEqual(set(roles[topo]["roles"]), params)
            self.assertEqual(set(roles[topo]["short_roles"]), params)
            self.assertTrue(roles[topo]["names"] and roles[topo]["short_names"])
        self.assertNotIn("twin_t_active", describe_ru.ROLES)  # baseline и генератор v1 новой схемы не видят

    def test_sample(self):
        rng = random.Random(1)
        for _ in range(10):
            ref = filters_v3.sample("twin_t_active", rng)
            self.assertEqual(ref["expected"]["type"], "BS")
            self.assertAlmostEqual(ref["expected"]["f0"] / ref["analytic"]["f0"], 1, delta=0.03)
            self.assertAlmostEqual(ref["expected"]["Q"] / ref["analytic"]["Q"], 1, delta=0.1)
            self.assertEqual(sum(e["type"] == "OPAMP" for e in elements(ref["netlist"])), 2)
            self.assertIn("делителя", describe_ru.describe_a1(ref, rng, 3))
            self.assertIn("Rд(вых)", describe_ru.describe_a1_brief(ref, rng, 3))

    def test_same_structure_as_mc12_real(self):
        """Схема совпадает по структуре с twin_t_notch из mc12_real: в отчёте эту запись показывать отдельно."""
        recs = [json.loads(l) for l in (ROOT / "data/dataset/mc12_real_v1/mc12_real.jsonl").read_text(encoding="utf-8").splitlines()]
        notch = next(r for r in recs if r["id"] == "twin_t_notch")
        ours = elements(synth.twin_t_active(10e3, 10e-9, 10e3, 30e3)[0])
        self.assertTrue(iso.match(elements(notch["netlist"]), ours, "topology")["match"])
        same = [r["id"] for r in recs if iso.match(elements(r["netlist"]), ours, "topology")["match"]]
        self.assertEqual(same, ["twin_t_notch"])


class SkCascade(unittest.TestCase):
    def test_sample(self):
        rng = random.Random(2)
        seen = set()
        for _ in range(25):
            ref = filters_v3.sk_cascade(rng)
            d = ref["design"]
            n = int(d["order"])
            seen.add((d["type"], n, d["response"]))
            tops = [s["topology"] for s in ref["params"]]
            sk = "sallen_key_lp" if d["type"] == "LP" else "sallen_key_hp"
            self.assertEqual(tops, [sk] * (n // 2) + (["rc_lp" if d["type"] == "LP" else "rc_hp"] if n % 2 else []))
            self.assertEqual(ref["class"], "cascade:" + "+".join(tops))
            self.assertEqual(ref["expected"]["type"], d["type"])
            self.assertEqual(ref["netlist"], canon.canonical_netlist(ref["netlist"]))
            # эталон собирается из названных звеньев так же, как каскады v1: запрос A′ его определяет
            nets = [{"netlist": synth._builder(s["topology"])(**{k: v for k, v in s.items() if k != "topology"})[0]}
                    for s in ref["params"]]
            self.assertEqual(cascade.compose(nets), ref["netlist"])
            if d["response"] == "Butterworth":  # срез -3 дБ - заданная частота (с точностью округления к рядам)
                self.assertAlmostEqual(ref["expected"]["fc"] / canon.parse_value(d["fc"]), 1, delta=0.05)
        self.assertGreater(len(seen), 8)
        self.assertEqual({o for _, o, _ in seen}, set(filters_v3.SK_ORDERS))

    def test_prompts(self):
        rng = random.Random(3)
        ref = filters_v3.sk_cascade(rng)
        long, brief = describe_ru.describe_a1(ref, rng, 3), describe_ru.describe_a1_brief(ref, rng, 3)
        self.assertIn(f"Каскад из {len(ref['params'])} звеньев", long)
        self.assertEqual(brief.count("["), len(ref["params"]))
        for _ in range(30):
            name = filters_v3.design_name(ref, rng)
            self.assertRegex(name, r"(ФНЧ|ФВЧ|фильтр (нижних|верхних) частот) (Баттерворта|Чебышёва) [3-6]-го порядка")

    def test_same_circuit_as_mode2(self):
        """Каскад из рядов совпадает со схемой, которую режим 2 строит по тем же характеристикам."""
        rng = random.Random(4)
        done = 0
        while done < 5:
            ref = filters_v3.sk_cascade(rng)
            vals = [(k[0], v) for s in ref["params"] for k, v in s.items() if k[0] in "RC" and k != "topology"]
            if any(min(abs(v / x - 1) for x in synth.series(t)) > 1e-6 for t, v in vals):
                continue  # точные номиналы - не из рядов
            d = ref["design"]
            spec = " ".join(f"{k}={v}" for k, v in d.items())
            rep = synth.run(f".SPEC {spec} active=yes")
            ours = [dict(e, value="0") if e["type"] == "V" else e for e in elements(ref["netlist"])]  # источник - из запроса
            self.assertTrue(iso.match(ours, elements(rep["netlist"]), "values")["match"], spec)
            done += 1


class Pool(unittest.TestCase):
    def test_composition(self):
        pool = filters_v3.named_pool()
        for t in ("khn", "twin_t", "mfb_hp", "twin_t_active", "sk_cascade"):  # отложенные в v2 и новые
            self.assertIn(t, pool)
        self.assertEqual(set(filters_v3.names()), set(formulas.TOPOLOGIES) | set(filters_v2.TOPOLOGIES_V2) | {"twin_t_active"})
        self.assertFalse([t for t in pool if re.search(r"allpass|all_pass|sc_|switched", t)])  # вне состава v3
        for t in filters_v3.NOT_FILTERS:
            self.assertLess(pool[t], pool["rc_lp"])
        self.assertGreater(filters_v3.filter_share(), 0.9)
        v2 = {t: 1 if t in {"rlc_series_bp", "rlc_parallel_bp", "mfb_bp", "rlc_series_bs", "tow_thomas", "lc_ladder5"} else 3
              for t in list(formulas.TOPOLOGIES) + list(filters_v2.TOPOLOGIES_V2) if t not in ("khn", "twin_t", "mfb_hp")}
        v2.update({"cascade:" + k: 1 for k in cascade.KINDS})
        self.assertLess(filters_v3.filter_share(v2), filters_v3.filter_share())

    def test_every_name_samples(self):
        rng = random.Random(5)
        for name in filters_v3.named_pool():
            with self.subTest(name):
                ref = filters_v3.sample_named(name, rng)
                self.assertEqual(canon.parse_netlist(ref["netlist"])[1], [])
                if isinstance(ref["params"], dict):
                    self.assertIn(ref["class"], describe_ru.ROLES_ALL)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Admission(unittest.TestCase):
    def test_batch(self):
        rng = random.Random(6)
        refs = [filters_v3.sample("twin_t_active", rng) for _ in range(6)] + [filters_v3.sk_cascade(rng) for _ in range(10)]
        refs += [filters_v3.sample_named(t, rng) for t in ("khn", "twin_t", "mfb_hp")]
        for i, r in enumerate(refs):
            r["id"] = f"n{i:02d}"
        for r in pilot.admit(refs, ROOT / "out" / "test_filters_v3"):
            with self.subTest(r["id"], cls=r["class"]):
                self.assertTrue(r["admission"]["admitted"], r["admission"])
                self.assertTrue(spice2cir.build(r["netlist"], r["f_char"]))


if __name__ == "__main__":
    unittest.main()
