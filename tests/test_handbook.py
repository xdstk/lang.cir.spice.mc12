"""Справочник фильтров (handbook.py, handbook_articles.py, вкладка окна): статьи, числа примеров и формулы против
узлового расчёта, файлы из MC12 в data/handbook и то, что статьи говорят о кривых MC12. Без MC12: файлы собирает
`python src/mcnl/handbook.py --build`.
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import cmath
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, handbook, mna, schematic  # noqa: E402
from mcnl.handbook_articles import ORDER  # noqa: E402
from mcnl.handbook_schematics import SCHEMATICS  # noqa: E402

FIELDS = ("id", "group", "title", "summary", "principle", "netlist", "opamp", "elements", "formulas", "properties",
          "use", "pros", "cons", "example", "check", "prompts")
TWO_PI = 2 * math.pi


def values(a):
    return {e["name"]: canon.parse_value(e["value"]) for e in canon.parse_netlist(a["netlist"])[0] if e["value"]}


def elems(a):
    return canon.parse_netlist(a["netlist"])[0]


def db(x):
    return 20 * math.log10(abs(x))


# Формулы статей на номиналах примеров: id -> v -> {величина: значение по формуле статьи}. Сравниваются с узловым
# расчётом (handbook.measured) - так проверяется, что формула в тексте статьи верна для схемы на чертеже.
FORMULAS = {
    "divider": lambda v: {"k_pass_db": db(v["R2"] / (v["R1"] + v["R2"]))},
    "rc_lp": lambda v: {"fc": 1 / (TWO_PI * v["R1"] * v["C1"])},
    "rc_hp": lambda v: {"fc": 1 / (TWO_PI * v["R1"] * v["C1"])},
    "rl_lp": lambda v: {"fc": v["R1"] / (TWO_PI * v["L1"])},
    "rl_hp": lambda v: {"fc": v["R1"] / (TWO_PI * v["L1"])},
    "rc_ladder2": lambda v: {
        "f0": 1 / (TWO_PI * math.sqrt(v["R1"] * v["R2"] * v["C1"] * v["C2"])),
        "Q": math.sqrt(v["R1"] * v["R2"] * v["C1"] * v["C2"]) / (v["R1"] * v["C1"] + v["R2"] * v["C2"]
                                                                 + v["R1"] * v["C2"]),
        "fc": 0.374 / (TWO_PI * v["R1"] * v["C1"])},
    "rc_ladder3": lambda v: {"fc": 0.194 / (TWO_PI * v["R1"] * v["C1"])},
    "twin_t": lambda v: {"f0": 1 / (TWO_PI * v["R1"] * v["C1"]), "Q": 0.25},
    "rlc_series": lambda v: {"f0": 1 / (TWO_PI * math.sqrt(v["L1"] * v["C1"])), "Q": math.sqrt(v["L1"] / v["C1"]) / v["R1"]},
    "rlc_parallel_bp": lambda v: {"f0": 1 / (TWO_PI * math.sqrt(v["L1"] * v["C1"])),
                                  "Q": v["R1"] * math.sqrt(v["C1"] / v["L1"])},
    "lc_ladder3": lambda v: {"k_pass_db": db(v["R2"] / (v["R1"] + v["R2"]))},
    "lc_ladder5": lambda v: {"k_pass_db": db(v["R2"] / (v["R1"] + v["R2"]))},
    "active_lp_inv": lambda v: {"k_pass_db": db(v["R2"] / v["R1"]), "fc": 1 / (TWO_PI * v["R2"] * v["C1"])},
    "active_hp_inv": lambda v: {"k_pass_db": db(v["R2"] / v["R1"]), "fc": 1 / (TWO_PI * v["R1"] * v["C1"])},
    "sallen_key_lp": lambda v: {
        "f0": 1 / (TWO_PI * math.sqrt(v["R1"] * v["R2"] * v["C1"] * v["C2"])),
        "Q": math.sqrt(v["R1"] * v["R2"] * v["C1"] * v["C2"]) / (v["C2"] * (v["R1"] + v["R2"]))},
    "sallen_key_hp": lambda v: {
        "f0": 1 / (TWO_PI * math.sqrt(v["R1"] * v["R2"] * v["C1"] * v["C2"])),
        "Q": math.sqrt(v["R1"] * v["R2"] * v["C1"] * v["C2"]) / (v["R1"] * (v["C1"] + v["C2"]))},
    "mfb_lp": lambda v: (lambda w0: {
        "f0": w0 / TWO_PI, "k_pass_db": db(v["R2"] / v["R1"]),
        "Q": w0 * v["C1"] / (1 / v["R1"] + 1 / v["R2"] + 1 / v["R3"])})(
        1 / math.sqrt(v["R2"] * v["R3"] * v["C1"] * v["C2"])),
    "mfb_bp": lambda v: (lambda f0: {
        "f0": f0, "Q": math.pi * f0 * v["R3"] * v["C1"], "k_pass_db": db(v["R3"] / (2 * v["R1"]))})(
        math.sqrt((v["R1"] + v["R2"]) / (v["R1"] * v["R2"] * v["R3"])) / (TWO_PI * v["C1"])),
    "mfb_hp": lambda v: (lambda w0: {
        "f0": w0 / TWO_PI, "k_pass_db": db(v["C1"] / v["C3"]),
        "Q": w0 * v["R2"] * v["C2"] * v["C3"] / (v["C1"] + v["C2"] + v["C3"])})(
        1 / math.sqrt(v["R1"] * v["R2"] * v["C2"] * v["C3"])),
    "twin_t_active": lambda v: {"f0": 1 / (TWO_PI * v["R1"] * v["C1"]),
                                "Q": 1 / (4 * (1 - v["R5"] / (v["R4"] + v["R5"])))},
    "tow_thomas": lambda v: (lambda w0: {
        "f0": w0 / TWO_PI, "Q": w0 * v["R3"] * v["C1"], "k_pass_db": db(v["R2"] / v["R1"])})(
        1 / math.sqrt(v["R2"] * v["R4"] * v["C1"] * v["C2"])),
    "khn": lambda v: (lambda w0: {
        "f0": w0 / TWO_PI,
        "Q": w0 * v["R5"] * v["C2"] * (v["R1"] + v["R2"]) / (v["R1"] * (1 + v["R6"] / v["R4"])),
        "k_pass_db": db((1 + v["R4"] / v["R6"]) * v["R2"] / (v["R1"] + v["R2"]))})(
        math.sqrt(v["R6"] / (v["R3"] * v["R4"] * v["R5"] * v["C1"] * v["C2"]))),
    "inverting": lambda v: {"k_pass_db": db(v["R2"] / v["R1"])},
    "noninverting": lambda v: {"k_pass_db": db(1 + v["R2"] / v["R1"])},
    "follower": lambda v: {"k_pass_db": 0.0},
    "integrator": lambda v: {"f_u": 1 / (TWO_PI * v["R1"] * v["C1"])},
    "integrator_rf": lambda v: (lambda fc, k: {"fc": fc, "k_dc_db": db(k), "f_u": fc * math.sqrt(k * k - 1)})(
        1 / (TWO_PI * v["R2"] * v["C1"]), v["R2"] / v["R1"]),
    "differentiator": lambda v: {"f_u": 1 / (TWO_PI * v["R1"] * v["C1"])},
    "tl071": lambda v: {"k_pass_db": db(v["R2"] / v["R1"])},
}
DB_KEYS = ("k_pass_db", "k_dc_db")


class Articles(unittest.TestCase):
    def test_structure(self):
        self.assertEqual([a["id"] for a in handbook.ARTICLES], ORDER)  # все схемы класса и статья об ОУ
        self.assertEqual(len(handbook.ARTICLES), 29)
        self.assertEqual(len(handbook.BY_ID), len(handbook.ARTICLES))
        groups = [a["group"] for a in handbook.ARTICLES]
        self.assertEqual(sorted(set(groups), key=groups.index), list(handbook.GROUPS))  # группы идут по порядку
        for a in handbook.ARTICLES:
            with self.subTest(a=a["id"]):
                self.assertEqual([f for f in FIELDS if f not in a], [])
                es = elems(a)
                names = {e["name"] for e in es} | {"+15 В", "−15 В"}
                for label, _ in a["elements"]:  # подписи элементов есть в схеме примера
                    self.assertTrue(set(x.strip() for x in label.split(",")) <= names, label)
                has_opamp = any(e["type"] == "OPAMP" for e in es)
                if a["opamp"] is None and has_opamp:  # идеальный ОУ на чертеже - только если реальный насытился бы
                    self.assertTrue(mna.saturation(es))
                else:
                    self.assertEqual(a["opamp"] is not None, has_opamp)
                self.assertTrue(a["prompts"] and all(m in (1, 2) for m, _ in a["prompts"]))
                self.assertNotIn("{", a["example"])

    def test_example_numbers(self):
        # числа раздела «Пример расчёта» (check) - узловой расчёт схемы примера с идеальным ОУ
        for a in handbook.ARTICLES:
            with self.subTest(a=a["id"]):
                m = handbook.measured(a)
                for k, want in a["check"].items():
                    if k == "type":
                        self.assertEqual(m["type"], want)
                    elif k in DB_KEYS:
                        self.assertAlmostEqual(m[k], want, delta=0.05)
                    else:
                        self.assertAlmostEqual(m[k] / want, 1, delta=0.005, msg=(k, m[k], want))

    def test_formulas(self):
        self.assertEqual(set(FORMULAS), set(handbook.BY_ID))
        for a in handbook.ARTICLES:
            with self.subTest(a=a["id"]):
                m = handbook.measured(a)
                for k, want in FORMULAS[a["id"]](values(a)).items():
                    if k in DB_KEYS:
                        self.assertAlmostEqual(m[k], want, delta=0.05)
                    else:
                        self.assertAlmostEqual(m[k] / want, 1, delta=0.01, msg=(k, m[k], want))

    def test_special_claims(self):
        # трёхзвенная RC-цепочка: на f = √6/(2πRC) фаза −180°, |K| = 1/29
        a = handbook.BY_ID["rc_ladder3"]
        v = values(a)
        h = mna.response(elems(a), math.sqrt(6) / (TWO_PI * v["R1"] * v["C1"]))
        self.assertAlmostEqual(abs(h) * 29, 1, delta=1e-4)
        self.assertAlmostEqual(abs(cmath.phase(h)), math.pi, delta=1e-4)
        # Tow-Thomas: на выходе первого ОУ - полосовой, K(f0) = −Rq/Rin
        a = handbook.BY_ID["tow_thomas"]
        v = values(a)
        bp = [dict(e, nodes=["OUT" if n == "N2" else "N9" if n == "OUT" else n for n in e["nodes"]]) for e in elems(a)]
        f0 = 1 / (TWO_PI * math.sqrt(v["R2"] * v["R4"] * v["C1"] * v["C2"]))
        h = mna.response(bp, f0)
        self.assertAlmostEqual(abs(h), v["R3"] / v["R1"], delta=1e-3)
        # двухзвенная цепочка с равными R и C: Q = 1/3
        self.assertAlmostEqual(FORMULAS["rc_ladder2"](values(handbook.BY_ID["rc_ladder2"]))["Q"], 1 / 3, places=9)
        # LC-лестница 5-го порядка (Баттерворт): на 2·fc ослабление около 30 дБ от уровня полосы
        a = handbook.BY_ID["lc_ladder5"]
        m = handbook.measured(a)
        drop = m["k_pass_db"] - db(mna.response(elems(a), 2 * m["fc"]))
        self.assertAlmostEqual(drop, 30, delta=1.5)


class Schematics(unittest.TestCase):
    """Схемы статей по ГОСТ (handbook_schematics.py): каждая изображает схему статьи с теми же обозначениями."""

    def test_every_drawing_is_the_article_circuit(self):
        self.assertEqual(set(SCHEMATICS), set(handbook.BY_ID))
        for a in handbook.ARTICLES:
            with self.subTest(a=a["id"]):
                sk = SCHEMATICS[a["id"]]()
                ok, drawn, ref = schematic.same_circuit(sk, a["netlist"])
                self.assertTrue(ok, f"\nрисунок {drawn}\nнетлист {ref}")
                # подписи раздела «Элементы» есть на рисунке (обозначения элементов или текст, например «+15 В»)
                shown = {n for _, n, _ in sk.parts} | {p[3] for p in sk.prims if p[0] == "text"}
                for label, _ in a["elements"]:
                    self.assertTrue(set(x.strip() for x in label.split(",")) <= shown, label)
                svg = schematic.to_svg(sk)
                self.assertTrue(svg.startswith("<svg") and svg.count("<line") > 5)

    def test_check_catches_wrong_drawing(self):
        # проверка не пустая: не тот символ, R1 и C1 поменяны местами, обрыв провода - всё ловится
        net = handbook.BY_ID["rc_lp"]["netlist"]
        s = schematic.Sketch()
        s.terminal((0, 0), "IN", "Uвх").terminal((40, 0), "OUT", "Uвых", side="right").rail(20, 0, 40)
        s.wire((0, 0), (5, 0)).C("R1", (5, 0), (20, 0)).wire((20, 0), (40, 0))
        s.two("R", "C1", (30, 0), (30, 20))
        self.assertFalse(schematic.same_circuit(s, net)[0])  # обозначения на месте, но символы не того типа
        s = schematic.Sketch()
        s.terminal((0, 0), "IN", "Uвх").terminal((40, 0), "OUT", "Uвых", side="right").rail(20, 0, 40)
        s.wire((0, 0), (5, 0)).R("C1", (5, 0), (20, 0)).wire((20, 0), (40, 0))
        s.C("R1", (30, 0), (30, 20))
        self.assertFalse(schematic.same_circuit(s, net)[0])  # R1 и C1 поменяны местами
        s = schematic.Sketch()
        s.terminal((0, 0), "IN", "Uвх").terminal((40, 0), "OUT", "Uвых", side="right").rail(20, 0, 40)
        s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).wire((25, 0), (40, 0))
        s.C("C1", (30, 0), (30, 20))
        self.assertFalse(schematic.same_circuit(s, net)[0])  # обрыв между R1 и узлом выхода
        with self.assertRaises(ValueError):
            schematic.Sketch().wire((0, 0), (5, 5))  # провод не по сетке


class Assets(unittest.TestCase):
    def test_files_from_mc12(self):
        for a in handbook.ARTICLES:
            with self.subTest(a=a["id"]):
                cir = handbook.DIR / f"{a['id']}.cir"
                self.assertEqual(cir.read_bytes().decode("ascii"), handbook.cir_text(a).replace("\r\n", "\n")
                                 .replace("\n", "\r\n"))  # файл в data совпадает с тем, что строит код
                self.assertEqual("TL071C" in cir.read_text(encoding="ascii"), a["opamp"] is not None)
                d = handbook.curve(a)
                self.assertEqual(len(d["f"]), len(d["db"]))
                self.assertEqual("db_ideal" in d, a["opamp"] is not None)
                self.assertEqual(json.loads((handbook.DIR / f"{a['id']}_ac.json").read_text(encoding="utf-8")), d)

    def test_curves_agree_with_text(self):
        # то, что статьи говорят о кривых MC12 с реальным ОУ
        def curve(aid):
            return handbook.curve(handbook.BY_ID[aid])

        def peak(d):
            k = max(range(len(d["db"])), key=d["db"].__getitem__)
            return d["f"][k], d["db"][k]
        d = curve("sallen_key_lp")
        k = min(range(len(d["db"])), key=d["db"].__getitem__)
        self.assertLess(abs(math.log10(d["f"][k] / 1e5)), 0.1)  # минимум около 100 кГц
        self.assertGreater(d["db"][-1], d["db"][k] + 15)  # затем поднимается
        self.assertLess(d["db_ideal"][-1], -115)  # с идеальным ОУ - нет
        self.assertLess(curve("mfb_lp")["db"][-1], d["db"][-1] - 40)  # МОС подавляет высокие частоты лучше
        d = curve("tl071")
        k = next(i for i, x in enumerate(d["db"]) if x < 17)  # срез усилителя ×10 по −3 дБ
        self.assertAlmostEqual(d["f"][k] / 288e3, 1, delta=0.03)
        f, top = peak(curve("differentiator"))  # пик дифференциатора около √(f_u·GBW) ≈ 22 кГц
        self.assertAlmostEqual(f / 22e3, 1, delta=0.05)
        f, top = peak(curve("mfb_hp"))  # резонансный подъём ФВЧ с МОС около 370 кГц
        self.assertAlmostEqual(f / 370e3, 1, delta=0.05)
        self.assertGreater(top, 15)
        self.assertLess(curve("active_hp_inv")["db"][-1], 10)  # инвертирующий ФВЧ с реальным ОУ - полосовой


if __name__ == "__main__":
    unittest.main()
