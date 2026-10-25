"""Тесты отдельных примитивов и старого формата .CIR на синтетических схемах.

Схемы собираются из минимальных секций, ответ известен заранее (не из вывода конвертера).
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl.cir2spice import convert  # noqa: E402

# выводы в шагах сетки, как в compdef примеров DATA
PINS = {
    "Resistor": [("Plus", 0, 0), ("Minus", 6, 0)],
    "Ground": [("PinA", 0, 0)],
    "Tie": [("PinA", 0, 0)],
    "Sine source": [("Plus", 0, 0), ("Minus", 0, 6)],
    "NFV": [("Plus", 0, 0), ("Minus", 0, 6)],
    "S": [("Plus Output", 0, 0), ("Minus Output", 0, 6), ("Plus Control", -4, 1), ("Minus Control", -4, 5)],
}


def cir(comps, text_area=(), old=False):
    """comps: [(definition, x, y, {атрибут: значение})]; old=True - формат ONV/Grid/Len."""
    out = []
    for d in sorted({c[0] for c in comps}):
        out += ["[compdef]", f"Name={d}", f"Definition={d}"]
        out += [f'Pin="{n}",{x},{y},0,0' for n, x, y in PINS[d]] + [""]
    for d, x, y, attrs in comps:
        out += ["[Comp]", f"Name={d}", f"Grid={x},{y}" if old else f"Px={x * 8},{y * 8}", "Rot=0", ""]
        for k, v in attrs.items():
            out += ["[Attr]", f"ONV=0,0,{k},{v}", ""] if old else ["[Attr]", f"ON=0,0,{k}", f"V={v}", ""]
    out += ["[Text Area]", "Page=1"]
    out += (["Len=1"] + list(text_area)) if old else [f"Text={t}" for t in text_area]
    return "\n".join(out) + "\n"


def body(net):
    return [l for l in net.splitlines() if l and not l.startswith("*")]


class Primitives(unittest.TestCase):
    def test_tie_joins_by_name(self):
        # R1: (0,0)-(6,0), R2: (20,0)-(26,0); Tie "A" на R1.Minus и на R2.Plus, земля на R1.Plus
        net, info = convert(cir([
            ("Resistor", 0, 0, {"PART": "R1", "RESISTANCE": "1k"}),
            ("Resistor", 20, 0, {"PART": "R2", "RESISTANCE": "2k"}),
            ("Tie", 6, 0, {"PART": "A"}),
            ("Tie", 20, 0, {"PART": "a"}),  # регистр имени не важен
            ("Ground", 0, 0, {}),
        ]))
        self.assertEqual(info["errors"], [])
        self.assertIn("R1 0 A 1k", body(net))
        self.assertTrue(any(l.startswith("R2 A ") for l in body(net)), net)

    def test_tie_without_name_is_error(self):
        _, info = convert(cir([("Resistor", 0, 0, {"PART": "R1", "RESISTANCE": "1k"}), ("Tie", 6, 0, {})]))
        self.assertTrue(info["errors"])

    def test_sine_source(self):
        # PH в радианах -> градусы; TAU -> THETA = 1/TAU; в AC-анализе 1 В
        net, info = convert(cir([("Sine source", 0, 0, {"PART": "V1", "MODEL": "SRC"}), ("Ground", 0, 6, {})],
                                [".MODEL SRC SIN (F=50 A=2 DC=1 PH=1.5707963 TAU=1m)"]))
        self.assertEqual(info["errors"], [])
        self.assertIn("V1 1 0 SIN(1 2 50 0 1000 90) AC 1", body(net))
        self.assertFalse(any(l.startswith(".MODEL SRC") for l in body(net)))

    def test_sine_missing_model(self):
        _, info = convert(cir([("Sine source", 0, 0, {"PART": "V1", "MODEL": "NOPE"}), ("Ground", 0, 6, {})]))
        self.assertTrue(any("SIN" in e for e in info["errors"]))

    def test_nfv(self):
        net, info = convert(cir([("NFV", 0, 0, {"PART": "E1", "VALUE": "V(In,Out)*2"}), ("Ground", 0, 6, {})]))
        self.assertEqual(info["errors"], [])
        self.assertIn("E1 1 0 VALUE = {V(In,Out)*2}", body(net))

    def test_switch(self):
        net, info = convert(cir([("S", 0, 0, {"PART": "S1", "MODEL": "SW"}), ("Ground", 0, 6, {})],
                                [".MODEL SW VSWITCH (RON=1 ROFF=1MEG VON=1 VOFF=0)"]))
        self.assertEqual(info["errors"], [])
        self.assertIn("S1 1 0 2 3 SW", body(net))
        self.assertIn(".MODEL SW VSWITCH (RON=1 ROFF=1MEG VON=1 VOFF=0)", body(net))

    def test_old_format_attrs_and_text_area(self):
        # ONV=x,y,ИМЯ,ЗНАЧЕНИЕ и сырой текст после Len= (так записаны 63 файла DATA)
        net, info = convert(cir([("Resistor", 0, 0, {"PART": "R1", "RESISTANCE": "10K"}), ("Ground", 0, 0, {}),
                                 ("NFV", 6, 0, {"PART": "E1", "VALUE": "V(A,B)"}), ("Ground", 6, 6, {})],
                                [".MODEL D0 D (IS=10F)"], old=True))
        self.assertEqual(info["errors"], [])
        self.assertIn("R1 0 1 10K", body(net))
        self.assertIn("E1 1 0 VALUE = {V(A,B)}", body(net))  # запятая внутри значения сохранена
        self.assertIn(".MODEL D0 D (IS=10F)", body(net))

    def test_missing_value_is_error(self):
        _, info = convert(cir([("Resistor", 0, 0, {"PART": "R1"}), ("Ground", 0, 0, {})]))
        self.assertTrue(any("RESISTANCE" in e for e in info["errors"]))


MC12_DATA = Path(os.environ.get("MC12_DATA", r"C:\MC12\DATA"))


@unittest.skipUnless((MC12_DATA / "CHOKE.cir").exists(), "нет примеров MC12 DATA")
class Choke(unittest.TestCase):
    """CHOKE.cir из поставки MC12: старый формат, два Sine source, модели в текстовой области."""

    def test_choke(self):
        net, info = convert((MC12_DATA / "CHOKE.cir").read_text(encoding="cp1251"), "CHOKE")
        self.assertEqual(info["errors"], [])
        lines = body(net)
        self.assertEqual(sum("SIN(0 100 50 0 0 0) AC 1" in l for l in lines), 2)
        self.assertIn("C1 0 B 2UF", lines)
        self.assertIn(".MODEL D0 D (IS=10F CJO=10P)", lines)


if __name__ == "__main__":
    unittest.main()
