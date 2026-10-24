"""Регрессионные тесты: .CIR -> SPICE совпадает с эталонами по топологии и номиналам.

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl.cir2spice import convert  # noqa: E402
from mcnl.netlist_compare import compare  # noqa: E402

DATA, REF = ROOT / "tests" / "data", ROOT / "tests" / "reference"


class RoundTrip(unittest.TestCase):
    def check(self, name):
        text = (DATA / f"{name}.cir").read_text(encoding="cp1251", errors="replace")
        net, info = convert(text, name)
        self.assertEqual(info["errors"], [])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / f"{name}.ckt"
            out.write_text(net, encoding="utf-8")
            res = compare(REF / f"{name}.ckt", out)
        self.assertTrue(res["topology_match"], res)
        self.assertEqual(res["value_mismatch"], [])

    def test_sallen_key(self):
        self.check("Sallen-Key_High_Pass_Filter")

    def test_opamp1(self):
        self.check("OPAMP1")

    def test_rc_wires(self):
        self.check("RC_test_wires")

    def test_rc_labels(self):
        self.check("RC_test_labels")

    def test_digital_rejected(self):
        text = (DATA / "Counter.cir").read_text(encoding="cp1251", errors="replace")
        _, info = convert(text, "Counter")
        self.assertTrue(info["errors"])


if __name__ == "__main__":
    unittest.main()
