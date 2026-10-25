"""Регрессионные тесты: .CIR -> SPICE совпадает с эталонами по топологии и номиналам.

Эталоны OPAMP1, Sallen-Key и *.mc12.ckt - вывод File > Translate самого MC12 12.2;
RC_test_* - ручные (схемы сгенерированы make_test_cir.py и проверены в MC12 по AC-анализу).
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl.cir2spice import convert  # noqa: E402
from mcnl.netlist_compare import compare  # noqa: E402

DATA, REF = ROOT / "tests" / "data", ROOT / "tests" / "reference"


MC12_DATA = Path(os.environ.get("MC12_DATA", r"C:\MC12\DATA"))


def check(case, cir_path, ref_path):
    net, info = convert(cir_path.read_text(encoding="cp1251", errors="replace"), cir_path.stem)
    case.assertEqual(info["errors"], [])
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / f"{cir_path.stem}.ckt"
        out.write_text(net, encoding="utf-8")
        res = compare(ref_path, out)
    case.assertTrue(res["topology_match"], res)
    case.assertEqual(res["value_mismatch"], [])


class RoundTrip(unittest.TestCase):
    def check(self, name):
        check(self, DATA / f"{name}.cir", REF / f"{name}.ckt")

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


@unittest.skipUnless(MC12_DATA.is_dir(), "нет примеров MC12 DATA")
class AgainstMC12Translate(unittest.TestCase):
    """Схемы из DATA против вывода File > Translate (25.10.2026): Rot 4/6 у MOSFET, Tie, Sine с RS,
    NFV, ключ S, порядок выводов библиотечной подсхемы, старый формат .CIR."""

    def test_mosdiff(self):
        check(self, MC12_DATA / "MOSDIFF.cir", REF / "MOSDIFF.mc12.ckt")

    def test_choke(self):
        check(self, MC12_DATA / "CHOKE.cir", REF / "CHOKE.mc12.ckt")

    def test_s_buck_cm(self):
        check(self, MC12_DATA / "S_BUCK_CM.cir", REF / "S_BUCK_CM.mc12.ckt")

    def test_bax(self):
        check(self, MC12_DATA / "BAX.cir", REF / "BAX.mc12.ckt")


if __name__ == "__main__":
    unittest.main()
