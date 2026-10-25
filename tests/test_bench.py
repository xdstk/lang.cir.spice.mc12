"""Тесты обвязки проверки (docs/eval_spec.md, раздел 4).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import bench, canon, mc12run, mcout  # noqa: E402

RC = "V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n"
# ФВЧ Саллена-Ки с усилением 2 на OPAMP_IDEAL (те же номиналы, что в tests/data/sk_ideal.ckt)
SK = ("V1 IN 0 DC 0\nC1 IN N1 1n\nC2 N1 N2 1n\nR1 N1 OUT 18.2k\nR3 N2 0 14k\n"
      "R4 OUT N3 10k\nR5 N3 0 10k\nX1 N2 N3 OUT OPAMP_IDEAL\n")


def elems(text):
    e, err = canon.parse_netlist(text)
    assert not err, err
    return e


class Build(unittest.TestCase):
    def test_rc(self):
        ckt = bench.build(elems(RC), f_char=1591.55, title="rc")
        lines = ckt.splitlines()
        self.assertEqual(lines[0], "* rc")
        self.assertIn("V1 IN 0 DC 0 AC 1", lines)
        self.assertIn(".AC DEC 100 1.59155 1.59155MEG", lines)
        self.assertIn(".PRINT AC VDB(OUT) VP(OUT)", lines)
        self.assertEqual(lines[-1], ".END")
        self.assertNotIn(".SUBCKT", ckt)

    def test_opamp_subckt_once(self):
        ckt = bench.build(elems(SK + "X2 N3 N2 0 OPAMP_IDEAL\n"))
        self.assertEqual(ckt.count(".SUBCKT OPAMP_IDEAL PLUS MINUS OUT"), 1)
        self.assertIn("E1 OUT 0 PLUS MINUS 1E6", ckt)

    def test_default_range_without_fchar(self):
        self.assertIn(".AC DEC 100 1 1MEG", bench.build(elems(RC)))

    def test_ac_only_on_input_source(self):
        ckt = bench.build(elems("V1 IN 0 DC 5\nV2 N1 0 DC 1\nR1 IN N1 1k\nR2 N1 OUT 1k\n"))
        self.assertIn("V1 IN 0 DC 5 AC 1", ckt)
        self.assertIn("V2 N1 0 DC 1\n", ckt)

    def test_written_as_ascii_crlf(self):
        p = ROOT / "out" / "test_bench" / "w.ckt"
        p.parent.mkdir(parents=True, exist_ok=True)
        bench.write(p, bench.build(elems(RC)))
        raw = p.read_bytes()
        self.assertIn(b"\r\n", raw)
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))  # только CRLF
        raw.decode("ascii")


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class RunInMC12(unittest.TestCase):
    """Готово по eval_spec 12.3: схемы, собранные через bench, дают прежние fc в MC12."""

    @classmethod
    def setUpClass(cls):
        cls.res = bench.run([("b_rc", elems(RC), 1591.55), ("b_sk", elems(SK), 9895.3)],
                            ROOT / "out" / "test_bench")

    def fc(self, name):
        r = self.res[name]
        self.assertTrue(r["ok"], r)
        o = mcout.read(r["output"])
        return mcout.cutoff_3db_from_max(mcout.column(o, "F"), mcout.column(o, "DB(V(OUT))"))

    def test_rc(self):
        self.assertAlmostEqual(self.fc("b_rc") / (1 / (2 * math.pi * 1e3 * 100e-9)), 1, delta=1e-3)

    def test_sallen_key_ideal_opamp(self):
        self.assertAlmostEqual(self.fc("b_sk") / 9895.3, 1, delta=1e-3)  # узловой расчёт, test_mc12run


if __name__ == "__main__":
    unittest.main()
