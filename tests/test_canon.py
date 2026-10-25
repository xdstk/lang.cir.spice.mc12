"""Тесты канонической формы и JSON-IR (docs/eval_spec.md, разделы 2-3).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon  # noqa: E402

DATA, REF = ROOT / "tests" / "data", ROOT / "tests" / "reference"

SK_LP = """* Саллен-Ки ФНЧ, повторитель, запись в произвольном порядке
X7 a OUT OUT opamp_ideal
C9 x OUT 10n
R5 x a 10K
C2 a 0 10n
R3 in x 10000
Vin in GND 0
.end
"""


class Values(unittest.TestCase):
    def test_parse(self):
        cases = {"10M": 1e-2, "10MEG": 1e7, "10meg": 1e7, "0.1u": 1e-7, "100n": 1e-7, "1000": 1e3,
                 "1k": 1e3, "4.7k": 4.7e3, "1e3": 1e3, "2.2uF": 2.2e-6, "470p": 4.7e-10, "-5": -5}
        for s, v in cases.items():
            self.assertAlmostEqual(canon.parse_value(s) / v, 1, places=12, msg=s)
        for bad in ("µ", "1,5k", "", "k10", "4k7"):
            self.assertIsNone(canon.parse_value(bad), bad)

    def test_format(self):
        cases = {1e7: "10MEG", 1e-2: "10m", 1e-7: "100n", 1e3: "1k", 4.7e3: "4.7k", 1e6: "1MEG",
                 1e9: "1G", 4.7e-10: "470p", 100: "100", 0: "0", -5: "-5", 2.2e-6: "2.2u",
                 999.9999: "1k", 1e5: "100k", 1.5e-11: "15p"}
        for v, s in cases.items():
            self.assertEqual(canon.format_value(v), s, v)

    def test_equivalent_spellings(self):
        for a, b in (("0.1u", "100n"), ("1000", "1k"), ("10MEG", "10meg"), ("1e-7", "100n")):
            self.assertEqual(canon.canon_value(a), canon.canon_value(b))
        self.assertNotEqual(canon.canon_value("10M"), canon.canon_value("10MEG"))  # M - милли


class Parse(unittest.TestCase):
    def test_spec_example_is_canonical(self):
        spec = "V1 IN 0 DC 5\nR1 IN OUT 1k\nC1 OUT 0 100n\n"
        self.assertEqual(canon.canonical_netlist(spec), spec)

    def test_source_forms(self):
        for tail, dc in (("DC 5", "5"), ("5", "5"), ("DC 0 AC 1 0", "0"), ("AC 1", "0"), ("", "0")):
            elems, err = canon.parse_netlist(f"V1 IN 0 {tail}\n")
            self.assertEqual(err, [], tail)
            self.assertEqual(elems[0]["value"], dc, tail)

    def test_errors(self):
        _, err = canon.parse_netlist("R1 IN OUT\nQ1 a b c MOD\n.TRAN 1u 1m\nX1 a b c LM741\nC1 a 0 1µ\n")
        self.assertEqual(len(err), 5)

    def test_reference_rc_circuits(self):
        # эталоны класса A из tests: RC-фильтры (ручные эталоны и rc_probe для MC12)
        for f in (REF / "RC_test_wires.ckt", REF / "RC_test_labels.ckt", DATA / "rc_probe.ckt"):
            text = f.read_text(encoding="cp1251")
            elems, err = canon.parse_netlist("\n".join(l for l in text.splitlines()[1:]
                                                       if not l.lstrip().startswith(".")))
            self.assertEqual(err, [], f.name)
            self.assertEqual(sorted(e["type"] for e in elems), ["C", "R", "V"], f.name)


class Canonicalize(unittest.TestCase):
    def test_sallen_key_shuffled(self):
        self.assertEqual(canon.canonical_netlist(SK_LP),
                         "V1 IN 0 DC 0\n"
                         "R1 IN N1 10k\n"
                         "R2 N1 N2 10k\n"
                         "C1 N2 0 10n\n"
                         "C2 N1 OUT 10n\n"
                         "X1 N2 OUT OUT OPAMP_IDEAL\n")

    def test_idempotent_and_deterministic(self):
        once = canon.canonical_netlist(SK_LP)
        self.assertEqual(canon.canonical_netlist(once), once)
        self.assertEqual(canon.canonical_netlist(SK_LP), once)

    def test_line_order_does_not_matter_without_symmetry(self):
        lines = [l for l in SK_LP.splitlines() if l and not l.startswith(("*", "."))]
        self.assertEqual(canon.canonical_netlist("\n".join(reversed(lines))), canon.canonical_netlist(SK_LP))

    def test_disconnected_part(self):
        out = canon.canonical_netlist("V1 IN 0 DC 1\nR1 IN 0 1k\nR2 a b 2k\n")
        self.assertIn("R2 N1 N2 2k", out)


class IR(unittest.TestCase):
    def test_roundtrip(self):
        for text in (SK_LP, "V1 IN 0 DC 5\nR1 IN OUT 1k\nC1 OUT 0 100n\n", "I1 IN 0 1m\nL1 IN OUT 10u\nR1 OUT 0 50\n"):
            can = canon.canonicalize(canon.parse_netlist(text)[0])
            ir = canon.to_ir(can)
            self.assertEqual(canon.validate_ir(ir), [])
            self.assertEqual(canon.to_netlist(canon.canonicalize(canon.from_ir(ir))), canon.to_netlist(can))

    def test_ir_value_spelling_normalized(self):
        ir = {"schema": "mcnl.ir/1", "elements": [{"type": "V", "nodes": ["IN", "0"], "dc": "5"},
                                                 {"type": "C", "nodes": ["IN", "OUT"], "value": "0.1u"}]}
        self.assertEqual(canon.from_ir(ir)[1]["value"], "100n")

    def test_validate_errors(self):
        errs = canon.validate_ir({"schema": "mcnl.ir/1", "elements": [
            {"type": "R", "nodes": ["IN"], "value": "1k"},
            {"type": "Q", "nodes": []},
            {"type": "C", "nodes": ["A", "0"], "value": "100 nF", "x": 1},
            {"type": "OPAMP", "nodes": ["A", "B", "OUT"], "value": "1"}]})
        self.assertEqual(len(errs), 5)
        self.assertTrue(canon.validate_ir({"elements": []}))


if __name__ == "__main__":
    unittest.main()
