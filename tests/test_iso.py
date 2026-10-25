"""Тесты сравнения через изоморфизм (docs/eval_spec.md, разделы 5 и 8).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import iso  # noqa: E402

# активный ФНЧ 1-го порядка на инвертирующем включении: K = -R2/R1, fc = 1/(2 pi R2 C1)
REF = """V1 IN 0 DC 1
R1 IN N1 10k
R2 N1 OUT 1MEG
C1 N1 OUT 1n
R3 OUT 0 10k
X1 0 N1 OUT OPAMP_IDEAL
"""


def cmp(test):
    return iso.compare_text(REF, test)


class ShouldMatch(unittest.TestCase):
    """Уровень S3 (топология и номиналы) должен засчитываться."""

    def assertS3(self, test):
        r = cmp(test)
        self.assertTrue(r.get("topology") and r.get("values"), r)

    def test_identical(self):
        self.assertS3(REF)

    def test_renamed_nodes_and_elements(self):
        self.assertS3("Vs IN 0 DC 1\nRa IN inv 10k\nRf inv OUT 1MEG\nCf inv OUT 1n\nRl OUT 0 10k\nU1 0 inv OUT OPAMP_IDEAL\n"
                      .replace("U1", "XU1"))

    def test_shuffled_lines(self):
        self.assertS3("\n".join(reversed(REF.strip().splitlines())))

    def test_swapped_passive_pins(self):
        self.assertS3(REF.replace("R2 N1 OUT 1MEG", "R2 OUT N1 1MEG").replace("R3 OUT 0", "R3 0 OUT"))

    def test_other_value_spelling(self):
        self.assertS3(REF.replace("10k", "10000").replace("1n", "0.001u").replace("1MEG", "1e6"))

    def test_gnd_alias(self):
        self.assertS3(REF.replace(" 0 ", " GND ").replace("X1 0 ", "X1 gnd "))


class ShouldNotMatch(unittest.TestCase):
    def test_reversed_source(self):
        r = cmp(REF.replace("V1 IN 0", "V1 0 IN"))
        self.assertFalse(r["topology"])

    def test_swapped_opamp_inputs(self):
        r = cmp(REF.replace("X1 0 N1 OUT", "X1 N1 0 OUT"))
        self.assertFalse(r["topology"])

    def test_milli_instead_of_mega(self):
        r = cmp(REF.replace("1MEG", "1M"))  # 1M = 1 мОм: ошибка в 1e9
        self.assertTrue(r["topology"])
        self.assertFalse(r["values"])
        self.assertEqual(r["unmatched_ref"], ["R2"])

    def test_milli_instead_of_kilo(self):
        r = cmp(REF.replace("R1 IN N1 10k", "R1 IN N1 10m"))
        self.assertTrue(r["topology"])
        self.assertFalse(r["values"])
        self.assertEqual(r["unmatched_ref"], ["R1"])

    def test_extra_element(self):
        r = cmp(REF + "R9 N1 0 100k\n")
        self.assertFalse(r["topology"])
        self.assertEqual(r["unmatched_test"], ["R9"])

    def test_missing_element(self):
        r = cmp(REF.replace("C1 N1 OUT 1n\n", ""))
        self.assertFalse(r["topology"])
        self.assertEqual(r["unmatched_ref"], ["C1"])

    def test_element_on_neighbour_node(self):
        r = cmp(REF.replace("C1 N1 OUT 1n", "C1 IN OUT 1n"))
        self.assertFalse(r["topology"])

    def test_two_series_resistors_are_not_s3(self):
        # функционально то же (S4), но другая схема: S3 не засчитывается
        r = cmp(REF.replace("R1 IN N1 10k", "R1 IN M 5k\nR4 M N1 5k"))
        self.assertFalse(r["topology"])

    def test_parse_error_reported(self):
        self.assertIn("parse_errors", cmp(REF + "Q1 a b c NPN\n"))


class Search(unittest.TestCase):
    def test_symmetric_ladder(self):
        # симметричная RC-лестница: много равных меток, поиск должен справиться с перестановками
        ref = "V1 IN 0 DC 0\n" + "".join(f"R{i} {'IN' if i == 1 else f'N{i-1}'} N{i} 1k\nC{i} N{i} 0 1n\n"
                                         for i in range(1, 9)) + "R9 N8 OUT 1k\nC9 OUT 0 1n\n"
        test = "\n".join(reversed(ref.strip().splitlines()))
        self.assertTrue(iso.compare_text(ref, test)["values"])


if __name__ == "__main__":
    unittest.main()
