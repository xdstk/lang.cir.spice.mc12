"""Тесты инструментов человеческого теста (docs/human_test.md): импорт, разметка, сборка.

Работают в out/test_human (реальные data/human_test не трогают); сборка - с MC12 (иначе пропуск).
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import human_test, mc12run  # noqa: E402

WORK = ROOT / "out" / "test_human"


class Base(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(WORK, ignore_errors=True)
        WORK.mkdir(parents=True)
        self._saved = (human_test.DIR, human_test.RAW, human_test.LABELS)
        human_test.DIR, human_test.RAW, human_test.LABELS = WORK, WORK / "raw.jsonl", WORK / "labels.md"

    def tearDown(self):
        human_test.DIR, human_test.RAW, human_test.LABELS = self._saved

    def write_csv(self, name, rows):
        p = WORK / name
        p.write_text("prompt,note\n" + "".join(f'"{r}",\n' for r in rows), encoding="utf-8")
        return p


class ImportAndLabels(Base):
    def test_import_csv_and_txt(self):
        p = self.write_csv("a02.csv", ["RC ФНЧ: R 1к, C 100н; сигнал на вх", "усилитель на транзисторе"])
        self.assertEqual(len(human_test.import_file(p, "A02")), 2)
        self.assertEqual(len(human_test.import_file(p, "A02")), 0)  # повторный импорт не дублирует
        t = WORK / "a03.txt"
        t.write_text("делитель 10к/4,7к от 5 В\n\n", encoding="utf-8")
        added = human_test.import_file(t, "A03")
        self.assertEqual(added[0]["id"], "H0003")
        with self.assertRaises(ValueError):
            human_test.import_file(t, "Иванов")

    def test_single_text_file_with_authors(self):
        t = WORK / "all.txt"
        t.write_text("# от руководителя и студентов\nA02: RC ФНЧ: R 1к, C 100н\nA03: делитель 10к и 4,7к\n"
                     "без префикса - автор по умолчанию\nA02: RC ФНЧ: R 1к, C 100н\n", encoding="utf-8")
        added = human_test.import_file(t, "A01")
        self.assertEqual([(a["author"], a["prompt"]) for a in added],
                         [("A02", "RC ФНЧ: R 1к, C 100н"), ("A03", "делитель 10к и 4,7к"),
                          ("A01", "без префикса - автор по умолчанию")])

    def test_template_example_row_skipped(self):
        added = human_test.import_file(ROOT / "data" / "human_test" / "collect_template.csv", "A09")
        self.assertEqual(added, [])

    def test_propose_and_parse(self):
        p = self.write_csv("a02.csv", ["RC ФНЧ: R 1к, C 100н; сигнал на вх", "нарисуй что-нибудь"])
        human_test.import_file(p, "A02")
        self.assertEqual(len(human_test.propose()), 2)
        labels = human_test.parse_labels(human_test.LABELS.read_text(encoding="utf-8"))
        self.assertEqual(labels["H0001"]["class"], "A1")
        self.assertIn("R1 IN OUT 1k", labels["H0001"]["netlist"])
        self.assertEqual(labels["H0002"]["class"], "?")
        self.assertEqual(human_test.propose(), [])  # повторно не дописывает
        self.assertEqual(human_test.status()["by_class"], {"A1": 1, "?": 1})

    def test_reference_fields(self):
        lab = {"author": "A02", "prompt": "x", "class": "A1", "note": "",
               "netlist": "V1 IN 0 DC 0\nR1 IN N1 10k\nR2 N1 OUT 100k\nC1 N1 OUT 1n\nX1 0 N1 OUT OPAMP_IDEAL\n"}
        r = human_test.reference("H0007", lab)
        self.assertEqual((r["split"], r["level"], r["topology"]), ("human_test", "L2", "human:LP"))
        self.assertAlmostEqual(r["expected"]["fc"] / 1591.55, 1, delta=3e-3)
        self.assertEqual(human_test.reference("H0008", dict(lab, author="A01"))["split"], "human_dev")
        # запросы, написанные Claude, - синтетика: в human_test не попадают
        self.assertEqual(human_test.reference("H0009", dict(lab, author="C01"))["split"], "paraphrase")


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Build(Base):
    def test_build(self):
        human_test.import_file(self.write_csv("a01.csv", ["RC ФНЧ: R 1к, C 100н; сигнал на вх"]), "A01")
        human_test.import_file(self.write_csv("a02.csv", ["инвертирующее включение ОУ: Rin=4,7к, Rf=47к, вход 1 В",
                                                          "усилитель на КТ315"]), "A02")
        human_test.propose()
        text = human_test.LABELS.read_text(encoding="utf-8").replace("class: ?", "class: out")
        human_test.LABELS.write_text(text, encoding="utf-8")
        m = human_test.build(WORK / "v1", WORK / "runs")
        self.assertEqual(m["files"]["human_dev.jsonl"]["records"], 1)
        self.assertEqual(m["files"]["human_test.jsonl"]["records"], 1)
        self.assertEqual(m["report"]["skipped"], {"out": 1})
        rec = json.loads((WORK / "v1" / "human_test.jsonl").read_text(encoding="utf-8"))
        self.assertTrue(rec["admission"]["admitted"])
        self.assertEqual(rec["measured"]["type"], "flat")


if __name__ == "__main__":
    unittest.main()
