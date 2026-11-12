"""Проверка парафраз train v3 (docs/generator_v3_spec.md, «Парафразы v3»).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import generate_v3, paraphrase, paraphrase_v3  # noqa: E402

NAMED = {"mode": 1, "prompt": "150 Ом от входа к узлу a; 75 Ом от узла b на землю; 270 нФ от узла a к узлу b; 1к от узла b к выходу; сигнал на вх",
         "netlist": "V1 IN 0 DC 0\nR1 IN A 150\nR2 B 0 75\nR3 B OUT 1k\nC1 A B 270n\n", "expected": {"type": "LP"}}
NUMBERED = {"mode": 1, "prompt": "R1 10 кОм от входа к узлу 1; C1 100 нФ от узла 1 к выходу; R2 1 кОм от выхода на землю; сигнал на вх",
            "netlist": "V1 IN 0 DC 0\nR1 IN N1 10k\nR2 OUT 0 1k\nC1 N1 OUT 100n\n", "expected": {"type": "HP"}}
UNNAMED = {"mode": 1, "prompt": "1к в разрыв, 100н на землю; выход в конце цепочки",
           "netlist": "V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n", "expected": {"type": "LP"}}
MODE2 = {"mode": 2, "kind": "named", "unknown": ["R1"], "spec": {"type": "LP", "fc": 2000.0, "K": 5},
         "prompt": "активный ФНЧ (инв.): Rвх ?, Rос 10к, Cос 10н; срез 2 кГц, K = 5", "output": "V1 IN 0 DC 0\nR1 IN N1 ?\n.SPEC type=LP fc=2k K=5"}
SPEC = {"mode": 2, "kind": "spec", "unknown": [], "spec": {"type": "LP", "fc": 1000.0, "order": 4, "active": "yes"},
        "prompt": "подбери активный ФНЧ 4-го порядка на 1 кГц", "output": ".SPEC type=LP fc=1k order=4 active=yes"}


class NodeNames(unittest.TestCase):
    def test_names(self):
        f = paraphrase_v3.node_names
        self.assertEqual(f("от узла a к точке B, затем в узел 3 и т.4, у5"), {"A", "B", "N3", "N4", "N5"})
        self.assertEqual(f("Узел K и общий провод; Точка 2 и выход"), {"K", "N2"})       # слово с заглавной
        self.assertEqual(f("между узлом С и узлом A"), {"C", "A"})                        # кириллическая «С» - тот же узел
        self.assertEqual(f("от узла 1 к выходу, от точки 2 на землю"), {"N1", "N2"})     # «1 к» - не 1 кОм
        self.assertEqual(f("к отдельному узлу 1,5к, с этого узла 39н на землю"), set())  # номинал после слова «узел»
        self.assertEqual(f("с узла 16 кОм на землю, в узел и обратно"), set())


class Check(unittest.TestCase):
    def ok(self, rec, new):
        self.assertEqual(paraphrase_v3.check(rec, new), [], new)

    def bad(self, rec, new, word):
        errs = paraphrase_v3.check(rec, new)
        self.assertTrue(any(word in e for e in errs), (new, errs))

    def test_named_nodes_must_stay(self):
        self.ok(NAMED, "150 Ом со входа в узел a, из узла a 270 нФ в узел b, с узла b 75 Ом на землю и 1к на выход")
        self.ok(NAMED, "150 Ом от входа в a, 270 нФ из a в b, с b на землю 75 Ом, из b на выход 1к")   # без слова «узел»
        self.bad(NAMED, "150 Ом со входа в узел a, из узла a 270 нФ в узел c, с узла c 75 Ом на землю и 1к на выход", "узлы потеряны")
        self.bad(NAMED, "150 Ом со входа в узел a, из узла a 270 нФ в узел b, с узла b 75 Ом на землю и 1к на выход, узел d свободен",
                 "новые имена")

    def test_numbered_nodes(self):
        self.ok(NUMBERED, "10 кОм от входа в узел 1, из узла 1 100 нФ на выход, на выходе 1 кОм на землю")
        self.ok(NUMBERED, "10 кОм от входа в т1, из т1 100 нФ на выход, на выходе 1 кОм на землю")
        self.bad(NUMBERED, "10 кОм от входа в узел 2, из узла 2 100 нФ на выход, на выходе 1 кОм на землю", "узлы потеряны")

    def test_no_new_names_for_unnamed_nodes(self):
        self.ok(UNNAMED, "резистор 1к последовательно и конденсатор 100н с выхода на землю")
        self.bad(UNNAMED, "1к от входа в узел A, из узла A на выход, 100н на землю", "новые имена")

    def test_values_and_voltage(self):
        self.bad(UNNAMED, "1,5к последовательно и 100н с выхода на землю", "номиналы")
        self.bad(UNNAMED, "1к последовательно, 100н с выхода на землю, на входе 5 В", "напряжение")

    def test_filter_name(self):
        self.bad(UNNAMED, "ФВЧ: 1к последовательно и 100н с выхода на землю", "название")

    def test_mode2(self):
        self.ok(MODE2, "инвертирующий активный ФНЧ со срезом 2 кГц и усилением 5: Rос 10к, Cос 10н, входной резистор подобрать")
        self.bad(MODE2, "инвертирующий активный ФНЧ со срезом 2 кГц и усилением 5: Rвх, Rос 10к, Cос 10н", "подбираемых")
        self.bad(MODE2, "инвертирующий активный ФНЧ со срезом 2 кГц: Rос 10к, Cос 10н, входной резистор подобрать", ".SPEC K")
        self.bad(MODE2, "инвертирующий активный ФНЧ со срезом 3 кГц и усилением 5: Rос 10к, Cос 10н, входной резистор подобрать", "номиналы")

    def test_spec_cues(self):
        self.ok(SPEC, "ФНЧ 4-го порядка на ОУ, срез 1 кГц")
        self.bad(SPEC, "ФНЧ 4-го порядка, срез 1 кГц", ".SPEC active")
        self.bad(SPEC, "активный ФНЧ на 1 кГц", ".SPEC order")
        self.bad(SPEC, "активный ФНЧ Баттерворта 4-го порядка на 1 кГц", ".SPEC response")

    def test_length_and_refusals(self):
        self.bad(UNNAMED, "1к в разрыв, 100н на землю " + "и так далее " * 60, "длина")
        self.assertEqual(paraphrase_v3.check({"mode": 1, "refuse": "out_of_class", "prompt": "счётчик на D-триггерах"},
                                             "двоичный счётчик на триггерах"), [])


class Parts(unittest.TestCase):
    def test_files(self):
        """Части парафраз: «id, табуляция, текст», id не повторяются, текст не пуст и не длиннее предела."""
        folder = ROOT / "data" / "paraphrase_v3"
        ids = []
        for f in sorted(folder.glob("part*.tsv")):
            for line in f.read_text(encoding="utf-8").splitlines():
                rid, text = line.split("\t", 1)
                self.assertRegex(rid, r"^train-\d{5}$")
                self.assertTrue(text.strip())
                self.assertLessEqual(len(text), generate_v3.MAX_PROMPT)
                self.assertNotIn("\t", text)
                ids.append(rid)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 605)
        self.assertEqual(set(paraphrase.read_tsv(folder)), set(ids))

    def test_frozen_dataset_paraphrases_pass_check(self):
        """После заморозки: каждая применённая парафраза проходит проверку против своего шаблона."""
        train = ROOT / "data" / "dataset" / "v3" / "train.jsonl"
        if not train.exists():
            self.skipTest("датасет v3 ещё не заморожен")
        para = paraphrase.read_tsv(ROOT / "data" / "paraphrase_v3")
        n = 0
        for r in paraphrase.read_jsonl(train):
            if r.get("family") == "paraphrase":
                n += 1
                self.assertEqual(paraphrase_v3.check(dict(r, prompt=r["prompt_template"]), para[r["id"]]), [], r["id"])
        self.assertEqual(n, len(para))


if __name__ == "__main__":
    unittest.main()
