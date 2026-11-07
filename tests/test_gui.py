"""Данные для окна программы (docs/gui_spec.md): подписи, карточки, геометрия АЧХ, замечания. Без запуска окна."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcnl import gui  # noqa: E402


class Data(unittest.TestCase):
    def test_freq_label(self):
        self.assertEqual(gui.freq_label(1530), "1,53 кГц")
        self.assertEqual(gui.freq_label(12.3), "12,3 Гц")
        self.assertEqual(gui.freq_label(2.5e6), "2,5 МГц")
        self.assertEqual(gui.freq_label(None), "-")

    def test_cards(self):
        lp = {"type": "LP", "fc": 1070.0, "f0": 900.0, "k_pass_db": -0.0001, "Q": 0.74}
        self.assertEqual(gui.cards(lp), [("Тип", "ФНЧ"), ("fc", "1,07 кГц"), ("K", "0,0 дБ"), ("Q", "0,74")])
        bp = {"type": "BP", "fc": None, "f0": 5000.0, "k_pass_db": 6.02, "Q": 2.0}
        self.assertEqual(gui.cards(bp)[1], ("f0", "5 кГц"))
        flat = {"type": "flat", "fc": None, "f0": None, "k_pass_db": 20.0, "Q": None}
        self.assertEqual(gui.cards(flat)[3], ("Q", "-"))
        self.assertEqual(len(gui.cards(None)), 4)

    def test_plot_geometry(self):
        f = [10 ** (i / 10) for i in range(10, 61)]  # 10 Гц - 1 МГц
        db = [0.0 if fi < 1e3 else -20 * (len(str(int(fi))) - 4) for fi in f]
        g = gui.plot_geometry(f, db, 400, 200)
        self.assertEqual(len(g["lines"]), 1)
        xs, ys = g["lines"][0][0::2], g["lines"][0][1::2]
        self.assertEqual(len(xs), len(f))
        self.assertAlmostEqual(xs[0], 42, places=1)
        self.assertAlmostEqual(xs[-1], 390, places=1)
        self.assertTrue(all(10 <= y <= 176 for y in ys))
        self.assertEqual([lab for _, lab in g["xgrid"]], ["10 Гц", "100 Гц", "1 кГц", "10 кГц", "100 кГц", "1 МГц"])
        self.assertEqual(gui.plot_geometry([1], [0], 400, 200)["lines"], [])
        # спад глубже шкалы (80 дБ): кривая обрывается, а не ложится полкой на нижний край
        deep = gui.plot_geometry([10, 100, 1e3, 1e4], [0, 0, -120, -160], 400, 200)
        self.assertEqual([len(seg) for seg in deep["lines"]], [4])

    def test_notes(self):
        rep = {"refused": {"code": "out_of_class", "reason": "транзистор"}, "parse_errors": ["строка 2"],
               "messages": ["MC12 не выполнил схему (S1)"], "notes": ["R2: второй вывод никуда не подключён"]}
        self.assertEqual(gui.notes_of(rep), ["Модель отказала: транзистор", "Ошибка разбора: строка 2",
                                             "MC12 не выполнил схему (S1)", "Замечание: R2: второй вывод никуда не подключён"])

    def test_rules(self):
        kinds = [k for k, _ in gui.RULES]
        self.assertTrue(set(kinds) <= {"h", "p", "ex"})
        self.assertEqual(kinds[0], "h")
        self.assertGreaterEqual(kinds.count("ex"), 3)  # примеры для кнопки «В запрос»


if __name__ == "__main__":
    unittest.main()
