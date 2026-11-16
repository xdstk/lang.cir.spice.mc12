"""Выборка program - запросы, которые программа показывает пользователю (справочник, «Примеры», «Правила»), с эталонами
(program_prompts.py). Каждая новая модель должна проходить её на 100% (решение автора 16.11.2026).

- Split: файл data/program_prompts/program.jsonl совпадает с запросами программы; у каждого запроса есть эталон.
- InMC12: эталоны проходят сами (схемы считаются в MC12, ответы режима 2 решаются и попадают в цели).
- ModelAnswers: ответы модели - 100%. Запускается, если задана переменная MCNL_PROGRAM_ANSWERS - файлы ответов
  (predict.py ... --records data/program_prompts/program.jsonl), через ';' (os.pathsep).
Запуск из корня репозитория:  python -m unittest tests.test_program_prompts
"""
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import mc12run, program_prompts as pp, score  # noqa: E402
from mcnl.handbook_articles import ARTICLES  # noqa: E402


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


class Split(unittest.TestCase):
    def test_file_matches_program(self):
        # запросы программы поменялись - перегенерировать: python src/mcnl/program_prompts.py --export
        fresh = [json.loads(json.dumps(r, ensure_ascii=False, sort_keys=True)) for r in pp.records()]
        self.assertEqual(pp.read(), fresh)

    def test_every_program_prompt_is_checked(self):
        recs = pp.read()
        keys = {(r["mode"], r["prompt"]) for r in recs}
        self.assertEqual({(m, p) for m, p, _ in pp.program_prompts()}, keys)
        self.assertEqual(len({r["id"] for r in recs}), len(recs))
        self.assertTrue(all(r["sources"] for r in recs))
        # таблицы эталонов без лишних строк: каждая используется
        self.assertTrue(set(pp.REFERENCES) <= keys)
        mode2 = {a["id"] for a in ARTICLES if any(m == 2 for m, _ in a["prompts"])}
        self.assertEqual(set(pp.HANDBOOK_MODE2), mode2)
        self.assertGreaterEqual(sum(r["mode"] == 2 for r in recs), 20)


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class InMC12(unittest.TestCase):
    def test_references_pass(self):
        recs = pp.read()
        preds = [{"id": r["id"], "model": "reference", "format": "netlist",
                  "output": r["output"] if (r.get("refuse") or r["mode"] == 2) else r["netlist"]} for r in recs]
        rows, fails = pp.check(preds, ROOT / "out" / "test_program_prompts", recs)
        self.assertEqual(len(rows), len(recs))
        self.assertEqual(fails, [], pp.report(rows, fails, "эталоны"))


ANSWERS = [Path(p) for p in os.environ.get("MCNL_PROGRAM_ANSWERS", "").split(os.pathsep) if p]


@unittest.skipUnless(ANSWERS and mc12_available(), "MCNL_PROGRAM_ANSWERS не задана или нет MC12")
class ModelAnswers(unittest.TestCase):
    def test_model_passes_all_program_prompts(self):
        preds = [p for f in ANSWERS for p in score.read_jsonl(f)]
        name = preds[0].get("model", "model")
        rows, fails = pp.check(preds, ROOT / "out" / "test_program_prompts" / f"model_{name}")
        self.assertEqual(fails, [], pp.report(rows, fails, name))


if __name__ == "__main__":
    unittest.main()
