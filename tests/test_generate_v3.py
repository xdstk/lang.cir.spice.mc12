"""Сборка датасета v3 (docs/generator_v3_spec.md): пробная партия с MC12, без него пропускается.

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import random
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, generate_v2, generate_v3, iso, mc12run, mode2_data, score, training  # noqa: E402

WORK = ROOT / "out" / "test_gen3"


def elements(net):
    return canon.parse_netlist(net)[0]


class Plan(unittest.TestCase):
    def test_shares(self):
        c = Counter(generate_v3.plan(1000, random.Random(1)))
        self.assertEqual(sum(c.values()), 1000)
        self.assertEqual(c["mode2"], 180)                       # 15-20% - режим 2
        self.assertEqual(c["addon"], round(820 * 0.15))         # около 12% всех записей (П5)
        self.assertEqual(c["multi"], round(820 * 0.06))         # около 5% (П2)
        self.assertEqual(c["mismatch"], 8)                      # около 1% (П7)
        self.assertEqual(c["refuse"], 41)
        self.assertAlmostEqual(sum(generate_v3.KINDS.values()), 1.0)
        self.assertNotIn("mode2", generate_v3.plan(100, random.Random(1), mode2=False))

    def test_path_family_in_train(self):
        self.assertIn("path", generate_v3.FAMILY_A)
        self.assertAlmostEqual(sum(generate_v3.FAMILY_A.values()), 1.0)

    def test_excluded_circuits(self):
        """Исключаются схемы llm_test, llm_test2, mc12_real и C01."""
        ex = generate_v3.excluded_circuits()
        self.assertGreater(len(ex), len(generate_v2.excluded_circuits()))
        for name in ("llm_test_v1/llm_test.jsonl", "llm_test2_v1/llm_test2.jsonl", "mc12_real_v1/mc12_real.jsonl"):
            recs = [json.loads(l) for l in (ROOT / "data/dataset" / name).read_text(encoding="utf-8").splitlines()]
            ids = {r["circuit_id"] for r in recs if r.get("circuit_id")}
            self.assertTrue(ids and ids <= ex, name)

    def test_v2_generator_unchanged(self):
        self.assertEqual(generate_v2.VERSION, "gen-2")
        self.assertEqual(generate_v2.HELD_OUT, {"khn", "twin_t", "mfb_hp"})
        self.assertNotIn("llm_test2", str(generate_v2.excluded_circuits.__code__.co_consts))


class Prompts(unittest.TestCase):
    def test_mutate_prefix_share(self):
        rng = random.Random(2)
        n = sum("prefix" in generate_v3.mutate("RC ФНЧ: R 1к, C 100н", rng)[1] for _ in range(3000))
        self.assertAlmostEqual(n / 3000, generate_v3.PREFIX_SHARE, delta=0.015)
        for _ in range(200):
            self.assertFalse({"prefix", "suffix"} & set(generate_v3.mutate("нужен ФНЧ на 1 кГц", rng, mode2=True)[1]))

    def test_connections_answer_matches_prompt_names(self):
        rng = random.Random(3)
        net = "V1 IN 0 DC 0\nR1 IN N1 1k\nR2 N1 N2 2.2k\nC1 N1 0 100n\nC2 N2 OUT 10n\nR3 OUT 0 10k\n"
        named = fams = 0
        seen = set()
        for _ in range(300):
            fam, prompt, answer, info = generate_v3.connections(net, rng)
            seen.add(fam)
            self.assertTrue(iso.match(elements(net), elements(answer), "values")["match"])
            if info.get("names") not in (None, "seq"):
                named += 1
            fams += fam != "brief"
        self.assertEqual(seen, set(generate_v3.FAMILY_A))
        self.assertAlmostEqual(named / fams, generate_v3.NAMED_SHARE, delta=0.1)

    def test_long_prompts_fall_back_to_compact_family(self):
        """Запрос не длиннее MAX_PROMPT: большая схема описывается более кратким семейством (замечание автора)."""
        rng = random.Random(6)
        fams = set()
        for _ in range(12):
            ref = generate_v3.sample_circuit("multi", rng, "train")
            for _ in range(6):
                fam, prompt, answer, _info = generate_v3.connections(ref["netlist"], rng)
                fams.add(fam)
                self.assertLessEqual(len(prompt), generate_v3.MAX_PROMPT, (fam, prompt))
                self.assertTrue(iso.match(elements(ref["netlist"]), elements(answer), "values")["match"])
        self.assertIn("freeform", fams)
        self.assertEqual(generate_v3.MAX_PROMPT, mode2_data.MAX_PROMPT)

    def test_mismatch_follows_connections(self):
        rng = random.Random(4)
        ref = generate_v3.sample_circuit("mismatch", rng, "train")
        task, fam, prompt, answer, info = generate_v3.make_prompt(ref, "mismatch", "train", rng)
        self.assertEqual(answer, ref["netlist"])
        self.assertEqual(info["named_as"], generate_v3.MISMATCH[ref["class"]])

    def test_needs_design_reason(self):
        rng = random.Random(5)
        recs = [generate_v3.refusal_record(rng, "x", "train", 1) for _ in range(40)]
        design = [r for r in recs if r["refuse"] == "needs_design"]
        self.assertTrue(design)
        for r in design:
            self.assertEqual(r["output"], "REFUSE needs_design: номиналы не заданы, расчёт номиналов - режим 2")
            self.assertEqual(score.parse_refusal(r["output"])[0], "needs_design")


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Trial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)
        cls.m = generate_v3.generate(WORK / "ds", seed=3, scale=0.03, workdir=WORK / "runs")
        cls.recs = {s: [json.loads(l) for l in (WORK / "ds" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
                    for s in generate_v3.SPLITS}

    def test_splits_filled(self):
        for s, n in generate_v3.SPLITS.items():
            self.assertGreaterEqual(len(self.recs[s]), max(1, round(n * 0.03)) - 3, s)
        self.assertNotIn("gen_topology", self.recs)

    def test_circuits_do_not_repeat(self):
        ids = [r["circuit_id"] for rs in self.recs.values() for r in rs if r["circuit_id"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & generate_v3.excluded_circuits())

    def test_mode1_records(self):
        for r in self.recs["train"] + self.recs["test"]:
            if r["mode"] != 1 or r.get("refuse"):
                continue
            self.assertEqual(r["gen"]["version"], "gen-3")
            self.assertTrue(r["admission"]["admitted"] and r["admission"]["cir"], r["id"])
            self.assertEqual(r["circuit_id"], generate_v3.circuit_id(r["canonical"]))
            # ответ модели - та же схема, что канон; имена узлов могут быть из запроса
            self.assertTrue(iso.match(elements(r["canonical"]), elements(r["netlist"]), "values")["match"], r["id"])
            self.assertEqual(r["output"], r["netlist"])
            self.assertEqual(training.answer_for(r), r["netlist"].strip())
            self.assertIn("узел 3 - N3", training.system_for(r))

    def test_mode2_records(self):
        recs = [r for r in self.recs["train"] + self.recs["test"] + self.recs["dev"] if r["mode"] == 2]
        self.assertTrue(recs)
        total = len(self.recs["train"])
        self.assertAlmostEqual(sum(r["mode"] == 2 for r in self.recs["train"]) / total, generate_v3.MODE2_SHARE, delta=0.06)
        for r in recs:
            self.assertIn(".SPEC type=", training.system_for(r))
            self.assertEqual(training.answer_for(r), r["output"].strip())
            if r["kind"] == "refuse":
                self.assertTrue(r["output"].startswith("REFUSE "))
                continue
            self.assertIsNotNone(mode2_data.solve(r["output"]), r["id"])  # ответ решается через synth.run
            if r["solution"]["route"] != "designer":
                self.assertTrue(r["admission"]["admitted"] and r["admission"]["cir"], r["id"])
        self.assertFalse([r for r in self.recs["gen_family"] + self.recs["gen_range"] if r["mode"] == 2])

    def test_refusal_records(self):
        for r in self.recs["train"]:
            if r.get("refuse"):
                self.assertEqual(score.parse_refusal(r["output"])[0], r["refuse"])

    def test_subset_nested_and_export(self):
        self.assertEqual({r["subset"] for r in self.recs["train"]} - set(generate_v3.SUBSETS), set())
        rep = training.export(WORK / "ds", WORK / "export", "netlist")
        self.assertEqual(rep["train"]["records"], len(self.recs["train"]))
        rows = [json.loads(l) for l in (WORK / "export" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        systems = {row["messages"][0]["content"] for row in rows}
        self.assertEqual(systems, {training.SYSTEM_V3["netlist"], training.SYSTEM_V3["mode2"]})
        ir = training.export(WORK / "ds", WORK / "export_ir", "ir")  # режим 2 - только нетлист
        self.assertEqual(ir["train"]["records"], sum(r["mode"] == 1 for r in self.recs["train"]))


if __name__ == "__main__":
    unittest.main()
