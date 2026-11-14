"""Датасет v3.1: LC-схемы без резистора, отбор записей v3, дефект Д5 режима 2, инструкция (docs/generator_v3_spec.md, «v3.1»).

Запуск из корня репозитория:  python -m unittest discover -s tests
Тесты без MC12; сборка с допуском в MC12 - generate_v31.py.
"""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, compose_v3, generate_v31, lc_v31, mode2_data, training  # noqa: E402


class LC(unittest.TestCase):
    def test_no_resistor_and_stable_quantities(self):
        rng = random.Random(1)
        for shape in lc_v31.SHAPES:
            ref = lc_v31.sample_lc(rng, shape)
            types = {e["type"] for e in canon.parse_netlist(ref["netlist"])[0]}
            self.assertEqual(types - {"V"}, {"L", "C"}, shape)
            self.assertEqual(ref["quantities"], ["type", "f0"])  # K и Q без потерь зависят от сетки

    def test_prompts_parse_back(self):
        rng = random.Random(2)
        for _ in range(300):
            ref = lc_v31.sample_lc(rng)
            task, family, text = lc_v31.make_prompt(ref, rng)  # make_prompt сам сверяет обратный разбор
            self.assertIn(task, ("A", "A1"))
            self.assertNotRegex(text, r"\bR\d")

    def test_pi_section_starts_with_shunt(self):
        ref = lc_v31.sample_lc(random.Random(3), "pi_lp")
        self.assertRegex(ref["netlist"], r"(?m)^C\d IN 0 ")

    def test_lossless_rejected_by_v3_rule(self):
        """Причина, по которой таких схем нет в v2 и v3: compose._reference отвергает их как резонанс."""
        from mcnl import compose
        secs = [("ser", "L", 330e-6), ("shunt", "C", 4.7e-9)]  # f0 около 128 кГц
        self.assertEqual(compose._reference(compose_v3.chain_netlist(secs), "x", {}, band=True)[1], "resonance")
        self.assertIsNotNone(lc_v31.lc_reference(compose_v3.chain_netlist(secs), "lc:g_lp", {})[0])


class Selection(unittest.TestCase):
    def test_protected(self):
        self.assertTrue(generate_v31.protected({"prompt": "rc фнч 1к 1н", "family": "terse", "topology": "rc_lp"}))
        self.assertTrue(generate_v31.protected({"prompt": "x" * 200, "family": "paraphrase", "topology": "rc_lp"}))
        self.assertTrue(generate_v31.protected({"prompt": "x" * 200, "family": "roles", "topology": "sallen_key_hp"}))
        self.assertFalse(generate_v31.protected({"prompt": "x" * 200, "family": "roles", "topology": "mfb_lp"}))

    def test_drop_only_unprotected(self):
        pool = [{"id": str(i), "prompt": "x" * (30 if i % 3 == 0 else 120), "family": "freeform", "topology": "rc_lp"}
                for i in range(30)]
        kept, gone = generate_v31.drop_long(pool, 10, random.Random(1))
        self.assertEqual(len(kept), 20)
        self.assertTrue(all(len(r["prompt"]) >= generate_v31.SHORT for r in pool if r["id"] in gone))
        with self.assertRaises(ValueError):
            generate_v31.drop_long(pool, 21, random.Random(1))


class Mode2(unittest.TestCase):
    def test_default_sampling_unchanged(self):
        """mode2_data.sample без новых параметров тратит rng как в v3: v3 воспроизводится."""
        a, b = random.Random(7), random.Random(7)
        for _ in range(15):
            kind = b.choices(list(mode2_data.MIX), list(mode2_data.MIX.values()))[0]
            self.assertEqual(mode2_data.sample(a)["output"], mode2_data.SAMPLERS[kind](b)["output"])

    def test_dc0_named_all(self):
        rng = random.Random(4)
        found = 0
        for _ in range(60):
            r = mode2_data.sample_named(rng, dc0=True)
            if r["family"] == "named_all":
                found += 1
                self.assertTrue(r["output"].startswith("V1 IN 0 DC 0\n"), r["output"])
        self.assertGreater(found, 0)

    def test_d5_detected(self):
        self.assertTrue(generate_v31.dc_defect({"output": "V1 IN 0 DC -5\nR1 IN OUT ?\n", "prompt": "ФНЧ RC; срез 1 кГц"}))
        self.assertFalse(generate_v31.dc_defect({"output": "V1 IN 0 DC -5\nR1 IN OUT ?\n", "prompt": "RC ФНЧ, на вх -5 В"}))
        self.assertFalse(generate_v31.dc_defect({"output": "V1 IN 0 DC 0\nR1 IN OUT ?\n", "prompt": "ФНЧ RC"}))

    def test_d5_in_v3_train(self):
        recs = [json.loads(l) for l in (ROOT / "data/dataset/v3/train.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(generate_v31.dc_defect(r) for r in recs if r.get("mode") == 2), 14)


class Instruction(unittest.TestCase):
    def test_gen31_uses_v3_instruction(self):
        rec = {"mode": 1, "gen": {"version": "gen-3.1"}}
        self.assertEqual(training.system_for(rec), training.SYSTEM_V3["netlist"])
        self.assertEqual(training.system_for(rec), training.instruction(rec, "netlist", "v3"))


if __name__ == "__main__":
    unittest.main()
