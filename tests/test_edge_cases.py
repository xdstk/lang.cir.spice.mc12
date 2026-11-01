"""Странные схемы и отказы v2 (docs/generator_v2_spec.md, разделы 2.3-2.4).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import app, canon, edge_cases, iso, mc12run, pilot, prompts_v2, score  # noqa: E402


def elements(net):
    return canon.parse_netlist(net)[0]


class Strange(unittest.TestCase):
    def test_kinds_are_literal(self):
        rng = random.Random(1)
        for kind in edge_cases.STRANGE:
            for _ in range(15):
                ref = edge_cases.strange(kind, rng)
                self.assertEqual(ref["class"], f"strange:{kind}")
                back = prompts_v2.parse_freeform(edge_cases.strange_prompt(ref, rng))
                self.assertTrue(iso.match(elements(ref["netlist"]), back, "values")["match"], ref["netlist"])

    def test_dangling_has_single_pin_node(self):
        ref = edge_cases.strange("dangling", random.Random(2))
        nodes = [n for e in elements(ref["netlist"]) for n in e["nodes"]]
        self.assertTrue(any(nodes.count(n) == 1 and n not in ("IN", "OUT", "0") for n in nodes))

    def test_source_shunt(self):
        ref = edge_cases.strange("source_shunt", random.Random(3))
        self.assertTrue(any(e["type"] in "RCL" and set(e["nodes"]) == {"IN", "0"} for e in elements(ref["netlist"])))


class Notes(unittest.TestCase):
    def test_notes(self):
        rng = random.Random(7)
        for kind, word in (("dangling", "не подключён"), ("source_shunt", "на источнике")):
            ref = edge_cases.strange(kind, rng)
            self.assertTrue(any(word in n for n in app.circuit_notes(elements(ref["netlist"]))), ref["netlist"])

    def test_clean_circuit_has_none(self):
        clean = "\n".join(["V1 IN 0 DC 0", "R1 IN OUT 1k", "C1 OUT 0 100n", ""])
        self.assertEqual(app.circuit_notes(elements(clean)), [])


class Refusals(unittest.TestCase):
    def test_every_code(self):
        rng = random.Random(4)
        for code in score.REFUSE_CODES:
            for _ in range(60):
                r = edge_cases.refusal(code, rng)
                self.assertNotIn("{", r["prompt"])
                self.assertEqual(score.parse_refusal(r["output"]), (code, r["reason"]))

    def test_ir_output(self):
        out = edge_cases.refusal_output("ambiguous", "неясно", "ir")
        self.assertEqual(json.loads(out)["refuse"], "ambiguous")
        self.assertEqual(score.parse_refusal(out, "ir"), ("ambiguous", "неясно"))

    def test_llm_test_wording_not_copied(self):
        recs = [json.loads(l) for l in (ROOT / "data/dataset/llm_test_v1/llm_test.jsonl").read_text(encoding="utf-8").splitlines()]
        frozen = {r["prompt"].lower() for r in recs if r.get("refuse")}
        rng = random.Random(5)
        for code in score.REFUSE_CODES:
            for _ in range(200):
                self.assertNotIn(edge_cases.refusal(code, rng)["prompt"].lower(), frozen)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Mc12(unittest.TestCase):
    def test_strange_run(self):
        rng = random.Random(6)
        refs = []
        for kind in edge_cases.STRANGE:
            for _ in range(3):
                r = edge_cases.strange(kind, rng)
                r["id"] = f"s{len(refs):02d}"
                refs.append(r)
        out = pilot.admit(refs, ROOT / "out" / "test_edge")
        self.assertTrue(all(r["admission"]["mc12_run"] for r in out),
                        [(r["class"], r["admission"]["run_messages"]) for r in out if not r["admission"]["mc12_run"]])


if __name__ == "__main__":
    unittest.main()
