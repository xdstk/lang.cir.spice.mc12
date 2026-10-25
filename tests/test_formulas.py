"""Тесты формул топологий L1/L2, узлового расчёта и допуска эталонов (eval_spec, разделы 9-10).

Формулы сверяются с численным узловым расчётом (mna.py) без MC12; допуск пилота - с MC12
(пропускается, если его нет). Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, formulas, mc12run, mna, pilot  # noqa: E402


class Mna(unittest.TestCase):
    def test_inverting_amp(self):
        h = mna.response(canon.parse_netlist("V1 IN 0 DC 1\nR1 IN N1 1k\nR2 N1 OUT 10k\nX1 0 N1 OUT OPAMP_IDEAL\n")[0], 1e3)
        self.assertAlmostEqual(h.real, -10 / (1 + 11 / mna.OPAMP_GAIN), places=9)

    def test_current_source_sign(self):
        # SPICE: ток I1 IN 0 течёт из IN через источник в 0 -> V(IN) = -1 * (1k || 2k)
        h = mna.response(canon.parse_netlist("I1 IN 0 DC 0\nR1 IN 0 1k\nR2 IN OUT 1k\nR3 OUT 0 1k\n")[0], 1)
        self.assertAlmostEqual(h.real, -1000 / 3, places=6)

    def test_rc_lp(self):
        e = canon.parse_netlist("V1 IN 0 DC 0\nR1 IN OUT 1k\nC1 OUT 0 100n\n")[0]
        fc = 1 / (2 * math.pi * 1e-4)
        self.assertAlmostEqual(abs(mna.response(e, fc)), 1 / math.sqrt(2), places=12)


class FormulasVsNodal(unittest.TestCase):
    def test_all_pilot_topologies(self):
        for rid, topo, params in pilot.PILOT:
            ref = formulas.reference(topo, params, rid)
            with self.subTest(ref=rid):
                ok, detail = pilot.compare(ref["expected"], pilot.mna_measure(ref), ref["quantities"])
                self.assertTrue(ok, detail)

    def test_every_topology_in_pilot(self):
        self.assertEqual({t for _, t, _ in pilot.PILOT}, set(formulas.TOPOLOGIES))

    def test_wrong_formula_is_rejected(self):
        ref = formulas.reference("sallen_key_hp", {"R1": 18e3, "R2": 15e3, "C1": 1e-9, "C2": 1e-9,
                                                   "Ra": 10e3, "Rb": 10e3})
        bad = dict(ref["expected"], Q=ref["expected"]["Q"] * 1.02)  # ошибка формулы на 2%
        self.assertFalse(pilot.compare(bad, pilot.mna_measure(ref), ref["quantities"])[0])
        bad = dict(ref["expected"], f0=ref["expected"]["f0"] * 1.002)  # на 0,2% - больше допуска 0,1%
        self.assertFalse(pilot.compare(bad, pilot.mna_measure(ref), ref["quantities"])[0])

    def test_canonical_netlists(self):
        for rid, topo, params in pilot.PILOT:
            self.assertTrue(pilot.roundtrip_ok(formulas.reference(topo, params, rid)["netlist"]), rid)

    def test_values_from_series(self):
        # номиналы пилота - из рядов E24 (R) и E12 (C, L)
        for rid, _, params in pilot.PILOT:
            for k, v in params.items():
                if k == "dc":
                    continue
                series = formulas.E24 if k.startswith("R") else formulas.E12
                m = v / 10 ** math.floor(math.log10(v) + 1e-9)
                self.assertTrue(any(abs(m - s) < 1e-9 for s in series), (rid, k, v))


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class PilotAdmission(unittest.TestCase):
    def test_all_admitted(self):
        for r in pilot.build_pilot(ROOT / "out" / "test_pilot"):
            with self.subTest(ref=r["id"]):
                self.assertTrue(r["admission"]["admitted"], r["admission"])

    def test_committed_pilot_matches(self):
        # data/pilot/refs.jsonl - результат того же построения
        saved = [json.loads(l) for l in (ROOT / "data" / "pilot" / "refs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["id"] for r in saved], [rid for rid, _, _ in pilot.PILOT])
        self.assertTrue(all(r["admission"]["admitted"] for r in saved))


if __name__ == "__main__":
    unittest.main()
