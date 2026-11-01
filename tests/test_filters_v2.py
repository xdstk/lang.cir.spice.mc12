"""Топологии датасета v2: формулы против узлового расчёта, структура против дизайнера MC12, допуск в MC12.

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, describe_ru, filters_v2, formulas, iso, mc12run, measure, mna, pilot  # noqa: E402

DESIGNER = ROOT / "data" / "designer"
ROLES = json.loads((ROOT / "data" / "roles_ru.json").read_text(encoding="utf-8"))


def parse(net):
    return canon.parse_netlist(net)[0]


class Analytic(unittest.TestCase):
    def test_every_topology_in_pilot(self):
        self.assertEqual({t for _, t, _ in filters_v2.PILOT}, set(filters_v2.TOPOLOGIES_V2))

    def test_v1_untouched(self):
        self.assertFalse(set(filters_v2.TOPOLOGIES_V2) & set(formulas.TOPOLOGIES))
        self.assertEqual(len(formulas.TOPOLOGIES), 17)
        self.assertFalse(set(filters_v2.TOPOLOGIES_V2) & set(describe_ru.ROLES))

    def test_formulas_against_mna(self):
        for rid, topo, params in filters_v2.PILOT:
            with self.subTest(topo):
                net, ana = filters_v2.TOPOLOGIES_V2[topo][0](**params)
                el = parse(net)
                f_char = ana.get("f_u") or filters_v2.characteristic_frequency(el)
                m = measure.measure(*mna.sweep(el, f_char / 1000, f_char * 1000, per_decade=200))
                for key, val in ana.items():
                    if key == "type":
                        self.assertEqual(m["type"], val)
                    elif key == "k_pass_db":
                        self.assertAlmostEqual(m["k_pass_db"], val, delta=0.05)
                    else:  # у Q 2-го порядка и f0 допуск сетки и конечного A
                        self.assertAlmostEqual(m[key] / val, 1, delta=0.02, msg=key)

    def test_notch_has_null(self):
        el = parse(filters_v2.twin_t(10e3, 10e-9)[0])
        self.assertLess(abs(mna.response(el, 1 / (2 * math.pi * 1e-4))), 1e-4)


class AgainstDesigner(unittest.TestCase):
    """Те же топологии и номиналы, что создал дизайнер MC12 (data/designer)."""

    def check(self, name, net):
        ref = parse((DESIGNER / f"{name}.net").read_text(encoding="ascii"))
        self.assertTrue(iso.match(ref, parse(net), "values")["match"], name)

    def test_mfb_lp(self):
        self.check("L_B_M_1111_2", filters_v2.mfb_lp(25.261e3, 25.261e3, 1.2377e6, 8.1016e-9, 81.016e-12)[0])

    def test_mfb_hp(self):
        self.check("H_B_M_1111_2", filters_v2.mfb_hp(8.3354e3, 37.509e3, 8.1016e-9, 8.1016e-9, 8.1016e-9)[0])

    def test_tow_thomas(self):
        r, ra = 17.682e3, 12.503e3
        self.check("L_B_T_1111_2", filters_v2.tow_thomas(r, r, ra, r, ra, 8.1016e-9, 8.1016e-9)[0])

    def test_khn(self):
        self.check("L_B_K_1111_2", filters_v2.khn(17.682e3, 17.682e3, 25.006e3, 17.682e3, 12.503e3, 17.682e3,
                                                  8.1016e-9, 8.1016e-9)[0])


class Roles(unittest.TestCase):
    def test_all_params_have_roles(self):
        import inspect
        roles = ROLES["topologies_v2"]
        self.assertEqual(set(roles), set(filters_v2.TOPOLOGIES_V2))
        for topo, (fn, *_rest) in filters_v2.TOPOLOGIES_V2.items():
            params = {p for p in inspect.signature(fn).parameters if p != "dc"}
            self.assertEqual(set(roles[topo]["roles"]), params, topo)
            self.assertTrue(roles[topo]["names"] and roles[topo]["short_names"], topo)
            self.assertEqual(set(roles[topo]["short_roles"]), params, topo)


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class Admission(unittest.TestCase):
    def test_pilot_in_mc12(self):
        refs = [filters_v2.reference(t, p, rid) for rid, t, p in filters_v2.PILOT]
        for r in pilot.admit(refs, ROOT / "out" / "test_filters_v2"):
            with self.subTest(r["id"]):
                self.assertTrue(r["admission"]["admitted"], r["admission"])


if __name__ == "__main__":
    unittest.main()
