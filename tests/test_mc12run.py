"""Тесты пакетного запуска MC12 и разбора его числового вывода.

Разбор проверяется на выводе MC12 из tests/data/mc12_out (получен 25.10.2026 из rc_probe.ckt и
float.ckt, журнал - из пакета с ошибочными схемами). Запуск самого MC12 пропускается, если его нет.
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import math
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import mc12run, mcout  # noqa: E402

DATA = ROOT / "tests" / "data"
OUT = DATA / "mc12_out"
RC_FC = 1 / (2 * math.pi * 1e3 * 100e-9)  # R=1k, C=100n: 1591.55 Гц


class ParseOutput(unittest.TestCase):
    def test_ac_table(self):
        r = mcout.read(OUT / "rc_probe.ANO")
        self.assertEqual((r["analysis"], r["circuit"]), ("AC", "RC"))
        self.assertEqual(r["limits"]["Frequency Range"], "Log 1MEG,10")
        self.assertEqual([x["banner"] for x in r["runs"]], ["Temperature=27"])
        t = r["runs"][0]["tables"][0]
        self.assertEqual(t["columns"], ["F", "DB(V(OUT))", "PH(V(OUT))"])
        self.assertEqual(t["units"], ["(Hz)", "", "(Degrees)"])
        self.assertEqual(len(t["data"]["F"]), 506)
        self.assertIsNone(t["data"]["PH(V(OUT))"][0])  # NA в первой точке
        self.assertAlmostEqual(t["data"]["F"][-1], 1e6)

    def test_limits_single_space_key(self):
        # самый длинный ключ отделён от значения одним пробелом
        r = mcout.read(OUT / "rc_probe.TNO")
        self.assertEqual(r["limits"]["Output Start Time (tstart)"], "0")
        self.assertEqual(r["limits"]["Maximum Run Time"], "2M")
        self.assertEqual(mcout.column(r, "t")[:2], [0.0, 1e-5])

    def test_expression_column(self):
        r = mcout.read(OUT / "float_probe.ANO")
        self.assertEqual(r["runs"][0]["tables"][0]["columns"], ["F", "V(X)", "V(Y)", "DB(V(Y)/V(IN))"])
        self.assertAlmostEqual(mcout.column(r, "db(v(y)/v(in))")[0], -30.06)

    def test_cutoff_rc(self):
        r = mcout.read(OUT / "rc_probe.ANO")
        fc = mcout.cutoff_3db(mcout.column(r, "F"), mcout.column(r, "DB(V(OUT))"))
        self.assertAlmostEqual(fc / RC_FC, 1, delta=1e-3)

    def test_cutoff_highpass_synthetic(self):
        f = [10 ** (k / 100) for k in range(0, 601)]
        g = [20 * math.log10(x / 1e3 / math.sqrt(1 + (x / 1e3) ** 2)) for x in f]  # ФВЧ 1-го порядка, 1 кГц
        self.assertAlmostEqual(mcout.cutoff_3db(f, g) / 1e3, 1, delta=1e-3)


class ParseBatchLog(unittest.TestCase):
    def setUp(self):
        self.log = mc12run.parse_doc((OUT / "batch_probe.DOC").read_text(encoding="cp1251"))

    def test_success_rows(self):
        self.assertTrue(self.log[("RC.CKT", "A")]["ok"])
        self.assertTrue(self.log[("RC.CKT", "T")]["ok"])
        self.assertTrue(self.log[("SK.CKT", "A")]["ok"])

    def test_load_errors(self):
        bad = self.log[("BAD.CKT", None)]
        self.assertFalse(bad["ok"])
        self.assertIn("Error: Missing model statement 'NOSUCHMODEL'.", bad["messages"])
        self.assertIn("Part:\tD1", bad["messages"])
        self.assertNotIn(("BAD.CKT", "A"), self.log)
        self.assertFalse(self.log[("NOVAL.CKT", None)]["ok"])

    def test_warning_does_not_fail(self):
        self.assertTrue(self.log[("FLOAT.CKT", None)]["ok"])
        self.assertIn("Warning: Node X has no DC path to ground.", self.log[("FLOAT.CKT", None)]["messages"])
        self.assertTrue(self.log[("FLOAT.CKT", "A")]["ok"])

    def test_command_line_error(self):
        self.assertFalse(self.log[("RC.CKT", None)]["ok"])

    def test_analyses_in(self):
        self.assertEqual(mc12run.analyses_in((DATA / "rc_probe.ckt").read_text()), ["A", "T"])


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class RunMC12(unittest.TestCase):
    """Настоящий запуск MC12 (~1 с). Рабочая папка - out/ проекта: MC12 видит её, в отличие от
    перенаправленных папок приложения."""

    @classmethod
    def setUpClass(cls):
        cls.work = ROOT / "out" / "test_mc12run"
        shutil.rmtree(cls.work, ignore_errors=True)
        cls.work.mkdir(parents=True)
        for n in ("rc_probe.ckt", "bad_probe.ckt", "sk_ideal.ckt"):
            shutil.copy(DATA / n, cls.work / n)
        cls.res = mc12run.run([cls.work / n for n in ("rc_probe.ckt", "bad_probe.ckt", "sk_ideal.ckt")],
                              timeout=120)
        cls.by = {(Path(r["ckt"]).name, r["analysis"]): r for r in cls.res}

    def test_rc_runs_and_matches_theory(self):
        ac = self.by[("rc_probe.ckt", "A")]
        self.assertTrue(ac["ok"], ac)
        self.assertTrue(self.by[("rc_probe.ckt", "T")]["ok"])
        r = mcout.read(ac["output"])
        fc = mcout.cutoff_3db(mcout.column(r, "F"), mcout.column(r, "DB(V(OUT))"))
        self.assertAlmostEqual(fc / RC_FC, 1, delta=1e-3)

    def test_broken_netlist_reported(self):
        bad = self.by[("bad_probe.ckt", "A")]
        self.assertFalse(bad["ok"])
        self.assertIsNone(bad["output"])
        self.assertTrue(any("NOSUCHMODEL" in m for m in bad["messages"]))

    def test_sallen_key_ideal_opamp_matches_nodal_analysis(self):
        # ФВЧ Саллена-Ки с усилением 2 (R1=18.2k, R3=14k, C1=C2=1n), идеальный ОУ: узловой расчёт
        R1, R3, C, K = 18.2e3, 14e3, 1e-9, 2.0

        def gain_db(f):
            s = 2j * math.pi * f
            a = s * C / (s * C + 1 / R3)
            v4 = s * C / (s * C + 1 / R1 - K * a / R1 + s * C - s * C * a)
            return 20 * math.log10(abs(K * a * v4))

        fs = [10 ** (k / 2000) for k in range(6000, 10001)]
        theory = mcout.cutoff_3db(fs, [gain_db(f) for f in fs])
        r = mcout.read(self.by[("sk_ideal.ckt", "A")]["output"])
        fc = mcout.cutoff_3db(mcout.column(r, "F"), mcout.db(mcout.column(r, "V(OUT)")))
        self.assertAlmostEqual(fc / theory, 1, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
