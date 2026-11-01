"""Режим дизайнера: спецификация, Active.flt, макрос, данные (docs/designer_mode.md); mc12_real.

Управление окном MC12 проверяется только при MCNL_GUI=1: тест открывает окно.
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import os
import random
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, designer, designer_data, mc12_real  # noqa: E402

LP = {"type": "LP", "response": "Butterworth", "order": 4, "fc": 1000}
FLT = ("[Data]\r\nType=High-Pass\r\nResponse=Chebyshev\r\nSave As=Circuit\r\nSame Circuit=No\r\nViewCir=Yes\r\n"
       "Low-Pass mode=1\r\nLow-Pass Butterworth Circuits=0,0,\r\nBandpass mode=1\r\n"
       "[Low-Pass High-Pass Mode 2]\r\nGain=0\r\nCutoff Frequency=1600\r\nRipple=3\r\nOrder=2\r\nSteepness Factor=1.5\r\n\r\n"
       "[Band Pass Notch Mode 2]\r\nGain=0\r\nCenter Frequency=1000\r\nRipple=1\r\nOrder=3\r\nQ=10\r\nSteepness Factor=1.5\r\n\r\n")


class Spec(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(designer.validate(LP), [])
        self.assertEqual(designer.validate({"type": "BP", "response": "Chebyshev", "order": 2, "f0": 1000, "Q": 5}), [])

    def test_errors(self):
        self.assertTrue(designer.validate(dict(LP, type="XX")))
        self.assertTrue(designer.validate(dict(LP, order=0)))
        self.assertTrue(designer.validate(dict(LP, fc=0)))
        self.assertTrue(designer.validate({"type": "BP", "response": "Butterworth", "order": 2, "f0": 1000}))
        self.assertTrue(designer.validate(dict(LP, style="Рауха")))

    def test_elliptic_rejected(self):
        self.assertTrue(designer.validate(dict(LP, response="Elliptic")))

    def test_from_json(self):
        spec, errs = designer.from_json('```json\n{"type": "LP", "response": "Butterworth", "order": 2, "fc": 50}\n```')
        self.assertEqual((spec["fc"], errs), (50, []))
        self.assertEqual(designer.from_json("не JSON")[0], None)


class Flt(unittest.TestCase):
    def test_lowpass(self):
        t = designer.flt_text(dict(LP, style="MFB", ripple_db=2), FLT)
        for line in ("Type=Low-Pass", "Response=Butterworth", "Save As=Macro", "ViewCir=No", "Low-Pass mode=2",
                     "Cutoff Frequency=1000", "Order=4", "Ripple=2", "Same Circuit=Yes"):
            self.assertIn(line + "\r\n", t)
        self.assertIn("Low-Pass Butterworth Circuits=1,1,", t)
        self.assertNotIn("\n\n", t.replace("\r\n", "\r"))

    def test_notch_keeps_lowpass_section(self):
        t = designer.flt_text({"type": "BS", "response": "Butterworth", "order": 2, "f0": 800, "Q": 2}, FLT.replace("Bandpass mode", "Notch mode"))
        self.assertIn("Center Frequency=800\r\n", t)
        self.assertIn("Q=2\r\n", t)
        self.assertIn("Cutoff Frequency=1600\r\n", t)

    def test_real_file_fields(self):
        p = designer.ACTIVE_FLT
        if not p.exists():
            self.skipTest("MC12 не установлен")
        out = designer.flt_text(dict(LP, style="Tow-Thomas"), p.read_bytes().decode("latin-1"))
        self.assertIn("Cutoff Frequency=1000\r\n", out)
        self.assertEqual(out.count("\r\n"), out.count("\n"))


class Macro(unittest.TestCase):
    def test_netlist_from_macro(self):
        p = designer.LIBRARY / "L_B_S_1000_2000.MAC"
        if not p.exists():
            self.skipTest("MC12 не установлен")
        raw = p.read_text(encoding="cp1251", errors="replace")
        self.assertEqual(designer.macro_title(raw), "Low-Pass Butterworth Sallen-Key")
        el = canon.parse_netlist(designer.macro_netlist(raw))[0]
        self.assertEqual(sum(e["type"] == "OPAMP" for e in el), 2)
        self.assertEqual(designer.check({"type": "LP", "fc": 1000}, canon.to_netlist(canon.canonicalize(el)))[2], [])


class Data(unittest.TestCase):
    def test_rules_read_their_templates(self):
        rng = random.Random(1)
        refs = []
        for _ in range(100):
            spec = designer_data.sample_spec(rng)
            self.assertEqual(designer.validate(spec), [])
            refs.append({"spec": spec, "prompt": designer_data.describe(spec, rng)})
        tr = designer_data.RuleDesigner()
        self.assertGreater(designer_data.score(refs, [tr.translate(r["prompt"]) for r in refs])["exact"], 0.9)

    def test_score_fields(self):
        ref = {"spec": {"type": "LP", "response": "Butterworth", "order": 4, "fc": 1000}}
        s = designer_data.score([ref, ref], [json.dumps(ref["spec"]), json.dumps(dict(ref["spec"], order=3))])
        self.assertEqual((s["exact"], s["fields"]["order"], s["fields"]["type"]), (0.5, 0.5, 1.0))
        self.assertEqual(designer_data.score([ref], ["мусор"])["exact"], 0.0)

    def test_frozen_files(self):
        d = ROOT / "data" / "dataset" / "designer_v1"
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({k: v["records"] for k, v in m["files"].items()}, {"train.jsonl": 200, "dev.jsonl": 30, "test.jsonl": 70, "llm_specs.jsonl": 30})


class Mc12Real(unittest.TestCase):
    def test_schemes_in_class(self):
        schemes = mc12_real.schemes()
        self.assertGreaterEqual(len(schemes), 13)
        for name, _src, net in schemes:
            el, err = canon.parse_netlist(net)
            self.assertEqual(err, [], name)
            self.assertIn("OUT", {n for e in el for n in e["nodes"]}, name)

    def test_task_lists_every_scheme(self):
        text = (ROOT / "data" / "dataset" / "mc12_real_v1" / "task.txt").read_text(encoding="utf-8")
        self.assertIn(f"\n{len(mc12_real.schemes())}. ", text)


@unittest.skipUnless(os.environ.get("MCNL_GUI"), "окно MC12 открывается только при MCNL_GUI=1")
class Gui(unittest.TestCase):
    def test_end_to_end(self):
        out = ROOT / "out" / "test_designer"
        shutil.rmtree(out, ignore_errors=True)
        r = designer.run(dict(LP, fc=1357, style="Sallen-Key"), out)
        self.assertTrue(r["ok"], r["notes"])
        self.assertTrue(Path(r["macro"]).exists())


if __name__ == "__main__":
    unittest.main()
