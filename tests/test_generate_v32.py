"""Датасет v3.2: снятие подписей ролей у кратких запросов Саллен-Ки (docs/generator_v3_spec.md, раздел 9).

Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import generate_v32  # noqa: E402

DIR = ROOT / "data" / "dataset" / "v3_2"


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


class Strip(unittest.TestCase):
    def test_lp_and_hp(self):
        self.assertEqual(generate_v32.strip_tags("саллен-ки фнч R1 1к R2 30к C1(ос) 12н C2(земля) 745п", "sallen_key_lp"),
                         "саллен-ки фнч R1 1к R2 30к C1 12н C2 745п")
        self.assertEqual(generate_v32.strip_tags("Саллен-Ки ФВЧ: R1(ос) 10к, R2(GND) 261к, C1 100н", "sallen_key_hp"),
                         "Саллен-Ки ФВЧ: R1 10к, R2 261к, C1 100н")
        self.assertEqual(generate_v32.strip_tags("ФНЧ: С1(ос) 2,2н, С2(земля) 330п", "sallen_key_lp"), "ФНЧ: С1 2,2н, С2 330п")

    def test_other_numbering_refused(self):
        with self.assertRaises(ValueError):  # «(ос)» не у той метки: снять подпись - значит поменять схему
            generate_v32.strip_tags("ФНЧ R1 1к R2 2к C1(земля) 1н C2(ос) 2н", "sallen_key_lp")
        with self.assertRaises(ValueError):
            generate_v32.strip_tags("ФНЧ R1 1к R2 2к C1 1н C2 2н", "sallen_key_lp")


class Frozen(unittest.TestCase):
    """v3.2 заморожен 15.11.2026: файлы не меняются; от v3.1 отличается только текстом 33 запросов."""

    def test_hashes(self):
        m = json.loads((DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((m["version"], m["frozen"], m["report"]["changed"]), ("v3.2", "2026-11-15", 33))
        readme = (DIR / "README.md").read_text(encoding="utf-8")
        for name, info in m["files"].items():
            b = (DIR / name).read_bytes()
            self.assertEqual(hashlib.sha256(b).hexdigest(), info["sha256"], name)
            self.assertEqual(b.count(b"\n"), info["records"], name)
            self.assertIn(info["sha256"], readme, name)

    def test_only_prompts_changed(self):
        old = {r["id"]: r for r in read(ROOT / "data/dataset/v3_1/mode1/train.jsonl")}
        new = read(DIR / "mode1/train.jsonl")
        self.assertEqual([r["id"] for r in new], list(old))
        changed = [r for r in new if r.get("v32")]
        self.assertEqual(len(changed), 33)
        for r in new:
            o = old[r["id"]]
            if r.get("v32"):
                self.assertEqual(r["prompt_v31"], o["prompt"])
                self.assertEqual(generate_v32.strip_tags(o["prompt"], o["topology"]), r["prompt"])
                self.assertIn(r["topology"], generate_v32.SK)
            self.assertEqual({k: v for k, v in r.items() if k not in ("prompt", "prompt_v31", "v32")},
                             {k: v for k, v in o.items() if k != "prompt"}, r["id"])
        for s in ("dev", "test", "gen_family", "gen_range"):
            self.assertEqual((DIR / "mode1" / f"{s}.jsonl").read_bytes(),
                             (ROOT / "data/dataset/v3_1/mode1" / f"{s}.jsonl").read_bytes(), s)

    def test_watch_sets(self):
        for name, src in (("watch_v3", "data/dataset/v3/test.jsonl"), ("watch_v2", "data/dataset/v2/test.jsonl")):
            ref = {r["id"]: r for r in read(ROOT / src)}
            w = read(DIR / "mode1" / f"{name}.jsonl")
            self.assertTrue(all(r == ref[r["id"]] for r in w), name)
            self.assertEqual(sum(bool(generate_v32._SALLEN.search(r["prompt"])) for r in w),
                             sum(bool(generate_v32._SALLEN.search(r["prompt"])) for r in ref.values()
                                 if r.get("mode", 1) == 1))


if __name__ == "__main__":
    unittest.main()
