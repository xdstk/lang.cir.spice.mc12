"""Исправления дефектов v2 для генератора v3 (docs/generator_v3_spec.md, раздел 2): Д1-Д4.
Включаются только с gen=3; по умолчанию (gen 2) текст прежний - v1 и v2 воспроизводятся."""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import canon, describe_ru, filters_v2, paraphrase, prompts_v2  # noqa: E402

SK_HP = {"class": "sallen_key_hp", "params": {"C1": 330e-12, "C2": 180e-12, "R1": 8106.5, "R2": 892.7e3}}  # test-00241
MFB = {"class": "mfb_bp", "params": {"R1": 620.0, "R2": 128.28, "R3": 18e3, "Ca": 47e-9, "Cb": 6.8e-9, "dc": 5}}  # test-00216


class D1Picofarads(unittest.TestCase):
    def options(self, value, gen, n=60):
        return {describe_ru.Style(random.Random(i), gen).value("C", value) for i in range(n)}

    def test_gen3_keeps_reference_precision(self):
        got = self.options("3.9354n", 3)
        self.assertIn("3935,4 пФ", {s.replace(".", ",") for s in got})
        self.assertNotIn("3935 пФ", got)

    def test_gen2_unchanged(self):
        self.assertIn("3935 пФ", self.options("3.9354n", 2))

    def test_gen3_values_parse_back_exactly(self):
        """Обратный разбор запроса (как verified_a генератора) возвращает номинал эталона."""
        for value in ("3.9354n", "4.6499n", "0.5p", "820p", "1.2345n"):
            net = f"V1 IN 0 DC 0\nR1 IN OUT 10k\nC1 OUT 0 {value}\n"
            for family in ("list", "narrative", "table"):
                for i in range(15):
                    prompt = describe_ru.describe_a(net, family, random.Random(i), gen=3)
                    c = [e for e in describe_ru.parse_a(prompt, family) if e["type"] == "C"][0]
                    self.assertEqual(canon.parse_value(c["value"]), canon.parse_value(value), prompt)

    def test_same_rng_sequence_as_gen2(self):
        """Тот же выбор вариантов: меняется только запись пикофарад."""
        net = "V1 IN 0 DC 0\nR1 IN OUT 10k\nC1 OUT 0 3.9354n\n"
        for i in range(30):
            a = describe_ru.describe_a(net, "list", random.Random(i), gen=2)
            b = describe_ru.describe_a(net, "list", random.Random(i), gen=3)
            self.assertEqual(a.replace("3935 пФ", "X"), b.replace("3935,4 пФ", "X").replace("3935.4 пФ", "X"))


class D2MfbRoles(unittest.TestCase):
    def test_gen3_roles_without_our_convention(self):
        for i in range(20):
            text = describe_ru.describe_a1_brief(MFB, random.Random(i), gen=3)
            self.assertNotRegex(text, r"C[аб]\b")
            self.assertIn("C(ос)", text)
            self.assertIn("C(инв. вход)", text)

    def test_gen2_unchanged(self):
        self.assertIn("Cа 47н", describe_ru.describe_a1_brief(MFB, random.Random(2)))

    def test_v3_labels_cover_all_roles(self):
        roles = describe_ru.ROLES_ALL["mfb_bp"]
        self.assertEqual(set(describe_ru.short_roles("mfb_bp", 3)), set(roles["short_roles"]))


class D3Terse(unittest.TestCase):
    def test_ambiguous_roles(self):
        self.assertEqual(prompts_v2.ambiguous_roles(SK_HP), {"C1", "C2", "R1", "R2"})
        self.assertEqual(prompts_v2.ambiguous_roles({"class": "rc_lp", "params": {"R": 1e3, "C": 1e-7}}), set())
        equal = {"class": "sallen_key_lp", "params": {"R1": 1e4, "R2": 1e4, "C1": 2.2e-8, "C2": 1e-8}}
        self.assertEqual(prompts_v2.ambiguous_roles(equal), {"C1", "C2"})  # равные R можно не подписывать

    def test_gen3_labels_ambiguous_values(self):
        text = prompts_v2.describe_terse(SK_HP, random.Random(5), gen=3)
        for label in ("C1", "C2", "R1(ос)", "R2(земля)"):
            self.assertIn(label, text)

    def test_unambiguous_text_unchanged(self):
        rng = random.Random(7)
        for topo in ("rc_lp", "rc_hp", "rl_lp", "rlc_series_lp", "twin_t", "integrator"):
            for i in range(10):
                ref = filters_v2.sample(topo, rng)[0] if topo in filters_v2.TOPOLOGIES_V2 else \
                    {"class": topo, "params": {"R": 1e3, "C": 1e-7} if topo.startswith("rc") else {"R": 1e3, "L": 1e-3}}
                self.assertEqual(prompts_v2.describe_terse(ref, random.Random(i), gen=3),
                                 prompts_v2.describe_terse(ref, random.Random(i)), topo)


class D4ParaphraseName(unittest.TestCase):
    def test_train_00752(self):
        tmpl = "1,3к последовательно; 220мкГн на минус; выход после последнего элемента"
        new = "Набросай, пожалуйста: RL ФНЧ: 1,3к последовательно и 220мкГн на минус, выход снимаем с катушки"
        self.assertIsNotNone(paraphrase.name_conflict(tmpl, new, "HP"))
        self.assertIsNone(paraphrase.name_conflict(tmpl, new.replace("ФНЧ", "ФВЧ"), "HP"))
        self.assertIsNone(paraphrase.name_conflict(tmpl, tmpl, "HP"))

    def test_name_from_template_allowed(self):
        """Добавка честно меняет тип: название из исходного запроса не ошибка парафразы."""
        tmpl = "RC ФНЧ: R 1к, C 100н; между входом и выходом ещё 10н"
        self.assertIsNone(paraphrase.name_conflict(tmpl, "Собери RC ФНЧ, R 1к, C 100н, плюс 10н от входа на выход", "BP"))

    def test_v2_train_has_one_conflict(self):
        bad = []
        for line in (ROOT / "data/dataset/v2/train.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("family") == "paraphrase" and not r.get("refuse"):
                if paraphrase.name_conflict(r["prompt_template"], r["prompt"], (r.get("expected") or {}).get("type")):
                    bad.append(r["id"])
        self.assertEqual(bad, ["train-00752"])


if __name__ == "__main__":
    unittest.main()
