"""Тесты справочной строки на API Z.ai (translators/glm_api.py) без сети и без настоящего ключа.
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import training  # noqa: E402
from mcnl.translators import glm_api, load  # noqa: E402


def fake_reply(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@mock.patch.dict(os.environ, {"ZAI_API_KEY": "test.key"})
class GLM(unittest.TestCase):
    def test_strip_fence(self):
        self.assertEqual(glm_api.strip_fence("```spice\nR1 IN OUT 1k\n```"), "R1 IN OUT 1k")
        self.assertEqual(glm_api.strip_fence("```\nR1 IN OUT 1k\n```"), "R1 IN OUT 1k")
        self.assertEqual(glm_api.strip_fence(" R1 IN OUT 1k "), "R1 IN OUT 1k")
        self.assertEqual(glm_api.strip_fence("текст ```R1```"), "текст ```R1```")  # не только обёртка - как есть

    def test_request_same_instruction_greedy_no_thinking(self):
        tr = load("glm", model="glm-4.7-flash")
        with mock.patch.object(tr, "_post", return_value=fake_reply("```spice\nV1 IN 0 DC 0\n```")) as post:
            self.assertEqual(tr.translate("RC ФНЧ"), "V1 IN 0 DC 0")
        body = post.call_args[0][0]
        self.assertEqual(body["messages"], training.messages("RC ФНЧ"))
        self.assertEqual((body["do_sample"], body["thinking"]), (False, {"type": "disabled"}))
        self.assertEqual(tr.name, "glm-4.7-flash")

    def test_shots_fixed_and_placed_before_query(self):
        a, b = glm_api.shots(10), glm_api.shots(10)
        self.assertEqual([x[2] for x in a], [x[2] for x in b])
        self.assertEqual(sum(x[1].startswith("REFUSE") for x in a), 3)
        tr = glm_api.GLMApi(n_shots=3)
        with mock.patch.object(tr, "_post", return_value=fake_reply("V1 IN 0 DC 0")) as post:
            tr.translate("запрос")
        msgs = post.call_args[0][0]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system"] + ["user", "assistant"] * 3 + ["user"])
        self.assertEqual((msgs[1]["content"], msgs[-1]["content"]), (a[0][0], "запрос"))
        self.assertEqual(tr.name, "glm-4.7-flash-fs3")

    def test_retries_on_overload(self):
        tr = glm_api.GLMApi(pause=0)
        busy = urllib.error.HTTPError(glm_api.URL, 429, "busy", {}, io.BytesIO(b'{"error":{"code":"1305"}}'))
        with mock.patch.object(tr, "_post", side_effect=[busy, busy, fake_reply("V1 IN 0 DC 0")]):
            self.assertEqual(tr.translate("x"), "V1 IN 0 DC 0")
        self.assertEqual(tr.last["attempts"], 3)
        bad = urllib.error.HTTPError(glm_api.URL, 401, "no", {}, io.BytesIO(b"token expired"))
        with mock.patch.object(tr, "_post", side_effect=bad), self.assertRaises(RuntimeError):
            tr.translate("x")


if __name__ == "__main__":
    unittest.main()
