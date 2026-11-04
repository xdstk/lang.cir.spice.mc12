#!/usr/bin/env python3
"""Справочная строка: свежая облачная модель без обучения (GLM, API Z.ai) - load("glm", model="glm-4.7-flash").

Модель видит ту же системную инструкцию, что обучаемые модели (training.SYSTEM), без примеров; декодирование
жадное (do_sample=false), размышление выключено - как у обученных моделей (enable_thinking=False). Из ответа
убирается только обёртка ```...``` (форма, не содержание). Ключ - переменная окружения ZAI_API_KEY (пользователь
кладёт её сам; в код и файлы ключ не попадает). Бесплатные модели Z.ai бывают перегружены (429, код 1305): запрос
повторяется с растущей паузой.

Ответы на выборку с продолжением прерванного запуска (оценка - evaluate.py --predictions):
  python src/mcnl/translators/glm_api.py --records data/dataset/v2/dev.jsonl --out out/glm/dev.jsonl
                                         [--model glm-4.7-flash] [--limit 20] [--thinking] [--shots 10]
--shots N - N примеров из train v2 перед запросом (shots(), выбор фиксирован); без него - только инструкция.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcnl import training  # noqa: E402
from mcnl.translators import Translator  # noqa: E402

URL = "https://api.z.ai/api/paas/v4/chat/completions"
RETRY_CODES = (429, 500, 502, 503, 504)
TRAIN = Path(__file__).resolve().parents[3] / "data" / "dataset" / "v2" / "train.jsonl"
# Примеры в запросе (few-shot) - «обучение» необученной модели: по одному из train v2 на каждое место,
# выбор фиксирован зерном. Место - (семейство формулировок, условие на запись).
SHOT_SLOTS = (("roles_brief", lambda r: r["level"] == "L1"), ("table", None), ("chain", None),
              ("narrative", lambda r: "OPAMP" in r["netlist"]), ("addon", lambda r: "OPAMP" in r["netlist"]),
              ("terse", None), (None, lambda r: r["level"] == "L3"),
              ("refuse", lambda r: r["output"].startswith("REFUSE out_of_class")),
              ("refuse", lambda r: r["output"].startswith("REFUSE needs_design")),
              ("refuse", lambda r: r["output"].startswith("REFUSE ambiguous")))
SHOT_SEED = 20261104


def shots(n, fmt="netlist"):
    """n первых мест SHOT_SLOTS -> [(запрос, ответ)] из train v2."""
    import random
    rows = training.read_jsonl(TRAIN)
    rng = random.Random(SHOT_SEED)
    out = []
    for family, cond in SHOT_SLOTS[:n]:
        cands = [r for r in rows if (family is None or r["family"] == family) and (r.get("netlist") or r.get("refuse"))
                 and (cond is None or cond(r))]
        r = rng.choice(cands)
        out.append((r["prompt"], training.answer_for(r, fmt), r["id"]))
    return out


def strip_fence(text):
    """Ответ без markdown-обёртки ```spice ... ``` или ```json ... ```."""
    text = (text or "").strip()
    m = re.fullmatch(r"```[\w+-]*[ \t]*\n?(.*?)\n?```", text, re.S)
    return m.group(1).strip() if m else text


def _key():
    key = os.environ.get("ZAI_API_KEY")
    if not key and os.name == "nt":  # переменная, заданная после запуска процесса, видна только в реестре
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "ZAI_API_KEY")[0]
        except OSError:
            key = None
    if not key:
        raise RuntimeError("нет переменной окружения ZAI_API_KEY")
    return key


class GLMApi(Translator):
    def __init__(self, model="glm-4.7-flash", output_format="netlist", thinking=False, n_shots=0, max_tokens=1024,
                 retries=8, pause=10.0, timeout=180):
        self.name = model + (f"-fs{n_shots}" if n_shots else "") + ("-think" if thinking else "")
        self.model, self.output_format, self.thinking = model, output_format, thinking
        self.examples = shots(n_shots, output_format) if n_shots else []
        self.max_tokens, self.retries, self.pause, self.timeout = max_tokens, retries, pause, timeout
        self.key = _key()
        self.last = {}  # время, токены и попытки последнего запроса

    def _post(self, body):
        req = urllib.request.Request(URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                     headers={"Authorization": "Bearer " + self.key,
                                              "Content-Type": "application/json; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def translate(self, prompt):
        msgs = training.messages(prompt, fmt=self.output_format)
        for q, a, _ in reversed(self.examples):
            msgs[1:1] = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        body = {"model": self.model, "messages": msgs,
                "do_sample": False, "max_tokens": self.max_tokens,
                "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        t0 = time.time()
        for attempt in range(self.retries + 1):
            try:
                res = self._post(body)
                break
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_CODES or attempt == self.retries:
                    raise RuntimeError(f"Z.ai {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"Z.ai недоступен: {e}") from e
            time.sleep(self.pause * (attempt + 1))
        choice = res["choices"][0]
        self.last = {"sec": round(time.time() - t0, 2), "attempts": attempt + 1, "finish": choice.get("finish_reason"),
                     "tokens_in": res.get("usage", {}).get("prompt_tokens"),
                     "tokens_out": res.get("usage", {}).get("completion_tokens")}
        return strip_fence(choice["message"].get("content"))


def main(argv):
    def opt(key, default=None):
        return argv[argv.index(key) + 1] if key in argv else default
    tr = GLMApi(opt("--model", "glm-4.7-flash"), opt("--format", "netlist"), thinking="--thinking" in argv,
                n_shots=int(opt("--shots", 0)))
    if tr.examples:
        print("примеры:", [i for *_, i in tr.examples], flush=True)
    recs = training.read_jsonl(opt("--records"))
    if opt("--limit"):
        recs = recs[:int(opt("--limit"))]
    out = Path(opt("--out"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in training.read_jsonl(out)} if out.exists() else set()
    left = [r for r in recs if r["id"] not in done]  # продолжение прерванного запуска
    print(f"{tr.name}: осталось {len(left)}, уже есть {len(done)}", flush=True)
    log = out.with_suffix(".log.jsonl")  # время и токены на запрос - для таблицы ресурсов
    with open(out, "a", encoding="utf-8", newline="\n") as fh, open(log, "a", encoding="utf-8", newline="\n") as lh:
        for rnd in range(int(opt("--passes", 5))):  # не прошедшие из-за перегрузки - следующим проходом
            failed = []
            for r in left:
                try:
                    o = tr.translate(r["prompt"])
                except RuntimeError as e:
                    if "недоступен" not in str(e) and not any(f"Z.ai {c}" in str(e) for c in RETRY_CODES):
                        raise  # ключ, формат запроса - повтор не поможет
                    failed.append(r)
                    print(f"{r['id']}: {e}; повторю проходом позже", flush=True)
                    continue
                fh.write(json.dumps({"id": r["id"], "model": tr.name, "format": tr.output_format, "output": o},
                                    ensure_ascii=False) + "\n")
                lh.write(json.dumps(dict(id=r["id"], **tr.last)) + "\n")
                fh.flush()
                lh.flush()
                done.add(r["id"])
                if len(done) % 10 == 0:
                    print(f"готово {len(done)}/{len(recs)}, последний {tr.last}", flush=True)
            left = failed
            if not left:
                break
            time.sleep(60 * (rnd + 1))
    print(f"готово {len(done)}/{len(recs)}" + (f", без ответа {len(left)}" if left else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv)
