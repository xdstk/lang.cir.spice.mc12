#!/usr/bin/env python3
"""Данные и оценка режима дизайнера: спецификация на русском -> JSON mcnl.flt/1 (docs/designer_mode.md).

Запуск:  python src/mcnl/designer_data.py --out data/dataset/designer_v1 [--seed 20261101]

Метрики: полное совпадение и точность по полям. Базовая линия - разбор правилами parse_rules (RuleDesigner),
подключается как переводчик с output_format = "flt".
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import designer, prompts_v2  # noqa: E402

SPLITS = {"train": 200, "dev": 30, "test": 70}
FIELDS = ("type", "response", "order", "fc", "f0", "Q", "style", "ripple_db", "gain_db", "gain")
TYPE_WORDS = {"LP": ["ФНЧ", "фильтр нижних частот", "фнч", "НЧ-фильтр", "low-pass"],
              "HP": ["ФВЧ", "фильтр верхних частот", "фвч", "ВЧ-фильтр", "high-pass"],
              "BP": ["полосовой фильтр", "полосовой", "ПФ", "bandpass"],
              "BS": ["режекторный фильтр", "режекторный", "заграждающий фильтр", "notch"]}
RESP_WORDS = {"Butterworth": ["Баттерворта", "баттерворта", "Баттерворт", "butterworth"],
              "Chebyshev": ["Чебышёва", "чебышева", "Чебышев", "chebyshev"],
              "Inverse Chebyshev": ["инверсный Чебышёв", "инверсный чебышев", "обратный Чебышёв"]}
STYLE_WORDS = {"Sallen-Key": ["Саллена-Ки", "саллен-ки", "Sallen-Key"], "MFB": ["МФБ", "MFB", "с многопетлевой ОС"],
               "Tow-Thomas": ["Тоу-Томаса", "Tow-Thomas"], "KHN": ["KHN", "КХН"], "Fleischer-Tow": ["Флейшера-Тоу", "Fleischer-Tow"],
               "DABP": ["DABP"], "Acker-Mossberg": ["Акера-Моссберга", "Acker-Mossberg"]}
STYLES_FOR = {"LP": ["Sallen-Key", "MFB", "Tow-Thomas", "KHN"], "HP": ["Sallen-Key", "MFB", "KHN"],
              "BP": ["MFB", "DABP", "Tow-Thomas", "KHN"], "BS": ["Tow-Thomas", "Fleischer-Tow", "KHN"]}
FREQS = [50, 60, 100, 150, 200, 300, 440, 500, 800, 1000, 1200, 1500, 2000, 3000, 3400, 4000, 5000, 8000, 10000, 15000, 20000, 50000, 100000]


def sample_spec(rng):
    t = rng.choices(["LP", "HP", "BP", "BS"], [4, 3, 1.5, 1.5])[0]
    resp = rng.choices(list(RESP_WORDS), [6, 3, 1])[0]
    spec = {"schema": designer.SCHEMA, "type": t, "response": resp}
    if t in ("LP", "HP"):
        spec.update(order=rng.randint(2, 6), fc=rng.choice(FREQS))
    else:
        spec.update(order=rng.randint(1, 3), f0=rng.choice(FREQS), Q=rng.choice([1, 2, 3, 5, 8, 10]))
    spec["style"] = rng.choice(STYLES_FOR[t]) if rng.random() < 0.4 else None
    spec["ripple_db"] = rng.choice([0.5, 1, 2, 3]) if resp == "Chebyshev" and rng.random() < 0.7 else None
    spec["gain_db"] = rng.choice([6, 10, 20]) if rng.random() < 0.15 else None
    return spec


def _freq(f, rng):
    if f >= 1000 and f % 1000 == 0:
        return rng.choice([f"{f // 1000} кГц", f"{f // 1000}кГц", f"{f} Гц"])
    if f >= 1000:
        return f"{f / 1000:g} кГц".replace(".", rng.choice([".", ","]))
    return rng.choice([f"{f} Гц", f"{f}Гц"])


def describe(spec, rng):
    """Спецификация -> запрос: части в случайном порядке, как пишут в чат."""
    kind = rng.choice(TYPE_WORDS[spec["type"]])
    resp = rng.choice(RESP_WORDS[spec["response"]])
    f = _freq(spec.get("fc") or spec.get("f0"), rng)
    n = spec["order"]
    ordr = rng.choice([f"{n} порядка", f"порядок {n}", f"{n}-го порядка"])
    parts = [ordr]
    if spec["type"] in ("LP", "HP"):
        parts.append(rng.choice([f"срез {f}", f"на {f}", f"частота среза {f}", f"fc {f}"]))
    else:
        parts += [rng.choice([f"на {f}", f"центр {f}", f"f0 {f}", f"центральная частота {f}"]), rng.choice([f"Q {spec['Q']}", f"добротность {spec['Q']}"])]
    if spec["ripple_db"]:
        parts.append(rng.choice([f"пульсации {spec['ripple_db']} дБ", f"неравномерность {spec['ripple_db']} дБ"]))
    if spec["gain_db"]:
        parts.append(rng.choice([f"усиление {spec['gain_db']} дБ", f"Ku {spec['gain_db']} дБ"]))
    if spec["style"]:
        parts.append(rng.choice(["схема ", "на схеме ", "звенья "]) + rng.choice(STYLE_WORDS[spec["style"]]))
    rng.shuffle(parts)
    head = rng.choice(["{k} {r}", "{k}, отклик {r}", "{r} {k}", "{k} {r}"]).format(k=kind, r=resp)
    lead = rng.choice(["", "", "спроектируй ", "рассчитай ", "нужен ", "подбери "])
    return lead + head + rng.choice([", ", " "]) + ", ".join(parts)


# ---------------------------------------------------------------- базовая линия
def parse_rules(prompt):
    """Разбор ключевыми словами -> спецификация; неразобранные поля None."""
    p = prompt.lower().replace("ё", "е")
    spec = {"schema": designer.SCHEMA, "type": None, "response": None, "order": None, "style": None, "ripple_db": None, "gain_db": None, "gain": None}
    for t, words in TYPE_WORDS.items():
        if any(w.lower() in p for w in words):
            spec["type"] = t
            break
    spec["response"] = ("Inverse Chebyshev" if re.search("инверс|обратн", p) else "Chebyshev" if "чебыш" in p or "chebyshev" in p
                        else "Butterworth" if "баттерворт" in p or "butterworth" in p else None)
    m = re.search(r"(\d+)(?:-[а-я]+)?\s*порядо?к|порядок\s*(\d+)", p)
    if m:
        spec["order"] = int(m.group(1) or m.group(2))
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(кгц|гц|мгц)", p)
    if m:
        v = float(m.group(1).replace(",", ".")) * {"гц": 1, "кгц": 1e3, "мгц": 1e6}[m.group(2)]
        spec["fc" if spec["type"] in ("LP", "HP") else "f0"] = int(v) if v == int(v) else v
    m = re.search(r"(?:\bq|добротност\w*)\s*(\d+(?:[.,]\d+)?)", p)
    if m and spec["type"] in ("BP", "BS"):
        spec["Q"] = int(float(m.group(1).replace(",", ".")))
    m = re.search(r"(?:пульсации|неравномерность)\s*(\d+(?:[.,]\d+)?)", p)
    if m:
        spec["ripple_db"] = float(m.group(1).replace(",", "."))
        spec["ripple_db"] = int(spec["ripple_db"]) if spec["ripple_db"] == int(spec["ripple_db"]) else spec["ripple_db"]
    m = re.search(r"(?:усиление|ku)\s*(\d+)(\s*дб)?", p)
    if m:
        spec["gain_db" if m.group(2) else "gain"] = int(m.group(1))  # без единицы - в разах
    for s, words in STYLE_WORDS.items():
        if any(w.lower().replace("ё", "е") in p for w in words):
            spec["style"] = s
    return spec


class RuleDesigner:
    name = "rules"
    output_format = "flt"

    def translate(self, prompt):
        return json.dumps(parse_rules(prompt), ensure_ascii=False)


# ---------------------------------------------------------------- оценка
def score(refs, outputs):
    """refs: записи с полем spec; outputs: JSON-строки -> {n, exact, по полям}."""
    per, exact = {f: 0 for f in FIELDS}, 0
    for ref, out in zip(refs, outputs):
        try:
            got = json.loads(out or "{}")
        except json.JSONDecodeError:
            got = {}
        if not isinstance(got, dict):
            got = {}
        same = {f: ref["spec"].get(f) == got.get(f) for f in FIELDS}
        exact += all(same.values())
        for f, ok in same.items():
            per[f] += ok
    n = len(refs)
    return {"n": n, "exact": exact / n, "fields": {f: v / n for f, v in per.items()}}


def generate(outdir, seed=20261101):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    seen, manifest = set(), {"seed": seed, "schema": designer.SCHEMA, "files": {}}
    for split, n in SPLITS.items():
        recs = []
        while len(recs) < n:
            spec = sample_spec(rng)
            key = json.dumps(spec, sort_keys=True)
            if key in seen or designer.validate(spec):
                continue
            seen.add(key)
            prompt, used = prompts_v2.mutate(describe(spec, rng), rng)
            recs.append({"id": f"{split}-{len(recs):04d}", "split": split, "prompt": prompt, "mutations": used, "spec": spec,
                         "output": json.dumps(spec, ensure_ascii=False)})
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8")
        (outdir / f"{split}.jsonl").write_bytes(data)
        manifest["files"][f"{split}.jsonl"] = {"records": n, "sha256": hashlib.sha256(data).hexdigest()}
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    out = argv[argv.index("--out") + 1] if "--out" in argv else "out/designer_data"
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 20261101
    generate(out, seed)
    recs = [json.loads(l) for l in (Path(out) / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    tr = RuleDesigner()
    print(json.dumps(score(recs, [tr.translate(r["prompt"]) for r in recs]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
