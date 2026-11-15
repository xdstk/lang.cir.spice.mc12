#!/usr/bin/env python3
"""Датасет v3.2: v3.1 (режим 1), у половины кратких запросов Саллен-Ки сняты подписи ролей (docs/generator_v3_spec.md, раздел 9).

Запуск:  python src/mcnl/generate_v32.py --out out/v3_2_full [--seed 20261115]

Причина - оценка q35_4b_v31_m1 (docs/stage2_plan.md, «Фаза 5: итоги»): на llm-выборках Саллен-Ки - 1 из 20, ОУ
включён как в MFB. В train v3.1 краткие запросы Саллен-Ки (terse, roles_brief) все с подписями ролей - «C1(ос)»,
«R2(земля)» (правка Д3), у MFB подписей нет ни разу; llm-запросы «Саллен-Ки ФНЧ R1 4.7к R2 8.2к C1 15н C2 6.8н» - без
подписей, то есть в формате MFB.

Правка (решения автора 15.11.2026): в SHARE таких запросов подписи в скобках снимаются, метки R1, R2, C1, C2 остаются.
Роль метки задаёт обычная нумерация, та же, что у подписей Д3 и у эталонов llm_test: у ФНЧ C1 - в обратной связи,
C2 - на землю; у ФВЧ R1 - в обратной связи, R2 - на землю. Голые номиналы, как в v2, не возвращаются - их
неоднозначность и исправляла Д3. Схемы, эталоны и остальные записи - как в v3.1 (байт в байт, кроме текста запроса и
двух полей у правленых записей). Режим 2 не меняется: адаптер режима 2 обучается на v3.1.

Выборки для наблюдения (решение автора: после обучения - не все выборки, а доля): watch_v3 - записи test v3 режима 1
со словом «Саллен» и случайные WATCH других; watch_v2 - то же из test v2.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V31 = ROOT / "data" / "dataset" / "v3_1" / "mode1"
SK = ("sallen_key_lp", "sallen_key_hp")
FAMILIES = ("terse", "roles_brief")
SHARE = 0.5
WATCH = 70
FEEDBACK = {"sallen_key_lp": "C1", "sallen_key_hp": "R1"}   # метка, которая у подписи «(ос)»
_TAG = re.compile(r"\b([RCРС])([12])\s*\(([^()]{1,15})\)")
_SALLEN = re.compile(r"салл|sallen", re.I)


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def strip_tags(prompt, topology):
    """Снимает подписи ролей у R1, R2, C1, C2. Проверяет нумерацию: «(ос)» - у метки FEEDBACK[topology]."""
    tags = _TAG.findall(prompt)
    if not tags:
        raise ValueError(f"нет подписей: {prompt}")
    for letter, num, tag in tags:
        label = ("C" if letter in "CС" else "R") + num
        if (tag.strip().lower() == "ос") != (label == FEEDBACK[topology]):
            raise ValueError(f"нумерация не та: {label}({tag}) в {topology}: {prompt}")
    return _TAG.sub(lambda m: m.group(1) + m.group(2), prompt)


def candidates(train):
    return [r for r in train if r["topology"] in SK and r["family"] in FAMILIES]


def choose(cands, rng):
    """Половина каждой топологии (ФНЧ и ФВЧ отдельно), с округлением вверх."""
    out = []
    for top in SK:
        group = sorted((r for r in cands if r["topology"] == top), key=lambda r: r["id"])
        out += rng.sample(group, (len(group) + 1) // 2)
    return {r["id"] for r in out}


def watch(records, rng, n=WATCH):
    sk = [r for r in records if _SALLEN.search(r["prompt"])]
    rest = [r for r in records if not _SALLEN.search(r["prompt"])]
    picked = {r["id"] for r in sk + rng.sample(rest, min(n, len(rest)))}
    return [r for r in records if r["id"] in picked]


def _write(path, records):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"records": len(records), "sha256": hashlib.sha256(data).hexdigest()}


def generate(outdir, seed=20261115):
    outdir = Path(outdir)
    rng = random.Random(seed)
    train = read(V31 / "train.jsonl")
    chosen = choose(candidates(train), rng)
    changed = []
    for r in train:
        if r["id"] in chosen:
            new = strip_tags(r["prompt"], r["topology"])
            r.update(prompt=new, prompt_v31=r["prompt"], v32="unlabeled") if new != r["prompt"] else None
            changed.append(r["id"])
    manifest = {"version": "v3.2", "seed": seed, "base": "v3.1 mode1", "files": {},
                "report": {"changed": len(changed), "changed_ids": changed,
                           "sk_short": len(candidates(train)), "share": SHARE}}
    d = outdir / "mode1"
    manifest["files"]["mode1/train.jsonl"] = _write(d / "train.jsonl", train)
    for s in ("dev", "test", "gen_family", "gen_range"):  # копии v3.1 (= v3 своего режима) байт в байт
        data = (V31 / f"{s}.jsonl").read_bytes()
        (d / f"{s}.jsonl").write_bytes(data)
        manifest["files"][f"mode1/{s}.jsonl"] = {"records": data.count(b"\n"), "sha256": hashlib.sha256(data).hexdigest()}
    manifest["files"]["mode1/watch_v3.jsonl"] = _write(d / "watch_v3.jsonl", watch(read(V31 / "test.jsonl"), rng))
    manifest["files"]["mode1/watch_v2.jsonl"] = _write(d / "watch_v2.jsonl",
                                                       watch(read(ROOT / "data/dataset/v2/test.jsonl"), rng))
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                          newline="\n")
    return manifest


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "out" / "v3_2_full"
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 20261115
    m = generate(out, seed)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({k: v["records"] for k, v in m["files"].items()}, ensure_ascii=False), "- правлено", m["report"]["changed"])


if __name__ == "__main__":
    main(sys.argv)
