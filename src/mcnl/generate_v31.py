#!/usr/bin/env python3
"""Сборка датасета v3.1 (docs/generator_v3_spec.md, раздел «v3.1»): два набора train для двух адаптеров.

Запуск:  python src/mcnl/generate_v31.py --out out/v3_1_full [--seed 20261114] [--scale 0.05]

Основа - замороженный v3 (data/dataset/v3), его файлы не меняются:
- mode1/train.jsonl - 2000 записей режима 1: записи train v3 режима 1 (пул 2457, со всеми парафразами) минус самые
  длинные, плюс новые записи gen-3.1 - одиночные Саллен-Ки в краткой записи, LC-схемы без резистора (lc_v31),
  краткие запросы к названным схемам и краткие описания соединений. Причина - разбор q35_4b_v3 (docs/stage2_plan.md,
  «Фаза 4: итоги»): на коротких человеческих запросах модель v3 путала Саллен-Ки с MFB и писала C или L буквой R;
- mode2/train.jsonl - 2000 записей режима 2: записи train v3 режима 2 без дефекта Д5 (напряжение источника в
  эталоне, которого нет в запросе) плюс новые (mode2_data, доля названных схем выше, dc0=True);
- dev, test, gen_family, gen_range - записи v3 своего режима как есть: числа сравнимы с моделями v2 и v3.
Схемы новых записей не совпадают ни с одной схемой v3 (все выборки), llm_test, llm_test2, mc12_real и C01.
Допуск - как в v3 (pilot.admit, .CIR; у режима 2 - generate_v3.admit_mode2). Сборку не вести одновременно с другими
запусками MC12.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import (canon, compose_v3, describe_ru, filters_v3, generate_v3, lc_v31, mode2_data, pilot,  # noqa: E402
                  prompts_v2)

VERSION = "gen-3.1"
ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "data" / "dataset" / "v3"
N_TRAIN = 2000                      # в каждом наборе, как у v2 (сравнение при равном объёме)
SUBSETS = (1000, 2000)
NEW1 = {"sk_short": 40, "lc": 80, "named_short": 120, "brief": 60}   # новые записи режима 1
SK = ("sallen_key_lp", "sallen_key_hp")
SHORT_FAMILY = {"terse": 0.6, "roles_brief": 0.4}
SHORT = 60                          # знаков: «короткий» запрос (у v2 таких 18%, у v3 - 13%)
MIX2 = {"spec": 0.25, "named": 0.45, "circuit": 0.25, "refuse": 0.05}  # v3: 0.35 / 0.35 / 0.25 / 0.05
BATCH = generate_v3.BATCH
_VOLT = re.compile(r"\d\s*В\b|вольт", re.I)


# ---------------------------------------------------------------- пулы v3
def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def dc_defect(rec):
    """Д5: в эталоне режима 2 напряжение источника не 0, а в запросе напряжения нет."""
    m = re.match(r"V1 IN 0 DC (\S+)", rec.get("output") or "")
    return bool(m and float(m.group(1)) != 0 and not _VOLT.search(rec["prompt"]))


def protected(rec):
    """Записи пула режима 1, которые не убираются: короткие, парафразы, одиночные Саллен-Ки, отказы."""
    return len(rec["prompt"]) < SHORT or rec["family"] == "paraphrase" or rec["topology"] in SK or rec.get("refuse")


def drop_long(pool, n_drop, rng):
    """Убирает n_drop случайных незащищённых записей (все они - не короче SHORT знаков). Равномерно, а не по длине:
    вес по длине срезал бы narrative, path и table втрое против v3, а у v2 длинных форматов было больше, и короткие
    запросы от этого не страдали (проверено на пуле 14.11.2026)."""
    cand = [r for r in pool if not protected(r)]
    if n_drop > len(cand):
        raise ValueError(f"убрать {n_drop}, а незащищённых {len(cand)}")
    gone = {r["id"] for r in rng.sample(cand, n_drop)}
    return [r for r in pool if r["id"] not in gone], gone


# ---------------------------------------------------------------- новые записи режима 1
def _named_short_pool():
    return {t: w for t, w in filters_v3.named_pool().items()
            if t not in SK and t != "sk_cascade" and not t.startswith("cascade:")}


def sample_ref(kind, rng):
    if kind == "lc":
        return lc_v31.sample_lc(rng)
    if kind == "sk_short":
        return filters_v3.sample_named(rng.choice(SK), rng)
    if kind == "named_short" or rng.random() < 0.6:  # brief: названная схема или цепочка
        pool = _named_short_pool()
        return filters_v3.sample_named(rng.choices(list(pool), list(pool.values()))[0], rng)
    return compose_v3.sample_chain(rng)


def make_prompt(ref, kind, rng):
    """-> (task, family, запрос, ответ, сведения)."""
    r2 = random.Random(rng.random())
    net = ref["netlist"]
    if kind == "lc":
        task, fam, text = lc_v31.make_prompt(ref, r2)
        return task, fam, text, net, {}
    if kind in ("sk_short", "named_short"):
        fam = rng.choices(list(SHORT_FAMILY), list(SHORT_FAMILY.values()))[0]
        if fam == "terse" and generate_v3._simple(ref):
            return "A1", "terse", prompts_v2.describe_terse(ref, r2, 3), net, {}
        return "A1", "roles_brief", describe_ru.describe_a1_brief(ref, r2, 3), net, {}
    fam, prompt, answer, info = generate_v3.connections(net, r2, families={"brief": 1.0})
    return "A", fam, prompt, answer, info


def new_mode1(rng, seen, workdir, seed, counts):
    slots, kinds = [], [k for k, n in counts.items() for _ in range(n)]
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds):
        while True:
            ref = sample_ref(kind, rng)
            cid = generate_v3.circuit_id(ref["netlist"])
            if cid not in seen:
                seen.add(cid)
                break
        ref["id"], ref["circuit_id"], ref["kind"] = f"v31m1-{i:05d}", cid, kind
        slots.append(ref)
    admitted = []
    for i in range(0, len(slots), BATCH):
        admitted += pilot.admit(slots[i:i + BATCH], workdir / "mode1" / f"b{i // BATCH:03d}")
    for r in admitted:
        r["admission"]["cir"] = generate_v3.cir_ok(r["netlist"], r["f_char"])
        r["admission"]["admitted"] = r["admission"]["admitted"] and r["admission"]["cir"]
    rejected = [{"id": r["id"], "class": r["class"], "netlist": r["netlist"], "admission": r["admission"]}
                for r in admitted if not r["admission"]["admitted"]]
    records = []
    for ref in (r for r in admitted if r["admission"]["admitted"]):
        for _ in range(20):  # запрос LC и краткое описание проверяются обратным разбором; редкий сбой - другой текст
            try:
                task, family, prompt, answer, info = make_prompt(ref, ref["kind"], rng)
                break
            except (AssertionError, ValueError):
                continue
        else:
            rejected.append({"id": ref["id"], "class": ref["class"], "netlist": ref["netlist"], "admission": "prompt"})
            continue
        prompt, used = generate_v3.mutate(prompt, rng)
        rec = {"id": ref["id"], "circuit_id": ref["circuit_id"], "split": "train", "mode": 1, "task": task,
               "topology": ref["class"], "level": ref["level"], "family": family, "kind": ref["kind"],
               "prompt": prompt, "mutations": used, "netlist": answer, "output": answer, "canonical": ref["netlist"],
               "ir": canon.to_ir(canon.parse_netlist(answer)[0]), "params": ref["params"],
               "f_char": ref["f_char"], "quantities": ref["quantities"], "expected": ref["expected"],
               "measured": ref["measured"], "admission": ref["admission"], "gen": {"version": VERSION, "seed": seed}}
        rec.update(info)
        if ref.get("design"):
            rec["design"] = ref["design"]
        records.append(rec)
    return records, rejected


# ---------------------------------------------------------------- новые записи режима 2
def new_mode2(n, rng, seen, workdir, seed):
    records, rejected, k = [], [], 0
    while len(records) < n:
        batch = []
        for _ in range(min(BATCH, (n - len(records)) + 10)):
            while True:
                rec = mode2_data.sample(rng, mix=MIX2, dc0=True)
                if rec["circuit_id"] is None or rec["circuit_id"] not in seen:
                    seen.add(rec["circuit_id"])
                    break
            rec["id"] = f"v31m2-{k:05d}"
            k += 1
            batch.append(rec)
        ok, bad = generate_v3.admit_mode2(batch, workdir / "mode2" / f"b{k:05d}")
        rejected += [{"id": r["id"], "class": "mode2:" + r["kind"], "netlist": r["output"], "admission": r["admission"]}
                     for r in bad]
        for rec in ok[:n - len(records)]:
            rec["prompt"], used = generate_v3.mutate(rec["prompt"], rng, mode2=True)
            rec.update(split="train", mutations=used, level=None, gen={"version": VERSION, "seed": seed})
            rec.setdefault("topology", "refuse" if rec["kind"] == "refuse" else "spec")
            records.append(rec)
    return records, rejected


# ---------------------------------------------------------------- сборка
def _subsets(records, rng):
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["subset"] = SUBSETS[min(i * len(SUBSETS) // len(records), len(SUBSETS) - 1)]
    return records


def _write(path, records):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"records": len(records), "sha256": hashlib.sha256(data).hexdigest()}


def composition(records):
    """Сводка для README и сверки с v2 и v3."""
    out = {"records": len(records), "short": sum(len(r["prompt"]) < SHORT for r in records),
           "paraphrase": sum(r["family"] == "paraphrase" for r in records),
           "new": sum((r.get("gen") or {}).get("version") == VERSION for r in records)}
    if records and records[0].get("mode", 1) != 2:
        out.update(sk_single=sum(r["topology"] in SK for r in records),
                   mfb=sum(r["topology"].startswith("mfb") for r in records),
                   lc_without_r=sum(r["topology"].startswith("lc:") for r in records))
    for key in ("kind", "family"):
        c = {}
        for r in records:
            c[r[key]] = c.get(r[key], 0) + 1
        out[f"by_{key}"] = dict(sorted(c.items(), key=lambda x: -x[1]))
    return out


def generate(outdir, seed=20261114, scale=1.0, workdir=None):
    outdir = Path(outdir)
    workdir = Path(workdir or outdir / "runs")
    master = random.Random(seed)
    n_train = max(10, round(N_TRAIN * scale))
    counts = {k: max(1, round(v * scale)) for k, v in NEW1.items()}
    v3 = {s: read(V3 / f"{s}.jsonl") for s in ("train", "dev", "test", "gen_family", "gen_range")}
    seen = generate_v3.excluded_circuits() | {r["circuit_id"] for rs in v3.values() for r in rs if r.get("circuit_id")}
    manifest = {"version": VERSION, "seed": seed, "scale": scale, "base": "v3", "files": {}, "report": {}}

    rng = random.Random(master.random())
    new1, rej1 = new_mode1(rng, seen, workdir, seed, counts)
    pool1 = [r for r in v3["train"] if r.get("mode", 1) != 2]
    if scale < 1:  # проба: пул уменьшается вместе с набором
        pool1 = rng.sample(pool1, round(len(pool1) * scale))
    pool1, gone = drop_long(pool1, len(pool1) + len(new1) - n_train, rng)
    train1 = _subsets(pool1 + new1, rng)

    rng = random.Random(master.random())
    pool2 = [r for r in v3["train"] if r.get("mode") == 2]
    defect = [r["id"] for r in pool2 if dc_defect(r)]
    pool2 = [r for r in pool2 if r["id"] not in set(defect)]
    if scale < 1:  # проба: пул уменьшается вместе с набором
        pool2 = pool2[:max(1, round(len(pool2) * scale))]
    new2, rej2 = new_mode2(n_train - len(pool2), rng, seen, workdir, seed)
    train2 = _subsets(pool2 + new2, rng)

    for mode, train in ((1, train1), (2, train2)):
        d = outdir / f"mode{mode}"
        manifest["files"][f"mode{mode}/train.jsonl"] = _write(d / "train.jsonl", train)
        splits = ("dev", "test", "gen_family", "gen_range") if mode == 1 else ("dev", "test")
        for s in splits:
            recs = [r for r in v3[s] if r.get("mode", 1) == mode]
            manifest["files"][f"mode{mode}/{s}.jsonl"] = _write(d / f"{s}.jsonl", recs)
        manifest["report"][f"mode{mode}"] = composition(train)
    manifest["report"]["mode1"].update(dropped_from_v3=len(gone), rejected=len(rej1), rejected_detail=rej1)
    manifest["report"]["mode2"].update(dropped_d5=defect, rejected=len(rej2), rejected_detail=rej2)
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for mode in (1, 2):
        rep = manifest["report"][f"mode{mode}"]
        print(f"mode{mode}: {rep['records']} записей, новых {rep['new']}, коротких {rep['short']}, "
              f"отвергнуто {rep['rejected']}", flush=True)
    return manifest


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "out" / "v3_1_trial"
    scale = float(argv[argv.index("--scale") + 1]) if "--scale" in argv else 1.0
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 20261114
    generate(out, seed=seed, scale=scale)


if __name__ == "__main__":
    main(sys.argv)
