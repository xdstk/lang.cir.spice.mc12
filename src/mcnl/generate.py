#!/usr/bin/env python3
"""Сборка синтетического датасета (docs/generator_spec.md; решения автора В1-В10).

Запуск:  python generate.py --out out/gen_trial --scale 0.07 [--seed 1]     (пробная партия ~200 записей)
         python generate.py --out data/dataset/v1                          (полная, ~2950 записей)

Выборки (В8): train - 1000 схем x 2 формулировки, dev 150, test 500, gen_topology / gen_family /
gen_range по 100. Доли уровней L1/L2/L3 - 30/45/25, задачи A/A' - 70/30. Схема попадает в датасет,
только если прошла допуск (pilot.admit: канон, MC12, формула или mna.py), а каждый запрос задачи A -
только если его обратный разбор изоморфен метке. circuit_id уникален по всем выборкам.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

try:
    from . import canon, cascade, describe_ru, formulas, iso, pilot, sampler
except ImportError:  # запуск как скрипта
    import canon
    import cascade
    import describe_ru
    import formulas
    import iso
    import pilot
    import sampler

VERSION = "gen-1"
ROOT = Path(__file__).resolve().parents[2]
SPLITS = {  # выборка: (число схем, формулировок на схему)
    "train": (1000, 2), "dev": (150, 1), "test": (500, 1),
    "gen_topology": (100, 1), "gen_family": (100, 1), "gen_range": (100, 1),
}
LEVEL_SHARE = {"L1": 0.30, "L2": 0.45, "L3": 0.25}
TASK_A_SHARE = 0.7
# семейства задачи A и их веса: краткая запись (brief) - чаще, реальные запросы короче (замечание автора);
# у A' - поровну полная (roles) и краткая (roles_brief) запись
FAMILY_WEIGHTS = {"brief": 0.4, "list": 0.2, "table": 0.2, "narrative": 0.2}
A1_BRIEF_SHARE = 0.5
HELD_OUT_TOPOLOGIES = {"mfb_bp", "rl_hp"}
HELD_OUT_CASCADES = {"HP+LP"}
HELD_OUT_FAMILY = "path"
OVERSAMPLE = 1.08          # запас на схемы, не прошедшие допуск
BATCH = 300                # схем на один запуск MC12


def circuit_id(netlist):
    return hashlib.sha256(netlist.encode("utf-8")).hexdigest()[:16]


def _pool(split):
    """Какие топологии / виды каскадов доступны выборке, по уровням."""
    l12 = {lvl: [t for t, v in formulas.TOPOLOGIES.items() if v[1] == lvl] for lvl in ("L1", "L2")}
    if split == "gen_topology":
        return {"L1": [t for t in l12["L1"] if t in HELD_OUT_TOPOLOGIES],
                "L2": [t for t in l12["L2"] if t in HELD_OUT_TOPOLOGIES],
                "L3": sorted(HELD_OUT_CASCADES)}
    pool = {lvl: [t for t in ts if t not in HELD_OUT_TOPOLOGIES] for lvl, ts in l12.items()}
    pool["L3"] = [k for k in cascade.KINDS if k not in HELD_OUT_CASCADES]
    if split == "gen_range":  # номиналы R вне диапазона обучения: только схемы с резисторами, без каскадов
        pool = {lvl: [t for t in ts if any(p.startswith("R") for p in _params(t))] for lvl, ts in pool.items()
                if lvl != "L3"}
    return pool


def _params(topology):
    import inspect
    return [p for p in inspect.signature(formulas.TOPOLOGIES[topology][0]).parameters if p != "dc"]


def level_plan(levels, n, rng):
    """Точное разбиение n схем по долям уровней (а не случайный выбор уровня на каждую схему):
    на малых выборках случайный выбор давал, например, 14% L3 вместо 25%."""
    shares = [LEVEL_SHARE[l] for l in levels]
    total = sum(shares)
    counts = [int(n * s / total) for s in shares]
    for k in sorted(range(len(levels)), key=lambda k: -(n * shares[k] / total - counts[k]))[:n - sum(counts)]:
        counts[k] += 1
    plan = [l for l, c in zip(levels, counts) for _ in range(c)]
    rng.shuffle(plan)
    return plan


def sample_circuits(split, n, rng, seen):
    pool = {lvl: list(ts) for lvl, ts in _pool(split).items() if ts}
    ranges = sampler.RANGES_GEN if split == "gen_range" else sampler.RANGES
    refs = []
    for lvl in level_plan(list(pool), n, rng):
        for _ in range(500):  # попытки внутри уровня: фильтр gen_range не должен смещать доли уровней
            name = rng.choice(pool[lvl])
            try:
                ref = cascade.sample(name, rng, ranges) if lvl == "L3" else sampler.sample(name, rng, ranges)[0]
            except RuntimeError as err:  # для топологии нет допустимых номиналов в этих диапазонах
                print(f"  {split}: {name} исключена из выборки: {err}", flush=True)
                pool[lvl].remove(name)
                if not pool[lvl]:
                    raise RuntimeError(f"{split}: на уровне {lvl} не осталось топологий") from err
                continue
            if split == "gen_range" and not any(isinstance(v, float) and k.startswith("R") and v > 1e6
                                                for k, v in ref["params"].items()):
                continue
            cid = circuit_id(ref["netlist"])
            if cid in seen:
                continue
            seen.add(cid)
            ref["id"] = f"{split}-{len(refs):05d}"
            ref["circuit_id"] = cid
            ref["split"] = split
            refs.append(ref)
            break
        else:
            raise RuntimeError(f"{split}: не набрана схема уровня {lvl}")
    return refs


def prompts_for(ref, k, rng, split):
    """k формулировок схемы: (task, family, prompt). Запросы A проверяются обратным разбором."""
    out = []
    families = {HELD_OUT_FAMILY: 1.0} if split == "gen_family" else FAMILY_WEIGHTS
    target = canon.parse_netlist(ref["netlist"])[0]
    while len(out) < k:
        task = "A" if split == "gen_family" or rng.random() < TASK_A_SHARE else "A1"
        if task == "A1":
            if rng.random() < A1_BRIEF_SHARE:
                prompt, family = describe_ru.describe_a1_brief(ref, random.Random(rng.random())), "roles_brief"
            else:
                prompt, family = describe_ru.describe_a1(ref, random.Random(rng.random())), "roles"
        else:
            family = rng.choices(list(families), list(families.values()))[0]
            prompt = describe_ru.describe_a(ref["netlist"], family, random.Random(rng.random()))
            try:
                ok = iso.match(target, describe_ru.parse_a(prompt, family), "values")["match"]
            except ValueError:
                ok = False
            if not ok:
                raise AssertionError(f"неоднозначный запрос для {ref['id']}: {prompt}")
        if prompt not in [p for _, _, p in out]:
            out.append((task, family, prompt))
    return out


def generate(outdir, seed=1, scale=1.0, workdir=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir or ROOT / "out" / "gen_runs")
    master = random.Random(seed)
    seen, report, manifest = set(), {}, {"version": VERSION, "seed": seed, "scale": scale, "files": {}}
    for split, (n_circ, k) in SPLITS.items():
        rng = random.Random(master.random())
        n = max(1, round(n_circ * scale))
        refs = sample_circuits(split, int(n * OVERSAMPLE) + 1, rng, seen)
        admitted = []
        for i in range(0, len(refs), BATCH):
            admitted += pilot.admit(refs[i:i + BATCH], workdir / split / f"b{i // BATCH:03d}")
        ok = [r for r in admitted if r["admission"]["admitted"]]
        rejected = [r for r in admitted if not r["admission"]["admitted"]]
        report[split] = {"sampled": len(refs), "admitted": len(ok), "rejected": len(rejected),
                         "rejected_detail": [{"id": r["id"], "class": r["class"], "netlist": r["netlist"],
                                              "f_char": r["f_char"], "admission": r["admission"]} for r in rejected]}
        records = []
        for ref in ok[:n]:
            for j, (task, family, prompt) in enumerate(prompts_for(ref, k, rng, split)):
                records.append({
                    "id": f"{ref['id']}-{j}", "circuit_id": ref["circuit_id"], "split": split, "task": task,
                    "topology": ref["class"], "level": ref["level"], "family": family, "prompt": prompt,
                    "netlist": ref["netlist"], "ir": canon.to_ir(canon.parse_netlist(ref["netlist"])[0]),
                    "params": ref["params"], "f_char": ref["f_char"], "quantities": ref["quantities"],
                    "expected": ref["expected"], "measured": ref["measured"], "admission": ref["admission"],
                    "gen": {"version": VERSION, "seed": seed}})
        path = outdir / f"{split}.jsonl"
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
        path.write_bytes(data)
        manifest["files"][path.name] = {"records": len(records), "circuits": len({r["circuit_id"] for r in records}),
                                        "sha256": hashlib.sha256(data).hexdigest()}
        print(f"{split:13s} circuits {manifest['files'][path.name]['circuits']:5d}  records {len(records):5d}  "
              f"rejected {len(rejected)}", flush=True)
    manifest["report"] = report
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "out" / "gen_trial"
    scale = float(argv[argv.index("--scale") + 1]) if "--scale" in argv else 1.0
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 1
    generate(out, seed=seed, scale=scale)


if __name__ == "__main__":
    main(sys.argv)
