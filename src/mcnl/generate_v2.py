#!/usr/bin/env python3
"""Сборка датасета v2, gen-2 (docs/generator_v2_spec.md).

Запуск:  python src/mcnl/generate_v2.py --out out/v2_trial [--scale 0.05] [--seed 1]

Состав каждой выборки - таблица KINDS: названные фильтры (запрос A′ и описание A), композиции, странные схемы,
отказы. ФНЧ и ФВЧ преобладают (веса WEIGHTS): они и в реальных запросах основные. Схемы llm_test и C01 в
выборки не попадают. Запросы A и freeform проверяются обратным разбором до правок текста. 20% записей train
помечены paraphrase: их запросы заменит парафраза (шаг 7 спецификации).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import canon, cascade, compose, describe_ru, edge_cases, filters_v2, formulas, iso, pilot, prompts_v2, sampler, score  # noqa: E402

VERSION = "gen-2"
ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"train": 3000, "dev": 150, "test": 500, "gen_topology": 100, "gen_family": 100, "gen_range": 100}
KINDS = {"named_a1": 0.30, "named_a": 0.25, "compose": 0.30, "strange": 0.10, "refuse": 0.05}
HELD_OUT = {"khn", "twin_t", "mfb_hp"}
HELD_OUT_FAMILY = "path"
RARE = {"rlc_series_bp", "rlc_parallel_bp", "mfb_bp", "rlc_series_bs", "twin_t", "tow_thomas", "khn", "lc_ladder5"}
WEIGHTS = {"common": 3, "rare": 1, "cascade": 1}
FAMILY_A = {"brief": 0.30, "freeform": 0.30, "list": 0.15, "table": 0.10, "narrative": 0.15}
FAMILY_A1 = {"roles_brief": 0.4, "roles": 0.3, "terse": 0.3}
TERSE_MAX_PARAMS = 5   # без подписей ролей длинный список номиналов нечитаем
PARAPHRASE_SHARE = 0.20
BATCH = 300
SUBSETS = (1000, 2000, 3000)  # вложенные подмножества train


def circuit_id(netlist):
    return hashlib.sha256(netlist.encode("utf-8")).hexdigest()[:16]


def excluded_circuits():
    """Схемы замороженной и отладочной выборок LLM: train их не видит."""
    out = set()
    for p in (ROOT / "data/dataset/llm_test_v1/llm_test.jsonl", ROOT / "data/human_test/v1/paraphrase.jsonl"):
        if p.exists():
            out |= {r["circuit_id"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r.get("circuit_id")}
    return out


def named_pool(split):
    names = list(formulas.TOPOLOGIES) + list(filters_v2.TOPOLOGIES_V2)
    if split == "gen_topology":
        return {t: 1 for t in HELD_OUT}
    pool = {t: WEIGHTS["rare" if t in RARE else "common"] for t in names if t not in HELD_OUT}
    if split == "gen_range":
        return {t: w for t, w in pool.items() if any(p.startswith("R") for p in _params(t))}
    pool.update({"cascade:" + k: WEIGHTS["cascade"] for k in cascade.KINDS})
    return pool


def _params(topology):
    import inspect
    fn = filters_v2.TOPOLOGIES_V2[topology][0] if topology in filters_v2.TOPOLOGIES_V2 else formulas.TOPOLOGIES[topology][0]
    return [p for p in inspect.signature(fn).parameters if p != "dc"]


def sample_named(pool, rng, split):
    name = rng.choices(list(pool), list(pool.values()))[0]
    if name.startswith("cascade:"):
        return cascade.sample(name.split(":")[1], rng)
    if split != "gen_range":
        return filters_v2.sample_any(name, rng)
    if name in filters_v2.TOPOLOGIES_V2:
        return filters_v2.sample(name, rng, sampler.RANGES_GEN)[0]
    return sampler.sample(name, rng, sampler.RANGES_GEN, pick=filters_v2.pick)[0]


def sample_circuit(kind, pool, bases, rng, split):
    if kind in ("named_a1", "named_a"):
        return sample_named(pool, rng, split)
    if kind == "compose":
        return compose.sample_chain(rng) if rng.random() < 0.5 else compose.sample_addon(rng, bases)
    return edge_cases.strange(rng.choice(edge_cases.STRANGE), rng)


def verified_a(netlist, family, rng):
    """Запрос A с обратным разбором: схема должна восстанавливаться однозначно."""
    target = canon.parse_netlist(netlist)[0]
    if family == "freeform":
        prompt = prompts_v2.describe_freeform(netlist, rng)
        back = prompts_v2.parse_freeform(prompt)
    else:
        prompt = describe_ru.describe_a(netlist, family, rng)
        back = describe_ru.parse_a(prompt, family)
    if not iso.match(target, back, "values")["match"]:
        raise AssertionError(f"неоднозначный запрос: {prompt}")
    return prompt


def _simple(ref):
    return isinstance(ref["params"], dict) and len([k for k in ref["params"] if k != "dc"]) <= TERSE_MAX_PARAMS


def make_prompt(ref, kind, split, rng):
    """-> (task, family, prompt)."""
    r2 = random.Random(rng.random())
    families = {HELD_OUT_FAMILY: 1.0} if split == "gen_family" else FAMILY_A
    if kind == "named_a1" and split != "gen_family":
        fam = rng.choices(list(FAMILY_A1), list(FAMILY_A1.values()))[0]
        if fam == "terse" and _simple(ref):
            return "A1", "terse", prompts_v2.describe_terse(ref, r2)
        if fam == "roles":
            return "A1", "roles", describe_ru.describe_a1(ref, r2)
        return "A1", "roles_brief", describe_ru.describe_a1_brief(ref, r2)
    if kind == "compose" and split != "gen_family" and rng.random() < 0.6:
        if "sections" in ref["params"]:
            return "A", "chain", compose.describe_chain(ref["params"]["sections"], r2)
        return "A1", "addon", compose.describe_addon(ref, r2)
    fam = rng.choices(list(families), list(families.values()))[0]
    return "A", fam, verified_a(ref["netlist"], fam, r2)


def plan(n, rng):
    """Точные доли видов: последовательность из n видов в случайном порядке."""
    kinds = [k for k, share in KINDS.items() for _ in range(round(n * share))]
    kinds += ["named_a"] * (n - len(kinds))
    rng.shuffle(kinds)
    return kinds[:n]


def refusal_record(rng, i, split, seed):
    r = edge_cases.refusal(rng.choice(score.REFUSE_CODES), rng)
    prompt, used = prompts_v2.mutate(r["prompt"], rng)
    return {"id": f"{split}-{i:05d}", "circuit_id": None, "split": split, "task": "refuse", "topology": "refuse", "level": None,
            "family": "refuse", "prompt": prompt, "mutations": used, "refuse": r["refuse"], "reason": r["reason"],
            "output": r["output"], "gen": {"version": VERSION, "seed": seed}}


def build_split(split, n, rng, seen, workdir, seed):
    pool = named_pool(split)
    bases = [b for b in compose.BASES if b not in HELD_OUT]
    kinds = plan(n, rng) if split not in ("gen_topology", "gen_range") else ["named_a1", "named_a"] * (n // 2 + 1)
    records, slots = [], []
    for i, kind in enumerate(kinds[:n]):
        if kind == "refuse":
            records.append(refusal_record(rng, i, split, seed))
            continue
        while True:
            ref = sample_circuit(kind, pool, bases, rng, split)
            cid = circuit_id(ref["netlist"])
            if cid not in seen:
                seen.add(cid)
                break
        ref["id"], ref["circuit_id"], ref["kind"] = f"{split}-{i:05d}", cid, kind
        slots.append(ref)
    admitted = []
    for i in range(0, len(slots), BATCH):
        admitted += pilot.admit(slots[i:i + BATCH], workdir / split / f"b{i // BATCH:03d}")
    ok = [r for r in admitted if r["admission"]["admitted"]]
    rejected = [r for r in admitted if not r["admission"]["admitted"]]
    for ref in ok:
        task, family, prompt = make_prompt(ref, ref["kind"], split, rng)
        prompt, used = prompts_v2.mutate(prompt, rng, force=split == "gen_family")
        records.append({"id": ref["id"], "circuit_id": ref["circuit_id"], "split": split, "task": task,
                        "topology": ref["class"], "level": ref["level"], "family": family, "kind": ref["kind"],
                        "prompt": prompt, "mutations": used, "netlist": ref["netlist"], "output": ref["netlist"],
                        "ir": canon.to_ir(canon.parse_netlist(ref["netlist"])[0]), "params": ref["params"],
                        "f_char": ref["f_char"], "quantities": ref["quantities"], "expected": ref["expected"],
                        "measured": ref["measured"], "admission": ref["admission"],
                        "gen": {"version": VERSION, "seed": seed}})
    records.sort(key=lambda r: r["id"])
    return records, rejected


def generate(outdir, seed=1, scale=1.0, workdir=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir or ROOT / "out" / "gen2_runs")
    master = random.Random(seed)
    seen = excluded_circuits()
    manifest = {"version": VERSION, "seed": seed, "scale": scale, "files": {}, "report": {}}
    for split, n_full in SPLITS.items():
        rng = random.Random(master.random())
        n = max(1, round(n_full * scale))
        records, rejected = build_split(split, n, rng, seen, workdir, seed)
        if split == "train":
            for i, r in enumerate(records):
                r["subset"] = SUBSETS[min(i * 3 // n, 2)]
                r["paraphrase"] = rng.random() < PARAPHRASE_SHARE
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
        (outdir / f"{split}.jsonl").write_bytes(data)
        manifest["files"][f"{split}.jsonl"] = {"records": len(records), "sha256": hashlib.sha256(data).hexdigest()}
        manifest["report"][split] = {"rejected": len(rejected),
                                     "rejected_detail": [{"id": r["id"], "class": r["class"], "netlist": r["netlist"]} for r in rejected]}
        print(f"{split:13s} records {len(records):5d}  rejected {len(rejected)}", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "out" / "gen2_trial"
    scale = float(argv[argv.index("--scale") + 1]) if "--scale" in argv else 1.0
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 1
    generate(out, seed=seed, scale=scale)


if __name__ == "__main__":
    main(sys.argv)
