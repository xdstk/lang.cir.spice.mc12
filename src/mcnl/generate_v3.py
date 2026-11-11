#!/usr/bin/env python3
"""Сборка датасета v3, gen-3 (docs/generator_v3_spec.md).

Запуск:  python src/mcnl/generate_v3.py --out out/v3_trial [--scale 0.05] [--seed 1]

Отличия от gen-2 (generate_v2.py не меняется, v2 воспроизводится):
- режим 2 (mode2_data) - доля MODE2_SHARE в train, dev и test; у записи поле mode = 2 и своя инструкция модели;
- виды записей режима 1 (KINDS): названные схемы (набор filters_v3: фокус на фильтрах, бывшие отложенные топологии,
  активный двойной Т-мост, каскады Саллен-Ки), цепочки и добавки compose_v3, схемы на 2-4 ОУ, название против
  соединений (П7), странные схемы, отказы;
- описания соединений - prompts_v3: имена узлов из запроса сохраняются в эталоне (доля NAMED_SHARE), описание не по
  порядку; семейство path - в train;
- исправления Д1-Д4 (gen=3 у describe_ru, prompts_v2);
- исключаются схемы llm_test, llm_test2, mc12_real и C01;
- допуск: MC12 и сверка с узловым расчётом (pilot.admit), как в v2, плюс каждая схема строится в .CIR
  (spice2cir.build); у режима 2 в MC12 считается подобранная схема.
Поле netlist записи - ответ модели (с именами узлов из запроса), canonical - канонический нетлист той же схемы.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import (bench, canon, compose_v3, describe_ru, edge_cases, filters_v2, filters_v3, generate_v2, iso, measure,  # noqa: E402
                  mode2_data, pilot, prompts_v2, prompts_v3, score, spice2cir)

VERSION = "gen-3"
ROOT = Path(__file__).resolve().parents[2]
# gen_topology в v3 нет: отложенные топологии v2 перенесены в train, своя отложенная топология - вопрос В1
SPLITS = {"train": 3000, "dev": 150, "test": 500, "gen_family": 100, "gen_range": 100}
MODE2_SHARE = 0.18          # доля примеров режима 2 в train, dev, test (15-20%)
KINDS = {"named_a1": 0.28, "named_a": 0.24, "chain": 0.13, "addon": 0.15, "multi": 0.06, "strange": 0.08,
         "mismatch": 0.01, "refuse": 0.05}                     # доли внутри режима 1
FAMILY_A = {"brief": 0.22, "freeform": 0.30, "list": 0.13, "table": 0.10, "narrative": 0.13, "path": 0.12}
FAMILY_A1 = {"roles_brief": 0.4, "roles": 0.3, "terse": 0.3}
OWN_PROMPT = 0.7            # цепочка и добавка: доля запросов своим шаблоном, остальные - описание соединений
NAMED_SHARE = 0.45          # описание соединений: доля запросов с буквенными и произвольными именами узлов (П1)
DESIGN_NAME = 0.5           # каскад Саллен-Ки: доля запросов с названием фильтра («ФНЧ Баттерворта 4-го порядка»)
PARAPHRASE_SHARE = 0.20
MAX_PROMPT = 600            # знаков: длиннее - схема описывается более кратким семейством (замечание автора 11.11.2026;
                            # запросы mc12_real - до 503 знаков, llm_test и llm_test2 - до 129)
PREFIX_SHARE = 0.03         # вступления «собери», «сделай» - реже, чем в v2 (0,08): замечание автора 11.11.2026
BATCH = 300
SUBSETS = (1000, 2000, 3000)
REASON_DESIGN = "номиналы не заданы, расчёт номиналов - режим 2"
MISMATCH = {"rc_lp": "rc_hp", "rc_hp": "rc_lp", "rl_lp": "rl_hp", "rl_hp": "rl_lp"}
MC12_FREQ_TOL = 0.01        # режим 2: срез или f0 подобранной схемы, MC12 против узлового расчёта


def circuit_id(netlist):
    return hashlib.sha256(netlist.encode("utf-8")).hexdigest()[:16]


def excluded_circuits():
    """Схемы замороженных выборок и C01: train v3 их не видит."""
    out = set()
    for p in ("data/dataset/llm_test_v1/llm_test.jsonl", "data/dataset/llm_test2_v1/llm_test2.jsonl",
              "data/dataset/mc12_real_v1/mc12_real.jsonl", "data/human_test/v1/paraphrase.jsonl"):
        p = ROOT / p
        if p.exists():
            out |= {r["circuit_id"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r.get("circuit_id")}
    return out


def plan(n, rng, mode2=True):
    """Виды записей: точные доли, случайный порядок. "mode2" - пример режима 2."""
    n2 = round(n * MODE2_SHARE) if mode2 else 0
    n1 = n - n2
    counts = {k: round(n1 * share) for k, share in KINDS.items()}
    counts["named_a"] = max(0, counts["named_a"] + n1 - sum(counts.values()))  # остаток округления - в самый частый вид
    kinds = ([k for k, c in counts.items() for _ in range(c)] + ["named_a"] * n1)[:n1] + ["mode2"] * n2
    rng.shuffle(kinds)
    return kinds


# ---------------------------------------------------------------- схемы
def sample_circuit(kind, rng, split):
    if kind in ("named_a1", "named_a"):
        if split == "gen_range":
            return generate_v2.sample_named(generate_v2.named_pool("gen_range"), rng, "gen_range")
        pool = filters_v3.named_pool()
        return filters_v3.sample_named(rng.choices(list(pool), list(pool.values()))[0], rng)
    if kind == "chain":
        return compose_v3.sample_chain(rng)
    if kind == "addon":
        return compose_v3.sample_addon(rng)
    if kind == "multi":
        return compose_v3.sample_multi(rng)
    if kind == "mismatch":
        return filters_v2.sample_any(rng.choice(list(MISMATCH)), rng)
    return edge_cases.strange(rng.choice(edge_cases.STRANGE), rng)


# ---------------------------------------------------------------- запросы
def mutate(prompt, rng, force=False, mode2=False):
    """Правки текста v2 с другой долей вступлений; у режима 2 - без вступлений и хвостов о чертеже."""
    used = []
    for name, p, fn in prompts_v2.MUTATIONS:
        if name in ("prefix", "suffix") and (force or mode2):
            continue
        p = PREFIX_SHARE if name == "prefix" else p
        if force or rng.random() < p:
            new = fn(prompt, rng)
            if new != prompt:
                prompt, used = new, used + [name]
    return prompt, used


def connections(netlist, rng, families=None):
    """Описание соединений -> (семейство, запрос, ответ модели, сведения). brief - describe_ru (узлы т1 -> N1),
    остальные - prompts_v3 с проверкой обратным разбором; если семейство не подошло схеме - freeform."""
    families = families or FAMILY_A
    fam = rng.choices(list(families), list(families.values()))[0]
    target = canon.parse_netlist(netlist)[0]
    scheme = None if rng.random() < NAMED_SHARE else "seq"
    found = []
    for f in dict.fromkeys((fam, "freeform", "brief")):  # выбранное семейство, затем более краткие
        try:
            if f == "brief":
                prompt = describe_ru.describe_a(netlist, "brief", rng, gen=3)
                if not iso.match(target, describe_ru.parse_a(prompt, "brief"), "values")["match"]:
                    continue
                cand = ("brief", prompt, netlist, {})
            else:
                d = prompts_v3.verified(netlist, f, rng, scheme=scheme)
                cand = (f, d["prompt"], d["netlist"], {"names": d["names"], "shuffled": d["shuffled"]})
        except (AssertionError, ValueError):  # разбор v1 не рассчитан на элемент на источнике и т.п.
            continue
        if len(cand[1]) <= MAX_PROMPT:
            return cand
        found.append(cand)
    if not found:
        raise AssertionError(f"запрос не построен: {netlist}")
    return min(found, key=lambda c: len(c[1]))  # схема велика для любого семейства: самый короткий запрос


def _simple(ref):
    return isinstance(ref["params"], dict) and len([k for k in ref["params"] if k != "dc"]) <= generate_v2.TERSE_MAX_PARAMS


def make_prompt(ref, kind, split, rng):
    """-> (task, family, запрос, ответ модели, сведения о запросе)."""
    r2 = random.Random(rng.random())
    net = ref["netlist"]
    if kind == "named_a1":
        fam = rng.choices(list(FAMILY_A1), list(FAMILY_A1.values()))[0]
        if fam == "terse" and _simple(ref):
            return "A1", "terse", prompts_v2.describe_terse(ref, r2, 3), net, {}
        fam = "roles" if fam == "roles" else "roles_brief"
        text = describe_ru.describe_a1(ref, r2, 3) if fam == "roles" else describe_ru.describe_a1_brief(ref, r2, 3)
        if len(text) > MAX_PROMPT and fam == "roles":  # длинный перечень ролей (каскады) - краткой записью
            fam, text = "roles_brief", describe_ru.describe_a1_brief(ref, r2, 3)
        info = {}
        if ref.get("design") and r2.random() < DESIGN_NAME:
            name = filters_v3.design_name(ref, r2)
            text = f"{name[:1].upper() + name[1:]}. {text}" if fam == "roles" else f"{name}, {text}"
            info["design_name"] = True
        return "A1", fam, text, net, info
    if kind == "chain" and r2.random() < OWN_PROMPT:
        d = compose_v3.describe_chain(ref["params"]["sections"], r2)
        return "A", "chain", d["prompt"], net, {"reordered": d["reordered"]}
    if kind == "addon" and r2.random() < OWN_PROMPT:
        d = min((compose_v3.describe_addon(ref, r2) for _ in range(4)), key=lambda x: max(len(x["prompt"]), MAX_PROMPT))
        return "A1", "addon", d["prompt"], net, {"reordered": d["reordered"], "explicit_out": d["explicit_out"]}
    if kind == "mismatch":  # П7: названа одна схема, описаны соединения другой - эталон по соединениям
        d = prompts_v3.verified(net, "freeform", r2, scheme="seq", shuffle=0)
        wrong = r2.choice(describe_ru.ROLES_ALL[MISMATCH[ref["class"]]]["short_names"])
        return "A", "freeform", f"{wrong}: {d['prompt']}", d["netlist"], {"named_as": MISMATCH[ref["class"]]}
    fam, prompt, answer, info = connections(net, r2)
    return "A", fam, prompt, answer, info


def refusal_record(rng, rid, split, seed):
    r = edge_cases.refusal(rng.choice(score.REFUSE_CODES), rng)
    if r["refuse"] == "needs_design":  # в v3 расчёт номиналов - режим 2, не дизайнер
        r["reason"], r["output"] = REASON_DESIGN, edge_cases.refusal_output("needs_design", REASON_DESIGN)
    prompt, used = mutate(r["prompt"], rng)
    return {"id": rid, "circuit_id": None, "split": split, "mode": 1, "task": "refuse", "topology": "refuse", "level": None,
            "family": "refuse", "kind": "refuse", "prompt": prompt, "mutations": used, "refuse": r["refuse"],
            "reason": r["reason"], "output": r["output"], "gen": {"version": VERSION, "seed": seed}}


# ---------------------------------------------------------------- допуск
def cir_ok(netlist, f_char=None):
    """Схема строится в .CIR (программа сможет открыть её в MC12)."""
    try:
        return bool(spice2cir.build(netlist, f_char))
    except Exception:  # любая ошибка раскладки - схема в датасет не идёт
        return False


def admit_mode2(recs, workdir):
    """Подобранные схемы режима 2 в MC12: запуск, тип АЧХ, срез или f0 против узлового расчёта, .CIR.
    -> (допущенные, отвергнутые). Отказы и маршрут дизайнера (схему строит не synth) проходят без прогона."""
    run_me = [r for r in recs if r.get("solution") and r["solution"]["route"] != "designer"]
    runs = bench.run([(r["id"], canon.parse_netlist(r["solution"]["netlist"])[0], r["spec"].get("fc") or r["spec"].get("f0"))
                      for r in run_me], workdir, timeout=1800) if run_me else {}
    ok, bad = [], []
    for r in recs:
        if r not in run_me:
            ok.append(r)
            continue
        run, sol = runs[r["id"]], r["solution"]
        adm = {"mc12_run": run["ok"], "cir": cir_ok(sol["netlist"])}
        if run["ok"]:
            got = measure.from_output(run["output"])
            key = "fc" if "fc" in sol["wanted"] else "f0"
            adm["type_ok"] = got.get("type") == sol["achieved"]["type"]
            adm["freq_rel"] = (got[key] / sol["achieved"][key] - 1) if got.get(key) and sol["achieved"].get(key) else None
            adm["admitted"] = bool(adm["cir"] and adm["type_ok"] and adm["freq_rel"] is not None
                                   and abs(adm["freq_rel"]) <= MC12_FREQ_TOL)
        else:
            adm["admitted"] = False
        r["admission"] = adm
        (ok if adm["admitted"] else bad).append(r)
    return ok, bad


# ---------------------------------------------------------------- сборка
def build_split(split, n, rng, seen, workdir, seed):
    if split == "gen_range":
        kinds = (["named_a1", "named_a"] * (n // 2 + 1))[:n]
    else:
        kinds = plan(n, rng, mode2=split != "gen_family")
    records, slots, mode2 = [], [], []
    for i, kind in enumerate(kinds):
        rid = f"{split}-{i:05d}"
        if kind == "refuse":
            records.append(refusal_record(rng, rid, split, seed))
            continue
        if kind == "mode2":
            while True:
                rec = mode2_data.sample(rng)
                if rec["circuit_id"] is None or rec["circuit_id"] not in seen:
                    seen.add(rec["circuit_id"])
                    break
            rec["id"] = rid
            mode2.append(rec)
            continue
        while True:
            ref = sample_circuit(kind, rng, split)
            cid = circuit_id(ref["netlist"])
            if cid not in seen:
                seen.add(cid)
                break
        ref["id"], ref["circuit_id"], ref["kind"] = rid, cid, kind
        slots.append(ref)
    admitted = []
    for i in range(0, len(slots), BATCH):
        admitted += pilot.admit(slots[i:i + BATCH], workdir / split / f"b{i // BATCH:03d}")
    for r in admitted:
        r["admission"]["cir"] = cir_ok(r["netlist"], r["f_char"])
        r["admission"]["admitted"] = r["admission"]["admitted"] and r["admission"]["cir"]
    rejected = [{"id": r["id"], "class": r["class"], "netlist": r["netlist"], "cir": r["admission"]["cir"]}
                for r in admitted if not r["admission"]["admitted"]]
    for ref in (r for r in admitted if r["admission"]["admitted"]):
        task, family, prompt, answer, info = make_prompt(ref, ref["kind"], split, rng)
        prompt, used = mutate(prompt, rng, force=split == "gen_family")
        rec = {"id": ref["id"], "circuit_id": ref["circuit_id"], "split": split, "mode": 1, "task": task,
               "topology": ref["class"], "level": ref["level"], "family": family, "kind": ref["kind"],
               "prompt": prompt, "mutations": used, "netlist": answer, "output": answer, "canonical": ref["netlist"],
               "ir": canon.to_ir(canon.parse_netlist(answer)[0]), "params": ref["params"], "f_char": ref["f_char"],
               "quantities": ref["quantities"], "expected": ref["expected"], "measured": ref["measured"],
               "admission": ref["admission"], "gen": {"version": VERSION, "seed": seed}}
        rec.update(info)
        if ref.get("design"):
            rec["design"] = ref["design"]
        records.append(rec)
    ok2, bad2 = admit_mode2(mode2, workdir / split / "mode2")
    rejected += [{"id": r["id"], "class": "mode2:" + r["kind"], "netlist": r["output"], "admission": r["admission"]} for r in bad2]
    for rec in ok2:
        rec["prompt"], used = mutate(rec["prompt"], rng, mode2=True)
        rec.update(split=split, mutations=used, level=None, gen={"version": VERSION, "seed": seed})
        rec.setdefault("topology", "refuse" if rec["kind"] == "refuse" else "spec")
        records.append(rec)
    records.sort(key=lambda r: r["id"])
    return records, rejected


def generate(outdir, seed=1, scale=1.0, workdir=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir or ROOT / "out" / "gen3_runs")
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
        manifest["report"][split] = {"rejected": len(rejected), "rejected_detail": rejected}
        print(f"{split:13s} records {len(records):5d}  rejected {len(rejected)}", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "out" / "v3_trial"
    scale = float(argv[argv.index("--scale") + 1]) if "--scale" in argv else 1.0
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 1
    generate(out, seed=seed, scale=scale, workdir=out / "runs")


if __name__ == "__main__":
    main(sys.argv)
