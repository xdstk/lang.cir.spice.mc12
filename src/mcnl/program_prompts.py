#!/usr/bin/env python3
"""Запросы, которые программа сама показывает пользователю, с эталонами - выборка program (решение автора 16.11.2026:
«было бы неловко, если в новой модели примеры из справочника не работали»). Каждая новая модель должна проходить её
на 100%.

Источники: справочник (handbook_articles: запросы режимов 1 и 2), меню «Примеры» (gui.EXAMPLES, gui.EXAMPLES_MODE2),
вкладка «Правила» (gui.RULES, примеры «В запрос»; режим - по разделу). Одинаковые запросы склеены (поле sources).
Эталон режима 1 - схема статьи справочника или REFERENCES ниже (схема или код отказа), режима 2 - ответ модели с `?`
и `.SPEC` (как в инструкции режима 2). Новый пример в программе без эталона роняет тест.

Проверка: режим 1 - схема с номиналами совпала с эталоном (S3) и MC12 её посчитал (S1), отказ - верный код;
режим 2 - подобранная по ответу схема попала в цели в MC12 (score2 behavior: частота и Q - 10%, усиление - 1 дБ).

Запуск:
  python src/mcnl/program_prompts.py --export            запросы -> data/program_prompts/program.jsonl (тест сверяет)
  python src/mcnl/program_prompts.py --check A.jsonl [B.jsonl ...]   ответы (predict.py в Colab или локально) -> MC12
  python src/mcnl/program_prompts.py --local --adapter папка [--trained-on v2] [--adapter-mode2 папка]
         [--trained-on-mode2 v3]       ответы локальной модели (venv-hf) -> out/program_prompts/<имя>.jsonl -> --check
В Colab:  python colab/predict.py ... --records data/program_prompts/program.jsonl --mode 1 (или 2) --out program.jsonl
"""
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import bench, canon, evaluate, measure, mna, score, synth  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data" / "program_prompts" / "program.jsonl"
OUT = ROOT / "out" / "program_prompts"

# Эталоны запросов меню «Примеры» и вкладки «Правила» (запросы справочника берут схему своей статьи).
# Режим 1: нетлист или ("REFUSE", код); режим 2: ответ модели (нетлист с ? и .SPEC или одна строка .SPEC).
REFERENCES = {
    (1, "Делитель напряжения: верхний резистор 10 кОм, нижний 4,7 кОм; на вход 5 В"):
        "V1 IN 0 DC 5\nR1 IN OUT 10k\nR2 OUT 0 4.7k\n",
    (1, "RC ФВЧ: конденсатор 100 нФ от входа к выходу, резистор 10 кОм с выхода на землю; сигнал на вход"):
        "V1 IN 0 DC 0\nC1 IN OUT 100n\nR1 OUT 0 10k\n",
    (1, "R1 10 кОм от входа к точке 1, C1 100 нФ от точки 1 на землю, R2 22 кОм от точки 1 к выходу, C2 10 нФ с выхода "
        "на землю; сигнал на вход"):
        "V1 IN 0 DC 0\nR1 IN N1 10k\nC1 N1 0 100n\nR2 N1 OUT 22k\nC2 OUT 0 10n\n",
    (1, "Инвертирующий усилитель на ОУ: входной резистор 4,7 кОм, резистор обратной связи 47 кОм; сигнал на вход"):
        "V1 IN 0 DC 0\nR1 IN N1 4.7k\nR2 N1 OUT 47k\nX1 0 N1 OUT OPAMP_IDEAL\n",
    (1, "Неинвертирующий усилитель на ОУ: резистор на землю 10 кОм, резистор обратной связи 22 кОм; сигнал на вход"):
        "V1 IN 0 DC 0\nR1 N1 0 10k\nR2 N1 OUT 22k\nX1 IN N1 OUT OPAMP_IDEAL\n",
    (1, "RC ФНЧ: резистор 1 кОм, конденсатор 100 нФ; после выхода ещё 470 Ом последовательно, выход за ним"):
        "V1 IN 0 DC 0\nR1 IN N1 1k\nC1 N1 0 100n\nR2 N1 OUT 470\n",
    (1, "Усилитель на транзисторе КТ315 с коэффициентом усиления 20"): ("REFUSE", "out_of_class"),
    (1, "Собери ФНЧ на 1 кГц"): ("REFUSE", "needs_design"),
    (2, "Нужен ФНЧ на 1 кГц"): ".SPEC type=LP fc=1k",
    (2, "Активный ФНЧ Баттерворта 4-го порядка на 2 кГц"): ".SPEC type=LP fc=2k order=4 response=Butterworth active=yes",
    (2, "Активный полосовой фильтр на 5 кГц с добротностью 5"): ".SPEC type=BP f0=5k Q=5 active=yes",
    (2, "ФНЧ Саллена-Ки на повторителе, срез 1 кГц; C1 (ОС) 22 нФ, C2 (земля) 10 нФ, резисторы подобрать"):
        "V1 IN 0 DC 0\nR1 IN N1 ?\nR2 N1 N2 ?\nC1 N1 OUT 22n\nC2 N2 0 10n\nX1 N2 OUT OUT OPAMP_IDEAL\n.SPEC type=LP fc=1k",
}
# Ответы режима 2 на запросы справочника (схему статьи режим 2 не задаёт: модель пишет цели, номиналы подбирает код)
HANDBOOK_MODE2 = {
    "rc_lp": ".SPEC type=LP fc=1.6k",
    "rc_hp": ".SPEC type=HP fc=1.6k",
    "rl_lp": "V1 IN 0 DC 0\nL1 IN OUT ?\nR1 OUT 0 1k\n.SPEC type=LP fc=16k",
    "rl_hp": "V1 IN 0 DC 0\nR1 IN OUT ?\nL1 OUT 0 10m\n.SPEC type=HP fc=16k",
    "twin_t": "V1 IN 0 DC 0\nR1 IN N1 ?\nR2 N1 OUT ?\nC3 N1 0 ?\nC1 IN N2 ?\nC2 N2 OUT ?\nR3 N2 0 ?\n.SPEC type=BS f0=1.6k",
    "rlc_series": ".SPEC type=BP f0=5k Q=3",
    "lc_ladder3": ".SPEC type=LP fc=10k order=3 response=Butterworth",
    "lc_ladder5": ".SPEC type=LP fc=10k order=5 response=Butterworth",
    "active_lp_inv": ".SPEC type=LP fc=1k active=yes K=10",
    "active_hp_inv": ".SPEC type=HP fc=1k active=yes K=10",
    "sallen_key_lp": ".SPEC type=LP fc=1k order=2 response=Butterworth active=yes",
    "sallen_key_hp": ".SPEC type=HP fc=1k order=2 response=Butterworth active=yes",
    "mfb_lp": ".SPEC type=LP fc=1k order=2 response=Butterworth active=yes K=2",
    "mfb_bp": ".SPEC type=BP f0=1k Q=5 active=yes",
    "mfb_hp": ".SPEC type=HP fc=1k order=2 response=Butterworth active=yes K=2",
    "twin_t_active": ".SPEC type=BS f0=1k Q=2 active=yes",
}


def program_prompts():
    """-> [(режим, запрос, источник)] - все запросы, которые программа показывает пользователю, по порядку."""
    from mcnl import gui
    from mcnl.handbook_articles import ARTICLES
    out = [(m, p, f"handbook:{a['id']}") for a in ARTICLES for m, p in a["prompts"]]
    out += [(1, p, "gui:examples") for _, p in gui.EXAMPLES]
    out += [(2, p, "gui:examples_mode2") for _, p in gui.EXAMPLES_MODE2]
    mode = 1
    for kind, body in gui.RULES:
        if kind == "h":
            mode = 2 if "Режим 2" in body else 1
        elif kind == "ex":
            out.append((mode, body, "gui:rules"))
    return out


def _mode1_record(rid, prompt, netlist, sources):
    elems = canon.parse_netlist(netlist)[0]
    from mcnl import app
    f_char = app.characteristic_frequency(elems)
    lo, hi = bench.ac_range(f_char)
    m = measure.measure(*mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE))
    quantities = ["type", "k_pass_db"] + ([k for k in ("fc", "f0", "Q") if m.get(k)] if m["type"] != "flat" else [])
    return {"id": rid, "split": "program", "mode": 1, "task": "A", "topology": "program", "family": "program",
            "kind": "program", "level": None, "prompt": prompt, "netlist": canon.canonical_netlist(netlist),
            "f_char": f_char, "measured": {k: v for k, v in m.items() if v is not None}, "quantities": quantities,
            "sources": sources}


def _mode2_record(rid, prompt, answer, sources):
    elems, unknown, spec, err = synth.parse(answer)
    if err:
        raise ValueError(f"эталон режима 2 не разбирается: {prompt}: {err}")
    kind = "spec" if not [e for e in elems if e["type"] != "V"] else "named"
    return {"id": rid, "split": "program", "mode": 2, "task": "B", "topology": kind, "family": "program", "kind": kind,
            "level": None, "prompt": prompt, "output": answer, "spec": spec, "unknown": unknown, "sources": sources}


def records():
    """Выборка program: запись на каждый уникальный (режим, запрос) с эталоном."""
    from mcnl.handbook_articles import ARTICLES
    by_id = {a["id"]: a for a in ARTICLES}
    seen, out = {}, []
    for mode, prompt, source in program_prompts():
        key = (mode, prompt)
        if key in seen:
            seen[key]["sources"].append(source)
            continue
        rid = f"program-{len(out) + 1:03d}"
        aid = source.split(":", 1)[1] if source.startswith("handbook:") else None
        if mode == 1:
            ref = by_id[aid]["netlist"] if aid else REFERENCES.get(key)
            if ref is None:
                raise KeyError(f"нет эталона для запроса режима 1: {prompt}")
            if isinstance(ref, tuple):
                rec = {"id": rid, "split": "program", "mode": 1, "task": "refuse", "topology": "refuse",
                       "family": "refuse", "kind": "refuse", "level": None, "prompt": prompt, "refuse": ref[1],
                       "output": f"REFUSE {ref[1]}: эталон", "sources": [source]}
            else:
                rec = _mode1_record(rid, prompt, ref, [source])
        else:
            ref = HANDBOOK_MODE2.get(aid) if aid else REFERENCES.get(key)
            if ref is None:
                raise KeyError(f"нет эталона для запроса режима 2: {prompt}")
            rec = _mode2_record(rid, prompt, ref, [source])
        seen[key] = rec
        out.append(rec)
    return out


def export(path=FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records())
    path.write_bytes(text.encode("utf-8"))
    return path


def read(path=FILE):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def passed(row):
    """Запись пройдена: режим 1 - S3 и S1 (отказ - верный код), режим 2 - поведение в MC12 (отказ - верный код)."""
    if row.get("refuse_ref"):
        return row.get("refuse_pred") == row["refuse_ref"]
    if row.get("mode") == 2:
        return bool(row.get("behavior"))
    return bool(row.get("S3") and row.get("S1"))


def check(preds, outdir=None, recs=None):
    """Ответы -> (строки оценки с полем passed, провалы). Оцениваются записи, на которые есть ответы (модель режима 1
    отвечает только на режим 1: predict.py --mode 1)."""
    recs = recs or read()
    ids = {p["id"] for p in preds}
    unknown = ids - {r["id"] for r in recs}
    if unknown:
        raise ValueError(f"ответы не из выборки program: {sorted(unknown)[:3]}")
    sub = [r for r in recs if r["id"] in ids]
    outdir = Path(outdir or OUT / "check")
    evaluate.evaluate_predictions(preds, sub, outdir)
    rows = score.read_jsonl(outdir / "scores.jsonl")
    meta = {r["id"]: r for r in sub}
    out = {p["id"]: p.get("output") for p in preds}
    for row in rows:
        row["passed"] = passed(row)
        row["prompt"], row["mode"] = meta[row["id"]]["prompt"], meta[row["id"]]["mode"]
        row["answer"] = out[row["id"]]
    return rows, [r for r in rows if not r["passed"]]


def report(rows, fails, name):
    by_mode = {m: [r for r in rows if r["mode"] == m] for m in (1, 2)}
    head = "; ".join(f"режим {m}: {sum(r['passed'] for r in rs)} из {len(rs)}" for m, rs in by_mode.items() if rs)
    lines = [f"{name}: {head} - " + ("100%, проверка пройдена" if not fails else f"провалов {len(fails)}")]
    for r in fails:
        why = r.get("error") or ("код отказа" if r.get("refuse_ref") else "")
        lines.append(f"  {r['id']} режим {r['mode']} [{why}] {r['prompt']}\n    ответ: {(r['answer'] or '').strip()!r}")
    return "\n".join(lines)


def local_answers(adapter, trained_on="v2", adapter_mode2=None, trained_on_mode2="v3", base="Qwen/Qwen3.5-4B"):
    """Ответы локальной модели (venv-hf) на все записи тех режимов, которые она умеет."""
    from mcnl import translators
    tr = translators.load("hf", base=base, adapter=adapter, trained_on=trained_on, adapter_mode2=adapter_mode2,
                          trained_on_mode2=trained_on_mode2)
    name = Path(adapter).parent.name if Path(adapter).name == "adapter" else tr.name  # out/<запуск>/adapter -> запуск
    recs, preds = read(), []
    for mode in tr.modes:
        sub = [r for r in recs if r["mode"] == mode]
        outs = tr.translate_batch([r["prompt"] for r in sub], mode)
        preds += [{"id": r["id"], "model": name, "format": "netlist", "output": o} for r, o in zip(sub, outs)]
    return name, preds


def main(argv):
    def opt(key, default=None):
        return argv[argv.index(key) + 1] if key in argv else default
    sys.stdout.reconfigure(encoding="utf-8")
    if "--export" in argv:
        print(f"-> {export()} ({len(read())} записей)")
        return 0
    files = []
    if "--local" in argv:
        name, preds = local_answers(opt("--adapter"), opt("--trained-on", "v2"), opt("--adapter-mode2"),
                                    opt("--trained-on-mode2", "v3"))
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{name}.jsonl"
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds), encoding="utf-8")
        files = [p]
    if "--check" in argv:
        files += [Path(a) for a in argv[argv.index("--check") + 1:] if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 2
    preds = [p for f in files for p in score.read_jsonl(f)]
    name = preds[0].get("model", "model") if preds else "model"
    rows, fails = check(preds, OUT / f"check_{name}")
    print(report(rows, fails, name))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
