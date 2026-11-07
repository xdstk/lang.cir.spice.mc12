#!/usr/bin/env python3
"""Программа: запрос на русском -> нетлист -> проверка и расчёт в MC12 -> схема .CIR -> MC12.

Запуск:  python src/mcnl/app.py "RC ФНЧ: R 1к, C 100н; сигнал на вх" [--model baseline] [--open]
         python src/mcnl/app.py --netlist схема.txt [--open]          (без переводчика)
         --out папка  (по умолчанию out/app; должна быть обычной папкой - MC12 не видит AppData пакета)

Переводчик - сменная «батарейка» (mcnl.translators): --model baseline | echo | модуль:Класс | файл.py:Класс.
Шаги: перевод -> разбор и канон (S0) -> оценка характерной частоты узловым расчётом (эталона у
пользователя нет) -> расчёт в MC12 через обвязку (S1) и измерения -> .CIR (spice2cir) -> открыть в MC12.
"""
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск как скрипта: python src/mcnl/app.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import bench, canon, mc12run, measure, mna, score, spice2cir, translators  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WIDE = (1.0, 1e8)  # широкая развёртка для оценки характерной частоты без эталона
# подсказка, а не обязательный формат: обученной модели единый разделитель не нужен, baseline - помогает
INPUT_HINT = ("Совет: назовите схему и перечислите элементы через «;» или с новой строки, указывая роль "
              "или выводы и номинал, например: «RC ФНЧ: R 1 кОм; C 100 нФ; сигнал на вх» или "
              "«R1 1к вх-вых; C1 100н вых-0; сигнал на вх».")


def characteristic_frequency(elems):
    """Оценка f_хар узловым расчётом на широкой сетке: fc (LP/HP), f0 (BP/BS) или None (flat)."""
    try:
        m = measure.measure(*mna.sweep(elems, *WIDE, per_decade=50))
    except ZeroDivisionError:
        return None
    return m.get("fc") or m.get("f0") or m.get("f_u")


def circuit_notes(elems):
    """Замечания к схеме: модель схему не чинит (docs/generator_v2_spec.md, раздел 2.3), программа предупреждает."""
    notes, use = [], {}
    for e in elems:
        for n in set(e["nodes"]):
            use.setdefault(n, []).append(e)
    for n, es in use.items():
        if n not in ("IN", "OUT", "0") and len(es) == 1:
            notes.append(f"{es[0]['name']}: второй вывод никуда не подключён")
    for e in elems:
        if e["type"] in "RCL" and set(e["nodes"]) == {"IN", "0"}:
            notes.append(f"{e['name']} стоит прямо на источнике и на выход не влияет")
    if "OUT" not in use:
        notes.append("в схеме нет узла выхода")
    elif len(use["OUT"]) == 1 and use["OUT"][0]["type"] in "RCL":
        notes.append(f"выход подключён только через {use['OUT'][0]['name']}: нагрузки нет")
    return notes


def run(prompt=None, translator=None, netlist=None, outdir=None, name="circuit", open_mc12=False, mc12=True):
    """-> отчёт {prompt, model, output, parse_errors, netlist, f_char, run, measured, cir, messages}."""
    outdir = Path(outdir or ROOT / "out" / "app")
    outdir.mkdir(parents=True, exist_ok=True)
    rep = {"prompt": prompt, "model": getattr(translator, "name", None), "messages": []}
    if netlist is None:
        out = translator.translate(prompt)
        fmt = getattr(translator, "output_format", "netlist")
    else:
        out, fmt = netlist, "netlist"
    rep["output"] = out
    refused = score.parse_refusal(out, fmt) if netlist is None else None
    if refused:
        rep["refused"] = {"code": refused[0], "reason": refused[1]}
        rep["messages"].append(f"схема не построена: {refused[1] or refused[0]}")
        return rep
    elems, errs = score.parse_output(out, fmt)
    rep["parse_errors"] = errs
    if elems is None:
        rep["messages"].append("ответ переводчика не разобран (S0): схема не построена")
        if netlist is None:
            rep["messages"].append(INPUT_HINT)
        return rep
    elems = canon.canonicalize(elems)
    rep["netlist"] = canon.to_netlist(elems)
    rep["notes"] = circuit_notes(elems)
    rep["f_char"] = characteristic_frequency(elems)
    if mc12:
        res = bench.run([(name, elems, rep["f_char"])], outdir)[name]
        rep["run"] = {"ok": res["ok"], "messages": res["messages"], "output": res["output"]}
        if res["ok"]:
            rep["measured"] = measure.from_output(res["output"])
            if prompt:
                note = name_mismatch(prompt, rep["measured"]["type"])
                if note:
                    rep["notes"].append(note)
        else:
            rep["messages"].append("MC12 не выполнил схему (S1)")
    cir = outdir / f"{name}.cir"
    try:
        spice2cir.write(cir, spice2cir.build(rep["netlist"], rep["f_char"], name))
        rep["cir"] = str(cir)
    except (ValueError, AssertionError) as err:
        rep["messages"].append(f".CIR не построен: {err}")
    if open_mc12 and rep.get("cir"):
        exe = mc12run.find_exe()
        subprocess.Popen([exe, rep["cir"]], cwd=str(Path(exe).parent))
        rep["messages"].append(f"схема открыта в MC12: {rep['cir']}")
    return rep


# Тип фильтра, названный в запросе (решение автора 06.11.2026, В5 в generator_v3_spec): схема строится по соединениям
# и номиналам, а программа предупреждает, если измеренный тип с названием расходится. Слова - по описаниям схем в
# roles_ru.json и запросах C01/llm_test; «интегратор», «усилитель», «контур» тип не задают и не учитываются.
NAMED_TYPES = {
    "LP": (r"\bфнч\b", r"нижних\s+частот", r"низких\s+частот", r"\bнч[- ]?фильтр", r"low[- ]?pass", r"\blpf\b"),
    "HP": (r"\bфвч\b", r"верхних\s+частот", r"высоких\s+частот", r"\bвч[- ]?фильтр", r"high[- ]?pass", r"\bhpf\b"),
    "BP": (r"полосов", r"полосно[- ]?пропуск", r"\bппф\b", r"band[- ]?pass", r"\bbpf\b"),
    "BS": (r"режектор", r"заграждающ", r"\bпзф\b", r"\bnotch\b", r"band[- ]?stop", r"двойн\w*\s+т\b"),
}
TYPE_WORD = {"LP": "ФНЧ", "HP": "ФВЧ", "BP": "полосовой фильтр", "BS": "режекторный фильтр"}


def named_type(prompt):
    """Тип фильтра, названный в запросе: LP / HP / BP / BS; None - не назван или названо несколько разных
    (каскад «ФВЧ -> ФНЧ» полосовой по смыслу, но не по названию)."""
    import re
    text = prompt.lower().replace("ё", "е")
    found = {t for t, pats in NAMED_TYPES.items() if any(re.search(p, text) for p in pats)}
    return found.pop() if len(found) == 1 else None


def name_mismatch(prompt, measured_type):
    """Замечание, если по запросу назван один тип фильтра, а MC12 насчитал другой; иначе None."""
    named = named_type(prompt)
    if named is None or measured_type == named:
        return None
    got = TYPE_WORD.get(measured_type, "схема без частотной зависимости")
    # добавки к схеме могут честно менять её тип (на эталонах v2 - 25 случаев из 903), поэтому не «ошибка», а проверка
    return (f"в запросе назван {TYPE_WORD[named]}, а по расчёту MC12 получился {got}: схема построена "
            "по описанным соединениям - проверьте их и добавленные элементы, если они есть")


def run_designer(prompt, translator, outdir=None, name="filter", open_mc12=False):
    """Режим дизайнера: запрос -> mcnl.flt/1 -> дизайнер MC12 -> .CIR (docs/designer_mode.md)."""
    from mcnl import designer
    outdir = Path(outdir or ROOT / "out" / "app")
    rep = {"prompt": prompt, "model": getattr(translator, "name", None), "messages": []}
    out = translator.translate(prompt)
    rep["output"] = out
    refused = score.parse_refusal(out, "ir")
    if refused:
        rep["refused"] = {"code": refused[0], "reason": refused[1]}
        rep["messages"].append(f"схема не построена: {refused[1] or refused[0]}")
        return rep
    spec, errs = designer.from_json(out)
    if errs:
        rep["messages"] += errs
        return rep
    rep["spec"] = spec
    rep.update(designer.run(spec, outdir))
    rep["messages"] += rep["notes"]
    cir = outdir / f"{name}.cir"
    spice2cir.write(cir, spice2cir.build(rep["netlist"], spec.get("fc") or spec.get("f0"), name))
    rep["cir"] = str(cir)
    if open_mc12:
        exe = mc12run.find_exe()
        subprocess.Popen([exe, str(cir)], cwd=str(Path(exe).parent))
    return rep


def main(argv):
    def opt(key, default=None):
        return argv[argv.index(key) + 1] if key in argv else default
    model = opt("--model", "baseline")
    netlist_file = opt("--netlist")
    skip = {argv.index(k) + i for k in ("--model", "--netlist", "--out", "--name") if k in argv for i in (0, 1)}
    words = [a for i, a in enumerate(argv[1:], 1) if i not in skip and not a.startswith("--")]
    prompt = " ".join(words) or None
    netlist = Path(netlist_file).read_text(encoding="utf-8") if netlist_file else None
    if prompt is None and netlist is None:
        sys.exit(__doc__)
    tr = None if netlist else translators.load(model)
    rep = run(prompt, tr, netlist, opt("--out"), opt("--name", "circuit"), "--open" in argv, "--no-mc12" not in argv)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
