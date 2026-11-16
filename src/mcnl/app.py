#!/usr/bin/env python3
"""Программа: запрос на русском -> нетлист -> проверка и расчёт в MC12 -> схема .CIR -> MC12.

Запуск:  python src/mcnl/app.py "RC ФНЧ: R 1к, C 100н; сигнал на вх" [--model baseline] [--open]
         python src/mcnl/app.py --netlist схема.txt [--open]          (без переводчика)
         --out папка  (по умолчанию out/app; должна быть обычной папкой - MC12 не видит AppData пакета)
         --opamp TL071C  (реальный ОУ с питанием на чертеже; по умолчанию - идеальный)
         --mode 2        (подбор номиналов; с --netlist - файл с `?` и `.SPEC`, как ответ модели)

Переводчик - сменная «батарейка» (mcnl.translators): --model baseline | echo | модуль:Класс | файл.py:Класс.
Шаги: перевод -> разбор и канон (S0) -> оценка характерной частоты узловым расчётом (эталона у
пользователя нет) -> расчёт в MC12 через обвязку (S1) и измерения -> .CIR (spice2cir) -> открыть в MC12.
--mode 2: подбор номиналов (docs/mode2_spec.md): ответ модели - нетлист с `?` и строкой `.SPEC` или одна `.SPEC`;
номиналы считает synth.run, дальше подобранная схема идёт тем же путём, что в режиме 1.
--opamp TL071C: на чертеже реальный ОУ с питанием (решение автора 16.11.2026, docs/gui_spec.md); тот же запуск MC12
считает и обвязку с идеальным ОУ, и сам чертёж. Если реальный ОУ насытился бы по постоянному току (mna.saturation),
чертёж остаётся с идеальным ОУ, а замечание объясняет почему.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # запуск как скрипта: python src/mcnl/app.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import bench, canon, mc12run, measure, mna, score, spice2cir, synth, translators  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OPAMP_REAL = "TL071C"  # реальный ОУ на чертеже (выбор автора 16.11.2026): модель OPA уровня 3 из tiopamp.lbr MC12
WIDE = (1.0, 1e8)  # широкая развёртка для оценки характерной частоты без эталона
WIDE_PER_DECADE = 50
# подсказка, а не обязательный формат: обученной модели единый разделитель не нужен, baseline - помогает
INPUT_HINT = ("Совет: назовите схему и перечислите элементы через «;» или с новой строки, указывая роль "
              "или выводы и номинал, например: «RC ФНЧ: R 1 кОм; C 100 нФ; сигнал на вх» или "
              "«R1 1к вх-вых; C1 100н вых-0; сигнал на вх».")


def characteristic_frequency(elems):
    """Оценка f_хар узловым расчётом на широкой сетке: fc (LP/HP), f0 (BP/BS) или None (flat)."""
    try:
        f, db, ph = mna.sweep(elems, *WIDE, per_decade=WIDE_PER_DECADE)
    except ZeroDivisionError:
        return None
    m = measure.measure(f, db, ph)
    edge = {"LP": db[0] - db[WIDE_PER_DECADE], "HP": db[-1] - db[-1 - WIDE_PER_DECADE]}.get(m["type"], 0)
    if m.get("f_u") and abs(edge) > 3:
        # на краю сетки, где должна быть полоса пропускания, АЧХ ещё не ровная (дифференциатор, интегратор): полосы
        # нет, fc - край сетки; рабочая точка - частота единичного усиления, как в датасете (filters_v2)
        return m["f_u"]
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


def saturation_note(opamp, sat):
    """Замечание: почему на чертеже идеальный ОУ, хотя выбран реальный (mna.saturation)."""
    name, value, why = sat[0]
    head = f"на чертеже идеальный ОУ: реальный {opamp} здесь "
    if why == "source":
        return (head + f"насытился бы - выход {name} по постоянному току около {value:+.0f} В при питании "
                f"±{spice2cir.SUPPLY_V} В (усилено постоянное напряжение источника); уменьшите DC источника или усиление")
    if why == "current":
        return (head + f"упёрся бы в ограничение тока - выход {name} отдаёт по постоянному току около "
                f"{value * 1e3:.0f} мА; увеличьте сопротивления")
    return (head + f"насытился бы - у {name} нет обратной связи по постоянному току, и смещение нуля уводит выход "
            "за питание (как у интегратора без резистора параллельно конденсатору)")


def card_band(m):
    """Верхняя частота для карточек по чертежу с реальным ОУ: 10 x верхний край полосы схемы с идеальным ОУ (fc, у
    полосовых и режекторных - f2). Выше у ФВЧ сказывается полоса ОУ, и без границы ФВЧ мерился бы по уровню на краю
    диапазона; саму кривую окно рисует целиком. У схем без частотной зависимости - весь диапазон."""
    if not m:
        return None
    if m["type"] in ("LP", "HP") and m.get("fc"):
        return 10 * m["fc"]
    if m["type"] in ("BP", "BS") and m.get("f0") and m.get("Q"):
        q = m["Q"]
        return 10 * m["f0"] * (math.sqrt(1 + 1 / (4 * q * q)) + 1 / (2 * q))
    return None


KIND_WORD = {"LP": "ФНЧ", "HP": "ФВЧ", "BP": "полосовой", "BS": "режекторный", "flat": "без частотной зависимости"}

# Режим 2: схемы, которые строит synth (docs/mode2_spec.md, разделы 3-4), - короткие названия для замечаний
TOPOLOGY_RU = {
    "rc_lp": "RC ФНЧ", "rc_hp": "RC ФВЧ", "rl_lp": "RL ФНЧ", "rl_hp": "RL ФВЧ",
    "rlc_series_lp": "последовательный RLC, ФНЧ", "rlc_series_hp": "последовательный RLC, ФВЧ",
    "rlc_series_bp": "последовательный RLC, полосовой", "rlc_series_bs": "последовательный RLC, режекторный",
    "active_lp_inv": "инвертирующий активный ФНЧ", "active_hp_inv": "инвертирующий активный ФВЧ",
    "sallen_key_lp": "ФНЧ Саллена-Ки", "sallen_key_hp": "ФВЧ Саллена-Ки", "mfb_lp": "ФНЧ с МОС (Рауха)",
    "mfb_hp": "ФВЧ с МОС", "mfb_bp": "полосовой с МОС", "twin_t": "двойной Т-мост",
    "twin_t_active": "активный двойной Т-мост", "lc_ladder3": "LC-лестница 3-го порядка",
    "lc_ladder5": "LC-лестница 5-го порядка",
}
MODE2_HINT = ("Режим 2: напишите тип фильтра и частоту («ФНЧ на 1 кГц», «активный полосовой на 5 кГц, Q = 5»), "
              "при желании - схему и часть номиналов («Саллен-Ки, конденсаторы 10 нФ»); остальное подберёт программа.")


def is_mode2_answer(text):
    """Ответ режима 2: есть неизвестный номинал `?` или строка `.SPEC`. Нетлист без них - обычная схема (режим 1)."""
    return any(l.strip().upper().startswith(".SPEC") or l.strip().endswith("?") for l in text.splitlines())


def _hz(f):
    for scale, unit in ((1e6, "МГц"), (1e3, "кГц"), (1, "Гц")):
        if abs(f) >= scale or scale == 1:
            return f"{f / scale:.3g} {unit}".replace(".", ",")


def _value(key, v):
    if key in ("fc", "f0"):
        return _hz(v)
    if key == "k_pass_db":
        return f"{round(v, 1) + 0.0:+.1f} дБ".replace(".", ",")
    return f"{v:.3g}".replace(".", ",")


def synth_notes(s, answer, measured=None):
    """Замечания режима 2: какая схема, какие номиналы подобраны, что задано и что получилось (MC12, если посчитан,
    иначе узловой расчёт synth), замечания synth."""
    lines = []
    _, unknown, _, _ = synth.parse(answer)
    top = s.get("topology") or ""
    name = " + ".join(TOPOLOGY_RU.get(t, t) for t in top.split("+")) if top else "схема из запроса"
    exact = s["spec"].get("series") == "EXACT"
    rows = "точные значения" if exact else "R - ряд E24, C и L - E12"
    if unknown:
        vals = {e["name"]: e["value"] for e in canon.parse_netlist(s["netlist"])[0]}
        picked = ", ".join(f"{n} = {vals[n]}" for n in unknown if n in vals)
        lines.append(f"подбор номиналов ({name}; {rows}): {picked}")
    else:  # схему по характеристикам выбрал код: показать её целиком (без источника)
        body = "; ".join(l for l in s["netlist"].splitlines() if l.strip() and not l.startswith("V"))
        lines.append(f"схему выбрала программа - {name}, номиналы подобраны ({rows}): {body}")
    got = measured or s.get("achieved") or {}
    where = "MC12" if measured else "узловой расчёт"
    parts = []
    for k, label in (("fc", "fc"), ("f0", "f0"), ("Q", "Q"), ("k_pass_db", "K")):
        want = s.get("wanted", {}).get(k)
        if want is None:
            continue
        have = got.get(k)
        if have is None:
            parts.append(f"{label} {_value(k, want)} - не измерено")
            continue
        dev = (f"{round(have - want, 1) + 0.0:+.1f} дБ" if k == "k_pass_db"
               else f"{round((have / want - 1) * 100, 1) + 0.0:+.1f}%")
        parts.append(f"{label} {_value(k, want)} -> {_value(k, have)} ({dev.replace('.', ',')})")
    if parts:
        lines.append(f"задано -> получено ({where}): " + "; ".join(parts))
    return lines + list(s.get("notes", []))


def run(prompt=None, translator=None, netlist=None, outdir=None, name="circuit", open_mc12=False, mc12=True,
        opamp=None, mode=1):
    """-> отчёт {prompt, model, output, parse_errors, netlist, f_char, run, measured, cir, messages}.
    mode=2 - подбор номиналов: переводчик отвечает в режиме 2 (нетлист с `?` и `.SPEC`), synth.run подбирает номиналы,
    отчёт synth - в synth, output - ответ модели, netlist - подобранная схема. С netlist=... текст с `?` или `.SPEC`
    тоже идёт через synth (пересчёт правленого ответа), без них - обычная схема.
    opamp - модель реального ОУ для чертежа (OPAMP_REAL) или None (идеальный). С реальным ОУ: opamp - модель на
    чертеже (None, если пришлось оставить идеальный), run_real и measured - расчёт самого чертежа,
    measured_ideal - обвязки с идеальным ОУ; run - всегда обвязка (S1)."""
    outdir = Path(outdir or ROOT / "out" / "app")
    outdir.mkdir(parents=True, exist_ok=True)
    rep = {"prompt": prompt, "model": getattr(translator, "name", None), "messages": [], "mode": mode}
    if netlist is None:
        if mode == 2 and 2 not in getattr(translator, "modes", (1,)):
            rep["messages"].append(f"переводчик {rep['model']} не умеет режим 2 (подбор номиналов)")
            return rep
        out = translator.translate(prompt, mode=2) if mode == 2 else translator.translate(prompt)
        fmt = getattr(translator, "output_format", "netlist")
    else:
        out, fmt = netlist, "netlist"
    rep["output"] = out
    refused = score.parse_refusal(out, fmt) if netlist is None else None
    if refused:
        rep["refused"] = {"code": refused[0], "reason": refused[1]}
        rep["messages"].append(f"схема не построена: {refused[1] or refused[0]}")
        if refused[0] == "needs_design" and mode == 1:
            rep["messages"].append("номиналы не заданы - это задача режима 2 (подбор номиналов)")
        return rep
    if mode == 2 and fmt == "netlist" and is_mode2_answer(out):
        s = rep["synth"] = synth.run(out)
        if "error" in s:
            rep["messages"].append(f"номиналы не подобраны: {s['error']}")
            if netlist is None:
                rep["messages"].append(MODE2_HINT)
            return rep
        if s.get("route") == "designer":
            rep["messages"].append("полосовой и режекторный фильтр выше 2-го порядка программа сама не считает - это "
                                   "работа дизайнера MC12, а он в программу пока не подключён; задайте порядок 2")
            return rep
        out = s["netlist"]
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
    real = opamp if opamp and any(e["type"] == "OPAMP" for e in elems) else None
    if real:
        sat = mna.saturation(elems)
        if sat:
            rep["notes"].append(saturation_note(real, sat))
            real = None
    cir = outdir / f"{name}.cir"
    try:
        try:
            text = spice2cir.build(rep["netlist"], rep["f_char"], name, opamp=real)
        except (ValueError, AssertionError) as err:
            if not real:
                raise
            rep["messages"].append(f"чертёж с реальным ОУ не построен ({err}): на чертеже идеальный ОУ")
            real = None
            text = spice2cir.build(rep["netlist"], rep["f_char"], name)
        spice2cir.write(cir, text)
        rep["cir"] = str(cir)
    except (ValueError, AssertionError) as err:
        rep["messages"].append(f".CIR не построен: {err}")
    rep["opamp"] = real
    if mc12:
        outdir.mkdir(parents=True, exist_ok=True)
        ckt = outdir / f"{name}.ckt"
        bench.write(ckt, bench.build(elems, rep["f_char"], title=name))
        paths = [ckt]
        if real and rep.get("cir"):  # копия чертежа с числовым выводом: файл пользователя остаётся чистым
            calc = outdir / f"{name}_real.cir"
            spice2cir.write(calc, spice2cir.build(rep["netlist"], rep["f_char"], name, opamp=real, numeric=True))
            paths.append(calc)
        runs = mc12run.run(paths, analyses=["A"], timeout=300)
        res = runs[0]
        rep["run"] = {"ok": res["ok"], "messages": res["messages"], "output": res["output"]}
        if res["ok"]:
            rep["measured"] = measure.from_output(res["output"])
            if prompt:  # название сверяется с идеальной схемой: у реального ОУ ФВЧ выше его полосы - полосовой
                note = name_mismatch(prompt, rep["measured"]["type"])
                if note:
                    rep["notes"].append(note)
        else:
            rep["messages"].append("MC12 не выполнил схему (S1)")
        if len(runs) > 1:
            rr = runs[1]
            rep["run_real"] = {"ok": rr["ok"], "messages": rr["messages"], "output": rr["output"]}
            if rr["ok"]:
                ideal = rep["measured_ideal"] = rep.get("measured")
                rep["measured"] = measure.from_output(rr["output"], card_band(ideal))
                if ideal and rep["measured"]["type"] != ideal["type"]:
                    rep["notes"].append(
                        f"с {real} АЧХ вида «{KIND_WORD[rep['measured']['type']]}», с идеальным ОУ - "
                        f"«{KIND_WORD[ideal['type']]}»: сказывается конечная полоса и выходное сопротивление ОУ "
                        "(на графике - сплошная и пунктирная кривые)")
            else:
                rep["messages"].append(f"MC12 не посчитал чертёж с {real}: " + "; ".join(rr["messages"]))
    if rep.get("synth"):  # отклонения - по схеме с идеальным ОУ: под неё считались номиналы
        rep["notes"] = synth_notes(rep["synth"], rep["output"], rep.get("measured_ideal") or rep.get("measured")) \
            + rep["notes"]
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
    skip = {argv.index(k) + i for k in ("--model", "--netlist", "--out", "--name", "--opamp", "--mode") if k in argv
            for i in (0, 1)}
    words = [a for i, a in enumerate(argv[1:], 1) if i not in skip and not a.startswith("--")]
    prompt = " ".join(words) or None
    netlist = Path(netlist_file).read_text(encoding="utf-8") if netlist_file else None
    if prompt is None and netlist is None:
        sys.exit(__doc__)
    tr = None if netlist else translators.load(model)
    rep = run(prompt, tr, netlist, opt("--out"), opt("--name", "circuit"), "--open" in argv, "--no-mc12" not in argv,
              opt("--opamp"), int(opt("--mode", "1")))
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
