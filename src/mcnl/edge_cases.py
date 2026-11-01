#!/usr/bin/env python3
"""Странные схемы и отказы датасета v2 (docs/generator_v2_spec.md, разделы 2.3-2.4).

Странная схема описана однозначно, поэтому эталон дословный: модель схему не чинит, замечания выдаёт
проверка программы. Отказ - запрос вне класса, неоднозначный или спецификация без номиналов; эталон - ответ
REFUSE. Шаблоны отказов составлены по docs/scope.md, а не по запросам llm_test.
"""
import random

try:
    from . import canon, compose, describe_ru, filters_v2, prompts_v2, score
except ImportError:  # запуск как скрипта
    import canon
    import compose
    import describe_ru
    import filters_v2
    import prompts_v2
    import score

STRANGE = ("dangling", "source_shunt", "unreal_value", "open_output")
UNREAL = {"L": [5e-9, 10.0, 100.0], "R": [0.1, 1e9], "C": [0.5e-12, 10e-3]}


def _chain_lines(rng):
    ref = compose.sample_chain(rng)
    return ref, [l for l in ref["netlist"].splitlines() if l and not l.startswith("V")]


def strange(kind, rng, max_tries=500):
    """Странная схема -> эталон compose._reference с class 'strange:<вид>'."""
    for _ in range(max_tries):
        base, lines = _chain_lines(rng)
        src = "V1 IN 0 DC 0\n"
        if kind == "dangling":  # элемент с одним подключённым выводом
            k = rng.choice("RC")
            compose._element(lines, k, rng.choice(["OUT", "IN"]), compose._next(lines, "D"), compose._pick(k, rng))
        elif kind == "source_shunt":  # на идеальном источнике элемент ни на что не влияет
            k = rng.choice("RCL")
            compose._element(lines, k, "IN", "0", compose._pick(k, rng))
        elif kind == "open_output":  # выход через один элемент, без нагрузки
            node = compose._next(lines, "A")
            lines = compose._rename(lines, "OUT", node)
            k = rng.choice("RC")
            compose._element(lines, k, node, "OUT", compose._pick(k, rng))
        elif kind == "unreal_value":
            i = rng.randrange(len(lines))
            t = lines[i].split()
            t[3] = canon.format_value(rng.choice(UNREAL[t[0][0]]))
            lines[i] = " ".join(t)
        else:
            raise ValueError(kind)
        ref, _why = compose._reference(src + "\n".join(lines) + "\n", f"strange:{kind}", {"kind": kind},
                                       unusual=kind == "unreal_value")
        if ref:
            return ref
    raise RuntimeError(f"{kind}: схема не собрана")


def strange_prompt(ref, rng):
    """Запрос к странной схеме: обычные семейства задачи A, схема описана полностью."""
    return prompts_v2.describe_freeform(ref["netlist"], rng)


# ---------------------------------------------------------------- отказы
OUT_OF_CLASS = (
    ("усилитель с общим эмиттером на КТ3102: Rк {R}, Rэ {R}", "транзистор вне класса схем"),
    ("каскад на полевом транзисторе, Rс {R}, Rи {R}", "транзистор вне класса схем"),
    ("mosfet с резистором {R} в затворе", "транзистор вне класса схем"),
    ("мост из четырёх диодов, на выходе {C} и нагрузка {R}", "диод вне класса схем"),
    ("детектор на диоде: {C} и {R} на землю", "диод вне класса схем"),
    ("сумматор на ОУ: два входа через {R} и {R}, Rос {R}", "у схемы два входа"),
    ("вычитатель на ОУ, все резисторы {R}", "у схемы два входа"),
    ("смеситель двух сигналов, на каждом входе {R}", "у схемы два входа"),
    ("генератор на NE555, {R} и {C}", "микросхема таймера вне класса схем"),
    ("счётчик на D-триггерах", "цифровая схема вне класса"),
    ("логический элемент И-НЕ с резистором {R} на выходе", "цифровая схема вне класса"),
    ("трансформатор 220 на 12 В, нагрузка {R}", "трансформатор вне класса схем"),
)
NEEDS_DESIGN = (
    "фильтр нижних частот Баттерворта, срез {f}, порядок {n}",
    "фвч чебышева на {f} порядка {n}",
    "рассчитай полосовой на {f} с добротностью {q}",
    "подбери номиналы ФНЧ на {f}",
    "режекторный на {f}, схему и номиналы выбери сам",
    "нужен Бессель {n} порядка, срез {f}",
)
REASON_DESIGN = "номиналы не заданы, расчёт - режим дизайнера"


def refusal_output(code, reason, fmt="netlist"):
    """Эталонный ответ-отказ в формате выхода модели."""
    if fmt == "ir":
        import json
        return json.dumps({"refuse": code, "reason": reason}, ensure_ascii=False)
    return f"REFUSE {code}: {reason}"


def _freq(rng):
    v = rng.choice([50, 100, 300, 500, 1000, 2000, 5000, 10000, 20000])
    return f"{v} Гц" if v < 1000 else f"{v // 1000} кГц"


def _filled(template, rng):
    b = describe_ru.Brief(rng)
    out = template
    while "{R}" in out:
        out = out.replace("{R}", b.value("R", canon.format_value(compose._pick("R", rng))), 1)
    while "{C}" in out:
        out = out.replace("{C}", b.value("C", canon.format_value(compose._pick("C", rng))), 1)
    return out.replace("{f}", _freq(rng)).replace("{n}", str(rng.randint(2, 6))).replace("{q}", str(rng.choice([2, 5, 10])))


def _ambiguous(rng):
    """Порча описания цепочки: подключение или номинал пропали, номиналы противоречат друг другу."""
    ref = compose.sample_chain(rng)
    secs = ref["params"]["sections"]
    parts = compose.describe_chain(secs, rng).split("; выход")[0]
    b = describe_ru.Brief(rng)
    k = rng.choice("RC")
    word = {"R": "резистор", "C": "конденсатор"}[k]
    val = b.value(k, canon.format_value(compose._pick(k, rng)))
    other = b.value(k, canon.format_value(compose._pick(k, rng)))
    kind = rng.choice(["no_link", "no_value", "conflict"])
    if kind == "no_link":
        return f"{parts}; ещё {word} {val}", f"не сказано, куда подключён {word} {val}"
    if kind == "no_value":
        return f"{parts}; {word} на выходе на землю, номинал пока не знаю", f"не указан номинал: {word}"
    return f"{parts}; последний {word} {val}, нет, лучше {other} или {val}", f"два значения для элемента: {word}"


def refusal(code, rng):
    """-> {'prompt', 'refuse', 'reason', 'output'}."""
    if code == "out_of_class":
        tpl, reason = rng.choice(OUT_OF_CLASS)
        prompt = _filled(tpl, rng)
    elif code == "needs_design":
        prompt, reason = _filled(rng.choice(NEEDS_DESIGN), rng), REASON_DESIGN
    elif code == "ambiguous":
        prompt, reason = _ambiguous(rng)
    else:
        raise ValueError(code)
    return {"prompt": prompt, "refuse": code, "reason": reason, "output": refusal_output(code, reason)}


if __name__ == "__main__":
    rng = random.Random(1)
    for c in score.REFUSE_CODES:
        print(refusal(c, rng))
    for k in STRANGE:
        r = strange(k, rng)
        print(k, strange_prompt(r, rng))
