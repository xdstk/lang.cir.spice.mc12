#!/usr/bin/env python3
"""LC-схемы без резистора для датасета v3.1 (docs/generator_v3_spec.md, «v3.1»).

В v2 и v3 таких схем нет: compose._reference отвергает их как «resonance» - без потерь высота пика и Q зависят от
сетки частот. На llm_test и llm_test2 они есть (LC ФНЧ и ФВЧ, LC-лестницы, П- и Т-звенья), и модель v3 записывала
в них конденсатор или катушку буквой R. Здесь они читаются как обычные схемы режима 1: эталон - нетлист, а в
поведение (S4) идут только величины, которые не зависят от сетки: тип АЧХ и f0 (проверка - lc_reference).

Формы (секции compose_v3.chain_netlist; выход - последний узел линии):
  g_lp  - L в линию, C на землю;          g_hp  - C в линию, L на землю;
  ladder_lp - L, C, L[, C[, L]] по очереди; ladder_hp - C, L, C[, L];
  pi_lp - C на входе, L в линию, C на выходе; t_lp - L, C на землю, L; t_hp - C, L на землю, C.
"""
import random
import re

try:
    from . import bench, canon, compose, compose_v3, describe_ru, filters_v2, measure, mna, sampler
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import compose
    import compose_v3
    import describe_ru
    import filters_v2
    import measure
    import mna
    import sampler

SHAPES = {  # форма: (секции без номиналов, тип по построению, вес)
    "g_lp": ((("ser", "L"), ("shunt", "C")), "LP", 3),
    "g_hp": ((("ser", "C"), ("shunt", "L")), "HP", 3),
    "ladder_lp": (None, "LP", 3),
    "ladder_hp": (None, "HP", 1),
    "pi_lp": ((("shunt", "C"), ("ser", "L"), ("shunt", "C")), "LP", 1),
    "t_lp": ((("ser", "L"), ("shunt", "C"), ("ser", "L")), "LP", 1),
    "t_hp": ((("ser", "C"), ("shunt", "L"), ("ser", "C")), "HP", 1),
}
NAMES = {
    "g_lp": ["LC ФНЧ", "LC-фильтр НЧ", "LC-НЧ", "Г-звено LC ФНЧ", "LC фильтр нижних частот"],
    "g_hp": ["LC ФВЧ", "LC-фильтр ВЧ", "LC-ВЧ", "Г-звено LC ФВЧ", "LC фильтр верхних частот"],
    "ladder_lp": ["LC-лестница", "LC лестница", "лестница LC", "лестничный LC ФНЧ"],
    "ladder_hp": ["LC-лестница ФВЧ", "верхнечастотная LC-лестница", "лестничный LC ФВЧ"],
    "pi_lp": ["П-звено LC", "LC П-звено", "П-образный LC ФНЧ"],
    "t_lp": ["Т-звено LC", "LC Т-звено", "Т-образный LC ФНЧ"],
    "t_hp": ["Т-звено LC ФВЧ", "Т-образный LC ФВЧ"],
}
GRID_F0 = 0.02          # f0 на двух сетках (сдвиг на полшага) - не больше 2%: иначе резонанс между точками
_LETTER = {"L": "L", "C": "C"}


def sections_of(shape, rng):
    """Секции формы со случайными номиналами (filters_v2.pick: ряд или точное значение)."""
    base, _t, _w = SHAPES[shape]
    if base is None:  # лестница: 3-5 элементов по очереди
        first, second = ("L", "C") if shape == "ladder_lp" else ("C", "L")
        n = rng.choice([3, 4, 4, 5]) if shape == "ladder_lp" else rng.choice([3, 4])
        base = tuple(("ser", first) if i % 2 == 0 else ("shunt", second) for i in range(n))
    return [(where, k, filters_v2.pick(k, rng)) for where, k in base]


def lc_reference(netlist, cls, params):
    """Нетлист -> эталон или (None, причина). Величины поведения - только устойчивые к сдвигу сетки."""
    elems, err = canon.parse_netlist(netlist)
    if err:
        return None, "parse"
    if not filters_v2.spread_ok(netlist):
        return None, "spread"
    elems = canon.canonicalize(elems)
    try:
        f_char = filters_v2.characteristic_frequency(elems)
        if f_char is None or not sampler.F_RANGE[0] <= f_char <= sampler.F_RANGE[1]:
            return None, "f_char"
        lo, hi = bench.ac_range(f_char)
        f, db, ph = mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE)
        m = measure.measure(f, db, ph)
        half = 10 ** (0.5 / bench.POINTS_PER_DECADE)
        f2, db2, ph2 = mna.sweep(elems, lo * half, hi * half, per_decade=bench.POINTS_PER_DECADE)
        m2 = measure.measure(f2, db2, ph2)
    except (ZeroDivisionError, ValueError, TypeError):
        return None, "solve"
    if m["type"] != m2["type"] or not m.get("f0") or not m2.get("f0") or abs(m["f0"] / m2["f0"] - 1) > GRID_F0:
        return None, "grid"
    q = ["type", "f0"]
    return {"id": cls, "class": cls, "level": "L1", "params": params, "netlist": canon.to_netlist(elems),
            "expected": {k: m[k] for k in q}, "quantities": q, "f_char": f_char}, None


def sample_lc(rng, shape=None, max_tries=500):
    """Эталон LC-схемы без резистора; params - форма и секции (для запроса)."""
    for _ in range(max_tries):
        sh = shape or rng.choices(list(SHAPES), [v[2] for v in SHAPES.values()])[0]
        secs = sections_of(sh, rng)
        ref, _why = lc_reference(compose_v3.chain_netlist(secs), f"lc:{sh}", {"shape": sh, "sections": secs})
        if ref:
            return ref
    raise RuntimeError("LC-схема не собрана")


# ---------------------------------------------------------------- запросы
def describe_terse(ref, rng):
    """Название и номиналы по пути сигнала: «LC-лестница 22мкГн 270п 82мкГн 100п», иногда с буквами элементов."""
    sh, secs = ref["params"]["shape"], ref["params"]["sections"]
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.3
    letters = rng.random() < 0.4
    vals = [(f"{_LETTER[k]} " if letters else "") + b.value(k, compose._fmt(v)) for _w, k, v in secs]
    name = rng.choice(NAMES[sh])
    return " ".join([name if rng.random() < 0.6 else name.lower()] + vals)


def describe_named_chain(ref, rng):
    """Название и описание по цепочке: «LC ФВЧ: 680п в линию, 8,2мГн на землю; выход в конце цепочки»."""
    sh = ref["params"]["shape"]
    d = compose_v3.describe_chain(ref["params"]["sections"], rng, reorder=False)
    name = rng.choice(NAMES[sh])
    return f"{name}{rng.choice([': ', ' - ', ', '])}{d['prompt']}"


def describe_words(ref, rng):
    """Словесное описание по порядку: «катушка 27 мкГн в линию, затем конденсатор 390 пФ на землю, выход с конденсатора»."""
    secs = ref["params"]["sections"]
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.7
    gnd, ser = rng.choice(compose.SINK), rng.choice(["последовательно", "в линию", "в разрыв"])
    noun = {"L": "катушка", "C": "конденсатор"}
    parts = []
    for i, (where, k, v) in enumerate(secs):
        head = noun[k] if rng.random() < 0.7 else ("ёмкость" if k == "C" else "индуктивность")
        where_text = ("на входе " + gnd if i == 0 else gnd) if where == "shunt" else ser
        parts.append(f"{head} {b.value(k, compose._fmt(v))} {where_text}")
    last = secs[-1]
    end = rng.choice([f"выход с {'конденсатора' if last[1] == 'C' else 'катушки'}" if last[0] == "shunt" else
                      "выход после последнего элемента", "выход в конце цепочки", "выход с последнего узла"])
    name = rng.choice(NAMES[ref["params"]["shape"]]) + ": " if rng.random() < 0.5 else ""
    text = rng.choice([", ", ", затем ", "; "]).join(parts)
    return f"{name}{text[:1].upper() + text[1:] if not name else text}, {end}"


def parse_words(prompt):
    """Обратный разбор describe_words и describe_named_chain по порядку -> нетлист (проверка запроса)."""
    body = prompt.split(": ", 1)[1] if ": " in prompt.split(",")[0] else prompt
    body = re.split(r"[;,]?\s*выход ", body)[0]
    secs = []
    for seg in re.split(compose_v3._SPLIT, body):
        vals = compose_v3._values(seg)
        if not vals:
            continue
        if len(vals) != 1:
            raise ValueError(f"не один номинал: {seg}")
        k, v = vals[0]
        secs.append(("shunt" if re.search(compose._GND_RE, seg) else "ser", k, v))
    return compose_v3.chain_netlist(secs)


STYLES = {"terse": 0.35, "named_chain": 0.25, "words": 0.25, "chain": 0.15}


def make_prompt(ref, rng):
    """-> (task, family, запрос). Запрос проверяется обратным разбором, кроме краткой записи (номиналы по порядку)."""
    style = rng.choices(list(STYLES), list(STYLES.values()))[0]
    if style == "terse":
        return "A1", "terse", describe_terse(ref, rng)
    if style == "chain":
        secs = ref["params"]["sections"]  # ветвь до первого элемента линии (П-звено) описывается только по порядку
        text = compose_v3.describe_chain(secs, rng, reorder=False if secs[0][0] == "shunt" else None)["prompt"]
        net = compose_v3.parse_chain(text)
        family, task = "chain", "A"
    else:
        text = describe_named_chain(ref, rng) if style == "named_chain" else describe_words(ref, rng)
        net = parse_words(text)
        family, task = "lc_" + style, "A1" if style == "named_chain" or ": " in text.split(",")[0] else "A"
    target = canon.canonicalize(canon.parse_netlist(ref["netlist"])[0])
    if canon.to_netlist(canon.canonicalize(canon.parse_netlist(net)[0])) != canon.to_netlist(target):
        raise AssertionError(f"запрос не разбирается обратно: {text}")
    return task, family, text


if __name__ == "__main__":
    import sys
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    for sh in SHAPES:
        ref = sample_lc(rng, sh)
        for _ in range(2):
            print(sh, "|", make_prompt(ref, rng)[2], "=>", ref["netlist"].replace("\n", " | "), ref["expected"])
