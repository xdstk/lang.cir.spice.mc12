#!/usr/bin/env python3
"""Запросы датасета v2 сверх семейств v1 (docs/generator_v2_spec.md, раздел 4).

freeform - схема словами на уровне соединений, с обратным разбором parse_freeform.
terse    - название и номиналы в порядке ролей («фнч 1к 100н»).
mutate   - правки текста поверх готового запроса: раскладка, запись чисел, слова земли, опечатка в слове,
           лишние фразы. Номиналы правка не меняет (проверяется в тестах), обратный разбор после неё не требуется.
"""
import re

try:
    from . import canon, compose, describe_ru
except ImportError:  # запуск как скрипта
    import canon
    import compose
    import describe_ru

ANCHORS = {"IN": ("вход", "входа", "входу", "входом"), "OUT": ("выход", "выхода", "выходу", "выходом"),
           "0": ("земля", "земли", "земле", "землёй")}
GND_WORDS = {"земля": ["общий", "общий провод", "корпус", "GND", "минус"], "землю": ["общий", "корпус", "GND"],
             "земле": ["общему", "корпусу"]}
LETTERS = (["A", "B", "C", "D", "E", "F", "G", "H"], ["X", "Y", "Z", "U", "W", "P", "Q", "S"],
           ["т1", "т2", "т3", "т4", "т5", "т6", "т7", "т8"])
PREFIXES = ["нарисуй ", "нужна схема: ", "собери ", "сделай ", "накидай ", "схема: ", "надо: "]
SUFFIXES = [" подпиши узлы", " вход слева, выход справа", " питание ±12 В", " без питания", " земля внизу"]
_NODE = r"(вход\w*|выход\w*|земл\w*|узл\w*\s+\S+|узел\s+\S+|точк\w+\s+\S+)"


# ---------------------------------------------------------------- freeform
def describe_freeform(netlist, rng):
    """Соединения словами: каждая деталь - откуда и куда. Узлы - буквами, обратный разбор возможен всегда."""
    elems, err = canon.parse_netlist(netlist)
    if err:
        raise ValueError(err)
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.6
    inner = sorted({n for e in elems for n in e["nodes"] if n not in ANCHORS})
    letters = rng.choice(LETTERS)
    if len(inner) > len(letters):
        letters = [f"т{i}" for i in range(1, len(inner) + 1)]
    label = dict(zip(inner, letters))
    word = rng.choice(["узел", "точка"])
    form = {"узел": ("узел {}", "узла {}", "узлу {}", "узлом {}"), "точка": ("точка {}", "точки {}", "точке {}", "точкой {}")}[word]

    def node(n, i):
        return ANCHORS[n][i] if n in ANCHORS else form[i].format(label[n])

    two = [e for e in elems if e["type"] in "RCL"]
    if rng.random() < 0.3:
        rng.shuffle(two)
    kind = rng.choice(["from_to", "between", "colon"])
    parts = []
    for e in two:
        a, c = e["nodes"]
        v = b.value(e["type"], e["value"])
        parts.append({"from_to": f"{v} от {node(a, 1)} к {node(c, 2)}", "between": f"{v} между {node(a, 3)} и {node(c, 3)}",
                      "colon": f"{v}: {node(a, 0)} - {node(c, 0)}"}[kind])
    for e in elems:
        if e["type"] == "OPAMP":
            p, m, o = e["nodes"]
            parts.append(f"ОУ: плюс - {node(p, 0)}, минус - {node(m, 0)}, выход ОУ - {node(o, 0)}")
    src = next(e for e in elems if e["type"] == "V")
    return "; ".join(parts) + "; " + b.source(src["value"])


def parse_freeform(prompt):
    """Обратный разбор запроса freeform -> элементы (canon.parse_netlist)."""
    def node(s):
        s = s.lower()
        if s.startswith("вход"):
            return "IN"
        if s.startswith("выход"):
            return "OUT"
        if s.startswith("земл"):
            return "0"
        return "N" + s.split()[-1].strip(",.").upper()

    lines, count, source = [], {}, "0"
    for seg in prompt.split(";"):
        seg = seg.strip()
        if seg.startswith("ОУ:"):
            nodes = re.findall(r"(?:плюс|минус|выход ОУ) - (" + _NODE[1:-1] + ")", seg)
            count["X"] = count.get("X", 0) + 1
            lines.append(f"X{count['X']} {node(nodes[0])} {node(nodes[1])} {node(nodes[2])} OPAMP_IDEAL")
            continue
        vals = list(compose._VALUE_RE.finditer(seg))
        nodes = re.findall(_NODE, seg[vals[0].end():], re.I) if vals else []
        if len(nodes) >= 2:
            k, mult = compose._UNITS[vals[0].group(2) or ""]
            count[k] = count.get(k, 0) + 1
            lines.append(f"{k}{count[k]} {node(nodes[0])} {node(nodes[1])} "
                         f"{canon.format_value(float(vals[0].group(1).replace(',', '.')) * mult)}")
            continue
        m = re.search(r"(-?\d+(?:[.,]\d+)?) В(?![а-яё])", seg)
        if m:
            source = canon.format_value(float(m.group(1).replace(",", ".")))
    return canon.parse_netlist("\n".join([f"V1 IN 0 DC {source}"] + lines) + "\n")[0]


# ---------------------------------------------------------------- terse
def describe_terse(ref, rng):
    """Название и номиналы в порядке ролей, без подписей ролей."""
    r = describe_ru.ROLES_ALL[ref["class"]]
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.3
    name = rng.choice(r["short_names"] + r.get("aliases", {}).get("names", [])[:3])
    vals = [b.value(k[0], canon.format_value(ref["params"][k])) for k in r["roles"] if k in ref["params"]]
    text = " ".join([name.lower() if rng.random() < 0.7 else name] + vals)
    dc = ref["params"].get("dc", 0)  # постоянное напряжение входит в эталон, запрос обязан его назвать
    return f"{text}, {b.source(canon.format_value(dc))}" if dc else text


# ---------------------------------------------------------------- правки текста
def _cyrillic_c(s):
    return re.sub(r"\bC(?=\d|\b)", "С", s)


def _swap_decimal(s):
    return re.sub(r"(?<=\d)[.,](?=\d)", lambda m: "." if m.group() == "," else ",", s)


def _glue_units(s):
    return re.sub(r"(\d) (кОм|МОм|Ом|мкФ|нФ|пФ|мкГн|мГн)", r"\1\2", s)


def _ground(s, rng):
    for w, alts in GND_WORDS.items():
        s = re.sub(rf"\b{w}\b", lambda _m: rng.choice(alts), s)
    return s


def _short_anchors(s, rng):
    pair = rng.choice([("in", "out"), ("вх", "вых")])
    s = re.sub(r"\bвход\b", pair[0], s)
    return re.sub(r"\bвыход\b", pair[1], s)


def _typo(s, rng):
    """Перестановка двух соседних букв в одном длинном слове без цифр."""
    words = [m for m in re.finditer(r"[а-яё]{7,}", s)]
    if not words:
        return s
    m = rng.choice(words)
    i = rng.randrange(1, len(m.group()) - 2)
    w = m.group()
    return s[:m.start()] + w[:i] + w[i + 1] + w[i] + w[i + 2:] + s[m.end():]


def _lower(s):
    """Строчная буква в начале запроса; единицы не трогает: МОм и мОм - разные величины."""
    return re.sub(r"^[А-ЯЁ](?=[а-яё]{2})", lambda m: m.group().lower(), s)


MUTATIONS = (("cyrillic_c", 0.15, lambda s, r: _cyrillic_c(s)), ("decimal", 0.10, lambda s, r: _swap_decimal(s)),
             ("glue", 0.15, lambda s, r: _glue_units(s)), ("ground", 0.15, _ground), ("anchors", 0.10, _short_anchors),
             ("typo", 0.08, _typo), ("lower", 0.15, lambda s, r: _lower(s)),
             ("prefix", 0.08, lambda s, r: r.choice(PREFIXES) + s), ("suffix", 0.05, lambda s, r: s.rstrip(".") + "," + r.choice(SUFFIXES)))


def mutate(prompt, rng, force=False):
    """-> (запрос, список применённых правок). force - все правки текста сразу, без вступлений и хвостов (gen_family)."""
    used = []
    for name, p, fn in MUTATIONS:
        if (force and name not in ("prefix", "suffix")) or (not force and rng.random() < p):
            new = fn(prompt, rng)
            if new != prompt:
                prompt = new
                used.append(name)
    return prompt, used
