#!/usr/bin/env python3
"""Запросы датасета v3: имена узлов из запроса и описание не по порядку (docs/generator_v3_spec.md, П1 и П6).

П1. Внутренние узлы названы в запросе буквами, буквой с номером или числами; эталон сохраняет эти имена
(решение автора 06.11.2026). Правило имени - node_name: латинское имя переписывается как есть (в верхнем регистре,
как узлы канона: «точка a» -> A), число k -> Nk («узел 3» -> N3, как в семействах v1 и v2). Порядок строк и имена
элементов эталона - как у канонического нетлиста, заменены только узлы.
П6. С вероятностью shuffle элементы описаны не в порядке канона: переставлены детали, а у narrative и path - и
предложения, у table - строки; у части деталей выводы названы в обратном порядке.

Семейства list, table, narrative, path строит describe_ru (gen 3), имена подставляются в готовый текст; freeform -
свой вариант (обозначения R1, имена без слова «узел», две записи ОУ). prompts_v2 и describe_ru не меняются.
Каждый запрос проверяется обратным разбором parse_named: схема и имена узлов должны совпасть с эталоном.
"""
import random
import re

try:
    from . import canon, compose, describe_ru, iso
except ImportError:  # запуск как скрипта
    import canon
    import compose
    import describe_ru
    import iso

FAMILIES = ("freeform", "list", "table", "narrative", "path")
SHUFFLE = 0.3   # доля запросов с перестановкой (П6)
FLIP = 0.3      # в переставленном запросе - доля деталей с обратным порядком выводов
ABC = list("ABCDEFGHJKLM")      # без I и O: похожи на 1 и 0
XYZ = list("XYZUWPQSTK")
PREFIXES = ["P", "A", "K", "T", "S", "M", "n", "u", "a"]  # не R, C, L, V, I, X: имя узла не должно совпасть с элементом
# схемы именования и их веса; "seq" (номера канона, как в v2) случайно не выбирается
SCHEMES = {"abc": 3, "xyz": 2, "lower": 1, "indexed": 1.5, "numbers": 1.5, "mixed": 1}
RESERVED = {"IN", "OUT", "0", "GND"}
_ELEMENT_LIKE = re.compile(r"^[RCLVIX]\d+$")
_BARE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_INNER_RE = re.compile(r"(точк(?:а|и|е|у|ой)|узл(?:а|у|ом)|узел) ([A-Za-z0-9_]+)", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=\.) (?=[А-ЯЁ])")
_ANCHORS = {"IN": ("вход", "входа", "входу", "входом"), "OUT": ("выход", "выхода", "выходу", "выходом"),
            "0": ("земля", "земли", "земле", "землёй")}
_FF_NODE = re.compile(r"(вход\w*|выход\w*|земл\w*)|(?:узл\w*|узел|точк\w+)\s+([A-Za-z0-9_]+)"
                      r"|(?<![\w.,])([A-Za-z][A-Za-z0-9_]*)(?!\w)")
_VOLTS = re.compile(r"(-?\d+(?:[.,]\d+)?) В(?![а-яёА-ЯЁ\w])")


# ---------------------------------------------------------------- имена узлов
def node_name(token):
    """Имя узла в запросе -> имя узла в эталоне: число k («узел 3», «т3», «т.3», «у3») -> Nk, иначе имя как есть,
    в верхнем регистре."""
    t = token.strip(" .,;:()")
    m = re.fullmatch(r"[тТуУ]?\.?(\d+)", t)
    return f"N{int(m.group(1))}" if m else t.upper()


def pick_names(inner, rng, scheme=None):
    """Имена для внутренних узлов канона (N1, N2, ...) -> (схема именования, {узел: имя в запросе})."""
    inner = sorted(inner, key=lambda n: int(n[1:]))
    n = len(inner)
    if scheme is None:
        scheme = rng.choices(list(SCHEMES), list(SCHEMES.values()))[0]
    if (scheme in ("abc", "lower") and n > len(ABC)) or (scheme == "xyz" and n > len(XYZ)):
        scheme = "indexed"
    if scheme == "seq":
        names = [x[1:] for x in inner]
    elif scheme in ("abc", "lower"):
        names = [c.lower() if scheme == "lower" else c for c in ABC[:n]]
        if rng.random() < 0.6:  # буквы не в порядке обхода схемы (mc12_real)
            rng.shuffle(names)
    elif scheme == "xyz":
        names = rng.sample(XYZ, n)
    elif scheme == "indexed":
        p = rng.choice(PREFIXES)
        names = [f"{p}{k}" for k in range(1, n + 1)]
        if rng.random() < 0.5:
            rng.shuffle(names)
    elif scheme == "numbers":
        names = [str(k) for k in rng.sample(range(1, max(n + 3, 9) + 1), n)]
    elif scheme == "mixed":
        letters = rng.sample(sorted(set(ABC + XYZ) - {"N"}), n)
        numbers = rng.sample(range(1, n + 4), n)
        names = [letters[i] if rng.random() < 0.5 else str(numbers[i]) for i in range(n)]
    else:
        raise ValueError(f"неизвестная схема имён {scheme}")
    label = dict(zip(inner, names))
    out = [node_name(x) for x in names]
    if len(set(out)) != n or any(x in RESERVED or _ELEMENT_LIKE.match(x) for x in out):
        raise AssertionError(f"недопустимые имена узлов: {names}")
    return scheme, label


def _inner(elems):
    nodes = {n for e in elems for n in e["nodes"]} - set(canon.ANCHORS)
    bad = [n for n in nodes if not re.fullmatch(r"N\d+", n)]
    if bad:
        raise ValueError(f"нужен канонический нетлист, узлы {sorted(bad)}")
    return nodes


def same_circuit(a, b):
    """Те же элементы на тех же узлах (имена узлов совпадают буквально, имена элементов не важны)."""
    def key(e):
        nodes = tuple(sorted(e["nodes"])) if e["type"] in "RCL" else tuple(e["nodes"])
        return e["type"], nodes, canon.parse_value(e["value"]) if e["value"] is not None else 0.0
    a, b = sorted(a, key=key), sorted(b, key=key)
    return len(a) == len(b) and all(key(x)[:2] == key(y)[:2] and iso._same_value(x["value"], y["value"])
                                    for x, y in zip(a, b))


# ---------------------------------------------------------------- запрос
def describe_named(netlist, family, rng, scheme=None, shuffle=SHUFFLE):
    """Канонический нетлист -> {"prompt", "netlist" (эталон с именами узлов из запроса), "names" (схема именования;
    None, если внутренних узлов нет), "shuffled"}."""
    elems, err = canon.parse_netlist(netlist)
    if err:
        raise ValueError(err)
    inner = _inner(elems)
    scheme, label = pick_names(inner, rng, scheme)
    target = [dict(e, nodes=[node_name(label[n]) if n in label else n for n in e["nodes"]]) for e in elems]
    shuffled = shuffle > 0 and rng.random() < shuffle
    order = [dict(e, nodes=list(e["nodes"])) for e in elems]
    if shuffled:
        src = [e for e in order if e["type"] in "VI"]
        two = [e for e in order if e["type"] in "RCL"]
        ops = [e for e in order if e["type"] == "OPAMP"]
        for e in two:
            if "0" not in e["nodes"] and rng.random() < FLIP:
                e["nodes"].reverse()
        if family == "freeform":  # ОУ вперемешку с деталями
            rest = two + ops
            rng.shuffle(rest)
        else:
            rng.shuffle(two)
            rest = two + ops
        order = src + rest
    if family == "freeform":
        prompt = _freeform(order, label, rng, shuffled)
    elif family in FAMILIES:
        text = describe_ru.describe_a(canon.to_netlist(order), family, rng, gen=3)
        if shuffled and family in ("narrative", "path"):
            parts = _SENTENCE.split(text)
            rng.shuffle(parts)
            text = " ".join(parts)
        elif shuffled and family == "table":
            rows = text.split("\n")
            body = rows[1:]
            rng.shuffle(body)
            text = "\n".join(rows[:1] + body)
        prompt = _INNER_RE.sub(lambda m: f"{m.group(1)} {label['N' + m.group(2)]}", text)
    else:
        raise ValueError(f"неизвестное семейство {family}")
    return {"prompt": prompt, "netlist": canon.to_netlist(target), "names": scheme if inner else None, "shuffled": shuffled}


def _freeform(elems, label, rng, shuffled):
    """Соединения словами, как prompts_v2.describe_freeform; новое: обозначения деталей, имена узлов без слова
    «узел», «на землю», нумерация и вторая запись ОУ, источник в начале."""
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.6
    # шесть значащих цифр, как в каноне: у Brief их пять, и номинал вида 19.1305k (R/2 двойного Т-моста) терялся бы
    b.num = lambda x: f"{x:.6g}".replace(".", ",") if b.comma else f"{x:.6g}"
    word = rng.choice(["узел", "точка"])
    form = {"узел": ("узел {}", "узла {}", "узлу {}", "узлом {}"), "точка": ("точка {}", "точки {}", "точке {}", "точкой {}")}[word]
    bare = bool(label) and all(_BARE.match(x) for x in label.values()) and rng.random() < 0.4
    des = rng.random() < 0.4
    kind = rng.choice(["from_to", "between", "colon"])
    op_style = rng.choice(["pm", "inv"])
    n_ops = sum(e["type"] == "OPAMP" for e in elems)
    numbered = n_ops > 1 and rng.random() < 0.5

    def node(n, i):
        if n in _ANCHORS:
            return _ANCHORS[n][i]
        return label[n] if bare else form[i].format(label[n])

    parts = []
    for e in elems:
        if e["type"] in "RCL":
            a, c = e["nodes"]
            v = (e["name"] + " " if des else "") + b.value(e["type"], e["value"])
            if kind == "from_to":
                to = "на землю" if c == "0" and rng.random() < 0.5 else f"к {node(c, 2)}"
                parts.append(f"{v} от {node(a, 1)} {to}")
            elif kind == "between":
                parts.append(f"{v} между {node(a, 3)} и {node(c, 3)}")
            else:
                parts.append(f"{v}: {node(a, 0)} - {node(c, 0)}")
        elif e["type"] == "OPAMP":
            p, m, o = (node(x, 0) for x in e["nodes"])
            tag = f"ОУ{e['name'][1:]}" if numbered else "ОУ"
            parts.append(f"{tag}: плюс - {p}, минус - {m}, выход ОУ - {o}" if op_style == "pm" else
                         f"{tag}: неинв. вход - {p}, инв. вход - {m}, выход ОУ - {o}")
    src = b.source(next(e for e in elems if e["type"] == "V")["value"])
    parts = [src] + parts if shuffled and rng.random() < 0.3 else parts + [src]
    return "; ".join(parts)


# ---------------------------------------------------------------- обратный разбор
def _ff_node(text):
    m = _FF_NODE.search(text)
    if not m:
        raise ValueError(f"нет узла: {text}")
    if m.group(1):
        w = m.group(1).lower()
        return "IN" if w.startswith("вход") else "OUT" if w.startswith("выход") else "0"
    return node_name(m.group(2) or m.group(3))


def parse_freeform(prompt):
    """Запрос freeform v3 -> элементы с именами узлов эталона."""
    elems, count, source = [], {}, None
    for seg in prompt.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        if seg.startswith("ОУ"):
            pins = [item.split(" - ", 1)[1] for item in seg.split(":", 1)[1].split(", ")]
            if len(pins) != 3:
                raise ValueError(f"у ОУ не три вывода: {seg}")
            count["X"] = count.get("X", 0) + 1
            elems.append({"type": "OPAMP", "name": f"X{count['X']}", "nodes": [_ff_node(p) for p in pins], "value": None})
            continue
        if "сигнал" in seg or _VOLTS.search(seg):
            vm = _VOLTS.search(seg)
            source = "0" if vm is None else canon.format_value(float(vm.group(1).replace(",", ".")))
            continue
        vm = compose._VALUE_RE.search(seg)
        if not vm:
            raise ValueError(f"нет номинала: {seg}")
        k, mult = compose._UNITS[vm.group(2) or ""]
        rest = seg[vm.end():]
        nodes = [_ff_node(m.group()) for m in _FF_NODE.finditer(rest)]
        if len(nodes) != 2:
            raise ValueError(f"у двухполюсника не два узла: {seg}")
        count[k] = count.get(k, 0) + 1
        elems.append({"type": k, "name": f"{k}{count[k]}", "nodes": nodes,
                      "value": canon.format_value(float(vm.group(1).replace(",", ".")) * mult)})
    if source is None:
        raise ValueError("не найден источник")
    return [{"type": "V", "name": "V1", "nodes": ["IN", "0"], "value": source}] + elems


def parse_named(prompt, family):
    """Обратный разбор запроса v3 -> элементы с именами узлов эталона. Для семейств describe_ru имена на время
    разбора заменяются номерами (parse_a понимает только «точка k»), затем возвращаются."""
    if family == "freeform":
        return parse_freeform(prompt)
    index = {}

    def sub(m):
        k = index.setdefault(m.group(2), len(index) + 1)
        return f"{m.group(1)} {k}"

    elems = describe_ru.parse_a(_INNER_RE.sub(sub, prompt), family)
    back = {f"N{k}": node_name(name) for name, k in index.items()}
    return [dict(e, nodes=[back.get(n, n) for n in e["nodes"]]) for e in elems]


def verified(netlist, family, rng, scheme=None, shuffle=SHUFFLE):
    """describe_named с проверкой: обратный разбор запроса даёт эталон с теми же именами узлов, а эталон - та же
    схема, что исходный нетлист. Иначе AssertionError (генератор берёт другое семейство)."""
    d = describe_named(netlist, family, rng, scheme, shuffle)
    target = canon.parse_netlist(d["netlist"])[0]
    if not same_circuit(target, parse_named(d["prompt"], family)):
        raise AssertionError(f"неоднозначный запрос: {d['prompt']}")
    if not iso.match(canon.parse_netlist(netlist)[0], target, "values")["match"]:
        raise AssertionError(f"эталон не совпал со схемой: {d['netlist']}")
    return d


if __name__ == "__main__":
    import sys
    net = open(sys.argv[1], encoding="utf-8").read()
    for fam in FAMILIES:
        d = verified(net, fam, random.Random(1))
        print(f"--- {fam} ({d['names']}, shuffled={d['shuffled']})\n{d['prompt']}\n{d['netlist']}")
