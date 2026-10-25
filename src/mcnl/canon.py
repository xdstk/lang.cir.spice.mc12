#!/usr/bin/env python3
"""Каноническая форма нетлиста класса A и JSON-IR `mcnl.ir/1` (docs/eval_spec.md, разделы 2-3).

Запуск:  python canon.py файл.ckt          -> канонический нетлист
         python canon.py файл.ckt --ir     -> JSON-IR

Элемент - словарь {"type", "name", "nodes", "value"}:
  type  - R, C, L, V, I, OPAMP;
  nodes - выводы в фиксированном порядке: R/C/L (2, порядок не важен), V/I (+, -), OPAMP (+, -, выход);
  value - номинал R/C/L или DC-значение V/I строкой в канонической записи; у OPAMP - None.
Номиналы хранятся строками (как в IR), числовое значение даёт parse_value.
"""
import json
import math
import re
import sys
from collections import deque

SCHEMA = "mcnl.ir/1"
TYPES = ("V", "I", "R", "C", "L", "OPAMP")  # он же порядок строк канонического нетлиста
NPINS = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "OPAMP": 3}
OPAMP_SUBCKT = "OPAMP_IDEAL"
ANCHORS = ("IN", "OUT", "0")

# суффиксы SPICE; M - милли, MEG - мега (RM12 и любой SPICE); MIL не поддерживается
SUFFIX = [("MEG", 1e6), ("T", 1e12), ("G", 1e9), ("K", 1e3), ("M", 1e-3),
          ("U", 1e-6), ("N", 1e-9), ("P", 1e-12), ("F", 1e-15)]
CANON_SUFFIX = [(1e9, "G"), (1e6, "MEG"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
                (1e-6, "u"), (1e-9, "n"), (1e-12, "p")]
VALUE_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z]*)$")


# ---------------------------------------------------------------- номиналы
def parse_value(s):
    """'4.7k' -> 4700.0; '10M' -> 0.01 (милли); '10MEG' -> 1e7; '100nF' -> 1e-7 (буквы после
    суффикса, как в SPICE, игнорируются). Нечитаемое ('µ', '1,5k', '') -> None."""
    m = VALUE_RE.match(str(s).strip())
    if not m:
        return None
    v, tail = float(m.group(1)), m.group(2).upper()
    for suf, mult in SUFFIX:
        if tail.startswith(suf):
            return v * mult
    return v  # без суффикса или только единицы измерения ('100V')


def format_value(x):
    """Каноническая инженерная запись: мантисса 1 <= m < 1000, до 6 значащих цифр без лишних
    нулей, суффиксы p n u m k MEG G; 0 -> '0'. Отрицательные значения - со знаком (DC источника)."""
    if x == 0:
        return "0"
    sign, a = ("-" if x < 0 else ""), abs(x)
    for mult, suf in CANON_SUFFIX:
        if a >= mult * (1 - 1e-12):
            m = a / mult
            break
    else:
        mult, suf, m = 1e-12, "p", a / 1e-12
    m = float(f"{m:.6g}")
    if m >= 1000 and suf != "G":  # округление дало 1000: перейти к следующему суффиксу
        i = [s for _, s in CANON_SUFFIX].index(suf)
        mult, suf = CANON_SUFFIX[i - 1]
        m = float(f"{a / mult:.6g}")
    txt = f"{m:.6g}"
    if "e" in txt:  # вне диапазона суффиксов
        return sign + f"{a:.6g}"
    return sign + txt + suf


def canon_value(s):
    v = parse_value(s)
    return None if v is None else format_value(v)


# ---------------------------------------------------------------- разбор строк
def parse_netlist(text):
    """Строки элементов класса A -> (элементы, ошибки). Комментарии (*, ;), пустые строки, .END
    и заголовок не нужны модели, но допускаются; остальные директивы - ошибка разбора (S0)."""
    elems, errors = [], []
    for no, raw in enumerate(text.replace("\r", "").split("\n"), 1):
        line = raw.split(";")[0].strip()
        if not line or line.startswith("*") or line.upper() == ".END":
            continue
        t = line.split()
        head = t[0].upper()
        letter = head[0]
        try:
            if head.startswith("."):
                raise ValueError(f"директива {t[0]} не входит в ответ")
            if letter in "RCL":
                if len(t) != 4:
                    raise ValueError("ожидается: имя узел узел номинал")
                val = canon_value(t[3])
                if val is None:
                    raise ValueError(f"нечитаемый номинал {t[3]}")
                elems.append({"type": letter, "name": t[0], "nodes": t[1:3], "value": val})
            elif letter in "VI":
                if len(t) < 3:
                    raise ValueError("ожидается: имя узел+ узел- [DC] значение")
                elems.append({"type": letter, "name": t[0], "nodes": t[1:3], "value": source_dc(t[3:])})
            elif letter == "X":
                if len(t) != 5 or t[4].upper() != OPAMP_SUBCKT:
                    raise ValueError(f"ожидается: X имя + - выход {OPAMP_SUBCKT}")
                elems.append({"type": "OPAMP", "name": t[0], "nodes": t[1:4], "value": None})
            else:
                raise ValueError(f"элемент {t[0]} вне класса A")
        except ValueError as e:
            errors.append(f"строка {no}: {e}: {raw.strip()}")
    for e in elems:
        e["nodes"] = [norm_node(n) for n in e["nodes"]]
    return elems, errors


def source_dc(rest):
    """Хвост строки источника -> DC-значение строкой. 'DC 5', '5', 'DC 0 AC 1 0', '' (-> '0').
    AC-часть ставит обвязка, в каноне она не хранится."""
    toks = [x for x in re.sub(r"[(),]", " ", " ".join(rest)).split()]
    i = 0
    while i < len(toks):
        u = toks[i].upper()
        if u == "DC" and i + 1 < len(toks):
            v = canon_value(toks[i + 1])
            if v is None:
                raise ValueError(f"нечитаемое DC-значение {toks[i + 1]}")
            return v
        if u == "AC":
            i += 2
            if i < len(toks) and parse_value(toks[i]) is not None:
                i += 1  # фаза
            continue
        if i == 0 and parse_value(toks[0]) is not None:
            return canon_value(toks[0])
        raise ValueError(f"источник вне класса A: {' '.join(rest)}")
    return "0"


def norm_node(n):
    u = n.upper()
    return "0" if u in ("0", "GND") else u


# ---------------------------------------------------------------- канонизация
def _type_rank(t):
    return TYPES.index(t)


def _num(e):
    v = parse_value(e["value"]) if e["value"] is not None else None
    return v if v is not None else -math.inf


def canonicalize(elems):
    """Канонический порядок, имена элементов и узлов (раздел 2). Детерминирована: одинаковый вход
    -> одинаковый выход; при симметрии допускает разные записи для перестановок входа - оценка
    всё равно идёт через изоморфизм (iso.py)."""
    incident = {}
    for k, e in enumerate(elems):
        for n in e["nodes"]:
            incident.setdefault(n, []).append(k)
    # метка и ранг узла: якоря - сами себе; внутренние - N1, N2, ... в порядке обхода
    label = {a: a for a in ANCHORS}
    rank = {"IN": 0, "OUT": 1e9, "0": 1e10}
    queued, order, seen = set(), [], set()

    def key(k):  # (тип, номинал, ранги соседних узлов; ещё не названные - последними, вход)
        e = elems[k]
        return (_type_rank(e["type"]), _num(e), sorted(rank.get(n, math.inf) for n in e["nodes"]), k)

    def visit(start):
        queue = deque([start])
        queued.add(start)
        while queue:
            u = queue.popleft()
            for k in sorted((k for k in incident.get(u, []) if k not in seen), key=key):
                seen.add(k)
                order.append(k)
                for n in elems[k]["nodes"]:
                    if n not in label:
                        label[n] = f"N{len(label) - len(ANCHORS) + 1}"
                        rank[n] = len(label)
                    if n not in queued:
                        queued.add(n)
                        queue.append(n)

    if "IN" in incident:
        visit("IN")
    while len(seen) < len(elems):  # часть схемы не связана с IN: продолжить с первого её узла
        start = elems[min(k for k in range(len(elems)) if k not in seen)]["nodes"][0]
        if start not in label:
            label[start] = f"N{len(label) - len(ANCHORS) + 1}"
            rank[start] = len(label)
        visit(start)
    out, counters = [], {}
    for k in sorted(order, key=lambda k: (_type_rank(elems[k]["type"]), order.index(k))):
        e = elems[k]
        nodes = [label[n] for n in e["nodes"]]
        if e["type"] in "RCL":
            nodes.sort(key=_node_rank)
        counters[e["type"]] = counters.get(e["type"], 0) + 1
        prefix = "X" if e["type"] == "OPAMP" else e["type"]
        out.append({"type": e["type"], "name": f"{prefix}{counters[e['type']]}", "nodes": nodes,
                    "value": e["value"]})
    return out


def _node_rank(n):
    if n == "IN":
        return (0, 0)
    if n.startswith("N") and n[1:].isdigit():
        return (1, int(n[1:]))
    if n == "OUT":
        return (2, 0)
    if n == "0":
        return (4, 0)
    return (3, n)


def to_netlist(elems):
    lines = []
    for e in elems:
        if e["type"] == "OPAMP":
            lines.append(f"{e['name']} {' '.join(e['nodes'])} {OPAMP_SUBCKT}")
        elif e["type"] in "VI":
            lines.append(f"{e['name']} {' '.join(e['nodes'])} DC {e['value']}")
        else:
            lines.append(f"{e['name']} {' '.join(e['nodes'])} {e['value']}")
    return "\n".join(lines) + "\n"


def canonical_netlist(text):
    elems, errors = parse_netlist(text)
    if errors:
        raise ValueError("; ".join(errors))
    return to_netlist(canonicalize(elems))


# ---------------------------------------------------------------- IR
IR_VALUE_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:[pnumkMG]|MEG|meg)?$")
NODE_RE = re.compile(r"^[A-Za-z0-9_]+$")


def to_ir(elems):
    out = []
    for e in elems:
        d = {"type": e["type"], "nodes": list(e["nodes"])}
        if e["type"] in "VI":
            d["dc"] = e["value"]
        elif e["type"] != "OPAMP":
            d["value"] = e["value"]
        out.append(d)
    return {"schema": SCHEMA, "elements": out}


def validate_ir(ir):
    """Проверка по схеме mcnl.ir/1 -> список ошибок (пустой - валиден)."""
    errs = []
    if not isinstance(ir, dict) or ir.get("schema") != SCHEMA:
        return [f"нет поля schema = {SCHEMA}"]
    els = ir.get("elements")
    if not isinstance(els, list) or not els:
        return ["elements - непустой список"]
    for i, e in enumerate(els):
        where = f"elements[{i}]"
        if not isinstance(e, dict):
            errs.append(f"{where}: не объект")
            continue
        t = e.get("type")
        if t not in NPINS:
            errs.append(f"{where}: type {t!r} не из {list(NPINS)}")
            continue
        nodes = e.get("nodes")
        if not isinstance(nodes, list) or len(nodes) != NPINS[t] or \
                not all(isinstance(n, str) and NODE_RE.match(n) for n in nodes):
            errs.append(f"{where}: nodes - {NPINS[t]} имени узла")
        field = {"V": "dc", "I": "dc", "OPAMP": None}.get(t, "value")
        allowed = {"type", "nodes", "name"} | ({field} if field else set())
        extra = set(e) - allowed
        if extra:
            errs.append(f"{where}: лишние поля {sorted(extra)}")
        if field == "value" and not (isinstance(e.get("value"), str) and IR_VALUE_RE.match(e["value"])):
            errs.append(f"{where}: value - строка номинала ('10k', '100n')")
        if field == "dc" and "dc" in e and not (isinstance(e["dc"], str) and IR_VALUE_RE.match(e["dc"])):
            errs.append(f"{where}: dc - строка значения ('5', '0')")
    return errs


def from_ir(ir):
    errs = validate_ir(ir)
    if errs:
        raise ValueError("; ".join(errs))
    out, counters = [], {}
    for e in ir["elements"]:
        t = e["type"]
        counters[t] = counters.get(t, 0) + 1
        name = e.get("name") or (("X" if t == "OPAMP" else t) + str(counters[t]))
        value = None if t == "OPAMP" else canon_value(e.get("dc", "0") if t in "VI" else e["value"])
        out.append({"type": t, "name": name, "nodes": [norm_node(n) for n in e["nodes"]], "value": value})
    return out


if __name__ == "__main__":
    elems, errors = parse_netlist(open(sys.argv[1], encoding="cp1251", errors="replace").read())
    if errors:
        print("\n".join(errors), file=sys.stderr)
    can = canonicalize(elems)
    print(json.dumps(to_ir(can), ensure_ascii=False, indent=1) if "--ir" in sys.argv else to_netlist(can), end="")
