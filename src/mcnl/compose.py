#!/usr/bin/env python3
"""Композиции датасета v2: схемы без названия (docs/generator_v2_spec.md, раздел 2.2).

Цепочка - последовательность секций от входа к выходу: элемент в разрыв, элемент или RC на землю.
Добавка - типовая схема плюс операции: R или RC на входе, R или RC на выходе, нагрузка, C параллельно ОС,
повторитель, ветвь к отдельному узлу, элемент между входом и выходом. Ожидаемые величины - из mna.py;
эталон всегда называет, откуда снимать выход. Запрос к цепочке проверяется обратным разбором parse_chain.
"""
import math
import random
import re

try:
    from . import bench, canon, describe_ru, filters_v2, measure, mna, sampler
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import describe_ru
    import filters_v2
    import measure
    import mna
    import sampler

MAX_ELEMENTS = 12
MAX_OPAMPS = 3
GRID_DB = 0.2       # уровень полосы не должен зависеть от сдвига сетки на полшага
MAX_PEAK_DB = 20.0  # подъём над краями: Q около 10, как в генераторе v1
CHAIN_SECTIONS = (2, 6)
SINK = ["на землю", "на общий", "на общий провод", "на корпус", "на минус"]
SERIES = ["в разрыв", "последовательно", "в линию"]
STEP = ["затем", "потом", "дальше"]
END = ["выход с последнего узла", "выход снимаем в конце", "выход после последнего элемента", "выход в конце цепочки"]
_GND_RE = r"на земл|на общ|на корпус|на минус"
_NUM = r"\d+(?:[.,]\d+)?"
_VALUE_RE = re.compile(r"(?<![\w.,])(" + _NUM + r")\s*(кОм|МОм|Ом|мкГн|мГн|мкФ|нФ|пФ|мк|к|М|н|п)?(?![а-яёА-ЯЁa-zA-Z])")
_UNITS = {"кОм": ("R", 1e3), "к": ("R", 1e3), "МОм": ("R", 1e6), "М": ("R", 1e6), "Ом": ("R", 1.0), "": ("R", 1.0),
          "мкФ": ("C", 1e-6), "мк": ("C", 1e-6), "нФ": ("C", 1e-9), "н": ("C", 1e-9), "пФ": ("C", 1e-12), "п": ("C", 1e-12),
          "мкГн": ("L", 1e-6), "мГн": ("L", 1e-3)}


def _pick(kind, rng):
    return filters_v2.pick(kind, rng)


def _fmt(x):
    return canon.format_value(x)


# ---------------------------------------------------------------- цепочки
def random_chain(rng):
    """Секции: ("ser", тип, номинал), ("shunt", тип, номинал), ("shunt_rc", R, C). Первая - в разрыв, последняя
    на землю или резистор в разрыв после неё; два одинаковых элемента в разрыв подряд не ставятся."""
    n = rng.randint(*CHAIN_SECTIONS)
    out = []
    for i in range(n):
        prev = out[-1] if out else None
        last = i == n - 1
        if i == 0 or (prev[0] != "ser" and not last and rng.random() < 0.6):
            kinds = [k for k in "RLC" if not prev or prev[:2] != ("ser", k)]
            k = rng.choices(kinds, [5 if x == "R" else 2 for x in kinds])[0]
            out.append(("ser", k, _pick(k, rng)))
        elif last and prev[0] != "ser" and rng.random() < 0.3:
            out.append(("ser", "R", _pick("R", rng)))
        elif rng.random() < 0.2:
            out.append(("shunt_rc", _pick("R", rng), _pick("C", rng)))
        else:
            k = rng.choice("CCCRL")
            out.append(("shunt", k, _pick(k, rng)))
    return out


def chain_netlist(sections):
    """Секции -> нетлист; выход - последний узел цепочки."""
    n_nodes = sum(1 for s in sections if s[0] == "ser")
    lines, count, cur, node = ["V1 IN 0 DC 0"], {}, "IN", 0

    def add(kind, a, b, value):
        count[kind] = count.get(kind, 0) + 1
        lines.append(f"{kind}{count[kind]} {a} {b} {_fmt(value)}")

    extra = 0
    for s in sections:
        if s[0] == "ser":
            node += 1
            nxt = "OUT" if node == n_nodes else f"N{node}"
            add(s[1], cur, nxt, s[2])
            cur = nxt
        elif s[0] == "shunt":
            add(s[1], cur, "0", s[2])
        else:
            extra += 1
            add("R", cur, f"M{extra}", s[1])
            add("C", f"M{extra}", "0", s[2])
    return "\n".join(lines) + "\n"


def describe_chain(sections, rng):
    """Запрос к цепочке. Единицы различают тип элемента, поэтому слова «резистор» и т.п. необязательны."""
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.5
    gnd = rng.choice(SINK)
    ser = rng.choice(SERIES)
    words = rng.random() < 0.4
    name = {"R": "резистор", "C": "конденсатор", "L": "катушка"}

    def val(k, v):
        return (name[k] + " " if words else "") + b.value(k, _fmt(v))

    parts = []
    for s in sections:
        if s[0] == "ser":
            parts.append(f"{val(s[1], s[2])} {ser}")
        elif s[0] == "shunt":
            parts.append(f"{val(s[1], s[2])} {gnd}")
        else:
            parts.append(f"{val('R', s[1])} и {val('C', s[2])} последовательно {gnd}")
    join = rng.choice([", ", "; ", " → ", None])
    text = join.join(parts) if join else (" " + rng.choice(STEP) + " ").join(parts)
    return f"{text}; {rng.choice(END)}"


def parse_chain(prompt):
    """Обратный разбор запроса цепочки -> нетлист."""
    body = re.split(r"[;,]?\s*выход ", prompt)[0]
    sections = []
    for seg in re.split(r";|→|(?<!\d),|,(?!\d)|\bзатем\b|\bпотом\b|\bдальше\b", body):
        vals = []
        for m in _VALUE_RE.finditer(seg):
            k, mult = _UNITS[m.group(2) or ""]
            vals.append((k, float(m.group(1).replace(",", ".")) * mult))
        if not vals:
            continue
        ground = re.search(_GND_RE, seg)
        if len(vals) == 2:
            sections.append(("shunt_rc", vals[0][1], vals[1][1]))
        else:
            sections.append(("shunt" if ground else "ser", vals[0][0], vals[0][1]))
    return chain_netlist(sections)


# ---------------------------------------------------------------- добавки
def _rename(lines, old, new):
    return [" ".join(new if t == old else t for t in l.split()) for l in lines]


def _next(lines, prefix):
    used = {t for l in lines for t in l.split()[1:]}
    k = 1
    while f"{prefix}{k}" in used:
        k += 1
    return f"{prefix}{k}"


def _element(lines, kind, a, b, value):
    k = 1
    names = {l.split()[0] for l in lines}
    while f"{kind}{k}" in names:
        k += 1
    lines.append(f"{kind}{k} {a} {b} {_fmt(value)}")


def _feedback_resistor(lines):
    """Резистор между инвертирующим входом и выходом ОУ -> его узлы."""
    pins = [l.split() for l in lines if l.startswith("X")]
    for l in lines:
        t = l.split()
        if t[0][0] == "R" and any({t[1], t[2]} == {p[2], p[3]} and p[2] != p[3] for p in pins):
            return t[1], t[2]
    return None


def _op(lines, op, rng):
    """Применить операцию -> (строки, параметры для описания) или None, если неприменима."""
    node = _next(lines, "A")
    if op == "in_r":
        lines = _rename(lines, "IN", node)
        r = _pick("R", rng)
        _element(lines, "R", "IN", node, r)
        return lines, (r,)
    if op == "in_rc":
        lines = _rename(lines, "IN", node)
        r, c = _pick("R", rng), _pick("C", rng)
        _element(lines, "R", "IN", node, r)
        _element(lines, "C", node, "0", c)
        return lines, (r, c)
    if op in ("out_r", "out_rc"):
        lines = _rename(lines, "OUT", node)
        r = _pick("R", rng)
        _element(lines, "R", node, "OUT", r)
        if op == "out_r":
            return lines, (r,)
        c = _pick("C", rng)
        _element(lines, "C", "OUT", "0", c)
        return lines, (r, c)
    if op == "load":
        r = _pick("R", rng)
        _element(lines, "R", "OUT", "0", r)
        return lines, (r,)
    if op == "side":
        r, c = _pick("R", rng), _pick("C", rng)
        _element(lines, "R", "OUT", node, r)
        _element(lines, "C", node, "0", c)
        return lines, (r, c)
    if op == "buffer":
        lines = _rename(lines, "OUT", node)
        k = 1 + sum(l.startswith("X") for l in lines)
        lines.append(f"X{k} {node} OUT OUT OPAMP_IDEAL")
        return lines, ()
    if op == "par_fb":
        fb = _feedback_resistor(lines)
        if not fb:
            return None
        c = _pick("C", rng)
        _element(lines, "C", fb[0], fb[1], c)
        return lines, (c,)
    if op == "bridge":
        c = _pick("C", rng)
        _element(lines, "C", "IN", "OUT", c)
        return lines, (c,)
    raise ValueError(op)


OPS = ("in_r", "in_rc", "par_fb", "out_r", "out_rc", "buffer", "load", "side", "bridge")  # порядок применения
GROUP = {"in_r": "in", "in_rc": "in", "out_r": "out", "out_rc": "out", "buffer": "out"}  # из группы - одна операция


def op_phrase(op, vals, b, rng):
    r = b.value("R", _fmt(vals[0])) if vals and op not in ("par_fb", "bridge") else ""
    c = b.value("C", _fmt(vals[-1])) if vals and op in ("in_rc", "out_rc", "side", "par_fb", "bridge") else ""
    gnd = rng.choice(SINK)
    return rng.choice({
        "in_r": [f"на входе ещё {r} последовательно", f"перед схемой {r} {rng.choice(SERIES)}"],
        "in_rc": [f"на входе {r} и {c} {gnd}", f"перед схемой {r} {rng.choice(SERIES)} и {c} {gnd}"],
        "out_r": [f"после выхода {r} последовательно, выход за ним", f"с выхода {r} {rng.choice(SERIES)}, снимаем после него"],
        "out_rc": [f"после выхода {r} и {c} {gnd}, выход с конденсатора", f"с выхода {r} {rng.choice(SERIES)}, затем {c} {gnd}, выход с конденсатора"],
        "load": [f"нагрузка {r} {gnd}", f"на выходе ещё {r} {gnd}"],
        "side": [f"от выхода к отдельному узлу {r}, с этого узла {c} {gnd}, выход остаётся на месте"],
        "buffer": ["на выходе повторитель на ОУ, выход с него", "после схемы буфер на ОУ, выход с буфера"],
        "par_fb": [f"параллельно резистору ОС {c}", f"{c} параллельно резистору обратной связи"],
        "bridge": [f"между входом и выходом ещё {c}", f"{c} от входа прямо на выход"],
    }[op])


def describe_addon(ref, rng):
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.5
    base = {"params": ref["params"]["base"], "class": ref["params"]["base_topology"]}
    head = describe_ru.describe_a1_brief(base, rng) if rng.random() < 0.6 else describe_ru.describe_a1(base, rng)
    tail = [op_phrase(op, vals, b, rng) for op, vals in ref["params"]["ops"]]
    return f"{head.rstrip('. ')}; {'; '.join(tail)}"


# ---------------------------------------------------------------- эталоны
BASES = ["rc_lp", "rc_hp", "rl_lp", "rl_hp", "rlc_series_bp", "rlc_parallel_bp", "rc_ladder2", "inverting", "noninverting",
         "active_lp_inv", "active_hp_inv", "sallen_key_lp", "sallen_key_hp", "follower", "mfb_bp",
         "rlc_series_lp", "rlc_series_hp", "rlc_series_bs", "mfb_lp", "mfb_hp"]


def _reference(netlist, cls, params):
    """Нетлист -> эталон или причина отказа."""
    elems, err = canon.parse_netlist(netlist)
    if err:
        return None, "parse"
    n_op = sum(e["type"] == "OPAMP" for e in elems)
    if len(elems) - 1 > MAX_ELEMENTS or n_op > MAX_OPAMPS:
        return None, "size"
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
        # узкий резонанс без потерь: пик попадает между точками сетки, и результат зависит от самой сетки
        half = 10 ** (0.5 / bench.POINTS_PER_DECADE)
        f2, db2, ph2 = mna.sweep(elems, lo * half, hi * half, per_decade=bench.POINTS_PER_DECADE)
        m2 = measure.measure(f2, db2, ph2)
    except (ZeroDivisionError, ValueError, TypeError):
        return None, "solve"
    if m["type"] != m2["type"] or abs(m["k_pass_db"] - m2["k_pass_db"]) > GRID_DB or max(db) - min(db[0], db[-1]) > MAX_PEAK_DB + abs(db[0] - db[-1]):
        return None, "resonance"
    if not sampler.GAIN_DB[0] <= m["k_pass_db"] <= sampler.GAIN_DB[1]:
        return None, "gain"
    q = ["type", "k_pass_db"] + (["fc"] if m["type"] in ("LP", "HP") and m.get("fc") else [])
    return {"id": cls, "class": cls, "level": "L1" if n_op == 0 else "L2" if n_op == 1 else "L3", "params": params,
            "netlist": canon.to_netlist(elems), "expected": {k: m[k] for k in q}, "quantities": q, "f_char": f_char}, None


def sample_chain(rng, max_tries=500):
    for _ in range(max_tries):
        sections = random_chain(rng)
        ref, _why = _reference(chain_netlist(sections), "compose:chain", {"sections": sections})
        if ref:
            return ref
    raise RuntimeError("цепочка не собрана")


def sample_addon(rng, bases=BASES, max_tries=500):
    for _ in range(max_tries):
        name = rng.choice(bases)
        base = filters_v2.sample_any(name, rng)
        net = canon.parse_netlist(base["netlist"])[0]
        lines = [l for l in base["netlist"].splitlines() if l and not l.startswith("V")]
        # порядок применения фиксирован: слова «вход» и «выход» в запросе значат итоговые узлы схемы
        groups, want = {}, rng.randint(1, 3)
        for op in rng.sample(OPS, len(OPS)):
            groups.setdefault(GROUP.get(op, op), op)
        ops = []
        for op in sorted(rng.sample(list(groups.values()), want), key=OPS.index):
            res = _op(lines, op, rng)
            if res:
                lines, vals = res
                ops.append((op, vals))
        if not ops:
            continue
        src = next(e for e in net if e["type"] == "V")
        netlist = f"V1 IN 0 DC {src['value']}\n" + "\n".join(lines) + "\n"
        params = {"base_topology": name, "base": base["params"], "ops": ops}
        ref, _why = _reference(netlist, f"compose:{name}", params)
        if ref:
            return ref
    raise RuntimeError("добавка не собрана")


if __name__ == "__main__":
    import json
    import sys
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    a, b = sample_chain(rng), sample_addon(rng)
    print(json.dumps({"chain": describe_chain(a["params"]["sections"], rng), "chain_net": a["netlist"],
                      "addon": describe_addon(b, rng), "addon_net": b["netlist"]}, ensure_ascii=False, indent=1))
