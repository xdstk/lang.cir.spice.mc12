#!/usr/bin/env python3
"""Композиции датасета v3 (docs/generator_v3_spec.md, П2-П6). compose.py (v2) не меняется, кроме необязательных
пределов размера в compose._reference.

П3. Цепочки до 8 секций; ветвь на землю из двух элементов - RC, CR, RL, LR (первый названный элемент - от линии к
    внутреннему узлу, второй - от него на землю); доля цепочек с ветвью - BRANCH_SHARE.
П4. Добавки без указания выхода (доля NO_OUT_SHARE): «после выхода 1к последовательно» - OUT за добавкой, как в
    соглашении v2; ветвь «к отдельному узлу» выход не сдвигает.
П5. Основы добавок - все названные топологии v1 и v2 (BASES).
П6. Описание не по порядку (доля REORDER): у цепочки - отдельно элементы линии и отдельно ветви на землю с номером
    места («после 2-го - 100н»); у добавки - фразы в другом порядке, основа не первой.
П2. Схемы на 2-4 ОУ без названия (sample_multi): разнородные каскады звеньев с пассивной секцией между ними, петли
    из двух интеграторов (Tow-Thomas, KHN) с выходом на любом ОУ, петля со звеном до или после неё. Запрос к ним -
    описание соединений (prompts_v3). Петли устойчивы по построению; произвольные перекрёстные связи не добавляются:
    устойчивость такой схемы расчёт на переменном токе не проверяет.
"""
import random
import re

try:
    from . import canon, cascade, compose, describe_ru, filters_v2, formulas
except ImportError:  # запуск как скрипта
    import canon
    import cascade
    import compose
    import describe_ru
    import filters_v2
    import formulas

CHAIN_SECTIONS = (2, 8)
BRANCH_SHARE = 0.6
BRANCH_KINDS = [("R", "C"), ("R", "C"), ("C", "R"), ("R", "L"), ("L", "R")]
NO_OUT_SHARE = 0.5
REORDER = 0.3
# интегратор без резистора в ОС основой не бывает: у него нет полосы, характерная частота не определяется
BASES = [t for t in list(formulas.TOPOLOGIES) + list(filters_v2.TOPOLOGIES_V2) if t != "integrator"]
MULTI_ELEMENTS = 18
MULTI_OPAMPS = 4
LOOPS = ["tow_thomas", "khn"]
LINE_HEAD = ["в линии от входа", "последовательно от входа к выходу", "в разрыв по порядку"]
ORDINALS = ["первого", "второго", "третьего", "четвёртого", "пятого", "шестого", "седьмого", "восьмого"]
HEAD_LAST = ["основа - ", "сама схема: ", "всё это к схеме: "]
_NAME = {"R": "резистор", "C": "конденсатор", "L": "катушка"}
_AFTER = re.compile(r"после (\S+) - (.*)$")
_SPLIT = r";|→|(?<!\d),|,(?!\d)|\bзатем\b|\bпотом\b|\bдальше\b"


def _stage_pool():
    """Звенья на одном ОУ с полосой или усилением (без интегратора и дифференциатора: у них нет полосы)."""
    names = [t for t, v in formulas.TOPOLOGIES.items() if v[1] == "L2"]
    return names + [t for t, v in filters_v2.TOPOLOGIES_V2.items() if v[1] == "L2" and t not in ("integrator", "differentiator")]


STAGES = _stage_pool()


# ---------------------------------------------------------------- цепочки (П3, П6)
def _chain(rng, branch):
    n = rng.randint(*CHAIN_SECTIONS)
    out = []
    for i in range(n):
        prev = out[-1] if out else None
        last = i == n - 1
        if i == 0 or (prev[0] != "ser" and not last and rng.random() < 0.6):
            kinds = [k for k in "RLC" if not prev or prev[:2] != ("ser", k)]
            k = rng.choices(kinds, [5 if x == "R" else 2 for x in kinds])[0]
            out.append(("ser", k, compose._pick(k, rng)))
        elif last and prev[0] != "ser" and rng.random() < 0.3:
            out.append(("ser", "R", compose._pick("R", rng)))
        elif branch and rng.random() < 0.4:
            a, b = rng.choice(BRANCH_KINDS)
            out.append(("branch", a, compose._pick(a, rng), b, compose._pick(b, rng)))
        else:
            k = rng.choice("CCCRL")
            out.append(("shunt", k, compose._pick(k, rng)))
    return out


def random_chain(rng, branch=None):
    """Секции: ("ser", тип, номинал), ("shunt", тип, номинал), ("branch", тип, номинал, тип, номинал).
    branch - есть ли ветвь из двух элементов; None - с вероятностью BRANCH_SHARE."""
    want = rng.random() < BRANCH_SHARE if branch is None else branch
    while True:
        s = _chain(rng, want)
        if any(x[0] == "branch" for x in s) == want:
            return s


def chain_netlist(sections):
    """Секции -> нетлист; выход - последний узел линии."""
    n_nodes = sum(1 for s in sections if s[0] == "ser")
    lines, count, cur, node, extra = ["V1 IN 0 DC 0"], {}, "IN", 0, 0

    def add(kind, a, b, value):
        count[kind] = count.get(kind, 0) + 1
        lines.append(f"{kind}{count[kind]} {a} {b} {compose._fmt(value)}")

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
            add(s[1], cur, f"M{extra}", s[2])
            add(s[3], f"M{extra}", "0", s[4])
    return "\n".join(lines) + "\n"


def describe_chain(sections, rng, reorder=None):
    """-> {"prompt", "reordered"}. По порядку - как в v2; иначе линия и ветви на землю описаны отдельно."""
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.5
    gnd = rng.choice(compose.SINK)
    words = rng.random() < 0.4
    split = (rng.random() < REORDER) if reorder is None else reorder

    def val(k, v):
        return (_NAME[k] + " " if words else "") + b.value(k, compose._fmt(v))

    def shunt(s):
        if s[0] == "shunt":
            return val(s[1], s[2])
        return f"{val(s[1], s[2])} и {val(s[3], s[4])} последовательно"

    end = rng.choice(compose.END)
    if not split:
        ser = rng.choice(compose.SERIES)
        parts = [f"{val(s[1], s[2])} {ser}" if s[0] == "ser" else f"{shunt(s)} {gnd}" for s in sections]
        join = rng.choice([", ", "; ", " → ", None])
        text = join.join(parts) if join else (" " + rng.choice(compose.STEP) + " ").join(parts)
        return {"prompt": f"{text}; {end}", "reordered": False}
    line = [val(s[1], s[2]) for s in sections if s[0] == "ser"]
    digits = rng.random() < 0.5
    items, pos = [], 0
    for s in sections:
        if s[0] == "ser":
            pos += 1
            continue
        where = "последнего" if pos == len(line) and rng.random() < 0.5 else f"{pos}-го" if digits else ORDINALS[pos - 1]
        items.append(f"после {where} - {shunt(s)}")
    rng.shuffle(items)
    blocks = [f"{rng.choice(LINE_HEAD)}: {', '.join(line)}"] + ([f"{gnd}: {', '.join(items)}"] if items else [])
    if rng.random() < 0.3:
        blocks.reverse()
    return {"prompt": "; ".join(blocks + [end]), "reordered": True}


def _values(seg):
    out = []
    for m in compose._VALUE_RE.finditer(seg):
        k, mult = compose._UNITS[m.group(2) or ""]
        out.append((k, float(m.group(1).replace(",", ".")) * mult))
    return out


def _shunt_section(vals):
    if len(vals) == 2:
        return ("branch", vals[0][0], vals[0][1], vals[1][0], vals[1][1])
    if len(vals) == 1:
        return ("shunt", vals[0][0], vals[0][1])
    raise ValueError(f"в ветви {len(vals)} номиналов")


def parse_chain(prompt):
    """Обратный разбор запроса цепочки v3 -> нетлист."""
    body = re.split(r"[;,]?\s*выход (?:с |снимаем|после|в конце)", prompt)[0]
    if "после " not in body and not any(body.lstrip().startswith(h) or f"; {h}" in body for h in LINE_HEAD):
        sections = []
        for seg in re.split(_SPLIT, body):
            vals = _values(seg)
            if not vals:
                continue
            if re.search(compose._GND_RE, seg):
                sections.append(_shunt_section(vals))
            elif len(vals) == 1:
                sections.append(("ser", vals[0][0], vals[0][1]))
            else:
                raise ValueError(f"не один номинал в разрыв: {seg}")
        return chain_netlist(sections)
    line, shunts = [], {}
    for block in body.split(";"):
        block = block.strip()
        head, _, rest = block.partition(": ")
        if head in LINE_HEAD:
            line = [_values(x)[0] for x in re.split(r"(?<!\d),|,(?!\d)", rest) if _values(x)]
            continue
        for item in re.split(r"(?<!\d),|,(?!\d)", rest):
            m = _AFTER.search(item.strip())
            if not m:
                raise ValueError(f"нет места ветви: {item}")
            shunts.setdefault(m.group(1), []).append(_shunt_section(_values(m.group(2))))
    sections = []
    for i, (k, v) in enumerate(line, 1):
        sections.append(("ser", k, v))
        keys = [f"{i}-го", ORDINALS[i - 1]] + (["последнего"] if i == len(line) else [])
        for key in keys:
            sections += shunts.pop(key, [])
    if shunts:
        raise ValueError(f"ветви без места: {sorted(shunts)}")
    return chain_netlist(sections)


def sample_chain(rng, branch=None, max_tries=2000):
    for _ in range(max_tries):
        sections = random_chain(rng, branch)
        ref, _why = compose._reference(chain_netlist(sections), "compose:chain", {"sections": sections}, band=True)
        if ref:
            return ref
    raise RuntimeError("цепочка не собрана")


# ---------------------------------------------------------------- добавки (П4-П6)
def op_phrase(op, vals, b, rng, explicit_out):
    """Фраза добавки; explicit_out=False - без слов о том, откуда снимать выход (П4)."""
    if explicit_out or op not in ("out_r", "out_rc", "buffer", "side"):
        return compose.op_phrase(op, vals, b, rng)
    r = b.value("R", compose._fmt(vals[0])) if vals else ""
    c = b.value("C", compose._fmt(vals[-1])) if op in ("out_rc", "side") else ""
    gnd, ser = rng.choice(compose.SINK), rng.choice(compose.SERIES)
    return rng.choice({
        "out_r": [f"после выхода {r} последовательно", f"последовательно с выходом {r}", f"на выходе {r} {ser}"],
        "out_rc": [f"после выхода {r} {ser} и {c} {gnd}", f"с выхода {r} {ser}, затем {c} {gnd}"],
        "buffer": ["на выходе повторитель на ОУ", "после схемы буфер на ОУ"],
        "side": [f"от выхода к отдельному узлу {r}, с этого узла {c} {gnd}"],
    }[op])


def describe_addon(ref, rng, explicit_out=None, reorder=None):
    """-> {"prompt", "explicit_out", "reordered"}: основа названа по ролям (gen 3), затем добавки."""
    b = describe_ru.Brief(rng)
    b.units = rng.random() < 0.5
    base = {"params": ref["params"]["base"], "class": ref["params"]["base_topology"]}
    head = describe_ru.describe_a1_brief(base, rng, 3) if rng.random() < 0.6 else describe_ru.describe_a1(base, rng, 3)
    head = head.rstrip(". ")
    explicit = (rng.random() >= NO_OUT_SHARE) if explicit_out is None else explicit_out
    tail = [op_phrase(op, vals, b, rng, explicit) for op, vals in ref["params"]["ops"]]
    moved = (rng.random() < REORDER) if reorder is None else reorder
    if not moved:
        return {"prompt": f"{head}; {'; '.join(tail)}", "explicit_out": explicit, "reordered": False}
    rng.shuffle(tail)
    k = rng.randint(1, len(tail))  # основа - после k фраз о добавках
    parts = tail[:k] + [rng.choice(HEAD_LAST) + head] + tail[k:]
    return {"prompt": "; ".join(parts), "explicit_out": explicit, "reordered": True}


def sample_addon(rng, bases=BASES, max_tries=500):
    """Как compose.sample_addon, но основы - все названные топологии, а полосовые схемы отсеиваются по Q."""
    for _ in range(max_tries):
        name = rng.choice(bases)
        base = filters_v2.sample_any(name, rng)
        lines = [l for l in base["netlist"].splitlines() if l and not l.startswith("V")]
        groups, want = {}, rng.randint(1, 3)
        for op in rng.sample(compose.OPS, len(compose.OPS)):
            groups.setdefault(compose.GROUP.get(op, op), op)
        ops = []
        for op in sorted(rng.sample(list(groups.values()), want), key=compose.OPS.index):
            res = compose._op(lines, op, rng)
            if res:
                lines, vals = res
                ops.append((op, vals))
        if not ops:
            continue
        src = next(e for e in canon.parse_netlist(base["netlist"])[0] if e["type"] == "V")
        netlist = "\n".join([f"V1 IN 0 DC {src['value']}"] + lines) + "\n"
        ref, _why = compose._reference(netlist, f"compose:{name}", {"base_topology": name, "base": base["params"], "ops": ops},
                                       band=True)
        if ref:
            return ref
    raise RuntimeError("добавка не собрана")


# ---------------------------------------------------------------- схемы на 2-4 ОУ (П2)
def _move_out(netlist, rng):
    """Выход - на случайном ОУ петли (у Tow-Thomas - полосовой, инвертированный и прямой ФНЧ)."""
    elems = canon.parse_netlist(netlist)[0]
    outs = sorted({e["nodes"][2] for e in elems if e["type"] == "OPAMP"})
    new = rng.choice(outs)
    if new == "OUT":
        return netlist, outs.index(new)
    swap = {"OUT": "T0", new: "OUT"}
    return canon.to_netlist([dict(e, nodes=[swap.get(n, n) for n in e["nodes"]]) for e in elems]), outs.index(new)


def _multi(rng):
    kind = rng.choice(["cascade", "cascade", "loop", "loop+stage"])
    if kind == "cascade":
        stages = [filters_v2.sample_any(rng.choice(STAGES), rng) for _ in range(rng.randint(2, 4))]
        if rng.random() < 0.3:  # пассивная секция между звеньями: ОУ следующего звена её нагружает
            stages.insert(rng.randint(1, len(stages) - 1), filters_v2.sample_any(rng.choice(["rc_lp", "rc_hp"]), rng))
        return cascade.compose(stages), {"kind": kind, "stages": [s["class"] for s in stages]}
    loop = filters_v2.sample_any(rng.choice(LOOPS), rng)
    net, out = _move_out(loop["netlist"], rng)
    names = [loop["class"]]
    if kind == "loop+stage":
        stage = filters_v2.sample_any(rng.choice(STAGES), rng)
        first = rng.random() < 0.5
        pair = [stage, dict(loop, netlist=net)] if first else [dict(loop, netlist=net), stage]
        net, names = cascade.compose(pair), [r["class"] for r in pair]
    return net, {"kind": kind, "stages": names, "out": out}


def sample_multi(rng, max_tries=500):
    """Схема на 2-4 ОУ без названия -> эталон (class compose:multi, уровень L3)."""
    for _ in range(max_tries):
        try:
            net, params = _multi(rng)
        except RuntimeError:
            continue
        ref, _why = compose._reference(net, "compose:multi", params, max_elements=MULTI_ELEMENTS, max_opamps=MULTI_OPAMPS,
                                       band=True)
        if ref and sum(l.startswith("X") for l in ref["netlist"].splitlines()) >= 2:
            return ref
    raise RuntimeError("схема на нескольких ОУ не собрана")


if __name__ == "__main__":
    import json
    import sys
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    a, b, m = sample_chain(rng), sample_addon(rng), sample_multi(rng)
    print(json.dumps({"chain": describe_chain(a["params"]["sections"], rng), "chain_net": a["netlist"],
                      "addon": describe_addon(b, rng), "addon_net": b["netlist"], "multi": m["params"],
                      "multi_net": m["netlist"]}, ensure_ascii=False, indent=1))
