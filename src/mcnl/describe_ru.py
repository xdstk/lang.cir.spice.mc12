#!/usr/bin/env python3
"""Формулировки запросов на русском (docs/generator_spec.md, раздел 4) и их обратный разбор.

Задача A (элементы перечислены) - по каноническому нетлисту, четыре семейства: list, table, narrative,
path. Задача A′ (task "A1", топология названа) - по параметрам и словарю ролей data/roles_ru.json.
Обратный разбор parse_a восстанавливает нетлист из запроса A; генератор принимает запрос, только если
разбор изоморфен метке по номиналам - так исключаются неоднозначные формулировки.

Устойчивые обороты (на них опирается разбор): узлы-якоря - «вход схемы», «выход схемы», «общий провод»
или «земля»; внутренние - «точка k» или «узел k»; выводы ОУ - «неинвертирующий (+) вход — …,
инвертирующий (−) вход — …, выход ОУ — …»; тип элемента однозначен по единице номинала.
"""
import json
import re
from pathlib import Path

try:
    from . import canon
except ImportError:  # запуск как скрипта
    import canon

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("list", "table", "narrative", "path")
ROLES = json.loads((ROOT / "data" / "roles_ru.json").read_text(encoding="utf-8"))["topologies"]

# падежи: им., род., дат., вин., твор.
CASES = ("nom", "gen", "dat", "acc", "ins")
ANCHOR_FORMS = {
    "IN": ("вход схемы", "входа схемы", "входу схемы", "вход схемы", "входом схемы"),
    "OUT": ("выход схемы", "выхода схемы", "выходу схемы", "выход схемы", "выходом схемы"),
}
GND_FORMS = {"провод": ("общий провод", "общего провода", "общему проводу", "общий провод", "общим проводом"),
             "земля": ("земля", "земли", "земле", "землю", "землёй")}
INNER_FORMS = {"точка": ("точка {}", "точки {}", "точке {}", "точку {}", "точкой {}"),
               "узел": ("узел {}", "узла {}", "узлу {}", "узел {}", "узлом {}")}
# тип: (им. п., вин. п., согласованное «включён», слово для номинала)
WORDS = {"R": ("резистор", "резистор", "включён", "сопротивлением"),
         "C": ("конденсатор", "конденсатор", "включён", "ёмкостью"),
         "L": ("катушка индуктивности", "катушку индуктивности", "включена", "индуктивностью")}


class Style:
    """Выбор вариантов, единый для одного запроса."""

    def __init__(self, rng):
        self.rng = rng
        self.gnd = rng.choice(list(GND_FORMS))
        self.inner = rng.choice(list(INNER_FORMS))
        self.comma = rng.random() < 0.7  # десятичная запятая чаще точки
        self.designators = rng.random() < 0.7

    def node(self, n, case):
        i = CASES.index(case)
        if n in ANCHOR_FORMS:
            return ANCHOR_FORMS[n][i]
        if n == "0":
            return GND_FORMS[self.gnd][i]
        return INNER_FORMS[self.inner][i].format(n[1:])

    def num(self, x):
        s = f"{x:.4g}"
        if "e" in s:
            s = f"{x:.10f}".rstrip("0").rstrip(".")
        return s.replace(".", ",") if self.comma else s

    def value(self, t, value):
        """Номинал в русской записи (решение В10): несколько привычных вариантов."""
        v = canon.parse_value(value)
        if t == "R":
            opts = [f"{self.num(v)} Ом"] if v < 1e3 else (
                [f"{self.num(v / 1e3)} кОм", f"{self.num(v / 1e3)} килоом"]
                + ([f"{int(round(v))} Ом"] if v < 1e5 and abs(v - round(v)) < 1e-9 else []) if v < 1e6 else
                [f"{self.num(v / 1e6)} МОм", f"{self.num(v / 1e6)} мегаом"])
        elif t == "C":
            if v < 1e-9:
                opts = [f"{self.num(v / 1e-12)} пФ"]
            elif v < 1e-6:
                opts = [f"{self.num(v / 1e-9)} нФ"] + ([f"{self.num(v / 1e-6)} мкФ"] if v >= 100e-9 else []) \
                    + ([f"{int(round(v / 1e-12))} пФ"] if v < 10e-9 else [])
            else:
                opts = [f"{self.num(v / 1e-6)} мкФ"]
        else:
            opts = [f"{self.num(v / 1e-6)} мкГн"] if v < 1e-3 else [f"{self.num(v / 1e-3)} мГн"] + (
                [f"{self.num(v / 1e-6)} мкГн"] if v < 10e-3 else [])
        return self.rng.choice(opts)

    def volts(self, value):
        return f"{self.num(canon.parse_value(value))} В"


def _cap(s):
    return s[:1].upper() + s[1:]


def _source_text(e, st, table=False):
    if table:
        val = "сигнал" if canon.parse_value(e["value"]) == 0 else st.volts(e["value"])
        return f"{e['name']} | {val} | плюс — {st.node('IN', 'nom')}, минус — {st.node('0', 'nom')}"
    if canon.parse_value(e["value"]) == 0:
        return st.rng.choice([
            f"На вход схемы подаётся сигнал (источник напряжения между {st.node('IN', 'ins')} и {st.node('0', 'ins')}).",
            f"Источник сигнала включён между {st.node('IN', 'ins')} и {st.node('0', 'ins')}, плюсом ко входу схемы."])
    return st.rng.choice([
        f"Источник постоянного напряжения {st.volts(e['value'])}: плюс — {st.node('IN', 'nom')}, минус — {st.node('0', 'nom')}.",
        f"Между {st.node('IN', 'ins')} и {st.node('0', 'ins')} включён источник напряжения {st.volts(e['value'])} (плюсом ко входу схемы)."])


def _opamp_text(e, st):
    des = f" {e['name']}" if st.designators else ""
    p, m, o = (st.node(n, "nom") for n in e["nodes"])
    return f"идеальный ОУ{des}: неинвертирующий (+) вход — {p}, инвертирующий (−) вход — {m}, выход ОУ — {o}"


def _label(e, st):
    return f" {e['name']}" if st.designators else ""


def describe_a(netlist, family, rng):
    """Запрос задачи A по каноническому нетлисту."""
    elems, err = canon.parse_netlist(netlist)
    if err:
        raise ValueError(err)
    st = Style(rng)
    src = [e for e in elems if e["type"] == "V"]
    two = [e for e in elems if e["type"] in "RCL"]
    ops = [e for e in elems if e["type"] == "OPAMP"]
    if family == "table":
        rows = ["Элемент | Номинал | Подключение"] + [_source_text(e, st, table=True) for e in src]
        for e in two:
            name = e["name"] if st.designators else WORDS[e["type"]][0]
            rows.append(f"{name} | {st.value(e['type'], e['value'])} | "
                        f"{st.node(e['nodes'][0], 'nom')} – {st.node(e['nodes'][1], 'nom')}")
        rows += [f"{e['name']} | — | {_opamp_text(e, st)}" for e in ops]
        return "\n".join(rows)
    parts = [_source_text(e, st) for e in src]
    if family == "list":
        items = []
        for e in two:
            a, b = (st.node(n, "ins") for n in e["nodes"])
            val = st.value(e["type"], e["value"])
            items.append(f"{e['name']} = {val} между {a} и {b}" if st.designators
                         else f"{WORDS[e['type']][0]} {val} между {a} и {b}")
        items += [_opamp_text(e, st) for e in ops]
        parts.append(rng.choice(["Схема: ", "Состав схемы: ", ""]) + "; ".join(items) + ".")
    elif family == "narrative":
        for e in two:
            w, w_acc, verb, mw = WORDS[e["type"]]
            a, b = e["nodes"]
            val = st.value(e["type"], e["value"])
            parts.append(rng.choice([
                f"Между {st.node(a, 'ins')} и {st.node(b, 'ins')} {verb} {w}{_label(e, st)} {mw} {val}.",
                f"{_cap(w)}{_label(e, st)} {val} {verb} между {st.node(a, 'ins')} и {st.node(b, 'ins')}.",
                f"{_cap(st.node(a, 'nom'))} и {st.node(b, 'nom')} соединены через {w_acc}{_label(e, st)} {val}."]))
        parts += [f"Используется {_opamp_text(e, st)}." for e in ops]
    elif family == "path":
        order = _path_order(two)
        for u, items in order:
            txt = []
            for e, v in items:
                to = f"на {st.node(v, 'acc')}" if v == "0" else f"к {st.node(v, 'dat')}"
                txt.append(f"{WORDS[e['type']][0]}{_label(e, st)} {st.value(e['type'], e['value'])} {to}")
            parts.append(f"От {st.node(u, 'gen')}: " + ", ".join(txt) + ".")
        parts += [_cap(_opamp_text(e, st)) + "." for e in ops]
    else:
        raise ValueError(f"неизвестное семейство {family}")
    return " ".join(parts)


def _path_order(two):
    """Группировка двухполюсников по узлам в порядке обхода от входа: каждый элемент - один раз,
    у первого встреченного узла (упрощённое семейство path, решение В4)."""
    seen, out, queue, visited = set(), [], ["IN"], {"IN"}
    nodes = {n for e in two for n in e["nodes"]}
    while queue or len(seen) < len(two):
        if not queue:  # оставшиеся элементы не связаны с входом через двухполюсники (например, через ОУ)
            queue.append(sorted(n for e in two if id(e) not in seen for n in e["nodes"])[0])
        u = queue.pop(0)
        items = []
        for e in two:
            if id(e) in seen or u not in e["nodes"]:
                continue
            seen.add(id(e))
            v = e["nodes"][1] if e["nodes"][0] == u else e["nodes"][0]
            items.append((e, v))
            if v not in visited and v != "0":
                visited.add(v)
                queue.append(v)
        if items:
            out.append((u, items))
    return out


def describe_a1(ref, rng):
    """Запрос задачи A′: топология названа словами, номиналы - по ролям (data/roles_ru.json)."""
    st = Style(rng)
    stages = ref["params"] if isinstance(ref["params"], list) else [dict(ref["params"], topology=ref["class"])]

    def stage_text(p):
        r = ROLES[p["topology"]]
        names = r.get("names_gain") if p.get("Ra") and r.get("names_gain") else r["names"]
        roles = [f"{r['roles'][k]} — {st.value(k[0], canon.format_value(v))}" for k, v in p.items()
                 if k in r["roles"]]
        return rng.choice(names) + (": " + ", ".join(roles) if roles else "")

    dc = canon.format_value(stages[0].get("dc", 0))
    src = _source_text({"name": "V1", "type": "V", "nodes": ["IN", "0"], "value": dc}, st)
    if len(stages) == 1:
        return _cap(stage_text(stages[0])) + ". " + src
    body = "; ".join(f"{i}) {stage_text(p)}" for i, p in enumerate(stages, 1))
    return (f"Каскад из {len(stages)} звеньев на идеальных ОУ, соединённых последовательно (выход каждого звена — "
            f"вход следующего): {body}. " + src)


# ---------------------------------------------------------------- обратный разбор задачи A
_NODE = (r"(вход(?:а|у|ом)? схемы|выход(?:а|у|ом)? схемы|общ(?:ий|его|ему|им) провод(?:а|у|ом)?|"
         r"земл(?:я|и|е|ю|ёй|ей)|(?:точк(?:а|и|е|у|ой)|узл(?:а|у|ом)|узел) (\d+))")
NODE_RE = re.compile(_NODE, re.IGNORECASE)  # «Точка 2 и выход схемы соединены …» - с заглавной
VALUE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(килоом|мегаом|мкГн|мГн|Гн|пФ|нФ|мкФ|Ф|кОм|МОм|Ом)(?![а-яё])")
UNITS = {"килоом": ("R", 1e3), "мегаом": ("R", 1e6), "кОм": ("R", 1e3), "МОм": ("R", 1e6), "Ом": ("R", 1),
         "пФ": ("C", 1e-12), "нФ": ("C", 1e-9), "мкФ": ("C", 1e-6), "Ф": ("C", 1),
         "мкГн": ("L", 1e-6), "мГн": ("L", 1e-3), "Гн": ("L", 1)}
OPAMP_RE = re.compile(r"неинвертирующий \(\+\) вход — " + _NODE + r", инвертирующий \(−\) вход — " + _NODE
                      + r", выход ОУ — " + _NODE)


def _node_id(m_text, num):
    m_text = m_text.lower()
    if m_text.startswith("вход"):
        return "IN"
    if m_text.startswith("выход"):
        return "OUT"
    if m_text.startswith(("общ", "земл")):
        return "0"
    return f"N{num}"


def _nodes_in(text):
    return [_node_id(m.group(1), m.group(2)) for m in NODE_RE.finditer(text)]


def parse_a(prompt):
    """Запрос задачи A -> элементы (canon). Бросает ValueError, если запрос не разбирается однозначно."""
    elems = []
    text = prompt
    for m in OPAMP_RE.finditer(text):
        g = m.groups()
        elems.append({"type": "OPAMP", "name": f"X{len(elems) + 1}", "value": None,
                      "nodes": [_node_id(g[0], g[1]), _node_id(g[2], g[3]), _node_id(g[4], g[5])]})
    text = OPAMP_RE.sub(" ", text)
    segments = []
    for line in text.split("\n"):
        if line.startswith("Элемент | Номинал"):
            continue
        for seg in re.split(r"(?<=[.;])\s+|;\s*", line):
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"От (.+?): (.*)$", seg)
            if m:  # path: узел из заголовка + узел в каждом элементе
                head = _nodes_in(m.group(1))
                if len(head) != 1:
                    raise ValueError(f"заголовок path без узла: {seg}")
                segments += [(head, item) for item in m.group(2).rstrip(".").split(", ")]
            else:
                segments.append(([], seg))
    src = None
    for head, seg in segments:
        if "источник" in seg.lower() or "сигнал" in seg or seg.startswith("V1 |"):
            vm = re.search(r"(-?\d+(?:[.,]\d+)?) В(?![а-яё])", seg)
            src = "0" if vm is None else canon.format_value(float(vm.group(1).replace(",", ".")))
            continue
        vals = VALUE_RE.findall(seg)
        if not vals:
            continue
        if len(vals) != 1:
            raise ValueError(f"в фрагменте не один номинал: {seg}")
        num, unit = vals[0]
        t, mult = UNITS[unit]
        nodes = head + _nodes_in(seg)
        if len(nodes) != 2:
            raise ValueError(f"у двухполюсника не два узла: {seg}")
        elems.append({"type": t, "name": f"{t}{len(elems) + 1}", "nodes": nodes,
                      "value": canon.format_value(float(num.replace(",", ".")) * mult)})
    if src is None:
        raise ValueError("не найден источник")
    return [{"type": "V", "name": "V1", "nodes": ["IN", "0"], "value": src}] + elems


if __name__ == "__main__":
    import random
    import sys
    net = open(sys.argv[1], encoding="utf-8").read()
    for fam in FAMILIES:
        print(f"--- {fam}\n{describe_a(net, fam, random.Random(1))}\n")
