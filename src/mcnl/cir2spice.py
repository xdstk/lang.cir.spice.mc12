#!/usr/bin/env python3
"""Прототип конвертера схем Micro-Cap (.CIR, текстовый формат MC10-MC12) в SPICE-нетлист.

Запуск:  python cir2spice.py <файл.cir | папка> [--out папка] [--report report.json]

Опирается на правила из MC12 Reference Manual (гл. 2, "Wires", "Node number assignment"):
  * провода соединяются концом к концу, концом к выводу элемента, концом к внутренней
    точке другого провода; пересечение внутренних точек НЕ соединяет;
  * вывод элемента, лежащий на проводе, соединяется с ним;
  * текст (Grid Text), чей якорь лежит на узле, именует узел; узлы с одинаковым
    именем соединены; Ground - узел 0.
Кодировка поворота Rot=0..7 восстановлена по примерам (Rot k+4 = поворот k, затем y -> -y)
и проверена только на выводах, лежащих на оси x; для многовыводных повёрнутых элементов
её нужно сверить с экспортом самого MC12 (File > Translate > Schematic to SPICE Text File).

Ограничения прототипа: только аналоговые примитивы из таблицы DEVICES; цифровые части,
макросы и примитивы без SPICE-аналога (OPAMP уровней 2-3) помечаются в отчёте.
"""
import json
import math
import os
import re
import sys
from pathlib import Path

GRID_PX = 8  # в MC12 координаты Px - пиксели, 1 шаг сетки = 8 px (Grid= в MC10 - шаги)

# Definition -> (буква SPICE, атрибут значения, порядок выводов по именам или None = как в compdef)
DEVICES = {
    "Resistor": ("R", "RESISTANCE", ["Plus", "Minus"]),
    "Capacitor": ("C", "CAPACITANCE", ["Plus", "Minus"]),
    "Inductor": ("L", "INDUCTANCE", ["Plus", "Minus"]),
    "Battery": ("V", "VALUE", ["Plus", "Minus"]),
    "VSpice": ("V", "VALUE", ["Plus", "Minus"]),
    "ISpice": ("I", "VALUE", ["Plus", "Minus"]),
    "Diode": ("D", "MODEL", ["Anode", "Cathode"]),
    "NPN": ("Q", "MODEL", ["Collector", "Base", "Emitter"]),
    "PNP": ("Q", "MODEL", ["Collector", "Base", "Emitter"]),
    "NMOS": ("M", "MODEL", ["Drain", "Gate", "Source", "Bulk"]),
    "PMOS": ("M", "MODEL", ["Drain", "Gate", "Source", "Bulk"]),
    "Pulse source": ("V", "MODEL", ["Plus", "Minus"]),
    "Sine source": ("V", "MODEL", ["Plus", "Minus"]),
    "S": ("S", "MODEL", ["Plus Output", "Minus Output", "Plus Control", "Minus Control"]),
    "NFV": ("E", "VALUE", ["Plus", "Minus"]),
    "Opamp": ("X", "MODEL", ["Plus input", "Minus input", "VEE", "Output", "VCC"]),
    "Subckt": ("X", "NAME", None),
}
IGNORED = {"Ground"}
# идеальный ОУ канона (docs/eval_spec.md, раздел 4; то же, что bench.OPAMP_DEF)
OPAMP_IDEAL_DEF = ".SUBCKT OPAMP_IDEAL PLUS MINUS OUT\nE1 OUT 0 PLUS MINUS 1E6\n.ENDS OPAMP_IDEAL"
# Tie (и "Output Port" с тем же Definition): соединитель по имени. В руководствах не описан;
# по примерам DATA все Tie с одинаковым PART образуют один узел с этим именем.
CONNECTORS = {"Tie"}
# другие имена тех же выводов в compdef: в примерах DATA MC12 вывод подложки MOSFET - "Body"
PIN_ALIASES = {"Bulk": ["Body"]}


# ---------------------------------------------------------------- разбор файла
def parse_sections(text):
    secs, cur = [], None
    for raw in text.replace("\r", "").split("\n"):
        line = raw.rstrip()
        m = re.match(r"^\[(.+)\]$", line)
        if m:
            cur = {"_name": m.group(1), "_lines": []}
            secs.append(cur)
        elif cur is not None and line:
            cur["_lines"].append(line)
    return secs


def kv(line):
    if "=" in line:
        k, v = line.split("=", 1)
        return k.strip(), v
    return line.strip(), ""


def nums(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def to_grid(sec_lines, single_px="Px", single_grid="Grid"):
    for l in sec_lines:
        k, v = kv(l)
        if k == single_px:
            return [x / GRID_PX for x in nums(v)]
        if k == single_grid:
            return nums(v)
    return None


def parse_cir(text):
    secs = parse_sections(text)
    compdefs, comps, wires, gtexts, textlines, limits, waves = {}, [], [], [], [], [], []
    owner = None
    for s in secs:
        n, L = s["_name"], s["_lines"]
        if n == "compdef":
            d = {"pins": []}
            for l in L:
                k, v = kv(l)
                if k == "Name":
                    d["name"] = v
                elif k == "Definition":
                    d["definition"] = v
                elif k == "Pin":
                    m = re.match(r'"([^"]*)"\s*,(.*)', v)
                    if m:
                        coords = [t.strip() for t in m.group(2).split(",")]
                        coords = [t for t in coords if re.fullmatch(r"-?\d+", t)]
                        d["pins"].append((m.group(1), float(coords[0]), float(coords[1])))
            compdefs[d.get("name")] = d
            owner = None
        elif n == "Comp":
            c = {"attrs": {}, "rot": 0}
            for l in L:
                k, v = kv(l)
                if k == "Name":
                    c["name"] = v
                elif k == "Rot":
                    c["rot"] = int(v)
            c["pos"] = to_grid(L)
            comps.append(c)
            owner = c
        elif n == "Attr":
            on, val = None, None
            for l in L:
                k, v = kv(l)
                if k == "ON":
                    on = v.split(",")[-1].strip()
                elif k == "V":
                    val = v
                elif k == "ONV":  # старый формат: ONV=x,y,ИМЯ,ЗНАЧЕНИЕ (в значении бывают запятые)
                    parts = v.split(",", 3)
                    if len(parts) == 4:
                        on, val = parts[2].strip(), parts[3]
            if owner is not None and on:
                owner["attrs"][on] = val if val is not None else ""
        elif n == "Wire":
            for l in L:
                k, v = kv(l)
                if k in ("Pxs", "Grids"):
                    p = nums(v)
                    if k == "Pxs":
                        p = [x / GRID_PX for x in p]
                    wires.append(((p[0], p[1]), (p[2], p[3])))
            owner = None
        elif n == "Grid Text":
            t = [kv(l)[1].strip('"') for l in L if kv(l)[0] == "Text"]
            gtexts.append({"text": t, "pos": to_grid(L)})
            owner = None
        elif n == "Text Area":
            # новый формат: строки Text=...; старый: служебные ключи, Len=N, затем сырой текст
            raw = False
            for l in L:
                k, v = kv(l)
                if raw:
                    textlines.append(l)
                elif k == "Text":
                    textlines.append(v)
                elif k == "Len":
                    raw = True
            owner = None
        elif n == "Limits":
            limits.append(dict(kv(l) for l in L))
            owner = None
        elif n == "WaveForm":
            waves.append(dict(kv(l) for l in L))
            owner = None
        else:
            owner = None
    return compdefs, comps, wires, gtexts, textlines, limits, waves


# ---------------------------------------------------------------- геометрия
def rotate(x, y, rot):
    k, mirror = rot % 4, rot >= 4
    x, y = [(x, y), (-y, x), (-x, -y), (y, -x)][k]
    if mirror:
        y = -y
    return x, y


def on_segment(p, a, b, eps=1e-6):
    (px, py), (ax, ay), (bx, by) = p, a, b
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps:
        return False
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps) and (min(ay, by) - eps <= py <= max(ay, by) + eps)


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def key(pt):
    return (round(pt[0], 3), round(pt[1], 3))


# ---------------------------------------------------------------- конвертация
SUBCKT_RE = re.compile(r"^[ \t]*\.SUBCKT[ \t]+(\S+)([^\n]*(?:\n[ \t]*\+[^\n]*)*)", re.I | re.M)


def subckt_nodes(text):
    """{ИМЯ: [[узлы], ...]} для всех .SUBCKT в тексте; узлы - до PARAMS:/TEXT:/OPTIONAL:."""
    found = {}
    for m in SUBCKT_RE.finditer(text):
        nodes = []
        for t in m.group(2).replace("\n", " ").replace("+", " ").split():
            if t.upper().rstrip(":") in ("PARAMS", "TEXT", "OPTIONAL") or "=" in t or t.startswith(";"):
                break
            nodes.append(t)
        found.setdefault(m.group(1).upper(), []).append(nodes)
    return found


_LIB_INDEX = {}


def library_index(lib_dir):
    """Индекс .SUBCKT библиотеки MC12 в порядке поиска: файлы из NOM.LIB (рекурсивно), затем остальные .lib.
    MC12 ищет подсхему схемы в тексте, затем через неявный .LIB NOM.LIB (RM12 с. 748)."""
    lib_dir = Path(lib_dir)
    if lib_dir in _LIB_INDEX:
        return _LIB_INDEX[lib_dir]
    order, seen = [], set()

    def visit(name):
        p = lib_dir / name
        if p.name.lower() in seen or not p.is_file():
            return
        seen.add(p.name.lower())
        order.append(p)
        if p.suffix.lower() == ".lib":
            for inc in re.findall(r'^\s*\.LIB\s+"?([^"\s]+)"?', p.read_text("cp1251", "replace"), re.I | re.M):
                visit(inc)

    visit("NOM.LIB")
    for p in sorted(lib_dir.glob("*.lib")):
        visit(p.name)
    index = {}
    for p in order:
        for k, v in subckt_nodes(p.read_text("cp1251", "replace")).items():
            index.setdefault(k, []).extend(v)
    _LIB_INDEX[lib_dir] = index
    return index


def default_lib_dir():
    d = Path(os.environ.get("MC12_LIBRARY", r"C:\MC12\LIBRARY"))
    return d if d.is_dir() else None


def subckt_pin_order(d, subname, models_text, warn, part, lib_dir):
    """Порядок выводов в вызове X: узлы вызова идут в порядке оператора .SUBCKT (RM12 с. 748),
    а выводы в compdef названы узлами подсхемы (RM12 с. 749). Порядок выводов в compdef бывает
    другим: у PWMCM (power456.lib) он 3,4,6,5,2,1 при .SUBCKT PWMCM 1 2 3 4 5 6."""
    pins = [p[0] for p in d["pins"]]
    by_up = {p.upper(): p for p in pins}
    candidates = subckt_nodes(models_text).get(subname.upper(), [])
    if lib_dir:
        candidates = candidates + library_index(lib_dir).get(subname.upper(), [])
    for nodes in candidates:
        if sorted(n.upper() for n in nodes) == sorted(by_up):
            return [by_up[n.upper()] for n in nodes]
    warn.append(f"{part}: .SUBCKT {subname} с выводами {pins} не найден ни в схеме, ни в библиотеке; "
                f"выводы в порядке compdef (может не совпасть с MC12)")
    return pins


def convert(text, title="circuit", lib_dir="default"):
    """lib_dir: каталог библиотеки MC12 для порядка выводов подсхем; "default" - MC12_LIBRARY
    или C:\\MC12\\LIBRARY, если есть; None - не искать."""
    if lib_dir == "default":
        lib_dir = default_lib_dir()
    compdefs, comps, wires, gtexts, textlines, limits, waves = parse_cir(text)
    warn, errors = [], []
    dsu = DSU()
    pins_of = []  # (comp, pinname, point)
    for c in comps:
        d = compdefs.get(c.get("name"))
        if d is None or c.get("pos") is None:
            errors.append(f"нет описания компонента {c.get('name')}")
            continue
        for pname, px, py in d["pins"]:
            dx, dy = rotate(px, py, c["rot"])
            pt = key((c["pos"][0] + dx, c["pos"][1] + dy))
            pins_of.append((c, pname, pt))
            dsu.find(pt)
    # точки интереса: концы проводов, выводы, якоря текста
    points = {key(a) for w in wires for a in w} | {p for _, _, p in pins_of}
    label_pts = []
    for g in gtexts:
        if g["pos"] is not None and len(g["text"]) == 1 and not g["text"][0].startswith("."):
            label_pts.append((g["text"][0].strip(), key(g["pos"])))
    points |= {p for _, p in label_pts}
    for a, b in wires:
        wa = key(a)
        dsu.union(wa, key(b))
        for p in points:
            if on_segment(p, a, b):
                dsu.union(p, wa)
    # метки узлов: только если якорь действительно лежит на узле
    node_points = {key(a) for w in wires for a in w} | {p for _, _, p in pins_of}
    on_wire = lambda p: p in node_points or any(on_segment(p, a, b) for a, b in wires)
    names, comments = {}, []
    by_label = {}  # имя узла без учёта регистра -> точки
    for lab, p in label_pts:
        if on_wire(p):
            by_label.setdefault(lab.upper(), []).append(p)
    label_case = {lab.upper(): lab for lab, _ in label_pts}
    for c, _, p in pins_of:
        if compdefs[c["name"]].get("definition") in CONNECTORS:
            lab = c["attrs"].get("PART", "").strip()
            if not lab:
                errors.append(f"{c['name']} без имени (PART): соединение неизвестно")
                continue
            by_label.setdefault(lab.upper(), []).append(p)
            label_case.setdefault(lab.upper(), lab)
    for lab, pts in by_label.items():
        for p in pts[1:]:
            dsu.union(pts[0], p)
    for g in gtexts:
        t = g["text"]
        if len(t) == 1 and t[0].strip().upper() in by_label:
            continue
        if t and t[0].startswith("."):
            textlines.extend(t)
        else:
            comments.append(" ".join(x.strip() for x in t))
    # земля
    ground_roots = {dsu.find(p) for c, _, p in pins_of if c.get("name") == "Ground"}
    for lab, pts in by_label.items():
        names[dsu.find(pts[0])] = re.sub(r"\s+", "_", label_case[lab])
    counter = [0]

    def node(pt):
        r = dsu.find(pt)
        if r in ground_roots:
            return "0"
        if r not in names:
            counter[0] += 1
            names[r] = str(counter[0])
        return names[r]

    # модели из текстовой области
    models_text = "\n".join(textlines)
    sch_models = {"PUL": {}, "OPA": {}, "SIN": {}}  # модели, у которых нет SPICE-синтаксиса
    for m in re.finditer(r"\.MODEL\s+(\S+)\s+(PUL|OPA|SIN)\s*\(([^)]*)\)", models_text, re.I):
        params = dict((k.upper(), v) for k, v in re.findall(r"(\w+)\s*=\s*([^\s)]+)", m.group(3)))
        sch_models[m.group(2).upper()][m.group(1).upper()] = params
    pul, opa, sin = sch_models["PUL"], sch_models["OPA"], sch_models["SIN"]

    lines, subckts = [f"* {title}  (converted from Micro-Cap .CIR by cir2spice prototype)"], {}
    for cm in comments:
        lines.append("* " + cm)
    pins_by_comp = {}
    for c, pname, pt in pins_of:
        pins_by_comp.setdefault(id(c), {})[pname] = pt
    for c in comps:
        cname = c.get("name")
        d = compdefs.get(cname)
        if d is None or cname in IGNORED or d.get("definition") in CONNECTORS:
            continue
        definition = d.get("definition", "")
        dev = DEVICES.get(definition)
        part = c["attrs"].get("PART", "") or "U?"
        if dev is None:
            errors.append(f"{part}: примитив '{definition}' не поддерживается прототипом")
            continue
        letter, vattr, order = dev
        pmap = pins_by_comp.get(id(c), {})
        if order is None:
            order = subckt_pin_order(d, c["attrs"].get(vattr, ""), models_text, warn, part, lib_dir)
        order = [next((a for a in PIN_ALIASES.get(p, []) if a in pmap and p not in pmap), p) for p in order]
        try:
            nodes = [node(pmap[p]) for p in order]
        except KeyError as e:
            errors.append(f"{part}: нет вывода {e}")
            continue
        name = part if part.upper().startswith(letter) else letter + "_" + part
        val = c["attrs"].get(vattr, "")
        if definition == "Pulse source":
            p = pul.get(val.upper())
            if p is None:
                errors.append(f"{part}: нет .MODEL {val} PUL")
                continue
            f = lambda k, dflt: p.get(k, dflt)
            t = [f("P1", "100n"), f("P2", "110n"), f("P3", "500n"), f("P4", "510n"), f("P5", "1u")]
            tv = [spice_num(x) for x in t]
            # Pulse и Sine source дают в AC-анализе фиксированные 1 В (UG12 с. 100, RM12 с. 220)
            # как пишет File > Translate самого MC12: AC и DC до функции, иначе при запуске
            # MC12 сообщает "Extra text 'AC' found" (проверено пакетным прогоном); DC = VZERO
            lines.append(f"{name} {nodes[0]} {nodes[1]} AC 1 DC {f('VZERO','0')} PULSE({f('VZERO','0')} "
                         f"{f('VONE','5')} {t[0]} {fmt(tv[1]-tv[0])} {fmt(tv[3]-tv[2])} {fmt(tv[2]-tv[1])} {t[4]})")
        elif definition == "Sine source":
            # RM12 с. 745: V = A*exp(-T/TAU)*sin(2*pi*F*TIME + PH) + DC, T = TIME mod RP, PH в радианах
            p = sin.get(val.upper())
            if p is None:
                errors.append(f"{part}: нет .MODEL {val} SIN")
                continue
            f = lambda k, dflt: p.get(k, dflt)
            tau, rp = spice_num(f("TAU", "0")), spice_num(f("RP", "0"))
            theta = fmt(1 / tau) if tau else "0"
            if tau and rp:
                warn.append(f"{part}: SIN RP={f('RP', '0')} (повтор затухания) в SPICE SIN не выражается, опущен")
            phase = fmt(math.degrees(spice_num(f("PH", "0"))))
            # RS (по умолчанию 1 мОм) - последовательный резистор со стороны плюса через внутренний
            # узел, как в выводе File > Translate самого MC12 (V1 INTERIOR 0 SIN(...); RV1 1 INTERIOR RS)
            rs, plus = f("RS", "0.001"), nodes[0]
            if spice_num(rs):
                plus = f"{name}_RS"
                lines.append(f"R{name} {nodes[0]} {plus} {rs}")
            lines.append(f"{name} {plus} {nodes[1]} AC 1 SIN({f('DC', '0')} {f('A', '1')} {f('F', '1E6')} 0 "
                         f"{theta} {phase})")
        elif definition == "NFV":
            # RM12 с. 562: в SPICE-файле MC12 формула задаётся как E<name> + - VALUE = {expr}
            if not val.strip():
                errors.append(f"{part}: пустая формула NFV")
                continue
            lines.append(f"{name} {nodes[0]} {nodes[1]} VALUE = {{{val.strip()}}}")
            if c["attrs"].get("FREQ", "").strip():
                warn.append(f"{part}: атрибут FREQ ({c['attrs']['FREQ']}) не переносится")
        elif definition == "Opamp" and val.upper() == "OPAMP_IDEAL":
            # схема, построенная spice2cir: идеальный ОУ канона (3 вывода), питание не подключено
            subckts["OPAMP_IDEAL"] = OPAMP_IDEAL_DEF
            lines.append(f"{name} {nodes[0]} {nodes[1]} {nodes[3]} OPAMP_IDEAL")
        elif definition == "Opamp":
            p = opa.get(val.upper(), {})
            lev = p.get("LEVEL", "1")
            sub = f"OPA_{val.upper()}_L1EQ"
            # RM12 с. 728 (умолчания), с. 732: R = ROUTAC+ROUTDC, GM = A/R; RIN и резисторы питания -
            # как в подсхеме уровня 1, которую выдаёт File > Translate самого MC12
            a = spice_num(p.get("A", "2E5"))
            rout = spice_num(p.get("ROUTAC", "75")) + spice_num(p.get("ROUTDC", "125"))
            subckts[sub] = (f".SUBCKT {sub} INP INM VEE OUT VCC\n"
                            f"RSUPPLUS VCC 0 1\nRSUPMIN VEE 0 1\nROUT OUT 0 {fmt(rout)}\n"
                            f"GOUT 0 OUT INP INM {fmt(a / rout)}\nRIN INP INM 1G\n.ENDS {sub}")
            if lev != "1":
                warn.append(f"{part}: OPAMP LEVEL={lev} заменён эквивалентом уровня 1 "
                            f"(MC12 Translate разворачивает его в подсхему; здесь не реализовано)")
            lines.append(f"{name} {' '.join(nodes)} {sub}")
        elif letter == "X":
            lines.append(f"{name} {' '.join(nodes)} {val}")
            warn.append(f"{part}: подсхема {val} не включена; MC12 найдёт её через неявный "
                        f".LIB NOM.LIB, другой симулятор - нет")
        else:
            if letter in "RCL" and not val.strip():
                errors.append(f"{part}: нет значения {vattr}")
                continue
            v = val.strip()
            if letter in "RCL" and " " not in v and not v.startswith("{") and not re.fullmatch(
                    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:E[+-]?\d+)?[A-Z]*", v, re.I):
                val = "{" + v + "}"  # выражение (10K-R(R1)) - в фигурных скобках, как у MC12 Translate
            model = c["attrs"].get("MODEL", "") if vattr != "MODEL" else ""
            lines.append(" ".join(x for x in [name, *nodes, model, val] if x))
    # текстовая область без моделей PUL/OPA/SIN (они преобразованы) вместе с их строками-продолжениями
    in_converted = False
    for tl in textlines:
        if re.match(r"\s*\.MODEL\s+\S+\s+(PUL|OPA|SIN)\b", tl, re.I):
            in_converted = True
            lines.append("* (converted above) " + tl)
        elif in_converted and tl.lstrip().startswith("+"):
            lines.append("* (converted above) " + tl)
        elif tl.strip():
            in_converted = False
            lines.append(tl)
    lines += list(subckts.values())
    # анализы; в .PRINT только напряжения узлов, которые есть в этом нетлисте: графики схемы
    # ссылаются на номера узлов MC12, а они с нашей нумерацией не совпадают
    known = {n.upper() for n in names.values()} | {"0"}
    printed = {"Transient": [], "AC": []}
    for w in waves:
        an, y = w.get("Analysis"), w.get("YExp", "")
        mm = re.fullmatch(r"(?:dB\()?\s*(V\(([^()]+)\))\s*\)?", y.strip(), re.I)
        if an not in printed or not mm or mm.group(1) in printed[an]:
            continue
        if all(n.strip().upper() in known for n in mm.group(2).split(",")):
            printed[an].append(mm.group(1))
        else:
            warn.append(f"{an}: график {mm.group(1)} ссылается на узел, которого нет в нетлисте; в .PRINT не включён")
    # анализ выводим, только если в схеме для него заданы графики
    used = {w.get("Analysis") for w in waves if w.get("YExp", "").strip()}
    for lim in limits:
        an = lim.get("Analysis")
        if an not in used:
            continue
        # диапазон времени: TMax= в файлах старых версий, TRange= в новых ("tmax[,tstart]")
        trange = lim.get("TMax") or lim.get("TRange")
        if an == "Transient" and trange:
            tmax = trange.split(",")[0].strip()
            step = lim.get("TStep", "0")
            step = step if spice_num(step) else fmt(spice_num(tmax) / 1000)
            lines.append(f".TRAN {step} {tmax}")
            if printed["Transient"]:
                lines.append(".PRINT TRAN " + " ".join(printed["Transient"]))
        if an == "AC" and "FRange" in lim:
            sweep = lim.get("Sweep type", "Auto")
            fr = [x.strip() for x in lim["FRange"].split(",")]
            if sweep == "List" or len(fr) != 2:
                warn.append(f"AC: развёртка '{sweep}' с FRange={lim['FRange']} не переведена в .AC")
                continue
            fmax, fmin = fr
            npts = int(spice_num(lim.get("NPts", "101")))  # бывает с суффиксом: NPts=1k
            if sweep == "Linear":
                lines.append(f".AC LIN {max(1, npts)} {fmin} {fmax}")
            else:
                dec = max(1e-9, math.log10(spice_num(fmax) / spice_num(fmin)))
                lines.append(f".AC DEC {max(1, math.ceil(npts / dec))} {fmin} {fmax}")
            if printed["AC"]:
                lines.append(".PRINT AC " + " ".join(printed["AC"]))
    lines.append(".END")
    n_comp = sum(1 for c in comps if c.get('name') not in IGNORED)
    if n_comp == 0:
        warn.append("в схеме нет компонентов (текстовый файл или страница без чертежа)")
    return "\n".join(lines) + "\n", {"warnings": warn, "errors": errors,
                                     "n_components": n_comp,
                                     "n_nodes": len(set(names.values()))}


SUF = {"T": 1e12, "G": 1e9, "MEG": 1e6, "K": 1e3, "M": 1e-3, "U": 1e-6, "N": 1e-9, "P": 1e-12, "F": 1e-15}


def spice_num(s):
    m = re.match(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)", str(s))
    if not m:
        return 0.0
    v, suf = float(m.group(1)), m.group(2).upper()
    for k in ("MEG", "T", "G", "K", "M", "U", "N", "P", "F"):
        if suf.startswith(k):
            return v * SUF[k]
    return v


def fmt(x):
    return f"{x:.6g}"


def main(argv):
    src = Path(argv[1])
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    rep_path = argv[argv.index("--report") + 1] if "--report" in argv else None
    # на Windows glob не различает регистр: без set каждый файл попадал бы дважды
    files = sorted({f for f in src.iterdir() if f.suffix.lower() == ".cir"}) if src.is_dir() else [src]
    report = {}
    for f in files:
        text = f.read_text(encoding="cp1251", errors="replace")
        try:
            net, info = convert(text, f.stem)
        except Exception as e:  # один сломанный файл не должен останавливать прогон по папке
            report[f.name] = {"warnings": [], "errors": [f"сбой конвертера: {type(e).__name__}: {e}"],
                              "n_components": 0, "n_nodes": 0}
            print(f"# {f.name}: CRASH {type(e).__name__}: {e}", file=sys.stderr)
            continue
        report[f.name] = info
        if out:
            out.mkdir(parents=True, exist_ok=True)
            (out / (f.stem + ".ckt")).write_text(net, encoding="cp1251", errors="replace")
        else:
            print(net)
        print(f"# {f.name}: {json.dumps(info, ensure_ascii=False)}", file=sys.stderr)
    if rep_path:
        Path(rep_path).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
