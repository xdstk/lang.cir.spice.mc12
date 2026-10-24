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
    "Opamp": ("X", "MODEL", ["Plus input", "Minus input", "VEE", "Output", "VCC"]),
    "Subckt": ("X", "NAME", None),
}
IGNORED = {"Ground"}
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
            for l in L:
                k, v = kv(l)
                if k == "Text":
                    textlines.append(v)
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
def convert(text, title="circuit"):
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
    by_label = {}
    for lab, p in label_pts:
        if on_wire(p):
            by_label.setdefault(lab, []).append(p)
    for lab, pts in by_label.items():
        for p in pts[1:]:
            dsu.union(pts[0], p)
    for g in gtexts:
        t = g["text"]
        if len(t) == 1 and t[0].strip() in by_label:
            continue
        if t and t[0].startswith("."):
            textlines.extend(t)
        else:
            comments.append(" ".join(x.strip() for x in t))
    # земля
    ground_roots = {dsu.find(p) for c, _, p in pins_of if c.get("name") == "Ground"}
    for lab, pts in by_label.items():
        names[dsu.find(pts[0])] = re.sub(r"\s+", "_", lab)
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
    pul = {}
    opa = {}
    for m in re.finditer(r"\.MODEL\s+(\S+)\s+(PUL|OPA)\s*\(([^)]*)\)", models_text, re.I):
        params = dict((k.upper(), v) for k, v in re.findall(r"(\w+)\s*=\s*([^\s)]+)", m.group(3)))
        (pul if m.group(2).upper() == "PUL" else opa)[m.group(1).upper()] = params

    lines, subckts = [f"* {title}  (converted from Micro-Cap .CIR by cir2spice prototype)"], {}
    for cm in comments:
        lines.append("* " + cm)
    pins_by_comp = {}
    for c, pname, pt in pins_of:
        pins_by_comp.setdefault(id(c), {})[pname] = pt
    for c in comps:
        cname = c.get("name")
        d = compdefs.get(cname)
        if d is None or cname in IGNORED:
            continue
        definition = d.get("definition", "")
        dev = DEVICES.get(definition)
        part = c["attrs"].get("PART", "") or "U?"
        if dev is None:
            errors.append(f"{part}: примитив '{definition}' не поддерживается прототипом")
            continue
        letter, vattr, order = dev
        pmap = pins_by_comp.get(id(c), {})
        order = order or [p[0] for p in d["pins"]]
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
            lines.append(f"{name} {nodes[0]} {nodes[1]} PULSE({f('VZERO','0')} {f('VONE','5')} {t[0]} "
                         f"{fmt(tv[1]-tv[0])} {fmt(tv[3]-tv[2])} {fmt(tv[2]-tv[1])} {t[4]})")
        elif definition == "Opamp":
            p = opa.get(val.upper(), {})
            lev = p.get("LEVEL", "1")
            sub = f"OPA_{val.upper()}_L1EQ"
            a, rout = p.get("A", "2E5"), p.get("ROUTDC", "125")
            subckts[sub] = (f".SUBCKT {sub} INP INM VEE OUT VCC\n"
                            f"* level-1 equivalent (VCCS + Rout), see MC12 Reference Manual, OPAMP\n"
                            f"G1 0 OUT INP INM {{{a}/{rout}}}\nROUT OUT 0 {rout}\n.ENDS {sub}")
            if lev != "1":
                warn.append(f"{part}: OPAMP LEVEL={lev} заменён эквивалентом уровня 1 (нет SPICE-аналога)")
            lines.append(f"{name} {' '.join(nodes)} {sub}")
        elif letter == "X":
            lines.append(f"{name} {' '.join(nodes)} {val}")
            warn.append(f"{part}: подсхема {val} не включена; MC12 найдёт её через неявный "
                        f".LIB NOM.LIB, другой симулятор - нет")
        else:
            model = c["attrs"].get("MODEL", "") if vattr != "MODEL" else ""
            lines.append(" ".join(x for x in [name, *nodes, model, val] if x))
    # текстовая область без моделей PUL/OPA (они преобразованы)
    for tl in textlines:
        if re.match(r"\s*\.MODEL\s+\S+\s+(PUL|OPA)\b", tl, re.I):
            lines.append("* (converted above) " + tl)
        elif tl.strip():
            lines.append(tl)
    lines += list(subckts.values())
    # анализы
    printed = {"Transient": [], "AC": []}
    for w in waves:
        an, y = w.get("Analysis"), w.get("YExp", "")
        mm = re.fullmatch(r"(?:dB\()?\s*(V\([^()]+\))\s*\)?", y.strip(), re.I)
        if an in printed and mm and mm.group(1) not in printed[an]:
            printed[an].append(mm.group(1))
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
