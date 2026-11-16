#!/usr/bin/env python3
"""Схемы справочника по ГОСТ (решение автора 17.11.2026): рисунок из символов, свой вывод в окно (tkinter Canvas) и в SVG.

Символы (размеры в мм, по ГОСТ 2.728-74, 2.723-68, 2.721-74, 2.759-82 - пропорции, не точные допуски):
резистор - прямоугольник 10 x 4; конденсатор - две обкладки; катушка - четыре полуокружности; ОУ - прямоугольник
со знаком ▷, инвертирующий вход - с кружком; заземление - три убывающие черты; вывод - кружок; узел - точка.
Номиналы не пишутся (схема в общем виде), только обозначения элементов.

Рисунок (Sketch) строится по координатам в мм (y - вниз). Связность рисунка (провода, выводы, земля, обозначения)
восстанавливается (connectivity) и сверяется с нетлистом статьи (same_circuit) - так тест проверяет, что рисунок
изображает ту же схему с теми же обозначениями.

Запуск:  python src/mcnl/schematic.py --svg папка     (SVG всех схем справочника - для рисунков в тексте работы)
"""
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import canon  # noqa: E402

LINE = 0.35        # толщина линий, мм (основная линия схем ГОСТ - 0,2...1 мм; на экране - пропорционально)
R_LEN, R_W = 10.0, 4.0
C_GAP, C_PLATE = 2.0, 7.0
L_TURNS, L_R = 4, 1.5
OA_W, OA_H, OA_LEAD = 15.0, 20.0, 5.0
TERM_R, DOT_R = 1.0, 0.8
FONT = 3.2          # высота шрифта обозначений, мм


class Sketch:
    """Рисунок: примитивы для вывода и элементы для проверки связности."""

    def __init__(self):
        self.prims = []      # ("line", x1, y1, x2, y2) | ("rect", x1, y1, x2, y2) | ("circle", cx, cy, r, fill)
        self.wires = []      # (точка, точка) - отрезки проводов
        self.parts = []      # (тип, имя, [точки выводов]) - для ОУ: [+, -, выход]
        self.marks = []      # (точка, узел) - выводы IN, OUT и земля 0

    # --- примитивы
    def _line(self, a, b):
        self.prims.append(("line", a[0], a[1], b[0], b[1]))

    def _text(self, p, s, anchor="c", size=FONT):
        self.prims.append(("text", p[0], p[1], s, anchor, size))

    # --- провода и выводы
    def wire(self, *pts):
        for a, b in zip(pts, pts[1:]):
            if a[0] != b[0] and a[1] != b[1]:
                raise ValueError(f"провод не по сетке: {a} - {b}")
            self._line(a, b)
            self.wires.append((a, b))
        return self

    def dot(self, p):
        self.prims.append(("circle", p[0], p[1], DOT_R, True))
        return self

    def terminal(self, p, node, text, side="left"):
        """Вывод двухполюсника (кружок) - узел IN или OUT; подпись слева или справа."""
        self.prims.append(("circle", p[0], p[1], TERM_R, False))
        self.marks.append((p, node))
        dx = -2.5 if side == "left" else 2.5
        self._text((p[0] + dx, p[1] - 2.5), text, "e" if side == "left" else "w")
        return self

    def ground(self, p, side="down"):
        """Заземление (ГОСТ 2.721): отвод и три убывающие черты; side - куда смотрит символ."""
        x, y = p
        s = 1 if side in ("down", "right") else -1
        if side in ("down", "up"):
            self._line((x, y), (x, y + s * 3))
            for i, w in enumerate((3.0, 2.0, 1.0)):
                yy = y + s * (3 + i * 1.0)
                self._line((x - w, yy), (x + w, yy))
        else:
            self._line((x, y), (x + s * 3, y))
            for i, w in enumerate((3.0, 2.0, 1.0)):
                xx = x + s * (3 + i * 1.0)
                self._line((xx, y - w), (xx, y + w))
        self.marks.append((p, "0"))
        return self

    def rail(self, y, x1, x2, text_in="", text_out=""):
        """Общий провод двухполюсника: выводы слева и справа, узел 0."""
        self.wire((x1, y), (x2, y))
        for x, side in ((x1, "left"), (x2, "right")):
            self.prims.append(("circle", x, y, TERM_R, False))
            self.marks.append(((x, y), "0"))
        if text_in:
            self._text((x1 - 2.5, y - 2.5), text_in, "e")
        if text_out:
            self._text((x2 + 2.5, y - 2.5), text_out, "w")

    def label(self, p, s, anchor="c"):
        self._text(p, s, anchor)

    # --- двухполюсники
    def two(self, kind, name, a, b, side=None):
        """Резистор, конденсатор или катушка между выводами a и b (по горизонтали или вертикали), корпус посередине.
        side - где подпись: above / below (горизонтальный), left / right (вертикальный)."""
        (x1, y1), (x2, y2) = a, b
        horizontal = y1 == y2
        if not horizontal and x1 != x2:
            raise ValueError(f"{name}: выводы не на одной линии")
        length = abs(x2 - x1) if horizontal else abs(y2 - y1)
        body = {"R": R_LEN, "C": C_GAP, "L": 2 * L_R * L_TURNS}[kind]
        if length < body + 2:
            raise ValueError(f"{name}: мало места ({length} мм)")
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ux, uy = ((1 if x2 > x1 else -1), 0) if horizontal else (0, (1 if y2 > y1 else -1))
        h = body / 2
        s1 = (cx - ux * h, cy - uy * h)
        s2 = (cx + ux * h, cy + uy * h)
        self._line(a, s1)
        self._line(s2, b)
        if kind == "R":
            hw = R_W / 2
            if horizontal:
                self.prims.append(("rect", cx - h, cy - hw, cx + h, cy + hw))
            else:
                self.prims.append(("rect", cx - hw, cy - h, cx + hw, cy + h))
            off = hw + 2.5
        elif kind == "C":
            hp = C_PLATE / 2
            for s in (s1, s2):
                if horizontal:
                    self._line((s[0], s[1] - hp), (s[0], s[1] + hp))
                else:
                    self._line((s[0] - hp, s[1]), (s[0] + hp, s[1]))
            off = hp + 2.0
        else:  # катушка: полуокружности выпуклостью вверх (горизонтальная) или вправо (вертикальная)
            start = min(s1[0], s2[0]) if horizontal else min(s1[1], s2[1])
            for i in range(L_TURNS):
                c = start + L_R * (2 * i + 1)
                self.prims.append(("arc", c if horizontal else cx, cy if horizontal else c, L_R,
                                   0 if horizontal else 270, 180))
            off = L_R + 2.0
        side = side or ("above" if horizontal else "right")
        pos = {"above": (cx, cy - off - 1.2), "below": (cx, cy + off + 1.2),
               "left": (cx - off - 0.8, cy), "right": (cx + off + 0.8, cy)}[side]
        self._text(pos, name, {"above": "s", "below": "n", "left": "e", "right": "w"}[side])
        self.parts.append((kind, name, [a, b]))
        return self

    def R(self, name, a, b, side=None):
        return self.two("R", name, a, b, side)

    def C(self, name, a, b, side=None):
        return self.two("C", name, a, b, side)

    def L(self, name, a, b, side=None):
        return self.two("L", name, a, b, side)

    # --- ОУ
    def opamp(self, name, x, y, inv_top=True, power=False, label="above", mirror=False):
        """ОУ по ГОСТ 2.759: прямоугольник OA_W x OA_H с верхним левым углом (x, y), знак ▷; входы слева (кружок -
        указатель инверсии на инвертирующем входе), выход справа; mirror - входы справа, выход слева (знак ◁).
        -> {"plus", "minus", "out"} - концы выводов. power - выводы питания."""
        self.prims.append(("rect", x, y, x + OA_W, y + OA_H))
        # знак усилителя ▷ (◁ у отражённого) - линиями: глиф шрифта на экране выходит цветным значком
        tx, ty, a = x + OA_W / 2, y + 4, 1.6
        tip, back = (tx + a, tx - a) if not mirror else (tx - a, tx + a)
        self._line((back, ty - a), (back, ty + a))
        self.prims.append(("line", back, ty - a, tip, ty))
        self.prims.append(("line", back, ty + a, tip, ty))
        edge_in, edge_out = (x + OA_W, x) if mirror else (x, x + OA_W)
        s = 1 if mirror else -1  # куда от корпуса идут входы
        top, bot = (edge_in - s * -OA_LEAD, y + 5), (edge_in - s * -OA_LEAD, y + OA_H - 5)
        out = (edge_out - s * OA_LEAD, y + OA_H / 2)
        minus, plus = (top, bot) if inv_top else (bot, top)
        self._line(plus, (edge_in, plus[1]))
        self._line(out, (edge_out, out[1]))
        # указатель инверсии - кружок на инвертирующем входе у корпуса
        self.prims.append(("circle", edge_in + s * 0.9, minus[1], 0.9, False))
        self._line(minus, (edge_in + s * 1.8, minus[1]))
        if power:
            cx = x + OA_W / 2
            self._line((cx, y), (cx, y - 4))
            self._line((cx, y + OA_H), (cx, y + OA_H + 4))
            self._text((cx + 1.5, y - 4), "+15 В", "w", FONT * 0.85)
            self._text((cx + 1.5, y + OA_H + 4), "−15 В", "w", FONT * 0.85)
        lp = {"above": (x + OA_W / 2, y - 1.5), "below": (x + OA_W / 2, y + OA_H + 1.5),
              "right": (x + OA_W + 1.5, y + 2.5)}[label]
        self._text(lp, name, {"above": "s", "below": "n", "right": "w"}[label])
        self.parts.append(("OPAMP", name, [plus, minus, out]))
        return {"plus": plus, "minus": minus, "out": out}

    # --- размеры
    def bbox(self):
        xs, ys = [], []
        for p in self.prims:
            k = p[0]
            if k in ("line", "rect"):
                xs += [p[1], p[3]]
                ys += [p[2], p[4]]
            elif k in ("circle", "arc"):
                xs += [p[1] - p[3], p[1] + p[3]]
                ys += [p[2] - p[3], p[2] + p[3]]
            elif k == "text":
                w = len(p[3]) * p[5] * 0.6
                x0 = {"e": p[1] - w, "w": p[1]}.get(p[4], p[1] - w / 2) if p[4] in ("e", "w", "c", "s", "n") else p[1]
                xs += [x0, x0 + w]
                ys += [p[2] - p[5], p[2] + p[5]]
        return min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2


# ---------------------------------------------------------------- связность рисунка
def _on_segment(p, a, b):
    (x, y), (x1, y1), (x2, y2) = p, a, b
    if x1 == x2 == x:
        return min(y1, y2) <= y <= max(y1, y2)
    if y1 == y2 == y:
        return min(x1, x2) <= x <= max(x1, x2)
    return False


def connectivity(sk):
    """Рисунок -> {точка вывода: класс узла}: провода соединяют свои концы, а конец провода, вывод или метка, лежащие
    на проводе, соединяются с ним (Т-образное примыкание). Пересечение проводов без конца в точке - не соединение."""
    points = set()
    for a, b in sk.wires:
        points |= {a, b}
    for _, _, pins in sk.parts:
        points |= set(pins)
    for p, _ in sk.marks:
        points.add(p)
    parent = {p: p for p in points}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in sk.wires:
        union(a, b)
        for p in points:
            if _on_segment(p, a, b):
                union(p, a)
    for p, node in sk.marks:  # одноимённые метки соединены (земля в разных местах - один узел)
        for q, n2 in sk.marks:
            if n2 == node:
                union(p, q)
    return {p: find(p) for p in points}


def node_signatures(elems):
    """Схема -> мультимножество узлов, каждый - множество (обозначение, вывод) и метка IN/OUT/0. Двухполюсники без
    порядка выводов, ОУ - с выводами +, -, выход. Совпадение множеств = та же схема с теми же обозначениями."""
    nodes = {}
    for e in elems:
        if e["type"] == "OPAMP":
            roles = ("+", "-", "out")
        elif e["type"] in "VI":
            roles = ("p", "n")
        else:
            roles = ("", "")
        for n, r in zip(e["nodes"], roles):
            nodes.setdefault(n, set()).add(f"{e['name']}{r}")
    for n in ("IN", "OUT", "0"):
        if n in nodes:
            nodes[n].add(f"@{n}")
    return sorted(sorted(s) for s in nodes.values())


def drawn_elements(sk):
    """Элементы рисунка с узлами из связности; вход и земля дают источник V1 IN 0, как в нетлисте статьи."""
    cls = connectivity(sk)
    names = {}
    for p, node in sk.marks:
        names[cls[p]] = node
    seq = iter(f"N{i}" for i in range(1, 100))
    for c in sorted(set(cls.values())):
        names.setdefault(c, next(seq))
    out = [{"type": t, "name": n, "nodes": [names[cls[p]] for p in pins]} for t, n, pins in sk.parts]
    return out + [{"type": "V", "name": "V1", "nodes": ["IN", "0"]}]


def same_circuit(sk, netlist):
    """Рисунок изображает схему нетлиста с теми же обозначениями элементов -> (да/нет, рисунок, нетлист)."""
    ref = canon.parse_netlist(netlist)[0]
    drawn = drawn_elements(sk)
    a, b = node_signatures(drawn), node_signatures(ref)
    types_ok = {e["name"]: e["type"] for e in drawn} == {e["name"]: e["type"] for e in ref}  # символ того же типа
    return a == b and types_ok, a, b


# ---------------------------------------------------------------- вывод
def draw_tk(canvas, sk, mm=3.6, pad=6, color="#1d1d1b"):
    """Рисунок на холсте tkinter; mm - пикселей на мм. Размер холста - по рисунку."""
    x0, y0, x1, y1 = sk.bbox()
    sx = lambda x: (x - x0) * mm + pad  # noqa: E731
    sy = lambda y: (y - y0) * mm + pad  # noqa: E731
    canvas.configure(width=int((x1 - x0) * mm + 2 * pad), height=int((y1 - y0) * mm + 2 * pad))
    w = max(1.0, LINE * mm)
    for p in sk.prims:
        k = p[0]
        if k == "line":
            canvas.create_line(sx(p[1]), sy(p[2]), sx(p[3]), sy(p[4]), fill=color, width=w, capstyle="projecting")
        elif k == "rect":
            canvas.create_rectangle(sx(p[1]), sy(p[2]), sx(p[3]), sy(p[4]), outline=color, width=w, fill="white")
        elif k == "circle":
            r = p[3] * mm
            canvas.create_oval(sx(p[1]) - r, sy(p[2]) - r, sx(p[1]) + r, sy(p[2]) + r, outline=color, width=w,
                               fill=color if p[4] else "white")
        elif k == "arc":
            r = p[3] * mm
            canvas.create_arc(sx(p[1]) - r, sy(p[2]) - r, sx(p[1]) + r, sy(p[2]) + r, start=p[4], extent=p[5],
                              style="arc", outline=color, width=w)
        elif k == "text":
            anchor = {"c": "center", "s": "s", "n": "n", "e": "e", "w": "w"}[p[4]]
            canvas.create_text(sx(p[1]), sy(p[2]), text=p[3], anchor=anchor, fill=color,
                               font=("Segoe UI", -max(9, int(p[5] * mm))))


def to_svg(sk, mm=3.6, pad=6):
    """Рисунок в SVG (для рисунков в тексте работы)."""
    x0, y0, x1, y1 = sk.bbox()
    sx = lambda x: round((x - x0) * mm + pad, 2)  # noqa: E731
    sy = lambda y: round((y - y0) * mm + pad, 2)  # noqa: E731
    W, H = int((x1 - x0) * mm + 2 * pad), int((y1 - y0) * mm + 2 * pad)
    w = round(max(1.0, LINE * mm), 2)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<g stroke="black" stroke-width="{w}" fill="none" font-family="Arial, sans-serif">']
    for p in sk.prims:
        k = p[0]
        if k == "line":
            out.append(f'<line x1="{sx(p[1])}" y1="{sy(p[2])}" x2="{sx(p[3])}" y2="{sy(p[4])}" stroke-linecap="square"/>')
        elif k == "rect":
            out.append(f'<rect x="{sx(p[1])}" y="{sy(p[2])}" width="{round((p[3] - p[1]) * mm, 2)}" '
                       f'height="{round((p[4] - p[2]) * mm, 2)}" fill="white"/>')
        elif k == "circle":
            out.append(f'<circle cx="{sx(p[1])}" cy="{sy(p[2])}" r="{round(p[3] * mm, 2)}" '
                       f'fill="{"black" if p[4] else "white"}"/>')
        elif k == "arc":
            cx, cy, r, start, ext = p[1], p[2], p[3], p[4], p[5]
            a1, a2 = math.radians(start), math.radians(start + ext)
            # tkinter: углы против часовой стрелки от оси x, ось y вниз - точки дуги (cos, -sin)
            xa, ya = cx + r * math.cos(a1), cy - r * math.sin(a1)
            xb, yb = cx + r * math.cos(a2), cy - r * math.sin(a2)
            out.append(f'<path d="M {sx(xa)} {sy(ya)} A {round(r * mm, 2)} {round(r * mm, 2)} 0 0 0 {sx(xb)} {sy(yb)}"/>')
        elif k == "text":
            anchor = {"c": "middle", "s": "middle", "n": "middle", "e": "end", "w": "start"}[p[4]]
            base = {"c": "central", "s": "auto", "n": "hanging", "e": "central", "w": "central"}[p[4]]
            text = p[3].replace("&", "&amp;").replace("<", "&lt;")
            out.append(f'<text x="{sx(p[1])}" y="{sy(p[2])}" font-size="{round(p[5] * mm, 1)}" text-anchor="{anchor}" '
                       f'dominant-baseline="{base}" stroke="none" fill="black">{text}</text>')
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if "--svg" in sys.argv:
        from mcnl.handbook_schematics import SCHEMATICS
        d = Path(sys.argv[sys.argv.index("--svg") + 1])
        d.mkdir(parents=True, exist_ok=True)
        for aid, fn in SCHEMATICS.items():
            (d / f"{aid}.svg").write_text(to_svg(fn()), encoding="utf-8")
        print(f"-> {d} ({len(SCHEMATICS)} схем)")
    else:
        sys.exit(__doc__)
