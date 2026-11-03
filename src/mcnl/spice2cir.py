#!/usr/bin/env python3
"""Нетлист класса A -> схема Micro-Cap (.CIR), которую пользователь открывает в MC12 (решение В7).

Запуск:  python spice2cir.py нетлист.ckt схема.cir [--fchar 1591]

Раскладка по пути сигнала (_flow_layout): главная линия IN -> OUT слева направо, источник слева, элементы на землю
и боковые цепочки висят под своим узлом, ОУ стоит в разрыве линии (вход сигнала сверху, при сигнале на минус ОУ
отражён), обратные связи - перемычки над линией, делитель обратной связи - под ОУ. Схема вне её правил (несколько
источников, ОУ вне главной линии) раскладывается прежней сеткой (_grid_layout). Узлы соединяются проводами
(трассировка по сетке - route.py); цепь, которую развести не удалось, соединяется метками узлов (Grid Text на
выводах, RM12 с. 45-46, 67). Узлы IN и OUT
всегда подписаны: по их именам строятся анализ и график. На узел 0 ставится символ земли, направленный от
элемента наружу; элементы, провода, метки и земли не перекрываются. ОУ - элемент Opamp с моделью OPAMP_IDEAL
уровня 1: A = 1e6, R = ROUTAC + ROUTDC = 1 Ом (RM12 с. 728, 732), выводы питания не подключены
(у уровня 1 они не влияют на передачу). Описания фигур и элементов - data/cir_templates.cir.
В схему кладутся текст с исходным нетлистом и анализ AC (диапазон обвязки bench) с графиком dB(V(OUT)).
Файл - ASCII с CRLF (CLAUDE.md).
"""
import math
import re
import sys
from collections import namedtuple
from pathlib import Path

try:
    from . import bench, canon, route
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import route

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "data" / "cir_templates.cir"
OPAMP_MODEL = ".MODEL OPAMP_IDEAL OPA (LEVEL=1 A=1E6 ROUTAC=0.5 ROUTDC=0.5)"
COMP = {"R": ("Resistor", "RESISTANCE"), "C": ("Capacitor", "CAPACITANCE"), "L": ("Inductor", "INDUCTANCE"),
        "V": ("Voltage Source", "VALUE"), "OPAMP": ("Opamp", "MODEL")}
# выводы в шагах сетки при Rot=0 (data/cir_templates.cir): минус (0,0), плюс (6,0)
TWO_PIN = {"minus": (0, 0), "plus": (6, 0)}
OPAMP_PINS = {"plus": (0, 0), "minus": (0, 6), "out": (9, 3), "vee": (4, 7), "vcc": (4, -1)}
ROT_UP = 3          # (x, y) -> (y, -x): плюс источника над минусом
# Раскладка, шаги сетки. Метка узла привязана за нижний левый угол и тянется вправо-вверх, поэтому к выводам слева
# и снизу ведёт отвод STUB (провод), а метка стоит на его конце - иначе она легла бы на сам элемент.
X0, STUB, STUB_DOWN = 20, 3, 2
SERIES_DX, SERIES_DY, SERIES_COLS = 16, 10, 4
SHUNT_DX, SHUNT_DY, SHUNT_COLS = 11, 14, 6
OPAMP_DX, SOURCE_X, SOURCE_DY = 24, 6, 14
GROUND_ROT = {"right": 0, "down": 1, "left": 2, "up": 3}  # линия земли (12,0) px, поворот как в rotate
GROUND_LEN = 3
# подписи PART и значения, px от точки привязки (верхний левый угол текста): у горизонтального - под элементом,
# у вертикального - справа (как у резисторов в примерах DATA), у ОУ - справа снизу, мимо земель питания
ATTRS = {"h": ((18, 6), (18, 20)), "down": ((8, 11), (8, 25)), "up": ((8, -37), (8, -23)),
         "source": ((14, -37), (14, -23)), "opamp": ((76, 40), (76, 54)),
         # раскладка по пути сигнала: у перемычки над линией подписи сверху, у ОУ - справа от земли питания над ним
         "hb": ((18, -34), (18, -20)), "opamp_t": ((44, -32), (44, -20)), "opamp_f": ((44, -80), (44, -68))}
# Раскладка по пути сигнала, шаги сетки: главная линия IN -> OUT на высоте 0; элементы на землю и боковые цепочки
# висят под своим узлом (верхний вывод на HANG_TOP ниже линии, следующее звено цепочки - на HANG_DY ниже);
# ОУ стоит в разрыве линии выходом на ней; делитель обратной связи - под ОУ на FB_Y; остальное - перемычки над линией.
FLOW_DX, HANG_TOP, HANG_DY, FB_Y = 10, 5, 8, 11
PATH_STEPS = 20000  # предел перебора путей IN -> OUT
OPAMP_GAP = 16  # от точки привязки ОУ до следующего узла линии: подпись ОУ не должна лечь на провод к узлу
Placed = namedtuple("Placed", "e x y rot pins kind ends")  # ends: (точка метки, направление наружу) на вывод
# Чертёж: раскладка и всё, что к ней дорисовано. wires: (a, b, цепь); labels: (текст, точка, номер элемента);
# grounds: (точка, направление, номер элемента); routed: цепи, соединённые проводами.
Drawing = namedtuple("Drawing", "placed wires labels grounds routed")
NAMED = ("IN", "OUT")  # подписываются всегда


def ground_at(pin, end, d):
    """-> (точка земли, направление). Земля у вывода смотрит наружу; у входов ОУ (отвод влево) земля ставится на
    конце отвода и смотрит вниз: повёрнутая влево, она читается хуже."""
    return (end, "down") if d == "left" else (pin, d)
# Текст с нетлистом: MC12 привязывает многострочный Grid Text за нижний левый угол и поле JustifyV=Top в нашем
# файле не принимает (проверено 03.11.2026), поэтому блок ставится справа от схемы с учётом его высоты.
# Строка шрифта по умолчанию (Verdana 12) - около 2 шагов сетки, знак - около 0,9 шага.
TEXT_LINE, TEXT_CHAR, TEXT_GAP = 2, 0.9, 6
# Масштаб окна схемы: поля Scale 0..3 страницы Main - сколько единиц чертежа на пиксель, меньше 1 - крупнее
# (проверено 03.11.2026; [Schematic] Scale MC12 при загрузке не читает). Принимаются только шаги масштаба MC12:
# 0.5, 2/3 и 2 сработали, 0.6, 0.95 и 1.5 MC12 молча заменил на 1. VIEW - видимая часть окна при масштабе 1 в шагах
# сетки от края содержимого, замер на ноутбуке автора (развёрнутое окно MC12, 4,5 px на шаг на снимке 0,6). Берётся самый крупный шаг, при котором всё помещается.
VIEW_W, VIEW_H = 155, 60
SCALES = (2 / 3, 1, 2)  # 0.5 тоже работает, но маленькая схема при нём непомерно крупная


def rotate(x, y, rot):
    """Как в cir2spice: Rot 0-3 - поворот на 90 градусов * k ((x, y) -> (-y, x) за шаг)."""
    for _ in range(rot % 4):
        x, y = -y, x
    return x, y


def _comp(name, x, y, rot, attrs):
    s = [f"[Comp]\nName={name}\nGrid={x},{y}\nRot={rot}"]
    for (ox, oy), key, val in attrs:
        s.append(f"[Attr]\nON={ox},{oy},{key}\nV={val}\nStyle=8192")
    return "\n\n".join(s)


def _text(lines, x, y):
    body = "\n".join(f'Text="{t}"' for t in lines)
    return f"[Grid Text]\n{body}\nGrid={x},{y}\nCBorder=None"


def _attr_texts(e):
    value = OPAMP_MODEL.split()[1] if e["type"] == "OPAMP" else (
        _source_value(e) if e["type"] in "VI" else e["value"])
    return e["name"], value


def _text_box(x, y, text, top=False):
    """Прямоугольник текста в шагах сетки: знак - TEXT_CHAR, строка - 1,5 шага; Grid Text привязан снизу слева."""
    w = TEXT_CHAR * len(text)
    return (x, y, x + w, y + 1.5) if top else (x, y - 1.5, x + w, y)


def _element_boxes(placed):
    """Корпуса и подписи элементов: [(вид, номер элемента, прямоугольник)]."""
    out = []
    for i, p in enumerate(placed):
        e, x, y = p.e, p.x, p.y
        if p.kind in ("h", "hb"):
            out.append(("body", i, (x + 0.5, y - 0.75, x + 5.5, y + 0.75)))
        elif p.kind in ("down", "up"):
            y0, y1 = sorted(py for _, py in p.pins)
            out.append(("body", i, (x - 0.75, y0 + 0.5, x + 0.75, y1 - 0.5)))
        elif p.kind == "source":
            out.append(("body", i, (x - 1.5, y - 5, x + 1.5, y - 1)))
        elif p.rot == 4:  # ОУ, отражённый по вертикали: минус сверху
            out.append(("body", i, (x + 1, y - 6.5, x + 8, y + 0.5)))
        else:
            out.append(("body", i, (x + 1, y - 0.5, x + 8, y + 6.5)))
        for (ox, oy), text in zip(ATTRS[p.kind], _attr_texts(e)):
            out.append(("text", i, _text_box(x + ox / 8, y + oy / 8, text, top=True)))
    return out


def _ground_box(pt, d):
    gx, gy = pt
    return {"right": (gx, gy - 1, gx + GROUND_LEN, gy + 1), "down": (gx - 1, gy, gx + 1, gy + GROUND_LEN),
            "left": (gx - GROUND_LEN, gy - 1, gx, gy + 1), "up": (gx - 1, gy - GROUND_LEN, gx + 1, gy)}[d]


def _points_in(r, margin):
    """Целые точки сетки внутри прямоугольника, расширенного на margin."""
    return {(x, y) for x in range(math.floor(r[0] - margin), math.ceil(r[2] + margin) + 1)
            for y in range(math.floor(r[1] - margin), math.ceil(r[3] + margin) + 1)
            if r[0] - margin < x < r[2] + margin and r[1] - margin < y < r[3] + margin}


def _segment_points(a, b):
    (x0, y0), (x1, y1) = a, b
    if x0 == x1:
        return [(x0, y) for y in range(min(y0, y1), max(y0, y1) + 1)]
    return [(x, y0) for x in range(min(x0, x1), max(x0, x1) + 1)]


def draw(placed, wires=True):
    """Дорисовывает к раскладке земли, провода и метки -> Drawing. wires=False - только метки, как раньше."""
    grounds, gwires, labels = [], [], []
    nets = {}  # цепь -> [(номер элемента, вывод, точка метки, направление)]
    for i, p in enumerate(placed):
        for node, pin, (end, d) in zip(p.e["nodes"], p.pins, p.ends):
            if node == "0":
                gpt, gd = ground_at(pin, end, d)
                grounds.append((gpt, gd, i))
                if gpt != pin:
                    gwires.append((pin, gpt, "0"))
            else:
                nets.setdefault(node, []).append((i, pin, end, d))
    pins = {pin for p in placed for pin in p.pins}
    if len(pins) != sum(len(p.pins) for p in placed):
        raise AssertionError("совпали точки выводов")  # иначе выводы замкнулись бы

    def label_at(net, terms):
        """Точка подписи цепи: вывод, где метка не ложится на элемент (справа или сверху), иначе конец отвода."""
        for i, pin, end, d in terms:
            if d in ("right", "up") and end == pin:
                return i, pin, []
        i, pin, end, d = terms[0]
        return i, end, ([(pin, end, net)] if end != pin else [])

    eb = _element_boxes(placed)
    blocked = set()
    for kind, _, r in eb:
        blocked |= _points_in(r, 0.3 if kind == "body" else 0.45)
    for gpt, gd, _ in grounds:
        blocked |= _points_in(_ground_box(gpt, gd), 0.3)
    named, under = {}, {}
    spec = getattr(placed, "labels", {})  # раскладка по пути сигнала ставит метки сама
    for net in NAMED:
        if net in nets:
            if net in spec:
                i, pt, stub = spec[net]
                named[net] = (i, pt, [(a, b, net) for a, b in stub])
            else:
                named[net] = label_at(net, nets[net])
            pt = named[net][1]
            box = _points_in(_text_box(pt[0], pt[1], net), 0.45)
            under[net] = {q for q in box if q[1] == pt[1]}  # по строке привязки может идти провод своей цепи
            blocked |= box - under[net]
    allb = [r for *_, r in eb] + [_ground_box(g, d) for g, d, _ in grounds]
    bounds = (math.floor(min(r[0] for r in allb)) - 8, math.floor(min(r[1] for r in allb)) - 6,
              math.ceil(max(r[2] for r in allb)) + 8, math.ceil(max(r[3] for r in allb)) + 8)
    grid = route.Grid(bounds, blocked)
    for p in placed:
        for node, pin in zip(p.e["nodes"], p.pins):
            grid.owner[pin] = node
    for a, b, net in gwires:
        for q in _segment_points(a, b):
            grid.owner[q] = net
    for gpt, gd, _ in grounds:
        grid.owner[gpt] = "0"
    out_wires = list(gwires)
    start = {}
    for net, (i, pt, stub) in named.items():
        grid.owner[pt] = net
        for q in under[net]:
            grid.owner.setdefault(q, net)
        for a, b, _ in stub:
            for q in _segment_points(a, b):
                grid.owner[q] = net
                start.setdefault(net, set()).add(q)
        out_wires += stub
        labels.append((net, pt, i))
    routed = set()
    order = sorted(nets, key=lambda n: (max(t[1][0] for t in nets[n]) - min(t[1][0] for t in nets[n])
                                        + max(t[1][1] for t in nets[n]) - min(t[1][1] for t in nets[n])))
    for net in order if wires else []:
        terms = [t[1] for t in nets[net]]
        segs = route.route_net(grid, net, terms, start.get(net, ()))
        if segs is not None:
            routed.add(net)
            out_wires += [(a, b, net) for a, b in segs]
    for net, terms in nets.items():
        if net in routed:
            continue
        for i, pin, end, d in terms:  # метки на каждом выводе, как до трассировки
            if net in named and named[net][1] in (pin, end):
                continue
            ok = end == pin or all(grid.free_for(q, net) for q in _segment_points(pin, end))
            if ok and end != pin:
                out_wires.append((pin, end, net))
                for q in _segment_points(pin, end):
                    grid.owner[q] = net
            labels.append((net, end if ok else pin, i))
    return Drawing(placed, _split_at_joints(out_wires, pins), labels, grounds, routed)


def _split_at_joints(wires, pins):
    """Режет провода в точках, где к ним примыкает конец другого провода или вывод той же цепи: при загрузке файла
    MC12 соединяет провода только концами (Т-образное примыкание к середине провода в файле не соединилось:
    MC12 предупреждал о висящих узлах, проверено 03.11.2026)."""
    ends = {}
    for a, b, net in wires:
        ends.setdefault(net, set()).update((a, b))
    out = []
    for a, b, net in wires:
        pts = _segment_points(a, b)
        cut = [q for q in pts[1:-1] if q in ends[net] or q in pins]
        if not cut:
            out.append((a, b, net))
            continue
        marks = [pts[0]] + cut + [pts[-1]]
        out += [(marks[k], marks[k + 1], net) for k in range(len(marks) - 1)]
    return out


def _drawing(x):
    return x if isinstance(x, Drawing) else draw(x)


def boxes(drawing):
    """-> [(вид, владелец, (x0, y0, x1, y1))] - что занимает место на чертеже: корпуса, подписи, земли, провода
    и метки. Владелец - номер элемента или имя цепи для проводов. Нужно для проверки наложений и границ."""
    d = _drawing(drawing)
    out = _element_boxes(d.placed)
    out += [("ground", i, _ground_box(pt, gd)) for pt, gd, i in d.grounds]
    out += [("wire", net, (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
            for a, b, net in d.wires]
    out += [("label", i, _text_box(pt[0], pt[1], text)) for text, pt, i in d.labels]
    return out


def extent(drawing):
    """Правый и нижний край чертежа в шагах сетки."""
    b = [r for *_, r in boxes(drawing)]
    return math.ceil(max(r[2] for r in b)), math.ceil(max(r[3] for r in b))


def overlaps(drawing, margin=0.1):
    """Что накладывается: корпуса, подписи, земли и метки не задевают друг друга, провод не идёт по корпусу,
    подписи или земле. Провод касается корпуса и земли только у вывода; метка стоит на своём проводе или выводе.
    Касание по краю (в пределах margin) не считается."""
    d = _drawing(drawing)
    b = boxes(d)

    def name(k, o):
        return (k, o if k == "wire" else d.placed[o].e["name"])

    def inside(r, p):  # точка строго внутри прямоугольника
        return r[0] + margin < p[0] < r[2] - margin and r[1] + margin < p[1] < r[3] - margin

    hit = []
    for j, (k1, o1, r1) in enumerate(b):
        for k2, o2, r2 in b[j + 1:]:
            if "wire" in (k1, k2):
                if k1 == k2 or "label" in (k1, k2):
                    continue  # провода разных цепей разводит трассировка; метка стоит на проводе
                kw, rw, ko, ro = (k1, r1, k2, r2) if k1 == "wire" else (k2, r2, k1, r1)
                seg = ((rw[0], rw[1]), (rw[2], rw[3]))
                pts = [(x, y) for x, y in _segment_points(*seg)]
                if any(inside(ro, q) for q in pts):
                    hit.append((name(k1, o1), name(k2, o2)))
                continue
            if o1 == o2 and "body" not in (k1, k2) and "text" not in (k1, k2):
                continue  # земля и метка у своего вывода
            if o1 == o2 and {k1, k2} in ({"body", "label"}, {"body", "ground"}):
                continue  # метка и земля начинаются у вывода своего элемента
            if min(r1[2], r2[2]) - max(r1[0], r2[0]) > margin and min(r1[3], r2[3]) - max(r1[1], r2[1]) > margin:
                hit.append((name(k1, o1), name(k2, o2)))
    return hit


def note_place(drawing, lines):
    """-> (x, y) привязки текста: справа от схемы, первая строка у верхнего края листа."""
    right, _ = extent(drawing)
    return right + TEXT_GAP, 2 + TEXT_LINE * len(lines)


def view_scale(drawing, lines):
    """Шаг масштаба MC12, при котором схема и текст помещаются в окно."""
    drawing = _drawing(drawing)
    b = [r for *_, r in boxes(drawing)]
    right, bottom = extent(drawing)
    x, y = note_place(drawing, lines)
    # MC12 открывает окно с верхнего левого края содержимого, а не с начала листа (видно по снимкам)
    width = x + TEXT_CHAR * max(len(l) for l in lines) + 2 - min(r[0] for r in b)
    height = max(bottom, y) + 2 - min(r[1] for r in b)
    need = max(width / VIEW_W, height / VIEW_H)
    return next((k for k in SCALES if k >= need), SCALES[-1])


def text_area(lines):
    """Текстовая область (страница Text) в формате файла версии 10.01: строки Text=..., как в образце
    Sallen-Key. Формат «Len=N + сырой текст» - у файлов версии 9 (DATA/CHOKE.cir); в нашем файле MC12
    его не читает («Extra or unknown data»). Блок должен стоять после [Section]/[Page]: раньше них
    MC12 сообщает «Bad format in loading file» (проверено 25.10.2026)."""
    return "[Text Area]\nSection=0\nPage=1\nText=\n" + "\n".join(f"Text={l}" for l in lines)


def _source_value(e):
    return f"DC {e['value']} AC 1" if e["nodes"][0] == "IN" else f"DC {e['value']}"


def _two_pin(e, x, y, vertical):
    if not vertical:
        return Placed(e, x, y, 0, [(x, y), (x + 6, y)], "h", [((x - STUB, y), "left"), ((x + 6, y), "right")])
    if e["nodes"][1] == "0":  # земля на плюс-выводе: Rot 1, корпус вниз от точки привязки
        return Placed(e, x, y, 1, [(x, y), (x, y + 6)], "down", [((x, y), "up"), ((x, y + 6), "down")])
    # земля на минус-выводе: Rot 3, точка привязки (минус) внизу
    return Placed(e, x, y + 6, 3, [(x, y + 6), (x, y)], "up", [((x, y + 6), "down"), ((x, y), "up")])


class Layout(list):
    """[Placed] раскладки по пути сигнала и её метки: {цепь: (номер элемента, точка, [(a, b)] - отвод к метке)}."""
    labels = {}


def _swap_to(e, first):
    """Выводы R, C, L равноправны (iso их не различает): чтобы не поворачивать элемент, переставляются узлы."""
    a, b = e["nodes"]
    return e if a == first or e["type"] not in "RCL" else dict(e, nodes=[b, a])


def _horizontal(e, x, y, left, kind="h"):
    e = _swap_to(e, left)
    return Placed(e, x, y, 0, [(x, y), (x + 6, y)], kind, [((x - STUB, y), "left"), ((x + 6, y), "right")])


def _vertical(e, x, y, top):
    """Вертикальный элемент, вывод узла top - сверху в (x, y). Источник - плюсом вверх (Rot 3, привязка снизу)."""
    if e["type"] in "VI":
        return Placed(e, x, y + 6, ROT_UP, [(x, y), (x, y + 6)], "source", [((x, y), "up"), ((x, y + 6), "down")])
    e = _swap_to(e, top)
    return Placed(e, x, y, 1, [(x, y), (x, y + 6)], "down", [((x, y), "up"), ((x, y + 6), "down")])


def _opamp(e, x, y, flip):
    """ОУ с выходом в (x + 9, y). flip - отражение по вертикали (Rot 4): вход сигнала (минус) сверху."""
    e = dict(e, nodes=list(e["nodes"]) + ["0", "0"])  # питание на землю, см. _grid_layout
    if not flip:
        y0 = y - 3
        pins = [(x + OPAMP_PINS[p][0], y0 + OPAMP_PINS[p][1]) for p in ("plus", "minus", "out", "vee", "vcc")]
        ends = [((x - STUB, pins[0][1]), "left"), ((x - STUB, pins[1][1]), "left"), (pins[2], "right"),
                (pins[3], "down"), (pins[4], "up")]
        return Placed(e, x, y0, 0, pins, "opamp_t", ends)
    y0 = y + 3
    pins = [(x + OPAMP_PINS[p][0], y0 - OPAMP_PINS[p][1]) for p in ("plus", "minus", "out", "vee", "vcc")]
    ends = [((x - STUB, pins[0][1]), "left"), ((x - STUB, pins[1][1]), "left"), (pins[2], "right"),
            (pins[3], "up"), (pins[4], "down")]
    return Placed(e, x, y0, 4, pins, "opamp_f", ends)


def _hang_width(e):
    """Ширина места под висящий элемент: корпус и подписи справа от него."""
    body, text = (1.5, 1.75) if e["type"] in "VI" else (0.75, 1)
    return math.ceil(body + text + TEXT_CHAR * max(map(len, _attr_texts(e))) + 1.5)


def _flow_layout(elems, spread=0):
    """Раскладка по пути сигнала -> Layout или None, если схема не укладывается в её правила.

    Главная линия - кратчайший путь IN -> OUT; в узел, который ведёт ОУ, путь входит только через этот ОУ (иначе
    обратная связь стала бы главной линией). Второй вход ОУ вне линии - узел обратной связи, его элементы ставятся
    под ОУ. Узлы вне линии висят под узлом, от которого до них дошёл поиск. spread - добавка к промежуткам."""
    srcs = [e for e in elems if e["type"] in "VI"]
    ops = [e for e in elems if e["type"] == "OPAMP"]
    two = [e for e in elems if e["type"] in "RCL"]
    if len(srcs) != 1 or srcs[0]["nodes"] != ["IN", "0"]:
        return None
    driven = {o["nodes"][2] for o in ops}
    if len(driven) != len(ops) or "0" in driven or "IN" in driven:
        return None
    adj = {}
    for e in two:
        a, b = e["nodes"]
        if "0" not in (a, b) and a != b:
            adj.setdefault(a, []).append((b, e))
            adj.setdefault(b, []).append((a, e))
    # перебор простых путей (схемы маленькие, шагов не больше PATH_STEPS): больше ОУ, затем короче
    best, budget = [None], [PATH_STEPS]

    def walk(n, path, edges):
        budget[0] -= 1
        if budget[0] < 0:
            return
        if n == "OUT":
            key = (sum(e["type"] == "OPAMP" for e in edges), -len(edges))
            if best[0] is None or key > best[0][0]:
                best[0] = (key, list(path), list(edges))
            return
        steps = [(m, e) for m, e in adj.get(n, []) if m not in driven]
        steps += [(o["nodes"][2], o) for o in ops if n in o["nodes"][:2]]
        for m, e in steps:
            if m not in path:
                walk(m, path + [m], edges + [e])

    walk("IN", ["IN"], [])
    if best[0] is None:
        return None
    _, main, via = best[0]
    on_main = set(main)
    if sum(e["type"] == "OPAMP" for e in via) != len(ops):
        return None  # ОУ вне главной линии (кольцо из нескольких ОУ и т.п.)
    fb = {}
    for i, e in enumerate(via):
        if e["type"] == "OPAMP":
            other = [n for n in e["nodes"][:2] if n != main[i]]
            if other and other[0] != "0" and other[0] not in on_main:
                if other[0] in fb:
                    return None
                fb[other[0]] = i
    used = {id(e) for e in via}
    items = {}  # узел -> [(элемент, узел под ним или None)]: что висит под узлом
    for e in two:
        a, b = e["nodes"]
        if "0" in (a, b):
            if a == b:
                return None
            items.setdefault(b if a == "0" else a, []).append((e, None))
            used.add(id(e))
    parent, queue = {}, list(main) + list(fb)
    for n in queue:
        for m, e in adj.get(n, []):
            if id(e) in used or m in on_main or m in fb or m in parent:
                continue
            parent[m] = n
            used.add(id(e))
            items.setdefault(n, []).append((e, m))
            queue.append(m)
    nodes = {n for e in elems for n in e["nodes"] if n != "0"}
    if nodes - on_main - set(fb) - set(parent):
        return None  # часть схемы не связана с линией
    bridges = [e for e in two if id(e) not in used]

    out, nx = Layout(), {}

    def hang(n, x, y):
        """Подвешивает под узел n с верхним выводом в (x, y) -> ширина."""
        cur = x
        for e, child in items.get(n, []):
            out.append(_vertical(e, cur, y, n))
            w = _hang_width(e) + spread
            if child is not None:
                nx[child] = cur
                w = max(w, hang(child, cur, y + HANG_DY))
            cur += w
        return cur - x

    x = 0
    gaps = []  # (номер ОУ в out, узел обратной связи, x ОУ)
    for i, n in enumerate(main):
        nx[n] = x
        w = 0
        if n == "IN":
            out.append(_vertical(srcs[0], x, HANG_TOP, "IN"))
            w = _hang_width(srcs[0]) + spread
        w += hang(n, x + w, HANG_TOP)
        if i == len(main) - 1:
            break
        e = via[i]
        if e["type"] == "OPAMP":
            f = next((m for m, j in fb.items() if j == i), None)
            fw = 0
            if f is not None:
                nx[f] = x + w + 1
                fw = hang(f, x + w + 1, FB_Y) + 1
            xo = x + max(4, w + fw + 5) + spread
            out.append(_opamp(e, xo, 0, flip=e["nodes"][1] == n))
            gaps.append((len(out) - 1, f, xo))
            x = xo + OPAMP_GAP + spread
        else:
            gap = max(FLOW_DX, w + 2) + spread
            out.append(_horizontal(e, x + (gap - 6) // 2, 0, n))
            x += gap
    xo_end = nx["OUT"]
    src = next(k for k, p in enumerate(out) if p.kind == "source")
    out_el = next(k for k, p in enumerate(out) if "OUT" in p.e["nodes"])
    out.labels = {"IN": (src, (0, 0), [((0, HANG_TOP), (0, 0))]),
                  "OUT": (out_el, (xo_end + 3, 0), [((xo_end, 0), (xo_end + 3, 0))])}

    # под ОУ: элементы между узлом обратной связи и линией, рядами вниз
    rows = {}
    for e in list(bridges):
        a, b = e["nodes"]
        f, m = (a, b) if a in fb else (b, a)
        if f in fb and m in on_main:
            xo = next(g[2] for g in gaps if g[1] == f)
            r = rows.get(f, 0)
            rows[f] = r + 1
            out.append(_horizontal(e, xo + 3, FB_Y + 6 * r, f))
            bridges.remove(e)

    # остальное - перемычки над линией, короткие ниже; высота - по верхнему краю того, что под ними
    obstacles = [r for _, _, r in _element_boxes(out)]
    for p in out:
        for node, (end, d) in zip(p.e["nodes"], p.ends):
            if node == "0" and d == "up":
                obstacles.append(_ground_box(end, d))
    for net, (_, pt, _) in out.labels.items():
        obstacles.append(_text_box(pt[0], pt[1], net))
    for e in sorted(bridges, key=lambda e: abs(nx[e["nodes"][0]] - nx[e["nodes"][1]])):
        a, b = sorted(e["nodes"], key=lambda n: nx[n])
        left = (nx[a] + nx[b]) // 2 - 3
        lo, hi = min(nx[a], left) - 1, max(nx[b], left + 6) + 1
        top = min([r[1] for r in obstacles if r[0] < hi and r[2] > lo] + [-1])
        p = _horizontal(e, left, math.floor(top - 2), a, "hb")
        out.append(p)
        obstacles += [r for _, _, r in _element_boxes([p])]
    return out


def _shifted(lay, dx, dy):
    """Сдвиг раскладки на целое число шагов."""
    mv = lambda q: (q[0] + dx, q[1] + dy)
    out = Layout(Placed(p.e, p.x + dx, p.y + dy, p.rot, [mv(q) for q in p.pins], p.kind,
                        [(mv(q), d) for q, d in p.ends]) for p in lay)
    out.labels = {n: (i, mv(pt), [(mv(a), mv(b)) for a, b in stub]) for n, (i, pt, stub) in lay.labels.items()}
    return out


def _frame(lay):
    """Чертёж без трассировки: элементы, земли, отводы и метки IN/OUT - для проверки наложений до разводки."""
    grounds, wires, labels = [], [], []
    for i, p in enumerate(lay):
        for node, pin, (end, d) in zip(p.e["nodes"], p.pins, p.ends):
            if node == "0":
                gpt, gd = ground_at(pin, end, d)
                grounds.append((gpt, gd, i))
                if gpt != pin:
                    wires.append((pin, gpt, "0"))
    for net, (i, pt, stub) in lay.labels.items():
        wires += [(a, b, net) for a, b in stub]
        labels.append((net, pt, i))
    return Drawing(lay, wires, labels, grounds, set())


def layout(elems):
    """-> [Placed]: элемент, точка привязки и поворот, выводы в порядке nodes, точки меток. Шаги сетки.
    Сначала раскладка по пути сигнала (с растущими промежутками, пока что-то накладывается), иначе - сетка."""
    for spread in (0, 2, 4):
        try:
            lay = _flow_layout(elems, spread)
        except (KeyError, ValueError, StopIteration):
            lay = None
        if lay is None:
            break
        b = [r for _, _, r in _element_boxes(lay)]
        lay = _shifted(lay, 4 - math.floor(min(r[0] for r in b)), 8 - math.floor(min(r[1] for r in b)))
        pins = [q for p in lay for q in p.pins]
        if len(set(pins)) == len(pins) and not overlaps(_frame(lay)):
            return lay
    return _grid_layout(elems)


def _grid_layout(elems):
    """Прежняя раскладка сеткой: источники слева, последовательные элементы строками, на землю - ниже, ОУ внизу."""
    placed = []
    for k, e in enumerate(e for e in elems if e["type"] in "VI"):
        x, y = SOURCE_X, 16 + SOURCE_DY * k
        down = (x, y + STUB_DOWN) if e["nodes"][1] != "0" else (x, y)
        placed.append(Placed(e, x, y, ROT_UP, [(x, y - 6), (x, y)], "source", [((x, y - 6), "up"), (down, "down")]))
    two = [e for e in elems if e["type"] in "RCL"]
    series = [e for e in two if "0" not in e["nodes"]]
    shunt = [e for e in two if "0" in e["nodes"]]
    y = 8
    for i, e in enumerate(series):
        placed.append(_two_pin(e, X0 + SERIES_DX * (i % SERIES_COLS), y + SERIES_DY * (i // SERIES_COLS), False))
    y += SERIES_DY * math.ceil(len(series) / SERIES_COLS)
    for i, e in enumerate(shunt):  # +2: место для меток над верхними выводами
        placed.append(_two_pin(e, X0 + SHUNT_DX * (i % SHUNT_COLS), y + 2 + SHUNT_DY * (i // SHUNT_COLS), True))
    y = max(y + SHUNT_DY * math.ceil(len(shunt) / SHUNT_COLS) + 5, 24)  # +5: земля питания над ОУ; не выше источника
    for j, e in enumerate(e for e in elems if e["type"] == "OPAMP"):
        x = X0 + OPAMP_DX * j
        # выводы питания - на землю: у модели уровня 1 они не влияют на передачу, а неподключённые
        # дают предупреждение MC12 «Nodes have no DC path to ground. Resistors added»
        e = dict(e, nodes=list(e["nodes"]) + ["0", "0"])
        pins = [(x + OPAMP_PINS[p][0], y + OPAMP_PINS[p][1]) for p in ("plus", "minus", "out", "vee", "vcc")]
        ends = [((x - STUB, y), "left"), ((x - STUB, y + 6), "left"), (pins[2], "right"), (pins[3], "down"),
                (pins[4], "up")]
        placed.append(Placed(e, x, y, 0, pins, "opamp", ends))
    return placed


def _templates():
    text = TEMPLATES.read_bytes().decode("ascii").replace("\r", "")
    return [b.strip() for b in re.split(r"\n(?=\[)", text) if b.startswith(("[shapedef]", "[compdef]"))]


def build(netlist, f_char=None, title="spice2cir", wires=True):
    elems, err = canon.parse_netlist(netlist)
    if err:
        raise ValueError("; ".join(err))
    if any(e["type"] == "I" for e in elems):
        raise ValueError("источник тока spice2cir не поддерживает (генератор его не использует)")
    head = ("[Main]\nFileType=CIR\nVersion=10.01\nProgram=Micro-Cap\nComponent Version=10.00\n"
            "Shape Version=10.01\nScreen=1920,1200\nMCAPVersion=10.0.4.0\nKey ID=0\nNLCnt=0\nTXCnt=0")
    parts = [head] + _templates()
    parts.append("[Circuit]\nSpiceType=AMBIGUOUS\nShow Nodes=False\nShow Attribute Text=True\nShow Grid Text=True")
    drawing = draw(layout(elems), wires)
    note = [f"{title}: netlist (spice2cir)"] + netlist.strip().splitlines() + ["AC analysis: Alt+2"]
    scale = view_scale(drawing, note)
    for p in drawing.placed:
        e = p.e
        name, vattr = COMP["V" if e["type"] in "VI" else e["type"]]
        part, value = _attr_texts(e)
        (op, ov) = ATTRS[p.kind]
        parts.append(_comp(name, p.x, p.y, p.rot, [(op, "PART", part), (ov, vattr, value)]))
    for pt, gd, _ in drawing.grounds:
        parts.append(_comp("Ground", pt[0], pt[1], GROUND_ROT[gd], []))
    for (x0, y0), (x1, y1), _ in drawing.wires:
        parts.append(f"[Wire]\nGrids={x0},{y0},{x1},{y1}")
    for text, (x, y), _ in drawing.labels:
        parts.append(_text([text], x, y))
    parts.append(_text(note, *note_place(drawing, note)))
    fl, fh = bench.ac_range(f_char)
    parts.append("[Section]\nIndex=0\nName=\"spice2cir\"\nType=Main\nEnable=True")
    for i, (pname, extra) in enumerate((("Main", ""), ("Text", "Type=Text\n"), ("Models", "Type=Text\n"),
                                        ("Info", "Type=Info\n"))):
        zoom = "".join(f"\nScale {k}={scale!r}" for k in range(4)) if i == 0 else ""
        parts.append(f"[Page]\nIndex={i}\nSection=0\n{extra}Print={'True' if i == 0 else 'False'}\nEnable=True\nName={pname}{zoom}")
    if any(e["type"] == "OPAMP" for e in elems):
        # после [Section]/[Page], как в образцах MC12: текстовая область ссылается на страницу Page=1
        parts.append(text_area([OPAMP_MODEL]))
    parts.append(f"[Limits]\nAnalysis=AC\nFRange={canon.format_value(fh)},{canon.format_value(fl)}\n"
                 f"NPts={int(round(bench.POINTS_PER_DECADE * 6)) + 1}\nTemp=27\nMaxChng=5\nNInput=NONE\nNOutput=2\n"
                 "Sweep type=Log\nStateVar=Zero\nRunOption=Normal\nTemperature Type=Linear\nAccumulate Plots=False")
    parts.append("[WaveForm]\nAnalysis=AC\nPlt=1\nAliasID=1\nXExp=F\nYExp=dB(V(OUT))\nXRng=FMAX,FMIN\n"
                 "YRng=80,-100\nOptions=LINEARY\nEnable=Enable")
    text = "\r\n\r\n".join(p.replace("\n", "\r\n") for p in parts) + "\r\n"
    text.encode("ascii")  # кириллица в файлах для MC12 недопустима
    return text


def write(path, text):
    Path(path).write_bytes(text.encode("ascii"))


if __name__ == "__main__":
    net = Path(sys.argv[1]).read_text(encoding="utf-8")
    f_char = float(sys.argv[sys.argv.index("--fchar") + 1]) if "--fchar" in sys.argv else None
    write(sys.argv[2], build(net, f_char, Path(sys.argv[1]).stem))
    print(f"-> {sys.argv[2]}")
