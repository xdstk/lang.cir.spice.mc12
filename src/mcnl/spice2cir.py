#!/usr/bin/env python3
"""Нетлист класса A -> схема Micro-Cap (.CIR), которую пользователь открывает в MC12 (решение В7).

Запуск:  python spice2cir.py нетлист.ckt схема.cir [--fchar 1591]

Раскладка простая (CLAUDE.md): источник слева, последовательные двухполюсники строками, элементы на землю -
вертикально отдельными строками, ОУ ниже. Соединения задаются метками узлов (Grid Text на выводах, RM12
с. 45-46, 67; вариант без проводов проверен в MC12); провода - только короткие отводы, на которых стоят метки.
На узел 0 ставится символ земли, направленный от элемента наружу; элементы, метки и земли не перекрываются. ОУ - элемент Opamp с моделью OPAMP_IDEAL
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
    from . import bench, canon
except ImportError:  # запуск как скрипта
    import bench
    import canon

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
         "source": ((14, -37), (14, -23)), "opamp": ((76, 40), (76, 54))}
Placed = namedtuple("Placed", "e x y rot pins kind ends")  # ends: (точка метки, направление наружу) на вывод


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


def boxes(placed):
    """-> [(вид, номер элемента, (x0, y0, x1, y1))] - что занимает место на чертеже: корпуса, земли, отводы,
    метки узлов и подписи. Нужно для проверки, что ничего не накладывается, и для границ чертежа."""
    out = []
    for i, p in enumerate(placed):
        e, x, y = p.e, p.x, p.y
        if p.kind == "h":
            out.append(("body", i, (x + 0.5, y - 0.75, x + 5.5, y + 0.75)))
        elif p.kind in ("down", "up"):
            y0, y1 = sorted(py for _, py in p.pins)
            out.append(("body", i, (x - 0.75, y0 + 0.5, x + 0.75, y1 - 0.5)))
        elif p.kind == "source":
            out.append(("body", i, (x - 1.5, y - 5, x + 1.5, y - 1)))
        else:
            out.append(("body", i, (x + 1, y - 0.5, x + 8, y + 6.5)))
        for (ox, oy), text in zip(ATTRS[p.kind], _attr_texts(e)):
            out.append(("text", i, _text_box(x + ox / 8, y + oy / 8, text, top=True)))
        for node, (px, py), ((ex, ey), d) in zip(e["nodes"], p.pins, p.ends):
            if node == "0":
                (gx, gy), gd = ground_at((px, py), (ex, ey), d)
                g = {"right": (gx, gy - 1, gx + GROUND_LEN, gy + 1), "down": (gx - 1, gy, gx + 1, gy + GROUND_LEN),
                     "left": (gx - GROUND_LEN, gy - 1, gx, gy + 1), "up": (gx - 1, gy - GROUND_LEN, gx + 1, gy)}[gd]
                out.append(("ground", i, g))
                if (gx, gy) != (px, py):
                    out.append(("wire", i, (min(px, gx), min(py, gy), max(px, gx), max(py, gy))))
                continue
            if (ex, ey) != (px, py):
                out.append(("wire", i, (min(px, ex), min(py, ey), max(px, ex), max(py, ey))))
            out.append(("label", i, _text_box(ex, ey, node)))
    return out


def extent(placed):
    """Правый и нижний край чертежа в шагах сетки."""
    b = [r for *_, r in boxes(placed)]
    return math.ceil(max(r[2] for r in b)), math.ceil(max(r[3] for r in b))


def overlaps(placed, margin=0.1):
    """Пары того, что накладывается: корпуса, земли и тексты не должны задевать друг друга; отвод может касаться
    только своего элемента. Касание по краю (в пределах margin) не считается."""
    b = boxes(placed)
    hit = []
    for j, (k1, i1, r1) in enumerate(b):
        for k2, i2, r2 in b[j + 1:]:
            if i1 == i2 and "body" not in (k1, k2) and "text" not in (k1, k2):
                continue  # метка на своём отводе, земля у своего вывода
            if i1 == i2 and {k1, k2} in ({"body", "wire"}, {"body", "label"}, {"wire", "label"}):
                continue  # отвод и метка начинаются у вывода своего элемента
            if min(r1[2], r2[2]) - max(r1[0], r2[0]) > margin and min(r1[3], r2[3]) - max(r1[1], r2[1]) > margin:
                hit.append(((k1, placed[i1].e["name"]), (k2, placed[i2].e["name"])))
    return hit


def note_place(placed, lines):
    """-> (x, y) привязки текста: справа от схемы, первая строка у верхнего края листа."""
    right, _ = extent(placed)
    return right + TEXT_GAP, 2 + TEXT_LINE * len(lines)


def view_scale(placed, lines):
    """Шаг масштаба MC12, при котором схема и текст помещаются в окно."""
    b = [r for *_, r in boxes(placed)]
    right, bottom = extent(placed)
    x, y = note_place(placed, lines)
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


def layout(elems):
    """-> [Placed]: элемент, точка привязки и поворот, выводы в порядке nodes, точки меток. Шаги сетки."""
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


def build(netlist, f_char=None, title="spice2cir"):
    elems, err = canon.parse_netlist(netlist)
    if err:
        raise ValueError("; ".join(err))
    if any(e["type"] == "I" for e in elems):
        raise ValueError("источник тока spice2cir не поддерживает (генератор его не использует)")
    head = ("[Main]\nFileType=CIR\nVersion=10.01\nProgram=Micro-Cap\nComponent Version=10.00\n"
            "Shape Version=10.01\nScreen=1920,1200\nMCAPVersion=10.0.4.0\nKey ID=0\nNLCnt=0\nTXCnt=0")
    parts = [head] + _templates()
    parts.append("[Circuit]\nSpiceType=AMBIGUOUS\nShow Nodes=False\nShow Attribute Text=True\nShow Grid Text=True")
    used = set()
    placed = layout(elems)
    note = [f"{title}: netlist (spice2cir)"] + netlist.strip().splitlines() + ["AC analysis: Alt+2"]
    scale = view_scale(placed, note)
    for p in placed:
        e = p.e
        name, vattr = COMP["V" if e["type"] in "VI" else e["type"]]
        part, value = _attr_texts(e)
        (op, ov) = ATTRS[p.kind]
        parts.append(_comp(name, p.x, p.y, p.rot, [(op, "PART", part), (ov, vattr, value)]))
        for node, (px, py), ((ex, ey), d) in zip(e["nodes"], p.pins, p.ends):
            for pt in {(px, py), (ex, ey)}:
                if pt in used:
                    raise AssertionError(f"совпали точки выводов в {pt}")  # иначе выводы замкнулись бы
                used.add(pt)
            if node == "0":
                (gx, gy), gd = ground_at((px, py), (ex, ey), d)
                if (gx, gy) != (px, py):
                    parts.append(f"[Wire]\nGrids={px},{py},{gx},{gy}")
                parts.append(_comp("Ground", gx, gy, GROUND_ROT[gd], []))
                continue
            if (ex, ey) != (px, py):
                parts.append(f"[Wire]\nGrids={px},{py},{ex},{ey}")
            parts.append(_text([node], ex, ey))
    parts.append(_text(note, *note_place(placed, note)))
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
