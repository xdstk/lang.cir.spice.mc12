#!/usr/bin/env python3
"""Нетлист класса A -> схема Micro-Cap (.CIR), которую пользователь открывает в MC12 (решение В7).

Запуск:  python spice2cir.py нетлист.ckt схема.cir [--fchar 1591]

Раскладка простая (CLAUDE.md): источник слева, двухполюсники сеткой по строкам, ОУ ниже; провода не
рисуются - соединения задаются метками узлов (Grid Text на выводах, RM12 с. 45-46, 67; вариант без
проводов проверен в MC12). На узел 0 ставится символ земли. ОУ - элемент Opamp с моделью OPAMP_IDEAL
уровня 1: A = 1e6, R = ROUTAC + ROUTDC = 1 Ом (RM12 с. 728, 732), выводы питания не подключены
(у уровня 1 они не влияют на передачу). Описания фигур и элементов - data/cir_templates.cir.
В схему кладутся текст с исходным нетлистом и анализ AC (диапазон обвязки bench) с графиком dB(V(OUT)).
Файл - ASCII с CRLF (CLAUDE.md).
"""
import re
import sys
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
GRID_X, GRID_Y, COLS = 14, 10, 4
# Текст с нетлистом: MC12 привязывает многострочный Grid Text за нижний левый угол и поле JustifyV=Top в нашем
# файле не принимает (проверено 03.11.2026), поэтому блок ставится справа от схемы с учётом его высоты.
# Строка шрифта по умолчанию (Verdana 12) - около 2 шагов сетки, знак - около 0,9 шага.
TEXT_LINE, TEXT_CHAR, TEXT_GAP = 2, 0.9, 6
# Масштаб окна схемы: поля Scale 0..3 страницы Main - сколько единиц чертежа на пиксель, меньше 1 - крупнее
# (проверено 03.11.2026; [Schematic] Scale MC12 при загрузке не читает). Принимаются только шаги масштаба MC12:
# 0.5, 2/3 и 2 сработали, 0.6 и 0.95 MC12 молча заменил на 1. VIEW - видимая часть окна при масштабе 1 в шагах
# сетки, замер на ноутбуке автора (развёрнутое окно MC12). Берётся самый крупный шаг, при котором всё помещается.
VIEW_W, VIEW_H = 135, 52
SCALES = (0.5, 2 / 3, 1, 2)


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


def extent(placed):
    """Правый и нижний край раскладки в шагах сетки: по выводам, точкам привязки и подписям (до 4 шагов вправо)."""
    xs = [x for _, x, _, _, _ in placed] + [px for *_, pins in placed for px, _ in pins]
    ys = [y for _, _, y, _, _ in placed] + [py for *_, pins in placed for _, py in pins]
    return max(xs) + 4, max(ys) + 2


def note_place(placed, lines):
    """-> (x, y) привязки текста: справа от схемы, первая строка у верхнего края листа."""
    right, _ = extent(placed)
    return right + TEXT_GAP, 2 + TEXT_LINE * len(lines)


def view_scale(placed, lines):
    """Шаг масштаба MC12, при котором схема и текст помещаются в окно."""
    right, bottom = extent(placed)
    x, y = note_place(placed, lines)
    width = x + TEXT_CHAR * max(len(l) for l in lines) + 2
    height = max(bottom, y) + 2
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


def layout(elems):
    """-> [(элемент, x, y, rot, [точки выводов в порядке nodes])], координаты - шаги сетки."""
    placed, k = [], 0
    for e in elems:
        if e["type"] in "VI":
            x, y = 6, 30
            pins = {n: (x + px, y + py) for n, (px, py) in
                    (("plus", rotate(*TWO_PIN["plus"], ROT_UP)), ("minus", rotate(*TWO_PIN["minus"], ROT_UP)))}
            placed.append((e, x, y, ROT_UP, [pins["plus"], pins["minus"]]))
    two = [e for e in elems if e["type"] in "RCL"]
    for i, e in enumerate(two):
        x, y = 20 + GRID_X * (i % COLS), 14 + GRID_Y * (i // COLS)
        placed.append((e, x, y, 0, [(x + TWO_PIN["minus"][0], y), (x + TWO_PIN["plus"][0], y)]))
    y0 = 14 + GRID_Y * ((len(two) + COLS - 1) // COLS) + 4
    for j, e in enumerate(e for e in elems if e["type"] == "OPAMP"):
        x, y = 20 + 22 * j, y0
        # выводы питания - на землю: у модели уровня 1 они не влияют на передачу, а неподключённые
        # дают предупреждение MC12 «Nodes have no DC path to ground. Resistors added»
        e = dict(e, nodes=list(e["nodes"]) + ["0", "0"])
        placed.append((e, x, y, 0, [(x + OPAMP_PINS[p][0], y + OPAMP_PINS[p][1])
                                    for p in ("plus", "minus", "out", "vee", "vcc")]))
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
    for e, x, y, rot, pins in placed:
        name, vattr = COMP["V" if e["type"] in "VI" else e["type"]]
        value = OPAMP_MODEL.split()[1] if e["type"] == "OPAMP" else (
            _source_value(e) if e["type"] in "VI" else e["value"])
        parts.append(_comp(name, x, y, rot, [((8, -14), "PART", e["name"]), ((8, -26), vattr, value)]))
        for node, (px, py) in zip(e["nodes"], pins):
            if (px, py) in used:
                raise AssertionError(f"совпали точки выводов в ({px}, {py})")  # иначе выводы замкнулись бы
            used.add((px, py))
            if node == "0":
                parts.append(_comp("Ground", px, py, 7, []))
            else:
                parts.append(_text([node], px, py))
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
