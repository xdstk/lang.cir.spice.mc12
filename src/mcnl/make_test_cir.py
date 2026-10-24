#!/usr/bin/env python3
"""Генерация тестовых .CIR (формат MC10, который MC12 читает) для RC-фильтра НЧ.

Вариант A: элементы + провода.
Вариант B: элементы без проводов, соединение метками узлов (Grid Text на выводах).
Описания фигур и элементов ([shapedef], [compdef]) копируются из присланного
Sallen-Key_High_Pass_Filter.cir, чтобы не выдумывать их.

Запуск: python make_test_cir.py <образец Sallen-Key .cir> <папка вывода>
"""
import re
import sys
from pathlib import Path

NEED_SHAPES = {"Capacitor", "Ground", "Plus.root", "Resistor", "SPICE_V", "Vsource.root"}
NEED_COMPS = {"Capacitor", "Ground", "Resistor", "Voltage Source"}


def blocks(text):
    out, cur = [], None
    for line in text.replace("\r", "").split("\n"):
        if re.match(r"^\[.+\]$", line):
            cur = [line]
            out.append(cur)
        elif cur is not None:
            cur.append(line)
    return out


def pick(src, header, wanted):
    res = []
    for b in blocks(src):
        if b[0] == header:
            name = next((l.split("=", 1)[1] for l in b if l.startswith("Name=")), None)
            if name in wanted:
                res.append("\n".join(x for x in b if x.strip()))
    return res


def comp(name, x, y, rot, attrs):
    s = [f"[Comp]\nName={name}\nGrid={x},{y}\nRot={rot}"]
    for (ox, oy), key, val in attrs:
        s.append(f"[Attr]\nON={ox},{oy},{key}\nV={val}\nStyle=8192")
    return "\n\n".join(s)


def wire(x1, y1, x2, y2):
    return f"[Wire]\nGrids={x1},{y1},{x2},{y2}"


def text(t, x, y):
    return f"[Grid Text]\nText=\"{t}\"\nGrid={x},{y}\nCBorder=None"


def build(src, with_wires):
    head = ("[Main]\nFileType=CIR\nVersion=10.01\nProgram=Micro-Cap\n"
            "Component Version=10.00\nShape Version=10.01\nScreen=1920,1200\n"
            "MCAPVersion=10.0.4.0\nKey ID=0\nNLCnt=0\nTXCnt=0")
    parts = [head] + pick(src, "[shapedef]", NEED_SHAPES) + pick(src, "[compdef]", NEED_COMPS)
    parts.append("[Circuit]\nSpiceType=AMBIGUOUS\nShow Nodes=False\nShow Attribute Text=True\n"
                 "Show Grid Text=True")
    # Источник: Grid=20,25 Rot=3 -> Plus (20,19), Minus (20,25)
    parts.append(comp("Voltage Source", 20, 25, 3,
                      [((-31, -27), "PART", "V1"), ((-110, -41), "VALUE", "DC 0 AC 1 0")]))
    parts.append(comp("Ground", 20, 25, 7, []))
    # R1: Grid=25,19 Rot=0 -> Minus (25,19), Plus (31,19)
    parts.append(comp("Resistor", 25, 19, 0,
                      [((18, -20), "PART", "R1"), ((18, -33), "RESISTANCE", "1k")]))
    # C1: Grid=37,19 Rot=1 -> Minus (37,19), Plus (37,25)
    parts.append(comp("Capacitor", 37, 19, 1,
                      [((12, 12), "PART", "C1"), ((12, 26), "CAPACITANCE", "100n")]))
    parts.append(comp("Ground", 37, 25, 7, []))
    if with_wires:
        parts += [wire(20, 19, 25, 19), wire(31, 19, 37, 19)]
        parts += [text("In", 20, 19), text("Out", 37, 19)]
    else:
        # без проводов: одинаковые метки на выводах соединяют узлы (RM12, с. 45-46, 67)
        parts += [text("In", 20, 19), text("In", 25, 19), text("Out", 31, 19), text("Out", 37, 19)]
    parts.append(text("RC low-pass test: R1=1k, C1=100n, fc=1.59kHz. Run AC analysis.", 20, 10))
    parts.append("[Section]\nIndex=0\nName=\"RC_test\"\nType=Main\nEnable=True")
    parts.append("[Page]\nIndex=0\nSection=0\nPrint=True\nEnable=True\nName=Main")
    parts.append("[Page]\nIndex=1\nSection=0\nType=Text\nPrint=False\nEnable=True\nName=Text")
    parts.append("[Page]\nIndex=2\nSection=0\nType=Text\nPrint=False\nEnable=True\nName=Models")
    parts.append("[Page]\nIndex=3\nSection=0\nType=Info\nPrint=False\nEnable=True\nName=Info")
    parts.append("[Limits]\nAnalysis=AC\nFRange=1Meg,10\nNPts=501\nTemp=27\nMaxChng=5\n"
                 "NInput=NONE\nNOutput=2\nSweep type=Log\nStateVar=Zero\nRunOption=Normal\n"
                 "Temperature Type=Linear\nAccumulate Plots=False")
    parts.append("[WaveForm]\nAnalysis=AC\nPlt=1\nAliasID=1\nXExp=F\nYExp=dB(V(Out))\n"
                 "XRng=FMAX,FMIN\nYRng=10,-60\nOptions=LINEARY\nEnable=Enable")
    return "\r\n\r\n".join(p.replace("\n", "\r\n") for p in parts) + "\r\n"


if __name__ == "__main__":
    src = Path(sys.argv[1]).read_text(encoding="cp1251", errors="replace")
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "RC_test_wires.cir").write_bytes(build(src, True).encode("ascii"))
    (out / "RC_test_labels.cir").write_bytes(build(src, False).encode("ascii"))
    print("ok")
