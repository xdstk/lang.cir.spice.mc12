#!/usr/bin/env python3
"""Численный узловой расчёт (модифицированный метод узловых потенциалов) для схем класса A.

Независимая от MC12 проверка формул и эталонов: H(f) = V(OUT) при AC 1 на источнике с плюсом в IN
(как в обвязке bench). OPAMP_IDEAL - ИНУН с тем же усилением, что в bench (1e6), выход относительно
земли. Только стандартная библиотека: комплексный метод Гаусса с выбором главного элемента.
"""
import cmath
import math

OPAMP_GAIN = 1e6  # как в bench.OPAMP_DEF

try:
    from . import canon
except ImportError:  # запуск как скрипта
    import canon


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-300:
            raise ZeroDivisionError("вырожденная матрица (узел без пути к земле?)")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            k = M[r][c] / M[c][c]
            if k:
                for j in range(c, n + 1):
                    M[r][j] -= k * M[c][j]
    x = [0j] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][j] * x[j] for j in range(r + 1, n))) / M[r][r]
    return x


def response(elems, f):
    """Комплексный V(OUT) на частоте f (Гц) при AC 1 на источниках с плюсом в IN, остальные - 0."""
    nodes = sorted({n for e in elems for n in e["nodes"]} - {"0"})
    idx = {n: i for i, n in enumerate(nodes)}
    branches = [e for e in elems if e["type"] in ("V", "OPAMP")]  # добавочные токи
    n = len(nodes) + len(branches)
    A = [[0j] * n for _ in range(n)]
    b = [0j] * n
    s = 2j * math.pi * f

    def stamp_y(a, c, y):
        for p, q, sg in ((a, a, 1), (c, c, 1), (a, c, -1), (c, a, -1)):
            if p != "0" and q != "0":
                A[idx[p]][idx[q]] += sg * y

    k = len(nodes)
    for e in elems:
        t, nd = e["type"], e["nodes"]
        v = canon.parse_value(e["value"]) if e["value"] is not None else None
        if t == "R":
            stamp_y(nd[0], nd[1], 1 / v)
        elif t == "C":
            stamp_y(nd[0], nd[1], s * v)
        elif t == "L":
            stamp_y(nd[0], nd[1], 1 / (s * v))
        elif t == "I":  # ток AC 1 из + через источник в - (направление SPICE: из + во внешнюю цепь через -)
            ac = 1.0 if nd[0] == "IN" else 0.0
            if nd[0] != "0":
                b[idx[nd[0]]] -= ac
            if nd[1] != "0":
                b[idx[nd[1]]] += ac
        elif t == "V":
            ac = 1.0 if nd[0] == "IN" else 0.0
            for node, sg in ((nd[0], 1), (nd[1], -1)):
                if node != "0":
                    A[idx[node]][k] += sg
                    A[k][idx[node]] += sg
            b[k] = ac
            k += 1
        elif t == "OPAMP":  # V(out) = A (V+ - V-), ток ветви втекает в выход
            plus, minus, out = nd
            if out != "0":
                A[idx[out]][k] += 1
                A[k][idx[out]] += 1
            for node, g in ((plus, -OPAMP_GAIN), (minus, OPAMP_GAIN)):
                if node != "0":
                    A[k][idx[node]] += g
            k += 1
    x = _solve(A, b)
    return x[idx["OUT"]] if "OUT" in idx else 0j


def sweep(elems, f_lo, f_hi, per_decade=200):
    """(f, дБ, градусы) на логарифмической сетке - тот же вид, что даёт mcout для MC12."""
    nd = int(round(math.log10(f_hi / f_lo) * per_decade))
    f = [f_lo * 10 ** (i / per_decade) for i in range(nd + 1)]
    h = [response(elems, x) for x in f]
    return f, [20 * math.log10(abs(v)) if v else -400.0 for v in h], [math.degrees(cmath.phase(v)) for v in h]


# Насыщение реального ОУ по постоянному току (чертёж с реальным ОУ, spice2cir.power). Пороги - умолчания модели OPA
# уровня 3 при питании +-15 В: размах выхода VPS/VNS = +-13 В, смещение нуля VOFF = 1 мВ, ток короткого замыкания
# IOSC = 20 мА (RM12 с. 728).
SWING = 13.0
OFFSET = 1e-3
CURRENT = 0.02
GMIN = 1e-12  # См, от каждого узла на землю, как GMIN в SPICE: узел за конденсаторами не повисает


def dc_outputs(elems, offset=0.0, currents=False):
    """Постоянные напряжения на выходах ОУ -> {имя ОУ: В} (currents=True - ещё {имя ОУ: ток выхода, А}).
    Конденсатор - разрыв, катушка - перемычка, источники - своё DC, ОУ - ИНУН с усилением OPAMP_GAIN и смещением нуля
    offset на входе."""
    nodes = sorted({n for e in elems for n in e["nodes"]} - {"0"})
    idx = {n: i for i, n in enumerate(nodes)}
    branches = [e for e in elems if e["type"] in ("V", "L", "OPAMP")]
    n = len(nodes) + len(branches)
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(len(nodes)):
        A[i][i] += GMIN

    def branch(k, plus, minus):
        for node, sg in ((plus, 1), (minus, -1)):
            if node != "0":
                A[idx[node]][k] += sg
                A[k][idx[node]] += sg

    k = len(nodes)
    for e in elems:
        t, nd = e["type"], e["nodes"]
        if t == "R":
            g = 1 / canon.parse_value(e["value"])
            for p, q, sg in ((nd[0], nd[0], 1), (nd[1], nd[1], 1), (nd[0], nd[1], -1), (nd[1], nd[0], -1)):
                if p != "0" and q != "0":
                    A[idx[p]][idx[q]] += sg * g
        elif t in ("V", "L"):  # катушка - источник 0 В
            branch(k, nd[0], nd[1])
            b[k] = canon.parse_value(e["value"]) if t == "V" else 0.0
            k += 1
        elif t == "OPAMP":  # V(out) = A (V+ - V- + offset)
            plus, minus, out = nd
            if out != "0":
                A[idx[out]][k] += 1
                A[k][idx[out]] += 1
            for node, g in ((plus, -OPAMP_GAIN), (minus, OPAMP_GAIN)):
                if node != "0":
                    A[k][idx[node]] += g
            b[k] = OPAMP_GAIN * offset
            k += 1
    x = _solve(A, b)
    volts = {e["name"]: x[idx[e["nodes"][2]]].real if e["nodes"][2] != "0" else 0.0
             for e in elems if e["type"] == "OPAMP"}
    if not currents:
        return volts
    amps = {e["name"]: x[len(nodes) + j].real for j, e in enumerate(branches) if e["type"] == "OPAMP"}
    return volts, amps


def saturation(elems, swing=SWING, offset=OFFSET, current=CURRENT):
    """ОУ, выход которого по постоянному току вышел бы за размах swing или ограничение тока current
    -> [(имя, значение, причина)], пусто - насыщения нет. Причина "source" - постоянное напряжение источника, усиленное
    схемой (значение - В); "offset" - смещение нуля без обратной связи по постоянному току (интегратор без резистора
    в ОС и т. п., В); "current" - выходной ток больше тока короткого замыкания (низкоомная нагрузка, А)."""
    if not any(e["type"] == "OPAMP" for e in elems):
        return []
    try:
        (plain, amps), shifted = dc_outputs(elems, currents=True), dc_outputs(elems, offset)
    except ZeroDivisionError:
        return [(e["name"], math.inf, "offset") for e in elems if e["type"] == "OPAMP"]
    out = []
    for name, v in shifted.items():
        if abs(plain[name]) > swing:
            out.append((name, plain[name], "source"))
        elif abs(v) > swing:
            out.append((name, v, "offset"))
        elif abs(amps[name]) > current:
            out.append((name, abs(amps[name]), "current"))
    return out
