#!/usr/bin/env python3
"""Разбор числового вывода Micro-Cap 12 (.ANO / .TNO / .DNO) и простые измерения по нему.

Запуск:  python mcout.py файл.ANO   -> JSON со столбцами таблиц и напряжениями рабочей точки

Формат восстановлен по выводу MC12 12.2.0.5 (проверка 25.10.2026, образцы в DATA):
  шапка из звёздочек с "AC Analysis of ИМЯ" -> блок Limits (ключ, значение по ширине) ->
  для каждого прогона баннер (Temperature=27 и т.п.) -> "DC Operating Point Voltages" (пары
  узел-напряжение) -> "Interpolated Waveform Values": строка имён, строка единиц, строки чисел.
  Числа - 4 значащие цифры; "NA" - нет значения (например, фаза в первой точке).
"""
import json
import math
import re
import sys
from pathlib import Path

NUM = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _value(tok):
    return float(tok) if NUM.match(tok) else None  # NA и прочее -> None


def _columns_by_data(lines):
    """Концы столбцов по первой строке данных: числа выровнены вправо под именами."""
    return [m.end() for m in re.finditer(r"\S+", lines[0])]


def _slice(line, ends):
    out, start = [], 0
    for e in ends:
        out.append(line[start:e].strip())
        start = e
    return out


def parse(text):
    """Текст вывода -> {"analysis", "circuit", "limits", "runs": [{"banner", "op_voltages", "tables"}]}.

    Таблица: {"columns": [...], "units": [...], "data": {столбец: [float|None, ...]}}."""
    lines = text.replace("\r", "").split("\n")
    res = {"analysis": None, "circuit": None, "limits": {}, "runs": []}
    run, limits_done = None, False
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r"\*\*\*\s+(.+?) Analysis of (\S+)\s+\*\*\*", line)
        if m and res["analysis"] is None:
            res["analysis"], res["circuit"] = m.group(1), m.group(2)
        elif line.strip() == "Limits" and lines[i + 1].startswith("="):
            i += 2
            block = []
            while i < len(lines) and lines[i].strip():
                block.append(lines[i].rstrip())
                i += 1
            # значения выровнены по одному столбцу; у самого длинного ключа отступ - один пробел
            col = max((m.end() for m in (re.search(r"\S\s{2,}", l) for l in block) if m), default=0)
            for l in block:
                res["limits"][l[:col].strip()] = l[col:].strip()
            limits_done = True
            continue
        elif re.match(r"^\*{3}\s+\S.*\*{3}$", line) and limits_done:
            run = {"banner": line.strip("* ").strip(), "op_voltages": {}, "tables": []}
            res["runs"].append(run)
        elif line.strip() == "DC Operating Point Voltages" and run is not None:
            i += 3  # заголовок, ====, "Node Voltage Node Voltage"
            while i < len(lines) and lines[i].strip():
                toks = lines[i].split()
                for n, v in zip(toks[0::2], toks[1::2]):
                    if _value(v) is not None:
                        run["op_voltages"][n] = float(v)
                i += 1
            continue
        elif line.strip() == "Interpolated Waveform Values":
            if run is None:  # вывод без баннера прогона
                run = {"banner": "", "op_voltages": {}, "tables": []}
                res["runs"].append(run)
            head, units = lines[i + 2], lines[i + 3]
            i += 4
            rows = []
            while i < len(lines) and lines[i].strip() and NUM.match(lines[i].split()[0]):
                rows.append(lines[i])
                i += 1
            if rows:
                ends = _columns_by_data(rows)
                cols = _slice(head, ends)
                table = {"columns": cols, "units": _slice(units, ends), "data": {c: [] for c in cols}}
                for r in rows:
                    for c, tok in zip(cols, r.split()):
                        table["data"][c].append(_value(tok))
                run["tables"].append(table)
            continue
        i += 1
    return res


def read(path):
    return parse(Path(path).read_text(encoding="cp1251", errors="replace"))


def column(res, name, run=0):
    """Столбец по имени (без учёта регистра и пробелов) из первой таблицы прогона, где он есть."""
    key = re.sub(r"\s+", "", name).upper()
    for t in res["runs"][run]["tables"]:
        for c in t["columns"]:
            if re.sub(r"\s+", "", c).upper() == key:
                return t["data"][c]
    raise KeyError(f"нет столбца {name}; есть: " +
                   ", ".join(c for t in res["runs"][run]["tables"] for c in t["columns"]))


# ---------------------------------------------------------------- измерения
def crossing(x, y, level, log_x=True, start=0):
    """Первое x (после индекса start), где y пересекает level; линейная интерполяция по y,
    по x - в логарифмическом масштабе (для частотных характеристик) или линейном."""
    for k in range(max(start, 1), len(x)):
        y0, y1 = y[k - 1], y[k]
        if y0 is None or y1 is None or (y0 - level) * (y1 - level) > 0 or y0 == y1:
            continue
        f = (level - y0) / (y1 - y0)
        if log_x:
            return 10 ** (math.log10(x[k - 1]) + f * (math.log10(x[k]) - math.log10(x[k - 1])))
        return x[k - 1] + f * (x[k] - x[k - 1])
    return None


def db(values):
    return [20 * math.log10(abs(v)) if v not in (None, 0) else None for v in values]


def cutoff_3db_from_max(freq, gain_db):
    """Частота по уровню -3 дБ от МАКСИМУМА АЧХ (первое пересечение при движении от максимума к краю).
    Для фильтров с подъёмом (Q > 0,707) это не fc по определению eval_spec (от уровня полосы) -
    для оценки использовать measure.fc."""
    vals = [g for g in gain_db if g is not None]
    gmax = max(vals)
    k = gain_db.index(gmax)
    level = gmax - 3.0103  # 20*log10(1/sqrt(2))
    right = crossing(freq, gain_db, level, start=k + 1)
    if right is not None:
        return right
    # максимум справа (ФВЧ): ищем пересечение слева от максимума
    return crossing(freq[:k + 1], gain_db[:k + 1], level)


if __name__ == "__main__":
    print(json.dumps(read(sys.argv[1]), ensure_ascii=False, indent=1)[:20000])
