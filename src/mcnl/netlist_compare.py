#!/usr/bin/env python3
"""Сравнение двух SPICE-нетлистов по топологии и номиналам, без учёта имён узлов.

Запуск:  python netlist_compare.py эталон.ckt проверяемый.ckt

Топология считается совпавшей, если совпадают разбиения выводов элементов по узлам:
вывод задаётся парой (имя элемента, номер вывода), узел - множеством таких пар;
земля (узел 0) сравнивается отдельно. Имена узлов не важны, порядок строк тоже.
Имена элементов сравниваются без учёта регистра и символов '_' (X_O1 == XO1).

Годится для двух задач: проверка конвертера .CIR -> SPICE против экспорта самого MC12
и метрика "топология совпала" при оценке сгенерированных моделью нетлистов.
Ограничение: элементы сопоставляются по именам; если модель назвала резистор иначе,
нужен поиск изоморфизма графов (не реализован в прототипе).
"""
import re
import sys

NPINS = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "B": 2, "Q": 3, "J": 3,
         "M": 4, "E": 4, "G": 4, "F": 2, "H": 2, "S": 4, "W": 2}
BEHAVIORAL = ("VALUE", "TABLE", "LAPLACE", "FREQ")  # E/G в форме PSpice: только два узла
SUF = [("MEG", 1e6), ("T", 1e12), ("G", 1e9), ("K", 1e3), ("M", 1e-3), ("U", 1e-6),
       ("N", 1e-9), ("P", 1e-12), ("F", 1e-15)]


def num(tok):
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:E[+-]?\d+)?)([A-Z]*)$", tok.upper())
    if not m:
        return None
    v, s = float(m.group(1)), m.group(2)
    for k, mult in SUF:
        if s.startswith(k):
            return v * mult
    return v


def norm_name(n):
    return re.sub(r"[_\W]", "", n.upper())


def parse(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    out = []
    for l in lines:
        s = l.strip()
        if s.startswith("+") and out:
            out[-1] += " " + s[1:]
        elif s and not s.startswith("*"):
            out.append(s.split(";")[0].strip())
    elems, in_sub = {}, False
    for s in out:
        t = s.split()
        k = t[0].upper()
        if k == ".SUBCKT":
            in_sub = True
            continue
        if k == ".ENDS":
            in_sub = False
            continue
        if in_sub or k.startswith("."):
            continue
        letter = k[0]
        if letter == "X":
            toks = [x for x in t[1:] if "=" not in x and x.upper() != "PARAMS:"]
            nodes, rest = toks[:-1], toks[-1:]
        elif letter in "EG" and len(t) > 3 and t[3].upper().startswith(BEHAVIORAL):
            nodes, rest = t[1:3], []
        elif letter in NPINS:
            nodes, rest = t[1:1 + NPINS[letter]], t[1 + NPINS[letter]:]
        else:
            continue
        val = None
        for r in rest:
            v = num(r)
            if v is not None:
                val = v
                break
        elems[norm_name(t[0])] = {"nodes": [n.upper() for n in nodes], "value": val, "raw": s}
    return elems


SYMMETRIC = set("RCL")  # у R, C, L выводы равноправны: ориентация не важна


def partition(elems):
    from collections import Counter
    groups = {}
    for name, e in elems.items():
        for i, n in enumerate(e["nodes"]):
            key = "0" if n in ("0", "GND") else n
            role = "*" if name[0] in SYMMETRIC else i
            groups.setdefault(key, Counter())[(name, role)] += 1
    sig = lambda c: frozenset(c.items())
    ground = sig(groups.pop("0", Counter()))
    from collections import Counter as C2
    return ground, C2(sig(g) for g in groups.values())


def compare(a, b):
    ea, eb = parse(a), parse(b)
    res = {"only_in_ref": sorted(set(ea) - set(eb)), "only_in_test": sorted(set(eb) - set(ea))}
    common = set(ea) & set(eb)
    ca = {k: ea[k] for k in common}
    cb = {k: eb[k] for k in common}
    ga, pa = partition(ca)
    gb, pb = partition(cb)
    res["ground_match"] = ga == gb
    res["topology_match"] = (pa == pb) and ga == gb and not res["only_in_ref"] and not res["only_in_test"]
    res["value_mismatch"] = sorted(k for k in common
                                   if ea[k]["value"] is not None and eb[k]["value"] is not None
                                   and abs(ea[k]["value"] - eb[k]["value"]) > 1e-9 * max(abs(ea[k]["value"]), 1e-30))
    res["nodes_ref"], res["nodes_test"] = len(pa), len(pb)
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=1))
