#!/usr/bin/env python3
"""Сравнение нетлистов без учёта имён: изоморфизм с метками и якорями (docs/eval_spec.md, раздел 5).

Запуск:  python iso.py эталон.ckt ответ.ckt [--topology]

Граф: элементы и узлы; ребро - вывод элемента на узле. Метка элемента - тип (режим "topology")
или тип + номинал (режим "values"; номиналы равны при относительной разнице <= 1e-6). Роли выводов:
у R, C, L неразличимы, у V, I - (+, -), у OPAMP - (+, -, выход). Узлы 0, IN, OUT отображаются
только сами в себя. Схемы класса A малы (до ~30 вершин): поиск с возвратом без внешних библиотек.
"""
import json
import sys
from itertools import permutations

try:
    from . import canon
except ImportError:  # запуск как скрипта
    import canon

ANCHORS = set(canon.ANCHORS)
SYMMETRIC = set("RCL")
REL_TOL = 1e-6


def _same_value(a, b):
    if a is None or b is None:
        return a is None and b is None
    va, vb = canon.parse_value(a), canon.parse_value(b)
    if va is None or vb is None:
        return False
    return abs(va - vb) <= REL_TOL * max(abs(va), abs(vb), 1e-300)


def _labels_equal(ea, eb, mode):
    if ea["type"] != eb["type"]:
        return False
    return mode == "topology" or _same_value(ea["value"], eb["value"])


def _pin_orders(e):
    n = len(e["nodes"])
    return list(permutations(range(n))) if e["type"] in SYMMETRIC else [tuple(range(n))]


def _try_nodes(na, nb, nmap, rmap):
    """Попытка продолжить отображение узлов na -> nb. Возвращает добавленные пары или None."""
    added = []
    for x, y in zip(na, nb):
        if x in ANCHORS or y in ANCHORS:
            if x != y:
                break
        if x in nmap:
            if nmap[x] != y:
                break
            continue
        if y in rmap:
            break
        nmap[x], rmap[y] = y, x
        added.append(x)
    else:
        return added
    for x in added:
        del rmap[nmap.pop(x)]
    return None


def _order(a):
    """Порядок перебора элементов A: сначала касающиеся якорей, затем соседи уже выбранных."""
    left, order, known = set(range(len(a))), [], set(ANCHORS)
    while left:
        best = max(left, key=lambda k: (sum(n in known for n in a[k]["nodes"]), -k))
        order.append(best)
        left.discard(best)
        known.update(a[best]["nodes"])
    return order


def match(a, b, mode="values", max_steps=200000):
    """a, b - списки элементов (canon.parse_netlist). Возвращает
    {"match", "mapping" {i_a: i_b}, "node_map", "unmatched_a", "unmatched_b"}; при несовпадении
    mapping - наилучшее найденное частичное сопоставление (максимум элементов)."""
    full = _search(a, b, mode, allow_skip=False, max_steps=max_steps) \
        if len(a) == len(b) and _multiset(a, mode) == _multiset(b, mode) else None
    if full is not None:
        mapping, nmap = full
        return {"match": True, "mapping": mapping, "node_map": nmap, "unmatched_a": [], "unmatched_b": []}
    mapping, nmap = _search(a, b, mode, allow_skip=True, max_steps=max_steps)
    return {"match": False, "mapping": mapping, "node_map": nmap,
            "unmatched_a": sorted(set(range(len(a))) - set(mapping)),
            "unmatched_b": sorted(set(range(len(b))) - set(mapping.values()))}


def _multiset(elems, mode):
    """Быстрый отсев: мультимножество меток (номиналы - канонической записью)."""
    return sorted((e["type"], canon.canon_value(e["value"]) if mode == "values" and e["value"] else "")
                  for e in elems)


def _search(a, b, mode, allow_skip, max_steps):
    order = _order(a)
    used, mapping, nmap, rmap = set(), {}, {}, {}
    best = {"n": -1, "mapping": {}, "nmap": {}}
    steps = [0]

    def rec(i):
        steps[0] += 1
        if steps[0] > max_steps:
            return False
        if len(mapping) > best["n"]:
            best.update(n=len(mapping), mapping=dict(mapping), nmap=dict(nmap))
        if i == len(order):
            return not allow_skip  # полный поиск: успех; частичный - продолжаем искать максимум
        if allow_skip and len(mapping) + (len(order) - i) <= best["n"]:
            return False  # даже сопоставив все оставшиеся, не превзойти лучшее
        ka = order[i]
        ea = a[ka]
        for kb, eb in enumerate(b):
            if kb in used or not _labels_equal(ea, eb, mode):
                continue
            for perm in _pin_orders(eb):
                added = _try_nodes(ea["nodes"], [eb["nodes"][p] for p in perm], nmap, rmap)
                if added is None:
                    continue
                used.add(kb)
                mapping[ka] = kb
                if rec(i + 1):
                    return True
                del mapping[ka]
                used.discard(kb)
                for x in added:
                    del rmap[nmap.pop(x)]
        if allow_skip:
            return rec(i + 1)  # элемент A без пары
        return False

    ok = rec(0)
    if not allow_skip:
        return (dict(mapping), dict(nmap)) if ok else None
    return best["mapping"], best["nmap"]


def compare_text(ref_text, test_text):
    """Удобная обёртка: оба уровня сразу. Ошибки разбора -> {"parse_errors": [...]}."""
    a, ea = canon.parse_netlist(ref_text)
    b, eb = canon.parse_netlist(test_text)
    if ea or eb:
        return {"parse_errors": ea + eb}
    topo = match(a, b, "topology")
    vals = match(a, b, "values") if topo["match"] else None
    worst = vals if topo["match"] else topo  # уровень, на котором сравнение разошлось
    return {"topology": topo["match"], "values": bool(vals and vals["match"]),
            "unmatched_ref": [a[k]["name"] for k in worst["unmatched_a"]],
            "unmatched_test": [b[k]["name"] for k in worst["unmatched_b"]]}


if __name__ == "__main__":
    ra = open(sys.argv[1], encoding="cp1251", errors="replace").read()
    rb = open(sys.argv[2], encoding="cp1251", errors="replace").read()
    print(json.dumps(compare_text(ra, rb), ensure_ascii=False, indent=1))
