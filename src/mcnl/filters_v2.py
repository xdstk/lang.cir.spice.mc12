#!/usr/bin/env python3
"""Топологии датасета v2 (docs/generator_v2_spec.md, раздел 2.1).

Отдельно от formulas.TOPOLOGIES: генератор v1 должен воспроизводиться, а baseline знает только v1.
Функция топологии даёт нетлист и аналитические значения там, где они есть (проверяются в тестах);
ожидаемые величины эталона - из узлового расчёта mna.py, как у каскадов. Структуры на нескольких ОУ и
MFB совпадают со схемами дизайнера MC12 (data/designer, проверка изоморфизма в tests/test_filters_v2.py).
"""
import inspect
import math

try:
    from . import bench, canon, measure, mna, sampler
    from .formulas import TWO_PI, _net, _v, db, fc_2nd_lp
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import measure
    import mna
    import sampler
    from formulas import TWO_PI, _net, _v, db, fc_2nd_lp

WIDE = (0.1, 1e9)  # развёртка для оценки характерной частоты
EXACT_SHARE = 0.3  # доля номиналов не из рядов, как у дизайнера MC12 (раздел 3)
MAX_SPREAD = 1000.0  # разброс номиналов одного типа внутри схемы


def _f0(L, C):
    return 1 / (TWO_PI * math.sqrt(L * C))


# ---------------------------------------------------------------- пассивные
def rlc_series_lp(R, L, C, dc=0):
    """Последовательный RLC, выход на C: ФНЧ 2-го порядка."""
    f0, Q = _f0(L, C), math.sqrt(L / C) / R
    return _net([f"R1 IN N1 {_v(R)}", f"L1 N1 OUT {_v(L)}", f"C1 OUT 0 {_v(C)}"], dc), \
        {"type": "LP", "k_pass_db": 0.0, "f0": f0, "Q": Q}


def rlc_series_hp(R, L, C, dc=0):
    """Последовательный RLC, выход на L: ФВЧ 2-го порядка."""
    f0, Q = _f0(L, C), math.sqrt(L / C) / R
    return _net([f"R1 IN N1 {_v(R)}", f"C1 N1 OUT {_v(C)}", f"L1 OUT 0 {_v(L)}"], dc), \
        {"type": "HP", "k_pass_db": 0.0, "f0": f0, "Q": Q}


def rlc_series_bs(R, L, C, dc=0):
    """R от входа к выходу, с выхода на землю последовательный LC: режекторный на f0."""
    return _net([f"R1 IN OUT {_v(R)}", f"L1 OUT N1 {_v(L)}", f"C1 N1 0 {_v(C)}"], dc), \
        {"type": "BS", "k_pass_db": 0.0, "f0": _f0(L, C)}


def rc_ladder3(R1, C1, R2, C2, R3, C3, dc=0):
    """Трёхзвенная RC-лестница."""
    return _net([f"R1 IN N1 {_v(R1)}", f"C1 N1 0 {_v(C1)}", f"R2 N1 N2 {_v(R2)}", f"C2 N2 0 {_v(C2)}",
                 f"R3 N2 OUT {_v(R3)}", f"C3 OUT 0 {_v(C3)}"], dc), {"type": "LP", "k_pass_db": 0.0}


def lc_ladder3(Rs, L1, C1, L2, Rl, dc=0):
    """LC-лестница 3-го порядка с сопротивлениями источника и нагрузки."""
    return _net([f"R1 IN N1 {_v(Rs)}", f"L1 N1 N2 {_v(L1)}", f"C1 N2 0 {_v(C1)}", f"L2 N2 OUT {_v(L2)}",
                 f"R2 OUT 0 {_v(Rl)}"], dc), {"k_pass_db": db(Rl / (Rs + Rl))}


def lc_ladder5(Rs, L1, C1, L2, C2, L3, Rl, dc=0):
    """LC-лестница 5-го порядка с сопротивлениями источника и нагрузки."""
    return _net([f"R1 IN N1 {_v(Rs)}", f"L1 N1 N2 {_v(L1)}", f"C1 N2 0 {_v(C1)}", f"L2 N2 N3 {_v(L2)}",
                 f"C2 N3 0 {_v(C2)}", f"L3 N3 OUT {_v(L3)}", f"R2 OUT 0 {_v(Rl)}"], dc), \
        {"k_pass_db": db(Rl / (Rs + Rl))}


def twin_t(R, C, dc=0):
    """Двойной Т-мост без нагрузки: режекция на 1/(2 pi R C)."""
    return _net([f"R1 IN N1 {_v(R)}", f"R2 N1 OUT {_v(R)}", f"C3 N1 0 {_v(2 * C)}", f"C1 IN N2 {_v(C)}",
                 f"C2 N2 OUT {_v(C)}", f"R3 N2 0 {_v(R / 2)}"], dc), \
        {"type": "BS", "k_pass_db": 0.0, "f0": 1 / (TWO_PI * R * C)}


# ---------------------------------------------------------------- на одном ОУ
def mfb_lp(R1, R2, R3, C1, C2, dc=0):
    """ФНЧ с многопетлевой ОС: R1 IN-N1, R2 N1-OUT, R3 N1-N2, C1 N1-0, C2 N2-OUT; K = -R2/R1."""
    w0 = 1 / math.sqrt(R2 * R3 * C1 * C2)
    return _net([f"R1 IN N1 {_v(R1)}", f"R2 N1 OUT {_v(R2)}", f"R3 N1 N2 {_v(R3)}", f"C1 N1 0 {_v(C1)}",
                 f"C2 N2 OUT {_v(C2)}", "X1 0 N2 OUT OPAMP_IDEAL"], dc), \
        {"type": "LP", "k_pass_db": db(R2 / R1), "f0": w0 / TWO_PI, "Q": w0 * C1 / (1 / R1 + 1 / R2 + 1 / R3)}


def mfb_hp(R1, R2, C1, C2, C3, dc=0):
    """ФВЧ с многопетлевой ОС: C1 IN-N1, C2 N1-N2, C3 N1-OUT, R1 N1-0, R2 N2-OUT; K = -C1/C3."""
    w0 = 1 / math.sqrt(R1 * R2 * C2 * C3)
    return _net([f"C1 IN N1 {_v(C1)}", f"C2 N1 N2 {_v(C2)}", f"C3 N1 OUT {_v(C3)}", f"R1 N1 0 {_v(R1)}",
                 f"R2 N2 OUT {_v(R2)}", "X1 0 N2 OUT OPAMP_IDEAL"], dc), \
        {"type": "HP", "k_pass_db": db(C1 / C3), "f0": w0 / TWO_PI, "Q": w0 * R2 * C2 * C3 / (C1 + C2 + C3)}


def integrator(R, C, dc=0):
    """Интегратор без резистора в ОС: f_u = 1/(2 pi R C); усиление на низких частотах ограничено A ОУ."""
    return _net([f"R1 IN N1 {_v(R)}", f"C1 N1 OUT {_v(C)}", "X1 0 N1 OUT OPAMP_IDEAL"], dc), \
        {"f_u": 1 / (TWO_PI * R * C)}


def differentiator(R, C, dc=0):
    """Дифференциатор без резистора на входе: f_u = 1/(2 pi R C)."""
    return _net([f"C1 IN N1 {_v(C)}", f"R1 N1 OUT {_v(R)}", "X1 0 N1 OUT OPAMP_IDEAL"], dc), \
        {"f_u": 1 / (TWO_PI * R * C)}


# ---------------------------------------------------------------- на трёх ОУ
def tow_thomas(Rin, Rf, Rq, Ri, Ra, C1, C2, dc=0):
    """Tow-Thomas, ФНЧ-выход за инвертором: сумматор-интегратор с потерями, интегратор, инвертор."""
    w0 = 1 / math.sqrt(Rf * Ri * C1 * C2)
    return _net([f"R1 IN N1 {_v(Rin)}", f"R2 N1 OUT {_v(Rf)}", f"R3 N1 N2 {_v(Rq)}", f"C1 N1 N2 {_v(C1)}",
                 f"R4 N2 N3 {_v(Ri)}", f"C2 N3 N4 {_v(C2)}", f"R5 N4 N5 {_v(Ra)}", f"R6 N5 OUT {_v(Ra)}",
                 "X1 0 N1 N2 OPAMP_IDEAL", "X2 0 N3 N4 OPAMP_IDEAL", "X3 0 N5 OUT OPAMP_IDEAL"], dc), \
        {"type": "LP", "k_pass_db": db(Rf / Rin), "f0": w0 / TWO_PI, "Q": w0 * Rq * C1}


def khn(R1, R2, R3, R4, R5, R6, C1, C2, dc=0):
    """KHN (переменные состояния), ФНЧ-выход: сумматор и два интегратора; R1, R2 задают Q, R3, R5 - частоту."""
    w0 = math.sqrt(R6 / (R3 * R4 * R5 * C1 * C2))
    return _net([f"R1 IN N1 {_v(R1)}", f"R2 N1 N4 {_v(R2)}", f"R3 N2 N4 {_v(R3)}", f"R4 N5 OUT {_v(R4)}",
                 f"R5 N3 N6 {_v(R5)}", f"R6 N5 N6 {_v(R6)}", f"C1 N2 OUT {_v(C1)}", f"C2 N3 N4 {_v(C2)}",
                 "X1 0 N2 OUT OPAMP_IDEAL", "X2 0 N3 N4 OPAMP_IDEAL", "X3 N1 N5 N6 OPAMP_IDEAL"], dc), \
        {"type": "LP", "f0": w0 / TWO_PI}


TOPOLOGIES_V2 = {  # имя: (функция, уровень, величины класса)
    "rlc_series_lp": (rlc_series_lp, "L1", ["type", "k_pass_db", "fc"]),
    "rlc_series_hp": (rlc_series_hp, "L1", ["type", "k_pass_db", "fc"]),
    "rlc_series_bs": (rlc_series_bs, "L1", ["type", "k_pass_db", "f0"]),
    "rc_ladder3": (rc_ladder3, "L1", ["type", "k_pass_db", "fc"]),
    "lc_ladder3": (lc_ladder3, "L1", ["k_pass_db"]),
    "lc_ladder5": (lc_ladder5, "L1", ["k_pass_db"]),
    "twin_t": (twin_t, "L1", ["type", "k_pass_db", "f0"]),
    "mfb_lp": (mfb_lp, "L2", ["type", "k_pass_db", "fc"]),
    "mfb_hp": (mfb_hp, "L2", ["type", "k_pass_db", "fc"]),
    "integrator": (integrator, "L2", ["f_u"]),
    "differentiator": (differentiator, "L2", ["f_u"]),
    "tow_thomas": (tow_thomas, "L3", ["type", "k_pass_db", "fc"]),
    "khn": (khn, "L3", ["type", "k_pass_db", "fc"]),
}

# значения для проверки в MC12 (id, топология, параметры); числа - как у дизайнера или типовые
PILOT = [
    ("v2_rlc_series_lp", "rlc_series_lp", dict(R=100.0, L=1e-3, C=100e-9)),
    ("v2_rlc_series_hp", "rlc_series_hp", dict(R=220.0, L=2.2e-3, C=47e-9)),
    ("v2_rlc_series_bs", "rlc_series_bs", dict(R=1e3, L=10e-3, C=100e-9)),
    ("v2_rc_ladder3", "rc_ladder3", dict(R1=1e3, C1=100e-9, R2=4.7e3, C2=22e-9, R3=22e3, C3=4.7e-9)),
    ("v2_lc_ladder3", "lc_ladder3", dict(Rs=50.0, L1=22e-6, C1=22e-9, L2=22e-6, Rl=50.0)),
    ("v2_lc_ladder5", "lc_ladder5", dict(Rs=50.0, L1=12e-6, C1=15e-9, L2=33e-6, C2=15e-9, L3=12e-6, Rl=50.0)),
    ("v2_twin_t", "twin_t", dict(R=10e3, C=10e-9)),
    ("v2_mfb_lp", "mfb_lp", dict(R1=10e3, R2=10e3, R3=22e3, C1=22e-9, C2=2.2e-9)),
    ("v2_mfb_hp", "mfb_hp", dict(R1=8.2e3, R2=39e3, C1=10e-9, C2=10e-9, C3=10e-9)),
    ("v2_integrator", "integrator", dict(R=10e3, C=100e-9)),
    ("v2_differentiator", "differentiator", dict(R=100e3, C=10e-9)),
    ("v2_tow_thomas", "tow_thomas", dict(Rin=18e3, Rf=18e3, Rq=12e3, Ri=18e3, Ra=12e3, C1=8.2e-9, C2=8.2e-9)),
    ("v2_khn", "khn", dict(R1=18e3, R2=18e3, R3=24e3, R4=18e3, R5=13e3, R6=18e3, C1=8.2e-9, C2=8.2e-9)),
]


def characteristic_frequency(elems):
    """Частота для развёртки обвязки: fc (LP/HP), f0 (BP/BS), None (flat)."""
    m = measure.measure(*mna.sweep(elems, *WIDE, per_decade=50))
    if m["type"] in ("BP", "BS"):
        return m["f0"]
    return m.get("fc")


def reference(topology, params, ref_id=None):
    """Эталон в формате formulas.reference; expected - из mna.py на сетке обвязки."""
    fn, level, quantities = TOPOLOGIES_V2[topology]
    net, analytic = fn(**params)
    elems = canon.canonicalize(canon.parse_netlist(net)[0])
    # у интегратора и дифференциатора нет полосы: рабочая точка - частота единичного усиления
    f_char = analytic.get("f_u") or characteristic_frequency(elems)
    lo, hi = bench.ac_range(f_char)
    m = measure.measure(*mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE))
    return {"id": ref_id or topology, "class": topology, "level": level, "params": params, "analytic": analytic,
            "netlist": canon.canonical_netlist(net), "expected": {k: m[k] for k in quantities},
            "quantities": quantities, "f_char": f_char}


def pick(kind, rng, ranges=sampler.RANGES):
    """Номинал v2: значение ряда или точное, 4-5 значащих цифр."""
    if rng.random() >= EXACT_SHARE:
        return sampler.pick(kind, rng, ranges)
    lo, hi = ranges[kind]
    x = 10 ** rng.uniform(math.log10(lo), math.log10(hi))
    return float(f"{x:.{rng.choice([3, 4])}e}")


def spread_ok(netlist):
    elems = canon.parse_netlist(netlist)[0]
    for k in "RCL":
        vals = [canon.parse_value(e["value"]) for e in elems if e["type"] == k]
        if vals and max(vals) / min(vals) > MAX_SPREAD:
            return False
    return True


def sample_any(topology, rng, max_tries=2000):
    """Эталон топологии v1 или v2 с номиналами v2."""
    if topology in TOPOLOGIES_V2:
        return sample(topology, rng, max_tries=max_tries)[0]
    for _ in range(50):
        ref = sampler.sample(topology, rng, max_tries=max_tries, pick=pick)[0]
        if spread_ok(ref["netlist"]):
            return ref
    raise RuntimeError(f"{topology}: разброс номиналов")


def sample(topology, rng, ranges=sampler.RANGES, max_tries=2000, ref_id=None):
    """Случайные номиналы, ограничения sampler.acceptable -> (эталон, отвергнутые по причинам)."""
    names = [p for p in inspect.signature(TOPOLOGIES_V2[topology][0]).parameters if p != "dc"]
    rejected = {}
    for _ in range(max_tries):
        params = {n: pick(n[0], rng, ranges) for n in names}
        try:
            ref = reference(topology, params, ref_id)
        except (ValueError, ZeroDivisionError, KeyError, TypeError):
            rejected["formula"] = rejected.get("formula", 0) + 1
            continue
        if not spread_ok(ref["netlist"]):
            rejected["spread"] = rejected.get("spread", 0) + 1
            continue
        if any(ref["expected"][k] is None for k in ref["quantities"]):
            rejected["measure"] = rejected.get("measure", 0) + 1
            continue
        ok, why = sampler.acceptable(ref)
        if ok:
            return ref, rejected
        rejected[why] = rejected.get(why, 0) + 1
    raise RuntimeError(f"{topology}: не найдено допустимых номиналов за {max_tries} попыток")
