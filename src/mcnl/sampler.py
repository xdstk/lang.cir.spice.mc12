#!/usr/bin/env python3
"""Выборка номиналов для генератора (docs/generator_spec.md, раздел 5 и решение В3).

Номиналы - случайные значения рядов (R - E24, C и L - E12), лог-равномерно по декадам диапазона;
схема отсеивается, если характерная частота вне 10 Гц - 1 МГц, усиление вне допустимого или Q вне
0,5-10 (для звеньев, где Q измеряется: при Q > 10 измерение на сетке 100 точек на декаду неточно,
см. measure.py). Детерминирована по seed (random.Random).
"""
import inspect
import math
import random

try:
    from . import formulas
except ImportError:  # запуск как скрипта
    import formulas

RANGES = {"R": (100.0, 1e6), "C": (100e-12, 10e-6), "L": (1e-6, 100e-3)}
RANGES_GEN = dict(RANGES, R=(100.0, 10e6))  # gen_range: R до 10 МОм; генератор берёт схемы, где хотя бы один R > 1 МОм
F_RANGE = (10.0, 1e6)
GAIN_DB = (-40.0, 60.0)       # |K| от 0,01 до 1000
Q_RANGE = (0.5, 10.0)
DC_CHOICES = [0, 0, 0, 0, 0, 0, 1, 2.5, 5, 10, 12, -5]  # чаще «сигнал без уровня» (DC 0)
MIN_INTEGRATOR_K = 10.0       # иначе f_u сливается с fc
MIN_GAIN_DB = {"inverting": 0.0, "active_lp_inv": -6.0, "active_hp_inv": -6.0, "mfb_bp": -6.0}


def series_values(kind, lo, hi):
    """Все значения ряда (E24 для R, E12 для C/L) в [lo, hi]."""
    base = formulas.E24 if kind == "R" else formulas.E12
    out = []
    for d in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in base:
            v = float(f"{m * 10 ** d:.6g}")
            if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                out.append(v)
    return sorted(set(out))


def pick(kind, rng, ranges=RANGES):
    """Лог-равномерно по диапазону: значение ряда, ближайшее к случайной точке на лог-шкале."""
    vals = series_values(kind, *ranges[kind])
    x = 10 ** rng.uniform(math.log10(vals[0]), math.log10(vals[-1]))
    return min(vals, key=lambda v: abs(math.log10(v / x)))


def _param_names(topology):
    fn = formulas.TOPOLOGIES[topology][0]
    return [p for p in inspect.signature(fn).parameters if p != "dc"], inspect.signature(fn).parameters


def draw(topology, rng, ranges=RANGES, pick=pick):
    """Один набор параметров без проверки ограничений."""
    names, sig = _param_names(topology)
    params = {}
    gain_variant = rng.random() < 0.5  # у Саллена-Ки Ra/Rb - необязательные (повторитель или K > 1)
    for n in names:
        if sig[n].default is None and not gain_variant:
            continue
        params[n] = pick(n[0], rng, ranges)
    params["dc"] = rng.choice(DC_CHOICES)
    return params


def acceptable(ref):
    """Ограничения раздела 5 по формулам эталона -> (годится, причина)."""
    e, q = ref["expected"], ref["quantities"]
    if ref["f_char"] is not None and not (F_RANGE[0] <= ref["f_char"] <= F_RANGE[1]):
        return False, "f_char"
    for k in ("k_pass_db", "k_dc_db"):
        if k in e and not (GAIN_DB[0] <= e[k] <= GAIN_DB[1]):
            return False, k
    if "Q" in q and not (e.get("Q") and Q_RANGE[0] <= e["Q"] <= Q_RANGE[1]):
        return False, "Q"
    for k in ("fc", "f0", "f_u"):  # измеряемые частоты должны лежать внутри развёртки обвязки
        if k in q and e.get(k) and ref["f_char"] and not (ref["f_char"] / 300 <= e[k] <= ref["f_char"] * 300):
            return False, k
    if ref["class"] == "integrator_rf" and 10 ** (e["k_dc_db"] / 20) < MIN_INTEGRATOR_K:
        return False, "integrator_k"
    if ref["class"] in MIN_GAIN_DB and e["k_pass_db"] < MIN_GAIN_DB[ref["class"]]:
        return False, "gain_low"  # «усилитель» с K = 0,01 формально допустим, но нелеп
    return True, None


def sample(topology, rng, ranges=RANGES, max_tries=2000, ref_id=None, pick=pick):
    """Эталон (formulas.reference) со случайными номиналами, прошедший ограничения.
    -> (эталон, число отвергнутых попыток по причинам)."""
    rejected = {}
    for _ in range(max_tries):
        params = draw(topology, rng, ranges, pick)
        try:
            ref = formulas.reference(topology, params, ref_id)
        except (ValueError, ZeroDivisionError) as err:
            rejected["formula"] = rejected.get("formula", 0) + 1
            continue
        ok, why = acceptable(ref)
        if ok:
            return ref, rejected
        rejected[why] = rejected.get(why, 0) + 1
    raise RuntimeError(f"{topology}: не найдено допустимых номиналов за {max_tries} попыток: {rejected}")


if __name__ == "__main__":
    import json
    import sys
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    ref, rej = sample(sys.argv[1], rng)
    print(json.dumps({"ref": ref, "rejected": rej}, ensure_ascii=False, indent=1))
