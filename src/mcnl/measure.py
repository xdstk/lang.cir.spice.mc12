#!/usr/bin/env python3
"""Измерения по АЧХ/ФЧХ и сравнение с допусками (docs/eval_spec.md, раздел 6).

Запуск:  python measure.py вывод.ANO     -> JSON с измерениями по столбцам F, DB(V(OUT)), PH(V(OUT))

Все величины - по VDB(OUT), VP(OUT) из .AC эталонной развёртки при AC 1 на входе.
  type     flat | LP | HP | BP | BS - по форме АЧХ: размах < 0,5 дБ - flat; оба края ниже максимума
           > 25 дБ - BP; разница краёв > 10 дБ - LP/HP; подъём/провал > 3 дБ относительно краёв - BP/BS
  k_dc_db  |H| на нижней частоте развёртки, дБ
  k_pass_db уровень полосы: LP - |H(fl)|, HP - |H(fh)|, flat - медиана, BP - максимум,
           BS - больший из краёв; дБ
  fc       |H| = k_pass/sqrt(2) (уровень полосы, не максимум): LP - первое пересечение снизу,
           HP - последнее сверху; логарифмическая интерполяция по частоте
  f0       LP/HP 2-го порядка - фаза -90/+90 градусов относительно фазы в полосе (ближайшее
           к фазе на краю развёртки кратное 90 градусов);
           BP/BS - sqrt(f1*f2) по точкам -3 дБ (для звеньев 2-го порядка совпадает с максимумом /
           минимумом АЧХ и не зависит от шага сетки)
  Q        LP/HP - |H(f0)|/k_pass; BP - f0/(f2-f1) по -3 дБ от максимума; BS - то же от уровня полосы
  f_u      |H| = 1 (0 дБ), первое пересечение
Рабочий диапазон Q - до 10 (eval_spec, раздел 1): при 100 точках на декаду шаг сетки 2,3%, и при
Q около 30 полоса -3 дБ (f0/Q) едва шире шага - Q полосового завышается на 13% (параллельный RLC,
одинаково в MC12 и mna.py). Генератор отсеивает Q > 10.
Величины, не имеющие смысла для данной АЧХ, - None (например, f0 у звена 1-го порядка: фаза
не доходит до 90 градусов в пределах развёртки).
"""
import json
import math
import statistics
import sys

try:
    from . import mcout
except ImportError:  # запуск как скрипта
    import mcout

DB3 = 20 * math.log10(math.sqrt(2))  # 3.0103 дБ
FLAT_DB = 0.5    # размах меньше - flat
ENDS_DB = 10.0   # разница краёв больше - LP / HP
BUMP_DB = 3.0    # подъём/провал относительно краёв больше - BP / BS
BP_DB = 25.0     # оба края ниже максимума больше чем на это - BP (подъём ФНЧ/ФВЧ при Q <= 10 - до 20 дБ)

# допуски уровня S4 (раздел 6): частоты - относительные, уровни - дБ, Q - относительный
TOL = {"fc": ("rel", 5e-3), "f0": ("rel", 5e-3), "f_u": ("rel", 5e-3),
       "k_dc_db": ("abs", 0.1), "k_pass_db": ("abs", 0.1), "Q": ("rel", 1e-2), "type": ("eq", None)}


def _clean(f, db, ph):
    pts = [(x, y, p) for x, y, p in zip(f, db, ph if ph else [None] * len(f)) if x and y is not None]
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def unwrap(ph):
    """Развёртка фазы в градусах: скачки больше 180 убираются; None сохраняются."""
    out, prev, shift = [], None, 0.0
    for p in ph:
        if p is None:
            out.append(None)
            continue
        if prev is not None:
            d = p + shift - prev
            if d > 180:
                shift -= 360
            elif d < -180:
                shift += 360
        prev = p + shift
        out.append(prev)
    return out


def value_at(f, y, x):
    """y(x) с линейной интерполяцией по lg f."""
    for k in range(1, len(f)):
        if f[k - 1] <= x <= f[k]:
            if y[k - 1] is None or y[k] is None:
                return None
            t = (math.log10(x) - math.log10(f[k - 1])) / (math.log10(f[k]) - math.log10(f[k - 1]))
            return y[k - 1] + t * (y[k] - y[k - 1])
    return None


def _last_crossing(f, y, level):
    x = mcout.crossing(f[::-1], y[::-1], level)
    return x


def _asymptote(p):
    """Фаза в полосе - ближайшее кратное 90 градусов: на краю развёртки (f0/1000) фаза звена
    2-го порядка ещё не дошла до асимптоты (при Q = 0,5 - на 0,11 градуса), и f0 смещался бы на 0,2%.
    У схем класса A с идеальным ОУ асимптотические фазы кратны 90 (0 или 180)."""
    return 90.0 * round(p / 90.0)


def classify(db):
    lo, hi, gmax, gmin = db[0], db[-1], max(db), min(db)
    if gmax - gmin < FLAT_DB:
        return "flat"
    if gmax - lo > BP_DB and gmax - hi > BP_DB:
        return "BP"  # до правила краёв: у полосы из звеньев разного порядка края затухают несимметрично
    if lo - hi > ENDS_DB:
        return "LP"
    if hi - lo > ENDS_DB:
        return "HP"
    if gmax - max(lo, hi) > BUMP_DB:
        return "BP"
    if min(lo, hi) - gmin > BUMP_DB:
        return "BS"
    return "flat"


def measure(f, db, ph=None):
    f, db, ph = _clean(f, db, ph)
    t = classify(db)
    res = {"type": t, "k_dc_db": db[0], "k_pass_db": None, "fc": None, "f0": None, "Q": None, "f_u": None}
    phu = unwrap(ph) if any(p is not None for p in ph) else None
    if t == "LP":
        kp = db[0]
        res["fc"] = mcout.crossing(f, db, kp - DB3)
        if phu:
            p_pass = _asymptote(next(p for p in phu if p is not None))
            res["f0"] = mcout.crossing(f, phu, p_pass - 90)
    elif t == "HP":
        kp = db[-1]
        res["fc"] = _last_crossing(f, db, kp - DB3)
        if phu:
            p_pass = _asymptote(next(p for p in reversed(phu) if p is not None))
            res["f0"] = _last_crossing(f, phu, p_pass + 90)
    elif t == "flat":
        kp = statistics.median(db)
    elif t == "BP":
        kp = max(db)
        k = db.index(kp)
        f1 = _last_crossing(f[:k + 1], db[:k + 1], kp - DB3)
        f2 = mcout.crossing(f[k:], db[k:], kp - DB3)
        if f1 and f2:
            res["f0"] = math.sqrt(f1 * f2)
            res["Q"] = res["f0"] / (f2 - f1)
    else:  # BS
        kp = max(db[0], db[-1])
        f1 = mcout.crossing(f, db, kp - DB3)
        f2 = _last_crossing(f, db, kp - DB3)
        if f1 and f2 and f2 > f1:
            res["f0"] = math.sqrt(f1 * f2)
            res["Q"] = res["f0"] / (f2 - f1)
    res["k_pass_db"] = kp
    if t in ("LP", "HP") and res["f0"]:
        h0 = value_at(f, db, res["f0"])
        res["Q"] = 10 ** ((h0 - kp) / 20) if h0 is not None else None
    res["f_u"] = mcout.crossing(f, db, 0.0)
    return res


def from_output(path):
    o = mcout.read(path)
    ph = None
    try:
        ph = mcout.column(o, "PH(V(OUT))")
    except KeyError:
        pass
    return measure(mcout.column(o, "F"), mcout.column(o, "DB(V(OUT))"), ph)


def within_tolerance(ref, test, keys):
    """Сравнение измерений ответа с измерениями эталона по набору величин класса.
    -> (все в допуске, {величина: (эталон, ответ, в допуске)})."""
    detail = {}
    for k in keys:
        a, b = ref.get(k), test.get(k)
        kind, tol = TOL[k]
        if a is None:
            ok = b is None
        elif b is None:
            ok = False
        elif kind == "eq":
            ok = a == b
        elif kind == "abs":
            ok = abs(a - b) <= tol + 1e-12
        else:
            ok = abs(a - b) <= tol * abs(a)
        detail[k] = (a, b, ok)
    return all(v[2] for v in detail.values()), detail


if __name__ == "__main__":
    print(json.dumps(from_output(sys.argv[1]), ensure_ascii=False, indent=1))
