#!/usr/bin/env python3
"""Названные схемы и состав датасета v3 (docs/generator_v3_spec.md, «Состав v3»; docs/mode2_spec.md, раздел 3).

Новое относительно v2:
- активный двойной Т-мост (synth.twin_t_active) - названная схема; роли - data/roles_ru.json, topologies_v3;
- каскады Саллен-Ки по Баттерворту и Чебышёву порядка 3-6 (sk_cascade): номиналы считает synth, как в режиме 2;
- отложенные в v2 топологии (khn, twin_t, mfb_hp) входят в общий набор;
- веса набора (named_pool): фокус на фильтрах - схемы без частотной характеристики реже.
formulas.TOPOLOGIES и filters_v2.TOPOLOGIES_V2 не меняются.
"""
import inspect
import math
import random

try:
    from . import bench, canon, cascade, filters_v2, formulas, measure, mna, sampler, synth
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import cascade
    import filters_v2
    import formulas
    import measure
    import mna
    import sampler
    import synth

TOPOLOGIES_V3 = {  # имя: (функция, уровень, величины класса)
    "twin_t_active": (synth.twin_t_active, "L3", ["type", "k_pass_db", "f0", "Q"]),
}
# схемы без частотной характеристики фильтра: усилители, делитель, интегратор и дифференциатор без резистора
NOT_FILTERS = {"divider", "inverting", "noninverting", "follower", "integrator", "differentiator"}
# полосовые, режекторные и фильтры на нескольких ОУ: в v2 вес 1 (RARE), здесь выше - это схемы режима 2
BAND = {"rlc_series_bp", "rlc_parallel_bp", "mfb_bp", "rlc_series_bs", "twin_t", "twin_t_active", "tow_thomas", "khn"}
WEIGHTS = {"lp_hp": 3, "band": 2, "not_filter": 1, "cascade": 1, "sk_cascade": 4}
SK_ORDERS = (3, 4, 5, 6)
SK_RIPPLE = (0.5, 1.0, 2.0)
SK_F = (20.0, 100e3)
SK_EXACT_SHARE = filters_v2.EXACT_SHARE  # доля каскадов с точными номиналами (как у дизайнера), остальные - из рядов


def names():
    return list(formulas.TOPOLOGIES) + list(filters_v2.TOPOLOGIES_V2) + list(TOPOLOGIES_V3)


def named_pool():
    """Названные схемы train v3 и их веса: {имя: вес}. «cascade:ВИД» - каскады v1, «sk_cascade» - каскады Саллен-Ки."""
    pool = {t: WEIGHTS["not_filter" if t in NOT_FILTERS else "band" if t in BAND else "lp_hp"] for t in names()}
    pool.update({"cascade:" + k: WEIGHTS["cascade"] for k in cascade.KINDS})
    pool["sk_cascade"] = WEIGHTS["sk_cascade"]
    return pool


def filter_share(pool=None):
    """Доля фильтров в наборе по весам (каскады - фильтры, кроме доли усилителей в них)."""
    pool = pool or named_pool()
    return 1 - sum(w for t, w in pool.items() if t in NOT_FILTERS) / sum(pool.values())


# ---------------------------------------------------------------- активный двойной Т-мост
def reference(topology, params, ref_id=None):
    """Эталон схемы TOPOLOGIES_V3 в формате filters_v2.reference."""
    fn, level, quantities = TOPOLOGIES_V3[topology]
    net, analytic = fn(**params)
    elems = canon.canonicalize(canon.parse_netlist(net)[0])
    f_char = filters_v2.characteristic_frequency(elems)
    lo, hi = bench.ac_range(f_char)
    m = measure.measure(*mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE))
    return {"id": ref_id or topology, "class": topology, "level": level, "params": params, "analytic": analytic,
            "netlist": canon.canonical_netlist(net), "expected": {k: m.get(k) for k in quantities},
            "quantities": quantities, "f_char": f_char}


def sample(topology, rng, max_tries=2000):
    """Случайные номиналы v2 (ряды и точные), ограничения sampler.acceptable."""
    params = [p for p in inspect.signature(TOPOLOGIES_V3[topology][0]).parameters if p != "dc"]
    for _ in range(max_tries):
        p = {n: filters_v2.pick(n[0], rng) for n in params}
        p["dc"] = rng.choice(sampler.DC_CHOICES)
        try:
            ref = reference(topology, p)
        except (ValueError, ZeroDivisionError, KeyError, TypeError):
            continue
        if not filters_v2.spread_ok(ref["netlist"]) or any(ref["expected"][k] is None for k in ref["quantities"]):
            continue
        if ref["expected"]["type"] == ref["analytic"]["type"] and sampler.acceptable(ref)[0]:
            return ref
    raise RuntimeError(f"{topology}: не найдено допустимых номиналов за {max_tries} попыток")


# ---------------------------------------------------------------- каскады Саллен-Ки
def _round_freq(f, rng):
    return float(f"{f:.{rng.choice([0, 1, 1, 2])}e}")  # 2 кГц, 1,5 кГц, 3,47 кГц


def sk_cascade(rng, max_tries=200):
    """Активный ФНЧ или ФВЧ порядка 3-6 по Баттерворту или Чебышёву: звенья Саллен-Ки (K = 1) и, при нечётном
    порядке, RC-звено в конце; номиналы - расчёт режима 2 (synth._cascade). Эталон - в формате каскадов v1:
    params - список звеньев с topology; design - характеристики, по которым каскад рассчитан."""
    for _ in range(max_tries):
        typ = rng.choice(["LP", "LP", "HP"])
        response = rng.choice(["Butterworth", "Butterworth", "Chebyshev"])
        raw = {"type": typ, "order": str(rng.choice(SK_ORDERS)), "active": "yes", "response": response,
               "fc": canon.format_value(_round_freq(10 ** rng.uniform(*map(math.log10, SK_F)), rng))}
        if response == "Chebyshev":
            raw["ripple"] = canon.format_value(rng.choice(SK_RIPPLE))
        spec, errors = synth.resolve_spec(raw)
        if errors:
            continue
        try:
            stages = synth._cascade(spec)
        except ValueError:
            continue
        exact = rng.random() < SK_EXACT_SHARE
        dc = rng.choice(sampler.DC_CHOICES)
        params = []
        for i, (top, p, q, _t) in enumerate(stages):
            vals = {k: float(f"{v:.5g}") for k, v in (p if exact else q).items()}
            params.append(dict(vals, topology=top, **({"dc": dc} if i == 0 else {})))
        nets = [{"netlist": synth._builder(s["topology"])(**{k: v for k, v in s.items() if k != "topology"})[0]} for s in params]
        netlist = cascade.compose(nets)
        if not filters_v2.spread_ok(netlist):
            continue
        f_char = spec["f"]
        exp = cascade.expected_by_mna(netlist, f_char)
        if exp["type"] != typ or not exp.get("fc") or not sampler.F_RANGE[0] <= exp["fc"] <= sampler.F_RANGE[1]:
            continue
        f_char = exp["fc"]
        exp = cascade.expected_by_mna(netlist, f_char)  # развёртка обвязки - вокруг измеренного среза, как у каскадов v1
        if exp["type"] != typ or not exp.get("fc"):
            continue
        q = ["type", "k_pass_db", "fc"]
        return {"id": "sk_cascade", "class": "cascade:" + "+".join(s["topology"] for s in params), "level": "L3",
                "params": params, "netlist": netlist, "expected": {k: exp[k] for k in q}, "quantities": q, "f_char": f_char,
                "design": {k: raw[k] for k in ("type", "order", "response", "fc", "ripple") if k in raw}}
    raise RuntimeError("каскад Саллен-Ки не собран")


def design_name(ref, rng):
    """Название рассчитанного каскада для запроса: «активный ФНЧ Баттерворта 4-го порядка на 16 кГц». Частота -
    та, по которой каскад рассчитан (у Чебышёва - граница неравномерности); в эталон она не входит."""
    d = ref["design"]
    kind = {"LP": rng.choice(["ФНЧ", "фильтр нижних частот"]), "HP": rng.choice(["ФВЧ", "фильтр верхних частот"])}[d["type"]]
    who = "Баттерворта" if d["response"] == "Butterworth" else "Чебышёва"
    f = canon.parse_value(d["fc"])
    freq = f"{f / 1e3:g} кГц" if f >= 1e3 else f"{f:g} Гц"
    text = f"{rng.choice(['активный ', ''])}{kind} {who} {d['order']}-го порядка"
    if d["response"] == "Chebyshev" and rng.random() < 0.5:
        text += f" (неравномерность {canon.parse_value(d['ripple']):g} дБ)"
    if rng.random() < 0.6:
        text += rng.choice([" на ", ", срез "]) + freq
    return text.replace(".", ",") if rng.random() < 0.7 else text


# ---------------------------------------------------------------- выборка названной схемы
def sample_named(name, rng):
    """Эталон по имени из named_pool."""
    if name == "sk_cascade":
        return sk_cascade(rng)
    if name.startswith("cascade:"):
        return cascade.sample(name.split(":", 1)[1], rng)
    if name in TOPOLOGIES_V3:
        return sample(name, rng)
    return filters_v2.sample_any(name, rng)


if __name__ == "__main__":
    import json
    import sys
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    print(json.dumps(sample_named(sys.argv[1] if len(sys.argv) > 1 else "sk_cascade", rng), ensure_ascii=False, indent=1))
