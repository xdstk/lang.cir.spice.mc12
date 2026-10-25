#!/usr/bin/env python3
"""Каскады L3 (docs/generator_spec.md, раздел 3 и решение В2).

Звенья - схемы на ОУ из formulas.TOPOLOGIES; выход ОУ - идеальный источник напряжения, поэтому звенья
не нагружают друг друга и H = H1*H2*... Сочетания: ФНЧ+ФНЧ, ФВЧ+ФВЧ, усилитель + фильтр,
ФВЧ + ФНЧ с разносом частот не меньше 10 (полоса). Ожидаемые значения - узловой расчёт mna.py
тем же измерением (measure.py) на той же сетке, что и для MC12; формулы для каскада не нужны.
"""
import math
import random

try:
    from . import bench, canon, measure, mna, sampler
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import measure
    import mna
    import sampler

LP = ["active_lp_inv", "sallen_key_lp"]
HP = ["active_hp_inv", "sallen_key_hp"]
AMP = ["inverting", "noninverting"]
KINDS = ["LP+LP", "HP+HP", "AMP+LP", "AMP+HP", "HP+LP", "LP+LP+LP", "AMP+LP+LP"]
BAND_RATIO = 10.0
STAGE_Q_MAX = 3.0  # подъём произведения звеньев остаётся ниже порога BP в measure.classify (25 дБ)
QUANTITIES = {"LP": ["type", "k_pass_db", "fc"], "HP": ["type", "k_pass_db", "fc"],
              "BP": ["type", "k_pass_db", "f0", "Q"]}


def _stage_lines(netlist, i, n_stages):
    """Строки звена i с переименованными узлами: IN -> вход звена, OUT -> выход звена,
    внутренние Nk -> S<i>Nk. Источник остаётся только у первого звена."""
    inp = "IN" if i == 0 else f"L{i}"
    out = "OUT" if i == n_stages - 1 else f"L{i + 1}"
    lines = []
    for e in canon.parse_netlist(netlist)[0]:
        if e["type"] == "V" and i > 0:
            continue
        nodes = [inp if n == "IN" else out if n == "OUT" else n if n == "0" else f"S{i}{n}" for n in e["nodes"]]
        lines.append(dict(e, nodes=nodes, name=f"{e['name']}S{i}"))
    return lines


def compose(stage_refs):
    elems = []
    for i, r in enumerate(stage_refs):
        elems += _stage_lines(r["netlist"], i, len(stage_refs))
    return canon.to_netlist(canon.canonicalize(elems))


def _stage_freq(r):
    e = r["expected"]
    return e.get("fc") or e.get("f0")


def expected_by_mna(netlist, f_char):
    lo, hi = bench.ac_range(f_char)
    return measure.measure(*mna.sweep(canon.parse_netlist(netlist)[0], lo, hi, per_decade=bench.POINTS_PER_DECADE))


def sample(kind, rng, ranges=sampler.RANGES, max_tries=500, ref_id=None):
    """Каскад вида kind ('LP+LP', 'AMP+HP', 'HP+LP', ...) -> эталон в формате formulas.reference."""
    parts = kind.split("+")
    pools = {"LP": LP, "HP": HP, "AMP": AMP}
    for _ in range(max_tries):
        stages = [sampler.sample(rng.choice(pools[p]), rng, ranges)[0] for p in parts]
        if any(s["expected"].get("Q", 0) > STAGE_Q_MAX for s in stages):
            continue
        for s in stages[1:]:
            s["params"]["dc"] = 0
        freqs = [_stage_freq(s) for s in stages if _stage_freq(s)]
        if kind == "HP+LP" and _stage_freq(stages[1]) < BAND_RATIO * _stage_freq(stages[0]):
            continue
        gain = sum(s["expected"]["k_pass_db"] for s in stages)
        if not (sampler.GAIN_DB[0] <= gain <= sampler.GAIN_DB[1]):
            continue
        if len(freqs) > 1 and kind != "HP+LP" and max(freqs) / min(freqs) > 30:
            continue  # звенья одного типа с далёкими частотами - фактически одно звено в полосе развёртки
        f_char = math.sqrt(freqs[0] * freqs[-1]) if kind == "HP+LP" else math.exp(sum(map(math.log, freqs)) / len(freqs))
        netlist = compose(stages)
        exp = expected_by_mna(netlist, f_char)
        want = "BP" if kind == "HP+LP" else ("LP" if parts[-1] == "LP" else "HP")
        if exp["type"] != want:
            continue
        q = QUANTITIES[want]
        if any(exp.get(k) is None for k in q):
            continue
        # характерная частота обвязки - измеренная fc / f0 каскада (развёртка ±3 декады вокруг неё)
        f_char = exp["fc"] if want != "BP" else exp["f0"]
        exp = expected_by_mna(netlist, f_char)
        if exp["type"] != want or any(exp.get(k) is None for k in q):
            continue  # после перецентровки развёртки форма АЧХ должна сохраниться
        if not (sampler.F_RANGE[0] <= f_char <= sampler.F_RANGE[1]):
            continue
        return {"id": ref_id or kind, "class": "cascade:" + "+".join(s["class"] for s in stages), "level": "L3",
                "params": [dict(s["params"], topology=s["class"]) for s in stages], "netlist": netlist,
                "expected": {k: exp[k] for k in q}, "quantities": q, "f_char": f_char}
    raise RuntimeError(f"{kind}: не удалось собрать каскад за {max_tries} попыток")


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(sample(sys.argv[1], random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)),
                     ensure_ascii=False, indent=1))
