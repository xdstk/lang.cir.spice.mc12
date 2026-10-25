#!/usr/bin/env python3
"""Пилот эталонов и допуск в датасет (docs/eval_spec.md, разделы 9-10, план 12.7).

Запуск:  python pilot.py [--out data/pilot/refs.jsonl]

Эталон принимается, если: (1) нетлист каноничен и проходит круговое преобразование нетлист <-> IR;
(2) запускается в MC12 без Error; (3) измерения совпадают с формулой (formulas.py): fc, f0, f_u, k -
0,1% (k - в разах), Q - 1%. Дополнительно формулу проверяет численный узловой расчёт (mna.py):
неверную формулу отвергают обе проверки, а не только MC12. В эталон записываются измерения MC12 -
с ними, а не с запрошенными значениями, сравниваются ответы моделей.
"""
import json
import sys
from pathlib import Path

try:
    from . import bench, canon, formulas, measure, mna
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import formulas
    import measure
    import mna

ROOT = Path(__file__).resolve().parents[2]
REL_ADMIT = {"fc": 1e-3, "f0": 1e-3, "f_u": 1e-3, "Q": 1e-2}
K_ADMIT = 1e-3  # для k_pass_db / k_dc_db: относительная разница в разах

# номиналы: R - E24, C и L - E12; характерные частоты 10 Гц - 1 МГц (eval_spec, раздел 1)
PILOT = [
    ("p01_divider", "divider", {"R1": 10e3, "R2": 4.7e3, "dc": 5}),
    ("p02_rc_lp", "rc_lp", {"R": 1.5e3, "C": 47e-9}),
    ("p03_rc_hp", "rc_hp", {"R": 33e3, "C": 2.2e-9}),
    ("p04_rl_lp", "rl_lp", {"R": 220, "L": 10e-3}),
    ("p05_rl_hp", "rl_hp", {"R": 1e3, "L": 47e-3}),
    ("p06_rlc_bp", "rlc_series_bp", {"R": 100, "L": 10e-3, "C": 100e-9}),
    ("p07_inverting", "inverting", {"R1": 4.7e3, "R2": 47e3}),
    ("p08_noninverting", "noninverting", {"R1": 10e3, "R2": 22e3}),
    ("p09_active_lp", "active_lp_inv", {"R1": 10e3, "R2": 100e3, "C": 1.5e-9}),
    ("p10_integrator", "integrator_rf", {"R1": 1e3, "Rf": 220e3, "C": 10e-9}),
    ("p11_sk_lp_unity", "sallen_key_lp", {"R1": 10e3, "R2": 10e3, "C1": 22e-9, "C2": 10e-9}),
    ("p12_sk_lp_gain", "sallen_key_lp", {"R1": 15e3, "R2": 15e3, "C1": 4.7e-9, "C2": 4.7e-9, "Ra": 10e3, "Rb": 5.6e3}),
    ("p13_sk_hp_gain", "sallen_key_hp", {"R1": 18e3, "R2": 15e3, "C1": 1e-9, "C2": 1e-9, "Ra": 10e3, "Rb": 10e3}),
    ("p14_sk_hp_unity", "sallen_key_hp", {"R1": 12e3, "R2": 24e3, "C1": 2.2e-9, "C2": 2.2e-9}),
    ("p15_mfb_bp", "mfb_bp", {"R1": 10e3, "R2": 1e3, "R3": 100e3, "Ca": 10e-9, "Cb": 10e-9}),
]


def compare(expected, measured, quantities):
    """-> (в допуске, {величина: [формула, измерено, отклонение, в допуске]})."""
    detail = {}
    for q in quantities:
        a, b = expected.get(q), measured.get(q)
        if q == "type":
            ok, dev = a == b, None
        elif a is None or b is None:
            ok, dev = False, None
        elif q in ("k_pass_db", "k_dc_db"):
            dev = 10 ** ((b - a) / 20) - 1
            ok = abs(dev) <= K_ADMIT
        else:
            dev = (b - a) / a
            ok = abs(dev) <= REL_ADMIT[q]
        detail[q] = [a, b, dev, ok]
    return all(v[3] for v in detail.values()), detail


def roundtrip_ok(netlist):
    elems, err = canon.parse_netlist(netlist)
    if err or canon.to_netlist(canon.canonicalize(elems)) != netlist:
        return False
    return canon.to_netlist(canon.canonicalize(canon.from_ir(canon.to_ir(elems)))) == netlist


def mna_measure(ref):
    lo, hi = bench.ac_range(ref["f_char"])
    return measure.measure(*mna.sweep(canon.parse_netlist(ref["netlist"])[0], lo, hi, per_decade=bench.POINTS_PER_DECADE))


def build_pilot(workdir, items=PILOT):
    refs = [formulas.reference(topo, params, rid) for rid, topo, params in items]
    runs = bench.run([(r["id"], canon.parse_netlist(r["netlist"])[0], r["f_char"]) for r in refs], workdir)
    for r in refs:
        run = runs[r["id"]]
        adm = {"canonical_roundtrip": roundtrip_ok(r["netlist"]), "mc12_run": run["ok"],
               "run_messages": run["messages"]}
        ok_mna, adm["formula_vs_mna"] = compare(r["expected"], mna_measure(r), r["quantities"])
        if run["ok"]:
            r["measured"] = measure.from_output(run["output"])
            ok_mc, adm["formula_vs_mc12"] = compare(r["expected"], r["measured"], r["quantities"])
        else:
            ok_mc = False
        adm["formula_ok"] = ok_mna
        adm["admitted"] = adm["canonical_roundtrip"] and run["ok"] and ok_mna and ok_mc
        r["admission"] = adm
    return refs


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "data" / "pilot" / "refs.jsonl"
    refs = build_pilot(ROOT / "out" / "pilot")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in refs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in refs:
        a = r["admission"]
        worst = max((abs(v[2]) for v in a.get("formula_vs_mc12", {}).values() if v[2] is not None), default=0)
        print(f"{r['id']:20s} {'OK ' if a['admitted'] else 'NO '} max dev MC12 {worst:.2e}  "
              f"{'' if a['formula_ok'] else 'FORMULA != MNA '}{a['run_messages'] if not a['mc12_run'] else ''}")
    print(f"admitted {sum(r['admission']['admitted'] for r in refs)} of {len(refs)} -> {out}")


if __name__ == "__main__":
    main(sys.argv)
