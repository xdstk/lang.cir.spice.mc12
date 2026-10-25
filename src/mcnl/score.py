#!/usr/bin/env python3
"""Оценка ответов моделей: уровни S0-S4, функциональная точность, F1, классы ошибок, сводка
(docs/eval_spec.md, раздел 7).

Запуск:  python score.py эталоны.jsonl predictions.jsonl --out папка

Эталон (строка эталонов):  {"id", "netlist" (канонический), "f_char" (Гц или null), "class", "level",
                            "quantities" [...], "measured" {...}}
Ответ (predictions.jsonl): {"id", "model", "format": "netlist"|"ir", "output"}
Результат: <out>/scores.jsonl (по примеру) и <out>/summary.json (доли по уровням с 95% интервалом
Уилсона, в разбивке по классам схем и уровням сложности).

Уровни строгие: S1 требует S0, S2 - S1 и т.д. Функциональная точность = S1 и поведение в допуске
независимо от S2/S3 (другая схема, работающая так же).
"""
import json
import math
import re
import sys
from pathlib import Path

try:
    from . import bench, canon, iso, measure
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import iso
    import measure

LEVELS = ("S0", "S1", "S2", "S3", "S4")


# ---------------------------------------------------------------- разбор ответа (S0)
def parse_output(output, fmt):
    """Ответ модели -> (элементы или None, ошибки)."""
    if fmt == "ir":
        txt = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", output or "")
        try:
            ir = json.loads(txt)
        except json.JSONDecodeError as e:
            return None, [f"JSON: {e}"]
        errs = canon.validate_ir(ir)
        return (None, errs) if errs else (canon.from_ir(ir), [])
    elems, errs = canon.parse_netlist(output or "")
    if not elems and not errs:
        errs = ["пустой ответ"]
    return (None, errs) if errs else (elems, [])


# ---------------------------------------------------------------- F1 и классы ошибок
def _key(e):
    return (e["type"], e["value"] or "")


def f1(ref, test):
    from collections import Counter
    a, b = Counter(map(_key, ref)), Counter(map(_key, test))
    tp = sum((a & b).values())
    if not tp:
        return 0.0
    p, r = tp / sum(b.values()), tp / sum(a.values())
    return 2 * p * r / (p + r)


def _swapped(elems, k, pins):
    out = [dict(e, nodes=list(e["nodes"])) for e in elems]
    n = out[k]["nodes"]
    n[pins[0]], n[pins[1]] = n[pins[1]], n[pins[0]]
    return out


def topology_error(ref, test, partial):
    """Класс ошибки при несовпадении топологии."""
    for k, e in enumerate(test):
        if e["type"] in "VI" and iso.match(ref, _swapped(test, k, (0, 1)), "topology")["match"]:
            return "reversed_source"
        if e["type"] == "OPAMP" and iso.match(ref, _swapped(test, k, (0, 1)), "topology")["match"]:
            return "swapped_opamp_inputs"
    if not partial["unmatched_a"] and partial["unmatched_b"]:
        return "extra_element"
    if partial["unmatched_a"] and not partial["unmatched_b"]:
        return "missing_element"
    return "wrong_connection"


def value_error(ref, test, topo_mapping):
    """Класс ошибки при совпавшей топологии и несовпавших номиналах (по парам из сопоставления)."""
    worst = None
    for ka, kb in topo_mapping.items():
        va, vb = ref[ka]["value"], test[kb]["value"]
        if va is None or iso._same_value(va, vb):
            continue
        a, b = canon.parse_value(va), canon.parse_value(vb)
        if not a or not b:
            return "wrong_value"
        order = math.log10(abs(b / a))
        if abs(order + 9) < 1e-6:
            return "M_instead_of_MEG"  # 1M (милли) вместо 1MEG: ошибка в 1e9
        if abs(order - round(order)) < 1e-6 and round(order) % 3 == 0:
            worst = worst or "order_1e3"  # k/MEG/m перепутаны
        else:
            worst = "wrong_value"
    return worst or "wrong_value"


# ---------------------------------------------------------------- оценка
def score_all(refs, preds, workdir, timeout=600):
    """refs: {id: эталон}; preds: [ответ]. Один запуск MC12 на все ответы, прошедшие S0."""
    rows, cases = [], []
    for i, p in enumerate(preds):
        ref = refs[p["id"]]
        ref_el, err = canon.parse_netlist(ref["netlist"])
        if err:
            raise ValueError(f"эталон {p['id']} не разбирается: {err}")
        test_el, perr = parse_output(p.get("output"), p.get("format", "netlist"))
        row = {"id": p["id"], "model": p.get("model"), "format": p.get("format", "netlist"),
               "class": ref.get("class"), "level": ref.get("level"),
               "S0": test_el is not None, "parse_errors": perr}
        rows.append((row, ref, ref_el, test_el))
        if test_el is not None:
            cases.append((f"p{i:05d}", test_el, ref.get("f_char")))
    runs = bench.run(cases, workdir, timeout=timeout) if cases else {}
    out = []
    for i, (row, ref, ref_el, test_el) in enumerate(rows):
        r = runs.get(f"p{i:05d}") if test_el is not None else None
        row["S1"] = bool(row["S0"] and r and r["ok"])
        row["run_messages"] = r["messages"] if r else []
        topo = iso.match(ref_el, test_el, "topology") if test_el is not None else None
        vals = iso.match(ref_el, test_el, "values") if topo and topo["match"] else None
        row["topology"] = bool(topo and topo["match"])
        row["values"] = bool(vals and vals["match"])
        behavior, detail = False, {}
        if row["S1"]:
            m = measure.from_output(r["output"])
            behavior, detail = measure.within_tolerance(ref["measured"], m, ref["quantities"])
            row["measured"] = m
        row["behavior"] = behavior
        row["behavior_detail"] = {k: list(v) for k, v in detail.items()}
        row["S2"] = row["S1"] and row["topology"]
        row["S3"] = row["S2"] and row["values"]
        row["S4"] = row["S3"] and behavior
        row["f1"] = f1(ref_el, test_el) if test_el is not None else 0.0
        row["error"] = classify_error(row, ref_el, test_el, topo)
        # малосигнальный AC с идеальным ОУ не видит знака обратной связи: схема с перепутанными
        # входами даёт ту же АЧХ, хотя реальная ушла бы в насыщение - функционально не засчитывать
        row["functional"] = row["S1"] and behavior and row["error"] != "swapped_opamp_inputs"
        out.append(row)
    return out


def classify_error(row, ref_el, test_el, topo):
    if not row["S0"]:
        return "syntax"
    if not row["topology"]:
        return topology_error(ref_el, test_el, topo)
    if not row["values"]:
        return value_error(ref_el, test_el, topo["mapping"])
    if not row["S1"]:
        return "run_error"  # схема верна, но MC12 её не выполнил - ошибка обвязки, не модели
    if not row["behavior"]:
        return "behavior"
    return None


# ---------------------------------------------------------------- сводка
def wilson(k, n, z=1.96):
    """95% интервал Уилсона для доли k/n."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(rows):
    def block(rs):
        n = len(rs)
        res = {"n": n}
        for lvl in LEVELS + ("functional",):
            k = sum(1 for r in rs if r[lvl])
            lo, hi = wilson(k, n)
            res[lvl] = {"k": k, "rate": k / n if n else None, "ci95": [lo, hi]}
        res["f1_mean"] = sum(r["f1"] for r in rs) / n if n else None
        errs = {}
        for r in rs:
            if r["error"]:
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        res["errors"] = dict(sorted(errs.items(), key=lambda x: -x[1]))
        return res

    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["format"]), []).append(r)
    summary = {}
    for (model, fmt), rs in groups.items():
        key = f"{model}/{fmt}"
        summary[key] = {"all": block(rs),
                        "by_level": {lv: block([r for r in rs if r["level"] == lv])
                                     for lv in sorted({r["level"] for r in rs if r["level"]})},
                        "by_class": {c: block([r for r in rs if r["class"] == c])
                                     for c in sorted({r["class"] for r in rs if r["class"]})}}
    return summary


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main(argv):
    refs = {r["id"]: r for r in read_jsonl(argv[1])}
    preds = read_jsonl(argv[2])
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("out") / "score"
    out.mkdir(parents=True, exist_ok=True)
    rows = score_all(refs, preds, out / "runs")
    with open(out / "scores.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
