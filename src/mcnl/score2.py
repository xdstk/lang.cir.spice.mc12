#!/usr/bin/env python3
"""Оценка ответов режима 2 (docs/mode2_spec.md, раздел 7): нетлист с `?` и строка `.SPEC`.

Эталон - запись датасета v3 с mode = 2 (mode2_data): output (ответ), spec, solution.wanted (названные цели).
Ответ - как у score.py: {"id", "model", "format": "netlist", "output"}.

Уровни (строгие, каждый требует предыдущего, кроме behavior):
  parse    - ответ разбирается: строки элементов и строка .SPEC (synth.parse) без ошибок;
  circuit  - схема та же: изоморфизм, `?` на тех же элементах, заданные номиналы совпадают; если эталон - одна
             строка .SPEC, элементов в ответе быть не должно;
  spec     - поля .SPEC совпадают с эталоном точно (те же поля; числа - с точностью 1e-6, слова - без учёта регистра);
  exact    - circuit и spec: ответ равен эталону (основная величина, как S3 в режиме 1);
  behavior - независимо от exact: по ответу synth.run подбирает номиналы, подобранная схема считается в MC12 и
             попадает в цели эталона (тип АЧХ; срез, f0, Q - в пределах 10%, усиление - 1 дБ). Маршрут дизайнера
             (ПФ, РФ выше 2-го порядка): схему строит не synth, поведение засчитывается по совпадению .SPEC.
Отказы считаются, как в score.py: refuse_ref, refuse_pred.
"""
import json
import sys
from pathlib import Path

try:
    from . import bench, canon, iso, measure, mode2_data, score, synth
except ImportError:  # запуск как скрипта
    import bench
    import canon
    import iso
    import measure
    import mode2_data
    import score
    import synth

LEVELS = ("parse", "circuit", "spec", "exact", "behavior")
UNKNOWN = "123.456G"                       # номинал-метка на месте `?`: совпадает только с другим `?`
NUMERIC = ("fc", "f0", "Q", "bw", "K", "ripple", "order")
TOL = mode2_data.TOL


def parse(output):
    """-> (элементы с меткой на месте `?`, поля .SPEC, ошибки)."""
    elems, unknown, spec, errors = synth.parse(output or "")
    if not spec and not errors:
        errors = ["нет строки .SPEC"]
    for e in elems:
        if e["name"] in unknown:
            e["value"] = UNKNOWN
    return elems, spec, errors


def same_spec(ref, test):
    """Поля .SPEC совпадают -> (да/нет, список расходящихся полей)."""
    diff = sorted(set(ref) ^ set(test))
    for k in set(ref) & set(test):
        a, b = str(ref[k]), str(test[k])
        if k in NUMERIC:
            x, y = canon.parse_value(a), canon.parse_value(b)
            if x is None or y is None or abs(x - y) > 1e-6 * max(abs(x), abs(y), 1e-30):
                diff.append(k)
        elif a.lower() != b.lower():
            diff.append(k)
    return not diff, sorted(diff)


def circuit_error(ref_el, test_el):
    """Класс ошибки схемы: другая схема, `?` не на тех элементах или другой заданный номинал."""
    topo = iso.match(ref_el, test_el, "topology")
    if not topo["match"]:
        return "wrong_circuit"
    flags = [(ref_el[a]["value"] == UNKNOWN, test_el[b]["value"] == UNKNOWN) for a, b in topo["mapping"].items()]
    return "wrong_unknowns" if any(x != y for x, y in flags) else "wrong_value"


def _within(wanted, got):
    """Измерение подобранной схемы против целей эталона -> (в допуске, отклонения)."""
    if got.get("type") != wanted.get("type"):
        return False, {"type": [wanted.get("type"), got.get("type")]}
    ok, detail = True, {}
    for k, v in wanted.items():
        if k == "type" or v is None:
            continue
        g = got.get(k)
        if g is None:
            ok, detail[k] = False, None
            continue
        d = g - v if k == "k_pass_db" else g / v - 1
        detail[k] = d
        ok = ok and abs(d) <= (TOL["db"] if k == "k_pass_db" else TOL["rel"])
    return ok, detail


def score_all(refs, preds, workdir, timeout=1800):
    """refs: {id: запись режима 2}; preds: [ответ]. Один запуск MC12 на все подобранные схемы."""
    rows, cases = [], []
    for i, p in enumerate(preds):
        ref = refs[p["id"]]
        refused = score.parse_refusal(p.get("output"))
        row = {"id": p["id"], "model": p.get("model"), "format": "netlist", "mode": 2, "kind": ref.get("kind"),
               "family": ref.get("family"), "refuse_ref": ref.get("refuse"), "refuse_pred": refused[0] if refused else None}
        rows.append(row)
        if ref.get("refuse"):
            continue
        ref_el, ref_spec, err = parse(ref["output"])
        if err:
            raise ValueError(f"эталон {p['id']} не разбирается: {err}")
        test_el, test_spec, perr = ([], {}, ["отказ"]) if refused else parse(p.get("output"))
        row.update(parse=not perr, parse_errors=perr, circuit=False, spec=False, exact=False, solved=False,
                   behavior=False, behavior_detail={}, error=None)
        if perr:
            row["error"] = "false_refusal" if refused else "syntax"
            continue
        has_ref = any(e["type"] != "V" for e in ref_el)
        has_test = any(e["type"] != "V" for e in test_el)
        if has_ref or has_test:
            row["circuit"] = bool(has_ref and has_test and iso.match(ref_el, test_el, "values")["match"])
        else:
            row["circuit"] = True  # оба ответа - одна строка .SPEC
        row["spec"], row["spec_diff"] = same_spec(ref_spec, test_spec)
        row["spec"] = row["circuit"] and row["spec"]          # уровни строгие
        row["exact"] = row["spec"]
        if not row["circuit"]:
            row["error"] = "extra_circuit" if has_test and not has_ref else "missing_circuit" if has_ref and not has_test \
                else circuit_error(ref_el, test_el)
        elif not row["exact"]:
            row["error"] = "wrong_spec"
        rep = synth.run(p.get("output") or "")
        if "error" in rep:
            row["solve_error"] = rep["error"]
            row["error"] = row["error"] or "unsolved"
            continue
        row["solved"] = True
        if rep.get("route") == "designer":
            row["behavior"] = same_spec(ref_spec, test_spec)[0]
            row["error"] = row["error"] or (None if row["behavior"] else "behavior")
            continue
        wanted = (ref.get("solution") or {}).get("wanted") or rep["wanted"]
        f = canon.parse_value(str(ref["spec"].get("fc") or ref["spec"].get("f0")))
        cases.append((f"m{i:05d}", canon.parse_netlist(rep["netlist"])[0], f, row, wanted))
    runs = bench.run([c[:3] for c in cases], workdir, timeout=timeout) if cases else {}
    for name, _el, _f, row, wanted in cases:
        r = runs[name]
        row["mc12_run"] = r["ok"]
        if r["ok"]:
            row["behavior"], row["behavior_detail"] = _within(wanted, measure.from_output(r["output"]))
        row["error"] = row["error"] or (None if row["behavior"] else "behavior")
    return rows


def summarize(rows):
    def block(rs):
        refuse = [r for r in rs if r.get("refuse_ref")]
        rs = [r for r in rs if not r.get("refuse_ref")]
        n = len(rs)
        res = {"n": n}
        if refuse:
            res["refusals"] = {"n": len(refuse), "caught": sum(1 for r in refuse if r["refuse_pred"]),
                               "code_correct": sum(1 for r in refuse if r["refuse_pred"] == r["refuse_ref"]),
                               "missed": sum(1 for r in refuse if not r["refuse_pred"])}
        res["false_refusals"] = sum(1 for r in rs if r.get("refuse_pred"))
        for lvl in LEVELS:
            k = sum(1 for r in rs if r[lvl])
            lo, hi = score.wilson(k, n)
            res[lvl] = {"k": k, "rate": k / n if n else None, "ci95": [lo, hi]}
        errs = {}
        for r in rs:
            if r["error"]:
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        res["errors"] = dict(sorted(errs.items(), key=lambda x: -x[1]))
        return res

    return {"all": block(rows), "by_kind": {k: block([r for r in rows if r["kind"] == k])
                                            for k in sorted({r["kind"] for r in rows if r["kind"]})}}


def main(argv):
    refs = {r["id"]: r for r in score.read_jsonl(argv[1]) if r.get("mode") == 2}
    preds = [p for p in score.read_jsonl(argv[2]) if p["id"] in refs]
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("out") / "score2"
    out.mkdir(parents=True, exist_ok=True)
    rows = score_all(refs, preds, out / "runs")
    with open(out / "scores.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
