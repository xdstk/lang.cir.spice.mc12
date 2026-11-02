#!/usr/bin/env python3
"""Оценка переводчика («батарейки») на выборке датасета: ответы -> score.py -> сводка.

Запуск:  python src/mcnl/evaluate.py --model baseline --split test [--data data/dataset/v1]
                                     [--limit 100] [--out out/eval] [--predictions ответы.jsonl]

Пишет <out>/<модель>_<выборка>/predictions.jsonl (формат eval_spec, раздел 7), scores.jsonl и summary.json;
сводка - доли уровней S0-S4 и функциональной точности с 95% интервалом Уилсона, в разбивке по уровню
сложности, топологии, задаче (A / A′) и семейству формулировок. Одна команда для baseline и для любых
обученных моделей - сравнение идёт на одних и тех же записях.
"""
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import score, translators  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def evaluate(translator, records, outdir, timeout=1800):
    outputs = translator.translate_batch([r["prompt"] for r in records]) if hasattr(translator, "translate_batch") \
        else [translator.translate(r["prompt"]) for r in records]
    name = getattr(translator, "name", type(translator).__name__)
    fmt = getattr(translator, "output_format", "netlist")
    preds = [{"id": r["id"], "model": name, "format": fmt, "output": o} for r, o in zip(records, outputs)]
    return evaluate_predictions(preds, records, outdir, timeout)


def evaluate_predictions(preds, records, outdir, timeout=1800):
    """Готовые ответы -> оценка; ответы могут прийти из Colab, где нет MC12. Набор id обязан совпасть с выборкой."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    missing = {r["id"] for r in records} ^ {p["id"] for p in preds}
    if missing:
        raise ValueError(f"ответы и выборка расходятся: {len(missing)} id, например {sorted(missing)[:3]}")
    with open(outdir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    refs = {r["id"]: dict(r, **{"class": r["topology"]}) for r in records}
    rows = score.score_all(refs, preds, outdir / "runs", timeout=timeout)
    meta = {r["id"]: r for r in records}
    for row in rows:
        row["task"], row["family"] = meta[row["id"]]["task"], meta[row["id"]]["family"]
    with open(outdir / "scores.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = score.summarize(rows)
    for key in summary:
        rs = [r for r in rows if f"{r['model']}/{r['format']}" == key]
        for field in ("task", "family"):
            summary[key][f"by_{field}"] = {v: score.summarize([r for r in rs if r[field] == v])[key]["all"]
                                           for v in sorted({r[field] for r in rs})}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main(argv):
    def opt(key, default=None):
        return argv[argv.index(key) + 1] if key in argv else default
    model, split = opt("--model", "baseline"), opt("--split", "test")
    data = Path(opt("--data", ROOT / "data" / "dataset" / "v1"))
    records = [json.loads(l) for l in (data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if opt("--limit"):
        records = records[:int(opt("--limit"))]
    if opt("--predictions"):  # ответы из Colab
        preds = score.read_jsonl(opt("--predictions"))
        out = Path(opt("--out", ROOT / "out" / "eval")) / f"{preds[0]['model']}_{split}"
        summary = evaluate_predictions(preds, records, out)
    else:
        tr = translators.load(model)
        out = Path(opt("--out", ROOT / "out" / "eval")) / f"{getattr(tr, 'name', 'model')}_{split}"
        summary = evaluate(tr, records, out)
    sys.stdout.reconfigure(encoding="utf-8")
    for key, s in summary.items():
        a = s["all"]
        print(f"{key}  n={a['n']}  " + "  ".join(
            f"{lvl} {a[lvl]['rate']:.3f} [{a[lvl]['ci95'][0]:.3f}-{a[lvl]['ci95'][1]:.3f}]"
            for lvl in score.LEVELS + ("S4c", "functional", "functional_c")))
        print("  refusals:", a.get("refusals"), " false:", a["false_refusals"])
        print("  by task:", {t: round(v["S3"]["rate"], 3) for t, v in s["by_task"].items() if v["n"]},
              " errors:", a["errors"])
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv)
