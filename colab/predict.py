#!/usr/bin/env python3
"""Ответы модели на выборку: в Colab нет MC12, оценка идёт на ноутбуке автора (evaluate.py --predictions).

Запуск:  python colab/predict.py --model <id> [--adapter папка] --records data/dataset/v2/dev.jsonl --out ответы.jsonl
Без --adapter - модель без обучения.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mcnl import hf_runtime, training  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="netlist", choices=training.FORMATS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", help="имя модели в таблицах")
    a = ap.parse_args()
    recs = training.read_jsonl(a.records)
    model, tok = hf_runtime.load(a.model, a.adapter)
    outputs, sec = hf_runtime.generate(model, tok, [r["prompt"] for r in recs], a.format, a.batch)
    name = a.name or (a.adapter or a.model).rstrip("/").split("/")[-1]
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for r, o in zip(recs, outputs):
            fh.write(json.dumps({"id": r["id"], "model": name, "format": a.format, "output": o}, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(recs), "sec_per_query": sec, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}, indent=1))


if __name__ == "__main__":
    main()
