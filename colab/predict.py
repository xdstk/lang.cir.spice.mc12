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
    ap.add_argument("--limit", type=int, help="только первые записи")
    a = ap.parse_args()
    recs = training.read_jsonl(a.records)[:a.limit]
    done = {r["id"] for r in training.read_jsonl(a.out)} if Path(a.out).exists() else set()
    recs = [r for r in recs if r["id"] not in done]  # продолжение прерванного запуска
    if not recs:
        print("все ответы уже есть:", a.out)
        return
    print(f"осталось запросов: {len(recs)}, уже есть: {len(done)}", flush=True)
    model, tok = hf_runtime.load(a.model, a.adapter)
    name = a.name or (a.adapter or a.model).rstrip("/").split("/")[-1]
    left = iter(recs)
    with open(a.out, "a", encoding="utf-8", newline="\n") as fh:
        def save(answers):
            for o in answers:
                fh.write(json.dumps({"id": next(left)["id"], "model": name, "format": a.format, "output": o}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"готово {sum(1 for _ in open(a.out, encoding='utf-8'))}", flush=True)
        _, sec = hf_runtime.generate(model, tok, [r["prompt"] for r in recs], a.format, a.batch, on_batch=save)
    print(json.dumps({"records": len(recs), "sec_per_query": sec, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}, indent=1))


if __name__ == "__main__":
    main()
