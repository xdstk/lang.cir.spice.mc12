#!/usr/bin/env python3
"""Обучение LoRA в Colab (docs/stage2_plan.md). Прерванный запуск продолжается с последней контрольной точки.

Запуск:  python colab/train_lora.py --model <id на Hugging Face> --data out/training/v2_2000 --out /content/drive/MyDrive/mcnl/run1
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # до импорта torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments,  # noqa: E402
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from mcnl import hf_runtime, training  # noqa: E402


def dataset(path, tok, max_len):
    rows = [training.encode(tok, r["messages"], max_len) for r in training.read_jsonl(path)]
    kept = [r for r in rows if r]
    return kept, len(rows) - len(kept)


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for b in batch:
            gap = n - len(b["input_ids"])  # при обучении дополняем справа
            out["input_ids"].append(b["input_ids"] + [pad_id] * gap)
            out["attention_mask"].append(b["attention_mask"] + [0] * gap)
            out["labels"].append(b["labels"] + [-100] * gap)
        return {k: torch.tensor(v) for k, v in out.items()}
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="папка training.py: train.jsonl, dev.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    # логиты батча: токены * словарь (150 тыс. у Qwen); батч 4 при 700 токенах не вошёл в 15 ГБ T4. Шаг обучения тот же: 2 * 4
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-records", type=int, default=40)
    a = ap.parse_args()
    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "run.json").exists():
        print("обучение уже завершено:", out / "run.json", flush=True)
        return
    bf16 = hf_runtime.native_bf16()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=dtype))
    model.config.use_cache = False  # кэш при обучении не нужен и мешает контрольным точкам градиента
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()

    train, cut_train = dataset(Path(a.data) / "train.jsonl", tok, a.max_len)
    dev, cut_dev = dataset(Path(a.data) / "dev.jsonl", tok, a.max_len)
    dev = dev[:a.eval_records]  # потери на dev - только для наблюдения, точность считается отдельно
    print(f"примеров: train {len(train)}, dev {len(dev)}; длиннее {a.max_len} токенов: {cut_train}, {cut_dev}", flush=True)

    steps = math.ceil(len(train) / a.batch / a.accum) * a.epochs
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * steps)), bf16=bf16, fp16=not bf16, logging_steps=max(1, int(steps) // 40), eval_strategy="epoch", save_strategy="steps", save_steps=100,
        save_total_limit=2, report_to=[], seed=a.seed, remove_unused_columns=False,
        prediction_loss_only=True,  # на dev нужны только потери: логиты всей выборки в память не поместятся
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev, data_collator=collate(tok.pad_token_id))
    last = get_last_checkpoint(str(out / "checkpoints")) if (out / "checkpoints").exists() else None
    if last:
        print("продолжаю с", last, flush=True)
    start = time.time()
    result = trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    import peft
    import transformers
    meta = {"args": vars(a), "train_seconds": time.time() - start, "train_loss": result.training_loss,
            "gpu": torch.cuda.get_device_name(0), "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "dropped_long": [cut_train, cut_dev], "log": trainer.state.log_history,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}}
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"готово: {meta['train_seconds'] / 60:.1f} мин, пик GPU {meta['gpu_peak_gb']:.1f} ГБ", flush=True)


if __name__ == "__main__":
    main()
