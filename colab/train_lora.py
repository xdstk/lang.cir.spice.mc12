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
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,  # noqa: E402
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from mcnl import hf_runtime, training  # noqa: E402


class SessionClock(TrainerCallback):
    """Время обучения по всем сессиям Colab: при каждой контрольной точке пишет в time.json {шаг: секунды с начала
    обучения}. Продолжение с точки берёт время, записанное для её шага, - время оборванной части после точки не
    считается, как и сами шаги, которые придётся повторить."""

    def __init__(self, path, resume_step):
        self.path = path
        self.marks = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.before = self.marks.get(str(resume_step), 0.0) if resume_step else 0.0
        self.resume_step = resume_step or 0
        self.start = time.time()

    def total(self):
        return self.before + time.time() - self.start

    def on_save(self, args, state, control, **kw):
        self.marks[str(state.global_step)] = self.total()
        self.path.write_text(json.dumps(self.marks, indent=1), encoding="utf-8")


def shrink_frozen(model, dtype):
    """prepare_model_for_kbit_training переводит в float32 всё, что не квантовано, в том числе встраивания и выходной
    слой (bitsandbytes их не квантует). При словаре 248 тыс. это 4 ГБ на слой у Qwen3.5-9B (слои не связаны: 8 ГБ),
    2,5 ГБ у 4B и 14 ГБ у встраиваний по слоям Gemma 4 E4B. Слои заморожены, поэтому хранятся снова в dtype (fp16 на T4).
    Счёт почти не меняется: выходной слой и так умножался в fp16 (autocast), а выход встраиваний переводится обратно
    в float32, как было. Возвращает освобождённые ГБ."""
    freed = 0
    head = model.get_output_embeddings()
    for mod in model.modules():
        if not (isinstance(mod, torch.nn.Embedding) or mod is head):
            continue
        w = mod.weight
        if w.dtype != torch.float32 or w.requires_grad:  # связанный с встраиваниями выходной слой уже переведён
            continue
        w.data = w.data.to(dtype)
        freed += w.numel() * 2
        if isinstance(mod, torch.nn.Embedding):
            mod.register_forward_hook(lambda m, inp, out: out.float() if out.is_floating_point() else out)
    return freed / 2 ** 30


def dataset(path, tok, max_len):
    end = training.end_of_turn(tok)
    rows = [training.encode(tok, r["messages"], max_len, end) for r in training.read_jsonl(path)]
    kept = [r for r in rows if r]
    return kept, len(rows) - len(kept)


def lora_targets(model):
    """У мультимодальных моделей, что грузятся целиком (Gemma 4), LoRA только на языковую часть: башни зрения и звука
    не трогаем. Текстовые модели (и Qwen3.5, у которой грузится только текст) - все линейные слои."""
    if any(".language_model." in name for name, _ in model.named_modules()):
        return r".*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    return "all-linear"


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
    ap.add_argument("--max-steps", type=int, default=-1, help="проба: столько шагов и оценка времени полного обучения")
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
    print(f"замороженные встраивания обратно в {dtype}: -{shrink_frozen(model, dtype):.1f} ГБ", flush=True)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=lora_targets(model)))
    model.print_trainable_parameters()

    train, cut_train = dataset(Path(a.data) / "train.jsonl", tok, a.max_len)
    dev, cut_dev = dataset(Path(a.data) / "dev.jsonl", tok, a.max_len)
    dev = dev[:a.eval_records]  # потери на dev - только для наблюдения, точность считается отдельно
    print(f"примеров: train {len(train)}, dev {len(dev)}; длиннее {a.max_len} токенов: {cut_train}, {cut_dev}", flush=True)

    steps = math.ceil(len(train) / a.batch / a.accum) * a.epochs
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * steps)), bf16=bf16, fp16=not bf16, logging_steps=max(1, int(steps) // 40), eval_strategy="epoch", save_strategy="steps", save_steps=50,
        save_total_limit=2, report_to=[], seed=a.seed, remove_unused_columns=False,
        prediction_loss_only=True,  # на dev нужны только потери: логиты всей выборки в память не поместятся
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    last = get_last_checkpoint(str(out / "checkpoints")) if (out / "checkpoints").exists() else None
    clock = SessionClock(out / "time.json", int(last.rsplit("-", 1)[1]) if last else 0)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev, data_collator=collate(tok.pad_token_id),
                      callbacks=[clock])
    if last:
        print(f"продолжаю с {last}; до неё обучение шло {clock.before / 60:.0f} мин", flush=True)
    start = time.time()
    result = trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    import peft
    import transformers
    # train_seconds - все сессии (до точки продолжения - из time.json); session_seconds и session_steps - последняя.
    # В run.json 4B (04.11.2026) train_seconds - только последняя сессия, с шага около 325: тогда учёта сессий не было.
    meta = {"args": vars(a), "train_seconds": clock.total(), "session_seconds": time.time() - start,
            "session_steps": trainer.state.global_step - clock.resume_step, "train_loss": result.training_loss,
            "gpu": torch.cuda.get_device_name(0), "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
            "dropped_long": [cut_train, cut_dev], "log": trainer.state.log_history,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}}
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"готово: {meta['train_seconds'] / 60:.1f} мин (эта сессия - {meta['session_steps']} шагов за "
          f"{meta['session_seconds'] / 60:.1f} мин), пик GPU {meta['gpu_peak_gb']:.1f} ГБ", flush=True)
    if a.max_steps > 0:  # первые шаги медленнее остальных, оценка с запасом
        sec = meta["session_seconds"] / a.max_steps
        print(f"проба: {sec:.1f} с на шаг; полное обучение - {int(steps)} шагов, около {sec * steps / 3600:.1f} ч", flush=True)


if __name__ == "__main__":
    main()
