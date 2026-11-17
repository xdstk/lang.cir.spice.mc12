#!/usr/bin/env python3
"""Модель режима для llama.cpp (переносная версия, docs/program.md): база Qwen3.5-4B с влитым адаптером LoRA -> GGUF.

Запуск (из venv-hf: нужны torch, transformers, peft; скрипты перевода - из исходников llama.cpp той же версии,
что llama-server):
  python src/mcnl/gguf_export.py m1 out/q35_4b/adapter --trained-on v2
  python src/mcnl/gguf_export.py m2 out/q35_4b_v31_m2/adapter --trained-on v3
-> out/merged/<имя> (HF, bf16), out/gguf/<имя>-bf16.gguf, -Q4_K_M.gguf (и другие --quant), запись в out/gguf/models.json.

Почему вливание, а не LoRA-файлы: convert_lora_to_gguf.py переводит наши адаптеры, но llama-server b12360 их не
подключает - «LoRA tensor 'blk.0.attn_gate.weight' does not exist in base model» (имена тензоров слоёв DeltaNet
Qwen3.5 в адаптере и в базе не совпадают; 17.11.2026). Вливание даёт по модели на режим (Q4_K_M - 2,6 ГБ).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA_SRC = ROOT / "out" / "tools" / "llama-src" / "llama.cpp-b12360"
LLAMA_BIN = ROOT / "out" / "tools" / "llama-bin"
MODE = {"m1": "mode1", "m2": "mode2"}


def merge(adapter, merged, base="Qwen/Qwen3.5-4B"):
    """Влить адаптер в базу (bf16, процессор: около 9 ГБ памяти) и сохранить с токенизатором."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    shutil.rmtree(merged, ignore_errors=True)
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    return model.config.architectures


def export(name, adapter, trained_on, quants=("Q4_K_M",), out=None, llama_src=LLAMA_SRC, llama_bin=LLAMA_BIN, log=print):
    out = Path(out or ROOT / "out" / "gguf")
    out.mkdir(parents=True, exist_ok=True)
    merged = ROOT / "out" / "merged" / name
    t0 = time.time()
    if not (merged / "config.json").exists():
        log("вливание", adapter, "->", merged, merge(adapter, merged))
    bf16 = out / f"{name}-bf16.gguf"
    if not bf16.exists():
        subprocess.run([sys.executable, str(Path(llama_src) / "convert_hf_to_gguf.py"), str(merged), "--outtype", "bf16",
                        "--outfile", str(bf16)], check=True)
    for q in quants:
        dst = out / f"{name}-{q}.gguf"
        if not dst.exists():
            subprocess.run([str(Path(llama_bin) / "llama-quantize.exe"), str(bf16), str(dst), q], check=True,
                           stdout=subprocess.DEVNULL)
    index = out / "models.json"
    spec = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
    spec[MODE.get(name, name)] = {"file": f"{name}-{quants[0]}.gguf", "trained_on": trained_on,
                                  "adapter": Path(adapter).parent.name if Path(adapter).name == "adapter" else str(adapter)}
    index.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{name}: готово за {time.time() - t0:.0f} с", sorted(p.name for p in out.glob(f"{name}-*.gguf")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="m1 (режим 1) или m2 (режим 2)")
    ap.add_argument("adapter", help="папка адаптера LoRA")
    ap.add_argument("--trained-on", required=True, choices=("v2", "v3"), help="на чём обучен адаптер (инструкция)")
    ap.add_argument("--quant", nargs="+", default=["Q4_K_M"], help="кванты llama-quantize; первый - в models.json")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    export(a.name, a.adapter, a.trained_on, tuple(a.quant))


if __name__ == "__main__":
    main()
