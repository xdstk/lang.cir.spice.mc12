"""Загрузка модели Hugging Face и генерация ответов: общий код Colab и программы (docs/stage2_plan.md).

torch и transformers подключаются только при вызове: основная программа работает на стандартной библиотеке.
"""
import time

try:
    from . import training
except ImportError:  # запуск как скрипта
    import training

MAX_NEW_TOKENS = 800  # самый длинный ответ v2: нетлист 303 знака, JSON-IR 1102 знака


def native_bf16():
    """bf16 только аппаратный: без этого флага torch отвечает «да» и на T4, где bf16 эмулируется и идёт в разы медленнее."""
    import torch
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)


def adapter_path(path):
    """Папка адаптера с adapter_config.json. Если его нет, берётся последняя контрольная точка того же запуска
    (<RUN>/checkpoints/checkpoint-N): на Drive после обрыва сессии в adapter/ остался один файл весов."""
    import re
    from pathlib import Path
    p = Path(path)
    if (p / "adapter_config.json").exists():
        return str(p)
    found = [c for c in (p.parent / "checkpoints").glob("checkpoint-*")
             if re.fullmatch(r"checkpoint-\d+", c.name) and (c / "adapter_config.json").exists()]
    if not found:
        raise FileNotFoundError(f"нет adapter_config.json ни в {p}, ни в {p.parent / 'checkpoints'}")
    last = max(found, key=lambda c: int(c.name.split("-")[1]))
    print(f"в {p} нет adapter_config.json, беру контрольную точку {last}", flush=True)
    return str(last)


def load(base, adapter=None, four_bit=True):
    """-> (модель, токенизатор). bf16 - если поддерживается, иначе fp16 (GTX 1660 - только fp16)."""
    if adapter:
        adapter = adapter_path(adapter)  # до загрузки весов: ошибка видна сразу, а не через минуту
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if native_bf16() else torch.float16
    # «auto» оставляет запас и на 6 ГБ отказывает заранее; модель целиком на GPU, нехватка памяти покажет себя честно
    kw = {"device_map": {"": 0} if torch.cuda.is_available() else "auto"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def generate(model, tok, prompts, fmt="netlist", batch=8, max_new_tokens=MAX_NEW_TOKENS, on_batch=None):
    """Ответы на запросы, без случайности -> (ответы, секунд на запрос). on_batch(ответы) вызывается после каждой
    пачки: прерванная генерация не теряется."""
    import torch
    out, start = [], time.time()
    for i in range(0, len(prompts), batch):
        texts = [training.chat_text(tok, training.messages(p, fmt=fmt)) for p in prompts[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        done = [t.strip() for t in tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        out += done
        if on_batch:
            on_batch(done)
    return out, (time.time() - start) / max(1, len(prompts))
