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


ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def adapter_missing(folder):
    """Чего не хватает папке адаптера: [] - полная. Без файла весов PEFT принимает папку за имя репозитория
    Hugging Face и падает с непонятной HFValidationError (17.11.2026, q35_4b_v32_m1: на Drive после конца сессии
    остался adapter_config.json без adapter_model.safetensors - 130 МБ не успели выгрузиться)."""
    missing = [] if (folder / "adapter_config.json").exists() else ["adapter_config.json"]
    if not any((folder / w).exists() for w in ADAPTER_WEIGHTS):
        missing.append(ADAPTER_WEIGHTS[0])
    return missing


def adapter_path(path):
    """Папка адаптера с adapter_config.json и файлом весов. Если чего-то нет, берётся последняя полная контрольная
    точка того же запуска (<RUN>/checkpoints/checkpoint-N): на Drive после обрыва сессии в adapter/ оставался
    то один файл весов (smoke1), то одна настройка без весов (q35_4b_v32_m1)."""
    import re
    from pathlib import Path
    p = Path(path)
    missing = adapter_missing(p)
    if not missing:
        return str(p)
    found = [c for c in (p.parent / "checkpoints").glob("checkpoint-*")
             if re.fullmatch(r"checkpoint-\d+", c.name) and not adapter_missing(c)]
    if not found:
        raise FileNotFoundError(
            f"адаптер неполный: в {p} нет {', '.join(missing)}, полных контрольных точек в {p.parent / 'checkpoints'} нет. "
            "Если обучение только что кончилось в другой сессии - Drive мог ещё не выгрузить файл; после обучения "
            "контрольные точки удаляются в корзину Drive, последнюю (с наибольшим N) можно восстановить оттуда")
    last = max(found, key=lambda c: int(c.name.split("-")[1]))
    print(f"в {p} нет {', '.join(missing)}, беру контрольную точку {last}", flush=True)
    return str(last)


def _checkpoints(folder):
    from pathlib import Path
    folder = Path(folder)
    if not folder.exists():
        return []
    return [d for d in folder.glob("checkpoint-*") if d.is_dir() and d.name.split("-")[-1].isdigit()]


def checkpoint_complete(d):
    """С контрольной точки можно продолжать обучение: есть состояние Trainer и оптимизатора."""
    return (d / "trainer_state.json").exists() and (d / "optimizer.pt").exists()


def last_complete(folder):
    """Последняя полная контрольная точка (путь) или None. Папка без файлов бывает, когда Drive перестал принимать
    запись (08.11.2026: кончилось место, `checkpoint-150` Phi-4-mini осталась пустой) - с неё продолжать нельзя."""
    done = [d for d in _checkpoints(folder) if checkpoint_complete(d)]
    return str(max(done, key=lambda d: int(d.name.split("-")[-1]))) if done else None


def drop_incomplete(folder):
    """Удалить неполные контрольные точки -> [удалённые]. Продолжать с них нельзя, а Trainer при save_total_limit
    упорядочивает точки по номеру и не трогает последнюю по номеру (transformers.trainer_utils.rotate_checkpoints):
    пустые checkpoint-400...500 прерванного запуска (q35_4b_v32_m1, 17.11.2026) удаляли бы и точку продолжения 300,
    и каждую новую - после обрыва сессии продолжать было бы не с чего."""
    import shutil
    gone = []
    for d in sorted(_checkpoints(folder), key=lambda d: int(d.name.split("-")[-1])):
        if not checkpoint_complete(d):
            shutil.rmtree(d, ignore_errors=True)
            gone.append(str(d))
    return gone


def load(base, adapter=None, four_bit=True, extra=None):
    """-> (модель, токенизатор). bf16 - если поддерживается, иначе fp16 (GTX 1660 - только fp16).
    extra - {имя: папка} ещё адаптеров на той же базе (программа: адаптер режима 2); переключение - set_adapter."""
    if adapter:
        adapter = adapter_path(adapter)  # до загрузки весов: ошибка видна сразу, а не через минуту
    extra = {k: adapter_path(v) for k, v in (extra or {}).items()}
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
        for name, path in extra.items():
            model.load_adapter(path, adapter_name=name)
        model.set_adapter("default")
    model.eval()
    return model, tok


def generate(model, tok, prompts, fmt="netlist", batch=8, max_new_tokens=MAX_NEW_TOKENS, on_batch=None, systems=None):
    """Ответы на запросы, без случайности -> (ответы, секунд на запрос). on_batch(ответы) вызывается после каждой
    пачки: прерванная генерация не теряется. systems - инструкция для каждого запроса (датасет v3:
    training.system_for(запись), у режима 2 она своя); без него - инструкция v1 и v2."""
    import torch
    # конец реплики шаблона (<|im_end|>) - eos токенизатора; у Qwen3.5 нет generation_config, и без этого
    # генерация останавливалась бы только на <|endoftext|>, то есть через max_new_tokens
    stop = model.generation_config.eos_token_id
    stop = sorted({tok.eos_token_id} | set(training.end_of_turn(tok))
                  | set(stop if isinstance(stop, list) else [stop] if stop is not None else []))
    out, start = [], time.time()
    for i in range(0, len(prompts), batch):
        texts = [training.chat_text(tok, training.messages(p, fmt=fmt, system=systems[i + k] if systems else None))
                 for k, p in enumerate(prompts[i:i + batch])]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=stop)
        done = [t.strip() for t in tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        out += done
        if on_batch:
            on_batch(done)
    return out, (time.time() - start) / max(1, len(prompts))
