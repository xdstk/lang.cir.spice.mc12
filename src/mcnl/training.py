#!/usr/bin/env python3
"""Подготовка обучения: системная инструкция, экспорт датасета в диалоги, разметка меток (docs/stage2_plan.md).

Запуск:  python src/mcnl/training.py --data data/dataset/v2 --subset 2000 --format netlist --out out/training/v2_2000

Только стандартная библиотека: файл работает и здесь, и в Colab. Инструкция одна для обучения и для работы
программы: модель должна видеть в программе ровно то, на чём училась.
"""
import json
import sys
from pathlib import Path

SYSTEM = {
    "netlist": """Ты переводишь описание электрической схемы на русском в нетлист. Отвечай только строками элементов, без пояснений.

Строки:
V1 IN 0 DC <напряжение> - источник, 0, если напряжение не задано;
R1 <узел> <узел> <номинал>, C1 ..., L1 ... - резисторы, конденсаторы, катушки;
X1 <плюс> <минус> <выход> OPAMP_IDEAL - идеальный ОУ.
Узлы: IN - вход, OUT - выход, 0 - земля, N1, N2, ... - внутренние. Номиналы: 4.7k, 100n, 22u, 3.3m, 1MEG, 470p.
Схему не исправляй, пиши то, что описано. Сначала источник, затем R, C, L, затем ОУ.

Если схема вне класса (транзисторы, диоды, цифровые элементы, несколько входов), неоднозначна или номиналы не заданы, ответь одной строкой:
REFUSE out_of_class: причина
REFUSE ambiguous: причина
REFUSE needs_design: причина""",
    "ir": """Ты переводишь описание электрической схемы на русском в JSON. Отвечай только JSON, без пояснений.

Формат: {"schema": "mcnl.ir/1", "elements": [...]}. Элементы:
{"type": "V", "nodes": ["IN", "0"], "dc": "<напряжение>"} - источник, "0", если напряжение не задано;
{"type": "R", "nodes": ["a", "b"], "value": "<номинал>"}, так же "C" и "L";
{"type": "OPAMP", "nodes": ["<плюс>", "<минус>", "<выход>"]} - идеальный ОУ.
Узлы: IN - вход, OUT - выход, 0 - земля, N1, N2, ... - внутренние. Номиналы: 4.7k, 100n, 22u, 3.3m, 1MEG, 470p.
Схему не исправляй, пиши то, что описано.

Если схема вне класса (транзисторы, диоды, цифровые элементы, несколько входов), неоднозначна или номиналы не заданы, ответь:
{"refuse": "out_of_class", "reason": "причина"}; код также "ambiguous" или "needs_design".""",
}
FORMATS = tuple(SYSTEM)


def messages(prompt, answer=None, fmt="netlist"):
    """Диалог: система, запрос и, для обучения, ответ."""
    out = [{"role": "system", "content": SYSTEM[fmt]}, {"role": "user", "content": prompt}]
    return out + ([{"role": "assistant", "content": answer}] if answer is not None else [])


def answer_for(rec, fmt="netlist"):
    """Эталонный ответ модели для записи датасета."""
    if rec.get("refuse"):
        if fmt == "ir":
            return json.dumps({"refuse": rec["refuse"], "reason": rec["reason"]}, ensure_ascii=False)
        return rec["output"]
    if fmt == "ir":
        return json.dumps(rec["ir"], ensure_ascii=False)
    return rec["netlist"].strip()


def chat_text(tokenizer, msgs):
    """Текст запроса к модели. У шаблонов без роли system (Gemma) инструкция идёт в начало запроса."""
    msgs = [m for m in msgs if m["role"] != "assistant"]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as err:  # шаблон модели запрещает system
        if msgs[0]["role"] != "system" or "system" not in str(err).lower():
            raise
        merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def encode(tokenizer, msgs, max_len):
    """Токены и метки одного примера: учим только ответ, запрос в потерю не входит. None - пример длиннее max_len."""
    prefix = tokenizer(chat_text(tokenizer, msgs), add_special_tokens=False)["input_ids"]
    answer = tokenizer(msgs[-1]["content"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = prefix + answer
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + answer}


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def export(data, out, fmt="netlist", subset=None):
    """train и dev в диалогах; subset - вложенное подмножество train (1000, 2000, 3000) -> отчёт о длинах."""
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"format": fmt, "subset": subset}
    for split in ("train", "dev"):
        recs = read_jsonl(data / f"{split}.jsonl")
        if split == "train" and subset:
            recs = [r for r in recs if r.get("subset", 0) <= subset]
        rows = [{"id": r["id"], "messages": messages(r["prompt"], answer_for(r, fmt), fmt)} for r in recs]
        with open(out / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        lengths = sorted(len(r["messages"][1]["content"]) + len(r["messages"][2]["content"]) for r in rows)
        report[split] = {"records": len(rows), "chars_p50": lengths[len(lengths) // 2], "chars_p99": lengths[int(len(lengths) * 0.99)],
                         "chars_max": lengths[-1]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main(argv):
    def opt(key, default=None):
        return argv[argv.index(key) + 1] if key in argv else default
    sys.stdout.reconfigure(encoding="utf-8")
    subset = opt("--subset")
    print(json.dumps(export(opt("--data", "data/dataset/v2"), opt("--out", "out/training/v2"), opt("--format", "netlist"),
                            int(subset) if subset else None), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
