#!/usr/bin/env python3
"""Сборка colab/stage2.ipynb. Ноутбук правится здесь, а не в JSON.

Запуск:  python colab/build_notebook.py
"""
import json
from pathlib import Path


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


CELLS = [
    md("""
# Этап 2: дообучение модели

Цель: научить модель переводить описание схемы на русском в нетлист для Micro-Cap. Метод - QLoRA: веса модели заморожены в 4-битном виде, обучается только малый добавочный слой (LoRA), поэтому хватает бесплатного GPU.

**Перед началом:** Среда выполнения → Сменить среду выполнения → GPU (T4). Ячейки запускаются сверху вниз, вывод виден прямо здесь. Всё важное пишется на Google Drive в папку `mcnl/<RUN>`. Если сессия оборвалась, запустите ячейки заново: обучение продолжится с последней контрольной точки.
"""),
    md("## 1. Среда\nСтавим библиотеки и проверяем, какой GPU выдан."),
    code("""
!pip -q install -U transformers peft accelerate bitsandbytes
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("## 2. Drive и код проекта\nDrive попросит разрешение. Код берётся из GitHub, внизу видно, какой коммит запущен."),
    code("""
from google.colab import drive
drive.mount('/content/drive')
!rm -rf /content/repo && git clone -q --depth 1 https://github.com/xdstk/lang.cir.spice.mc12 /content/repo
%cd /content/repo
!git log --oneline -1
"""),
    md("""
## 3. Настройки запуска
Значения ниже - проба конвейера (фаза 2): 200 записей, задача лишь показать, что всё работает и модель учится. Это не результат работы.
"""),
    code("""
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
RUN = 'smoke1'         # имя запуска: папка на Drive
SUBSET = 1000          # вложенное подмножество train v2: 1000, 2000 или 3000
LIMIT = 200            # сколько записей взять из подмножества; 0 - все
FORMAT = 'netlist'     # netlist или ir
EPOCHS = 5
SEED = 1
OUT = f'/content/drive/MyDrive/mcnl/{RUN}'
!mkdir -p {OUT}
!python src/mcnl/training.py --data data/dataset/v2 --subset {SUBSET} --limit {LIMIT} --format {FORMAT} --out out/training/{RUN}
"""),
    md("""
## 4. Модель без обучения
Первая загрузка скачивает веса. Ответы на 30 записей dev - точка отсчёта: что умеет модель до обучения, только по инструкции.
"""),
    code("""
!python colab/predict.py --model {MODEL} --format {FORMAT} --records data/dataset/v2/dev.jsonl --limit 30 --out {OUT}/untrained_dev.jsonl --name untrained
"""),
    md("""
## 5. Обучение
Здесь видны полоса прогресса, потери `loss` по шагам и `eval_loss` по эпохам. `loss` должен падать. Если он стоит на месте или становится `nan`, обучение идёт неправильно.
"""),
    code("""
!python colab/train_lora.py --model {MODEL} --data out/training/{RUN} --out {OUT} --epochs {EPOCHS} --seed {SEED}
"""),
    md("## 6. График обучения\nСохраняется на Drive как `loss.png`: годится для текста работы."),
    code("""
import json, matplotlib.pyplot as plt
run = json.load(open(f'{OUT}/run.json', encoding='utf-8'))
tr = [(x['step'], x['loss']) for x in run['log'] if 'loss' in x]
ev = [(x['step'], x['eval_loss']) for x in run['log'] if 'eval_loss' in x]
plt.figure(figsize=(7, 4))
plt.plot(*zip(*tr), label='обучающая выборка')
if ev:
    plt.plot(*zip(*ev), 'o-', label='dev')
plt.xlabel('шаг обучения'); plt.ylabel('потери'); plt.legend(); plt.grid(alpha=0.3)
plt.title(RUN); plt.savefig(f'{OUT}/loss.png', dpi=200, bbox_inches='tight'); plt.show()
print(f"Время обучения: {run['train_seconds'] / 60:.1f} мин, пик памяти GPU: {run['gpu_peak_gb']:.1f} ГБ, GPU: {run['gpu']}")
print('Отброшено длинных примеров (train, dev):', run['dropped_long'])
print('Версии:', run['versions'])
"""),
    md("""
## 7. Ответы обученной модели
Три набора: 30 записей из обучающей выборки (модель должна справиться почти полностью), dev и запросы C01, которых модель не видела и которые написаны не шаблонами.
"""),
    code("""
A = f'--model {MODEL} --adapter {OUT}/adapter --format {FORMAT} --name {RUN}'
!python colab/predict.py {A} --records data/dataset/v2/train.jsonl --limit 30 --out {OUT}/train30.jsonl
!python colab/predict.py {A} --records data/dataset/v2/dev.jsonl --limit 30 --out {OUT}/dev.jsonl
!python colab/predict.py {A} --records data/human_test/v1/paraphrase.jsonl --out {OUT}/paraphrase.jsonl
"""),
    md("""
## 8. Что получилось
Здесь быстрая прикидка: точное совпадение строк ответа с эталоном, без учёта порядка. Настоящая оценка с MC12 идёт отдельно, на ноутбуке с Micro-Cap: она считает схему верной и при других именах узлов, поэтому её цифры выше.
"""),
    code("""
import sys
sys.path.insert(0, 'src')
from mcnl import training
from IPython.display import display, Markdown


def lines(text):
    return sorted(l.strip() for l in (text or '').splitlines() if l.strip())


def show(title, records, answers, n=4):
    gold = {r['id']: r for r in training.read_jsonl(records)}
    rows = training.read_jsonl(answers)
    same = sum(lines(a['output']) == lines(training.answer_for(gold[a['id']], FORMAT)) for a in rows)
    display(Markdown(f'### {title}: совпало строка в строку {same} из {len(rows)}'))
    for a in rows[:n]:
        g = gold[a['id']]
        print('ЗАПРОС:', g['prompt'][:200]); print('ЭТАЛОН:', training.answer_for(g, FORMAT).replace(chr(10), ' | '))
        print('ОТВЕТ: ', a['output'].replace(chr(10), ' | ')[:300]); print()


show('Без обучения, dev', 'data/dataset/v2/dev.jsonl', f'{OUT}/untrained_dev.jsonl')
show('Обучающая выборка', 'data/dataset/v2/train.jsonl', f'{OUT}/train30.jsonl')
show('Обученная модель, dev', 'data/dataset/v2/dev.jsonl', f'{OUT}/dev.jsonl')
show('Запросы C01', 'data/human_test/v1/paraphrase.jsonl', f'{OUT}/paraphrase.jsonl')
"""),
]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("stage2.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
