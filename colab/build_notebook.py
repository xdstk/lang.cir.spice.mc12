#!/usr/bin/env python3
"""Сборка ноутбуков этапа 2: по одному на запуск, colab/stage2_<RUN>.ipynb (решение автора 03.11.2026).
Ноутбуки правятся здесь, а не в JSON, и отличаются только настройками из RUNS.

Запуск:  python colab/build_notebook.py
"""
import json
from pathlib import Path


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


RUNS = {
    "smoke1": {"model": "Qwen/Qwen2.5-7B-Instruct", "subset": 1000, "limit": 200, "epochs": 5,
               "about": "Проба конвейера: 200 записей train v2, 5 эпох."},
    "q35_4b": {"model": "Qwen/Qwen3.5-4B", "subset": 2000, "limit": 0, "epochs": 2,
               "about": "Train v2, 2000 записей, 2 эпохи, seed 1."},
    "q35_9b": {"model": "Qwen/Qwen3.5-9B", "subset": 2000, "limit": 0, "epochs": 2, "batch": 1,
               "about": "Train v2, 2000 записей, 2 эпохи, seed 1. Батч 1 с накоплением 8: шаг оптимизации тот же, "
                        "памяти меньше. Обучение может занять две сессии - ячейка 6 продолжит с контрольной точки."},
    "phi4_mini": {"model": "microsoft/Phi-4-mini-instruct", "subset": 2000, "limit": 0, "epochs": 2,
                  "about": "Train v2, 2000 записей, 2 эпохи, seed 1."},
    "q35_4b_seed2": {"model": "Qwen/Qwen3.5-4B", "subset": 2000, "limit": 0, "epochs": 2, "seed": 2,
                     "about": "Повтор `q35_4b` с seed 2 (порядок записей и начальные веса LoRA): разброс от обучения."},
    "yagpt5_lite": {"model": "yandex/YandexGPT-5-Lite-8B-instruct", "subset": 2000, "limit": 0, "epochs": 2,
                    "sentencepiece": True,
                    "about": "Train v2, 2000 записей, 2 эпохи, seed 1. Шаблон чата без роли system: инструкция идёт "
                             "в начало запроса."},
}


def settings(run, r):
    return "\n".join([
        f"MODEL = '{r['model']}'",
        f"RUN = '{run}'".ljust(23) + "# папка на Drive",
        f"SUBSET = {r['subset']}".ljust(23) + "# подмножество train v2: 1000, 2000 или 3000",
        f"LIMIT = {r['limit']}".ljust(23) + "# 0 - всё подмножество",
        "FORMAT = 'netlist'".ljust(23) + "# netlist или ir",
        f"EPOCHS = {r['epochs']}",
        f"BATCH = {r.get('batch', 2)}".ljust(23) + "# шаг оптимизации всегда 8 записей",
        "ACCUM = 8 // BATCH",
        f"SEED = {r.get('seed', 1)}",
        "OUT = f'/content/drive/MyDrive/mcnl/{RUN}'",
        "!mkdir -p {OUT}",
        "!python src/mcnl/training.py --data data/dataset/v2 --subset {SUBSET} --limit {LIMIT} --format {FORMAT} --out out/training/{RUN}",
    ])


def setup(r):
    if not r.get("sentencepiece"):
        return """
%cd /content
!pip -q install -U transformers peft accelerate bitsandbytes
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""
    return """
%cd /content
!pip -q install -U transformers peft accelerate bitsandbytes sentencepiece "protobuf>=7.36"
!nvidia-smi --query-gpu=name,memory.total --format=csv
# токенизатор SentencePiece читается через protobuf 7.x, в Colab стоит 6.x
import google.protobuf
print('protobuf', google.protobuf.__version__, '- ниже 7: Среда выполнения -> Перезапустить сеанс и снова эту ячейку'
      if int(google.protobuf.__version__.split('.')[0]) < 7 else '')
"""


def cells(run, r):
    return [
    md("# Этап 2: " + r["model"] + " (`" + run + "`)\n\n" + r["about"] + """

QLoRA: веса модели заморожены в 4 битах, обучается только добавочный слой LoRA. Среда выполнения - GPU (T4). Результаты пишутся на Drive в `mcnl/<RUN>`; после обрыва сессии ячейки запускаются заново, обучение и ответы продолжаются с места обрыва.
"""),
    md("## 1. Среда"),
    code(setup(r)),
    md("""
## 2. Drive и код проекта
Код берётся из GitHub. Для закрытого репозитория - секрет `GITHUB_TOKEN` (только чтение) в панели ключей слева.
"""),
    code("""
import os, subprocess
os.chdir('/content')
from google.colab import drive, userdata
drive.mount('/content/drive')
try:
    token = userdata.get('GITHUB_TOKEN')
except Exception:
    token = None
REPO = 'github.com/xdstk/lang.cir.spice.mc12'
!rm -rf /content/repo
done = subprocess.run(['git', 'clone', '-q', '--depth', '1', f'https://x-access-token:{token}@{REPO}' if token else f'https://{REPO}', '/content/repo'],
                      capture_output=True, text=True, env={'GIT_TERMINAL_PROMPT': '0', 'PATH': '/usr/bin:/bin:/usr/local/bin'})
if done.returncode:
    raise SystemExit('Код не скачан: ' + done.stderr.replace(token or '<нет>', '***').strip())
!git -C /content/repo remote set-url origin https://{REPO}
%cd /content/repo
!git log --oneline -1
"""),
    md("## 3. Настройки"),
    code(settings(run, r)),
    md("""
## 4. Проба
Шаблон чата и то, чему учится модель (только нетлист и знак конца реплики); затем 10 шагов обучения - память и время.
"""),
    code("""
import sys
sys.path.insert(0, 'src')
from transformers import AutoTokenizer
from mcnl import training
tok = AutoTokenizer.from_pretrained(MODEL)
recs = training.read_jsonl(f'out/training/{RUN}/train.jsonl')
end = training.end_of_turn(tok)
e = training.encode(tok, recs[0]['messages'], 10 ** 6, end)
print(tok.decode(e['input_ids'])[-300:])
print('УЧИТСЯ:', repr(tok.decode([t for t, l in zip(e['input_ids'], e['labels']) if l != -100])))
lens = sorted(len(training.encode(tok, r['messages'], 10 ** 6, end)['input_ids']) for r in recs)
print(f'токенов в примере: медиана {lens[len(lens) // 2]}, максимум {lens[-1]}')
!rm -rf /content/probe
!python colab/train_lora.py --model {MODEL} --data out/training/{RUN} --out /content/probe --epochs {EPOCHS} --seed {SEED} --batch {BATCH} --accum {ACCUM} --max-steps 10
"""),
    md("## 5. Модель без обучения\n30 записей dev - точка отсчёта."),
    code("""
!python colab/predict.py --model {MODEL} --format {FORMAT} --records data/dataset/v2/dev.jsonl --limit 30 --out {OUT}/untrained_dev.jsonl --name untrained
"""),
    md("## 6. Обучение\n`loss` должен падать; `nan` или плато - ошибка."),
    code("""
!python colab/train_lora.py --model {MODEL} --data out/training/{RUN} --out {OUT} --epochs {EPOCHS} --seed {SEED} --batch {BATCH} --accum {ACCUM}
"""),
    md("## 7. График обучения\nСохраняется на Drive как `loss.png`."),
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
print(f"Время обучения: {run['train_seconds'] / 60:.1f} мин (все сессии), пик памяти GPU: {run['gpu_peak_gb']:.1f} ГБ, GPU: {run['gpu']}")
print('Отброшено длинных примеров (train, dev):', run['dropped_long'])
print('Версии:', run['versions'])
"""),
    md("## 8. Ответы обученной модели\n30 записей train, весь dev (150), запросы C01."),
    code("""
A = f'--model {MODEL} --adapter {OUT}/adapter --format {FORMAT} --name {RUN}'
!python colab/predict.py {A} --records data/dataset/v2/train.jsonl --limit 30 --out {OUT}/train30.jsonl
!python colab/predict.py {A} --records data/dataset/v2/dev.jsonl --out {OUT}/dev.jsonl
!python colab/predict.py {A} --records data/human_test/v1/paraphrase.jsonl --out {OUT}/paraphrase.jsonl
"""),
    md("### 8б. Остальные выборки\n`mc12_real`, `gen_*`, test v2 - около 800 запросов; после обрыва - ячейки 1-3 и эта."),
    code("""
A = f'--model {MODEL} --adapter {OUT}/adapter --format {FORMAT} --name {RUN}'
!python colab/predict.py {A} --records data/dataset/mc12_real_v1/mc12_real.jsonl --out {OUT}/mc12_real.jsonl
for s in ['gen_topology', 'gen_family', 'gen_range', 'test']:
    !python colab/predict.py {A} --records data/dataset/v2/{s}.jsonl --out {OUT}/{s}.jsonl
"""),
    md("## 9. Быстрая проверка\nСовпадение строк ответа с эталоном без учёта порядка. Точная оценка - в MC12, отдельно."),
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
    display(Markdown(f'### {title}: совпало {same} из {len(rows)}'))
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

for run, r in RUNS.items():
    nb = {"cells": cells(run, r), "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(__file__).with_name(f"stage2_{run}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
