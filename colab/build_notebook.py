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
               "about": "Фаза 2, проба конвейера: 200 записей, 5 эпох. Обучение уже сделано (`run.json` на Drive), ячейки 4 и 6 "
                        "можно пропустить; ячейка 8 дописывает ответы dev до полных 150 - для итоговой таблицы."},
    "q35_4b": {"model": "Qwen/Qwen3.5-4B", "subset": 2000, "limit": 0, "epochs": 2,
               "about": "Фаза 3, отбор моделей: Qwen3.5-4B. Условия у всех кандидатов одни: v2, 2000 записей, 2 эпохи, seed 1."},
    "q35_9b": {"model": "Qwen/Qwen3.5-9B", "subset": 2000, "limit": 0, "epochs": 2, "batch": 1,
               "about": "Фаза 3, отбор моделей: Qwen3.5-9B, старшая модель того же семейства. Условия те же, что у 4B; "
                        "батч 1 с накоплением 8 вместо 2 x 4 - шаг оптимизации тот же, а памяти на логиты и активации "
                        "вдвое меньше (у 4B пик был 10,7 ГБ из 15). Сначала ячейка 4: 10 шагов, память и время. 4B "
                        "училась около 16 с на шаг (2,2 ч); 9B ожидается примерно вдвое дольше, то есть в две сессии - "
                        "ячейка 6 продолжает с контрольной точки на Drive (они каждые 50 шагов), время всех сессий "
                        "складывается в run.json."},
    "gemma4_e4b": {"model": "google/gemma-4-E4B-it", "subset": 2000, "limit": 0, "epochs": 2,
                   "about": "Фаза 3, отбор моделей: Gemma 4 E4B (Apache 2.0), другое семейство. Модель грузится целиком, "
                            "с башнями зрения и звука; LoRA - только на языковую часть. Из 7,9 млрд параметров 3,5 млрд - "
                            "встраивания по слоям, они не квантуются: память на T4 покажет ячейка 4."},
    "phi4_mini": {"model": "microsoft/Phi-4-mini-instruct", "subset": 2000, "limit": 0, "epochs": 2,
                  "about": "Фаза 3, отбор моделей: Phi-4-mini (3,8B, MIT), вторая модель 3-4B не от Qwen (решение автора "
                           "06.11.2026: 9B не лучше 4B значимо; Gemma 4 E4B на ноутбук автора целиком не входит). "
                           "Условия те же, что у Qwen3.5-4B (батч 2 x 4). Ответы на остальные выборки можно считать и "
                           "локально: модель в 4 битах около 3-3,5 ГБ."},
    "q35_4b_seed2": {"model": "Qwen/Qwen3.5-4B", "subset": 2000, "limit": 0, "epochs": 2, "seed": 2,
                     "about": "Разброс от повторного обучения (решение автора 07.11.2026): Qwen3.5-4B в тех же условиях, "
                              "что `q35_4b`, кроме seed (2 вместо 1) - меняются порядок записей и начальные веса LoRA. "
                              "Разница с `q35_4b` - мера шума обучения: с ней сравниваются различия между моделями."},
    "yagpt5_lite": {"model": "yandex/YandexGPT-5-Lite-8B-instruct", "subset": 2000, "limit": 0, "epochs": 2,
                    "about": "Фаза 3, отбор моделей: YandexGPT 5 Lite 8B, русскоязычная модель (собственная лицензия, "
                             "проверена автором). Шаблон чата не знает роли system - инструкция идёт в начало запроса."},
}


def settings(run, r):
    return "\n".join([
        f"MODEL = '{r['model']}'",
        f"RUN = '{run}'".ljust(23) + "# имя запуска: папка на Drive",
        f"SUBSET = {r['subset']}".ljust(23) + "# вложенное подмножество train v2: 1000, 2000 или 3000",
        f"LIMIT = {r['limit']}".ljust(23) + "# сколько записей взять из подмножества; 0 - все",
        "FORMAT = 'netlist'".ljust(23) + "# netlist или ir",
        f"EPOCHS = {r['epochs']}",
        f"BATCH = {r.get('batch', 2)}".ljust(23) + "# записей за проход; шаг оптимизации всегда 8 записей",
        "ACCUM = 8 // BATCH",
        f"SEED = {r.get('seed', 1)}".ljust(23) + "# порядок записей и начальные веса LoRA",
        "OUT = f'/content/drive/MyDrive/mcnl/{RUN}'",
        "!mkdir -p {OUT}",
        "!python src/mcnl/training.py --data data/dataset/v2 --subset {SUBSET} --limit {LIMIT} --format {FORMAT} --out out/training/{RUN}",
    ])


def cells(run, r):
    return [
    md("# Этап 2: " + r["model"] + " (`" + run + "`)\n\n" + r["about"] + """

Цель: научить модель переводить описание схемы на русском в нетлист для Micro-Cap. Метод - QLoRA: веса модели заморожены в 4-битном виде, обучается только малый добавочный слой (LoRA), поэтому хватает бесплатного GPU.

**Перед началом:** Среда выполнения → Сменить среду выполнения → GPU (T4). Ячейки запускаются сверху вниз, вывод виден прямо здесь. Всё важное пишется на Google Drive в папку `mcnl/<RUN>`. Если сессия оборвалась, запустите ячейки заново: обучение продолжится с последней контрольной точки, готовые шаги пропускаются, а ответы модели дописываются с места обрыва.
"""),
    md("## 1. Среда\nСтавим библиотеки и проверяем, какой GPU выдан."),
    code("""
%cd /content
!pip -q install -U transformers peft accelerate bitsandbytes sentencepiece "protobuf>=7.36"
!nvidia-smi --query-gpu=name,memory.total --format=csv
# токенизатор SentencePiece (YandexGPT) читается через protobuf: transformers собран под 7.x, в Colab стоит 6.x
import google.protobuf
print('protobuf', google.protobuf.__version__, '- ниже 7: Среда выполнения -> Перезапустить сеанс и снова эту ячейку'
      if int(google.protobuf.__version__.split('.')[0]) < 7 else '')
"""),
    md("""
## 2. Drive и код проекта
Drive попросит разрешение. Ячейка проверяет свободное место: нужно не меньше 3 ГБ (контрольные точки, адаптер, ответы). Удалённые файлы остаются в корзине Drive и занимают место, пока корзину не очистить. Код берётся из GitHub, внизу видно, какой коммит запущен.
Если репозиторий закрытый, нужен токен только на чтение: значок ключа слева → секрет `GITHUB_TOKEN`, доступ для ноутбука включён. В текст ячеек и в чат токен не вставлять.
"""),
    code("""
import os, subprocess
os.chdir('/content')  # при повторном запуске текущая папка - /content/repo, которую ячейка сейчас удалит
from google.colab import drive, userdata
drive.mount('/content/drive')


def drive_free(need=3):
    # Свободно на Drive, ГБ. Colab пишет на Drive через кэш: когда место кончилось, ячейки идут без ошибок, а файлы
    # на Drive не попадают (08.11.2026, Phi-4-mini). Удалённые контрольные точки лежат в корзине Drive и занимают место.
    import shutil
    free = shutil.disk_usage('/content/drive/MyDrive').free / 1e9
    print(f'Свободно на Drive: {free:.1f} ГБ')
    if free < need:
        raise SystemExit(f'На Drive меньше {need} ГБ: очистите корзину Drive (drive.google.com -> Корзина -> '
                         'Очистить корзину) и запустите ячейку снова. Иначе результаты не сохранятся.')
    return free


drive_free()
try:
    token = userdata.get('GITHUB_TOKEN')
except Exception:
    token = None  # открытый репозиторий: токен не нужен
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
    md("""
## 3. Настройки запуска
Настройки этого запуска; ноутбуки других моделей отличаются только ими.
"""),
    code(settings(run, r)),
    md("""
## 4. Проба модели
Шаблон чата: в начале - инструкция (у YandexGPT она внутри запроса пользователя), в конце запроса - начало ответа ассистента (у Qwen3.5 с пустым блоком `<think></think>`: рассуждение выключено). Учиться модель должна только нетлисту и знаку конца реплики (`<|im_end|>`, `<turn|>` или `</s>`). Затем 10 шагов обучения во временную папку: проверка, что модель обучается на T4, и оценка времени полного запуска.
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
    md("""
## 5. Модель без обучения
Первая загрузка скачивает веса. Ответы на 30 записей dev - точка отсчёта: что умеет модель до обучения, только по инструкции.
"""),
    code("""
!python colab/predict.py --model {MODEL} --format {FORMAT} --records data/dataset/v2/dev.jsonl --limit 30 --out {OUT}/untrained_dev.jsonl --name untrained
"""),
    md("""
## 6. Обучение
Здесь видны полоса прогресса, потери `loss` по шагам и `eval_loss` по эпохам. `loss` должен падать. Если он стоит на месте или становится `nan`, обучение идёт неправильно.
"""),
    code("""
!python colab/train_lora.py --model {MODEL} --data out/training/{RUN} --out {OUT} --epochs {EPOCHS} --seed {SEED} --batch {BATCH} --accum {ACCUM}
drive_free(1)
"""),
    md("## 7. График обучения\nСохраняется на Drive как `loss.png`: годится для текста работы."),
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
    md("""
## 8. Ответы обученной модели
Три набора: 30 записей из обучающей выборки (модель должна справиться почти полностью), весь dev (150) и запросы C01, которых модель не видела и которые написаны не шаблонами.
"""),
    code("""
A = f'--model {MODEL} --adapter {OUT}/adapter --format {FORMAT} --name {RUN}'
!python colab/predict.py {A} --records data/dataset/v2/train.jsonl --limit 30 --out {OUT}/train30.jsonl
!python colab/predict.py {A} --records data/dataset/v2/dev.jsonl --out {OUT}/dev.jsonl
!python colab/predict.py {A} --records data/human_test/v1/paraphrase.jsonl --out {OUT}/paraphrase.jsonl
drive_free(0.2)
"""),
    md("""
### 8б. Остальные выборки (для итоговой таблицы)
test v2 (500), `gen_*` (300), `mc12_real` (13): около 800 запросов. Нужны для моделей, которые не помещаются на ноутбук автора (9B); остальные можно считать локально. Каждый файл продолжается с места обрыва: если сессия кончится, запустить ячейки 1-3 и эту снова.
"""),
    code("""
A = f'--model {MODEL} --adapter {OUT}/adapter --format {FORMAT} --name {RUN}'
!python colab/predict.py {A} --records data/dataset/mc12_real_v1/mc12_real.jsonl --out {OUT}/mc12_real.jsonl
for s in ['gen_topology', 'gen_family', 'gen_range', 'test']:
    !python colab/predict.py {A} --records data/dataset/v2/{s}.jsonl --out {OUT}/{s}.jsonl
drive_free(0.2)
"""),
    md("""
## 9. Что получилось
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
    md("""
## 10. Дописать на Drive
Colab отправляет файлы на Drive не сразу. Перед закрытием сессии - эта ячейка: она дожидается отправки и отключает Drive. Ниже - что лежит в папке запуска; эти же файлы должны быть видны на drive.google.com.
"""),
    code("""
!ls -la {OUT}
drive_free(0)
drive.flush_and_unmount()
print('Drive отключён, файлы отправлены. Для новых ячеек - снова ячейка 2.')
"""),
]

for run, r in RUNS.items():
    nb = {"cells": cells(run, r), "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(__file__).with_name(f"stage2_{run}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
