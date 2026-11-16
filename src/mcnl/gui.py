#!/usr/bin/env python3
"""Окно программы (tkinter): запрос -> нетлист -> MC12 -> АЧХ и схема .CIR (docs/gui_spec.md).

Запуск:  ..\\venv-hf\\Scripts\\python.exe src/mcnl/gui.py      (Qwen3.5-4B нужен torch из venv-hf)
         python src/mcnl/gui.py                               (без torch доступен только baseline)

Окно - оболочка над app.run: своей логики схем здесь нет. Модель и MC12 работают в фоновом потоке,
результат приходит в главный поток через очередь. Рабочая папка - out/gui (обычная папка: MC12 не видит
перенаправленные папки приложения Claude). Настройки (пути к модели, реальный ОУ) - out/gui/settings.json.
"""
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import app, canon, handbook, handbook_schematics, measure, mc12run, schematic, translators  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "gui"
# адаптер режима 1 (trained_on - на чём обучен: инструкция), адаптер режима 2 - на той же базе; пока m2 v3.1 не обучен -
# общая модель v3 (docs/mode2_spec.md, раздел 9); нет папки - режим 2 у Qwen недоступен
DEFAULTS = {"base": "Qwen/Qwen3.5-4B", "adapter": str(ROOT / "out" / "q35_4b" / "adapter"), "trained_on": "v2",
            "adapter_mode2": str(ROOT / "out" / "q35_4b_v3" / "adapter"), "trained_on_mode2": "v3",
            "real_opamp": True}  # реальный ОУ на чертеже по умолчанию (решение автора 16.11.2026)
MODE_NAMES = {1: "перевод описания", 2: "подбор номиналов"}
QWEN, BASELINE = "Qwen3.5-4B (локально)", "baseline (проверка окна)"
KIND = {"LP": "ФНЧ", "HP": "ФВЧ", "BP": "полосовой", "BS": "режекторный", "flat": "без частотной зависимости"}
EXAMPLE = "ФНЧ Саллена-Ки на повторителе: R1 10 кОм, R2 10 кОм, C1 (ОС) 22 нФ, C2 (земля) 10 нФ; сигнал на вход"
# Меню «Примеры» (шаг 5): разные классы схем; все проверены на Qwen3.5-4B и MC12 07.11.2026 (тип, fc / f0, Q, K
# совпали с формулами), отказы - с верным кодом.
EXAMPLES = [
    ("Делитель напряжения", "Делитель напряжения: верхний резистор 10 кОм, нижний 4,7 кОм; на вход 5 В"),
    ("RC ФВЧ по соединениям", "RC ФВЧ: конденсатор 100 нФ от входа к выходу, резистор 10 кОм с выхода на землю; "
                              "сигнал на вход"),
    ("Двухзвенная RC-цепочка", "R1 10 кОм от входа к точке 1, C1 100 нФ от точки 1 на землю, R2 22 кОм от точки 1 "
                               "к выходу, C2 10 нФ с выхода на землю; сигнал на вход"),
    ("Последовательный RLC-контур", "Последовательный RLC-контур, выход на резисторе: катушка 10 мГн, конденсатор "
                                    "100 нФ, резистор 100 Ом; сигнал на вход"),
    ("Инвертирующий усилитель", "Инвертирующий усилитель на ОУ: входной резистор 4,7 кОм, резистор обратной связи "
                                "47 кОм; сигнал на вход"),
    ("Неинвертирующий усилитель", "Неинвертирующий усилитель на ОУ: резистор на землю 10 кОм, резистор обратной "
                                  "связи 22 кОм; сигнал на вход"),
    ("ФНЧ Саллена-Ки", EXAMPLE),
    ("RC ФНЧ с добавкой на выходе", "RC ФНЧ: резистор 1 кОм, конденсатор 100 нФ; после выхода ещё 470 Ом "
                                    "последовательно, выход за ним"),
    ("Отказ: транзистор", "Усилитель на транзисторе КТ315 с коэффициентом усиления 20"),
    ("Отказ: номиналы не заданы", "Собери ФНЧ на 1 кГц"),
]
# Примеры режима 2 (подбор номиналов): выбор переключает режим. Конвейер проверен на эталонных ответах test v3,
# ответы модели на эти запросы - не проверены (адаптера режима 2 локально ещё нет, 16.11.2026).
EXAMPLES_MODE2 = [
    ("Подбор: ФНЧ на 1 кГц", "Нужен ФНЧ на 1 кГц"),
    ("Подбор: активный ФНЧ 4-го порядка", "Активный ФНЧ Баттерворта 4-го порядка на 2 кГц"),
    ("Подбор: полосовой на ОУ, Q = 5", "Активный полосовой фильтр на 5 кГц с добротностью 5"),
    ("Подбор: Саллен-Ки с заданными C", "ФНЧ Саллена-Ки на повторителе, срез 1 кГц; C1 (ОС) 22 нФ, C2 (земля) 10 нФ, "
                                        "резисторы подобрать"),
]
# Вкладка «Правила» (docs/gui_spec.md, шаг 3): ("h", заголовок) | ("p", пункт) | ("ex", пример с кнопкой «В запрос»).
# Примеры проверены на Qwen3.5-4B и MC12 07.11.2026: схемы построены верно, отказы - с верным кодом.
RULES = [
    ("h", "Что умеет программа"),
    ("p", "Пассивные схемы из резисторов, конденсаторов и катушек: делители, RC, RL и RLC-фильтры, цепочки звеньев."),
    ("p", "Схемы на операционных усилителях: усилители, активные фильтры, каскады из 2-3 звеньев."),
    ("p", "Один источник сигнала между входом и землёй. Программа считает АЧХ в Micro-Cap 12 и строит чертёж схемы."),
    ("h", "Как писать запрос"),
    ("p", "Назовите каждый элемент и его номинал с единицами: 10 кОм, 4,7 кОм, 100 нФ, 22 мкФ, 3,3 мГн."),
    ("p", "Скажите, куда подключён каждый элемент: «от входа к точке 1», «с выхода на землю», «между точками 1 и 2»."),
    ("p", "Вместо соединений можно назвать типовую схему (инвертирующий усилитель, ФНЧ Саллена-Ки, фильтр Рауха) "
          "и дать номиналы по ролям: «резистор обратной связи 47 кОм»."),
    ("p", "Если после схемы добавлены элементы, укажите, откуда снимать выход: «выход за ним» или «выход остаётся на месте»."),
    ("p", "Питание ОУ указывать не нужно: на чертеже программа ставит ОУ TL071 с питанием ±15 В, без флажка "
          "«Реальный ОУ» - идеальный."),
    ("p", "Примеры - кнопка «В запрос» переносит пример на вкладку «Схема»:"),
    ("ex", "R1 10 кОм от входа к точке 1, C1 100 нФ от точки 1 на землю, R2 22 кОм от точки 1 к выходу, "
           "C2 10 нФ с выхода на землю; сигнал на вход"),
    ("ex", "Инвертирующий усилитель на ОУ: входной резистор 4,7 кОм, резистор обратной связи 47 кОм; сигнал на вход"),
    ("ex", "RC ФНЧ: резистор 1 кОм, конденсатор 100 нФ; после выхода ещё 470 Ом последовательно, выход за ним"),
    ("h", "Режим 2: подбор номиналов"),
    ("p", "Переключатель «Режим» - «подбор номиналов»: напишите тип фильтра и частоту, остальное подберёт программа."),
    ("p", "Можно добавить порядок, Баттерворт или Чебышёв, добротность, усиление, «активный» или «на ОУ», "
          "назвать схему и задать часть номиналов. Без слова «активный» фильтр пассивный."),
    ("p", "Модель пишет схему с неизвестными «?» и строку целей .SPEC, номиналы считает программа (R - ряд E24, C и L - "
          "E12) и проверяет в MC12. В «Замечаниях» - что подобрано и насколько попали в цель."),
    ("p", "Строку .SPEC в поле справа можно поправить (например, fc=2k) и нажать «Пересчитать» - без модели."),
    ("ex", "Нужен ФНЧ на 1 кГц"),
    ("h", "Когда программа откажет"),
    ("p", "Транзисторы, диоды, цифровые микросхемы - вне класса схем."),
    ("p", "Несколько входов (сумматор, вычитатель, дифференциальный усилитель)."),
    ("p", "Номиналы не заданы: «собери ФНЧ на 1 кГц» - это расчёт фильтра, он появится в режиме дизайнера."),
    ("p", "Описание неоднозначно: не сказано, куда подключён элемент, или у элемента два номинала."),
    ("p", "Пример отказа:"),
    ("ex", "Собери ФНЧ на 1 кГц"),
    ("h", "Что дальше"),
    ("p", "Проверьте нетлист справа: его можно поправить и нажать «Пересчитать нетлист» - модель при этом не вызывается."),
    ("p", "Замечания под кнопками подскажут, если элемент висит в воздухе или выход ни к чему не подключён."),
    ("p", "«Открыть в MC12» показывает чертёж схемы в Micro-Cap 12."),
    ("p", "Вкладка «Справочник» - схемы фильтров и ОУ: формулы, применение, достоинства и недостатки, пример с "
          "чертежом и АЧХ из MC12 и готовые запросы для программы."),
    ("p", "С реальным ОУ на графике две АЧХ: сплошная - чертежа с TL071, пунктир - с идеальным ОУ. Расходятся они на "
          "высоких частотах, где у TL071 кончается полоса (около 3 МГц на единичном усилении)."),
    ("p", "Если реальный ОУ насытился бы (большое постоянное напряжение на входе, интегратор без резистора в обратной "
          "связи), чертёж остаётся с идеальным ОУ, а замечание объясняет почему."),
]


# ---------------------------------------------------------------- данные для окна (без tkinter, есть тесты)

def freq_label(f):
    """12.3 -> «12,3 Гц», 1530 -> «1,53 кГц»."""
    if f is None:
        return "-"
    for scale, unit in ((1e9, "ГГц"), (1e6, "МГц"), (1e3, "кГц"), (1, "Гц")):
        if abs(f) >= scale or scale == 1:
            return f"{f / scale:.3g} {unit}".replace(".", ",")
    return "-"


def cards(m):
    """Карточки результата: [(подпись, значение)] из measure.from_output."""
    if not m:
        return [("Тип", "-"), ("fc / f0", "-"), ("K", "-"), ("Q", "-")]
    t = m["type"]
    if t in ("LP", "HP"):
        freq = ("fc", freq_label(m.get("fc")))
    elif t in ("BP", "BS"):
        freq = ("f0", freq_label(m.get("f0")))
    else:
        freq = ("fc / f0", "-")
    k = m.get("k_pass_db")
    q = m.get("Q") if t != "flat" else None
    return [("Тип", KIND.get(t, t)), freq,
            ("K", "-" if k is None else f"{round(k, 1) + 0.0:.1f} дБ".replace(".", ",")),
            ("Q", "-" if q is None else f"{q:.2g}".replace(".", ","))]


PLOT_TITLE = "АЧХ (расчёт MC12)"
NETLIST_TITLE = {1: "Нетлист (можно править и пересчитать)",
                 2: "Ответ модели: ? - номиналы подобрать, .SPEC - цели (можно править и пересчитать)"}


def plot_title(rep):
    """Подпись графика с двумя кривыми: что сплошная, что пунктир."""
    return f"АЧХ (расчёт MC12): сплошная - чертёж с {rep.get('opamp')}, пунктир - идеальный ОУ"


def plot_geometry(f, db, width, height, pad=(42, 10, 10, 24), span_db=80, also=None):
    """Точки АЧХ на холсте (логарифмическая ось частоты) и линии сетки.
    pad = (слева, сверху, справа, снизу). -> {"lines": [[x, y, ...], ...], "xgrid": [(x, подпись)], "ygrid": [(y, подпись)]}
    Ниже нижней границы шкалы (span_db от максимума) кривая обрывается, а не прижимается к краю: иначе
    глубокий спад выглядел бы ложной полкой. also = (f, db) - вторая кривая в том же масштабе ("also_lines")."""
    pts = [(fi, d) for fi, d in zip(f, db) if fi and fi > 0 and d is not None]
    if len(pts) < 2:
        return {"lines": [], "xgrid": [], "ygrid": [], "also_lines": []}
    pts2 = [(fi, d) for fi, d in zip(*also) if fi and fi > 0 and d is not None] if also else []
    l, t, r, b = pad
    w, h = width - l - r, height - t - b
    x0, x1 = math.log10(pts[0][0]), math.log10(pts[-1][0])
    # шкалу задаёт основная кривая, вторая обрезается; подъём меньше 0,5 дБ (Q чуть больше 0,707) шкалу не сдвигает
    top = math.ceil((max(d for _, d in pts) - 0.5) / 10) * 10 + 5
    bottom = max(min(d for _, d in pts), top - span_db)
    bottom = math.floor(bottom / 10) * 10

    def x(fi):
        return l + (math.log10(fi) - x0) / ((x1 - x0) or 1) * w

    def y(d):
        return t + (top - d) / ((top - bottom) or 1) * h
    def polylines(points):
        lines, cur = [], []
        for fi, d in points:
            if not bottom <= d <= top or not x0 - 1e-9 <= math.log10(fi) <= x1 + 1e-9:
                if len(cur) >= 4:
                    lines.append(cur)
                cur = []
                continue
            cur += [round(x(fi), 1), round(y(d), 1)]
        if len(cur) >= 4:
            lines.append(cur)
        return lines
    xgrid = [(round(x(10 ** e), 1), freq_label(10 ** e))
             for e in range(math.ceil(x0), math.floor(x1) + 1)]
    step = 20 if top - bottom > 40 else 10
    ygrid = [(round(y(v), 1), f"{v}") for v in range(int(bottom), int(top) + 1) if v % step == 0]
    return {"lines": polylines(pts), "xgrid": xgrid, "ygrid": ygrid, "x": x, "also_lines": polylines(pts2)}


def netlist_text(rep):
    """Текст поля нетлиста: в режиме 2 - ответ модели с ? и .SPEC (его правят и пересчитывают), иначе - нетлист."""
    if rep.get("synth"):
        return (rep.get("output") or "").strip()
    return (rep.get("netlist") or rep.get("output") or "").strip()


def notes_of(rep):
    """Строки замечаний: отказ, ошибки разбора, сообщения конвейера и замечания к схеме."""
    out = []
    if rep.get("refused"):
        out.append("Модель отказала: " + (rep["refused"].get("reason") or rep["refused"]["code"]))
    out += [f"Ошибка разбора: {e}" for e in rep.get("parse_errors") or []]
    out += rep.get("messages", [])
    out += [f"Замечание: {n}" for n in rep.get("notes", [])]
    return out


HISTORY_SIZE = 20


def history_result(rep):
    """Короткая подпись результата для меню истории: «ФНЧ 1,59 кГц», «отказ», «не разобран»."""
    if rep.get("refused"):
        return "отказ"
    if "netlist" not in rep:
        return "не разобран"
    m = rep.get("measured")
    if not m:
        return "без расчёта"
    kind, freq = cards(m)[0][1], cards(m)[1][1]
    return kind if freq == "-" else f"{kind} {freq}"


def read_history(path=None):
    """Последние запросы, новые первыми (файл out/gui/history.jsonl)."""
    p = Path(path or OUT / "history.jsonl")
    if not p.exists():
        return []
    items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return items[::-1]


def add_history(prompt, result, path=None, mode=1):
    """Запрос в историю: повтор (тот же запрос в том же режиме) поднимается наверх, хранится HISTORY_SIZE последних.
    Режим запоминается: запрос из истории возвращается в своём режиме."""
    p = Path(path or OUT / "history.jsonl")
    items = [h for h in read_history(p)[::-1] if (h["prompt"], h.get("mode", 1)) != (prompt, mode)]
    items.append({"prompt": prompt, "result": result, "time": time.strftime("%Y-%m-%d %H:%M"), "mode": mode})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(h, ensure_ascii=False) + "\n" for h in items[-HISTORY_SIZE:]), encoding="utf-8")


def load_settings():
    p = OUT / "settings.json"
    s = dict(DEFAULTS)
    if p.exists():
        s.update(json.loads(p.read_text(encoding="utf-8")))
    return s


def save_setting(key, value, path=None):
    """Одна настройка в out/gui/settings.json; остальные поля файла сохраняются."""
    p = Path(path or OUT / "settings.json")
    s = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    s[key] = value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


# Индикатор переводчика рядом со списком: серый мигает - модель загружается, зелёный - готов, жёлтый - запрос
# выполняется (перевод и MC12), красный - модель не загрузилась (решение автора 16.11.2026).
LAMP = {"loading": ("#b4b2a9", "загрузка…"), "ready": ("#2e9e44", "готов"), "busy": ("#e0a020", "работает…"),
        "error": ("#d03b3b", "ошибка"), "idle": ("#d3d1c7", "не выбран")}
LAMP_BLINK = "#e4e2db"  # второй цвет мигания при загрузке


def lamp_state(name, loaded, loading, errors, busy):
    """Состояние индикатора для выбранного переводчика name: ключ LAMP. busy - сколько запросов в очереди и в работе."""
    if name in errors and name not in loading:
        return "error"
    if name in loading:
        return "loading"
    if busy:
        return "busy"
    return "ready" if name in loaded else "idle"


# ---------------------------------------------------------------- окно

class Window:
    def __init__(self, root):
        self.root = root
        self.tasks, self.results = queue.Queue(), queue.Queue()
        self.translators = {}  # имя -> загруженный переводчик
        self.loading = set()
        self.load_errors = {}  # имя -> текст ошибки загрузки
        self.busy = 0  # запросов в очереди и в работе (меняется только в главном потоке)
        self.blink = False
        self.rep = None
        self.counter = 0
        self.settings = load_settings()
        OUT.mkdir(parents=True, exist_ok=True)
        root.title("mcnl - запрос в схему Micro-Cap")
        k = float(root.tk.call("tk", "scaling")) / (96 / 72)  # масштаб экрана Windows: 1,0 при 100%, 1,25 при 125%
        root.geometry(f"{int(1180 * k)}x{int(720 * k)}")
        root.minsize(int(980 * k), int(600 * k))
        self.k = k
        self._build()
        threading.Thread(target=self._worker, daemon=True).start()
        root.after(100, self._poll)
        root.after(500, self._blink)
        self._select_translator()

    # --- раскладка
    def _build(self):
        mono = ("Consolas", 10)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        main = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(main, text="Схема")
        book = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(book, text="Справочник")
        self._build_handbook(book)
        rules = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(rules, text="Правила")
        self._build_rules(rules)
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=6)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        head = ttk.Frame(left)
        head.pack(fill="x")
        ttk.Label(head, text="Запрос").pack(side="left")
        hist = ttk.Menubutton(head, text="История")
        hist.pack(side="right")
        self.hist_menu = tk.Menu(hist, tearoff=False, postcommand=self._fill_history)
        hist["menu"] = self.hist_menu
        ex = ttk.Menubutton(head, text="Примеры")
        ex.pack(side="right", padx=6)
        ex_menu = tk.Menu(ex, tearoff=False)
        for title, prompt in EXAMPLES:
            ex_menu.add_command(label=title, command=lambda s=prompt: self.use_example(s, 1))
        ex_menu.add_separator()
        for title, prompt in EXAMPLES_MODE2:
            ex_menu.add_command(label=title, command=lambda s=prompt: self.use_example(s, 2))
        ex["menu"] = ex_menu
        self.prompt = tk.Text(left, height=8, wrap="word", font=("Segoe UI", 10), undo=True)
        self.prompt.pack(fill="x", pady=(2, 8))
        self.prompt.insert("1.0", EXAMPLE)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text="Переводчик").pack(side="left")
        self.choice = tk.StringVar(value=QWEN)
        box = ttk.Combobox(row, textvariable=self.choice, values=[QWEN, BASELINE], state="readonly", width=28)
        box.pack(side="left", padx=8)
        box.bind("<<ComboboxSelected>>", lambda e: self._select_translator())
        d = int(14 * self.k)
        self.lamp = tk.Canvas(row, width=d, height=d, highlightthickness=0, background=self.root.cget("background"))
        self.lamp.pack(side="left")
        self.lamp_dot = self.lamp.create_oval(1, 1, d - 1, d - 1, fill=LAMP["loading"][0], outline="")
        self.lamp_text = tk.StringVar(value=LAMP["loading"][1])
        ttk.Label(row, textvariable=self.lamp_text, foreground="#5f5e5a").pack(side="left", padx=(5, 0))
        mrow = ttk.Frame(left)
        mrow.pack(fill="x", pady=(0, 8))
        ttk.Label(mrow, text="Режим").pack(side="left")
        self.mode = tk.IntVar(value=1)
        for m, title in MODE_NAMES.items():
            ttk.Radiobutton(mrow, text=title, value=m, variable=self.mode,
                            command=self._mode_changed).pack(side="left", padx=(8, 0))
        orow = ttk.Frame(left)
        orow.pack(fill="x", pady=(0, 8))
        self.real_opamp = tk.BooleanVar(value=bool(self.settings.get("real_opamp")))
        ttk.Checkbutton(orow, text="Реальный ОУ на чертеже (TL071, ±15 В)", variable=self.real_opamp,
                        command=lambda: save_setting("real_opamp", self.real_opamp.get())).pack(side="left")
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(0, 8))
        self.b_build = ttk.Button(buttons, text="Построить", command=self.build)
        self.b_build.pack(side="left")
        ttk.Button(buttons, text="Пересчитать нетлист", command=self.recalc).pack(side="left", padx=6)
        ttk.Button(buttons, text="Открыть в MC12", command=self.open_mc12).pack(side="left")
        ttk.Label(left, text="Замечания").pack(anchor="w", pady=(6, 0))
        self.notes = tk.Text(left, height=8, wrap="word", font=("Segoe UI", 9), state="disabled", relief="flat",
                             background=self.root.cget("background"))
        self.notes.pack(fill="both", expand=True, pady=(2, 0))

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        self.netlist_title = tk.StringVar(value=NETLIST_TITLE[1])
        ttk.Label(right, textvariable=self.netlist_title).pack(anchor="w")
        self.netlist = tk.Text(right, height=10, wrap="none", font=mono, undo=True)
        self.netlist.pack(fill="x", pady=(2, 8))
        self.card_frame = ttk.Frame(right)
        self.card_frame.pack(fill="x", pady=(0, 8))
        self.card_vars = []
        for i in range(4):
            cell = ttk.Frame(self.card_frame, padding=(8, 4), relief="groove")
            cell.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            self.card_frame.columnconfigure(i, weight=1)
            cap, val = tk.StringVar(), tk.StringVar()
            ttk.Label(cell, textvariable=cap, foreground="#5f5e5a").pack(anchor="w")
            ttk.Label(cell, textvariable=val, font=("Segoe UI", 12, "bold")).pack(anchor="w")
            self.card_vars.append((cap, val))
        self._show_cards(None)
        self.plot_title = tk.StringVar(value=PLOT_TITLE)
        ttk.Label(right, textvariable=self.plot_title).pack(anchor="w")
        self.canvas = tk.Canvas(right, height=260, background="white", highlightthickness=1,
                                highlightbackground="#d3d1c7")
        self.canvas.pack(fill="both", expand=True, pady=(2, 0))
        self.canvas.bind("<Configure>", lambda e: self._draw_plot())

        self.status = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status, padding=(10, 2), foreground="#5f5e5a").pack(fill="x")

    def _build_rules(self, frame):
        """Вкладка «Правила»: как писать запрос; примеры вставляются в запрос кнопкой."""
        text = tk.Text(frame, wrap="word", font=("Segoe UI", 10), relief="flat", padx=12, pady=8,
                       background=self.root.cget("background"), cursor="arrow")
        bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.tag_configure("h", font=("Segoe UI", 12, "bold"), spacing1=10, spacing3=4)
        text.tag_configure("p", spacing3=3, lmargin1=8, lmargin2=20)
        text.tag_configure("ex", font=("Segoe UI", 10, "italic"), lmargin1=20, lmargin2=20, spacing1=4)
        for kind, body in RULES:
            if kind == "h":
                text.insert("end", body + "\n", "h")
            elif kind == "p":
                text.insert("end", "•  " + body + "\n", "p")
            else:  # пример: текст и кнопка «В запрос»
                text.insert("end", body + "  ", "ex")
                b = ttk.Button(text, text="В запрос", command=lambda s=body: self.use_example(s))
                text.window_create("end", window=b)
                text.insert("end", "\n")
        text.configure(state="disabled")

    def _fill_history(self):
        self.hist_menu.delete(0, "end")
        items = read_history()
        if not items:
            self.hist_menu.add_command(label="Пока пусто", state="disabled")
        for h in items:
            label = h["prompt"] if len(h["prompt"]) <= 70 else h["prompt"][:67] + "…"
            mode = h.get("mode", 1)
            tag = h.get("result", "-") if mode == 1 else f"подбор: {h.get('result', '-')}"
            self.hist_menu.add_command(label=f"{label}   [{tag}]",
                                       command=lambda s=h["prompt"], m=mode: self.use_example(s, m))

    def use_example(self, prompt, mode=None):
        if mode:
            self.mode.set(mode)
            self._mode_changed()
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", prompt)
        self.tabs.select(0)
        self.status.set("Пример вставлен в запрос: нажмите «Построить»")

    # --- действия
    def _select_translator(self):
        name = self.choice.get()
        if name not in self.translators and name not in self.loading:  # после ошибки - новая попытка
            self.loading.add(name)
            self.load_errors.pop(name, None)
            self.status.set(f"{name}: загрузка…" + (" (модель грузится около 30-45 с)" if name == QWEN else ""))
            threading.Thread(target=self._load_task, args=(name,), daemon=True).start()  # не ждёт запросов в очереди
        self._update_lamp()

    def _load_task(self, name):
        try:
            self.results.put(("loaded", name, self._load(name)))
        except Exception as err:  # noqa: BLE001
            self.results.put(("load_error", name, f"{type(err).__name__}: {err}"))

    def _update_lamp(self):
        key = lamp_state(self.choice.get(), self.translators, self.loading, self.load_errors, self.busy)
        color, text = LAMP[key]
        if key == "loading" and self.blink:
            color = LAMP_BLINK
        self.lamp.itemconfigure(self.lamp_dot, fill=color)
        self.lamp_text.set(text)

    def _blink(self):
        """Мигание серого индикатора, пока модель загружается: видно, что окно живо."""
        self.blink = not self.blink
        try:
            self._update_lamp()
        finally:
            self.root.after(500, self._blink)

    def build(self):
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self.status.set("Введите запрос")
            return
        name = self.choice.get()
        if name in self.load_errors and name not in self.loading:
            self.status.set(f"{name} не загрузился: {self.load_errors[name]} - выберите переводчик в списке ещё раз")
            return
        if name in self.translators:
            self.status.set(f"Перевод запроса ({name}) и расчёт в MC12…")
        else:
            self.status.set(f"Запрос в очереди: выполнится сразу после загрузки {name}")
        self.busy += 1
        self._update_lamp()
        self.tasks.put(("build", name, (prompt, self._opamp(), self.mode.get())))

    def recalc(self):
        text = self.netlist.get("1.0", "end").strip()
        if not text:
            self.status.set("Нетлист пуст")
            return
        mode = self.mode.get()
        elems, errors = ([], None) if mode == 2 and app.is_mode2_answer(text) else canon.parse_netlist(text)
        if errors:
            self._show_notes([f"Ошибка разбора: {e}" for e in errors])
            self.status.set("Нетлист не разобран")
            return
        self.status.set("Расчёт в MC12…")
        self.busy += 1
        self._update_lamp()
        self.tasks.put(("recalc", None, (text, self._opamp(), mode)))

    def _mode_changed(self):
        self.netlist_title.set(NETLIST_TITLE[self.mode.get()])

    def _opamp(self):
        """Модель реального ОУ для чертежа или None; читается в главном потоке и уходит в задачу значением."""
        return app.OPAMP_REAL if self.real_opamp.get() else None

    def open_mc12(self):
        cir = (self.rep or {}).get("cir")
        if not cir:
            self.status.set("Сначала постройте схему")
            return
        try:
            exe = mc12run.find_exe()
        except FileNotFoundError as err:
            self.status.set(str(err))
            return
        subprocess.Popen([exe, cir], cwd=str(Path(exe).parent))
        self.status.set(f"Схема открыта в MC12: {cir}")

    # --- фоновый поток: модель и MC12
    def _worker(self):
        while True:
            kind, name, payload = self.tasks.get()
            try:
                if kind == "build":
                    tr = self.translators.get(name)
                    while tr is None and name in self.loading:  # ждём загрузку из её потока
                        time.sleep(0.2)
                        tr = self.translators.get(name)
                    if tr is None:
                        raise RuntimeError(f"{name} не загружен")
                    self.results.put(("started", name, None))
                    t0 = time.time()
                    prompt, opamp, mode = payload
                    rep = app.run(prompt, tr, outdir=OUT, name=self._next_name(), opamp=opamp, mode=mode)
                    rep["seconds"] = time.time() - t0
                    self.results.put(("done", name, rep))
                elif kind == "recalc":
                    t0 = time.time()
                    text, opamp, mode = payload
                    rep = app.run(netlist=text, outdir=OUT, name=self._next_name(), opamp=opamp, mode=mode)
                    rep["seconds"] = time.time() - t0
                    self.results.put(("done", None, rep))
            except Exception as err:  # noqa: BLE001 - ошибка показывается в окне, поток живёт дальше
                self.results.put(("error", name, f"{type(err).__name__}: {err}"))

    def _load(self, name):
        t0 = time.time()
        if name == BASELINE:
            tr = translators.load("baseline")
        else:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")  # модель уже в кэше, сеть не нужна
            st = self.settings
            m2 = st.get("adapter_mode2")
            m2 = m2 if m2 and Path(m2).exists() else None  # нет адаптера режима 2 - режим 2 только у модели на v3
            tr = translators.load("hf", base=st["base"], adapter=st["adapter"], trained_on=st.get("trained_on", "v2"),
                                  adapter_mode2=m2, trained_on_mode2=st.get("trained_on_mode2", "v3"))
        self.translators[name] = tr
        return time.time() - t0

    def _next_name(self):
        self.counter += 1
        return f"circuit_{self.counter:03d}"

    # --- главный поток: показ результатов
    def _poll(self):
        try:
            while True:
                kind, name, data = self.results.get_nowait()
                if kind == "loaded":
                    self.loading.discard(name)
                    queued = " Выполняю запрос из очереди: перевод и расчёт в MC12…" if self.busy else ""
                    self.status.set(f"{name}: загружен за {data:.0f} с" + self._gpu() + "." + queued)
                elif kind == "load_error":
                    self.loading.discard(name)
                    self.load_errors[name] = data
                    self.status.set(f"{name} не загрузился: {data}")
                elif kind == "started":
                    self.status.set(f"Перевод запроса ({name}) и расчёт в MC12…")
                elif kind == "error":
                    self.busy = max(0, self.busy - 1)
                    self.status.set(f"Ошибка: {data}")
                else:
                    self.busy = max(0, self.busy - 1)
                    self._show(data)
                self._update_lamp()
        except queue.Empty:
            pass
        except Exception as err:  # noqa: BLE001 - ошибка показа не должна останавливать опрос очереди
            self.status.set(f"Ошибка окна: {type(err).__name__}: {err}")
        finally:
            self.root.after(100, self._poll)

    def _gpu(self):
        if QWEN not in self.translators:  # пока модель грузится, torch может быть импортирован не до конца
            return ""
        try:
            import torch
            if torch.cuda.is_available():
                return f", GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} ГБ"
        except Exception:  # noqa: BLE001 - строка состояния не должна ронять окно
            pass
        return ""

    def _show(self, rep):
        self.rep = rep
        if rep.get("prompt"):
            try:
                add_history(rep["prompt"], history_result(rep), mode=rep.get("mode", 1))
            except OSError:  # история - удобство: не записалась - окно работает дальше
                pass
        self.netlist.delete("1.0", "end")
        self.netlist.insert("1.0", netlist_text(rep))
        self._show_cards(rep.get("measured"))
        self._show_notes(notes_of(rep))
        self._draw_plot()
        ok = rep.get("run", {}).get("ok")
        self.status.set(f"Готово за {rep.get('seconds', 0):.1f} с; MC12: " + ("расчёт выполнен" if ok else "нет расчёта")
                        + self._gpu())

    def _show_cards(self, m):
        for (cap, val), (c, v) in zip(self.card_vars, cards(m)):
            cap.set(c)
            val.set(v)

    def _show_notes(self, lines):
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", "\n".join(lines) if lines else "Замечаний нет")
        self.notes.configure(state="disabled")

    def _draw_plot(self):
        c = self.canvas
        c.delete("all")
        rep = self.rep or {}
        ideal = (rep.get("run") or {}).get("output")
        real = (rep.get("run_real") or {}).get("output")
        real = real if real and Path(real).exists() else None
        out = real or ideal
        if not out or not Path(out).exists():
            c.create_text(c.winfo_width() / 2, c.winfo_height() / 2, text="Здесь появится АЧХ", fill="#888780")
            return
        f, db, _ = measure.read_curve(out)
        also = measure.read_curve(ideal)[:2] if real and ideal and Path(ideal).exists() else None
        self.plot_title.set(plot_title(rep) if also else PLOT_TITLE)
        m = (self.rep or {}).get("measured") or {}
        mark = m.get("fc") or (m.get("f0") if m.get("type") in ("BP", "BS") else None)
        paint_plot(c, f, db, self.k, also, mark)

    # --- справочник (handbook.py)
    def _build_handbook(self, frame):
        """Вкладка «Справочник»: слева список статей по группам, справа статья с чертежом и АЧХ из MC12."""
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(frame, show="tree", selectmode="browse")
        tree.column("#0", width=int(260 * self.k))
        tree.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        for g in handbook.GROUPS:
            gid = tree.insert("", "end", text=g, open=True)
            for a in handbook.ARTICLES:
                if a["group"] == g:
                    tree.insert(gid, "end", iid=a["id"], text=a["title"])
        tree.bind("<<TreeviewSelect>>", lambda e: self._select_article(tree))
        t = tk.Text(frame, wrap="word", font=("Segoe UI", 10), relief="flat", padx=14, pady=8,
                    background=self.root.cget("background"), cursor="arrow")
        bar = ttk.Scrollbar(frame, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=bar.set)
        t.grid(row=0, column=1, sticky="nsew")
        bar.grid(row=0, column=2, sticky="ns")
        t.tag_configure("title", font=("Segoe UI", 15, "bold"), spacing3=6)
        t.tag_configure("h", font=("Segoe UI", 11, "bold"), spacing1=12, spacing3=4)
        t.tag_configure("p", spacing3=4)
        t.tag_configure("li", lmargin1=8, lmargin2=24, spacing3=3)
        t.tag_configure("mono", font=("Consolas", 10), lmargin1=12, lmargin2=12, spacing3=2)
        t.tag_configure("cap", font=("Segoe UI", 9, "italic"), foreground="#5f5e5a", spacing3=4)
        t.tag_configure("ex", font=("Segoe UI", 10, "italic"), lmargin1=8, lmargin2=8, spacing1=4)
        self.book_text, self.book_keep = t, []
        tree.selection_set(handbook.ARTICLES[0]["id"])

    def _select_article(self, tree):
        sel = tree.selection()
        if sel and sel[0] in handbook.BY_ID:
            self._show_article(handbook.BY_ID[sel[0]])

    def _show_article(self, a):
        t = self.book_text
        t.configure(state="normal")
        t.delete("1.0", "end")
        for w in self.book_keep:  # кнопки и холст прошлой статьи
            if hasattr(w, "destroy"):
                w.destroy()
        self.book_keep = []
        t.insert("end", a["title"] + "\n", "title")
        t.insert("end", a["summary"] + "\n", "p")
        sketch = handbook_schematics.SCHEMATICS[a["id"]]()  # схема по ГОСТ, разложена вручную
        c = tk.Canvas(t, background="white", highlightthickness=0)
        schematic.draw_tk(c, sketch, mm=3.4 * self.k)
        self.book_keep.append(c)
        t.window_create("end", window=c, pady=6)
        has_opamp = any(kind == "OPAMP" for kind, _, _ in sketch.parts)
        t.insert("end", "\nСхема по ГОСТ" + ("; питание ОУ не показано" if has_opamp and a["id"] != "tl071" else "")
                 + "; номиналы - в примере расчёта ниже\n", "cap")
        t.insert("end", "Как работает\n", "h")
        for par in a["principle"]:
            t.insert("end", par + "\n", "p")
        t.insert("end", "Элементы\n", "h")
        for name, role in a["elements"]:
            t.insert("end", f"{name} - {role}\n", "li")
        t.insert("end", "Формулы\n", "h")
        for line in a["formulas"]:
            t.insert("end", line + "\n", "mono")
        for key, head in handbook.SECTIONS:
            if a.get(key):
                t.insert("end", head + "\n", "h")
                for item in a[key]:
                    t.insert("end", "•  " + item + "\n", "li")
        t.insert("end", "Пример расчёта\n", "h")
        t.insert("end", a["example"] + "\n", "p")
        data = handbook.curve(a)
        if data:
            c = tk.Canvas(t, width=int(600 * self.k), height=int(240 * self.k), background="white",
                          highlightthickness=1, highlightbackground="#d3d1c7")
            also = (data["f"], data["db_ideal"]) if "db_ideal" in data else None
            mark = a["check"].get("fc") or a["check"].get("f0")
            c.bind("<Configure>", lambda e, c=c: paint_plot(c, data["f"], data["db"], self.k, also, mark))
            self.book_keep.append(c)
            t.window_create("end", window=c, pady=4)
            t.insert("end", "\nАЧХ для этих номиналов, расчёт MC12" + ("; сплошная - с TL071, пунктир - идеальный ОУ"
                                                             if also else "") + "\n", "cap")
        t.insert("end", "Как запросить в программе\n", "h")
        for mode, prompt in a["prompts"]:
            t.insert("end", f"{MODE_NAMES[mode].capitalize()}: «{prompt}»  ", "ex")
            b = ttk.Button(t, text="В запрос", command=lambda s=prompt, m=mode: self.use_example(s, m))
            self.book_keep.append(b)
            t.window_create("end", window=b)
            t.insert("end", "\n")
        b = ttk.Button(t, text="Открыть пример в MC12", command=lambda: self._open_example(a))
        self.book_keep.append(b)
        t.insert("end", "\n")
        t.window_create("end", window=b, pady=6)
        t.insert("end", "\n")
        t.configure(state="disabled")
        t.yview_moveto(0)

    def _open_example(self, a):
        """Чертёж примера - копия в out/gui/handbook (MC12 пишет рядом служебные файлы), открыть в MC12."""
        try:
            exe = mc12run.find_exe()
        except FileNotFoundError as err:
            self.status.set(str(err))
            return
        dst = OUT / "handbook" / f"{a['id']}.cir"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((handbook.DIR / f"{a['id']}.cir").read_bytes())
        subprocess.Popen([exe, str(dst)], cwd=str(Path(exe).parent))
        self.status.set(f"Пример открыт в MC12: {dst}")


def paint_plot(c, f, db, k=1.0, also=None, mark=None):
    """АЧХ на холсте: сетка и подписи, отметка частоты mark (пунктир), основная кривая и пунктиром also - кривая
    с идеальным ОУ. Общая для главной вкладки и справочника."""
    c.delete("all")
    w, h = c.winfo_width(), c.winfo_height()
    if w < 10:  # холст ещё не показан - размер из параметров
        w, h = int(c.cget("width")), int(c.cget("height"))
    pad = tuple(int(v * k) for v in (42, 10, 10, 24))
    g = plot_geometry(f, db, w, h, pad, also=also)
    for x, lab in g["xgrid"]:
        c.create_line(x, pad[1], x, h - pad[3], fill="#eeede8")
        edge = x > w - pad[2] - 30 * k  # подпись у правого края - влево от линии, чтобы не обрезалась
        c.create_text(x, h - pad[3] / 2, text=lab, fill="#5f5e5a", font=("Segoe UI", 8),
                      anchor="e" if edge else "center")
    for y, lab in g["ygrid"]:
        c.create_line(pad[0], y, w - pad[2], y, fill="#eeede8")
        c.create_text(pad[0] - 6, y, text=lab, anchor="e", fill="#5f5e5a", font=("Segoe UI", 8))
    c.create_text(pad[0] + 4, pad[1] + 4, text="дБ", anchor="w", fill="#888780", font=("Segoe UI", 8))
    if mark and g.get("x"):
        x = g["x"](mark)
        c.create_line(x, pad[1], x, h - pad[3], fill="#888780", dash=(3, 3))
    for line in g["lines"]:
        c.create_line(*line, fill="#185fa5", width=2)
    for line in g["also_lines"]:  # идеальный ОУ - пунктиром поверх: там, где кривые совпадают, видны обе
        c.create_line(*line, fill="#5f5e5a", width=1, dash=(4, 3))


def dpi_aware():
    """Windows: окно в настоящих пикселях экрана - иначе при масштабе 125-150% шрифты размыты."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001 - не Windows или старая версия: окно просто будет мягче
        pass


def main():
    dpi_aware()
    root = tk.Tk()
    Window(root)
    root.mainloop()


if __name__ == "__main__":
    main()
