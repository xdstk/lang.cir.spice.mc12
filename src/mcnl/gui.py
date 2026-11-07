#!/usr/bin/env python3
"""Окно программы (tkinter): запрос -> нетлист -> MC12 -> АЧХ и схема .CIR (docs/gui_spec.md).

Запуск:  ..\\venv-hf\\Scripts\\python.exe src/mcnl/gui.py      (Qwen3.5-4B нужен torch из venv-hf)
         python src/mcnl/gui.py                               (без torch доступен только baseline)

Окно - оболочка над app.run: своей логики схем здесь нет. Модель и MC12 работают в фоновом потоке,
результат приходит в главный поток через очередь. Рабочая папка - out/gui (обычная папка: MC12 не видит
перенаправленные папки приложения Claude). Настройки (пути к модели) - out/gui/settings.json.
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

from mcnl import app, canon, measure, mc12run, translators  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "gui"
DEFAULTS = {"base": "Qwen/Qwen3.5-4B", "adapter": str(ROOT / "out" / "q35_4b" / "adapter")}
QWEN, BASELINE = "Qwen3.5-4B (локально)", "baseline (проверка окна)"
KIND = {"LP": "ФНЧ", "HP": "ФВЧ", "BP": "полосовой", "BS": "режекторный", "flat": "без частотной зависимости"}
EXAMPLE = "ФНЧ Саллена-Ки на повторителе: R1 10 кОм, R2 10 кОм, C1 (ОС) 22 нФ, C2 (земля) 10 нФ; сигнал на вход"
# Вкладка «Правила» (docs/gui_spec.md, шаг 3): ("h", заголовок) | ("p", пункт) | ("ex", пример с кнопкой «В запрос»).
# Примеры проверены на Qwen3.5-4B и MC12 07.11.2026: схемы построены верно, отказы - с верным кодом.
RULES = [
    ("h", "Что умеет программа"),
    ("p", "Пассивные схемы из резисторов, конденсаторов и катушек: делители, RC, RL и RLC-фильтры, цепочки звеньев."),
    ("p", "Схемы на идеальных операционных усилителях: усилители, активные фильтры, каскады из 2-3 звеньев."),
    ("p", "Один источник сигнала между входом и землёй. Программа считает АЧХ в Micro-Cap 12 и строит чертёж схемы."),
    ("h", "Как писать запрос"),
    ("p", "Назовите каждый элемент и его номинал с единицами: 10 кОм, 4,7 кОм, 100 нФ, 22 мкФ, 3,3 мГн."),
    ("p", "Скажите, куда подключён каждый элемент: «от входа к точке 1», «с выхода на землю», «между точками 1 и 2»."),
    ("p", "Вместо соединений можно назвать типовую схему (инвертирующий усилитель, ФНЧ Саллена-Ки, фильтр Рауха) "
          "и дать номиналы по ролям: «резистор обратной связи 47 кОм»."),
    ("p", "Если после схемы добавлены элементы, укажите, откуда снимать выход: «выход за ним» или «выход остаётся на месте»."),
    ("p", "Питание ОУ указывать не нужно: усилитель идеальный."),
    ("p", "Примеры - кнопка «В запрос» переносит пример на вкладку «Схема»:"),
    ("ex", "R1 10 кОм от входа к точке 1, C1 100 нФ от точки 1 на землю, R2 22 кОм от точки 1 к выходу, "
           "C2 10 нФ с выхода на землю; сигнал на вход"),
    ("ex", "Инвертирующий усилитель на ОУ: входной резистор 4,7 кОм, резистор обратной связи 47 кОм; сигнал на вход"),
    ("ex", "RC ФНЧ: резистор 1 кОм, конденсатор 100 нФ; после выхода ещё 470 Ом последовательно, выход за ним"),
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
]


# ---------------------------------------------------------------- данные для окна (без tkinter, есть тесты)

def freq_label(f):
    """12.3 -> «12,3 Гц», 1530 -> «1,53 кГц»."""
    if f is None:
        return "-"
    for scale, unit in ((1e6, "МГц"), (1e3, "кГц"), (1, "Гц")):
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


def plot_geometry(f, db, width, height, pad=(42, 10, 10, 24), span_db=80):
    """Точки АЧХ на холсте (логарифмическая ось частоты) и линии сетки.
    pad = (слева, сверху, справа, снизу). -> {"lines": [[x, y, ...], ...], "xgrid": [(x, подпись)], "ygrid": [(y, подпись)]}
    Ниже нижней границы шкалы (span_db от максимума) кривая обрывается, а не прижимается к краю: иначе
    глубокий спад выглядел бы ложной полкой."""
    pts = [(fi, d) for fi, d in zip(f, db) if fi and fi > 0 and d is not None]
    if len(pts) < 2:
        return {"lines": [], "xgrid": [], "ygrid": []}
    l, t, r, b = pad
    w, h = width - l - r, height - t - b
    x0, x1 = math.log10(pts[0][0]), math.log10(pts[-1][0])
    top = math.ceil(max(d for _, d in pts) / 10) * 10 + 5
    bottom = max(min(d for _, d in pts), top - span_db)
    bottom = math.floor(bottom / 10) * 10

    def x(fi):
        return l + (math.log10(fi) - x0) / ((x1 - x0) or 1) * w

    def y(d):
        return t + (top - d) / ((top - bottom) or 1) * h
    lines, cur = [], []
    for fi, d in pts:
        if d < bottom:
            if len(cur) >= 4:
                lines.append(cur)
            cur = []
            continue
        cur += [round(x(fi), 1), round(y(d), 1)]
    if len(cur) >= 4:
        lines.append(cur)
    xgrid = [(round(x(10 ** e), 1), freq_label(10 ** e))
             for e in range(math.ceil(x0), math.floor(x1) + 1)]
    step = 20 if top - bottom > 40 else 10
    ygrid = [(round(y(v), 1), f"{v}") for v in range(int(bottom), int(top) + 1) if v % step == 0]
    return {"lines": lines, "xgrid": xgrid, "ygrid": ygrid, "x": x}


def notes_of(rep):
    """Строки замечаний: отказ, ошибки разбора, сообщения конвейера и замечания к схеме."""
    out = []
    if rep.get("refused"):
        out.append("Модель отказала: " + (rep["refused"].get("reason") or rep["refused"]["code"]))
    out += [f"Ошибка разбора: {e}" for e in rep.get("parse_errors") or []]
    out += rep.get("messages", [])
    out += [f"Замечание: {n}" for n in rep.get("notes", [])]
    return out


def load_settings():
    p = OUT / "settings.json"
    s = dict(DEFAULTS)
    if p.exists():
        s.update(json.loads(p.read_text(encoding="utf-8")))
    return s


# ---------------------------------------------------------------- окно

class Window:
    def __init__(self, root):
        self.root = root
        self.tasks, self.results = queue.Queue(), queue.Queue()
        self.translators = {}  # имя -> загруженный переводчик
        self.loading = set()
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
        self._select_translator()

    # --- раскладка
    def _build(self):
        mono = ("Consolas", 10)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=6, pady=(6, 0))
        main = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(main, text="Схема")
        rules = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(rules, text="Правила")
        self._build_rules(rules)
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=6)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(left, text="Запрос").pack(anchor="w")
        self.prompt = tk.Text(left, height=8, wrap="word", font=("Segoe UI", 10), undo=True)
        self.prompt.pack(fill="x", pady=(2, 8))
        self.prompt.insert("1.0", EXAMPLE)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text="Переводчик").pack(side="left")
        self.choice = tk.StringVar(value=QWEN)
        box = ttk.Combobox(row, textvariable=self.choice, values=[QWEN, BASELINE], state="readonly", width=30)
        box.pack(side="left", padx=8)
        box.bind("<<ComboboxSelected>>", lambda e: self._select_translator())
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
        ttk.Label(right, text="Нетлист (можно править и пересчитать)").pack(anchor="w")
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
        ttk.Label(right, text="АЧХ (расчёт MC12)").pack(anchor="w")
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

    def use_example(self, prompt):
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", prompt)
        self.tabs.select(0)
        self.status.set("Пример вставлен в запрос: нажмите «Построить»")

    # --- действия
    def _select_translator(self):
        name = self.choice.get()
        if name not in self.translators and name not in self.loading:
            self.loading.add(name)
            self.status.set(f"{name}: загрузка…" + (" (модель грузится около 30 с)" if name == QWEN else ""))
            threading.Thread(target=self._load_task, args=(name,), daemon=True).start()  # не ждёт запросов в очереди

    def _load_task(self, name):
        try:
            self.results.put(("loaded", name, self._load(name)))
        except Exception as err:  # noqa: BLE001
            self.results.put(("error", name, f"{type(err).__name__}: {err}"))

    def build(self):
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self.status.set("Введите запрос")
            return
        name = self.choice.get()
        wait = "" if name in self.translators else f" ({name} ещё загружается, запрос выполнится после загрузки)"
        self.status.set("Перевод и расчёт в MC12…" + wait)
        self.tasks.put(("build", name, prompt))

    def recalc(self):
        text = self.netlist.get("1.0", "end").strip()
        if not text:
            self.status.set("Нетлист пуст")
            return
        elems, errors = canon.parse_netlist(text)
        if errors:
            self._show_notes([f"Ошибка разбора: {e}" for e in errors])
            self.status.set("Нетлист не разобран")
            return
        self.status.set("Расчёт в MC12…")
        self.tasks.put(("recalc", None, text))

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
                    t0 = time.time()
                    rep = app.run(payload, tr, outdir=OUT, name=self._next_name())
                    rep["seconds"] = time.time() - t0
                    self.results.put(("done", name, rep))
                elif kind == "recalc":
                    t0 = time.time()
                    rep = app.run(netlist=payload, outdir=OUT, name=self._next_name())
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
            tr = translators.load("hf", base=self.settings["base"], adapter=self.settings["adapter"])
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
                    self.status.set(f"{name}: готов, загрузка {data:.0f} с" + self._gpu())
                elif kind == "error":
                    self.loading.discard(name)
                    self.status.set(f"Ошибка: {data}")
                else:
                    self._show(data)
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
        self.netlist.delete("1.0", "end")
        self.netlist.insert("1.0", (rep.get("netlist") or rep.get("output") or "").strip())
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
        out = ((self.rep or {}).get("run") or {}).get("output")
        if not out or not Path(out).exists():
            c.create_text(c.winfo_width() / 2, c.winfo_height() / 2, text="Здесь появится АЧХ", fill="#888780")
            return
        f, db, _ = measure.read_curve(out)
        w, h = c.winfo_width(), c.winfo_height()
        pad = tuple(int(v * self.k) for v in (42, 10, 10, 24))
        g = plot_geometry(f, db, w, h, pad)
        for x, lab in g["xgrid"]:
            c.create_line(x, pad[1], x, h - pad[3], fill="#eeede8")
            edge = x > w - pad[2] - 30 * self.k  # подпись у правого края - влево от линии, чтобы не обрезалась
            c.create_text(x, h - pad[3] / 2, text=lab, fill="#5f5e5a", font=("Segoe UI", 8),
                          anchor="e" if edge else "center")
        for y, lab in g["ygrid"]:
            c.create_line(pad[0], y, w - pad[2], y, fill="#eeede8")
            c.create_text(pad[0] - 6, y, text=lab, anchor="e", fill="#5f5e5a", font=("Segoe UI", 8))
        c.create_text(pad[0] + 4, pad[1] + 4, text="дБ", anchor="w", fill="#888780", font=("Segoe UI", 8))
        m = (self.rep or {}).get("measured") or {}
        mark = m.get("fc") or (m.get("f0") if m.get("type") in ("BP", "BS") else None)
        if mark and g.get("x"):
            x = g["x"](mark)
            c.create_line(x, pad[1], x, h - pad[3], fill="#888780", dash=(3, 3))
        for line in g["lines"]:
            c.create_line(*line, fill="#185fa5", width=2)


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
