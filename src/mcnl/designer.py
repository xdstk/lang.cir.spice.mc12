#!/usr/bin/env python3
"""Режим дизайнера: спецификация фильтра -> схема от дизайнера MC12 (docs/designer_mode.md).

Модель извлекает параметры mcnl.flt/1 и ничего не рассчитывает; номиналы даёт дизайнер. Настройки пишутся в
Active.flt, MC12 открывает диалог и нажимает OK сообщениями Windows (GUI-доступ только через Win32, без
скриншотов), готовый макрос забирается из LIBRARY и проверяется обвязкой против спецификации.

Запуск:  python src/mcnl/designer.py --spec '{"type": "LP", "response": "Butterworth", "order": 4, "fc": 1000}' --out out/designer
"""
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import bench, canon, cir2spice, measure, mna  # noqa: E402

MC12_DIR = Path(r"C:\MC12")
ACTIVE_FLT = MC12_DIR / "Active.flt"
LIBRARY = MC12_DIR / "LIBRARY"
SCHEMA = "mcnl.flt/1"
TYPES = {"LP": "Low-Pass", "HP": "High-Pass", "BP": "Bandpass", "BS": "Notch"}
# Bessel в MC12 - только тип Delay (RM12, с. 996). Elliptic не взят: в режиме 2 диалога MC12 молча заменяет его на Butterworth
RESPONSES = ("Butterworth", "Chebyshev", "Inverse Chebyshev")
STYLES = {"Sallen-Key": 0, "MFB": 1, "Tow-Thomas": 2, "Fleischer-Tow": 3, "KHN": 4, "Acker-Mossberg": 5, "DABP": 6, "Tow-Thomas 2": 7}
ORDER_MAX = 10
FREQ_RANGE = (0.01, 100e6)  # Гц; вне диапазона - ошибка спецификации
LEVEL_TOL = 1.5  # дБ: у Чебышёва fc по -3 дБ не равна частоте среза спецификации, поэтому допуск на частоту шире
FREQ_TOL = 0.25


# ---------------------------------------------------------------- спецификация
def validate(spec):
    """-> список ошибок спецификации на русском; пустой - годится."""
    errs = []
    if not isinstance(spec, dict):
        return ["спецификация не объект"]
    if spec.get("type") not in TYPES:
        errs.append(f"тип {spec.get('type')!r} не из {list(TYPES)}")
    if spec.get("response") not in RESPONSES:
        errs.append(f"отклик {spec.get('response')!r} не из {list(RESPONSES)}")
    if not isinstance(spec.get("order"), int) or not 1 <= spec["order"] <= ORDER_MAX:
        errs.append(f"порядок должен быть целым от 1 до {ORDER_MAX}")
    key = "fc" if spec.get("type") in ("LP", "HP") else "f0"
    f = spec.get(key)
    if not isinstance(f, (int, float)) or not FREQ_RANGE[0] <= f <= FREQ_RANGE[1]:
        errs.append(f"частота {key} вне диапазона")
    if key == "f0" and not (isinstance(spec.get("Q"), (int, float)) and spec["Q"] > 0):
        errs.append("для полосового и режекторного нужна добротность Q > 0")
    if spec.get("style") is not None and spec["style"] not in STYLES:
        errs.append(f"схема {spec['style']!r} не из {list(STYLES)}")
    return errs


def from_json(text):
    """Ответ модели -> (спецификация, ошибки). Отказ обрабатывает программа до этого вызова."""
    try:
        spec = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text or ""))
    except json.JSONDecodeError as e:
        return None, [f"JSON: {e}"]
    errs = validate(spec)
    return (None, errs) if errs else (spec, [])


# ---------------------------------------------------------------- Active.flt
def _sub(text, pattern, value):
    new, n = re.subn(pattern, lambda _m: value, text, count=1, flags=re.M)
    if n != 1:
        raise ValueError(f"в Active.flt нет поля: {pattern}")
    return new


def gain_db(spec):
    """Усиление в дБ: `gain_db` или `gain` в разах (люди пишут и так, и так); пересчёт делает программа, не модель."""
    if spec.get("gain"):
        return round(20 * math.log10(spec["gain"]), 3)
    return spec.get("gain_db") or 0


def flt_text(spec, text):
    """Подставить спецификацию в текст Active.flt (режим 2 диалога; CRLF файла сохраняются)."""
    kind, resp = TYPES[spec["type"]], spec["response"]
    for field, value in (("Type", kind), ("Response", resp), ("Save As", "Macro"), ("Same Circuit", "Yes"), ("ViewCir", "No")):
        text = _sub(text, rf"^{field}=.*?(?=\r?$)", f"{field}={value}")
    text = _sub(text, rf"^{kind} mode=.*?(?=\r?$)", f"{kind} mode=2")
    if spec.get("style"):
        text = _sub(text, rf"^{kind} {resp} Circuits=.*?(?=\r?$)", f"{kind} {resp} Circuits=" + f"{STYLES[spec['style']]}," * 60)
    gain, ripple = gain_db(spec), spec.get("ripple_db")
    if spec["type"] in ("LP", "HP"):
        block = {"Gain": gain, "Cutoff Frequency": spec["fc"], "Order": spec["order"]}
        section = r"(?<=\[Low-Pass High-Pass Mode 2\]\r\n)"
    else:
        block = {"Gain": gain, "Center Frequency": spec["f0"], "Order": spec["order"], "Q": spec["Q"]}
        section = r"(?<=\[Band Pass Notch Mode 2\]\r\n)"
    if ripple:
        block["Ripple"] = ripple
    head = re.search(section + r"(?:.+\r\n)+", text)
    if not head:
        raise ValueError("в Active.flt нет секции режима 2")
    body = head.group(0)
    for k, v in block.items():
        body, n = re.subn(rf"^{re.escape(k)}=.*$", lambda _m, k=k, v=v: f"{k}={v}\r", body, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"в секции режима 2 нет поля {k}")
    return text[:head.start()] + body + text[head.end():]


def write_flt(spec, path=ACTIVE_FLT):
    """Записать настройки; файл - байты latin-1, концы строк не трогаем (sed из Git Bash их портит)."""
    path = Path(path)
    data = flt_text(spec, path.read_bytes().decode("latin-1"))
    path.write_bytes(data.encode("latin-1"))


# ---------------------------------------------------------------- MC12 без скриншотов
def _user32():
    import ctypes
    import ctypes.wintypes as w
    u = ctypes.WinDLL("user32")
    u.SendMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
    u.SendMessageW.restype = ctypes.c_ssize_t
    u.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
    u.GetMenu.argtypes = [w.HWND]
    u.GetMenu.restype = w.HMENU
    u.GetSubMenu.argtypes = [w.HMENU, ctypes.c_int]
    u.GetSubMenu.restype = w.HMENU
    u.GetMenuItemCount.argtypes = [w.HMENU]
    u.GetMenuItemID.argtypes = [w.HMENU, ctypes.c_int]
    u.GetMenuItemID.restype = w.UINT
    u.GetMenuStringW.argtypes = [w.HMENU, w.UINT, w.LPWSTR, ctypes.c_int, w.UINT]
    return u, ctypes, w


class Mc12Window:
    """Отдельный экземпляр MC12: окна пользователя не трогаем. Окна ищем только в своём процессе."""
    WM_COMMAND, IDOK, IDYES, MF_BYPOSITION = 0x0111, 1, 6, 0x400

    def __init__(self, exe=MC12_DIR / "mc12.exe", timeout=30):
        self.u, self.ct, self.w = _user32()
        self.popups = []  # тексты всплывших окон: отчёт о том, что MC12 спросил сам
        self.proc = subprocess.Popen([str(exe)], cwd=str(MC12_DIR))
        self.main = self._wait("Micro-Cap 12", timeout)
        end = time.time() + timeout
        while not self.u.GetMenu(self.main) and time.time() < end:
            self._dismiss()
            time.sleep(0.25)
        time.sleep(2)  # окно появляется раньше, чем готово меню
        self._dismiss()

    def _title(self, h):
        b = self.ct.create_unicode_buffer(256)
        self.u.GetWindowTextW(h, b, 256)
        return b.value

    def _windows(self):
        found, pid = [], self.proc.pid
        proc = self.ct.WINFUNCTYPE(self.w.BOOL, self.w.HWND, self.w.LPARAM)

        def cb(h, _):
            owner = self.w.DWORD()
            self.u.GetWindowThreadProcessId(h, self.ct.byref(owner))
            if owner.value == pid and self.u.IsWindowVisible(h):
                found.append(h)
            return True
        self.u.EnumWindows(proc(cb), 0)
        return found

    def _find(self, text):
        return next((h for h in self._windows() if text in self._title(h)), None)

    def _wait(self, text, timeout):
        end = time.time() + timeout
        while time.time() < end:
            h = self._find(text)
            if h:
                return h
            self._dismiss()
            time.sleep(0.25)
        raise TimeoutError(f"нет окна «{text}»")

    def _texts(self, h):
        out = []
        proc = self.ct.WINFUNCTYPE(self.w.BOOL, self.w.HWND, self.w.LPARAM)

        def cb(c, _):
            t = self._title(c)
            if t:
                out.append(t)
            return True
        self.u.EnumChildWindows(h, proc(cb), 0)
        return out

    def _dismiss(self):
        """Закрыть всплывшее окно MC12 (ошибка, «файл существует»): иначе дизайнер ждёт ответа бесконечно."""
        for h in self._windows():
            t = self._title(h)
            if h == getattr(self, "main", None) or "Micro-Cap 12" in t or "Active Filter Designer" in t:
                continue
            b = self.ct.create_unicode_buffer(32)
            self.u.GetClassNameW(h, b, 32)
            texts = self._texts(h)
            # настоящий вопрос - стандартный диалог с кнопкой; панели и окошко «Generating Standard Values» не трогаем
            if b.value != "#32770" or not any(x in ("&Yes", "OK", "ОК", "&Да") for x in texts):
                continue
            self.popups.append(" | ".join([t] + texts)[:300])
            self.u.PostMessageW(h, self.WM_COMMAND, self.IDYES, 0)
            self.u.PostMessageW(h, self.WM_COMMAND, self.IDOK, 0)

    def _command(self, menu_name, item):
        """Код команды пункта меню: номер читается из самого меню, а не задаётся в программе."""
        menu = self.u.GetMenu(self.main)
        for i in range(self.u.GetMenuItemCount(menu)):
            b = self.ct.create_unicode_buffer(64)
            self.u.GetMenuStringW(menu, i, b, 64, self.MF_BYPOSITION)
            if menu_name in b.value:
                sub = self.u.GetSubMenu(menu, i)
                for j in range(self.u.GetMenuItemCount(sub)):
                    self.u.GetMenuStringW(sub, j, b, 64, self.MF_BYPOSITION)
                    if item in b.value:
                        return self.u.GetMenuItemID(sub, j)
        raise LookupError(f"нет пункта {menu_name} > {item}")

    def design(self, timeout=20):
        """Открыть диалог дизайнера и нажать OK; настройки уже в Active.flt."""
        self.u.PostMessageW(self.main, self.WM_COMMAND, self._command("Design", "Active"), 0)
        dlg = self._wait("Active Filter Designer", timeout)
        time.sleep(0.5)
        self.u.PostMessageW(dlg, self.WM_COMMAND, self.IDOK, 0)
        end = time.time() + timeout
        while self._find("Active Filter Designer") and time.time() < end:
            self._dismiss()
            time.sleep(0.25)
        self._dismiss()

    def close(self):
        self.proc.kill()


# ---------------------------------------------------------------- макрос -> схема класса
def macro_title(raw):
    m = re.search(r'^Text="((?:Low|High|Band|Notch)[^"]*)"', raw, re.M)
    return m.group(1) if m else None


def macro_netlist(raw):
    """Текст .MAC -> нетлист класса: V1 IN 0, идеальные ОУ, питание и блоки .define убраны."""
    text = cir2spice.convert(raw, title="designer")[0]
    lines = ["V1 IN 0 DC 0"]
    for l in text.splitlines():
        t = l.split()
        if not t or t[0][0] in "*.+1(":
            if t and t[0].upper() == ".SUBCKT":
                break
            continue
        if t[0][0] in "RCL" and len(t) >= 4:
            lines.append(" ".join(t[:3] + [t[-1]]))
        elif t[0][0] == "X" and len(t) >= 7:
            lines.append(f"{t[0]} {t[1]} {t[2]} {t[4]} OPAMP_IDEAL")
    net = re.sub(r"\bIn\b", "IN", "\n".join(lines) + "\n")
    return re.sub(r"\bOut\b", "OUT", net)


def check(spec, netlist):
    """Сверка схемы со спецификацией узловым расчётом -> (годится, измерения, замечания)."""
    elems = canon.canonicalize(canon.parse_netlist(netlist)[0])
    f = spec.get("fc") or spec.get("f0")
    lo, hi = bench.ac_range(f)
    m = measure.measure(*mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE))
    notes = []
    if m["type"] != spec["type"]:
        notes.append(f"тип АЧХ {m['type']}, запрошен {spec['type']}")
    got = m.get("fc") if spec["type"] in ("LP", "HP") else m.get("f0")
    if not got or abs(got / f - 1) > FREQ_TOL:
        notes.append(f"частота {got}, запрошена {f}")
    if abs(m["k_pass_db"] - gain_db(spec)) > LEVEL_TOL:
        notes.append(f"усиление {m['k_pass_db']:.2f} дБ, запрошено {gain_db(spec)}")
    return not notes, m, notes


def _stash(spec, outdir):
    """Убрать макросы с тем же именем на время работы: иначе MC12 спрашивает о замене, а файл пользователя пропал бы."""
    letter = {"LP": "L", "HP": "H", "BP": "B", "BS": "N"}[spec["type"]]
    f = spec.get("fc") or spec.get("f0")
    moved = []
    for p in LIBRARY.glob(f"{letter}_*_*_{f:g}_{spec['order']}.[mM][aA][cC]"):
        keep = Path(outdir) / ("stash_" + p.name)
        shutil.move(str(p), str(keep))
        moved.append((keep, p))
    return moved


def _unstash(moved):
    for keep, original in moved:
        shutil.move(str(keep), str(original))


def run(spec, outdir):
    """Спецификация -> {'macro', 'netlist', 'ok', 'measured', 'notes', 'title', 'popups'}. На каждую схему свой экземпляр
    MC12: каталог Filter.cmp возвращается к исходному, и работающий MC12 на это отвечает вопросом о перезаписи."""
    errs = validate(spec)
    if errs:
        raise ValueError("; ".join(errs))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = {p: p.read_bytes() for p in (ACTIVE_FLT, MC12_DIR / "Filter.cmp", MC12_DIR / "Filter.bin")}
    win = Mc12Window()
    start = time.time() - 1
    seen = len(win.popups)
    stash = _stash(spec, outdir)
    try:
        write_flt(spec)
        win.design()
        end = time.time() + 30
        made = []
        while time.time() < end and not made:
            win._dismiss()
            made = [p for p in LIBRARY.iterdir() if p.suffix.upper() == ".MAC" and p.stat().st_mtime >= start]
            time.sleep(0.25)
        time.sleep(0.5)  # файл появляется до окончания записи
    finally:
        win.close()
        for p, data in saved.items():  # каталог компонентов и настройки дизайнера - как были
            p.write_bytes(data)
    popups = win.popups[seen:]
    if not made:
        _unstash(stash)
        raise RuntimeError("дизайнер не создал макрос" + (": " + popups[0] if popups else ""))
    macro = max(made, key=lambda p: p.stat().st_mtime)
    target = outdir / macro.name
    raw = macro.read_text(encoding="cp1251", errors="replace")
    shutil.move(str(macro), str(target))  # в LIBRARY ничего не оставляем
    inx = macro.with_name(macro.stem + "_MAC.inx")
    if inx.exists():
        inx.unlink()
    _unstash(stash)
    net = canon.to_netlist(canon.canonicalize(canon.parse_netlist(macro_netlist(raw))[0]))
    ok, measured, notes = check(spec, net)
    title = macro_title(raw) or ""
    if spec["response"] not in title or (spec.get("style") and spec["style"] not in title):
        ok = False  # дизайнер мог молча подменить отклик или схему: подпись макроса говорит, что он построил на деле
        notes.append(f"дизайнер построил «{title}»")
    return {"macro": str(target), "title": title, "netlist": net, "ok": ok, "measured": measured, "notes": notes, "popups": popups}


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    spec, errs = from_json(argv[argv.index("--spec") + 1])
    if errs:
        sys.exit("; ".join(errs))
    out = argv[argv.index("--out") + 1] if "--out" in argv else "out/designer"
    print(json.dumps(run(spec, out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
