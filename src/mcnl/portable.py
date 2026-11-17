#!/usr/bin/env python3
"""Переносная версия программы для ПК без Python и видеокарты (17.11.2026, docs/program.md, «Переносная версия»).

Сборка:  python src/mcnl/portable.py [--quant Q4_K_M] [--out out/portable/MCNL] [--python папка_Python] [--mc12 C:\\MC12]

Папка (копируется куда угодно, путь - латиницей):
  start.bat      - запуск окна (pythonw без консоли; -E -s: Python владельца ПК и его переменные не мешают)
  README.txt     - что это и что попробовать
  python/        - копия установленного Python 3.12 без site-packages, тестов и документации (tkinter есть)
  app/           - корень программы (ROOT модулей):
    src, data    - код и нужные ему данные (справочник, шаблоны .CIR, словарь ролей)
    out/         - рабочие файлы окна (создаются при запуске)
    llama/       - llama-server.exe и его библиотеки (llama.cpp, процессор; вариант под процессор выбирается сам)
    llama-vk/    - то же с Vulkan: берётся, только если есть дискретная видеокарта (llama_server.pick_device)
    models/      - модели режимов в GGUF (база Qwen3.5-4B с влитым адаптером) и models.json
    MC12/        - Micro-Cap 12 без mcap.dat: без него MC12 ищет библиотеки в своей папке (проверено 17.11.2026 -
                   с mcap.dat копия читала бы C:\\MC12\\LIBRARY из настроек)
Программа находит llama-server и модели через translators.llama_server.find_runtime, MC12 - через
mc12run.DEFAULT_EXES (<корень>/MC12 раньше C:\\MC12). Torch в переносной версии нет: в окне - «процессор» и baseline.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ["handbook", "cir_templates.cir", "roles_ru.json"]  # всё, что читает программа (gui, app, spice2cir, baseline)
PY_SKIP = {"site-packages", "test", "idlelib", "ensurepip", "venv", "__pycache__", "lib2to3", "pydoc_data", "turtledemo"}
PY_TOP_SKIP = {"Doc", "include", "libs", "Scripts", "Tools", "NEWS.txt"}
MC12_SKIP = {"mcap.dat", "Logs", "clipboard.dat"}  # настройки с путями C:\MC12, журналы, буфер обмена владельца
LLAMA_KEEP = ("llama-server.exe",)  # и все .dll рядом
# для README: замеры на i5-10300H без видеокарты (docs/program.md, «Переносная версия»)
RAM_GB = "5"
ANSWER_TIME = "10-30 с на современном ПК и 1-3 мин на старом (без AVX)"

START_BAT = "\r\n".join([
    "@echo off",
    "rem MCNL: zapusk okna programmy (Python i model - v etoj papke)",
    'cd /d "%~dp0app"',
    'start "" "%~dp0python\\pythonw.exe" -E -s src\\mcnl\\gui.py',
    ""])


def _copytree(src, dst, skip=frozenset(), top_skip=frozenset()):
    def ignore(folder, names):
        here = Path(folder)
        out = {n for n in names if n in skip or n.endswith((".pyc", ".pyo"))}
        if here == Path(src):
            out |= {n for n in names if n in top_skip}
        return out
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def build(out, quant="Q4_K_M", python=None, mc12=r"C:\MC12", gguf=None, server=None, server_gpu=None, log=print):
    out = Path(out)
    python = Path(python or Path(sys.base_prefix))
    gguf = Path(gguf or ROOT / "out" / "gguf")
    server = Path(server or ROOT / "out" / "tools" / "llama-bin")
    server_gpu = Path(server_gpu or ROOT / "out" / "tools" / "llama-vk")
    for need in (python / "pythonw.exe", server / "llama-server.exe", gguf / f"m1-{quant}.gguf",
                 Path(mc12) / "Micro-Cap 12.exe"):
        if not need.is_file():
            raise FileNotFoundError(f"нет {need}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log("Python:", python)
    _copytree(python, out / "python", PY_SKIP, PY_TOP_SKIP)
    log("программа")
    _copytree(ROOT / "src", out / "app" / "src", {"__pycache__"})
    (out / "app" / "data").mkdir(parents=True)
    for name in DATA:
        src = ROOT / "data" / name
        if src.is_dir():
            shutil.copytree(src, out / "app" / "data" / name)
        else:
            shutil.copy2(src, out / "app" / "data" / name)
    log("llama-server")
    for src_dir, dst_name in ((server, "llama"), (server_gpu, "llama-vk")):
        if not (src_dir / "llama-server.exe").is_file():
            continue  # сборки с Vulkan может не быть: тогда только процессор
        lla = out / "app" / dst_name
        lla.mkdir()
        for f in src_dir.iterdir():
            if f.suffix.lower() == ".dll" or f.name in LLAMA_KEEP or f.name.startswith("LICENSE"):
                shutil.copy2(f, lla / f.name)
    log("модели", quant)
    mod = out / "app" / "models"
    mod.mkdir()
    spec = json.loads((gguf / "models.json").read_text(encoding="utf-8")) if (gguf / "models.json").is_file() else {}
    index = {}
    for mode, trained in (("mode1", "v2"), ("mode2", "v3")):
        src = gguf / f"m{mode[-1]}-{quant}.gguf"
        if src.is_file():
            shutil.copy2(src, mod / src.name)
            index[mode] = {"file": src.name, "trained_on": spec.get(mode, {}).get("trained_on", trained),
                           "adapter": spec.get(mode, {}).get("adapter", ""), "quant": quant}
    (mod / "models.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    log("MC12")
    _copytree(mc12, out / "app" / "MC12", MC12_SKIP | {n.name for n in Path(mc12).iterdir() if n.name.startswith("BACKUP-")})
    (out / "start.bat").write_bytes(START_BAT.encode("ascii"))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    readme = ROOT / "docs" / "portable_readme.txt"
    if readme.is_file():  # UTF-8 с BOM и CRLF - для Блокнота любой версии
        text = (readme.read_text(encoding="utf-8").replace("{DATE}", time.strftime("%d.%m.%Y"))
                .replace("{DISK}", f"{size / 2 ** 30 + 0.5:.0f}").replace("{RAM}", RAM_GB).replace("{TIME}", ANSWER_TIME))
        (out / "README.txt").write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
    log(f"готово: {out} ({size / 2 ** 30:.2f} ГБ)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "out" / "portable" / "MCNL"))
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--python", help="папка установленного Python 3.12 (по умолчанию - тот, что запускает сборку)")
    ap.add_argument("--mc12", default=r"C:\MC12")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    build(a.out, a.quant, a.python, a.mc12)


if __name__ == "__main__":
    main()
