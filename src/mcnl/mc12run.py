#!/usr/bin/env python3
"""Пакетный запуск SPICE-нетлистов (.ckt) в Micro-Cap 12 и сбор результатов.

Запуск:  python mc12run.py файл.ckt [...] [--timeout 120] [--exe путь]

Поведение MC12 12.2.0.5 (64 и 32 bit одинаково), подтверждено проверкой 25.10.2026 (UG12 с. 43-46 - синтаксис):
  * `MC12 /quiet @пакет.bat` выполняет строки `полный_путь.ckt /A` и закрывается сам (~1 с на пакет);
    код выхода всегда 0, в том числе при ошибках в схемах;
  * /quiet принимается только глобально, перед @пакет; в строке схемы это ошибка командной строки;
  * ошибки и предупреждения не вызывают окон: они пишутся в <пакет>.DOC рядом с .bat, пакет идёт дальше;
  * числовой вывод .PRINT пишется в ИМЯ.ANO / ИМЯ.TNO / ИМЯ.DNO рядом с .ckt (/NOF без полного пути
    пишет в папку DATA самого MC12, поэтому здесь не используется);
  * MC12 не видит перенаправленные папки приложения Claude (AppData\\Roaming в пакете MSIX):
    .ckt должны лежать в обычной папке, например в out/ проекта.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath

ANALYSES = {"A": ".ANO", "T": ".TNO", "D": ".DNO"}
DEFAULT_EXES = [r"C:\MC12\Micro-Cap 12.exe", r"C:\MC12\mc12.exe"]  # 64 бит, затем 32 бит


def find_exe(exe=None):
    for cand in [exe, os.environ.get("MC12_EXE"), *DEFAULT_EXES]:
        if cand and Path(cand).is_file():
            return str(cand)
    raise FileNotFoundError("Micro-Cap 12 не найден: задайте --exe или переменную MC12_EXE")


def analyses_in(ckt_text):
    """Анализы, заданные в нетлисте: .AC -> A, .TRAN -> T, .DC -> D."""
    found = []
    for line in ckt_text.upper().splitlines():
        for card, a in ((".AC", "A"), (".TRAN", "T"), (".DC", "D")):
            if re.match(re.escape(card) + r"\b", line.strip()) and a not in found:
                found.append(a)
    return found


def parse_doc(text):
    """Журнал пакета (.DOC) -> {(ИМЯ.CKT, анализ или None): {"ok": bool, "messages": [...]}}.

    Строка успешного прогона: `RC.CKT    /A      2  509 ...`;
    ошибка/предупреждение: `BAD.CKT Error Missing model ...` и строки с табуляцией (Part:, File:).
    Ошибка загрузки относится ко всей схеме (анализ None)."""
    res, last = {}, None
    for line in text.splitlines():
        m = re.match(r"^(\S+\.(?:CKT|CIR))\s+/(\w+)\s+\d", line, re.I)
        if m:
            last = res.setdefault((m.group(1).upper(), m.group(2).upper()), {"ok": True, "messages": []})
            continue
        m = re.match(r"^(\S+\.(?:CKT|CIR))\s+(Error|Warning)\s+(.*)$", line, re.I)
        if m:
            key = (m.group(1).upper(), None)
            entry = res.setdefault(key, {"ok": True, "messages": []})
            entry["messages"].append(f"{m.group(2)}: {m.group(3).strip()}")
            if m.group(2).lower() == "error":
                entry["ok"] = False
            last = entry
            continue
        m = re.match(r"^Error (.*?)'([^']+\.(?:CKT|CIR))'", line, re.I)
        if m:  # «Error Bad format in loading file '...\X.CIR'.» - имя схемы внутри строки, а не в начале
            last = res.setdefault((Path(m.group(2)).name.upper(), None), {"ok": True, "messages": []})
            last["messages"].append(line.strip())
            last["ok"] = False
            continue
        m = re.match(r"^Command Line\s+(\S+\.(?:CKT|CIR))", line, re.I)
        if m:
            last = res.setdefault((PureWindowsPath(m.group(1)).name.upper(), None), {"ok": True, "messages": []})
            continue
        if line.startswith("Error") and last is not None:
            last["messages"].append(line.strip())
            last["ok"] = False
        elif line.startswith("\t") and last is not None:
            last["messages"].append(line.strip())
    return res


def run(ckt_paths, timeout=120, exe=None, analyses=None):
    """Прогоняет нетлисты одним запуском MC12.

    analyses: None - взять из нетлиста (.AC/.TRAN/.DC), иначе список букв "A"/"T"/"D" для всех.
    Возвращает список {"ckt", "analysis", "ok", "output", "messages"} по каждой паре схема-анализ."""
    exe = find_exe(exe)
    paths = [Path(p).resolve() for p in ckt_paths]
    names = [p.name.upper() for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("в одном пакете имена файлов должны различаться (журнал MC12 пишет только имя)")
    jobs = []
    for p in paths:
        for a in (analyses or analyses_in(p.read_text(encoding="cp1251", errors="replace"))):
            out = p.with_suffix(ANALYSES[a])
            if out.exists():
                out.unlink()  # иначе старый вывод сойдёт за результат упавшего прогона
            jobs.append((p, a, out))
    results = []
    if not jobs:
        return [{"ckt": str(p), "analysis": None, "ok": False, "output": None,
                 "messages": ["в нетлисте нет .AC/.TRAN/.DC"]} for p in paths]
    workdir = paths[0].parent
    bat = workdir / f"mc12run_{os.getpid()}_{int(time.time() * 1000)}.bat"
    bat.write_bytes("".join(f"{p} /{a}\r\n" for p, a, _ in jobs).encode("ascii"))
    doc = bat.with_suffix(".DOC")
    timed_out = False
    proc = subprocess.Popen([exe, "/quiet", f"@{bat}"], cwd=str(Path(exe).parent))
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        timed_out = True
    log = parse_doc(doc.read_text(encoding="cp1251", errors="replace")) if doc.exists() else {}
    for p, a, out in jobs:
        load = log.get((p.name.upper(), None), {"ok": True, "messages": []})
        entry = log.get((p.name.upper(), a))
        msgs = list(load["messages"]) + (entry["messages"] if entry else [])
        ok = load["ok"] and entry is not None and out.exists()
        if timed_out and not ok:
            msgs.append(f"MC12 не завершился за {timeout} с, процесс снят")
        elif entry is None and load["ok"]:
            msgs.append("прогона нет в журнале MC12")
        results.append({"ckt": str(p), "analysis": a, "ok": ok,
                        "output": str(out) if out.exists() else None, "messages": msgs})
    return results


def main(argv):
    timeout = int(argv[argv.index("--timeout") + 1]) if "--timeout" in argv else 120
    exe = argv[argv.index("--exe") + 1] if "--exe" in argv else None
    skip = {argv.index(k) + i for k in ("--timeout", "--exe") if k in argv for i in (0, 1)}
    files = [a for i, a in enumerate(argv[1:], 1) if i not in skip]
    print(json.dumps(run(files, timeout=timeout, exe=exe), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
