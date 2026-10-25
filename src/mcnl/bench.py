#!/usr/bin/env python3
"""Обвязка проверки (testbench): строки элементов класса A -> полный .ckt для MC12 (docs/eval_spec.md, раздел 4).

Запуск:  python bench.py элементы.txt [--fchar 1591] [--run]

Обвязка добавляет то, чего модель не пишет: заголовок, `AC 1` источнику с плюсом в IN, подсхему
OPAMP_IDEAL, анализ и .PRINT. Диапазон .AC задаётся характерной частотой ЭТАЛОНА, а не ответа
(правило честности): [f_хар/1000, f_хар*1000]; без частоты (делитель и т.п.) - 1 Гц..1 МГц.

OPAMP_IDEAL: E-источник с усилением 1e6, выход относительно узла 0 подсхемы. MC12 соединяет 0
внутри .SUBCKT с глобальной землёй (проверено 25.10.2026), опорный вывод не нужен.
"""
import sys
from pathlib import Path

try:
    from . import canon, mc12run
except ImportError:  # запуск как скрипта
    import canon
    import mc12run

OPAMP_DEF = (f".SUBCKT {canon.OPAMP_SUBCKT} PLUS MINUS OUT\n"
             f"E1 OUT 0 PLUS MINUS 1E6\n"
             f".ENDS {canon.OPAMP_SUBCKT}")
POINTS_PER_DECADE = 100
DEFAULT_RANGE = (1.0, 1e6)


def ac_range(f_char=None):
    if not f_char:
        return DEFAULT_RANGE
    return f_char / 1000, f_char * 1000


def _element_line(e):
    nodes = " ".join(e["nodes"])
    if e["type"] == "OPAMP":
        return f"{e['name']} {nodes} {canon.OPAMP_SUBCKT}"
    if e["type"] in "VI":
        ac = " AC 1" if e["nodes"][0] == "IN" else ""  # AC после DC допустимо: функции источника нет
        return f"{e['name']} {nodes} DC {e['value']}{ac}"
    return f"{e['name']} {nodes} {e['value']}"


def build(elems, f_char=None, title="mcnl testbench"):
    """Элементы (canon.parse_netlist / canonicalize) -> текст .ckt (LF; при записи - CRLF)."""
    fl, fh = ac_range(f_char)
    lines = [f"* {title}"]
    lines += [_element_line(e) for e in elems]
    if any(e["type"] == "OPAMP" for e in elems):
        lines.append(OPAMP_DEF)
    lines.append(f".AC DEC {POINTS_PER_DECADE} {canon.format_value(fl)} {canon.format_value(fh)}")
    lines.append(".PRINT AC VDB(OUT) VP(OUT)")
    lines.append(".END")
    return "\n".join(lines) + "\n"


def write(path, text):
    """Файлы для MC12 - ASCII с CRLF (CLAUDE.md); кириллица в заголовке недопустима."""
    Path(path).write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))


def run(cases, workdir, timeout=300):
    """cases: [(имя, элементы, f_char)] -> {имя: результат mc12run (анализ A)}. Один запуск MC12.
    workdir должен быть обычной папкой (MC12 не видит перенаправленные папки приложения)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, elems, f_char in cases:
        p = workdir / f"{name}.ckt"
        write(p, build(elems, f_char, title=name))
        paths[name] = p
    res = mc12run.run(list(paths.values()), timeout=timeout, analyses=["A"])
    by_path = {Path(r["ckt"]).resolve(): r for r in res}
    return {name: by_path[p.resolve()] for name, p in paths.items()}


if __name__ == "__main__":
    elems, errors = canon.parse_netlist(open(sys.argv[1], encoding="utf-8").read())
    if errors:
        sys.exit("\n".join(errors))
    f_char = float(sys.argv[sys.argv.index("--fchar") + 1]) if "--fchar" in sys.argv else None
    if "--run" in sys.argv:
        import json
        print(json.dumps(run([(Path(sys.argv[1]).stem, elems, f_char)], Path("out") / "bench"),
                         ensure_ascii=False, indent=1))
    else:
        print(build(elems, f_char), end="")
