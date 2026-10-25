#!/usr/bin/env python3
"""Демонстрация без интернета: запрос на русском -> схема в MC12 (baseline на правилах).

Запуск:  python src/mcnl/demo.py --out C:/Users/.../Desktop/MC12_demo/schemy
         python src/mcnl/demo.py --examples C:/Users/.../Desktop/MC12_demo/examples   (готовые схемы)

Всё работает локально: baseline, расчёт в MC12, построение .CIR. Папки - латиницей (MC12 - старая
программа, кириллицу в путях к схемам лучше не допускать).
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import app, translators  # noqa: E402

EXAMPLES = [  # (имя файла, запрос) - разные классы схем, запросы в стиле, который понимает baseline
    ("01_rc_lpf", "RC ФНЧ: R 1 кОм, C 100 нФ; сигнал на вх"),
    ("02_rc_hpf", "RC-фильтр высоких частот: конденсатор 220 нФ, резистор 10 кОм"),
    ("03_series_rlc", "последовательный колебательный контур: катушка 10 мГн, конденсатор 0,1 мкФ, резистор 100 Ом"),
    ("04_inverting_amp", "инвертирующий усилитель: Rвх 4,7к, Rос 47к; на вх 1 В"),
    ("05_noninverting_amp", "неинвертирующий усилитель: Rз 10 кОм, Rос 22 кОм; сигнал на вх"),
    ("06_active_lpf", "активный ФНЧ (инв.): Rвх 10к, Rос 100к, Cос 1,5н; сигнал на вх"),
    ("07_sallen_key_lpf", "ФНЧ Саллена-Ки на повторителе: R1 10 кОм, R2 10 кОм, C1(ос) 22 нФ, C2(земля) 10 нФ"),
    ("08_mfb_bandpass", "фильтр Рауха: R1 10к, R2 1к, R3 100к, Cа 10н, Cб 10н"),
    ("09_cascade", "каскад: [неинв. усилитель: Rз 10к, Rос 22к] → [Саллен-Ки ФНЧ: R1 15к, R2 15к, C1(ос) 10н, C2(земля) 4,7н]; сигнал на вх"),
    ("10_elements", "R1 1к вх-т1; C1 100н т1-0; R2 22к т1-вых; C2 10н вых-0; сигнал на вх"),
]


def describe(rep):
    """Краткий отчёт для экрана."""
    lines = []
    if "netlist" not in rep:
        return "\n".join(["Не разобрано."] + rep["messages"])
    lines.append("Нетлист:\n  " + rep["netlist"].strip().replace("\n", "\n  "))
    m = rep.get("measured")
    if m:
        kind = {"LP": "ФНЧ", "HP": "ФВЧ", "BP": "полосовой", "BS": "режекторный", "flat": "без частотной зависимости"}[m["type"]]
        k = round(m["k_pass_db"], 2) + 0.0  # без «-0.00»
        parts = [f"тип АЧХ: {kind}", f"коэффициент передачи {k:.2f} дБ"]
        if m.get("fc"):
            parts.append(f"частота среза {m['fc']:.4g} Гц")
        if m.get("f0") and m["type"] in ("BP", "BS"):
            parts.append(f"f0 {m['f0']:.4g} Гц, Q {m['Q']:.3g}")
        lines.append("Расчёт в MC12: " + ", ".join(parts))
    elif "run" in rep:
        lines.append("MC12: " + "; ".join(rep["run"]["messages"]))
    if rep.get("cir"):
        lines.append(f"Схема: {rep['cir']}")
    return "\n".join(lines)


def make_examples(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tr = translators.load("baseline")
    index = ["Готовые схемы: файл .cir - и запрос, из которого программа его построила.", ""]
    for name, prompt in EXAMPLES:
        rep = app.run(prompt, tr, outdir=folder, name=name)
        ok = "cir" in rep
        index.append(f"{name}.cir\n  запрос: {prompt}\n  {describe(rep).splitlines()[-2] if ok and rep.get('measured') else ''}")
        print(("OK   " if ok else "FAIL ") + name)
    for junk in list(folder.glob("*.ANO")) + list(folder.glob("*.ckt")) + list(folder.glob("*.inx")) + \
            list(folder.glob("mc12run_*")):
        junk.unlink()
    (folder / "zaprosy.txt").write_text("\n".join(index) + "\n", encoding="utf-8-sig")


def interactive(outdir):
    sys.stdout.reconfigure(encoding="utf-8")
    tr = translators.load("baseline")
    print("Запрос на русском -> схема в Micro-Cap 12. Пустая строка - выход.")
    print("Примеры:\n  " + "\n  ".join(p for _, p in EXAMPLES[:4]) + "\n")
    n = 0
    while True:
        try:
            prompt = input("Запрос: ").strip()
        except EOFError:
            break
        if not prompt:
            break
        n += 1
        rep = app.run(prompt, tr, outdir=outdir, name=f"zapros_{n:02d}", open_mc12=True)
        print(describe(rep) + "\n")


def main(argv):
    if "--examples" in argv:
        make_examples(argv[argv.index("--examples") + 1])
    else:
        out = argv[argv.index("--out") + 1] if "--out" in argv else str(app.ROOT / "out" / "demo")
        interactive(out)


if __name__ == "__main__":
    main(sys.argv)
