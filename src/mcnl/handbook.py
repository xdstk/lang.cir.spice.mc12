#!/usr/bin/env python3
"""Справочник фильтров (вкладка окна, docs/gui_spec.md, «Справочник»): статьи, примеры схем, файлы из MC12.

Статья описывает тип цепи в общем виде (решение автора 16.11.2026: «менее конкретно, более общее описание цепи;
можно на примере, но чуть обезличить»): что это, как работает, элементы, формулы, характеристики, применение,
достоинства, недостатки, короткий пример расчёта и запросы для программы. К статье - файлы в data/handbook:
<id>.cir (пример с номиналами, его открывает кнопка «Открыть пример в MC12») и <id>_ac.json (АЧХ примера, расчёт
MC12: чертёж и, у схем с ОУ, обвязка с идеальным ОУ). Схема статьи - рисунок по ГОСТ (handbook_schematics.py,
schematic.py, решение автора 17.11.2026; раньше - картинка чертежа MC12). Числа примера (поле check) и формулы проверяет тест по узловому расчёту.

Статьи - handbook_articles.py: 29 (все схемы класса и реальный ОУ; пилот из 5 одобрен автором 16.11.2026). Внешний вид
(формат формул и др.) - позже, вместе со всем интерфейсом.

Сборка файлов:  python src/mcnl/handbook.py --build     (нужен MC12, пакетный режим)
"""
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import bench, canon, measure, mc12run, mna, spice2cir  # noqa: E402
from mcnl.handbook_articles import ARTICLES, GROUPS, OPAMP  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "handbook"
WORK = ROOT / "out" / "handbook_build"
SECTIONS = (("properties", "Характеристики"), ("use", "Применение"), ("pros", "Достоинства"),
            ("cons", "Недостатки и ограничения"), ("program", "В программе"))

BY_ID = {a["id"]: a for a in ARTICLES}


def f_char(article):
    """Характерная частота примера - как в программе (app.characteristic_frequency): по ней диапазон AC."""
    from mcnl import app
    return app.characteristic_frequency(canon.parse_netlist(article["netlist"])[0])


def measured(article):
    """Узловой расчёт примера с идеальным ОУ (без MC12): тип, fc, f0, Q, K - для проверки чисел статьи."""
    elems = canon.parse_netlist(article["netlist"])[0]
    lo, hi = bench.ac_range(f_char(article))
    return measure.measure(*mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE))


def curve(article, d=None):
    """АЧХ примера из data/handbook/<id>_ac.json: {"f": [...], "db": [...], "db_ideal": [...] (у схем с ОУ)}."""
    p = Path(d or DIR) / f"{article['id']}_ac.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cir_text(article):
    """Чертёж примера: тот же, что строит программа (с TL071 у схем с ОУ), без текста нетлиста справа."""
    return spice2cir.build(article["netlist"], f_char(article), article["id"], opamp=article["opamp"], note=False)


def build(articles=ARTICLES, out=DIR):
    """Файлы статей: .cir (чертёж примера для MC12) и АЧХ примера из MC12 (пакетный режим, /quiet)."""
    out, work = Path(out), WORK
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for a in articles:
        spice2cir.write(out / f"{a['id']}.cir", cir_text(a))
        fc = f_char(a)
        spice2cir.write(work / f"{a['id']}_w.cir", spice2cir.build(a["netlist"], fc, a["id"], opamp=a["opamp"],
                                                                     numeric=True, note=False))
        bench.write(work / f"{a['id']}_r.ckt", bench.build(canon.parse_netlist(a["netlist"])[0], fc, a["id"]))
        paths += [work / f"{a['id']}_w.cir", work / f"{a['id']}_r.ckt"]
    res = {Path(r["ckt"]).name: r for r in mc12run.run(paths, analyses=["A"], timeout=600)}
    report = {}
    for a in articles:
        w, r = res[f"{a['id']}_w.cir"], res[f"{a['id']}_r.ckt"]
        if not (w["ok"] and r["ok"]):
            raise RuntimeError(f"{a['id']}: MC12 не посчитал пример: {w['messages']} {r['messages']}")
        f, db, _ = measure.read_curve(w["output"])
        data = {"f": [float(f"{x:.5g}") for x in f], "db": [round(x, 3) for x in db]}
        if a["opamp"]:
            _, dbi, _ = measure.read_curve(r["output"])
            data["db_ideal"] = [round(x, 3) for x in dbi]
        (out / f"{a['id']}_ac.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        report[a["id"]] = {"real": measure.from_output(w["output"]), "ideal": measure.from_output(r["output"])}
    return report


if __name__ == "__main__":
    if "--build" in sys.argv:
        rep = build()
        sys.stdout.reconfigure(encoding="utf-8")
        for k, v in rep.items():
            print(k, {n: {q: (round(x, 4) if isinstance(x, float) else x) for q, x in m.items() if x is not None}
                      for n, m in v.items()})
    else:
        sys.exit(__doc__)
