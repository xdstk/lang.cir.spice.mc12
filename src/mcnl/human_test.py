#!/usr/bin/env python3
"""Человеческий тест: сбор запросов, разметка эталонов, сборка выборок (docs/human_test.md).

Запуск:  python src/mcnl/human_test.py import файл.csv|файл.txt --author A02
         python src/mcnl/human_test.py propose        -> дописать новые запросы в data/human_test/labels.md
         python src/mcnl/human_test.py status
         python src/mcnl/human_test.py build          -> data/human_test/v1/{human_dev,human_test}.jsonl + manifest

labels.md - разметка, которую правит человек; запись:
    ## H0001 · A02
    > текст запроса
    class: A1            (A | A1 | B | out | unclear | ? - не размечено)
    ```netlist
    V1 IN 0 DC 0
    R1 IN OUT 1k
    ```
    note: необязательный комментарий
Эталон предлагает baseline; разметчик исправляет нетлист и class. В выборки идут только A и A1.
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import app, bench, canon, describe_ru, measure, mna, pilot, translators  # noqa: E402
from mcnl.translators import baseline as rule_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "human_test"
RAW = DIR / "raw.jsonl"
LABELS = DIR / "labels.md"
CLASSES = {"A", "A1", "B", "out", "unclear", "?"}
DEV_AUTHORS = {"A01"}  # автор работы знает шаблоны генератора: его запросы - dev, не тест
# C01, C02, … - запросы, написанные языковой моделью (внешней или Claude): это синтетика, не человеческий тест;
# отдельная выборка paraphrase (для v2 и отладки), в human_test не попадает
SPLITS = ("human_dev", "human_test", "paraphrase")
AUTHOR_RE = r"[AC]\d{2}"
QUANTITIES = {"LP": ["type", "k_pass_db", "fc"], "HP": ["type", "k_pass_db", "fc"],
              "BP": ["type", "k_pass_db", "f0", "Q"], "BS": ["type", "k_pass_db", "f0", "Q"],
              "flat": ["type", "k_pass_db"]}
LABELS_HEAD = ("# Разметка человеческого теста\n\n"
               "Правила - docs/human_test.md. Исправьте нетлист-эталон и class: A (элементы описаны), "
               "A1 (названа типовая схема), B (номиналы надо рассчитать), out (вне класса), unclear (неоднозначно); "
               "? - ещё не размечено.\n")


# ---------------------------------------------------------------- сбор
def read_raw():
    return [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines()] if RAW.exists() else []


def import_file(path, author):
    """CSV со столбцом prompt (как collect_template.csv) или текст - по запросу в строке."""
    path = Path(path)
    if not re.fullmatch(AUTHOR_RE, author):
        raise ValueError("код автора - A01, A02, … (люди) или C01 (запросы Claude)")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        prompts = [(r.get("prompt") or "").strip() for r in rows
                   if "пример - удалить" not in (r.get("note") or "")]  # строка-образец из шаблона
        prompts = [p for p in prompts if p]
    else:
        prompts = [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    raw = read_raw()
    known = {(r["author"], r["prompt"]) for r in raw}
    added = []
    for p in prompts:
        # в одном текстовом файле - запросы разных авторов: «A02: текст» относит строку к A02
        m = re.match(r"^(" + AUTHOR_RE + r")\s*[:|]\s*(.+)$", p)
        who, p = (m.group(1), m.group(2).strip()) if m else (author, p)
        if (who, p) in known or any((a["author"], a["prompt"]) == (who, p) for a in added):
            continue
        rec = {"id": f"H{len(raw) + len(added) + 1:04d}", "author": who, "prompt": p, "source": path.name}
        added.append(rec)
    DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW, "a", encoding="utf-8", newline="\n") as fh:
        for r in added:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return added


# ---------------------------------------------------------------- разметка
def parse_labels(text):
    """labels.md -> {id: {author, prompt, class, netlist, note}}."""
    out = {}
    for m in re.finditer(r"^## (H\d{4}) · (" + AUTHOR_RE + r")\n(.*?)(?=^## H\d{4} · |\Z)", text, re.S | re.M):
        body = m.group(3)
        prompt = "\n".join(l[2:] if l.startswith("> ") else l[1:] for l in body.splitlines() if l.startswith(">"))
        cls = re.search(r"^class:\s*(\S+)", body, re.M)
        net = re.search(r"```netlist\n(.*?)```", body, re.S)
        note = re.search(r"^note:\s*(.*)$", body, re.M)
        out[m.group(1)] = {"author": m.group(2), "prompt": prompt, "class": cls.group(1) if cls else "?",
                           "netlist": net.group(1).strip() + "\n" if net and net.group(1).strip() else "",
                           "note": note.group(1).strip() if note else ""}
    return out


def _section(rec, cls, netlist, note=""):
    quoted = "\n".join("> " + l for l in rec["prompt"].splitlines())
    return (f"\n## {rec['id']} · {rec['author']}\n{quoted}\nclass: {cls}\n```netlist\n{netlist.strip()}\n```\n"
            + (f"note: {note}\n" if note else ""))


def propose(translator=None):
    """Дописать в labels.md неразмеченные запросы с эталоном, предложенным переводчиком (baseline)."""
    translator = translator or translators.load("baseline")
    text = LABELS.read_text(encoding="utf-8") if LABELS.exists() else LABELS_HEAD
    have = parse_labels(text)
    new = [r for r in read_raw() if r["id"] not in have]
    for r in new:
        out = translator.translate(r["prompt"])
        if out:
            names = any(n.lower() in r["prompt"].lower() for t in describe_ru.ROLES.values()
                        for n, _ in rule_baseline._topology_names(t))
            text += _section(r, "A1" if names else "A", out, "предложено baseline - проверить")
        else:
            text += _section(r, "?", "", "baseline не разобрал - разметить вручную")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(text, encoding="utf-8", newline="\n")
    return new


def status():
    labels = parse_labels(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    by = {}
    for v in labels.values():
        by[v["class"]] = by.get(v["class"], 0) + 1
    return {"raw": len(read_raw()), "labelled": len(labels), "by_class": by,
            "by_author": {a: sum(1 for v in labels.values() if v["author"] == a)
                          for a in sorted({v["author"] for v in labels.values()})}}


# ---------------------------------------------------------------- сборка
def reference(hid, lab):
    """Размеченный запрос -> эталон в формате датасета (без measured: его даёт допуск в MC12)."""
    elems, err = canon.parse_netlist(lab["netlist"])
    if err or not elems:
        raise ValueError(f"{hid}: эталон не разбирается: {err or 'пусто'}")
    elems = canon.canonicalize(elems)
    netlist = canon.to_netlist(elems)
    f_char = app.characteristic_frequency(elems)
    lo, hi = bench.ac_range(f_char)
    m = measure.measure(*mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE))
    q = QUANTITIES[m["type"]]
    n_op = sum(e["type"] == "OPAMP" for e in elems)
    return {"id": hid, "circuit_id": hashlib.sha256(netlist.encode()).hexdigest()[:16], "author": lab["author"],
            "split": ("paraphrase" if lab["author"].startswith("C") else
                      "human_dev" if lab["author"] in DEV_AUTHORS else "human_test"), "task": lab["class"],
            "family": "human", "topology": f"human:{m['type']}", "level": "L1" if n_op == 0 else "L2" if n_op == 1 else "L3",
            "prompt": lab["prompt"], "netlist": netlist, "ir": canon.to_ir(elems), "f_char": f_char,
            "quantities": q, "expected": {k: m[k] for k in q}, "note": lab["note"]}


def build(outdir=None, workdir=None):
    labels = parse_labels(LABELS.read_text(encoding="utf-8"))
    refs, report = [], {"skipped": {}, "errors": []}
    for hid, lab in labels.items():
        if lab["class"] not in ("A", "A1"):
            report["skipped"][lab["class"]] = report["skipped"].get(lab["class"], 0) + 1
            continue
        try:
            refs.append(reference(hid, lab))
        except (ValueError, KeyError, ZeroDivisionError) as err:
            report["errors"].append(str(err))
    admitted = pilot.admit(refs, Path(workdir or ROOT / "out" / "human_runs")) if refs else []
    report["rejected"] = [{"id": r["id"], "admission": r["admission"]} for r in admitted if not r["admission"]["admitted"]]
    outdir = Path(outdir or DIR / "v1")
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"files": {}, "report": report}
    for split in SPLITS:
        recs = [r for r in admitted if r["admission"]["admitted"] and r["split"] == split]
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8")
        (outdir / f"{split}.jsonl").write_bytes(data)
        manifest["files"][f"{split}.jsonl"] = {"records": len(recs), "sha256": hashlib.sha256(data).hexdigest()}
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "import":
        author = argv[argv.index("--author") + 1]
        added = import_file(argv[2], author)
        print(f"добавлено {len(added)} запросов от {author}")
    elif cmd == "propose":
        print(f"в labels.md добавлено {len(propose())} запросов - проверьте эталоны и class")
    elif cmd == "build":
        print(json.dumps(build(), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
