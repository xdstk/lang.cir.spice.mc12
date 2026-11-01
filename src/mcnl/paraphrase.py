#!/usr/bin/env python3
"""Парафразы train v2 (docs/generator_v2_spec.md, раздел 4): запросы, переписанные Claude (автор C02).

Запуск:  python src/mcnl/paraphrase.py export out/v2_full/train.jsonl out/v2_para_todo.txt
         python src/mcnl/paraphrase.py apply out/v2_full data/paraphrase_v2/C02.jsonl data/dataset/v2

Парафраза меняет слова, но не схему: номиналы и напряжение источника должны остаться теми же (числами в любой
записи: 0,1 мкФ = 100 нФ). Список соединений проверяет автор выборочно. Файл парафраз - jsonl {"id", "prompt"}.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

CHATTY_SHARE = 0.05
CHATTY = (("Подскажи, как собрать: ", ""), ("Нужна помощь со схемой: ", ""), ("Мне для лабы: ", ""), ("Набросай, пожалуйста: ", ""),
          ("", ". Нарисуешь?"), ("", ". Сможешь собрать?"), ("", ". Больше ничего не надо."), ("Такая задача: ", ". Сделаешь?"))
_NUM = r"\d+(?:[.,]\d+)?"
_UNIT = (r"килоом|кОм|ком|мегаом|МОм|Ом|ом|мкГн|мГн|Гн|мкФ|нФ|пФ|мкф|нф|пф|мк|к|М|н|п")
VALUE_RE = re.compile(r"(?<![\w.,])(" + _NUM + r")\s*(" + _UNIT + r")?(?![а-яёА-ЯЁa-zA-Z0-9])")
VOLT_RE = re.compile(r"(?<![\w.,])(-?" + _NUM + r")\s*В(?![а-яё])")
MULT = {"килоом": ("R", 1e3), "кОм": ("R", 1e3), "ком": ("R", 1e3), "к": ("R", 1e3), "мегаом": ("R", 1e6), "МОм": ("R", 1e6),
        "М": ("R", 1e6), "Ом": ("R", 1.0), "ом": ("R", 1.0), "": ("R", 1.0), "мкГн": ("L", 1e-6), "мГн": ("L", 1e-3), "Гн": ("L", 1.0),
        "мкФ": ("C", 1e-6), "мкф": ("C", 1e-6), "мк": ("C", 1e-6), "нФ": ("C", 1e-9), "нф": ("C", 1e-9), "н": ("C", 1e-9),
        "пФ": ("C", 1e-12), "пф": ("C", 1e-12), "п": ("C", 1e-12)}
# цифры в обозначениях (R1, C2, т3, порядок звена) и напряжение - не номиналы
_DESIGNATOR = re.compile(r"\b[RCLXVrclРСЛ]\w*\d+\b|\b[RCL]\w*\b|[тТу]\.?\d+|\b\d+[-‑]?(?:го|й|е|я|х)\b|\b\d\)|\d+\s*звен\w*|\d+\s*ОУ"
                         r"|±\s*\d+\s*В")
_NODE_NUMBER = re.compile(r"(?:точк|узл|узел)\w*\s+\d+(?!\d|[.,]\d|\s*(?:" + _UNIT + r")\b)", re.I)  # «узел 3», но не «с узла 39н»


def values(prompt):
    """Номиналы запроса -> отсортированный список (тип, значение до 4 значащих цифр)."""
    text = VOLT_RE.sub(" ", _NODE_NUMBER.sub(" ", _DESIGNATOR.sub(" ", prompt.replace("отдельному узлу", " "))))
    out = []
    for m in VALUE_RE.finditer(text):
        kind, mult = MULT[m.group(2) or ""]
        v = float(m.group(1).replace(",", "."))
        if v:  # «т1-0» - узел земли, не номинал
            out.append((kind, float(f"{v * mult:.4e}")))
    return sorted(out)


def volts(prompt):
    return sorted(float(m.group(1).replace(",", ".")) for m in VOLT_RE.finditer(_DESIGNATOR.sub(" ", prompt)))


def check(original, new):
    """-> список расхождений; пустой - парафраза сохранила номиналы."""
    errs = []
    a, b = values(original), values(new)
    if a != b:
        errs.append(f"номиналы: было {a}, стало {b}")
    if volts(original) != volts(new):
        errs.append(f"напряжение: было {volts(original)}, стало {volts(new)}")
    return errs


def read_tsv(folder):
    """Части data/paraphrase_v2/part*.tsv: id, табуляция, запрос."""
    out = {}
    for f in sorted(Path(folder).glob("part*.tsv")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rid, text = line.split("\t", 1)
                out[rid] = text.strip()
    return out


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def export(train, out):
    recs = [r for r in read_jsonl(train) if r.get("paraphrase")]
    lines = [f"{r['id']}\t{r['prompt']}".replace("\n", " / ") for r in recs]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(recs)


def apply(srcdir, para_file, outdir):
    """Подставить парафразы в train, остальные выборки скопировать; записать манифест с SHA-256."""
    srcdir, outdir = Path(srcdir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    para = read_tsv(para_file) if Path(para_file).is_dir() else {r["id"]: r["prompt"] for r in read_jsonl(para_file)}
    manifest = json.loads((srcdir / "manifest.json").read_text(encoding="utf-8"))
    applied, errors, rng = 0, [], random.Random(20261101)
    for name in list(manifest["files"]):
        recs = read_jsonl(srcdir / name)
        for r in recs:
            new = para.get(r["id"])
            if name == "train.jsonl" and r.get("paraphrase") and new:
                errs = [] if r.get("refuse") else check(r["prompt"], new)
                if errs:
                    errors.append({"id": r["id"], "prompt": r["prompt"], "new": new, "errors": errs})
                    continue
                chatty = rng.random() < CHATTY_SHARE
                if chatty:  # разговорная оболочка: малая доля train, эффект смотрим на llm_test_chatty
                    pre, post = rng.choice(CHATTY)
                    new = pre + new.rstrip(".") + post
                r.update(prompt_template=r["prompt"], prompt=new, family_template=r["family"], family="paraphrase",
                         author="C02", chatty=chatty)
                applied += 1
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8")
        (outdir / name).write_bytes(data)
        manifest["files"][name] = {"records": len(recs), "sha256": hashlib.sha256(data).hexdigest()}
    manifest["paraphrases"] = {"author": "C02", "applied": applied, "rejected": len(errors)}
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return applied, errors


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    if argv[1] == "export":
        print(export(argv[2], argv[3]))
    elif argv[1] == "check":  # проверка написанных частей без записи выборок
        recs = {r["id"]: r for r in read_jsonl(argv[2])}
        para = read_tsv(argv[3])
        bad = [(i, check(recs[i]["prompt"], t)) for i, t in para.items() if i in recs and not recs[i].get("refuse")]
        bad = [(i, e) for i, e in bad if e]
        print(f"частей: {len(para)}, неизвестных id: {sum(i not in recs for i in para)}, расхождений: {len(bad)}")
        for i, e in bad[:25]:
            print(i, e)
    else:
        applied, errors = apply(argv[2], argv[3], argv[4])
        print(f"применено {applied}, отклонено {len(errors)}")
        for e in errors[:20]:
            print(e["id"], e["errors"], "\n ", e["prompt"][:150], "\n ", e["new"][:150])


if __name__ == "__main__":
    main(sys.argv)
