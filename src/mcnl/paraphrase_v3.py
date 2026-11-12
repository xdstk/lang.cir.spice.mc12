#!/usr/bin/env python3
"""Парафразы train v3 (docs/generator_v3_spec.md): запросы, переписанные Claude (автор C03).

Запуск:  python src/mcnl/paraphrase_v3.py export out/v3_full/train.jsonl out/v3_para_work.txt
         python src/mcnl/paraphrase_v3.py check out/v3_full/train.jsonl data/paraphrase_v3
         python src/mcnl/paraphrase_v3.py apply out/v3_full data/paraphrase_v3 data/dataset/v3

Проверка парафразы (check) строже, чем в v2 (paraphrase.py не меняется):
- номиналы и напряжение источника те же (paraphrase.check); у режима 2 так же сверяются частота, Q, усиление;
- название фильтра не противоречит эталону (Д4, paraphrase.name_conflict);
- имена узлов: в v3 эталон сохраняет имена из запроса, поэтому парафраза обязана назвать те же узлы и не вводить
  новых («узел А» в парафразе запроса без имён дал бы эталон с узлом A);
- режим 2: число слов на месте подбираемых номиналов не меньше, чем в запросе; признаки полей .SPEC (активный,
  порядок, аппроксимация, добротность, полоса, усиление) - те же;
- длина не больше generate_v3.MAX_PROMPT.
Соединения программа не проверяет: их выборочно смотрит автор (30 парафраз).
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import canon, generate_v3, paraphrase  # noqa: E402

AUTHOR = "C03"
SEED = 20261111
# имя узла после слова «узел», «точка»: латинское имя или номер; «к отдельному узлу 1,5к» и «узел и» - не имена
# номер узла - число, за которым не идёт единица: слитная («узла 39к») или полная через пробел («узлу 16 кОм»);
# «от узла 1 к выходу» - узел 1 и предлог (paraphrase.values v2 читает это как 1 кОм)
_NOT_VALUE = (r"(?!\d|[.,]\d|(?:" + paraphrase._UNIT + r")(?![а-яёА-ЯЁa-zA-Z])"
              r"|\s+(?:килоом|кОм|ком|мегаом|МОм|Ом|ом|мкГн|мГн|Гн|мкФ|нФ|пФ|мкф|нф|пф)(?![а-яёА-ЯЁa-zA-Z]))")
# одиночная кириллическая буква-двойник («узел С» после правки текста cyrillic_c) - то же имя, что латинская
_LOOKALIKE = str.maketrans("АВСЕКМНОРТХ", "ABCEKMHOPTX")
_WORD_NODE = re.compile(r"(?:узл\w*|узел|точк\w+)\s+([A-Za-z][A-Za-z0-9_]*(?![а-яёА-ЯЁ])|[АВСЕКМНОРТХ](?![\wа-яёА-ЯЁ])|\d+"
                        + _NOT_VALUE + r")", re.I)
_NODE_NUMBER = re.compile(r"((?:узл\w*|узел|точк\w+)\s+)\d+" + _NOT_VALUE, re.I)
_BARE = re.compile(r"(?<![\w.,])([A-Za-z][A-Za-z0-9_]*)(?!\w)")
_SHORT_NODE = re.compile(r"(?<![\w.,])[тТуУ]\.?(\d+)(?!\w)")
_MARK = re.compile(r"\?|подобр|подбер|рассчит|найти|найди|неизвест|посчита|определи|вычисл", re.I)
_NOT_NODES = {"RC", "RL", "LC", "RLC", "MFB", "KHN", "GND", "IN", "OUT", "FC", "Q", "K", "F0", "FC", "TOW", "THOMAS", "SALLEN",
              "KEY", "NE555", "T", "TWIN", "NOTCH", "STATE", "VARIABLE", "U", "E12", "E24"}
CUES = {"active": r"активн|на ОУ|на оу|операционн|с ОУ", "order": r"порядк", "response": r"баттерворт|чебыш[её]в",
        "Q": r"добротност|Q\s*=", "bw": r"полос(?:а|ы|у|е|ой)\b|ширин", "K": r"усилени|(?-i:[KК])\s*=", "ripple": r"неравномерн|пульсац",
        "series": r"точн|без округлен"}


def node_names(prompt):
    """Имена узлов, названные в запросе, в записи эталона: буквы - заглавными, числа - Nk."""
    out = set()
    for m in _WORD_NODE.finditer(prompt):
        t = m.group(1)
        out.add(f"N{int(t)}" if t.isdigit() else t.upper().translate(_LOOKALIKE))
    out |= {f"N{int(m.group(1))}" for m in _SHORT_NODE.finditer(prompt)}
    return out


def answer_nodes(rec):
    if rec.get("refuse") or not rec.get("netlist") and rec.get("mode") != 2:
        return set()
    text = rec["netlist"] if rec.get("mode") != 2 else "\n".join(l for l in rec["output"].splitlines() if not l.startswith("."))
    nodes = set()
    for line in text.splitlines():
        t = line.split()
        if t and t[0][0].upper() in "RCLV":
            nodes |= set(t[1:3])
        elif t and t[0][0].upper() == "X":
            nodes |= set(t[1:4])
    return {n.upper() for n in nodes} - {"IN", "OUT", "0"}


def check(rec, new):
    """-> список расхождений парафразы new с записью rec; пустой - парафраза принята."""
    old, errs = rec["prompt"], []
    if len(new) > generate_v3.MAX_PROMPT:
        errs.append(f"длина {len(new)}")
    if rec.get("refuse"):
        return errs
    def clean(text):  # «к отдельному узлу 330» - номинал, не номер узла
        return _NODE_NUMBER.sub(r"\1", re.sub(r"отдельн\w+\s+уз\w+", " ", text))

    errs += paraphrase.check(clean(old), clean(new))
    conflict = paraphrase.name_conflict(old, new, (rec.get("expected") or rec.get("spec") or {}).get("type"))
    if conflict:
        errs.append(conflict)
    named = node_names(old)
    answer = answer_nodes(rec)
    if named:  # узлы названы в запросе: те же имена должны быть в парафразе (с буквой - и без слова «узел»)
        latin = {m.group(1).upper() for m in _BARE.finditer(new)} | node_names(new)
        missing = sorted(n for n in named & answer if n not in latin)
        if missing:
            errs.append(f"узлы потеряны: {missing}")
    extra = sorted(node_names(new) - named - _NOT_NODES)
    if extra:
        errs.append(f"новые имена узлов: {extra}")
    if rec.get("mode") == 2:
        # слово на месте подбираемого номинала; у запроса по характеристикам «подбери» - вступление, не пометка
        if rec.get("unknown") and _MARK.search(old) and not _MARK.search(new):
            errs.append("нет слова о подбираемых номиналах")
        for field, pattern in CUES.items():
            a, b = bool(re.search(pattern, old, re.I)), bool(re.search(pattern, new, re.I))
            if a != b and (rec["kind"] == "spec" or field in ("Q", "K")):
                errs.append(f".SPEC {field}: в запросе {'есть' if a else 'нет'}, в парафразе {'есть' if b else 'нет'}")
    return errs


def export(train, out):
    """Записи под парафразы, по семействам: «id <tab> семейство <tab> запрос»."""
    recs = [r for r in paraphrase.read_jsonl(train) if r.get("paraphrase")]
    recs.sort(key=lambda r: (r["family"], r["id"]))
    lines = [f"{r['id']}\t{r['family']}\t{r['prompt']}".replace("\n", " / ") for r in recs]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(recs)


def check_all(train, folder):
    recs = {r["id"]: r for r in paraphrase.read_jsonl(train)}
    para = paraphrase.read_tsv(folder)
    bad = [(i, check(recs[i], t)) for i, t in para.items() if i in recs]
    bad = [(i, e) for i, e in bad if e]
    todo = [i for i, r in recs.items() if r.get("paraphrase") and i not in para]
    return para, bad, todo, [i for i in para if i not in recs or not recs[i].get("paraphrase")]


def apply(srcdir, folder, outdir):
    """Подставить парафразы в train, остальные выборки скопировать; манифест с SHA-256. Парафраза, не прошедшая
    проверку, не применяется: запись остаётся с шаблонным запросом."""
    srcdir, outdir = Path(srcdir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    para = paraphrase.read_tsv(folder)
    manifest = json.loads((srcdir / "manifest.json").read_text(encoding="utf-8"))
    applied, errors, rng = 0, [], random.Random(SEED)
    for name in list(manifest["files"]):
        recs = paraphrase.read_jsonl(srcdir / name)
        for r in recs:
            new = para.get(r["id"])
            if name != "train.jsonl" or not r.get("paraphrase") or not new:
                continue
            errs = check(r, new)
            if errs:
                errors.append({"id": r["id"], "errors": errs})
                continue
            chatty = rng.random() < paraphrase.CHATTY_SHARE
            if chatty:
                pre, post = rng.choice(paraphrase.CHATTY)
                new = pre + new.rstrip(".") + post
            r.update(prompt_template=r["prompt"], prompt=new, family_template=r["family"], family="paraphrase",
                     author=AUTHOR, chatty=chatty)
            applied += 1
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8")
        (outdir / name).write_bytes(data)
        manifest["files"][name] = {"records": len(recs), "sha256": hashlib.sha256(data).hexdigest()}
    manifest["paraphrases"] = {"author": AUTHOR, "applied": applied, "rejected": len(errors)}
    (outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return applied, errors


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    if argv[1] == "export":
        print(export(argv[2], argv[3]))
    elif argv[1] == "check":
        para, bad, todo, unknown = check_all(argv[2], argv[3])
        print(f"парафраз: {len(para)}, осталось написать: {len(todo)}, лишних id: {len(unknown)}, расхождений: {len(bad)}")
        for i, e in bad[:40]:
            print(i, e)
    else:
        applied, errors = apply(argv[2], argv[3], argv[4])
        print(f"применено {applied}, отклонено {len(errors)}")
        for e in errors[:20]:
            print(e["id"], e["errors"])


if __name__ == "__main__":
    main(sys.argv)
