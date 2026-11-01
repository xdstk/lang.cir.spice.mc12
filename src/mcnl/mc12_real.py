#!/usr/bin/env python3
"""Выборка mc12_real: реальные схемы фильтров из примеров MC12 и дизайнера (docs/generator_v2_spec.md, раздел 5).

Запуск:  python src/mcnl/mc12_real.py schemes data/dataset/mc12_real_v1      -> схемы, проверка в MC12, текст задания
         python src/mcnl/mc12_real.py prompts файл.txt data/dataset/mc12_real_v1  -> запросы внешней модели, заморозка

Схемы написаны не нами: выборка показывает, покрывает ли класс реальные фильтры, независимо от генератора.
Перенос из примеров: библиотечный ОУ заменён на OPAMP_IDEAL, питание и смещение нуля по постоянному току убраны
(плюс ОУ, сидевший на смещении, - на земле), источник - V1 IN 0. Узлы и номиналы - как в файлах DATA/Filters.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcnl import app, canon, filters_v2, pilot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
X = "OPAMP_IDEAL"
FROM_DATA = {
    "twin_t_notch": ("Twin T Notch Filter", f"""R1 A IN 26.7k
R2 N1 A 26.7k
C1 IN B 100n
C2 B N1 100n
R3 N3 B 13.3k
C3 N3 A 200n
R4 N2 0 95.3k
R5 OUT N2 4.99k
X1 N1 OUT OUT {X}
X2 N2 N3 N3 {X}
"""),
    "state_variable": ("State Variable Filter", f"""R1 A IN 29.5k
C1 A B 10n
R2 C B 8.6k
C2 C D 10n
R3 E D 11.8k
C3 E F 10n
R4 G F 21.5k
C4 G OUT 10n
R5 OUT H 11.8k
X1 D A B {X}
X2 0 C D {X}
X3 H E F {X}
X4 0 G OUT {X}
"""),
    "butterworth_2stage": ("Butterworth Filter", f"""C1 A B 47p
R1 A C 909
R2 B C 2.67k
R3 C IN 909
C2 0 C 220p
C3 OUT D 22p
R4 OUT E 1.1k
R5 D E 2.21k
R6 E A 1.1k
C4 0 E 470p
X1 0 B A {X}
X2 0 D OUT {X}
"""),
    "chebyshev_2stage": ("Chebyshev Lowpass Filter", f"""C1 A B 33n
R1 A C 237k
R2 B C 249k
R3 C IN 237k
C2 0 C 100n
C3 OUT D 10n
R4 OUT E 154k
R5 D E 249k
R6 E A 154k
C4 0 E 330n
X1 0 B A {X}
X2 0 D OUT {X}
"""),
    "sallen_key_hp_gain": ("Sallen-Key High Pass Filter", f"""R1 OUT A 18.2k
C1 IN A 1n
C2 B A 1n
R3 0 B 14k
R4 OUT C 10k
R5 0 C 10k
X1 B C OUT {X}
"""),
    "bandpass_3stage": ("BPFILT", f"""R3 A IN 1.13k
R1 S1 B 4.48k
R4 S1 C 334
C6 B A 100n
C1 S1 A 100n
X1 0 B S1 {X}
R2 S2 D 6.35k
R5 S2 E 794
C2 D C 100n
C3 S2 C 100n
X2 0 D S2 {X}
R6 OUT F 15k
C4 F E 100n
C5 OUT E 100n
X3 0 F OUT {X}
"""),
    "sallen_key_lp": ("FILTER", f"""R1 A IN 28.422k
R2 B A 77.782k
C1 B 0 60p
C5 A OUT 50n
X1 B OUT OUT {X}
"""),
    "crossover_low": ("CROSSOVR, ветвь нижних частот", """R1 A IN 8
C1 A 0 14.06u
L1 A OUT 1.8m
R2 OUT 0 8
"""),
}
TASK_HEAD = """Ниже схемы фильтров из примеров Micro-Cap. На каждую напиши один запрос к программе, которая по русскому тексту строит схему: так, как написал бы человек, который хочет получить именно эту схему.
Все номиналы должны быть в запросе, соединения должны восстанавливаться однозначно. Обозначения узлов из списка не копируй, описывай своими словами: что откуда и куда идёт, или назови типовую схему, если это точно она. Разный стиль и длина, без вступлений и вопросов, без пояснений.
Формат: один запрос в строке, нумерация как в списке.

"""


def schemes():
    """-> [(имя, источник, нетлист)]: примеры DATA и схемы дизайнера, кроме двух самых длинных."""
    out = [(k, f"DATA/Filters: {src}", "V1 IN 0 DC 0\n" + net) for k, (src, net) in FROM_DATA.items()]
    for f in sorted((ROOT / "data" / "designer").glob("*.net")):
        if len(f.read_text(encoding="ascii").splitlines()) <= 15:
            out.append((f.stem, f"дизайнер MC12: {f.stem}", f.read_text(encoding="ascii")))
    return out


def references(workdir=None):
    refs = []
    for name, src, net in schemes():
        el = canon.canonicalize(canon.parse_netlist(net)[0])
        f = filters_v2.characteristic_frequency(el) or app.characteristic_frequency(el)
        text = canon.to_netlist(el)
        refs.append({"id": name, "class": name, "source": src, "level": "L1" if "OPAMP" not in text else "L2",
                     "netlist": text, "circuit_id": hashlib.sha256(text.encode()).hexdigest()[:16],
                     "f_char": f, "expected": {}, "quantities": []})
    return pilot.admit(refs, Path(workdir or ROOT / "out" / "mc12_real"))


def build_schemes(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    refs = references()
    lines = [TASK_HEAD]
    for i, r in enumerate(refs, 1):
        lines.append(f"{i}. " + "; ".join(l for l in r["netlist"].splitlines() if not l.startswith("V1")))
    (outdir / "task.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (outdir / "schemes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in refs), encoding="utf-8", newline="\n")
    return refs


def build_prompts(prompt_file, outdir):
    """Запросы внешней модели (порядок - как в task.txt) -> mc12_real.jsonl и манифест с SHA-256."""
    outdir = Path(outdir)
    refs = [json.loads(l) for l in (outdir / "schemes.jsonl").read_text(encoding="utf-8").splitlines()]
    lines = [re.sub(r"^\d+\.\s*", "", l.strip()) for l in Path(prompt_file).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if len(lines) != len(refs):
        raise ValueError(f"запросов {len(lines)}, схем {len(refs)}")
    recs = []
    for r, p in zip(refs, lines):
        recs.append({"id": r["id"], "circuit_id": r["circuit_id"], "split": "mc12_real", "task": "A", "topology": r["id"], "level": r["level"],
                     "family": "mc12_real", "source": r["source"], "prompt": p, "netlist": r["netlist"],
                     "ir": canon.to_ir(canon.parse_netlist(r["netlist"])[0]), "f_char": r["f_char"], "quantities": [],
                     "expected": {}, "measured": r["measured"], "admission": r["admission"]})
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8")
    (outdir / "mc12_real.jsonl").write_bytes(data)
    (outdir / "manifest.json").write_text(json.dumps({"frozen": "2026-11-01", "files": {"mc12_real.jsonl": {
        "records": len(recs), "sha256": hashlib.sha256(data).hexdigest()}}}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return recs


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    if argv[1] == "prompts":
        print(len(build_prompts(argv[2], argv[3])))
    if argv[1] == "schemes":
        for r in build_schemes(argv[2]):
            m = r.get("measured") or {}
            print(f"{r['id']:22} {r['admission']['mc12_run']} {m.get('type')} fc={m.get('fc')} f0={m.get('f0')} k={m.get('k_pass_db')}")


if __name__ == "__main__":
    main(sys.argv)
