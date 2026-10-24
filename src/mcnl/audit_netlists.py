#!/usr/bin/env python3
"""Статический аудит пар (подпись, SPICE-нетлист).

Запуск:  python3 -I audit.py <папка с captions/ spice/> <выходной_json>

Это НЕ симуляция. Проверяется то, что можно проверить без симулятора:
синтаксис, число узлов у элемента, неопределённые параметры и модели,
висячие узлы, наличие пути по постоянному току до земли, возбуждения и анализа,
следы markdown и плейсхолдеры, согласованность чисел подписи и нетлиста.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6,
          "µ": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "mil": 25.4e-6}
NUM = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)([a-zµμ]*)$", re.I)
FENCE_FIRST = {"plaintext", "spice", "text", "netlist", "txt", "plain", "ngspice"}
PLACEHOLDER = re.compile(
    r"\b(replace|assume|assuming|actual|if any|placeholder|missing|todo|"
    r"specify|as necessary|as needed|not shown|not specified|adjust)\b", re.I)


def spice_value(tok):
    """Число в SI по правилам SPICE ('m' = милли, 'meg' = мега). None если не число."""
    m = NUM.match(tok.strip())
    if not m:
        return None, None
    base, suf = float(m.group(1)), m.group(2).lower()
    if suf.startswith("meg"):
        return base * 1e6, "meg"
    if suf.startswith("mil"):
        return base * 25.4e-6, "mil"
    if suf and suf[0] in SUFFIX:
        return base * SUFFIX[suf[0]], suf[0]
    return base, ""


def join_continuations(lines):
    out = []
    for ln in lines:
        if ln.startswith("+") and out:
            out[-1] += " " + ln[1:].strip()
        else:
            out.append(ln)
    return out


def parse(text):
    flags = Counter()
    raw = text.replace("\r", "").split("\n")
    # остатки markdown-ограждения
    while raw and (raw[0].strip().lower() in FENCE_FIRST or raw[0].strip().startswith("```")):
        flags["fence_residue"] += 1
        raw = raw[1:]
    if any(l.strip().startswith("```") for l in raw):
        flags["fence_residue"] += 1
    comments = [l for l in raw if l.strip().startswith("*")]
    if any(PLACEHOLDER.search(c) for c in comments):
        flags["placeholder_comment"] = 1
    lines = []
    in_control = False
    for l in raw:
        s = l.strip()
        if not s or s.startswith("*") or s.startswith("```"):
            continue
        sl = s.lower()
        if sl.startswith(".control"):
            in_control = True
            flags["ngspice_control_block"] = 1
            continue
        if sl.startswith(".endc"):
            in_control = False
            continue
        if in_control:
            if re.match(r"(tran|ac|dc|op|tf|noise)\b", sl):
                flags["analysis_in_control"] = 1
            continue
        if ";" in s:
            body, cm = s.split(";", 1)
            if PLACEHOLDER.search(cm):
                flags["placeholder_comment"] = 1
            s = body.strip()
        if s:
            lines.append(s)
    return join_continuations(lines), flags, comments


def audit(text):
    lines, flags, comments = parse(text)
    elems, directives = [], []
    for s in lines:
        (directives if s.startswith(".") else elems).append(s)

    # определения
    models, params, subckts = set(), set(), {}
    cur = None
    has_include = False
    analyses = set()
    for d in directives:
        t = d.split()
        k = t[0].lower()
        if k == ".model" and len(t) > 1:
            models.add(t[1].lower())
        elif k in (".param", ".define"):
            for tok in re.findall(r"([A-Za-z_]\w*)\s*=", d):
                params.add(tok.lower())
            if k == ".define" and len(t) > 1:
                params.add(t[1].lower())
        elif k == ".subckt" and len(t) > 1:
            pins = [x for x in t[2:] if "=" not in x]
            subckts[t[1].lower()] = len(pins)
            cur = t[1].lower()
        elif k == ".ends":
            cur = None
        elif k in (".lib", ".include", ".inc"):
            has_include = True
        elif k in (".tran", ".ac", ".dc", ".op", ".tf", ".noise"):
            analyses.add(k)
    if not any(d.lower().split()[0] == ".end" for d in directives):
        flags["no_end"] = 1
    if not analyses and not flags.get("analysis_in_control"):
        flags["no_analysis"] = 1

    # разбор элементов (содержимое .subckt не входит в основную схему)
    body_elems, in_sub = [], False
    for s in lines:
        low = s.lower().split()[0]
        if low == ".subckt":
            in_sub = True
            continue
        if low == ".ends":
            in_sub = False
            continue
        if in_sub or s.startswith("."):
            continue
        body_elems.append(s)

    names, nodes_use = Counter(), Counter()
    node_kinds = {}
    graph = {}  # узел -> множество соседей по DC-путям
    n_ind_src, n_nonzero_src = 0, 0
    types = Counter()
    vnames = {e.split()[0].lower() for e in body_elems if e[0].upper() == "V"}
    r_meg_pitfall = 0
    nonnumeric_values = 0
    undefined_param = 0
    brace_values = 0

    def link(a, b):
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    for e in body_elems:
        t = e.split()
        name, kind = t[0], t[0][0].upper()
        types[kind] += 1
        names[name.lower()] += 1
        nn = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "B": 2,
              "E": 4, "G": 4, "F": 2, "H": 2, "Q": 3, "J": 3, "M": 4}.get(kind)
        if kind == "X":
            if len(t) < 3:
                flags["bad_element_line"] += 1
                continue
            sub = t[-1].lower()
            pins = t[1:-1]
            nodes = pins
            if sub not in subckts and not has_include:
                flags["undefined_subckt"] += 1
            elif sub in subckts and subckts[sub] != len(pins):
                flags["subckt_pin_mismatch"] += 1
            for p in pins[1:]:
                link(pins[0].lower(), p.lower())  # консервативно: все выводы связаны
            vals = []
        elif nn is None:
            flags["unknown_element_type"] += 1
            continue
        else:
            if len(t) < 1 + nn:
                flags["bad_element_line"] += 1
                continue
            nodes = t[1:1 + nn]
            rest = t[1 + nn:]
            vals = rest
            nl = [n.lower() for n in nodes]
            if kind in ("R", "L", "V", "D", "B", "E", "H"):
                link(nl[0], nl[1])
            if kind == "E" or kind == "H":
                pass  # выход - ветвь напряжения; управляющие узлы DC-путей не дают
            if kind == "Q":
                link(nl[0], nl[1]); link(nl[1], nl[2])
            if kind in ("J", "M"):
                link(nl[0], nl[2]) if kind == "J" else link(nl[0], nl[2])
            if kind in ("C", "I", "G", "F"):
                for n in nl:
                    graph.setdefault(n, set())
            if kind in ("D", "Q", "J", "M"):
                mname = None
                cand = [x for x in rest if "=" not in x and not NUM.match(x)]
                if kind == "M" and len(t) > 5:
                    cand = [t[5]] if "=" not in t[5] else cand
                mname = cand[0].lower() if cand else None
                if mname is None:
                    flags["no_model_name"] += 1
                elif mname not in models and not has_include:
                    flags["undefined_model"] += 1
            if len(set(nl)) < len(nl) and kind in ("R", "C", "L", "V", "I", "D"):
                flags["element_shorted_to_itself"] += 1
        for n in nodes:
            nodes_use[n.lower()] += 1
            node_kinds.setdefault(n.lower(), []).append(kind)
        # значения
        if kind in ("R", "C", "L"):
            if not vals:
                flags["missing_value"] += 1
            else:
                tok = vals[0]
                if "{" in tok:
                    brace_values += 1
                v, suf = spice_value(tok)
                inner = tok.strip("{}").lower()
                if v is None:
                    nonnumeric_values += 1
                    if not (set(re.findall(r"[A-Za-z_]\w*", inner)) <= params):
                        undefined_param += 1
                elif kind == "R" and suf == "m" and re.search(r"m$", tok.lower()) and not tok.lower().endswith("meg"):
                    r_meg_pitfall += 1
        if kind in ("V", "I"):
            n_ind_src += 1
            joined = " ".join(vals).lower()
            m = re.search(r"(?:dc\s+)?([+-]?\d[\w.+-]*)", joined)
            zero_only = (not joined) or re.fullmatch(r"(dc\s*)?0(\.0*)?", joined.strip()) is not None
            if not zero_only:
                n_nonzero_src += 1
            if joined and not re.search(r"\d", joined) and not re.search(r"sin|pulse|exp|pwl|sffm|ac", joined):
                flags["source_nonnumeric"] += 1
                if not (set(re.findall(r"[A-Za-z_]\w*", joined)) <= params | {"dc", "ac"}):
                    undefined_param += 1
        if kind in ("E", "G"):
            if vals and spice_value(vals[0].strip("{}"))[0] is None:
                nonnumeric_values += 1
                if not (set(re.findall(r"[A-Za-z_]\w*", vals[0].lower())) <= params):
                    undefined_param += 1
        if kind in ("F", "H"):
            if vals and vals[0].lower() not in vnames:
                flags["control_source_missing"] += 1

    if any(c > 1 for c in names.values()):
        flags["duplicate_names"] = 1
    if undefined_param:
        flags["undefined_param"] = undefined_param
    if nonnumeric_values:
        flags["nonnumeric_value"] = nonnumeric_values
    if brace_values:
        flags["brace_value"] = brace_values
    if r_meg_pitfall:
        flags["R_suffix_m_is_milli"] = r_meg_pitfall

    # узлы
    nodes = set(nodes_use)
    gnd = {"0", "gnd"}
    if not (nodes & gnd):
        flags["no_ground"] = 1
    dangling = [n for n, c in nodes_use.items() if c == 1 and n not in gnd]
    # реальный дефект: единственное подключение - вывод пассивного/токового элемента;
    # выход ОУ (X), источника напряжения (V/E/H) без нагрузки - просто точка съёма
    d_real = [n for n in dangling if node_kinds[n][0] in "RCLDIQMJGF"]
    d_probe = [n for n in dangling if n not in d_real]
    if d_real:
        flags["dangling_passive"] = len(d_real)
    if d_probe:
        flags["dangling_probe_only"] = len(d_probe)
    # путь по постоянному току до земли
    seen, stack = set(), [g for g in gnd if g in nodes]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(graph.get(x, ()))
    no_dc = [n for n in nodes if n not in seen and n not in gnd]
    if nodes and (nodes & gnd) and no_dc:
        flags["no_dc_path_to_ground"] = len(no_dc)
    if n_nonzero_src == 0:
        flags["no_excitation"] = 1
    if not body_elems:
        flags["empty"] = 1

    # класс схемы
    cls = []
    if types.keys() & {"Q", "M", "J"}:
        cls.append("transistor")
    if "D" in types:
        cls.append("diode")
    if "X" in types or "E" in types:
        cls.append("opamp_or_subckt")
    if not cls:
        cls.append("passive_or_sources")
    return {
        "n_elements": len(body_elems),
        "types": dict(types),
        "class": cls,
        "flags": dict(flags),
        "analyses": sorted(analyses),
        "n_nodes": len(nodes),
        "node_names": sorted(nodes),
        "values_si": {"R": [], "C": [], "L": []},
        "_values": _collect_values(body_elems),
    }


def _collect_values(body_elems):
    out = {"R": [], "C": [], "L": []}
    for e in body_elems:
        t = e.split()
        k = t[0][0].upper()
        if k in out and len(t) > 3:
            v, _ = spice_value(t[3].strip("{}"))
            if v is not None:
                out[k].append(v)
    return out


CAP_UNITS = {"Ω": "R", "F": "C", "H": "L"}


def caption_values(cap):
    c = cap
    for a, b in [("\\text{", ""), ("\\mathrm{", ""), ("\\,", " "), ("\\ ", " "), ("\\mu", "μ"),
                 ("\\Omega", "Ω"), ("\\(", " "), ("\\)", " "), ("}", ""), ("{", ""), ("$", " "),
                 ("ohms", "Ω"), ("ohm", "Ω"), ("kilo", "k")]:
        c = c.replace(a, b)
    out = {"R": [], "C": [], "L": []}
    for num, pre, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(k|M|m|μ|µ|u|n|p)?\s*(Ω|F|H)\b", c):
        mult = {"": 1, "k": 1e3, "M": 1e6, "m": 1e-3, "μ": 1e-6, "µ": 1e-6, "u": 1e-6,
                "n": 1e-9, "p": 1e-12}[pre]
        out[CAP_UNITS[unit]].append(float(num) * mult)
    return out


def close(a, b):
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-30)


def caption_features(cap):
    f = {}
    f["len_chars"] = len(cap)
    f["mentions_figure"] = bool(re.search(r"\b(figure|fig\.)\s*[\d.]*", cap, re.I))
    f["page_or_image_meta"] = bool(re.search(
        r"\b(the page|this page|the image|the table|the text|the graph|the plot|the figure shows|depicts a page)\b", cap, re.I))
    f["has_formula"] = "\\(" in cap or "\\frac" in cap
    return f


def main(root, out):
    root = Path(root)
    base = root if (root / "spice").is_dir() else next(p for p in root.rglob("spice") if p.is_dir()).parent
    recs = []
    for sp in sorted((base / "spice").glob("spice*.txt")):
        rid = sp.stem.replace("spice", "")
        cap_p = base / "captions" / f"cap{rid}.txt"
        text = sp.read_text(encoding="utf-8", errors="replace")
        cap = cap_p.read_text(encoding="utf-8", errors="replace") if cap_p.exists() else ""
        a = audit(text)
        cf = caption_features(cap)
        cv = caption_values(cap)
        nv = a["_values"]
        # согласованность чисел
        cap_total = sum(len(v) for v in cv.values())
        cap_matched = sum(1 for k in cv for x in cv[k] if any(close(x, y) for y in nv[k]))
        net_total = sum(len(v) for v in nv.values())
        net_in_cap = sum(1 for k in nv for y in nv[k] if any(close(y, x) for x in cv[k]))
        rec = {"id": rid, **{k: v for k, v in a.items() if not k.startswith("_") and k != "values_si"},
               "caption": cf,
               "cap_vals": cap_total, "cap_vals_matched": cap_matched,
               "net_vals": net_total, "net_vals_in_caption": net_in_cap,
               "has_caption": bool(cap)}
        recs.append(rec)
    Path(out).write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"records: {len(recs)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
