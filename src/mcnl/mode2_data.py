#!/usr/bin/env python3
"""Примеры режима 2 для датасета v3 (docs/mode2_spec.md, разделы 5 и 7; docs/generator_v3_spec.md, «Режим 2»).

Запрос - какой фильтр нужен; ответ модели - нетлист с `?` на месте подбираемых номиналов и строка `.SPEC`, либо одна
строка `.SPEC`, если схема не названа и не описана. Номиналы считает synth.py, модель их не считает.

Виды примеров (sample):
  spec    - только характеристики: «нужен активный ФНЧ Баттерворта 4-го порядка на 1 кГц» -> `.SPEC ...`;
  named   - названная схема, часть номиналов задана, остальные подобрать (или не задан ни один);
  circuit - схема описана соединениями (prompts_v3), 1-2 номинала подобрать, тип и частота названы;
  refuse  - частота не задана или схема вне класса.
В `.SPEC` пишется только названное в запросе; значения по умолчанию подставляет synth. У named и circuit цели - из
измерений эталона (срез, f0, Q, усиление, округлённые до 2-3 цифр), поэтому задача заведомо решается. Каждый ответ
проверяется: synth.run находит номиналы без ошибки и попадает в названные цели с допуском TOL.
"""
import hashlib
import math
import random
import re

try:
    from . import canon, compose_v3, describe_ru, edge_cases, filters_v2, filters_v3, prompts_v3, synth
except ImportError:  # запуск как скрипта
    import canon
    import compose_v3
    import describe_ru
    import edge_cases
    import filters_v2
    import filters_v3
    import prompts_v3
    import synth

MIX = {"spec": 0.35, "named": 0.35, "circuit": 0.25, "refuse": 0.05}
TOL = {"rel": 0.10, "db": 1.0}  # отклонение подобранной схемы от названных целей (номиналы - из рядов)
FIELDS = ("type", "fc", "f0", "order", "response", "ripple", "Q", "bw", "active", "K", "series")  # порядок в .SPEC
NAMED_ALL_SHARE = 0.3            # названная схема без единого номинала
TYPE_WORDS = {"LP": ["ФНЧ", "фильтр нижних частот", "фильтр низких частот"],
              "HP": ["ФВЧ", "фильтр верхних частот", "фильтр высоких частот"],
              "BP": ["полосовой фильтр", "ПФ", "полосно-пропускающий фильтр"],
              "BS": ["режекторный фильтр", "заграждающий фильтр", "РФ", "фильтр-пробка"]}
# глаголы-вступления - в четверти запросов: замечание автора 11.11.2026 (меньше внимания к «сделай», «собери»)
HEADS = ["", "", "", "", "", "", "нужен ", "нужен ", "требуется ", "рассчитай ", "подбери ", "сделай "]
MARKERS = ["?", "?", "?", "подобрать", "рассчитать", "найти"]
MANTISSAS = [1, 1.2, 1.5, 2, 2.2, 2.5, 3, 3.3, 4, 4.7, 5, 6.8, 8]
REASON_NO_FREQ = "не задана частота"
_UNIT = r"(?:\s?(?:кОм|килоом|нФ|пФ|мГн|мкГн)|к|н)"
_SCALE = {"R": ("k", 1e3), "C": ("n", 1e-9), "L": ("m", 1e-3)}


class _Tagged(str):
    """Номинал с номером исходной строки: canon.canonicalize переносит объект value как есть."""


# ---------------------------------------------------------------- запись чисел
def _num(x, rng, comma=None):
    s = f"{x:g}"
    return s.replace(".", ",") if (rng.random() < 0.7 if comma is None else comma) else s


def freq_text(f, rng):
    if f >= 1e6:
        return f"{_num(f / 1e6, rng)} МГц"
    if f >= 1e3 and not (f == 1e3 and rng.random() < 0.2):
        return f"{_num(f / 1e3, rng)} кГц"
    return f"{_num(f, rng)} Гц"


def nice_freq(rng, lo=20.0, hi=100e3):
    if rng.random() < 0.2:
        return float(f"{10 ** rng.uniform(math.log10(lo), math.log10(hi)):.3g}")
    while True:
        f = rng.choice(MANTISSAS) * 10 ** rng.randint(1, 4)
        if lo <= f <= hi:
            return float(f"{f:.3g}")


def spec_line(spec):
    """Поля -> строка `.SPEC` в установленном порядке; числа - в записи канона."""
    out = []
    for k in FIELDS:
        if k in spec:
            v = spec[k]
            if isinstance(v, (int, float)):  # частоты - в записи канона (1.5k), Q, K, порядок - числом (0.707)
                v = canon.format_value(v) if k in ("fc", "f0", "bw") else f"{v:g}"
            out.append(f"{k}={v}")
    return ".SPEC " + " ".join(out)


# ---------------------------------------------------------------- проверка решения
def solve(answer):
    """Ответ модели -> отчёт synth.run, если номиналы найдены и цели выполнены с допуском; иначе None."""
    rep = synth.run(answer)
    if "error" in rep:
        return None
    if rep.get("route") == "designer":
        return rep
    if rep["achieved"].get("type") != rep["wanted"]["type"]:
        return None
    for k, d in rep["deviation"].items():
        if abs(d) > (TOL["db"] if k == "k_pass_db" else TOL["rel"]):
            return None
    return rep


def _record(kind, family, prompt, output, spec, unknown, rep, extra=None):
    sol = None if rep is None else {k: rep.get(k) for k in ("route", "topology", "netlist", "achieved", "deviation", "wanted")}
    rec = {"mode": 2, "task": "B", "kind": kind, "family": family, "prompt": prompt, "output": output, "spec": spec,
           "unknown": unknown, "solution": sol,
           "circuit_id": hashlib.sha256(output.encode("utf-8")).hexdigest()[:16]}
    rec.update(extra or {})
    return rec


# ---------------------------------------------------------------- только характеристики
def _target_phrases(spec, rng, with_type=None):
    """Фразы о целях для запроса (без типа): частота, добротность или полоса, усиление."""
    out = []
    if "fc" in spec:
        f = freq_text(spec["fc"], rng)
        out.append(rng.choice([f"срез {f}", f"частота среза {f}", f"на {f}", f"fc = {f}", f"срез на {f}"]))
    if "f0" in spec:
        f = freq_text(spec["f0"], rng)
        out.append(rng.choice([f"центральная частота {f}", f"f0 = {f}", f"на {f}", f"настроен на {f}"]))
    if "Q" in spec:
        out.append(rng.choice([f"добротность {_num(spec['Q'], rng)}", f"Q = {_num(spec['Q'], rng)}"]))
    if "bw" in spec:
        out.append(rng.choice([f"полоса {freq_text(spec['bw'], rng)}", f"ширина полосы {freq_text(spec['bw'], rng)}"]))
    if "K" in spec:
        out.append(rng.choice([f"усиление {_num(spec['K'], rng)}", f"K = {_num(spec['K'], rng)}",
                               f"коэффициент усиления {_num(spec['K'], rng)}"]))
    return out


def _draw_spec(rng):
    t = rng.choices(["LP", "HP", "BP", "BS"], [4, 3, 2, 2])[0]
    spec = {"type": t, ("fc" if t in ("LP", "HP") else "f0"): nice_freq(rng)}
    active = rng.random() < 0.45
    if t in ("LP", "HP"):
        if rng.random() < 0.6:
            spec["order"] = rng.choice([1, 2, 2, 3, 4, 4, 5, 6] if active else ([1, 2, 2, 3, 5] if t == "LP" else [1, 2]))
        if spec.get("order", 2) >= 2 and rng.random() < 0.35:
            spec["response"] = rng.choice(["Butterworth", "Butterworth", "Chebyshev"])
            if spec["response"] == "Chebyshev" and rng.random() < 0.5:
                spec["ripple"] = rng.choice([0.5, 1, 2, 3])
        elif spec.get("order") == 2 and rng.random() < 0.3:
            spec["Q"] = rng.choice([0.5, 0.6, 0.707, 0.8, 1, 1.5, 2])
        if active and spec.get("order", 1) <= 2 and "Q" not in spec and rng.random() < 0.3:
            spec["K"] = rng.choice([2, 3, 5, 10, 20])
    else:
        r = rng.random()
        if r < 0.5:
            spec["Q"] = rng.choice([0.5, 1, 2, 3, 5, 8, 10])
        elif r < 0.7:
            spec["bw"] = float(f"{spec['f0'] / rng.choice([1, 2, 4, 5, 10]):.3g}")
        if active and t == "BP" and rng.random() < 0.3:
            spec["K"] = rng.choice([1, 2, 5])
        if rng.random() < 0.08:
            spec["order"] = 4  # выше 2-го порядка - маршрут дизайнера MC12
    if active:
        spec["active"] = "yes"
    if rng.random() < 0.05:
        spec["series"] = "exact"
    return spec


def spec_prompt(spec, rng):
    """Запрос только с характеристиками. Названо ровно то, что есть в spec."""
    t = spec["type"]
    noun = rng.choice(TYPE_WORDS[t])
    if "response" in spec:
        who = {"Butterworth": ["Баттерворта", "по Баттерворту"], "Chebyshev": ["Чебышёва", "по Чебышёву", "Чебышева"]}[spec["response"]]
        noun += " " + rng.choice(who)
    if "order" in spec:
        n = spec["order"]
        words = ["первого", "второго", "третьего", "четвёртого", "пятого", "шестого"]
        noun += " " + rng.choice([f"{n}-го порядка", f"{words[n - 1]} порядка", f"{n} порядка"])
    tail = _target_phrases(spec, rng)
    if "ripple" in spec:
        tail.append(f"неравномерность {_num(spec['ripple'], rng)} дБ")
    if "series" in spec:
        tail.append(rng.choice(["номиналы точные, без округления к рядам", "без округления номиналов"]))
    act = None
    if "active" in spec:
        act = rng.choice(["активный", "активный", "на ОУ", "на операционном усилителе"])
    elif rng.random() < 0.15:
        act = "пассивный"  # исполнение по умолчанию: в .SPEC не пишется
    if act in ("активный", "пассивный"):
        noun = f"{act} {noun}"
    elif act:
        tail.insert(rng.randint(0, len(tail)), act)
    first = tail.pop(0) if tail[0].startswith("на ") and rng.random() < 0.7 else None
    text = rng.choice(HEADS) + noun + (f" {first}" if first else "")
    return text + (rng.choice([", ", ": ", "; "]) + ", ".join(tail) if tail else "")


def sample_spec(rng, max_tries=200):
    for _ in range(max_tries):
        spec = _draw_spec(rng)
        out = spec_line(spec)
        rep = solve(out)
        if rep is not None:
            return _record("spec", "spec", spec_prompt(spec, rng), out, spec, [], rep)
    raise RuntimeError("запрос по характеристикам не собран")


# ---------------------------------------------------------------- цели из измерений эталона
def _round(x, digits):
    return float(f"{x:.{digits}g}")


def measured_spec(netlist, f_char, rng, want_k=False):
    """Цели по измерению схемы (узловой расчёт): тип, срез или f0, для полосовых и режекторных - Q."""
    m = synth.achieved(netlist, f_char)
    t = m.get("type")
    if t in ("LP", "HP") and m.get("fc"):
        spec = {"type": t, "fc": _round(m["fc"], 3)}
    elif t in ("BP", "BS") and m.get("f0") and m.get("Q"):
        spec = {"type": t, "f0": _round(m["f0"], 3), "Q": _round(m["Q"], 2)}
    else:
        return None
    k = 10 ** (abs(m["k_pass_db"]) / 20) if m.get("k_pass_db") is not None else 1.0
    if want_k and abs(m["k_pass_db"]) > 0.5:
        spec["K"] = _round(10 ** (m["k_pass_db"] / 20), 2)
    return spec if k < 1e3 else None


def _hide_in_text(prompt, n, marker):
    """Заменяет в запросе контрольные номиналы 7,77k7 (k = 1..n) словом marker; каждый должен встретиться один раз."""
    for i in range(1, n + 1):
        prompt, cnt = re.subn(rf"(?:7[.,]77{i}7|777{i}[.,]7){_UNIT}", marker, prompt)
        if cnt != 1:
            raise ValueError(f"контрольный номинал {i} найден {cnt} раз")
    return prompt


def _sentinel(kind, i):
    suffix, mult = _SCALE[kind]
    return f"7.77{i}7{suffix}", 7.7707 * mult + i * 1e-3 * mult


def _with_targets(body, spec, rng, named):
    """Описание схемы + цели. У схемы без названия тип фильтра называется обязательно."""
    tail = _target_phrases(spec, rng)
    word = rng.choice(TYPE_WORDS[spec["type"]])
    lead = rng.choice([f"нужен {word}", f"это {word}", f"должен получиться {word}", word]) if not named or rng.random() < 0.3 else None
    goal = ", ".join(([lead] if lead else []) + tail)
    ask = rng.choice(["", "", "; остальное подобрать", "; подобрать неизвестные номиналы"])
    body = body.rstrip(". ")
    nl = chr(10)
    if nl in body:  # таблица: цели - отдельной строкой
        return f"{goal}{ask}{nl}{body}" if rng.random() < 0.4 else f"{body}{nl}{goal}{ask}"
    return f"{goal}. {body}{ask}" if rng.random() < 0.4 else f"{body}; {goal}{ask}"


# ---------------------------------------------------------------- названная схема
def _named_ref(top, rng):
    for _ in range(50):
        ref = filters_v3.sample_named(top, rng)
        if not {"Ra", "Rb"} & set(ref["params"]):
            return ref
    raise RuntimeError(top)


def sample_named(rng, max_tries=300):
    """Названная схема из таблицы режима 2 (synth.DESIGNS): часть параметров задана, остальные - `?`."""
    for _ in range(max_tries):
        top = rng.choice(list(synth.DESIGNS))
        try:
            ref = _named_ref(top, rng)
        except RuntimeError:
            continue
        params = [p for p in ref["params"] if p != "dc"]
        hide_all = rng.random() < NAMED_ALL_SHARE
        hidden = params if hide_all else rng.sample(params, rng.randint(1, min(3, len(params))))
        build = synth._builder(top)
        base = {k: v for k, v in ref["params"].items()}
        raw = canon.parse_netlist(build(**base)[0])[0]
        other = canon.parse_netlist(build(**dict(base, **{h: base[h] * 1.37 for h in hidden}))[0])[0]
        for i, e in enumerate(raw):
            if e["value"] is not None:
                e["value"] = _Tagged(e["value"])
                e["value"].hidden = e["value"] != other[i]["value"]
        can = canon.canonicalize(raw)
        unknown = [e["name"] for e in can if getattr(e["value"], "hidden", False)]
        netlist = canon.to_netlist([dict(e, value="?") if e["name"] in unknown else e for e in can])
        want_k = top in synth.INVERTING
        if hide_all:  # ничего не задано: цели - круглые, как в запросе по характеристикам
            t = ref["expected"].get("type") or ref.get("analytic", {}).get("type")
            if t not in synth.TYPES:
                continue
            spec = {"type": t, ("fc" if t in ("LP", "HP") else "f0"): nice_freq(rng)}
            if t in ("BP", "BS") and top != "twin_t" and rng.random() < 0.5:
                spec["Q"] = rng.choice([1, 2, 3, 5])
            if want_k and rng.random() < 0.3:
                spec["K"] = rng.choice([1, 2, 5])
        else:
            spec = measured_spec(ref["netlist"], ref["f_char"], rng, want_k)
            if spec is None:
                continue
            if top == "twin_t":
                spec.pop("Q", None)  # у пассивного моста Q постоянно, в запросе его не задают
        output = netlist + spec_line(spec)
        rep = solve(output)
        if rep is None:
            continue
        r = describe_ru.ROLES_ALL[top]
        brief = rng.random() < 0.6
        if hide_all:
            body = rng.choice(r["short_names"] + r["names"])
            family = "named_all"
        else:
            shown = dict(base)
            for i, h in enumerate(hidden, 1):
                shown[h] = _sentinel(h[0], i)[1]
            fake = {"class": top, "params": shown}
            text = describe_ru.describe_a1_brief(fake, rng, 3) if brief else describe_ru.describe_a1(fake, rng, 3)
            try:
                body = _hide_in_text(text, len(hidden), rng.choice(MARKERS))
            except ValueError:
                continue
            family = "named_brief" if brief else "named_roles"
        prompt = _with_targets(body, spec, rng, named=True)
        return _record("named", family, prompt, output, spec, unknown, rep,
                       {"topology": top, "hidden": hidden, "reference": ref["netlist"]})
    raise RuntimeError("названная схема режима 2 не собрана")


# ---------------------------------------------------------------- схема, описанная соединениями
def _circuit_ref(rng):
    r = rng.random()
    if r < 0.45:
        return compose_v3.sample_chain(rng)
    if r < 0.75:
        return compose_v3.sample_addon(rng)
    return filters_v2.sample_any(rng.choice(["rc_ladder2", "rc_ladder3", "lc_ladder3", "rlc_parallel_bp", "rlc_series_bp",
                                              "sallen_key_lp", "sallen_key_hp", "mfb_lp", "mfb_bp"]), rng)


def sample_circuit(rng, max_tries=300):
    """Схема описана соединениями; 1-2 номинала - `?`; тип и частота (у полосовых - и Q) названы."""
    for _ in range(max_tries):
        ref = _circuit_ref(rng)
        spec = measured_spec(ref["netlist"], ref["f_char"], rng)
        if spec is None:
            continue
        elems = canon.parse_netlist(ref["netlist"])[0]
        two = [e["name"] for e in elems if e["type"] in "RCL"]
        hidden = rng.sample(two, min(len(two), rng.choice([1, 1, 1, 2])))
        marked, k = [], 0
        for e in elems:
            if e["name"] in hidden:
                k += 1
                marked.append(dict(e, value=_sentinel(e["type"], k)[0]))
            else:
                marked.append(e)
        family = rng.choice(prompts_v3.FAMILIES)
        try:
            d = prompts_v3.verified(canon.to_netlist(marked), family, rng)
            body = _hide_in_text(d["prompt"], len(hidden), rng.choice(MARKERS))
        except (AssertionError, ValueError):
            continue
        netlist = re.sub(r" 7\.77\d7[knm]$", " ?", d["netlist"], flags=re.M)
        if netlist.count(" ?\n") != len(hidden):
            continue
        output = netlist + spec_line(spec)
        rep = solve(output)
        if rep is None:
            continue
        prompt = _with_targets(body, spec, rng, named=False)
        return _record("circuit", family, prompt, output, spec, hidden, rep,
                       {"topology": ref["class"], "hidden": hidden, "reference": ref["netlist"], "names": d["names"],
                        "shuffled": d["shuffled"]})
    raise RuntimeError("схема режима 2 не собрана")


# ---------------------------------------------------------------- отказы
def sample_refusal(rng):
    if rng.random() < 0.6:  # частота не названа
        spec = _draw_spec(rng)
        spec.pop("fc", None), spec.pop("f0", None), spec.pop("bw", None)
        noun = rng.choice(TYPE_WORDS[spec["type"]])
        extra = _target_phrases({k: v for k, v in spec.items() if k in ("Q", "K")}, rng)
        if "order" in spec:
            extra.insert(0, f"{spec['order']}-го порядка")
        if "active" in spec:
            noun = "активный " + noun
        prompt = rng.choice(HEADS) + noun + (" " + ", ".join(extra) if extra else "")
        code, reason = "ambiguous", REASON_NO_FREQ
    else:
        r = edge_cases.refusal("out_of_class", rng)
        prompt, code, reason = r["prompt"], r["refuse"], r["reason"]
    rec = _record("refuse", "refuse", prompt, edge_cases.refusal_output(code, reason), None, [], None,
                  {"refuse": code, "reason": reason})
    rec["circuit_id"] = None
    return rec


SAMPLERS = {"spec": sample_spec, "named": sample_named, "circuit": sample_circuit, "refuse": sample_refusal}


def sample(rng, kind=None):
    """Один пример режима 2; kind=None - по долям MIX."""
    kind = kind or rng.choices(list(MIX), list(MIX.values()))[0]
    return SAMPLERS[kind](rng)


if __name__ == "__main__":
    import sys
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    for kind in MIX:
        for _ in range(3):
            r = sample(rng, kind)
            print(f"--- {kind} / {r['family']}\n{r['prompt']}\n{r['output']}\n")
