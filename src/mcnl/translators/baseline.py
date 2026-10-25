"""Baseline на правилах: запрос на русском -> канонический нетлист без машинного обучения.

Задача A: разбор описанных элементов (describe_ru.parse_brief, затем parse_a - полная запись).
Задача A′: топология узнаётся по названию из data/roles_ru.json (самое длинное совпадение), номиналы -
по фразам ролей («резистор обратной связи — 47 кОм» или «Rос 47к»), схема собирается функцией
топологии из formulas.py; каскады («Каскад из N звеньев … 1) …; 2) …» или «каскад: […] → […]») -
по звеньям через cascade.compose.

Ограничение (важно для выводов): правила написаны под формулировки генератора, поэтому на
синтетических выборках baseline близок к 100% по построению. Содержательная оценка - на запросах,
написанных людьми (человеческий тест).
"""
import re

from .. import canon, cascade, describe_ru, formulas
from . import Translator

_NUM = r"(-?\d+(?:[.,]\d+)?)"
# множители записи номиналов: полная («4,7 кОм», «0,1 мкФ») и краткая («4,7к», «100н», «1М»)
_MULT = [("мегаом", 1e6), ("килоом", 1e3), ("МОм", 1e6), ("кОм", 1e3), ("Ом", 1.0),
         ("мкФ", 1e-6), ("нФ", 1e-9), ("пФ", 1e-12), ("Ф", 1.0), ("мкГн", 1e-6), ("мГн", 1e-3), ("Гн", 1.0),
         ("мк", 1e-6), ("к", 1e3), ("М", 1e6), ("н", 1e-9), ("п", 1e-12), ("", 1.0)]
_UNIT_RE = "|".join(re.escape(u) for u, _ in _MULT if u)
VALUE_RE = re.compile(_NUM + r"\s*(" + _UNIT_RE + r")?(?![а-яёА-ЯЁ\w])")


def parse_value(text, kind):
    """Первое число с единицей в тексте -> значение в СИ (kind - R, C, L: для «1М» у R - мегаом)."""
    m = VALUE_RE.search(text)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    unit = m.group(2) or ""
    if kind == "C" and unit in ("", "к", "М"):
        return None  # у ёмкости единица обязательна
    if kind == "L" and unit not in ("мкГн", "мГн", "Гн"):
        return None
    return v * dict(_MULT)[unit]


def _source_dc(text):
    if "сигнал" in text:
        return 0.0
    m = re.search(_NUM + r"\s*В(?![а-яё])", text)
    return float(m.group(1).replace(",", ".")) if m else 0.0


def _norm(s):
    """Нормализация для сравнения: регистр, «ё» -> «е», разные тире в названиях -> «-».
    Замены посимвольные: позиции в нормализованном тексте совпадают с исходными."""
    return s.lower().replace("ё", "е").replace("–", "-").replace("‑", "-")


def _topology_names(r):
    """(название, признак варианта с усилением) - основные, краткие, с усилением и синонимы."""
    for key, gain in (("names", False), ("short_names", False), ("names_gain", True)):
        for name in r.get(key, []):
            yield name, gain
    for name in r.get("aliases", {}).get("names", []):
        yield name, False


def _find_topology(text):
    low = _norm(text)
    best = None
    for topo, r in describe_ru.ROLES.items():
        for name, gain in _topology_names(r):
            if _norm(name) in low and (best is None or len(name) > len(best[1])):
                best = (topo, name, gain)
    if best is None:
        return None, False, text
    # номиналы ищутся без названия: в «интегратор с Rос: Rвх 1.5к …» само название содержит роль Rос
    i = low.index(_norm(best[1]))
    return best[0], best[2] or "с усилением" in low, text[:i] + text[i + len(best[1]):]


def _role_phrases(r):
    """(параметр, обозначение роли): полные фразы, краткие обозначения и синонимы."""
    for param, phrase in r["roles"].items():
        yield param, phrase
    for param, phrase in r.get("short_roles", {}).items():
        yield param, phrase
    for param, phrases in r.get("aliases", {}).get("roles", {}).items():
        for phrase in phrases:
            yield param, phrase


def _stage_params(text, topo):
    """Номиналы звена по ролям. Обозначения - от длинных к коротким («резистор обратной связи» раньше
    «резистор»); фрагмент текста, уже давший номинал одной роли, другой роли не отдаётся."""
    low = _norm(text)
    params, used = {}, []
    # номинал - до разделителя ролей «, » / «;» / «]» / конца предложения; запятая внутри числа
    # («2,2к») пробела после себя не имеет. Между ролью и номиналом: «—», «=», «:» или пробел
    tail = r"(.+?)(?=,\s|;|\]|\.\s|\.$|$)"
    sep = r"(?:\s*[—=:]\s*|\s+)"
    for param, phrase in sorted(_role_phrases(describe_ru.ROLES[topo]), key=lambda pp: -len(pp[1])):
        if param in params:
            continue
        for m in re.finditer(r"(?<![\w(])" + re.escape(_norm(phrase)) + sep + tail, low):
            if any(m.start() < b and a < m.end() for a, b in used):
                continue
            v = parse_value(text[m.start(1):m.end(1)], param[0])
            if v is not None:
                params[param] = v
                used.append((m.start(), m.end()))
                break
    return params


def _build_stage(text):
    topo, gain, rest = _find_topology(text)
    if topo is None:
        raise ValueError("топология не узнана")
    params = _stage_params(rest, topo)
    fn = formulas.TOPOLOGIES[topo][0]
    import inspect
    sig = inspect.signature(fn).parameters
    need = [p for p, d in sig.items() if p != "dc" and d.default is inspect.Parameter.empty]
    if gain:
        need += [p for p in ("Ra", "Rb") if p in sig]
    missing = [p for p in need if p not in params]
    if missing:
        raise ValueError(f"{topo}: не найдены номиналы {missing}")
    params = {k: v for k, v in params.items() if k in need}
    return {"class": topo, "netlist": canon.canonical_netlist(fn(**params)[0])}


def parse_a1(prompt):
    dc = _source_dc(prompt)
    low = prompt.lower()
    if low.startswith("каскад"):
        if "→" in prompt:
            stages_txt = re.findall(r"\[([^\]]+)\]", prompt)
        else:
            body = prompt.split(":", 1)[1]
            stages_txt = [s for s in re.split(r"\s*\d\)\s*", body) if s.strip()]
            stages_txt[-1] = re.split(r"\.\s", stages_txt[-1])[0]
        stages = [_build_stage(s) for s in stages_txt]
        net = cascade.compose(stages)
    else:
        net = _build_stage(prompt)["netlist"]
    elems = canon.parse_netlist(net)[0]
    for e in elems:
        if e["type"] == "V":
            e["value"] = canon.format_value(dc)
    return canon.to_netlist(canon.canonicalize(elems))


class RuleBaseline(Translator):
    name = "baseline"
    output_format = "netlist"

    def translate(self, prompt):
        for parse in (lambda p: describe_ru.parse_a(p, "brief"), describe_ru.parse_a):
            try:
                elems = parse(prompt)
                if len(elems) > 1:
                    return canon.to_netlist(canon.canonicalize(elems))
            except (ValueError, KeyError):
                pass
        try:
            return parse_a1(prompt)
        except (ValueError, KeyError, IndexError, TypeError):
            return ""  # не разобрано: ответ пустой, оценка - S0 = нет
