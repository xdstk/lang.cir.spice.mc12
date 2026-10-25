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


def _find_topology(text):
    low = text.lower()
    best = None
    for topo, r in describe_ru.ROLES.items():
        for key, gain in (("names", False), ("short_names", False), ("names_gain", True)):
            for name in r.get(key, []):
                if name.lower() in low and (best is None or len(name) > len(best[1])):
                    best = (topo, name, gain)
    if best is None:
        return None, False, text
    # номиналы ищутся без названия: в «интегратор с Rос: Rвх 1.5к …» само название содержит роль Rос
    i = low.index(best[1].lower())
    return best[0], best[2] or "с усилением" in low, text[:i] + text[i + len(best[1]):]


def _stage_params(text, topo):
    """Номиналы звена по ролям: сначала полные фразы ролей, затем короткие обозначения."""
    r = describe_ru.ROLES[topo]
    params = {}
    for full in (True, False):
        roles = r["roles"] if full else r["short_roles"]
        for param, phrase in sorted(roles.items(), key=lambda kv: -len(kv[1])):
            if param in params:
                continue
            # номинал - до разделителя ролей «, » / «;» / «]» / конца предложения; запятая внутри
            # числа («2,2к») пробела после себя не имеет
            tail = r"(.+?)(?=,\s|;|\]|\.\s|\.$|$)"
            # минимальная общая терпимость: регистр не важен, между ролью и номиналом «—», «=», «:»
            sep = r"\s*[—=:]\s*" if full else r"(?:\s*[=:]\s*|\s+)"
            prefix = "" if full else r"(?<![\w(])"
            m = re.search(prefix + re.escape(phrase) + sep + tail, text, re.IGNORECASE)
            if m:
                v = parse_value(m.group(1), param[0])
                if v is not None:
                    params[param] = v
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
