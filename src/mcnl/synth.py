#!/usr/bin/env python3
"""Режим 2: подбор номиналов по характеристикам фильтра (docs/mode2_spec.md).

Модель пишет нетлист, где неизвестные номиналы - `?`, и строку целей `.SPEC`; либо одну строку `.SPEC`.
Номиналы считает этот модуль: свободный элемент (обычно конденсатор) перебирается по ряду, остальные
находятся решением уравнений прямых формул (formulas.py, filters_v2.py) методом Ньютона в логарифмах,
затем всё округляется к ряду с перебором соседних значений. Итог проверяется узловым расчётом (mna.py);
расчёт в MC12 делает вызывающий код, как в режиме 1.

Запуск:  python synth.py ".SPEC type=LP fc=1k active=yes"
         python synth.py файл_с_ответом_модели.txt
"""
import itertools
import json
import math
import re
import sys
from pathlib import Path

try:
    from . import bench, canon, cascade, filters_v2, formulas, iso, measure, mna, sampler
except ImportError:  # запуск как скрипта
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mcnl import bench, canon, cascade, filters_v2, formulas, iso, measure, mna, sampler

TWO_PI = 2 * math.pi
RANGES = sampler.RANGES                      # R 100 Ом...1 МОм, C 100 пФ...10 мкФ, L 1 мкГн...100 мГн
TYPES = ("LP", "HP", "BP", "BS")
RESPONSES = ("Butterworth", "Chebyshev", "Bessel")
DEFAULT_Q_BAND = 1.0                         # ПФ, РФ: полоса равна f0 (решение автора 11.11.2026)
DEFAULT_RIPPLE = 1.0                         # дБ, Чебышёв
MAX_ORDER = 8


# ---------------------------------------------------------------- разбор ответа модели
def parse(text):
    """-> (элементы canon, имена неизвестных, spec, ошибки). Неизвестный номинал - `?`."""
    spec, lines, unknown, errors = {}, [], [], []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith(".SPEC"):
            for tok in line.split()[1:]:
                if "=" not in tok:
                    errors.append(f".SPEC: не пара ключ=значение: {tok}")
                    continue
                k, v = tok.split("=", 1)
                spec[k.strip()] = v.strip()
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[-1] == "?" and parts[0][0].upper() in "RCL":
            unknown.append(parts[0])
            parts[-1] = "1"  # временный номинал: элемент нужен для разбора и топологии
        lines.append(" ".join(parts))
    elems, err = canon.parse_netlist("\n".join(lines) + "\n") if lines else ([], None)
    if err:
        errors += err if isinstance(err, list) else [err]
    return elems, unknown, spec, errors


def resolve_spec(raw):
    """Строка .SPEC (dict строк) -> цели с значениями по умолчанию (docs/mode2_spec.md, раздел 2)."""
    errors, s = [], {}
    t = raw.get("type", "").upper()
    if t not in TYPES:
        errors.append("не задан тип фильтра (type=LP, HP, BP, BS)")
    s["type"] = t
    num = {}
    for k in ("fc", "f0", "Q", "bw", "K", "ripple"):
        if k in raw:
            try:
                num[k] = canon.parse_value(raw[k])
            except (ValueError, TypeError):
                errors.append(f"{k}: не число: {raw[k]}")
    f = num.get("fc") or num.get("f0")
    if not f:
        errors.append("не задана частота (fc или f0)")
    s["f"] = f
    s["response"] = raw.get("response", "Butterworth").capitalize()
    if s["response"] not in RESPONSES:
        errors.append(f"аппроксимация {raw.get('response')} не поддержана")
    s["active"] = raw.get("active", "no").lower() in ("yes", "1", "true", "да")
    named_response = "response" in raw
    if "order" in raw:
        try:
            s["order"] = int(raw["order"])
        except ValueError:
            errors.append(f"order: не число: {raw['order']}")
            s["order"] = 2
    else:
        s["order"] = 2 if (t in ("BP", "BS") or named_response) else 1
    if not 1 <= s.get("order", 1) <= MAX_ORDER:
        errors.append(f"порядок {s['order']} вне 1...{MAX_ORDER}")
    if t in ("BP", "BS"):
        q = num.get("Q") or (f / num["bw"] if num.get("bw") and f else None)
        s["Q"] = q or DEFAULT_Q_BAND
    else:
        s["Q"] = num.get("Q")  # ФНЧ, ФВЧ 2-го порядка: по аппроксимации, если не задан
    s["K"] = abs(num.get("K", 1.0))
    s["ripple"] = num.get("ripple", DEFAULT_RIPPLE)
    s["series"] = raw.get("series", "E24").upper()
    s["explicit"] = {k for k in ("Q", "bw", "K") if k in num}
    return s, errors


# ---------------------------------------------------------------- полюса аппроксимаций
def sections(spec):
    """Звенья по аппроксимации: ([(f0, Q) пар полюсов], f вещественного полюса или None).
    Частота задаёт: Баттерворт - уровень -3 дБ, Чебышёв - границу неравномерности (как у дизайнера MC12)."""
    n, f, typ = spec["order"], spec["f"], spec["type"]
    if spec["response"] == "Bessel":
        raise ValueError("Бессель пока не поддержан")
    if spec["response"] == "Butterworth":
        pairs = [(1.0, 1 / (2 * math.sin((2 * k - 1) * math.pi / (2 * n)))) for k in range(1, n // 2 + 1)]
        real = 1.0 if n % 2 else None
    else:
        eps = math.sqrt(10 ** (spec["ripple"] / 10) - 1)
        a = math.asinh(1 / eps) / n
        pairs = []
        for k in range(1, n // 2 + 1):
            th = (2 * k - 1) * math.pi / (2 * n)
            sig, w = math.sinh(a) * math.sin(th), math.cosh(a) * math.cos(th)
            w0 = math.hypot(sig, w)
            pairs.append((w0, w0 / (2 * sig)))
        real = math.sinh(a) if n % 2 else None
    scale = (lambda w: f * w) if typ == "LP" else (lambda w: f / w)
    return [(scale(w), q) for w, q in pairs], (scale(real) if real else None)


# ---------------------------------------------------------------- ряды
def series(kind, name="E24"):
    if name == "EXACT":
        return None
    lo, hi = RANGES[kind]
    return sampler.series_values(kind, lo, hi)  # R - E24, C и L - E12 (sampler)


def nearest(kind, x, k=1):
    """k ближайших значений ряда снизу и сверху от x (лог-шкала)."""
    vals = series(kind)
    below = [v for v in vals if v <= x * (1 + 1e-9)][-k:]
    above = [v for v in vals if v >= x * (1 - 1e-9)][:k]
    return sorted(set(below + above))


def at_least(kind, x, k=2):
    return [v for v in series(kind) if v >= x * (1 - 1e-9)][:k]


# ---------------------------------------------------------------- топологии
def twin_t_active(R, C, Rk1, Rk2, dc=0):
    """Активный двойной Т-мост: двойной Т (R, R, R/2; C, C, 2C), повторитель на выходе, делитель Rk1-Rk2
    с выхода и второй повторитель на общую точку моста (R/2, 2C). k = Rk2/(Rk1 + Rk2), Q = 1/(4(1 - k))."""
    k = Rk2 / (Rk1 + Rk2)
    v = canon.format_value
    lines = [f"R1 IN N1 {v(R)}", f"R2 N1 N2 {v(R)}", f"C3 N1 N3 {v(2 * C)}", f"C1 IN N4 {v(C)}",
             f"C2 N4 N2 {v(C)}", f"R3 N4 N3 {v(R / 2)}", "X1 N2 OUT OUT OPAMP_IDEAL",
             f"R4 OUT N5 {v(Rk1)}", f"R5 N5 0 {v(Rk2)}", "X2 N5 N3 N3 OPAMP_IDEAL"]
    return formulas._net(lines, dc), {"type": "BS", "k_pass_db": 0.0, "f0": 1 / (TWO_PI * R * C),
                                       "Q": 1 / (4 * (1 - k))}


def _builder(top):
    if top == "twin_t_active":
        return twin_t_active
    if top in filters_v2.TOPOLOGIES_V2:
        return filters_v2.TOPOLOGIES_V2[top][0]
    return formulas.TOPOLOGIES[top][0]


def forward(top, p):
    """Прямая формула: параметры -> {f0, fc, Q, k_pass_db}. Q у последовательного RLC - sqrt(L/C)/R."""
    net, exp = _builder(top)(**p)
    exp = dict(exp)
    if top.startswith("rlc_series") and "Q" not in exp:
        exp["Q"] = math.sqrt(p["L"] / p["C"]) / p["R"]
    if "fc" not in exp and exp.get("type") in ("LP", "HP") and exp.get("f0") and exp.get("Q"):
        r = formulas.fc_2nd_lp(exp["f0"], exp["Q"])  # срез -3 дБ звена 2-го порядка
        exp["fc"] = r if exp["type"] == "LP" else exp["f0"] ** 2 / r
    return net, exp


def _sq_lp(spec):
    return 4 * spec["_Q"] ** 2


# Таблица схем режима 2: какие величины задаются, какой номинал перебирается по ряду (choose), предпочтение
# по сопротивлению (r_pref) - чтобы резисторы были близки к нему; остальные номиналы - решение уравнений.
DESIGNS = {
    "rc_lp": {"targets": ["fc"], "choose": lambda s: ({"C": c} for c in series("C"))},
    "rc_hp": {"targets": ["fc"], "choose": lambda s: ({"C": c} for c in series("C"))},
    "rl_lp": {"targets": ["fc"], "choose": lambda s: ({"L": x} for x in series("L")), "r_pref": 1e3},
    "rl_hp": {"targets": ["fc"], "choose": lambda s: ({"L": x} for x in series("L")), "r_pref": 1e3},
    "rlc_series_lp": {"targets": ["f0", "Q"], "choose": lambda s: ({"L": x} for x in series("L")), "r_pref": 1e3},
    "rlc_series_hp": {"targets": ["f0", "Q"], "choose": lambda s: ({"L": x} for x in series("L")), "r_pref": 1e3},
    "rlc_series_bp": {"targets": ["f0", "Q"], "choose": lambda s: ({"L": x} for x in series("L")), "r_pref": 1e3},
    "rlc_series_bs": {"targets": ["f0", "Q"], "choose": lambda s: ({"L": x} for x in series("L")), "r_pref": 1e3},
    "active_lp_inv": {"targets": ["fc", "k_pass_db"], "choose": lambda s: ({"C": c} for c in series("C"))},
    "active_hp_inv": {"targets": ["fc", "k_pass_db"], "choose": lambda s: ({"C": c} for c in series("C"))},
    "sallen_key_lp": {"targets": ["f0", "Q"],
                      "choose": lambda s: ({"C1": c1, "C2": c2} for c2 in series("C")
                                           for c1 in at_least("C", _sq_lp(s) * c2))},
    "sallen_key_hp": {"targets": ["f0", "Q"], "choose": lambda s: ({"C1": c, "C2": c} for c in series("C"))},
    "mfb_lp": {"targets": ["f0", "Q", "k_pass_db"],
               "choose": lambda s: ({"C1": c1, "C2": c2} for c2 in series("C")
                                    for c1 in at_least("C", _sq_lp(s) * (1 + s["K"]) * c2))},
    "mfb_hp": {"targets": ["f0", "Q", "k_pass_db"],  # K = C1/C3: пара из ряда с отношением, ближайшим к K
               "choose": lambda s: ({"C1": c1, "C2": c1, "C3": c3} for c3 in series("C")
                                    for c1 in ([c3 * s["K"]] if s["series"] == "EXACT" else nearest("C", c3 * s["K"])))},
    "mfb_bp": {"targets": ["f0", "Q", "k_pass_db"], "choose": lambda s: ({"Ca": c, "Cb": c} for c in series("C"))},
    "twin_t": {"targets": ["f0"], "choose": lambda s: ({"C": c} for c in series("C"))},
    "twin_t_active": {"targets": ["f0", "Q"], "choose": lambda s: ({"C": c, "Rk1": 10e3} for c in series("C"))},
}
INVERTING = {"active_lp_inv", "active_hp_inv", "mfb_lp", "mfb_hp", "mfb_bp"}


def params_of(top):
    import inspect
    return [p for p in inspect.signature(_builder(top)).parameters if p not in ("dc", "Ra", "Rb")]


# ---------------------------------------------------------------- решение уравнений
def _residuals(exp, targets):
    out = []
    for k, t in targets.items():
        m = exp.get(k)
        if m is None or (k != "k_pass_db" and m <= 0):
            return None
        out.append((m - t) / 20 * math.log(10) if k == "k_pass_db" else math.log(m / t))
    return out


def _lin_solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def newton(fn, fixed, unknown, targets, guess=None, iters=60, tol=1e-6, h=1e-6):
    """Решение fn(параметры) = цели относительно unknown (в логарифмах). Число неизвестных = числу целей;
    при большем числе целей - Гаусс-Ньютон (наименьшие квадраты). -> параметры или None."""
    names = list(unknown)
    x = [math.log(guess.get(n, 1e4) if guess else 1e4) for n in names]

    def res(xv):
        if any(abs(v) > 60 for v in xv):  # номинал вне 1e-26...1e26 - заведомо не решение
            return None
        p = dict(fixed, **{n: math.exp(v) for n, v in zip(names, xv)})
        try:
            return _residuals(fn(p), targets)
        except (ZeroDivisionError, ValueError, OverflowError):
            return None

    r = res(x)
    if r is None:
        return None
    for _ in range(iters):
        if max(abs(v) for v in r) < tol * 1e-4:
            break
        J = []
        for i in range(len(x)):
            xp = x[:]
            xp[i] += h
            rp = res(xp)
            if rp is None:
                return None
            J.append([(a - b) / h for a, b in zip(rp, r)])
        # J[i][j] = d r_j / d x_i; нормальные уравнения (J J^T) dx = -J r
        A = [[sum(J[i][k] * J[j][k] for k in range(len(r))) for j in range(len(x))] for i in range(len(x))]
        b = [-sum(J[i][k] * r[k] for k in range(len(r))) for i in range(len(x))]
        dx = _lin_solve(A, b)
        if dx is None:
            return None
        step = 1.0
        while step > 1e-4:
            xn = [a + step * d for a, d in zip(x, dx)]
            rn = res(xn)
            if rn is not None and sum(v * v for v in rn) < sum(v * v for v in r):
                x, r = xn, rn
                break
            step /= 2
        else:
            break
    if len(targets) <= len(names) and max(abs(v) for v in r) > tol:
        return None  # при большем числе целей остаток законен (например, K у ФВЧ МОС задают конденсаторы)
    return dict(fixed, **{n: math.exp(v) for n, v in zip(names, x)})


def _in_range(p):
    for k, v in p.items():
        kind = k[0] if k[0] in "RCL" else None
        if kind and not RANGES[kind][0] * 0.999 <= v <= RANGES[kind][1] * 1.001:
            return False
    return True


def _cost(p, r_pref):
    """Близость сопротивлений к предпочтительному (лог), штраф за выход из 1/10...10 от него."""
    rs = [v for k, v in p.items() if k[0] == "R"]
    return sum(abs(math.log10(v / r_pref)) + (1 if abs(math.log10(v / r_pref)) > 1 else 0) for v in rs)


def targets_for(top, spec):
    """Цели в обозначениях прямой формулы схемы."""
    want = DESIGNS[top]["targets"]
    t = {}
    if "fc" in want:
        t["fc"] = spec["f"]
    if "f0" in want:
        t["f0"] = spec["_f0"]
    if "Q" in want:
        t["Q"] = spec["_Q"]
    if "k_pass_db" in want:
        t["k_pass_db"] = 20 * math.log10(spec["K"])
    return t


def _guesses(base, unknown, f, r_pref):
    """Начальные приближения: величина - из частоты и известных номиналов; несколько несимметричных
    вариантов (у Саллен-Ки при R1 = R2 добротность максимальна и якобиан вырожден)."""
    cs = [v for k, v in base.items() if k[0] == "C"]
    ls = [v for k, v in base.items() if k[0] == "L"]
    w = TWO_PI * f
    c0 = math.exp(sum(map(math.log, cs)) / len(cs)) if cs else None
    l0 = math.exp(sum(map(math.log, ls)) / len(ls)) if ls else None
    g = {}
    for n in unknown:
        if n[0] == "R":
            g[n] = 1 / (w * c0) if c0 else (w * l0 if l0 and not cs else r_pref)
        elif n[0] == "C":
            g[n] = 1 / (w * w * l0) if l0 else 1 / (w * r_pref)
        else:
            g[n] = r_pref / w
    out = [g]
    for a, b in ((1 / 3, 3), (3, 1 / 3)):
        out.append({n: v * (a if i % 2 == 0 else b) for i, (n, v) in enumerate(g.items())})
    return out


OPTIONAL_ORDER = ("k_pass_db", "Q")  # что отбрасывать первым, если целей больше, чем неизвестных


def design(top, spec, fixed=None, optional=()):
    """Точный расчёт схемы: перебор свободного номинала по ряду + решение остальных. Из вариантов выбирается
    тот, у которого меньше ошибка после округления к ряду (с точностью 0,1%), затем - резисторы ближе к r_pref.
    optional - цели по умолчанию (не заданные пользователем): отбрасываются, если неизвестных меньше, чем целей
    (часть номиналов задал пользователь, и они уже определяют, например, Q).
    -> (точные параметры, использованные цели)."""
    d = DESIGNS[top]
    fixed = dict(fixed or {})
    targets = targets_for(top, spec)
    names = params_of(top)
    r_pref = d.get("r_pref", 1e4)
    fn = lambda p: forward(top, p)[1]  # noqa: E731
    f = spec.get("_f0") or spec["f"]
    best = None
    for chosen in d["choose"](spec):
        base = dict(chosen, **fixed)  # заданные пользователем номиналы важнее перебора
        unknown = [n for n in names if n not in base]
        if not unknown:
            continue
        tg = dict(targets)
        for k in OPTIONAL_ORDER:
            if len(tg) > len(unknown) and k in tg and k in optional:
                del tg[k]
        p = None
        for guess in _guesses(base, unknown, f, r_pref):
            p = newton(fn, base, unknown, tg, guess)
            if p is not None:
                break
        if p is None or not _in_range(p):
            continue
        # ошибка округления - по тому, что задал пользователь: у ФНЧ, ФВЧ 2-го порядка - по срезу -3 дБ, а не по f0
        et = dict({k: v for k, v in tg.items() if k != "f0"}, fc=spec["f"]) if spec.get("_by_fc") and "f0" in tg else tg
        rnd = round_to_series(top, p, et, keep=set(base))
        err = max(abs(v) for v in _residuals(fn(rnd), et))
        rs = [v for k, v in p.items() if k[0] == "R"]
        outside = sum(1 for v in rs if abs(math.log10(v / r_pref)) > 1)
        key = (outside, round(err, 3), _cost(p, r_pref))
        if best is None or key < best[0]:
            best = (key, p, et)
        if fixed and set(chosen) <= set(fixed):
            break  # перебирать нечего: свободный номинал задан пользователем
    if best is None:
        hint = (" с заданными номиналами - проверьте их или задайте меньше" if fixed else
                " - для низких частот попробуйте активный фильтр (active=yes)")
        raise ValueError(f"{top}: номиналы в пределах R {_fmt_range('R')}, C {_fmt_range('C')}, L {_fmt_range('L')} "
                         "не найдены" + hint)
    return best[1], best[2]


def _fmt_range(kind):
    lo, hi = RANGES[kind]
    return f"{canon.format_value(lo)}...{canon.format_value(hi)}"


def round_to_series(top, p, targets, keep=()):
    """Округление к ряду (R - E24, C, L - E12): перебор двух соседних значений у каждого номинала,
    минимум наибольшей относительной ошибки целей. keep - номиналы, которые не трогать (заданы пользователем)."""
    names = [n for n in p if n[0] in "RCL" and n not in keep]
    options = [nearest(n[0], p[n]) or [p[n]] for n in names]
    best = None
    for combo in itertools.product(*options):
        q = dict(p, **dict(zip(names, combo)))
        try:
            exp = forward(top, q)[1]
        except (ZeroDivisionError, ValueError):
            continue
        r = _residuals(exp, targets)
        if r is None:
            continue
        err = max(abs(v) for v in r)
        if best is None or err < best[0]:
            best = (err, q)
    return best[1] if best else p


# ---------------------------------------------------------------- выбор схемы по характеристикам
def default_topology(spec):
    """-> ('single', схема) | ('cascade', None) | ('ladder', None) | ('designer', причина); ValueError - не поддержано.
    Раздел 3 docs/mode2_spec.md: по умолчанию пассивный, активный - по запросу."""
    t, n, act = spec["type"], spec["order"], spec["active"]
    if t in ("BP", "BS"):
        if n > 2:
            return "designer", "ПФ и РФ выше 2-го порядка - дизайнер MC12"
        return "single", {"BP": "mfb_bp" if act else "rlc_series_bp", "BS": "twin_t_active" if act else "rlc_series_bs"}[t]
    if n == 1:
        if act:
            return "single", "active_lp_inv" if t == "LP" else "active_hp_inv"
        return "single", "rc_lp" if t == "LP" else "rc_hp"
    if n == 2:
        if act:
            sk = "sallen_key_lp" if t == "LP" else "sallen_key_hp"
            return "single", sk if spec["K"] == 1 else ("mfb_lp" if t == "LP" else "mfb_hp")
        return "single", "rlc_series_lp" if t == "LP" else "rlc_series_hp"
    if act:
        return "cascade", None
    if t == "LP" and n in (3, 5):
        return "ladder", None
    raise ValueError(f"пассивный {'ФНЧ' if t == 'LP' else 'ФВЧ'} {n}-го порядка не поддержан: есть пассивные ФНЧ "
                     "3-го и 5-го порядка (LC-лестница) или активный (active=yes)")


def _second_order(spec, f0, q):
    s = dict(spec, _f0=f0, _Q=q)
    return s


def _prep_single(top, spec):
    """f0 и Q для схемы 2-го порядка из частоты среза и аппроксимации."""
    s = dict(spec)
    if top in ("sallen_key_lp", "sallen_key_hp", "rlc_series_lp", "rlc_series_hp", "mfb_lp", "mfb_hp"):
        if spec["Q"] and "Q" in spec["explicit"]:
            q = spec["Q"]
            r = formulas.fc_2nd_lp(1.0, q)  # частота -3 дБ звена с f0 = 1
            f0 = spec["f"] / r if spec["type"] == "LP" else spec["f"] * r
        else:
            (f0, q), = sections(dict(spec, order=2))[0]
        # частота задана как срез -3 дБ (у Чебышёва - граница неравномерности: там ориентир - f0 и Q звена)
        s.update(_f0=f0, _Q=q, _by_fc=spec["response"] != "Chebyshev" or "Q" in spec["explicit"])
    else:
        s.update(_f0=spec["f"], _Q=spec["Q"] or DEFAULT_Q_BAND)
    return s


def _ladder(spec):
    """Пассивный ФНЧ 3-го или 5-го порядка: LC-лестница с равными нагрузками, g - Баттерворт или Чебышёв
    (у Чебышёва при равных нагрузках - только нечётный порядок)."""
    n, wc = spec["order"], TWO_PI * spec["f"]
    a = [math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)]
    if spec["response"] == "Butterworth":
        g = [2 * x for x in a]
    else:
        beta = math.log(1 / math.tanh(spec["ripple"] / 17.37))
        gam = math.sinh(beta / (2 * n))
        b = [gam ** 2 + math.sin(k * math.pi / n) ** 2 for k in range(1, n + 1)]
        g = [2 * a[0] / gam]
        for k in range(1, n):
            g.append(4 * a[k - 1] * a[k] / (b[k - 1] * g[k - 1]))
    top = "lc_ladder3" if n == 3 else "lc_ladder5"
    # ожидаемый срез -3 дБ (от уровня в полосе): у Чебышёва выше границы неравномерности
    eps = math.sqrt(10 ** (spec["ripple"] / 10) - 1)
    f3 = spec["f"] * (1.0 if spec["response"] == "Butterworth" else math.cosh(math.acosh(1 / eps) / n))
    best = None
    for R in series("R"):  # R нагрузок свободен: выбирается так, чтобы после округления L, C срез ушёл меньше всего
        if not 100 <= R <= 10e3:
            continue
        vals = [(g[k] * R / wc) if k % 2 == 0 else (g[k] / (R * wc)) for k in range(n)]  # L, C, L, ...
        p = {"Rs": R, "Rl": R}
        for k, v in enumerate(vals):
            p[("L" if k % 2 == 0 else "C") + str(k // 2 + 1)] = v
        if not _in_range(p):
            continue
        q = {k: min(nearest(k[0], v), key=lambda x: abs(math.log(x / v))) if k[0] in "LC" else v for k, v in p.items()}
        got = achieved(forward(top, q)[0], spec["f"]).get("fc")
        if not got:
            continue
        key = (round(abs(got / f3 - 1), 3), abs(math.log10(R / 1e3)))
        if best is None or key < best[0]:
            best = (key, p, q)
    if best is None:
        raise ValueError("LC-лестница: номиналы вне диапазонов")
    return top, best[1], best[2]


def _cascade(spec):
    """Активный ФНЧ, ФВЧ порядка >= 3: звенья Саллен-Ки (K = 1) и, при нечётном порядке, RC-звено в конце
    (выход ОУ - идеальный источник, RC на выходе никого не нагружает)."""
    pairs, real = sections(spec)
    sk = "sallen_key_lp" if spec["type"] == "LP" else "sallen_key_hp"
    stages = []
    for f0, q in pairs:
        s = dict(spec, _f0=f0, _Q=q)
        p, t = design(sk, s)
        stages.append((sk, p, round_to_series(sk, p, t), t))
    if real:
        rc = "rc_lp" if spec["type"] == "LP" else "rc_hp"
        s = dict(spec, f=real)
        p, t = design(rc, s)
        stages.append((rc, p, round_to_series(rc, p, t), t))
    return stages


# ---------------------------------------------------------------- проверка узловым расчётом
def achieved(netlist, f_char):
    elems = canon.parse_netlist(netlist)[0]
    lo, hi = bench.ac_range(f_char)
    return measure.measure(*mna.sweep(elems, lo, hi, per_decade=bench.POINTS_PER_DECADE))


def _report(route, top, exact_net, net, spec, notes, used=None):
    """used - цели, по которым считались номиналы (у схемы пользователя часть целей по умолчанию отброшена)."""
    got = achieved(net, spec["f"])
    want = {"type": spec["type"]}
    if spec["type"] in ("LP", "HP"):
        want["fc"] = spec["f"]
        if spec["response"] == "Chebyshev" and "Q" not in spec["explicit"]:  # срез -3 дБ выше границы неравномерности
            eps = math.sqrt(10 ** (spec["ripple"] / 10) - 1)
            r = math.cosh(math.acosh(1 / eps) / spec["order"])
            want["fc"] = spec["f"] * r if spec["type"] == "LP" else spec["f"] / r
    else:
        want["f0"] = spec["f"]
        if used is None or "Q" in used:
            want["Q"] = spec["Q"]
    if used is not None and "k_pass_db" in used or used is None and ("K" in spec["explicit"] or top in INVERTING):
        want["k_pass_db"] = 20 * math.log10(spec["K"]) if route != "ladder" else None
    if top in ("twin_t", "twin_t_active"):
        notes.append("2C и R/2 двойного Т-моста - по два одинаковых элемента параллельно")
    if top == "twin_t":
        want.pop("Q", None)
        notes.append("у пассивного двойного Т-моста Q = 0,25 при любых номиналах; Q задаётся в активном (active=yes)")
    dev = {}
    for k, v in want.items():
        if k == "type" or v is None or got.get(k) is None:
            continue
        dev[k] = got[k] - v if k == "k_pass_db" else got[k] / v - 1
    if route == "ladder":
        notes.append("LC-лестница с равными нагрузками: в полосе пропускания -6 дБ (делитель Rs, Rl)")
    if spec["response"] == "Chebyshev":
        notes.append("Чебышёв: частота - граница неравномерности, срез по -3 дБ выше (fc в «получено»)")
    return {"route": route, "topology": top, "netlist_exact": exact_net, "netlist": net, "spec": _clean(spec),
            "wanted": want, "achieved": {k: got.get(k) for k in ("type", "fc", "f0", "Q", "k_pass_db")},
            "deviation": dev, "notes": notes}


def _clean(spec):
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in spec.items() if not k.startswith("_")}


# ---------------------------------------------------------------- схема пользователя с `?`
def _marker_template(top):
    """Схема с различимыми номиналами: по номиналу элемента восстанавливается параметр."""
    names = params_of(top)
    marks = {n: (i + 2) * {"R": 1e3, "C": 1e-9, "L": 1e-3}[n[0]] for i, n in enumerate(names)}
    if top == "twin_t_active":
        marks["Rk1"], marks["Rk2"] = 13e3, 17e3
    net = _builder(top)(**marks)[0]
    elems = canon.parse_netlist(net)[0]
    by_value = {}
    for n, v in marks.items():
        by_value.setdefault(round(math.log10(v), 6), n)
    return elems, by_value


def identify(elems):
    """Схема пользователя -> (топология, {параметр: индекс элемента}, {индекс шаблона: индекс элемента})
    или (None, None, None). Шаблон и прямая формула строят элементы в одном порядке."""
    for top in DESIGNS:
        tmpl, by_value = _marker_template(top)
        if len(tmpl) != len(elems):
            continue
        m = iso.match(tmpl, elems, "topology")
        if not m["match"]:
            continue
        roles = {}
        for i_t, i_u in m["mapping"].items():
            e = tmpl[i_t]
            if e["type"] in "RCL":
                key = round(math.log10(canon.parse_value(e["value"])), 6)
                if key in by_value:
                    roles.setdefault(by_value[key], i_u)
        if set(roles) == set(params_of(top)):
            return top, roles, m["mapping"]
    return None, None, None


def _generic(elems, unknown, spec, notes):
    """Схема вне таблицы: Гаусс-Ньютон по узловому расчёту; лишние неизвестные - типовые номиналы."""
    targets = {("fc" if spec["type"] in ("LP", "HP") else "f0"): spec["f"]}
    if spec["type"] in ("BP", "BS") or "Q" in spec["explicit"]:
        targets["Q"] = spec["Q"]
    if "K" in spec["explicit"]:
        targets["k_pass_db"] = 20 * math.log10(spec["K"])
    typical = {"R": 10e3, "C": 10e-9, "L": 1e-3}
    free = list(unknown)
    fixed = {}
    while len(free) > len(targets):  # лишние: сначала конденсаторы, потом катушки
        extra = sorted(free, key=lambda n: "CLR".index(n[0]))[0]
        free.remove(extra)
        fixed[extra] = typical[extra[0]]
        notes.append(f"{extra} = {canon.format_value(typical[extra[0]])}: целей меньше, чем неизвестных")
    if len(free) < len(targets):
        notes.append("целей больше, чем неизвестных: подбор по наименьшим квадратам")

    def fn(p):
        es = [dict(e, value=canon.format_value(p[e["name"]])) if e["name"] in p else e for e in elems]
        got = achieved(canon.to_netlist(es), spec["f"])
        return got

    p = newton(fn, fixed, free, targets, {n: typical[n[0]] for n in free}, iters=30, tol=1e-3, h=1e-3)
    if p is None:
        raise ValueError("номиналы для этой схемы и характеристик не найдены")
    exact = canon.to_netlist([dict(e, value=canon.format_value(p[e["name"]])) if e["name"] in p else e for e in elems])
    if spec["series"] == "EXACT":
        return exact, exact
    best = None  # округление: перебор двух соседних значений ряда у каждого найденного номинала
    for combo in itertools.product(*[nearest(n[0], p[n]) or [p[n]] for n in free]):
        q = dict(fixed, **dict(zip(free, combo)))
        r = _residuals(fn(q), targets)
        if r is not None and (best is None or max(map(abs, r)) < best[0]):
            best = (max(map(abs, r)), q)
    q = best[1] if best else p
    rounded = canon.to_netlist([dict(e, value=canon.format_value(q[e["name"]])) if e["name"] in q else e for e in elems])
    return exact, rounded


def from_netlist(elems, unknown, spec):
    """Режим 2 со схемой пользователя: известная схема - формулы, иначе численный подбор."""
    notes = []
    if not unknown:  # все номиналы заданы: только проверка характеристик
        notes.append("неизвестных номиналов нет: схема проверена на заданные характеристики")
        net = canon.to_netlist(elems)
        return _report("check", None, net, net, spec, notes)
    top, roles, mapping = identify(elems)
    if top:
        user = {n: canon.parse_value(elems[i]["value"]) for n, i in roles.items() if elems[i]["name"] not in unknown}
        s = _prep_single(top, spec)
        optional = {k for k, e in (("Q", "Q"), ("k_pass_db", "K")) if e not in spec["explicit"]}
        p, t = design(top, s, fixed=user, optional=optional)
        q = round_to_series(top, p, t, keep=set(user)) if spec["series"] != "EXACT" else p

        def fill(params):
            """Номиналы спроектированной схемы -> на места `?` схемы пользователя (и производные: R/2, 2C)."""
            designed = canon.parse_netlist(forward(top, params)[0])[0]
            vals = {elems[i_u]["name"]: designed[i_t]["value"] for i_t, i_u in mapping.items()
                    if elems[i_u]["name"] in unknown}
            return canon.to_netlist([dict(e, value=vals[e["name"]]) if e["name"] in vals else e for e in elems])

        return _report("formula", top, fill(p), fill(q), spec, notes, used=t)
    exact_net, net = _generic(elems, unknown, spec, notes)
    return _report("numeric", None, exact_net, net, spec, notes)


# ---------------------------------------------------------------- вход
def run(text):
    """Ответ модели режима 2 -> отчёт: схема с номиналами, цели, полученное, отклонения, замечания."""
    elems, unknown, raw, errors = parse(text)
    if errors:
        return {"error": "; ".join(errors)}
    spec, errors = resolve_spec(raw)
    if errors:
        return {"error": "; ".join(errors)}
    try:
        if [e for e in elems if e["type"] != "V"]:
            return from_netlist(elems, unknown, spec)
        route, top = default_topology(spec)
        if route == "designer":
            return {"route": "designer", "reason": top, "spec": _clean(spec)}
        if route == "single":
            s = _prep_single(top, spec)
            p, t = design(top, s)
            q = round_to_series(top, p, t) if spec["series"] != "EXACT" else p
            notes = []
            return _report("single", top, forward(top, p)[0], forward(top, q)[0], spec, notes, used=t)
        if route == "ladder":
            top, p, q = _ladder(spec)
            return _report("ladder", top, forward(top, p)[0], forward(top, q if spec["series"] != "EXACT" else p)[0], spec, [])
        stages = _cascade(spec)
        exact = cascade.compose([{"netlist": forward(t, p)[0]} for t, p, _, _ in stages])
        net = exact if spec["series"] == "EXACT" else \
            cascade.compose([{"netlist": forward(t, q)[0]} for t, _, q, _ in stages])
        return _report("cascade", "+".join(t for t, *_ in stages), exact, net, spec, [])
    except ValueError as e:
        return {"error": str(e)}


def main(argv):
    arg = argv[1] if len(argv) > 1 else sys.stdin.read()
    text = Path(arg).read_text(encoding="utf-8") if Path(arg).is_file() else arg.replace("\\n", "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run(text), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv)
