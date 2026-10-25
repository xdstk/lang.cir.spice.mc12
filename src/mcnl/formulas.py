#!/usr/bin/env python3
"""Топологии L1/L2 и их аналитические характеристики (docs/eval_spec.md, раздел 10).

Для каждой топологии: построение канонического нетлиста по номиналам, формулы {k, fc, f0, Q, f_u},
набор измеряемых величин класса (раздел 6) и характерная частота для развёртки обвязки.
Формулы проверяются численным узловым расчётом (mna.py) и прогоном в MC12 (tests/test_formulas.py).

Обозначения: k_pass_db / k_dc_db - модуль коэффициента передачи в дБ; частоты - Гц.
Источник во всех схемах - V1 IN 0 (значение DC из параметра dc, по умолчанию 0).
"""
import math

try:
    from . import canon
except ImportError:  # запуск как скрипта
    import canon

TWO_PI = 2 * math.pi
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]


def db(x):
    return 20 * math.log10(abs(x))


def fc_2nd_lp(f0, Q):
    """|H| = k/sqrt(2) для звена ФНЧ 2-го порядка; для ФВЧ fc = f0^2 / fc_2nd_lp(f0, Q)."""
    a = 1 - 1 / (2 * Q * Q)
    return f0 * math.sqrt(a + math.sqrt(a * a + 1))


def _v(x):
    return canon.format_value(x)


def _net(lines, dc=0):
    return f"V1 IN 0 DC {_v(dc)}\n" + "".join(l + "\n" for l in lines)


# ---------------------------------------------------------------- топологии
def divider(R1, R2, dc=0):
    return (_net([f"R1 IN OUT {_v(R1)}", f"R2 OUT 0 {_v(R2)}"], dc),
            {"type": "flat", "k_pass_db": db(R2 / (R1 + R2))})


def rc_lp(R, C, dc=0):
    return _net([f"R1 IN OUT {_v(R)}", f"C1 OUT 0 {_v(C)}"], dc), \
        {"type": "LP", "k_pass_db": 0.0, "fc": 1 / (TWO_PI * R * C)}


def rc_hp(R, C, dc=0):
    return _net([f"R1 OUT 0 {_v(R)}", f"C1 IN OUT {_v(C)}"], dc), \
        {"type": "HP", "k_pass_db": 0.0, "fc": 1 / (TWO_PI * R * C)}


def rl_lp(R, L, dc=0):
    return _net([f"R1 OUT 0 {_v(R)}", f"L1 IN OUT {_v(L)}"], dc), \
        {"type": "LP", "k_pass_db": 0.0, "fc": R / (TWO_PI * L)}


def rl_hp(R, L, dc=0):
    return _net([f"R1 IN OUT {_v(R)}", f"L1 OUT 0 {_v(L)}"], dc), \
        {"type": "HP", "k_pass_db": 0.0, "fc": R / (TWO_PI * L)}


def rlc_series_bp(R, L, C, dc=0):
    """Последовательный RLC, выход на R: полосовой, максимум 0 дБ на f0."""
    return _net([f"R1 OUT 0 {_v(R)}", f"C1 N1 OUT {_v(C)}", f"L1 IN N1 {_v(L)}"], dc), \
        {"type": "BP", "k_pass_db": 0.0, "f0": 1 / (TWO_PI * math.sqrt(L * C)), "Q": math.sqrt(L / C) / R}


def inverting(R1, R2, dc=0):
    return _net([f"R1 IN N1 {_v(R1)}", f"R2 N1 OUT {_v(R2)}", "X1 0 N1 OUT OPAMP_IDEAL"], dc), \
        {"type": "flat", "k_pass_db": db(R2 / R1)}


def noninverting(R1, R2, dc=0):
    """R1 - от инвертирующего входа на землю, R2 - обратная связь: K = 1 + R2/R1."""
    return _net([f"R1 N1 0 {_v(R1)}", f"R2 N1 OUT {_v(R2)}", "X1 IN N1 OUT OPAMP_IDEAL"], dc), \
        {"type": "flat", "k_pass_db": db(1 + R2 / R1)}


def active_lp_inv(R1, R2, C, dc=0):
    """Инвертирующий активный ФНЧ 1-го порядка: K = -R2/R1, fc = 1/(2 pi R2 C)."""
    return _net([f"R1 IN N1 {_v(R1)}", f"R2 N1 OUT {_v(R2)}", f"C1 N1 OUT {_v(C)}",
                 "X1 0 N1 OUT OPAMP_IDEAL"], dc), \
        {"type": "LP", "k_pass_db": db(R2 / R1), "fc": 1 / (TWO_PI * R2 * C)}


def integrator_rf(R1, Rf, C, dc=0):
    """Интегратор с Rf (та же схема, что active_lp_inv, но оценивается по k_dc и f_u):
    f_u = fc * sqrt(k^2 - 1), k = Rf/R1 > 1."""
    k, fc = Rf / R1, 1 / (TWO_PI * Rf * C)
    net, _ = active_lp_inv(R1, Rf, C, dc)
    # fc - только для развёртки: от f_u нижний край (f_u/1000) лишь в ~10 раз ниже fc, и k_dc
    # на краю занижен на 0,5% при k = 100
    return net, {"type": "LP", "k_dc_db": db(k), "f_u": fc * math.sqrt(k * k - 1), "fc": fc}


def sallen_key_lp(R1, R2, C1, C2, Ra=None, Rb=None, dc=0):
    """ФНЧ Саллена-Ки: R1 IN-N1, R2 N1-N2, C1 N1-OUT (обратная связь), C2 N2-0; K = 1 + Rb/Ra
    (без Ra, Rb - повторитель, K = 1). w0^2 = 1/(R1 R2 C1 C2),
    w0/Q = 1/(R1 C1) + 1/(R2 C1) + (1 - K)/(R2 C2)."""
    K = 1 + Rb / Ra if Ra else 1.0
    lines = [f"R1 IN N1 {_v(R1)}", f"R2 N1 N2 {_v(R2)}", f"C1 N1 OUT {_v(C1)}", f"C2 N2 0 {_v(C2)}"]
    lines += ([f"R3 N3 0 {_v(Ra)}", f"R4 N3 OUT {_v(Rb)}", "X1 N2 N3 OUT OPAMP_IDEAL"] if Ra
              else ["X1 N2 OUT OUT OPAMP_IDEAL"])
    w0 = 1 / math.sqrt(R1 * R2 * C1 * C2)
    Q = w0 / (1 / (R1 * C1) + 1 / (R2 * C1) + (1 - K) / (R2 * C2))
    f0 = w0 / TWO_PI
    return _net(lines, dc), {"type": "LP", "k_pass_db": db(K), "f0": f0, "Q": Q, "fc": fc_2nd_lp(f0, Q)}


def sallen_key_hp(R1, R2, C1, C2, Ra=None, Rb=None, dc=0):
    """ФВЧ Саллена-Ки: C1 IN-N1, C2 N1-N2, R1 N1-OUT (обратная связь), R2 N2-0; K = 1 + Rb/Ra.
    w0^2 = 1/(R1 R2 C1 C2), w0/Q = (C1 + C2)/(R2 C1 C2) + (1 - K)/(R1 C1)."""
    K = 1 + Rb / Ra if Ra else 1.0
    lines = [f"R1 N1 OUT {_v(R1)}", f"R2 N2 0 {_v(R2)}", f"C1 IN N1 {_v(C1)}", f"C2 N1 N2 {_v(C2)}"]
    lines += ([f"R3 N3 0 {_v(Ra)}", f"R4 N3 OUT {_v(Rb)}", "X1 N2 N3 OUT OPAMP_IDEAL"] if Ra
              else ["X1 N2 OUT OUT OPAMP_IDEAL"])
    w0 = 1 / math.sqrt(R1 * R2 * C1 * C2)
    Q = w0 / ((C1 + C2) / (R2 * C1 * C2) + (1 - K) / (R1 * C1))
    f0 = w0 / TWO_PI
    return _net(lines, dc), {"type": "HP", "k_pass_db": db(K), "f0": f0, "Q": Q, "fc": f0 ** 2 / fc_2nd_lp(f0, Q)}


def mfb_bp(R1, R2, R3, Ca, Cb, dc=0):
    """Полосовой с многопетлевой ОС (MFB): R1 IN-N1, R2 N1-0, Ca N1-OUT, Cb N1-N2, R3 N2-OUT,
    ОУ: + на земле, - в N2. H = -(s/(R1 Cb)) / (s^2 + s (Ca+Cb)/(R3 Ca Cb) + (R1+R2)/(R1 R2 R3 Ca Cb));
    |H(f0)| = R3 Cb / (R1 (Ca + Cb)), Q = w0 R3 Ca Cb / (Ca + Cb)."""
    lines = [f"R1 IN N1 {_v(R1)}", f"R2 N1 0 {_v(R2)}", f"R3 N2 OUT {_v(R3)}", f"C1 N1 OUT {_v(Ca)}",
             f"C2 N1 N2 {_v(Cb)}", "X1 0 N2 OUT OPAMP_IDEAL"]
    w0 = math.sqrt((R1 + R2) / (R1 * R2 * R3 * Ca * Cb))
    return _net(lines, dc), {"type": "BP", "k_pass_db": db(R3 * Cb / (R1 * (Ca + Cb))),
                             "f0": w0 / TWO_PI, "Q": w0 * R3 * Ca * Cb / (Ca + Cb)}


# имя -> (функция, уровень, величины класса для S4, величина-характерная частота)
TOPOLOGIES = {
    "divider": (divider, "L1", ["type", "k_pass_db"], None),
    "rc_lp": (rc_lp, "L1", ["type", "k_pass_db", "fc"], "fc"),
    "rc_hp": (rc_hp, "L1", ["type", "k_pass_db", "fc"], "fc"),
    "rl_lp": (rl_lp, "L1", ["type", "k_pass_db", "fc"], "fc"),
    "rl_hp": (rl_hp, "L1", ["type", "k_pass_db", "fc"], "fc"),
    "rlc_series_bp": (rlc_series_bp, "L1", ["type", "k_pass_db", "f0", "Q"], "f0"),
    "inverting": (inverting, "L2", ["type", "k_pass_db"], None),
    "noninverting": (noninverting, "L2", ["type", "k_pass_db"], None),
    "active_lp_inv": (active_lp_inv, "L2", ["type", "k_pass_db", "fc"], "fc"),
    "integrator_rf": (integrator_rf, "L2", ["type", "k_dc_db", "f_u"], "fc"),
    "sallen_key_lp": (sallen_key_lp, "L2", ["type", "k_pass_db", "f0", "Q", "fc"], "f0"),
    "sallen_key_hp": (sallen_key_hp, "L2", ["type", "k_pass_db", "f0", "Q", "fc"], "f0"),
    "mfb_bp": (mfb_bp, "L2", ["type", "k_pass_db", "f0", "Q"], "f0"),
}


def reference(topology, params, ref_id=None):
    """Эталон без измерений: канонический нетлист, формулы, величины класса, характерная частота."""
    fn, level, quantities, fchar_key = TOPOLOGIES[topology]
    net, expected = fn(**params)
    return {"id": ref_id or topology, "class": topology, "level": level, "params": params,
            "netlist": canon.canonical_netlist(net), "expected": expected, "quantities": quantities,
            "f_char": expected[fchar_key] if fchar_key else None}
