"""Схемы статей справочника, разложенные вручную (schematic.py, по ГОСТ; решение автора 17.11.2026).

Координаты - мм, y вниз, сетка 2,5 мм. Пассивные цепи - четырёхполюсником: выводы Uвх, Uвых и общий провод снизу;
схемы с ОУ - выводы Uвх, Uвых и земля. Обозначения - как в нетлисте статьи: тест сверяет каждый рисунок с ним
(schematic.same_circuit). Номиналов нет: схема в общем виде.
"""
from mcnl.schematic import Sketch


def _two_port(s, x_out, h):
    """Выводы четырёхполюсника: Uвх (0, 0), Uвых (x_out, 0), общий провод на высоте h."""
    s.terminal((0, 0), "IN", "Uвх")
    s.terminal((x_out, 0), "OUT", "Uвых", side="right")
    s.rail(h, 0, x_out)


def _shunt(s, kind, name, x, h, side="right"):
    """Элемент с верхней линии на общий провод, с точками соединения."""
    s.two(kind, name, (x, 0), (x, h), side)
    s.dot((x, 0))
    s.dot((x, h))


# ---------------------------------------------------------------- пассивные RC и RL
def divider():
    s = Sketch()
    _two_port(s, 40, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).wire((20, 0), (40, 0))
    _shunt(s, "R", "R2", 30, 20)
    return s


def rc_lp():
    s = Sketch()
    _two_port(s, 40, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).wire((20, 0), (40, 0))
    _shunt(s, "C", "C1", 30, 20)
    return s


def rc_hp():
    s = Sketch()
    _two_port(s, 35, 20)
    s.wire((0, 0), (5, 0)).C("C1", (5, 0), (15, 0)).wire((15, 0), (35, 0))
    _shunt(s, "R", "R1", 25, 20)
    return s


def rl_lp():
    s = Sketch()
    _two_port(s, 42.5, 20)
    s.wire((0, 0), (5, 0)).L("L1", (5, 0), (22.5, 0)).wire((22.5, 0), (42.5, 0))
    _shunt(s, "R", "R1", 32.5, 20)
    return s


def rl_hp():
    s = Sketch()
    _two_port(s, 40, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).wire((20, 0), (40, 0))
    _shunt(s, "L", "L1", 30, 20)
    return s


def rc_ladder2():
    s = Sketch()
    _two_port(s, 60, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).wire((20, 0), (30, 0))
    s.R("R2", (30, 0), (45, 0)).wire((45, 0), (60, 0))
    _shunt(s, "C", "C1", 25, 20)
    _shunt(s, "C", "C2", 50, 20)
    return s


def rc_ladder3():
    s = Sketch()
    _two_port(s, 85, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).wire((20, 0), (30, 0))
    s.R("R2", (30, 0), (45, 0)).wire((45, 0), (55, 0))
    s.R("R3", (55, 0), (70, 0)).wire((70, 0), (85, 0))
    _shunt(s, "C", "C1", 25, 20)
    _shunt(s, "C", "C2", 50, 20)
    _shunt(s, "C", "C3", 75, 20)
    return s


def twin_t():
    """Ветвь C1-C2 (R3 на землю) сверху, ветвь R1-R2 (C3 на землю) снизу; R3 уходит на землю вверх."""
    s = Sketch()
    s.terminal((0, 12.5), "IN", "Uвх")
    s.terminal((65, 12.5), "OUT", "Uвых", side="right")
    s.rail(40, 0, 65)
    s.wire((0, 12.5), (5, 12.5)).wire((5, 0), (5, 25)).dot((5, 12.5))
    s.wire((60, 12.5), (65, 12.5)).wire((55, 0), (55, 25)).wire((55, 12.5), (60, 12.5)).dot((55, 12.5))
    s.wire((5, 0), (10, 0)).C("C1", (10, 0), (20, 0)).wire((20, 0), (40, 0))
    s.C("C2", (40, 0), (50, 0)).wire((50, 0), (55, 0))
    s.R("R3", (30, 0), (30, -15), side="right").dot((30, 0))
    s.ground((30, -15), side="up")
    s.R("R1", (5, 25), (20, 25)).wire((20, 25), (40, 25)).R("R2", (40, 25), (55, 25))
    s.C("C3", (30, 25), (30, 40), side="right").dot((30, 25)).dot((30, 40))
    return s


# ---------------------------------------------------------------- LC
def rlc_series():
    s = Sketch()
    _two_port(s, 50, 20)
    s.wire((0, 0), (5, 0)).L("L1", (5, 0), (22.5, 0)).C("C1", (22.5, 0), (32.5, 0)).wire((32.5, 0), (50, 0))
    _shunt(s, "R", "R1", 40, 20)
    return s


def rlc_parallel_bp():
    s = Sketch()
    _two_port(s, 50, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).wire((20, 0), (50, 0))
    _shunt(s, "L", "L1", 30, 20, side="left")
    _shunt(s, "C", "C1", 40, 20)
    return s


def lc_ladder3():
    s = Sketch()
    _two_port(s, 80, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).L("L1", (20, 0), (37.5, 0)).wire((37.5, 0), (47.5, 0))
    s.L("L2", (47.5, 0), (65, 0)).wire((65, 0), (80, 0))
    _shunt(s, "C", "C1", 42.5, 20)
    _shunt(s, "R", "R2", 72.5, 20)
    return s


def lc_ladder5():
    s = Sketch()
    _two_port(s, 107.5, 20)
    s.wire((0, 0), (5, 0)).R("R1", (5, 0), (20, 0)).L("L1", (20, 0), (37.5, 0)).wire((37.5, 0), (47.5, 0))
    s.L("L2", (47.5, 0), (65, 0)).wire((65, 0), (75, 0))
    s.L("L3", (75, 0), (92.5, 0)).wire((92.5, 0), (107.5, 0))
    _shunt(s, "C", "C1", 42.5, 20)
    _shunt(s, "C", "C2", 70, 20)
    _shunt(s, "R", "R2", 100, 20)
    return s


# ---------------------------------------------------------------- инвертирующие схемы на одном ОУ
def _inverting(s, x_in_end, feedback, power=False, label="above"):
    """ОУ с инвертирующим входом сверху, неинвертирующий - на земле; вход ОУ - x_in_end (узел инв. входа - на 2,5 мм
    левее вывода); feedback - [(вид, имя, y)] - ветви обратной связи от узла инв. входа к выходу. -> выводы ОУ."""
    x = x_in_end + 5
    p = s.opamp("X1", x, 0, power=power, label=label)
    node = (x_in_end - 2.5, 5)
    s.wire(node, p["minus"])
    s.ground((p["plus"][0] - 2.5, p["plus"][1]))
    s.wire(p["plus"], (p["plus"][0] - 2.5, p["plus"][1]))
    xr = p["out"][0] + 5
    top = min(y for _, _, y in feedback)
    s.wire(node, (node[0], top)).dot(node)
    s.wire((xr, top), (xr, p["out"][1])).dot((xr, p["out"][1]))
    s.wire(p["out"], (xr + 7.5, p["out"][1]))
    s.terminal((xr + 7.5, p["out"][1]), "OUT", "Uвых", side="right")
    for kind, name, y in feedback:
        mid = (node[0] + xr) / 2
        half = {"R": 7.5, "C": 5, "L": 8.75}[kind]
        s.wire((node[0], y), (mid - half, y)).two(kind, name, (mid - half, y), (mid + half, y)).wire((mid + half, y), (xr, y))
        if y != top:
            s.dot((node[0], y)).dot((xr, y))
    return p


def inverting():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    s.wire((0, 5), (2.5, 5)).R("R1", (2.5, 5), (17.5, 5)).wire((17.5, 5), (20, 5))
    _inverting(s, 22.5, [("R", "R2", -10)])
    return s


def tl071():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    s.wire((0, 5), (2.5, 5)).R("R1", (2.5, 5), (17.5, 5)).wire((17.5, 5), (20, 5))
    _inverting(s, 22.5, [("R", "R2", -12.5)], power=True, label="right")
    return s


def integrator():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    s.wire((0, 5), (2.5, 5)).R("R1", (2.5, 5), (17.5, 5)).wire((17.5, 5), (20, 5))
    _inverting(s, 22.5, [("C", "C1", -10)])
    return s


def integrator_rf():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    s.wire((0, 5), (2.5, 5)).R("R1", (2.5, 5), (17.5, 5)).wire((17.5, 5), (20, 5))
    _inverting(s, 22.5, [("R", "R2", -10), ("C", "C1", -25)])
    return s


def active_lp_inv():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    s.wire((0, 5), (2.5, 5)).R("R1", (2.5, 5), (17.5, 5)).wire((17.5, 5), (20, 5))
    _inverting(s, 22.5, [("R", "R2", -10), ("C", "C1", -25)])
    return s


def active_hp_inv():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    s.wire((0, 5), (2.5, 5)).R("R1", (2.5, 5), (17.5, 5)).C("C1", (17.5, 5), (27.5, 5)).wire((27.5, 5), (30, 5))
    _inverting(s, 32.5, [("R", "R2", -10)])
    return s


def differentiator():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    s.wire((0, 5), (5, 5)).C("C1", (5, 5), (15, 5)).wire((15, 5), (20, 5))
    _inverting(s, 22.5, [("R", "R1", -10)])
    return s


# ---------------------------------------------------------------- неинвертирующие
def noninverting():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    p = s.opamp("X1", 30, 0, inv_top=False)
    s.wire((0, 5), p["plus"])
    s.wire(p["minus"], (20, 15), (20, 25))
    s.R("R1", (20, 25), (20, 40), side="left").ground((20, 40))
    s.wire((20, 25), (27.5, 25)).R("R2", (27.5, 25), (42.5, 25), side="below").wire((42.5, 25), (55, 25), (55, 10)).dot((20, 25))
    s.wire(p["out"], (62.5, 10)).dot((55, 10))
    s.terminal((62.5, 10), "OUT", "Uвых", side="right")
    return s


def follower():
    s = Sketch()
    s.terminal((0, 5), "IN", "Uвх")
    p = s.opamp("X1", 30, 0, inv_top=False)
    s.wire((0, 5), p["plus"])
    s.wire(p["minus"], (22.5, 15), (22.5, 25), (55, 25), (55, 10))
    s.wire(p["out"], (62.5, 10)).dot((55, 10))
    s.terminal((62.5, 10), "OUT", "Uвых", side="right")
    return s


def _sk_tail(s, p):
    """Повторитель Саллена-Ки: инвертирующий вход - с выхода (снизу), выход - на вывод Uвых."""
    s.wire(p["minus"], (47.5, 30), (47.5, 42.5), (80, 42.5), (80, 25))
    s.wire(p["out"], (87.5, 25)).dot((80, 25))
    s.terminal((87.5, 25), "OUT", "Uвых", side="right")


def sallen_key_lp():
    s = Sketch()
    s.terminal((0, 20), "IN", "Uвх")
    p = s.opamp("X1", 55, 15, inv_top=False)
    s.wire((0, 20), (2.5, 20)).R("R1", (2.5, 20), (17.5, 20)).wire((17.5, 20), (22.5, 20))
    s.R("R2", (22.5, 20), (37.5, 20)).wire((37.5, 20), p["plus"])
    s.C("C2", (40, 20), (40, 32.5), side="left").ground((40, 32.5))
    s.wire((20, 20), (20, 5), (30, 5)).C("C1", (30, 5), (40, 5)).wire((40, 5), (80, 5), (80, 25))
    s.dot((20, 20)).dot((40, 20))
    _sk_tail(s, p)
    return s


def sallen_key_hp():
    s = Sketch()
    s.terminal((0, 20), "IN", "Uвх")
    p = s.opamp("X1", 55, 15, inv_top=False)
    s.wire((0, 20), (5, 20)).C("C1", (5, 20), (15, 20)).wire((15, 20), (25, 20))
    s.C("C2", (25, 20), (35, 20)).wire((35, 20), p["plus"])
    s.R("R2", (40, 20), (40, 37.5), side="left").ground((40, 37.5))
    s.wire((20, 20), (20, 5), (25, 5)).R("R1", (25, 5), (40, 5)).wire((40, 5), (80, 5), (80, 25))
    s.dot((20, 20)).dot((40, 20))
    _sk_tail(s, p)
    return s


# ---------------------------------------------------------------- МОС на одном ОУ
def _mfb(s, low, mid, high):
    """Схема МОС: вход - узел N1 (x 22,5) - элемент к инв. входу - узел N2 (x 45) - ОУ; low - (вид, имя) с N1 на
    землю, mid - с N2 на выход (y 5), high - с N1 на выход (y -5); input, series - элементы входа и к инв. входу."""
    p = s.opamp("X1", 55, 10, label="right")
    s.wire((45, 15), p["minus"]).dot((45, 15)).dot((22.5, 15))
    s.wire(p["plus"], (47.5, 25)).ground((47.5, 25))
    kind, name = low
    s.two(kind, name, (22.5, 15), (22.5, 32.5), "left")
    s.ground((22.5, 32.5))
    kind, name = mid
    half = 7.5 if kind == "R" else 5
    s.wire((45, 15), (45, 5), (57.5 - half, 5)).two(kind, name, (57.5 - half, 5), (57.5 + half, 5))
    s.wire((57.5 + half, 5), (85, 5))
    kind, name = high
    half = 7.5 if kind == "R" else 5
    s.wire((22.5, 15), (22.5, -7.5), (45 - half, -7.5)).two(kind, name, (45 - half, -7.5), (45 + half, -7.5))
    s.wire((45 + half, -7.5), (85, -7.5), (85, 20)).dot((85, 5)).dot((85, 20))
    s.wire(p["out"], (92.5, 20))
    s.terminal((92.5, 20), "OUT", "Uвых", side="right")
    s.terminal((0, 15), "IN", "Uвх")
    return p


def mfb_lp():
    s = Sketch()
    s.wire((0, 15), (2.5, 15)).R("R1", (2.5, 15), (17.5, 15)).wire((17.5, 15), (22.5, 15))
    s.R("R3", (27.5, 15), (42.5, 15), side="below").wire((22.5, 15), (27.5, 15)).wire((42.5, 15), (45, 15))
    _mfb(s, ("C", "C1"), ("C", "C2"), ("R", "R2"))
    return s


def mfb_bp():
    s = Sketch()
    s.wire((0, 15), (2.5, 15)).R("R1", (2.5, 15), (17.5, 15)).wire((17.5, 15), (22.5, 15))
    s.C("C2", (30, 15), (40, 15), side="below").wire((22.5, 15), (30, 15)).wire((40, 15), (45, 15))
    _mfb(s, ("R", "R2"), ("R", "R3"), ("C", "C1"))
    return s


def mfb_hp():
    s = Sketch()
    s.wire((0, 15), (5, 15)).C("C1", (5, 15), (15, 15)).wire((15, 15), (22.5, 15))
    s.C("C2", (30, 15), (40, 15), side="below").wire((22.5, 15), (30, 15)).wire((40, 15), (45, 15))
    _mfb(s, ("R", "R1"), ("R", "R2"), ("C", "C3"))
    return s


# ---------------------------------------------------------------- активный двойной Т-мост
def twin_t_active():
    """Мост: ветвь C1-C2 (R3 к общей точке) сверху, R1-R2 (C3 к общей точке) снизу; общая точка (N3) - линия внизу,
    её ведёт повторитель X2 (отражён: входы справа); X1 - повторитель на выходе моста; делитель R4-R5 - справа."""
    s = Sketch()
    s.terminal((0, 10), "IN", "Uвх")
    s.wire((0, 10), (5, 10)).wire((5, 0), (5, 20)).dot((5, 10))
    s.wire((5, 0), (10, 0)).C("C1", (10, 0), (20, 0)).wire((20, 0), (35, 0)).C("C2", (35, 0), (45, 0))
    s.wire((45, 0), (55, 0))
    s.R("R1", (5, 20), (20, 20)).wire((20, 20), (40, 20)).R("R2", (40, 20), (55, 20))
    s.wire((55, 0), (55, 20)).dot((55, 10))
    s.wire((25, 0), (25, 25)).dot((25, 0))  # к R3: проходит над ветвью R1-R2 без соединения
    s.R("R3", (25, 25), (25, 40), side="left")
    s.C("C3", (35, 20), (35, 40), side="right").dot((35, 20))
    s.wire((25, 40), (45, 40), (45, 60), (60, 60)).dot((35, 40))
    x1 = s.opamp("X1", 70, 5, inv_top=False)
    s.wire((55, 10), x1["plus"])
    s.wire(x1["minus"], (62.5, 20), (62.5, 30), (95, 30), (95, 15))
    s.wire(x1["out"], (110, 15)).dot((95, 15)).dot((100, 15))
    s.terminal((110, 15), "OUT", "Uвых", side="right")
    s.R("R4", (100, 15), (100, 32.5), side="right").wire((100, 32.5), (100, 40))
    s.R("R5", (100, 40), (100, 55), side="right").ground((100, 55))
    s.wire((100, 37.5), (90, 37.5), (90, 55), (85, 55)).dot((100, 37.5))
    x2 = s.opamp("X2", 65, 50, inv_top=False, mirror=True, label="above")
    s.wire(x2["minus"], (87.5, 65), (87.5, 75), (50, 75), (50, 60)).dot((50, 60))
    return s


# ---------------------------------------------------------------- на трёх ОУ
def _stage(s, name, x, node_x, feedback):
    """Инвертирующий каскад ОУ в цепочке: ОУ с верхним левым углом (x, 10), узел инв. входа - (node_x, 15),
    feedback - [(вид, имя, y)] - ветви с узла на выход каскада (x + 25). -> выводы ОУ."""
    p = s.opamp(name, x, 10)
    s.wire((node_x, 15), p["minus"]).dot((node_x, 15))
    s.ground((p["plus"][0] - 2.5, p["plus"][1]))
    s.wire(p["plus"], (p["plus"][0] - 2.5, p["plus"][1]))
    xr = x + 25
    top = min(y for _, _, y in feedback)
    s.wire((node_x, 15), (node_x, top)).wire((xr, top), (xr, 20)).wire(p["out"], (xr, 20)).dot((xr, 20))
    for kind, name_, y in feedback:
        mid = (node_x + xr) / 2
        half = {"R": 7.5, "C": 5}[kind]
        s.wire((node_x, y), (mid - half, y)).two(kind, name_, (mid - half, y), (mid + half, y)).wire((mid + half, y), (xr, y))
        if y != top:
            s.dot((node_x, y)).dot((xr, y))
    return p


def tow_thomas():
    s = Sketch()
    s.terminal((0, 15), "IN", "Uвх")
    s.wire((0, 15), (2.5, 15)).R("R1", (2.5, 15), (17.5, 15)).wire((17.5, 15), (20, 15))
    _stage(s, "X1", 30, 20, [("R", "R3", 0), ("C", "C1", -15)])
    s.wire((55, 20), (60, 20), (60, 15)).R("R4", (60, 15), (75, 15)).wire((75, 15), (77.5, 15))
    _stage(s, "X2", 85, 77.5, [("C", "C2", 0)])
    s.wire((110, 20), (115, 20), (115, 15)).R("R5", (115, 15), (130, 15)).wire((130, 15), (132.5, 15))
    _stage(s, "X3", 140, 132.5, [("R", "R6", 0)])
    s.wire((165, 20), (172.5, 20))
    s.terminal((172.5, 20), "OUT", "Uвых", side="right")
    # R2: обратная связь с выхода на вход первого ОУ - поверх всего
    s.wire((20, -15), (20, -30), (82.5, -30)).R("R2", (82.5, -30), (97.5, -30)).wire((97.5, -30), (165, -30), (165, 0))
    s.dot((20, -15)).dot((165, 0))
    return s


def khn():
    s = Sketch()
    s.terminal((0, 25), "IN", "Uвх")
    # X3 - сумматор: + от входа и полосового выхода (делитель R1, R2), - от ФНЧ-выхода (R4) с обратной связью R6
    x3 = s.opamp("X3", 30, 10)
    s.wire((0, 25), (2.5, 25)).R("R1", (2.5, 25), (17.5, 25)).wire((17.5, 25), x3["plus"]).dot((20, 25))
    s.R("R2", (20, 25), (20, 40), side="left").wire((20, 40), (20, 50), (110, 50), (110, 20))
    s.wire((20, 15), x3["minus"]).dot((20, 15))
    s.wire((20, 15), (20, -15)).wire((20, 0), (25, 0)).R("R6", (25, 0), (40, 0)).wire((40, 0), (55, 0), (55, 20))
    s.wire(x3["out"], (55, 20)).dot((20, 0)).dot((55, 20))
    s.wire((20, -15), (90, -15)).R("R4", (90, -15), (105, -15)).wire((105, -15), (165, -15), (165, 0))
    # X2 - первый интегратор (полосовой выход), X1 - второй (ФНЧ-выход)
    s.wire((55, 20), (60, 20), (60, 15)).R("R5", (60, 15), (75, 15)).wire((75, 15), (77.5, 15))
    _stage(s, "X2", 85, 77.5, [("C", "C2", 0)])
    s.dot((110, 20))
    s.wire((110, 20), (115, 20), (115, 15)).R("R3", (115, 15), (130, 15)).wire((130, 15), (132.5, 15))
    _stage(s, "X1", 140, 132.5, [("C", "C1", 0)])
    s.wire((165, 20), (172.5, 20)).dot((165, 0))
    s.terminal((172.5, 20), "OUT", "Uвых", side="right")
    return s


SCHEMATICS = {
    "divider": divider, "rc_lp": rc_lp, "rc_hp": rc_hp, "rl_lp": rl_lp, "rl_hp": rl_hp,
    "rc_ladder2": rc_ladder2, "rc_ladder3": rc_ladder3, "twin_t": twin_t,
    "rlc_series": rlc_series, "rlc_parallel_bp": rlc_parallel_bp, "lc_ladder3": lc_ladder3, "lc_ladder5": lc_ladder5,
    "active_lp_inv": active_lp_inv, "active_hp_inv": active_hp_inv,
    "sallen_key_lp": sallen_key_lp, "sallen_key_hp": sallen_key_hp, "mfb_lp": mfb_lp, "mfb_bp": mfb_bp,
    "mfb_hp": mfb_hp, "twin_t_active": twin_t_active, "tow_thomas": tow_thomas, "khn": khn,
    "inverting": inverting, "noninverting": noninverting, "follower": follower, "integrator": integrator,
    "integrator_rf": integrator_rf, "differentiator": differentiator, "tl071": tl071,
}
