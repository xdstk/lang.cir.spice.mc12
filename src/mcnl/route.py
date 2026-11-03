"""Трассировка проводов чертежа MC12 по сетке (spice2cir): цепи по одной, A* с ценой изгибов и пересечений.

Правила соединения MC12 (RM12 гл. 2, "Wires"; так же их читает cir2spice): провода соединяются концом к концу,
концом к выводу элемента и концом к внутренней точке другого провода; пересечение внутренних точек не соединяет;
вывод, лежащий на проводе, соединяется с ним. При загрузке файла MC12 строже - только концы (проверено 03.11.2026),
поэтому spice2cir потом режет провода в точках примыкания. Отсюда ограничения трассировки:
  * чужую цепь можно только пересечь под прямым углом через её внутреннюю точку, не поворачивая в ней;
  * нельзя проходить через чужие выводы и точки привязки меток (метка на проводе именует его узел);
  * нельзя поворачивать, ответвляться или заканчиваться на чужом проводе.
Координаты - целые шаги сетки.
"""
import heapq

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
STEP, BEND, CROSS = 1, 4, 8


class Grid:
    def __init__(self, bounds, blocked):
        self.x0, self.y0, self.x1, self.y1 = bounds
        self.blocked = set(blocked)  # недоступно никому: корпуса, подписи, земли
        self.owner = {}              # точка -> цепь: выводы, точки меток, точки земель
        self.wire = {}               # точка -> (цепь, "H" | "V" | "X"); X - конец, изгиб или ответвление

    def free_for(self, p, net):
        """Можно ли цепи net занять точку p как свою (повернуть, закончить, ответвиться)."""
        if not (self.x0 <= p[0] <= self.x1 and self.y0 <= p[1] <= self.y1):
            return False
        own = self.owner.get(p)
        if own is not None:
            return own == net
        if p in self.blocked:
            return False
        w = self.wire.get(p)
        return w is None or w[0] == net

    def crossable(self, p, net, horizontal):
        """Можно ли пройти через p напрямую, пересекая чужой провод."""
        if p in self.owner or p in self.blocked:
            return False
        w = self.wire.get(p)
        return w is not None and w[0] != net and w[1] == ("V" if horizontal else "H")

    def add(self, net, points):
        """Записывает путь (список точек с единичным шагом) как провода цепи net -> отрезки [(a, b)]."""
        segs, start = [], 0
        for i in range(1, len(points)):
            last = i == len(points) - 1
            turn = not last and (points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]) != \
                (points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            if last or turn:
                segs.append((points[start], points[i]))
                start = i
        for k, p in enumerate(points):
            if self.wire.get(p, (None,))[0] not in (None, net):
                continue  # пересечение чужого провода: его отметку не трогаем
            vertex = k in (0, len(points) - 1) or any(p in s for s in segs) or p in self.wire
            if vertex:
                self.wire[p] = (net, "X")
            else:
                self.wire[p] = (net, "H" if points[k - 1][1] == p[1] else "V")
        return segs


def _search(grid, net, start, goals):
    """A* от вывода start к любой точке goals (уже соединённая часть цепи) -> список точек или None."""
    gx0 = min(x for x, _ in goals)
    gx1 = max(x for x, _ in goals)
    gy0 = min(y for _, y in goals)
    gy1 = max(y for _, y in goals)

    def h(p):
        return max(gx0 - p[0], 0, p[0] - gx1) + max(gy0 - p[1], 0, p[1] - gy1)

    best = {(start, None): 0}
    prev = {}
    heap = [(h(start), 0, start, None)]
    while heap:
        _, cost, p, d = heapq.heappop(heap)
        if best.get((p, d), 1e18) < cost:
            continue
        if p in goals and p != start:
            path = [p]
            state = (p, d)
            while state in prev:
                state = prev[state]
                path.append(state[0])
            return path[::-1]
        crossing = p != start and not grid.free_for(p, net)
        for nd in DIRS:
            if d is not None and nd == (-d[0], -d[1]):
                continue
            if crossing and nd != d:
                continue  # на чужом проводе не поворачиваем
            q = (p[0] + nd[0], p[1] + nd[1])
            if grid.free_for(q, net):
                step = STEP
            elif grid.crossable(q, net, nd[1] == 0):
                step = STEP + CROSS
            else:
                continue
            c = cost + step + (BEND if d is not None and nd != d else 0)
            if c < best.get((q, nd), 1e18):
                best[(q, nd)] = c
                prev[(q, nd)] = (p, d)
                heapq.heappush(heap, (c + h(q), c, q, nd))
    return None


def route_net(grid, net, terminals):
    """Соединяет выводы цепи деревом проводов -> [(a, b)] или None (тогда сетка возвращается как была)."""
    if len(terminals) < 2:
        return []
    saved = dict(grid.wire)
    tree = {terminals[0]}
    segs = []
    rest = list(terminals[1:])
    while rest:
        rest.sort(key=lambda t: min(abs(t[0] - x) + abs(t[1] - y) for x, y in tree))
        t = rest.pop(0)
        if t in tree:
            continue
        path = _search(grid, net, t, tree)
        if path is None:
            grid.wire = saved
            return None
        segs += grid.add(net, path)
        tree.update(path)
    return segs
