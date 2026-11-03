"""Тесты построения схемы MC12 (.CIR) из нетлиста класса A (spice2cir.py, решение В7).

Круговая проверка нетлист -> .CIR -> cir2spice -> нетлист через изоморфизм - без MC12; загрузка и
расчёт в самом MC12 - с MC12 (иначе пропуск).
Запуск из корня репозитория:  python -m unittest discover -s tests
"""
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import bench, canon, cir2spice, iso, mc12run, measure, spice2cir  # noqa: E402

PILOT = [json.loads(l) for l in (ROOT / "data" / "pilot" / "refs.jsonl").read_text(encoding="utf-8").splitlines()]
TEST = [json.loads(l) for l in (ROOT / "data" / "dataset" / "v1" / "test.jsonl").read_text(encoding="utf-8").splitlines()]


def elements_of(spice):
    """Строки элементов из вывода cir2spice (без директив, комментариев и тел подсхем)."""
    out, insub = [], False
    for line in spice.splitlines():
        u = line.strip().upper()
        if u.startswith(".SUBCKT"):
            insub = True
        elif u.startswith(".ENDS"):
            insub = False
        elif not insub and u and not u.startswith(("*", ".")):
            out.append(line)
    elems, err = canon.parse_netlist("\n".join(out))
    assert not err, err
    return elems


def strict_problems(drawing):
    """Связность чертежа по правилам MC12 при загрузке файла: провод соединяется только концом с концом другого
    провода или с выводом; метка и земля - только в такой точке. -> список нарушений (пусто - узлы как в нетлисте)."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    pins = {pin for p in drawing.placed for pin in p.pins}
    ends = {q for a, b, _ in drawing.wires for q in (a, b)}
    joints = pins | ends
    problems = []
    inner = {}  # внутренняя точка провода -> [(цепь, горизонтален ли)]
    for a, b, net in drawing.wires:
        union(a, b)
        for q in spice2cir._segment_points(a, b)[1:-1]:
            inner.setdefault(q, []).append((net, a[1] == b[1]))
    for q, ws in inner.items():
        if q in joints:
            problems.append(f"конец провода или вывод на середине провода {q}")
        if len(ws) > 1 and not (len(ws) == 2 and ws[0][1] != ws[1][1]):
            problems.append(f"провода лежат друг на друге в {q}")
    for text, pt, _ in drawing.labels:
        if pt not in joints:
            problems.append(f"метка {text} не на конце провода или выводе {pt}")
        union(pt, ("label", text))
    for pt, _, _ in drawing.grounds:
        if pt not in joints:
            problems.append(f"земля не на конце провода или выводе {pt}")
        union(pt, ("label", "0"))
    comp = {}
    for p in drawing.placed:
        for node, pin in zip(p.e["nodes"], p.pins):
            comp.setdefault(node, set()).add(find(pin))
    for node, roots in comp.items():
        if len(roots) > 1:
            problems.append(f"узел {node} разорван на {len(roots)} части")
    seen = {}
    for node, roots in comp.items():
        for r in roots:
            if r in seen and seen[r] != node:
                problems.append(f"слились узлы {seen[r]} и {node}")
            seen[r] = node
    return problems


def roundtrip_ok(netlist, f_char=None):
    spice, info = cir2spice.convert(spice2cir.build(netlist, f_char), "t")
    return not info["errors"] and iso.match(canon.parse_netlist(netlist)[0], elements_of(spice), "values")["match"]


class Build(unittest.TestCase):
    def test_roundtrip_pilot(self):
        for r in PILOT:
            with self.subTest(ref=r["id"]):
                self.assertTrue(roundtrip_ok(r["netlist"], r["f_char"]))

    def test_roundtrip_frozen_test_set(self):
        bad = [r["id"] for r in TEST if not roundtrip_ok(r["netlist"], r["f_char"])]
        self.assertEqual(bad, [])

    def test_ascii_crlf(self):
        text = spice2cir.build(PILOT[12]["netlist"], PILOT[12]["f_char"])  # Саллен-Ки с ОУ
        raw = text.encode("ascii")
        self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))

    def test_text_area_after_pages(self):
        # блок модели раньше [Page] MC12 не загружает («Bad format in loading file»)
        text = spice2cir.build("V1 IN 0 DC 0\nX1 IN OUT OUT OPAMP_IDEAL\n")
        self.assertLess(text.rindex("[Page]"), text.index("[Text Area]"))
        self.assertIn("Text=" + spice2cir.OPAMP_MODEL, text)

    def test_opamp_supply_pins_grounded(self):
        placed = spice2cir.layout(canon.parse_netlist("V1 IN 0 DC 0\nX1 IN OUT OUT OPAMP_IDEAL\n")[0])
        op = [p for p in placed if p[0]["type"] == "OPAMP"][0]
        self.assertEqual(op[0]["nodes"][3:], ["0", "0"])
        self.assertEqual(len(op[4]), 5)

    def test_note_right_of_circuit_and_on_page(self):
        # MC12 привязывает многострочный текст снизу: блок не должен уходить выше листа и залезать на схему
        net = PILOT[12]["netlist"]
        placed = spice2cir.layout(canon.parse_netlist(net)[0])
        note = ["t: netlist (spice2cir)"] + net.strip().splitlines() + ["AC analysis: Alt+2"]
        x, y = spice2cir.note_place(placed, note)
        self.assertGreater(x, spice2cir.extent(placed)[0])
        self.assertGreaterEqual(y - spice2cir.TEXT_LINE * len(note), 0)

    def test_nothing_overlaps(self):
        # корпуса, земли, отводы, метки и подписи не накладываются (раньше земля питания ложилась на ОУ)
        for r in PILOT + TEST:
            with self.subTest(ref=r["id"]):
                self.assertEqual(spice2cir.overlaps(spice2cir.layout(canon.parse_netlist(r["netlist"])[0])), [])

    def test_wires_connect_only_at_ends(self):
        # cir2spice соединяет и конец провода с серединой другого, MC12 при загрузке файла - нет: проверка строже
        rows = PILOT + TEST + [json.loads(l) for l in (ROOT / "data" / "dataset" / "v2" / "test.jsonl")
                               .read_text(encoding="utf-8").splitlines()]
        unrouted = 0
        for r in rows:
            if not r.get("netlist"):
                continue
            try:
                placed = spice2cir.layout(canon.parse_netlist(r["netlist"])[0])
            except ValueError:
                continue  # вне класса чертежа (источник тока и т.п.)
            d = spice2cir.draw(placed)
            nets = {n for p in placed for n in p.e["nodes"] if n != "0"}
            unrouted += len(nets - d.routed)
            with self.subTest(ref=r["id"]):
                self.assertEqual(strict_problems(d), [])
        self.assertEqual(unrouted, 0)

    def test_signal_path_layout(self):
        # все схемы класса раскладываются по пути сигнала, сетка - только запасной вариант
        grid = [r["id"] for r in PILOT + TEST
                if not isinstance(spice2cir.layout(canon.parse_netlist(r["netlist"])[0]), spice2cir.Layout)]
        self.assertEqual(grid, [])
        lay = spice2cir.layout(canon.parse_netlist(
            "V1 IN 0 DC 0\nR1 IN N1 1k\nR2 N1 OUT 10k\nC1 OUT 0 1n\nX1 0 N1 OUT OPAMP_IDEAL\n")[0])
        by = {p.e["name"]: p for p in lay}
        self.assertEqual(by["X1"].rot, 4)  # сигнал на минус: ОУ отражён, минус сверху
        self.assertEqual(by["R1"].pins[0][1], by["X1"].pins[2][1])  # R1 и выход ОУ на одной линии
        self.assertLess(by["R2"].y, by["X1"].pins[1][1])  # обратная связь над ОУ
        self.assertEqual((by["C1"].kind, by["C1"].pins[0][0]), ("down", lay.labels["OUT"][2][0][0][0]))
        self.assertLess(by["V1"].x, by["R1"].x)

    def test_grid_fallback(self):
        # два источника - вне правил раскладки по пути сигнала: остаётся сетка, чертёж по-прежнему верный
        net = "V1 IN 0 DC 0\nV2 N1 0 DC 1\nR1 IN OUT 1k\nR2 N1 OUT 2k\nC1 OUT 0 1n\n"
        placed = spice2cir.layout(canon.parse_netlist(net)[0])
        self.assertNotIsInstance(placed, spice2cir.Layout)
        d = spice2cir.draw(placed)
        self.assertEqual((strict_problems(d), spice2cir.overlaps(d)), ([], []))
        self.assertTrue(roundtrip_ok(net))

    def test_grounds_point_away(self):
        placed = spice2cir.layout(canon.parse_netlist("V1 IN 0 DC 0\nR1 IN N1 1k\nC1 N1 0 1n\nX1 0 N1 OUT OPAMP_IDEAL\n")[0])
        c1 = [p for p in placed if p.e["name"] == "C1"][0]
        self.assertEqual(c1.kind, "down")  # элемент на землю стоит вертикально, земля снизу
        text = spice2cir.build("V1 IN 0 DC 0\nX1 0 IN OUT OPAMP_IDEAL\n")
        self.assertIn("[Wire]", text)  # земля на входе ОУ - на конце отвода

    def test_scale_is_an_mc12_step(self):
        # другие значения MC12 молча заменяет на 1
        for r in TEST[:50]:
            text = spice2cir.build(r["netlist"], r["f_char"])
            scale = float(text.split("Scale 0=")[1].split("\r\n")[0])
            self.assertIn(scale, spice2cir.SCALES)

    def test_rejects_current_source(self):
        with self.assertRaises(ValueError):
            spice2cir.build("I1 IN 0 DC 0\nR1 IN 0 1k\n")


def mc12_available():
    try:
        mc12run.find_exe()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(mc12_available(), "Micro-Cap 12 не установлен")
class InMC12(unittest.TestCase):
    def test_loads_and_runs(self):
        work = ROOT / "out" / "test_spice2cir"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        cases = [PILOT[1], PILOT[12], PILOT[14]] + [r for r in TEST if r["level"] == "L3"][:2]
        paths = []
        for i, r in enumerate(cases):
            p = work / f"c{i}.cir"
            spice2cir.write(p, spice2cir.build(r["netlist"], r["f_char"]))
            paths.append(p)
        for res in mc12run.run(paths, analyses=["A"], timeout=300):
            with self.subTest(cir=Path(res["ckt"]).name):
                self.assertTrue(res["ok"], res)
                self.assertEqual(res["messages"], [])

    def test_curve_matches_netlist(self):
        # провода и отражённый ОУ (Rot 4) MC12 читает так же, как cir2spice: АЧХ и ФЧХ чертежа = эталонный .ckt
        work = ROOT / "out" / "test_spice2cir_curve"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        ids = ("p07_inverting", "p08_noninverting", "p12_sk_lp_gain", "p15_mfb_bp", "p16_follower")
        cases = [r for r in PILOT if r["id"] in ids] + [r for r in TEST if r["level"] == "L3"][:2]
        pairs = []
        for i, r in enumerate(cases):
            text = spice2cir.build(r["netlist"], r["f_char"])
            wave = text[text.index("[WaveForm]"):]  # числовой вывод (OUTPUT) и фаза - только для сравнения
            phase = wave.replace("Plt=1", "Plt=2").replace("AliasID=1", "AliasID=2").replace("dB(V(OUT))", "PH(V(OUT))")
            text = text.replace(wave, (wave + "\r\n" + phase).replace("Options=LINEARY", "Options=OUTPUT,LINEARY"))
            spice2cir.write(work / f"w{i}.cir", text)
            bench.write(work / f"r{i}.ckt", bench.build(canon.parse_netlist(r["netlist"])[0], r["f_char"], f"r{i}"))
            pairs.append((r["id"], work / f"w{i}.cir", work / f"r{i}.ckt"))
        res = {Path(x["ckt"]).name: x for x in
               mc12run.run([p for _, *ps in pairs for p in ps], analyses=["A"], timeout=300)}
        for rid, w, ref in pairs:
            with self.subTest(ref=rid):
                self.assertTrue(res[w.name]["ok"] and res[ref.name]["ok"], (res[w.name], res[ref.name]))
                ok, diff = measure.curve_match(measure.read_curve(res[ref.name]["output"]),
                                               measure.read_curve(res[w.name]["output"]))
                self.assertTrue(ok, diff)


if __name__ == "__main__":
    unittest.main()
