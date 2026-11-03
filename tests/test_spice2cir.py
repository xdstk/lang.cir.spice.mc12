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

from mcnl import canon, cir2spice, iso, mc12run, spice2cir  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
