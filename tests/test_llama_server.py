"""Переводчик через llama.cpp (translators/llama_server.py) без сервера: шаблон запроса, модели режимов, поиск файлов,
выбор переводчика в окне; MC12 рядом с программой."""
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcnl import gui, mc12run, training  # noqa: E402
from mcnl.translators import llama_server  # noqa: E402

WORK = ROOT / "out" / "test_llama_server"


class Template(unittest.TestCase):
    def test_chat_text_is_qwen_template(self):
        """Тот же текст, что training.chat_text через apply_chat_template Qwen3.5 (сверено 17.11.2026 на 174 запросах
        программы и llm_test с инструкциями v2, v3 и режима 2)."""
        self.assertEqual(llama_server.chat_text("S", "P"),
                         "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\n"
                         "<think>\n\n</think>\n\n")
        self.assertIn("{x}", llama_server.chat_text("{x}", "}{"))  # фигурные скобки инструкции - как есть

    def test_mode_models(self):
        self.assertEqual(llama_server.mode_models("a.gguf"), {1: ("a.gguf", "v2")})
        self.assertEqual(llama_server.mode_models("a.gguf", "v2", "b.gguf"), {1: ("a.gguf", "v2"), 2: ("b.gguf", "v3")})
        self.assertEqual(llama_server.mode_models("a.gguf", "v3"), {1: ("a.gguf", "v3"), 2: ("a.gguf", "v3")})

    def test_request_body(self):
        tr = llama_server.LlamaServer.__new__(llama_server.LlamaServer)  # без запуска сервера
        tr.output_format, tr.name = "netlist", "m"
        tr.models = llama_server.mode_models("a.gguf", "v2", "b.gguf")
        b1, b2 = tr.request("RC ФНЧ", 1), tr.request("ФНЧ на 1 кГц", 2)
        self.assertEqual(b1["prompt"], llama_server.chat_text(training.instruction({"mode": 1}, "netlist", "v2"), "RC ФНЧ"))
        self.assertEqual(b2["prompt"], llama_server.chat_text(training.instruction({"mode": 2}, "netlist", "v3"),
                                                              "ФНЧ на 1 кГц"))
        self.assertEqual((b1["temperature"], b1["top_k"], b1["stop"]), (0, 1, ["<|im_end|>"]))
        with self.assertRaises(ValueError):
            tr.request("x", 3)
        # прогрев: начало запроса до текста пользователя - общее у всех запросов режима
        for mode, body in ((1, b1), (2, b2)):
            pre = tr.system_prefix(mode)
            self.assertTrue(body["prompt"].startswith(pre) and pre.endswith("<|im_start|>user\n"), pre[-40:])


class Device(unittest.TestCase):
    def test_discrete_names(self):
        """Vulkan - только на дискретной видеокарте: встроенная Intel UHD в 3 раза медленнее процессора (17.11.2026)."""
        for name in ("NVIDIA GeForce GTX 1660 Ti with Max-Q Design", "NVIDIA GeForce RTX 3060", "Quadro P2000",
                     "AMD Radeon RX 6600", "AMD Radeon Pro W6600", "Intel(R) Arc(TM) A770 Graphics"):
            self.assertTrue(llama_server.DISCRETE.search(name), name)
        for name in ("Intel(R) UHD Graphics", "Intel(R) HD Graphics 620", "Intel(R) Iris(R) Xe Graphics",
                     "AMD Radeon(TM) Graphics", "AMD Radeon Vega 8 Graphics", "Microsoft Basic Render Driver",
                     "llvmpipe (LLVM 15.0.7, 256 bits)"):
            self.assertFalse(llama_server.DISCRETE.search(name), name)

    def test_gpu_failure_falls_back_to_cpu(self):
        from unittest import mock
        WORK.mkdir(parents=True, exist_ok=True)
        for f in ("cpu.exe", "gpu.exe", "m1.gguf"):
            (WORK / f).write_bytes(b"")
        started = []

        def ensure(tr, mode):
            started.append((Path(tr.server).name, list(tr.extra_args)))
            if Path(tr.server).name == "gpu.exe":
                raise RuntimeError("llama-server завершился с кодом 1")
        with mock.patch.object(llama_server, "pick_device", return_value=("Vulkan1", "NVIDIA X")), \
                mock.patch.object(llama_server.LlamaServer, "_ensure", ensure):
            tr = llama_server.LlamaServer(WORK / "cpu.exe", WORK / "m1.gguf", server_gpu=WORK / "gpu.exe", prime=False)
        self.assertEqual([s for s, _ in started], ["gpu.exe", "cpu.exe"])
        self.assertIn("-dev", started[0][1])
        self.assertNotIn("-dev", started[1][1])
        self.assertEqual(tr.device, "процессор (видеокарта NVIDIA X не запустилась)")
        with mock.patch.object(llama_server, "pick_device", return_value=None), \
                mock.patch.object(llama_server.LlamaServer, "_ensure", ensure):
            tr = llama_server.LlamaServer(WORK / "cpu.exe", WORK / "m1.gguf", server_gpu=WORK / "gpu.exe", prime=False)
        self.assertEqual(tr.device, "процессор")


class Runtime(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(WORK, ignore_errors=True)
        (WORK / "llama").mkdir(parents=True)
        (WORK / "models").mkdir()

    def test_find_runtime(self):
        self.assertIsNone(llama_server.find_runtime(WORK))
        (WORK / "llama" / "llama-server.exe").write_bytes(b"")
        (WORK / "models" / "m1.gguf").write_bytes(b"")
        (WORK / "models" / "models.json").write_text(json.dumps(
            {"mode1": {"file": "m1.gguf", "trained_on": "v2"}, "mode2": {"file": "m2.gguf", "trained_on": "v3"}}))
        rt = llama_server.find_runtime(WORK)
        self.assertEqual(Path(rt["model"]).name, "m1.gguf")
        self.assertNotIn("model_mode2", rt)  # файла режима 2 нет - режима 2 нет
        (WORK / "models" / "m2.gguf").write_bytes(b"")
        rt = llama_server.find_runtime(WORK)
        self.assertEqual((Path(rt["model_mode2"]).name, rt["trained_on_mode2"]), ("m2.gguf", "v3"))

    def test_translator_choices(self):
        self.assertEqual(gui.translator_choices(WORK, torch_installed=True), ([gui.QWEN, gui.BASELINE], gui.QWEN))
        self.assertEqual(gui.translator_choices(WORK, torch_installed=False), ([gui.BASELINE], gui.BASELINE))
        (WORK / "llama" / "llama-server.exe").write_bytes(b"")
        (WORK / "models" / "m1.gguf").write_bytes(b"")
        (WORK / "models" / "models.json").write_text(json.dumps({"mode1": {"file": "m1.gguf"}}))
        self.assertEqual(gui.translator_choices(WORK, torch_installed=False), ([gui.LLAMA, gui.BASELINE], gui.LLAMA))
        self.assertEqual(gui.translator_choices(WORK, torch_installed=True)[1], gui.QWEN)

    def test_portable_layout(self):
        """Сборщик переносной версии: start.bat - ASCII с CRLF, Python изолирован (-E -s), данные программы есть,
        настройки MC12 с путями C:\\MC12 не копируются."""
        from mcnl import portable
        bat = portable.START_BAT.encode("ascii")
        self.assertIn(b"\r\n", bat)
        self.assertNotIn(b"\n", bat.replace(b"\r\n", b""))
        self.assertIn(b"pythonw.exe\" -E -s src\\mcnl\\gui.py", bat)
        for name in portable.DATA:
            self.assertTrue((ROOT / "data" / name).exists(), name)
        self.assertIn("mcap.dat", portable.MC12_SKIP)
        self.assertIn("site-packages", portable.PY_SKIP)
        readme = (ROOT / "docs" / "portable_readme.txt").read_text(encoding="utf-8")
        for key in ("{DATE}", "{DISK}", "{RAM}", "{TIME}"):
            self.assertIn(key, readme)

    def test_bundled_mc12_first(self):
        """Переносная версия: MC12 из <корень>/MC12 раньше установленного в C:\\MC12."""
        self.assertEqual(Path(mc12run.DEFAULT_EXES[0]), ROOT / "MC12" / "Micro-Cap 12.exe")
        self.assertEqual(mc12run.DEFAULT_EXES[2], r"C:\MC12\Micro-Cap 12.exe")


if __name__ == "__main__":
    unittest.main()
