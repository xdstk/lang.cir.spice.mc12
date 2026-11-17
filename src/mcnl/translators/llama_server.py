"""Обученная модель через llama.cpp (llama-server): работает на процессоре и без Python-библиотек модели (torch,
transformers) - для переносной версии и ПК без видеокарты (17.11.2026).

load("llama", server="…/llama-server.exe", model="…/m1.gguf", model_mode2="…/m2.gguf", trained_on="v2",
     trained_on_mode2="v3")

Модель режима - база Qwen3.5-4B с влитым адаптером, переведённая в GGUF (out/gguf, docs/stage2_plan.md, «Переносная
версия»): LoRA-файлы llama.cpp к базе Qwen3.5 не подключает (имена тензоров слоёв DeltaNet не совпадают). Сервер
держит одну модель: при смене режима он перезапускается с другой - памяти нужно вдвое меньше, чем на две модели сразу.
Сервер слушает свободный порт 127.0.0.1 и останавливается вместе с программой. Запрос - POST /completion: текст в шаблоне
чата Qwen (chat_text - тот же текст, что training.chat_text даёт через apply_chat_template; сверено на 174 запросах),
жадный выбор токена, остановка на <|im_end|>.
"""
import atexit
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from mcnl import training
from mcnl.hf_runtime import MAX_NEW_TOKENS
from mcnl.translators import Translator

END = "<|im_end|>"


def chat_text(system, prompt):
    """Запрос в шаблоне чата Qwen3.5 без рассуждений (enable_thinking=False), как в training.chat_text."""
    return ("<|im_start|>system\n" + system + END + "\n<|im_start|>user\n" + prompt + END
            + "\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


_JOB = None


def kill_with_parent(proc):
    """Сервер завершится вместе с программой при любом её завершении (и аварийном, и «Снять задачу»): Job Object
    Windows с JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Иначе сервер с моделью (4-5 ГБ памяти) остался бы висеть. -> True,
    если привязан (не Windows или отказ системы - False, тогда остаётся atexit)."""
    global _JOB
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return False
    if not hasattr(ctypes, "windll"):
        return False

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class Basic(ctypes.Structure):
        _fields_ = [("user_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", wintypes.DWORD),
                    ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("active", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO), ("proc_mem", ctypes.c_size_t), ("job_mem", ctypes.c_size_t),
                    ("peak_proc", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    if _JOB is None:
        job = k32.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
            return False
        _JOB = job  # описатель живёт до конца процесса программы: закрыла его система - сервер завершён
    return bool(k32.AssignProcessToJobObject(wintypes.HANDLE(_JOB), wintypes.HANDLE(int(proc._handle))))


# Дискретные видеокарты: только на них Vulkan быстрее процессора. Замер 17.11.2026 (запросы программы, i5-10300H):
# GTX 1660 Ti - 1,1 с на ответ, процессор - 7,1 с, встроенная Intel UHD - 20,9 с (в 3 раза медленнее процессора).
DISCRETE = re.compile(r"nvidia|geforce|quadro|tesla|\brtx\b|radeon\s*(rx|pro)|arc\(tm\)\s*a\d|\barc\s*a\d", re.I)


def pick_device(server_gpu, timeout=60):
    """Устройство для -dev из списка сборки Vulkan (--list-devices, около 1 с) -> ("VulkanN", имя) первой дискретной
    видеокарты или None (её нет, нет драйвера Vulkan, сборка не запустилась)."""
    try:
        out = subprocess.run([str(server_gpu), "--list-devices"], capture_output=True, text=True, errors="replace",
                             timeout=timeout, cwd=str(Path(server_gpu).parent),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (out.stdout or "").splitlines():
        m = re.match(r"\s*(Vulkan\d+):\s*(.+?)\s*\(", line)
        if m and DISCRETE.search(m.group(2)):
            return m.group(1), m.group(2)
    return None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mode_models(model, trained_on="v2", model_mode2=None, trained_on_mode2="v3"):
    """Режим -> (файл модели, на чём обучена). Без отдельной модели режима 2 он есть только у модели на v3
    (как hf_model.slots)."""
    out = {1: (str(model), trained_on)}
    if model_mode2:
        out[2] = (str(model_mode2), trained_on_mode2)
    elif trained_on != "v2":
        out[2] = (str(model), trained_on)
    return out


class LlamaServer(Translator):
    def __init__(self, server, model, model_mode2=None, output_format="netlist", trained_on="v2",
                 trained_on_mode2="v3", threads=None, ctx=2048, start_timeout=600, log=None, extra_args=(),
                 prime=True, server_gpu=None, gpu="auto"):
        # server_gpu - сборка llama.cpp с Vulkan: берётся, только если есть дискретная видеокарта (pick_device), иначе -
        # server (процессор). gpu="off" - всегда процессор. Не запустилась на видеокарте - процессор.
        # ctx: инструкция 276-499 токенов, запрос - до нескольких сотен, ответ - до MAX_NEW_TOKENS (800); меньше
        # контекст - меньше памяти на слабом ПК
        if output_format != "netlist":
            raise ValueError("llama-server: только формат netlist")
        for p in (server, model, model_mode2):
            if p and not Path(p).is_file():
                raise FileNotFoundError(f"нет файла {p}")
        # сервер запускается из своей папки (рядом его библиотеки): относительные пути - от неё, поэтому - абсолютные
        server, model = Path(server).resolve(), Path(model).resolve()
        model_mode2 = Path(model_mode2).resolve() if model_mode2 else None
        self.output_format = output_format
        self.name = Path(model).stem
        self.server, self.threads, self.ctx, self.start_timeout = str(server), threads, ctx, start_timeout
        self.extra_args = list(extra_args)
        # Прогрев (17.11.2026): у гибридной модели (слои DeltaNet) сервер не может откатить состояние к общему началу
        # запросов без контрольной точки, и инструкция (276-499 токенов) считалась заново в каждом запросе - 8 из 12 с
        # на i5-10300H. Точка после инструкции (-cms 0: точки без минимального шага) и запрос из одной инструкции
        # перед каждым запросом: обработка запроса 8,3 -> 1,3 с, ответы те же. Точек немного (-ctxcp 4) - память.
        self.prime = prime
        if prime:
            self.extra_args = ["-cms", "0", "-ctxcp", "4"] + self.extra_args
        self.log = Path(log) if log else None
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
        self.models = mode_models(model, trained_on, model_mode2, trained_on_mode2)
        self.modes = tuple(sorted(self.models))
        self.proc, self.current, self.url = None, None, None
        atexit.register(self.close)
        self.device = "процессор"
        cpu = (self.server, list(self.extra_args))
        pick = pick_device(server_gpu) if server_gpu and gpu != "off" and Path(server_gpu).is_file() else None
        if pick:
            self.server = str(Path(server_gpu).resolve())
            self.extra_args += ["-dev", pick[0], "-ngl", "auto"]  # -fit (по умолчанию): слои - сколько войдёт в память
            self.device = f"видеокарта {pick[1]}"
        try:
            self._ensure(1)  # модель режима 1 - сразу: «загружен» в окне значит «готов отвечать»
        except (RuntimeError, TimeoutError, OSError):
            if not pick:
                raise
            self.server, self.extra_args = cpu
            self.device = f"процессор (видеокарта {pick[1]} не запустилась)"
            self._ensure(1)

    def _ensure(self, mode):
        path = self.models[mode][0]
        if self.current == path and self.proc is not None and self.proc.poll() is None:
            return
        self.close()
        port = free_port()
        args = [self.server, "-m", path, "--host", "127.0.0.1", "--port", str(port), "-c", str(self.ctx),
                "-np", "1", "--no-webui"]
        if self.threads:
            args += ["-t", str(self.threads)]
        args += self.extra_args
        out = open(self.log, "a", encoding="utf-8", errors="replace") if self.log else subprocess.DEVNULL
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # без окна консоли под Windows
        self.proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, cwd=str(Path(self.server).parent),
                                     creationflags=flags)
        kill_with_parent(self.proc)
        self.url, self.current = f"http://127.0.0.1:{port}", path
        self._wait(self.start_timeout)
        if self.prime:  # прогрев - при загрузке, а не в первом запросе: на старом ПК без AVX он идёт 2-4 мин
            self._post({"prompt": self.system_prefix(mode), "n_predict": 0, "cache_prompt": True})

    def _wait(self, timeout):
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.proc = self.current = None
                raise RuntimeError(f"llama-server завершился с кодом {code}" + (f"; журнал: {self.log}" if self.log else ""))
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass  # сервер ещё загружает модель (503) или не слушает порт
            time.sleep(0.5)
        self.close()
        raise TimeoutError(f"llama-server не ответил за {timeout} с")

    def request(self, prompt, mode=1):
        """Тело запроса /completion (без сети: проверяется тестом)."""
        if mode not in self.models:
            raise ValueError(f"у {self.name} нет режима {mode}")
        system = training.instruction({"mode": mode}, self.output_format, self.models[mode][1])
        return {"prompt": chat_text(system, prompt), "n_predict": MAX_NEW_TOKENS, "temperature": 0, "top_k": 1,
                "stop": [END], "cache_prompt": True}

    def system_prefix(self, mode=1):
        """Начало каждого запроса режима: инструкция и открытая реплика пользователя (для прогрева)."""
        system = training.instruction({"mode": mode}, self.output_format, self.models[mode][1])
        return chat_text(system, "\0").split("\0")[0]

    def _post(self, body):
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read().decode("utf-8"))

    def complete(self, prompt, mode=1):
        """Ответ сервера целиком (content и timings)."""
        body = self.request(prompt, mode)
        self._ensure(mode)
        if self.prime:
            self._post({"prompt": self.system_prefix(mode), "n_predict": 0, "cache_prompt": True})
        return self._post(body)

    def translate(self, prompt, mode=1):
        return self.complete(prompt, mode)["content"].strip()

    def translate_batch(self, prompts, mode=1):
        return [self.translate(p, mode) for p in prompts]

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.proc = self.current = None


def find_runtime(root):
    """Сервер и модели: в переносной версии - <root>/llama и <root>/models, у разработчика - out/tools/llama-bin и
    out/gguf. Модели - models.json рядом с ними: {"mode1": {"file", "trained_on"}, "mode2": {...}}.
    Сборка с Vulkan (если есть) - <root>/llama-vk или out/tools/llama-vk.
    -> {"server", "model", "trained_on", "model_mode2", "trained_on_mode2", "server_gpu"} или None, если чего-то нет."""
    root = Path(root)
    server = next((p for p in (root / "llama" / "llama-server.exe", root / "out" / "tools" / "llama-bin" / "llama-server.exe")
                   if p.is_file()), None)
    index = next((p for p in (root / "models" / "models.json", root / "out" / "gguf" / "models.json") if p.is_file()), None)
    if server is None or index is None:
        return None
    spec = json.loads(index.read_text(encoding="utf-8"))
    m1, m2 = spec.get("mode1"), spec.get("mode2")
    if not m1 or not (index.parent / m1["file"]).is_file():
        return None
    out = {"server": str(server), "model": str(index.parent / m1["file"]), "trained_on": m1.get("trained_on", "v2")}
    if m2 and (index.parent / m2["file"]).is_file():
        out.update(model_mode2=str(index.parent / m2["file"]), trained_on_mode2=m2.get("trained_on", "v3"))
    gpu = next((p for p in (root / "llama-vk" / "llama-server.exe", root / "out" / "tools" / "llama-vk" / "llama-server.exe")
                if p.is_file()), None)
    if gpu:  # сборка с Vulkan - для дискретной видеокарты (pick_device)
        out["server_gpu"] = str(gpu)
    return out
