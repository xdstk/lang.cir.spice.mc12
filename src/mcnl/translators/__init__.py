"""Переводчики «запрос на русском -> нетлист» - сменные «батарейки» программы.

Любой переводчик - класс с полем output_format ("netlist" или "ir") и методом translate(prompt) -> str
(строки элементов класса A или JSON-IR mcnl.ir/1; обвязку, анализы и .END добавляет программа).
Подключение: load("baseline"), load("mcnl.translators.baseline:RuleBaseline") или
load("путь/к/файлу.py:Класс") - так обученная модель встаёт на место baseline без правок программы.
"""
import importlib
import importlib.util
from pathlib import Path


class Translator:
    """Базовый класс (необязателен: достаточно тех же поля и метода)."""
    name = "translator"
    output_format = "netlist"

    def translate(self, prompt):
        raise NotImplementedError

    def translate_batch(self, prompts):
        """Переопределяется моделями, которым выгодна пакетная обработка."""
        return [self.translate(p) for p in prompts]


REGISTRY = {
    "baseline": "mcnl.translators.baseline:RuleBaseline",
    "echo": "mcnl.translators.echo:Echo",
    "hf": "mcnl.translators.hf_model:HFModel",  # обученная модель: load("hf", base=..., adapter=...)
    "glm": "mcnl.translators.glm_api:GLMApi",  # справочная строка, API Z.ai: load("glm", model="glm-4.7-flash")
}


def load(spec, **kwargs):
    """Переводчик по имени из REGISTRY, по «модуль:Класс» или по «файл.py:Класс»."""
    spec = REGISTRY.get(spec, spec)
    if ":" not in spec:
        raise ValueError(f"неизвестный переводчик {spec!r}; есть {sorted(REGISTRY)} или «модуль:Класс»")
    target, cls_name = spec.rsplit(":", 1)
    if target.endswith(".py"):
        path = Path(target)
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    obj = getattr(module, cls_name)(**kwargs)
    for attr in ("translate", "output_format"):
        if not hasattr(obj, attr):
            raise TypeError(f"{spec}: у переводчика нет {attr}")
    return obj
