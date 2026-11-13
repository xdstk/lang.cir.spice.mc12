"""Обученная модель Hugging Face как переводчик: load("hf", base="...", adapter="папка_lora", trained_on="v3").

trained_on - на чём обучена модель (v2 или v3): от этого зависит инструкция (training.instruction). Модель на v3
с инструкцией v2 отвечает без ошибок, но хуже - поэтому параметр обязателен по смыслу для моделей v3.
mode - режим запроса: 1 - перевод описания, 2 - подбор номиналов (только у моделей v3)."""
from mcnl import hf_runtime, training
from mcnl.translators import Translator


class HFModel(Translator):
    def __init__(self, base, adapter=None, output_format="netlist", four_bit=True, batch=8, trained_on="v2"):
        self.name = (adapter or base).replace("\\", "/").rstrip("/").split("/")[-1]
        self.output_format = output_format
        self.batch = batch
        self.trained_on = trained_on
        self.model, self.tok = hf_runtime.load(base, adapter, four_bit)

    def translate(self, prompt, mode=1):
        return self.translate_batch([prompt], mode)[0]

    def translate_batch(self, prompts, mode=1):
        system = training.instruction({"mode": mode}, self.output_format, self.trained_on)
        return hf_runtime.generate(self.model, self.tok, prompts, self.output_format, self.batch,
                                   systems=[system] * len(prompts))[0]
