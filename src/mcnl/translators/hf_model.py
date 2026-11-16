"""Обученная модель Hugging Face как переводчик: load("hf", base="...", adapter="папка_lora", trained_on="v3").

trained_on - на чём обучена модель (v2 или v3): от этого зависит инструкция (training.instruction). Модель на v3
с инструкцией v2 отвечает без ошибок, но хуже - поэтому параметр обязателен по смыслу для моделей v3.
mode - режим запроса: 1 - перевод описания, 2 - подбор номиналов (только у моделей v3).

Два адаптера на одной базе (решение 14.11.2026, docs/stage2_plan.md): adapter - режим 1, adapter_mode2 - режим 2;
база грузится один раз, режим переключает set_adapter. Без adapter_mode2 режим 2 есть только у адаптера на v3
(общая модель v3 обучена обеим инструкциям)."""
from mcnl import hf_runtime, training
from mcnl.translators import Translator

MODE2_NAME = "mode2"  # имя второго адаптера в PeftModel; первый - "default", как у одиночного адаптера


def slots(adapter, trained_on="v2", adapter_mode2=None, trained_on_mode2="v3"):
    """Режим -> (имя адаптера в PeftModel, на чём обучен). Без torch: проверяется тестами."""
    out = {1: ("default", trained_on)}
    if adapter_mode2 and adapter_mode2 != adapter:
        out[2] = (MODE2_NAME, trained_on_mode2)
    elif adapter_mode2 or trained_on != "v2":
        out[2] = ("default", trained_on_mode2 if adapter_mode2 else trained_on)
    return out


class HFModel(Translator):
    def __init__(self, base, adapter=None, output_format="netlist", four_bit=True, batch=8, trained_on="v2",
                 adapter_mode2=None, trained_on_mode2="v3"):
        self.name = (adapter or base).replace("\\", "/").rstrip("/").split("/")[-1]
        self.output_format = output_format
        self.batch = batch
        self.trained_on = trained_on
        self.slots = slots(adapter, trained_on, adapter_mode2, trained_on_mode2)
        self.modes = tuple(sorted(self.slots))
        extra = {MODE2_NAME: adapter_mode2} if self.slots.get(2, ("",))[0] == MODE2_NAME else None
        self.model, self.tok = hf_runtime.load(base, adapter, four_bit, extra=extra)

    def translate(self, prompt, mode=1):
        return self.translate_batch([prompt], mode)[0]

    def translate_batch(self, prompts, mode=1):
        if mode not in self.slots:
            raise ValueError(f"у {self.name} нет режима {mode}: нужен адаптер, обученный на v3")
        name, trained_on = self.slots[mode]
        if len({n for n, _ in self.slots.values()}) > 1:
            self.model.set_adapter(name)
        system = training.instruction({"mode": mode}, self.output_format, trained_on)
        return hf_runtime.generate(self.model, self.tok, prompts, self.output_format, self.batch,
                                   systems=[system] * len(prompts))[0]
