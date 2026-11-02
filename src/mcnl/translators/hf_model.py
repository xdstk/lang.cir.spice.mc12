"""Обученная модель Hugging Face как переводчик: load("hf", base="...", adapter="папка_lora")."""
from mcnl import hf_runtime
from mcnl.translators import Translator


class HFModel(Translator):
    def __init__(self, base, adapter=None, output_format="netlist", four_bit=True, batch=8):
        self.name = (adapter or base).replace("\\", "/").rstrip("/").split("/")[-1]
        self.output_format = output_format
        self.batch = batch
        self.model, self.tok = hf_runtime.load(base, adapter, four_bit)

    def translate(self, prompt):
        return self.translate_batch([prompt])[0]

    def translate_batch(self, prompts):
        return hf_runtime.generate(self.model, self.tok, prompts, self.output_format, self.batch)[0]
