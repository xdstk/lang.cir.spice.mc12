"""Переводчик-заглушка для проверки программы: возвращает нетлист, который ему дали заранее
(или сам запрос, если он уже является нетлистом). Нужен тестам и для ручного ввода нетлиста."""
from . import Translator


class Echo(Translator):
    name = "echo"
    output_format = "netlist"
    modes = (1, 2)

    def __init__(self, answers=None):
        self.answers = answers or {}

    def translate(self, prompt, mode=1):
        return self.answers.get(prompt, prompt)
