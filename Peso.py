import random


class Peso:
    def __init__(self):
        self.value = random.random()
        self.gradiente = None
        self.deslocamento = 0
        self.deslocamentos = list()

    def calc_gradiente(self, delta_k, saida_ant):  # Delta_K é o delta do neurônio a direita do peso, ant a esquerda
        self.gradiente = delta_k * saida_ant

    def calc_deslocamento(self, taxa_aprendizado, alpha):
        self.deslocamento = (self.gradiente * taxa_aprendizado) + (alpha * self.deslocamento)

    def atualiza_peso(self):
        self.value = self.value + self.deslocamento
