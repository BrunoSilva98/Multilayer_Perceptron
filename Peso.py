import random


class Peso:
    def __init__(self, num_padroes):
        self.value = random.random()
        self.gradiente = None
        self.deslocamento = 0
        self.deslocamentos = list()
        self.inicializa_deslocamentos(num_padroes)

    def inicializa_deslocamentos(self, num_padroes):
        for i in range(num_padroes):
            self.deslocamentos.append(0)

    def calc_gradiente(self, delta_k, saida_ant):  # Delta_K é o delta do neurônio a direita do peso, ant a esquerda
        self.gradiente = delta_k * saida_ant

    def calc_deslocamento(self, taxa_aprendizado, alpha, index):
        self.deslocamento = (self.gradiente * taxa_aprendizado) + (alpha * self.deslocamentos[index])
        self.deslocamentos[index] = self.deslocamento

    def atualiza_peso(self):
        self.value = self.value + self.deslocamento
