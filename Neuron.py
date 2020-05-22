# Input deve ser uma lista com os valores, ex [1, 2, 3]

import numpy as np
from Peso import Peso


# noinspection PyMethodMayBeStatic
class Neuron:
    def __init__(self, input_length, num_padroes):
        self.entradas = list()
        self.pesos = list()
        self.saida = None
        self.delta = None
        self.cria_pesos(input_length, num_padroes)

    def set_entradas(self, entradas):
        self.entradas.clear()
        self.entradas.append(1)  # Bias
        for value in entradas:
            self.entradas.append(value)

    def cria_pesos(self, quantidade, num_padroes):
        quantidade += 1
        for i in range(quantidade):
            self.pesos.append(Peso(num_padroes))

    def somatorio(self):
        somatoria = 0
        for idx in range(len(self.entradas)):
            entrada = self.entradas[idx]
            peso = self.pesos[idx].value
            somatoria = somatoria + (entrada * peso)
        return somatoria

    def sigmoid_activation(self, somatoria):
        self.saida = 1 / (1 + np.exp(-somatoria))

    def derivada_sigmoid(self):
        derivada = self.saida * (1 - self.saida)
        return derivada

    def evaluate(self):
        somatoria = self.somatorio()
        self.sigmoid_activation(somatoria)

    def set_delta(self, delta):
        self.delta = delta
