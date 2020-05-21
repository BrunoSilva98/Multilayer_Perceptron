# Input deve ser uma lista com os valores, ex [1, 2, 3]

import numpy as np
from Peso import Peso


# noinspection PyMethodMayBeStatic
class Neuron:
    def __init__(self):
        self.entradas = list()
        self.saida = None
        self.delta = None

    def set_entradas(self, entradas):
        self.entradas.append([1, Peso()])  # Bias
        for value in entradas:
            self.entradas.append([value, Peso()])

    def somatorio(self):
        somatoria = 0
        for idx in range(len(self.entradas)):
            entrada = self.entradas[idx][0]
            peso = self.entradas[idx][1].value
            somatoria = somatoria + (entrada * peso)
        return somatoria

    def sigmoid_activation(self, somatoria):
        self.saida = 1 / (1 + np.exp(-somatoria))

    def evaluate(self):
        somatoria = self.somatorio()
        self.sigmoid_activation(somatoria)

    def set_delta(self, delta):
        self.delta = delta
