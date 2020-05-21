import random


class Peso:
    def __init__(self):
        self.value = random.random()
        self.gradiente = None
        self.neuron_prev = None
        self.neuron_after = None
        self.deslocamento = None

    def set_neurons(self, neuron_prev, neuron_after):
        self.neuron_prev = neuron_prev
        self.neuron_after = neuron_after

    def calc_gradiente(self):
        self.gradiente = self.neuron_after.delta * self.neuron_prev.output

    def calc_deslocamento(self, taxa_aprendizado, alpha):
        if self.deslocamento is None:
            self.deslocamento = self.gradiente * taxa_aprendizado
        else:
            self.deslocamento = (self.gradiente * taxa_aprendizado) + (alpha * self.deslocamento)

    def atualiza_peso(self):
        self.value = self.value + self.deslocamento
