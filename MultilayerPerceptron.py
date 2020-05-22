from Neuron import Neuron
from Peso import Peso


# Inputs devem ser sublistas [[], []]
class Perceptron:
    def __init__(self, inputs, saidas, taxa_aprendizado=0.2, alpha=0.2):
        self.padroes = inputs
        self.hidden_layer = list()
        self.output_layer = list()
        self.taxa_aprendizado = taxa_aprendizado
        self.alpha = alpha
        self.correct_outputs = saidas

    def add_layer(self, tipo, qtd_neurons):
        if tipo.lower() == "hidden":
            for i in range(qtd_neurons):
                self.hidden_layer.append(Neuron(len(self.padroes[0])))

        elif tipo.lower() == "output":
            for i in range(len(self.correct_outputs[0])):
                self.output_layer.append(Neuron(len(self.hidden_layer)))

    def get_saidas(self):
        saidas = list()
        for neuron in self.output_layer:
            saidas.append(neuron.saida)
        return saidas

    def loss_function(self, erros):  # MSE
        mse = 0
        for erro in erros:
            mse += erro
        mse = mse/len(self.padroes)
        return mse

    def calcula_erro(self, corretas):
        saidas = self.get_saidas()
        erro = 0
        for (saida, correta) in zip(saidas, corretas):
            erro = erro + (correta - saida)
        return erro**2

    def evaluate(self):
        saidas = list()
        erros = list()
        for (entrada, saida) in zip(self.padroes, self.correct_outputs):
            saidas.clear()
            for neuron in self.hidden_layer:
                neuron.set_entradas(entrada)
                neuron.evaluate()
                saidas.append(neuron.saida)

            for neuron in self.output_layer:
                neuron.set_entradas(saidas)
                neuron.evaluate()
            erros.append(self.calcula_erro(saida))

        mse = self.loss_function(erros)
        return mse