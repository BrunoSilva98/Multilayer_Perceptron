from Neuron import Neuron


# Inputs devem ser sublistas [[], []]
# noinspection PyMethodMayBeStatic
class Perceptron:
    def __init__(self, inputs, saidas, taxa_aprendizado=0.2, alpha=0.2):
        self.padroes = inputs
        self.hidden_layer = list()
        self.output_layer = list()
        self.taxa_aprendizado = taxa_aprendizado
        self.alpha = alpha
        self.correct_outputs = saidas
        self.mse = list()

    def add_layer(self, tipo, qtd_neurons=None):
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
            erro = erro**2
            mse += erro
        mse = mse / (len(self.padroes) * len(self.output_layer))
        return mse

    def calcula_erro(self, corretas, erros):
        for neuron, correta in zip(self.output_layer, corretas):
            neuron.erro = correta - neuron.saida
            erros.append(neuron.erro)

    def treina(self, erro_desejado):
        mse = 1
        epoch = 0
        while mse > erro_desejado:
            epoch += 1
            print(epoch)
            saidas = list()
            erros = list()
            for (entrada, saida, index) in zip(self.padroes, self.correct_outputs, range(len(self.padroes))):
                saidas.clear()

                for neuron in self.hidden_layer:
                    neuron.set_entradas(entrada)
                    neuron.evaluate()
                    saidas.append(neuron.saida)

                for neuron in self.output_layer:
                    neuron.set_entradas(saidas)
                    neuron.evaluate()

                self.calcula_erro(saida, erros)
                # TODO Gravar a variação dos pesos na iteração anterior. A variação é única pra cada input
                self.calculate_deltas_output()
                self.calculate_deltas_hidden()
                self.atualiza_pesos()

            mse = self.loss_function(erros)
            self.mse.append(mse)
        return epoch

    def calculate_deltas_output(self):
        for neuron in self.output_layer:
            derivada = neuron.derivada_sigmoid()
            delta = derivada * neuron.erro
            neuron.set_delta(delta)

    def calculate_deltas_hidden(self):
        for idx in range(len(self.hidden_layer)):
            somatoria_pesos_deltas = 0
            neuron = self.hidden_layer[idx]
            derivada = neuron.derivada_sigmoid()

            for neuron_k in self.output_layer:
                peso = neuron_k.pesos[idx+1]
                delta = neuron_k.delta
                somatoria_pesos_deltas = somatoria_pesos_deltas + (peso.value * delta)

            delta = derivada * somatoria_pesos_deltas
            neuron.delta = delta

    def atualiza_pesos(self):
        for neuron in self.output_layer:
            for idx in range(len(neuron.pesos)):
                peso = neuron.pesos[idx]
                entrada = neuron.entradas[idx]
                peso.calc_gradiente(neuron.delta, entrada)
                peso.calc_deslocamento(self.taxa_aprendizado, self.alpha)
                peso.atualiza_peso()

        for neuron in self.hidden_layer:
            for idx in range(len(neuron.pesos)):
                peso = neuron.pesos[idx]
                entrada = neuron.entradas[idx]
                peso.calc_gradiente(neuron.delta, entrada)
                peso.calc_deslocamento(self.taxa_aprendizado, self.alpha)
                peso.atualiza_peso()
