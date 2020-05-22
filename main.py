from MultilayerPerceptron import Perceptron


if __name__ == '__main__':
    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    saidas = [[0, 1], [1, 0], [1, 0], [0, 1]]
    mlp = Perceptron(entradas, saidas)
    mlp.add_layer("hidden", qtd_neurons=2)
    mlp.add_layer("output")
    mlp.treina(0.01)
