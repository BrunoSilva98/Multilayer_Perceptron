from MultilayerPerceptron import Perceptron
import matplotlib.pyplot as plt

if __name__ == '__main__':
    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    saidas = [[0, 1], [1, 0], [1, 0], [0, 1]]

    mlp = Perceptron(entradas, saidas, taxa_aprendizado=0.2, alpha=0.2)
    mlp.add_layer("hidden", qtd_neurons=5)
    mlp.add_layer("output")
    mlp.treina(erro_desejado=0.001, epocas=10000)

    erros = mlp.mse
    plt.plot(erros)
    plt.show()

    for entrada in entradas:
        print(mlp.evaluate(entrada))
