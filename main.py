import Neuron


if __name__ == '__main__':
    neuronios = list()
    entradas = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(2):
        neuronios.append(Neuron.Neuron("hidden"))
