import numpy as np
import random as rd
import matplotlib.pyplot as plt

class Mlp:

    def __init__(self, number_neurons, sample_size, x_min, x_max, learning_rate):
        self.number_neurons = number_neurons
        self.sample_size = sample_size
        self.x_min = x_min
        self.x_max = x_max
        self.learning_rate = learning_rate
        self.vi = np.random.uniform(-0.5, 0.5, self.number_neurons) # Bias dos neurônios da camada oculta
        self.wi = np.random.uniform(-0.5, 0.5, self.number_neurons) # Pesos da camada de entrada x oculta
        self.vy = rd.uniform(-0.5, 0.5) # Bias do neurônio de saída
        self.wy = np.random.uniform(-0.5, 0.5, self.number_neurons) # Pesos da camada oculta x saída
        self.threshold = 0.0
        self.inputs = self.get_inputs(x_min, x_max, sample_size)
        self.targets = self.get_targets(self.inputs)

        """
        Comentário importante:
        No vídeo do zequinha a correção dos pesos é feita somando à matriz de correção, mas o algoritmo necessita de subtrair o gradiente,
        efetivamente fazendo o gradiente descendente. A correção dos pesos está correta, mas a forma de atualização dos pesos está errada.
        Isso é o que pesquisamos e é uma dúvida pertinente.
        """

    def get_inputs(self, x_min, x_max, sample_size):

        return np.linspace(x_min, x_max, sample_size)

    def get_targets(self, inputs):

        targets = np.array([])

        for x in inputs:
            targets = np.append(targets, np.sin(x/2)*np.cos(2*x))

        return targets

    def train(self, min_error, update_callback = None):

        epochs = 0
        number_entries = len(self.inputs)
        error_history = [] # Histórico de erros para plotar o gráfico que será montado em tempo real na interface

        while epochs <= 1000:
            epochs += 1
            epoch_error = 0.0 # Erro de cada época que é zerado novamente a cada época
            # Percorre cada amostra de treinamento
            for i in range(number_entries):

                # Vetores para armazenar valores de entrada (zin) e saída (z)
                # de cada neurônio da camada oculta.
                zin = []
                z = []

                # 1) Camada Oculta: calcula saída de cada neurônio
                for j in range(self.number_neurons):
                    # net input do neurônio j:
                    net_in_j = self.vi[j] + self.wi[j] * self.inputs[i]
                    zin.append(net_in_j)

                    # saída do neurônio j após ativação tanh:
                    z_out_j = np.tanh(net_in_j)
                    z.append(z_out_j)

                # 2) Camada de Saída: combina as saídas dos neurônios ocultos
                sum_value = 0  # zera o acumulador para calcular a soma ponderada
                for k in range(self.number_neurons):
                    sum_value += z[k] * self.wy[k]

                # net input da saída
                yin = self.vy + sum_value
                # saída final (após ativação tanh)
                y = np.tanh(yin)

                # Erro quadrático para a amostra atual
                sample_error = 0.5 * ((y - self.targets[i]) ** 2)

                # Atualiza erro da época, somando o erro da amostra atual aos anteriormente calculados
                epoch_error += sample_error
                # epoch_error = sample_error

                # delta_k: derivada do erro em relação à saída (para o neurônio de saída)
                # (y - target) * derivada da tanh (1 - tanh^2(yin))
                delta_k = (y - self.targets[i]) * (1 - np.tanh(yin) ** 2)

                # Atualização dos pesos da camada de oculta x saída
                for j in range(self.number_neurons):
                    # Gradiente do peso que liga o neurônio oculto j à saída
                    delta_wy = self.learning_rate * delta_k * z[j]
                    # Gradiente do bias da saída
                    delta_vy = self.learning_rate * delta_k

                    # Atualiza pesos e bias
                    self.wy[j] -= delta_wy
                    self.vy -= delta_vy

                # Cada neurônio j na camada oculta recebe parte do erro vindo da saída
                for j in range(self.number_neurons):
                    # delta_in = erro que chega ao neurônio j
                    # como há apenas 1 neurônio de saída, é delta_k * peso_que_liga_j_à_saída
                    delta_in = self.wy[j] * delta_k

                    # delta_j = delta_in * derivada da ativação tanh(zin[j])
                    delta_j = delta_in * (1 - np.tanh(zin[j]) ** 2)

                    # Gradientes para os pesos de entrada do neurônio j
                    delta_wi = self.learning_rate * delta_j * self.inputs[i]
                    delta_vi = self.learning_rate * delta_j

                    # Atualiza pesos e bias do neurônio j na camada oculta x entrada
                    self.wi[j] -= delta_wi
                    self.vi[j] -= delta_vi

            # Se o erro da época for menor que min_error, paramos
            if epoch_error <= min_error:
                break