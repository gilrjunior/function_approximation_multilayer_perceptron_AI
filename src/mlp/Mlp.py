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
        self.vi = 0
        self.wi = []
        self.vy = 0
        self.wy = rd.uniform(-0.5, 0.5)
        self.threshold = 0.0
        self.inputs = self.get_inputs(x_min, x_max, sample_size)
        self.targets = self.get_targets(self.inputs)

    def get_inputs(self, x_min, x_max, sample_size):

        return np.linspace(x_min, x_max, sample_size)

    def get_targets(self, inputs):

        targets = np.array([])

        for x in inputs:
            targets = np.append(targets,np.sin(x/2)*np.cos(2*x))

        return targets

    def training(self, min_error):

        yin = 0
        y = 0

        error = 1.0
        epochs = 0

        zin = 0
        z = 0

        entries = len(self.inputs)

        for i in range(self.number_neurons):
            self.wi[i] = selfrd.uniform(-0.5, 0.5)
        
        while epochs <= 1000:

            epochs += 1

            while i in range(entries):

                z = self.inputs[i]

                while j in range(self.number_neurons):

                    zin = self.vi + self.wi[j] * z

                    z = np.tanh(zin)

                yin = self.vy + self.wy * z

                y = np.tanh(yin)

                error = 0.5 * ((y - self.targets[i])**2)

                #Fazer a correção dos pesos




        