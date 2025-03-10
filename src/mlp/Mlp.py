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
        self.inputs = self.get_inputs(x_min, x_max, sample_size)
        self.targets = self.get_targets(self.inputs)

    def get_inputs(self, x_min, x_max, sample_size):

        return np.linspace(x_min, x_max, sample_size)

    def get_targets(self, inputs):

        targets = np.array([])

        for x in inputs:
            targets = np.append(targets,np.sin(x/2)*np.cos(2*x))

        return targets

        