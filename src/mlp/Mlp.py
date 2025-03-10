import numpy as np
import random as rd
import matplotlib.pyplot as plt

class Mlp:

    def __init__(number_neurons, sample_size, x_min, x_max, learning_rate):
        self.number_neurons = number_neurons
        self.sample_size = sample_size
        self.x_min = x_min
        self.x_max = x_max
        self.learning_rate = learning_rate
        self.inputs = inputs(x_min, x_max, sample_size)
        self.targets = targets(self.inputs)

    def inputs(x_min, x_max, sample_size):

        return np.linspace(x_min, x_max, sample_size)

    def targets(inputs):

        targets = []

        for x in inputs:
            targets.append(np.sin(x/2)*np.cos(2*x))

        return targets