# neural_network/layer.py

import random
from neural_network import dot_product, activate, ActivationFunction

class Layer:
    """
    Classe représentant une couche d'un réseau de neurones.
    """
    def __init__(self, n_inputs: int, n_neurons: int, activation=ActivationFunction.SIGMOID):
        # Initialisation aléatoire des poids entre -0.5 et 0.5
        self.weights = [[random.uniform(-0.5, 0.5) for _ in range(n_inputs)] for _ in range(n_neurons)]
        self.biases = [random.uniform(-0.5, 0.5) for _ in range(n_neurons)]
        self.activation = activation

        # Variables de mémorisation pour la rétropropagation
        self.last_inputs = []
        self.last_outputs = []
        self.deltas = []

    def forward(self, inputs):
        self.last_inputs = inputs
        self.last_outputs = []
        for i in range(len(self.weights)):
            # Sommation
            z = dot_product(self.weights[i], inputs) + self.biases[i]
            # Activation
            a = activate(self.activation, z)
            self.last_outputs.append(a)
        return self.last_outputs