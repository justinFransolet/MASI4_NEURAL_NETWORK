#neural_network/neural_network.py

from neural_network import derivative, Layer
from enum import Enum

class TrainingType(Enum):
    STOCHASTIC = 'stochastic'
    FULL_BATCH = 'full_batch'

class NeuralNetwork:
    """
    Classe représentant un réseau de neurones.
    """
    def __init__(self):
        self.__layers = []

    def add_layer(self, layer: Layer)-> None:
        """
        Ajoute une couche au réseau de neurones.

        :param layer: Couche à ajouter.
        """
        self.__layers.append(layer)

    def predict(self, inputs: list[float])-> list[float]:
        """
        Effectuer une prédiction.

        :param inputs: Entrées du réseau de neurones.
        :raise NotTrainedModelError: Si le réseau de neurones n'est pas entraîné.
        :return: Sortie du réseau de neurones.
        """
        output = inputs
        for layer in self.__layers:
            output = layer.forward(output)

        return output

    def _init_gradients(self)-> tuple[list[list[list[float]]], list[list[float]]]:
        """
        Prépare les structures vides pour accumuler les gradients.
        """
        grad_w = [[[0.0] * len(l.weights[0]) for _ in range(len(l.weights))] for l in self.__layers]
        grad_b = [[0.0] * len(l.biases) for l in self.__layers]
        return grad_w, grad_b

    def _apply_batch_update(self, grad_w: list[list[list[float]]], grad_b: list[list[float]], learning_rate: float, n_samples: int)-> None:
        """Applique la moyenne des gradients accumulés (Full-Batch)."""
        for l, layer in enumerate(self.__layers):
            for j in range(len(layer.weights)):
                for k in range(len(layer.weights[j])):
                    layer.weights[j][k] -= learning_rate * (grad_w[l][j][k] / n_samples)
                layer.biases[j] -= learning_rate * (grad_b[l][j] / n_samples)

    def _backward_pass(self, y_true: list[float], y_pred: list[float]):
        """Calcule les deltas (signaux d'erreur) de la sortie vers l'entrée."""
        # 1. Couche de sortie
        output_layer = self.__layers[-1]
        output_layer.deltas = []
        for j in range(len(output_layer.weights)):
            error_signal = y_pred[j] - y_true[j]
            deriv = derivative(output_layer.activation, y_pred[j])
            output_layer.deltas.append(error_signal * deriv)

        # 2. Couches cachées (Rétropropagation)
        for l in reversed(range(len(self.__layers) - 1)):
            current_layer = self.__layers[l]
            next_layer = self.__layers[l + 1]
            current_layer.deltas = []

            for j in range(len(current_layer.weights)):
                error_signal = sum(next_layer.deltas[k] * next_layer.weights[k][j]
                                   for k in range(len(next_layer.weights)))
                deriv = derivative(current_layer.last_outputs[j], current_layer.activation)
                current_layer.deltas.append(error_signal * deriv)

    def _update_weights(self, learning_rate: float, grad_w: list[list[list[float]]], grad_b: list[list[float]], is_stochastic=True):
        """Applique la correction aux poids et biais."""
        for l, layer in enumerate(self.__layers):
            for j in range(len(layer.weights)):
                for k in range(len(layer.weights[j])):
                    gradient = layer.deltas[j] * layer.last_inputs[k]
                    if is_stochastic:
                        layer.weights[j][k] -= learning_rate * gradient
                    else:
                        grad_w[l][j][k] += gradient

                if is_stochastic:
                    layer.biases[j] -= learning_rate * layer.deltas[j]
                else:
                    grad_b[l][j] += layer.deltas[j]

    def train(self, x_train: list[list[float]], y_train: list[list[float]], epochs: int, learning_rate: float, strategy=TrainingType.STOCHASTIC):
        history = []
        n_samples = len(x_train)
        is_stochastic = (strategy == TrainingType.STOCHASTIC)

        for epoch in range(epochs):
            total_error = 0
            # Initialisation des accumulateurs si Batch
            grad_w, grad_b = self._init_gradients() if not is_stochastic else (None, None)

            for i in range(n_samples):
                # 1. Forward
                y_pred = self.predict(x_train[i])
                total_error += sum(0.5 * (yt - yp) ** 2 for yt, yp in zip(y_train[i], y_pred))

                # 2. Backward
                self._backward_pass(y_train[i], y_pred)

                # 3. Mise à jour ou Accumulation
                self._update_weights(learning_rate, grad_w, grad_b, is_stochastic)

            # Mise à jour finale pour le Full-Batch
            if not is_stochastic:
                self._apply_batch_update(grad_w, grad_b, learning_rate, n_samples)

            history.append(total_error / n_samples)

        return history