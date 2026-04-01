#neural_network/neural_network.py
from .metrics import MetricType
from .formula import derivative, matrix_confusion
from .layer import Layer
from .history import History
from enum import Enum

class TrainingType(Enum):
    STOCHASTIC = 'stochastic'
    FULL_BATCH = 'full_batch'


def _format_values(values: list[float], precision: int = 2) -> str:
    return ", ".join(f"{v:.{precision}f}" for v in values)

def _log_sample_details(x_sample: list[float], y_true: list[float], y_pred: list[float], sample_error: float, layers: list[Layer], verbose: bool,) -> None:
    if not verbose:
        return

    x_str = ", ".join(f"{v:.0f}" for v in x_sample)
    d_str = _format_values(y_true, precision=3)
    y_str = _format_values(y_pred, precision=3)

    layer_summaries = []
    for index, layer in enumerate(layers, start=1):
        biases_str = _format_values(layer.biases, precision=2)
        flat_weights = [w for neuron_weights in layer.weights for w in neuron_weights]
        weights_str = _format_values(flat_weights, precision=2)
        layer_summaries.append(f"L{index}: b=[{biases_str}], w=[{weights_str}]")

    print(
        f"x=[{x_str}], d=[{d_str}], y=[{y_str}], mse={sample_error:.6f}"
        + (f", {' | '.join(layer_summaries)}" if layer_summaries else "")
    )


def _log(text: str, verbose: bool)-> None:
    if verbose:
        print(text)

def _log_epoch_metrics(history: History, metrics: tuple, y_true: list[list[float]], y_pred: list[list[float]], threshold: float):
    """Calcule et enregistre les métriques demandées pour l'époque actuelle."""

    if MetricType.MSE in metrics:
        history.log_mse(y_true, y_pred)

    # 2. Métriques de Classification - Nécessitent des listes plates pour matrix_confusion
    # On aplatit les données pour traiter l'ensemble des prédictions du réseau
    y_true_flat = [val for sublist in y_true for val in sublist]
    y_pred_flat = [val for sublist in y_pred for val in sublist]

    tp, tn, fp, fn = matrix_confusion(y_true_flat, y_pred_flat, threshold)

    if MetricType.ACCURACY in metrics:
        history.log_accuracy(tp, tn, fp, fn)

    if MetricType.PRECISION in metrics:
        history.log_precision(tp, fp)

    if MetricType.RECALL in metrics:
        history.log_recall(tp, fn)

    if MetricType.F1_SCORE in metrics:
        history.log_f1_score(tp, fp, fn)

    if MetricType.ROC_AUC in metrics:
        history.log_roc(y_true_flat, y_pred_flat)

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
            # Calcul de l'erreur local
            error_signal = y_pred[j] - y_true[j]
            # Appliquer la dérivée de la fonction d'activation
            deriv = derivative(output_layer.activation, y_pred[j])
            output_layer.deltas.append(error_signal * deriv)

        # 2. Couches cachées (Rétropropagation)
        for l in reversed(range(len(self.__layers) - 1)):
            current_layer = self.__layers[l]
            next_layer = self.__layers[l + 1]
            current_layer.deltas = []

            for j in range(len(current_layer.weights)):
                # Somme pondérée des deltas de la couche suivante
                error_signal = sum(next_layer.deltas[k] * next_layer.weights[k][j] for k in range(len(next_layer.weights)))
                # IMPORTANT : La dérivée utilise la sortie mémorisée du neurone lors du forward
                # On passe d'abord la valeur, puis le type d'activation
                deriv = derivative(current_layer.activation, current_layer.last_outputs[j])
                current_layer.deltas.append(error_signal * deriv)

    def _update_weights(self, learning_rate: float, grad_w: list[list[list[float]]] | None, grad_b: list[list[float]] | None, is_stochastic=True):
        """Applique la correction aux poids et biais."""
        for l, layer in enumerate(self.__layers):
            for j in range(len(layer.weights)):
                for k in range(len(layer.weights[j])):
                    gradient = layer.deltas[j] * layer.last_inputs[k]
                    if is_stochastic:
                        layer.weights[j][k] -= learning_rate * gradient
                    else:
                        if grad_w is None:
                            raise AttributeError("Attribut manquant : grad_w")
                        grad_w[l][j][k] += gradient

                if is_stochastic:
                    layer.biases[j] -= learning_rate * layer.deltas[j]
                else:
                    if grad_b is None:
                        raise AttributeError("Attribut manquant : grad_b")
                    grad_b[l][j] += layer.deltas[j]

    def train(self, x_train: list[list[float]], y_train: list[list[float]], epochs: int, learning_rate: float, strategy=TrainingType.STOCHASTIC, verbose=False, metrics=()):
        history = History(metrics)
        n_samples = len(x_train)
        is_stochastic = (strategy == TrainingType.STOCHASTIC)

        for epoch in range(epochs):
            _log(f"\n---- Epoch {epoch + 1} ----", verbose)
            # Début d'époque
            # Initialisation des accumulateurs si Batch
            grad_w, grad_b = self._init_gradients() if not is_stochastic else (None, None)

            # Boucle sur les échantillons
            list_y_pred = []
            for i in range(n_samples):
                # 1. Forward
                y_pred = self.predict(x_train[i])
                list_y_pred.append(y_pred)

                # 2. Backward
                self._backward_pass(y_train[i], y_pred)

                # 3. Mise à jour ou Accumulation
                self._update_weights(learning_rate, grad_w, grad_b, is_stochastic)

                _log_sample_details(
                    x_sample=x_train[i],
                    y_true=y_train[i],
                    y_pred=y_pred,
                    sample_error=0,
                    layers=self.__layers,
                    verbose=verbose,
                )

            # Fin d'époque
            _log_epoch_metrics(history, metrics, y_train, list_y_pred, 0.5)

            # Mise à jour finale pour le Full-Batch
            if not is_stochastic:
                self._apply_batch_update(grad_w, grad_b, learning_rate, n_samples)

        return history