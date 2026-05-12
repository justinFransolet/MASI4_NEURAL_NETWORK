# neural_network/network.py

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

    def _is_sample_error(
            self,
            y_true: list[float],
            y_pred: list[float],
            threshold: float = 0.5
    ) -> bool:
        """
        Retourne True si l'échantillon est mal classé, sinon False.
        On binarise y_pred avec un seuil, puis on compare à y_true.
        """
        y_pred_class = [1 if value >= threshold else 0 for value in y_pred]
        y_true_class = [1 if value >= threshold else 0 for value in y_true]
        return y_pred_class != y_true_class

    def train(self, x_train: list[list[float]], y_train: list[list[float]], epochs: int, learning_rate: float, strategy=TrainingType.STOCHASTIC, verbose=False, metrics=(), update_on_error_only: bool = True, mse_threshold: float | None = None, classification_threshold: float = 0.5):
        history = History(metrics)
        n_samples = len(x_train)
        is_stochastic = (strategy == TrainingType.STOCHASTIC)

        for epoch in range(epochs):
            _log(f"\n---- Epoch {epoch + 1} ----", verbose)

            # Compteur d'erreurs pour l'époque
            nb_errors = 0

            # Début d'époque
            # Initialisation des accumulateurs si Batch
            grad_w, grad_b = self._init_gradients() if not is_stochastic else (None, None)

            # Boucle sur les échantillons
            list_y_pred = []
            for i in range(n_samples):
                # 1. Forward
                y_pred = self.predict(x_train[i])
                list_y_pred.append(y_pred)

                sample_mse = self._compute_sample_mse(y_train[i], y_pred)

                # 2. Vérifier si erreur
                sample_has_error = self._is_sample_error(y_train[i], y_pred, threshold=classification_threshold)

                # Modification UNIQUEMENT si erreur
                if sample_has_error:
                    nb_errors += 1

                # Cas perceptron simple : update seulement s'il y a erreur de classe
                # Cas gradient / ADALINE : update sur chaque exemple
                should_update = sample_has_error if update_on_error_only else True

                if should_update:
                    # 3. Backward uniquement si erreur
                    self._backward_pass(y_train[i], y_pred)

                    # 4. Update uniquement si erreur
                    self._update_weights(learning_rate, grad_w, grad_b, is_stochastic)

                _log_sample_details(
                    x_sample=x_train[i],
                    y_true=y_train[i],
                    y_pred=y_pred,
                    sample_error=sample_mse,
                    layers=self.__layers,
                    verbose=verbose,
                )

            # Mise à jour finale pour le Full-Batch
            if not is_stochastic:
                self._apply_batch_update(grad_w, grad_b, learning_rate, n_samples)

            # Recalcul des prédictions de l'époque avec les poids finaux de l'époque
            epoch_y_pred = [self.predict(x) for x in x_train]

            # Fin d'époque
            _log_epoch_metrics(history, metrics, y_train, epoch_y_pred, classification_threshold)

            # Sauvegarde des paramètres pour tracer la frontière de décision
            if len(self.__layers) == 1 and len(self.__layers[0].weights) == 1:
                history.log_params(
                    bias=self.__layers[0].biases[0],
                    weights=self.__layers[0].weights[0]
                )

            epoch_mse = None
            if MetricType.MSE in metrics:
                epoch_mse = history.metrics[MetricType.MSE.value][-1]

            _log(f"nbErreurs = {nb_errors}", verbose)
            if epoch_mse is not None:
                _log(f"MSE = {epoch_mse:.6f}", verbose)

            # Arrêt anticipé
            if update_on_error_only:
                # mode perceptron simple
                if nb_errors == 0:
                    _log("Arrêt anticipé : nbErreurs = 0", verbose)
                    break
            else:
                # mode descente du gradient / ADALINE
                if mse_threshold is not None and epoch_mse is not None and epoch_mse <= mse_threshold:
                    _log(f"Arrêt anticipé : MSE <= {mse_threshold}", verbose)
                    break

        return history

    def _compute_sample_mse(self, y_true: list[float], y_pred: list[float]) -> float:
        if len(y_true) == 0:
            raise ArithmeticError("Impossible de diviser par 0.")
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)