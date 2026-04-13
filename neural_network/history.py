# neural_network_history.py

from .metrics import calculate_mse, calculate_recall, calculate_precision, calculate_accuracy, calculate_roc_auc, calculate_f1_score

class History:
    """ Classe représentant les différentes métriques surveiller lors de l'entrainement d'un réseau de neurones."""

    def __init__(self, metrics = ()):
        self._metrics = {}
        self._params_history = []

        # Initialisation des listes de métriques
        for metric in metrics:
            self._metrics[str(metric.value)] = []

    @property
    def metrics(self)-> dict:
        return self._metrics

    @property
    def params_history(self):
        return self._params_history

    def log_params(self, bias: float, weights: list[float]) -> None:
        self._params_history.append({
            "bias": bias,
            "weights": weights[:]
        })

    def log_mse(self, y_true: list[list[float]], y_pred: list[list[float]])-> None:
        """
        Executer le calcul de l'Erreur Quadratique Moyenne (MSE).
        Attention : Utilisable seulement pour de la régression

        :param y_true: Les valeurs réelles de l'époque.
        :param y_pred: Les valeurs prédites de l'époque.
        :raise ValueError: Si les listes ne sont pas de même taille.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Les listes doivent avoir la même taille.")
        cumulative_error = 0.0
        for i in range(len(y_true)):
            cumulative_error += calculate_mse(y_true[i], y_pred[i])

        epoch_mse = cumulative_error / len(y_true)
        self._metrics['mse'].append(epoch_mse)

    def log_accuracy(self, tp: int, tn: int, fp: int, fn: int)-> None:
        """
        Executer le calcul de l'accuracy.

        :param tp: True Positive
        :param tn: True Negative
        :param fp: False Positive
        :param fn: False Negative
        """
        value = calculate_accuracy(tp, tn, fp, fn)
        self._metrics['accuracy'].append(value)

    def log_precision(self, tp: int, fp: int)-> None:
        """
        Executer le calcul de la précision.

        :param tp: True Positive
        :param fp: False Positive
        """
        value = calculate_precision(tp,fp)
        self._metrics['precision'].append(value)

    def log_recall(self, tp: int, fn: int)-> None:
        """
        Executer le calcul du recall.

        :param tp: True Positive
        :param fn: False Negative
        """
        value = calculate_recall(tp, fn)
        self._metrics['recall'].append(value)

    def log_f1_score(self, tp: int, fp: int, fn: int)-> None:
        """
        Executer le calcul du f1 score.

        :param tp: True Positive
        :param fp: False Positive
        :param fn: False Negative
        """
        value = calculate_f1_score(tp, fp, fn)
        self._metrics['f1_score'].append(value)

    def log_roc(self, y_true: list[float], y_pred_probs: list[float])-> None:
        """
        Executer le calcul de la courbe ROC_AUC.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.

        :raise ValueError: L'AUC nécessite au moins une instance positive et une négative.
        """
        value = calculate_roc_auc(y_true, y_pred_probs)
        self._metrics['roc_auc'].append(value)