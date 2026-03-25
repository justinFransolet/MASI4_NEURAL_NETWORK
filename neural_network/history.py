# neural_network_history.py

from .metrics import calculate_mse, calculate_recall, calculate_precision, calculate_accuracy, calculate_roc_auc, calculate_f1_score

class History:
    """ Classe représentant les différentes métriques surveiller lors de l'entrainement d'un réseau de neurones."""

    def __init__(self, metrics = ()):
        self._metrics = {}

        # Initialisation des listes de métriques
        for metric in metrics:
            self._metrics[str(metric.value)] = []

    @property
    def metrics(self)-> dict:
        return self._metrics

    def execute_mse(self, y_true: list[float], y_pred: list[float])-> None:
        """
        Executer le calcul de l'Erreur Quadratique Moyenne (MSE).
        Attention : Utilisable seulement pour de la régression

        :param y_true: Les valeurs réelles.
        :param y_pred: Les valeurs prédites.
        :raise ValueError: Si les listes ne sont pas de même taille.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Les listes doivent avoir la même taille.")
        self._metrics['mse'].append(calculate_mse(y_true, y_pred))

    def execute_accuracy(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> None:
        """
        Executer le calcul de l'accuracy.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        """
        value = calculate_accuracy(y_true, y_pred_probs, threshold)
        self._metrics['accuracy'].append(value)

    def execute_precision(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> None:
        """
        Executer le calcul de la précision.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        """
        value = calculate_precision(y_true,y_pred_probs, threshold)
        self._metrics['precision'].append(value)

    def execute_recall(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> None:
        """
        Executer le calcul du recall.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        """
        value = calculate_recall(y_true, y_pred_probs, threshold)
        self._metrics['recall'].append(value)

    def execute_f1_score(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> None:
        """
        Executer le calcul du f1 score.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        """
        value = calculate_f1_score(y_true, y_pred_probs, threshold)
        self._metrics['f1_score'].append(value)

    def execute_roc(self, y_true: list[float], y_pred_probs: list[float])-> None:
        """
        Executer le calcul de la courbe ROC_AUC.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.

        :raise ValueError: L'AUC nécessite au moins une instance positive et une négative.
        """
        value = calculate_roc_auc(y_true, y_pred_probs)
        self._metrics['roc_auc'].append(value)