# neural_network_history.py

from enum import Enum

class MetricType(Enum):
    MSE = 'mse'
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'
    ROC_AUC = 'roc_auc'

def get_confusion_matrix(y_true: list[float], y_pred_probs: list[float], threshold=0.5):
    """
    Calcule TP, TN, FP, FN en appliquant un seuil sur les probabilités.

    :param y_true: Les valeurs réelles.
    :param y_pred_probs: La probabilité que la valeur appartient à la classe.
    :param threshold: La valeur limite pour être dans une catégorie
    :raise ValueError: Si les listes ne sont pas de même taille.
    """
    if len(y_true) != len(y_pred_probs):
        raise ValueError("Les listes doivent avoir la même taille.")

    tp = tn = fp = fn = 0
    for yt, yp_prob in zip(y_true, y_pred_probs):
        # Binarisation de la prédiction selon le seuil
        yp = 1 if yp_prob >= threshold else 0

        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    return tp, tn, fp, fn


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

    def calculate_mse(self, y_true: list[float], y_pred: list[float])-> None:
        """
        Calcule l'Erreur Quadratique Moyenne (MSE).
        Attention : Utilisable seulement pour de la régression

        :param y_true: Les valeurs réelles.
        :param y_pred: Les valeurs prédites.
        :raise ValueError: Si les listes ne sont pas de même taille.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Les listes doivent avoir la même taille.")

        n = len(y_true)
        self._metrics['mse'].append(sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n)

    def calculate_accuracy(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> float:
        """
        Calculer le taux de fiabilité du modèle.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        :return: Le taux de fiabilité.
        """
        tp, tn, fp, fn = get_confusion_matrix(y_true, y_pred_probs, threshold)

        total = tp + tn + fp + fn
        value = (tp + tn) / total if total > 0 else 0.0
        self._metrics['accuracy'].append(value)
        return value

    def calculate_precision(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> float:
        """
        Calculer le taux de précision du modèle.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        :return: Le taux de précision.
        """
        tp, tn, fp, fn = get_confusion_matrix(y_true, y_pred_probs, threshold)

        value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        try:
            self._metrics['precision'].append(value)
        except KeyError:
            self._metrics['precision'] = [value]
        return value

    def calculate_recall(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> float:
        """
        Calculer la valeur "recall" est le taux de vrai positif sur la quantité réelle de positif.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        :return: Le taux de précision.
        """
        tp, tn, fp, fn = get_confusion_matrix(y_true, y_pred_probs, threshold)

        value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        try:
            self._metrics['recall'].append(value)
        except KeyError:
            self._metrics['recall'] = [value]
        return value

    def calculate_f1_score(self, y_true: list[float], y_pred_probs: list[float], threshold=0.5)-> float:
        """
        Calculer la moyenne harmonique entre la précision et le rappel.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :param threshold: La valeur limite pour être dans une catégorie.
        :return: Le taux de précision.
        """

        precision = self.calculate_precision(y_true,y_pred_probs, threshold)
        recall = self.calculate_recall(y_true,y_pred_probs, threshold)

        if (precision + recall) > 0:
            value = 2 * (precision * recall) / (precision + recall)
        else:
            value = 0.0

        self._metrics['f1_score'].append(value)
        return value

    def calculate_roc_auc(self, y_true: list[float], y_pred_probs: list[float])-> float:
        """
        Calcule l'aire sous la courbe ROC (AUC) à l'aide de la méthode des trapèzes.
        Nécessite de trier les prédictions par ordre décroissant de probabilité.

        :param y_true: Les valeurs réelles.
        :param y_pred_probs: La probabilité que la valeur appartient à la classe.
        :raise ValueError: L'AUC nécessite au moins une instance positive et une négative.
        :return: L'aire sous la courbe ROC.
        """
        # Trier les index par probabilité prédite décroissante
        indices = list(range(len(y_pred_probs)))
        indices.sort(key=lambda i: y_pred_probs[i], reverse=True)

        y_true_sorted = [y_true[i] for i in indices]

        num_pos = sum(y_true)
        num_neg = len(y_true) - num_pos

        if num_pos == 0 or num_neg == 0:
            raise ValueError("L'AUC nécessite au moins une instance positive et une négative.")

        tp = 0
        fp = 0
        tpr_list = [0.0]  # True Positive Rate
        fpr_list = [0.0]  # False Positive Rate

        # Parcours des valeurs triées pour construire la courbe
        for yt in y_true_sorted:
            if yt == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / num_pos)
            fpr_list.append(fp / num_neg)

        # Calcul de l'aire sous la courbe avec la règle des trapèzes
        auc = 0.0
        for i in range(1, len(tpr_list)):
            largeur = fpr_list[i] - fpr_list[i - 1]
            hauteur_moyenne = (tpr_list[i] + tpr_list[i - 1]) / 2.0
            auc += largeur * hauteur_moyenne

        self._metrics['roc_auc'].append(auc)
        return auc