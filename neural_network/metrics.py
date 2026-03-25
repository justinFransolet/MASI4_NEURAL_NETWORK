# metrics.py

from enum import Enum

class MetricType(Enum):
    MSE = 'mse'
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1_SCORE = 'f1_score'
    ROC_AUC = 'roc_auc'

def calculate_mse(y_true: list[float], y_pred: list[float])-> float:
    """
    Calcule l'Erreur Quadratique Moyenne (MSE).

    :param y_true: La valeur réelle de l'échantillion.
    :param y_pred: La valeur prédite de l'échantillion.
    """
    n = len(y_true)
    if n == 0:
        raise ArithmeticError("Impossible de diviser par 0.")
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))

def calculate_accuracy(tp: int, tn: int, fp: int, fn: int)-> float:
    """
    Calculer le taux de fiabilité du modèle.

    :param tp: True Positive
    :param tn: True Negative
    :param fp: False Positive
    :param fn: False Negative
    :return: Le taux de fiabilité.
    """
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0

def calculate_precision(tp: int, fp: int)-> float:
    """
    Calculer le taux de précision du modèle.

    :param tp: True Positive
    :param fp: False Positive
    :return: Le taux de précision.
    """
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calculate_recall(tp: int, fn: int)-> float:
    """
    Calculer la valeur "recall" est le taux de vrai positif sur la quantité réelle de positif.

    :param tp: True Positive
    :param fn: False Negative
    :return: Le taux de précision.
    """
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calculate_f1_score(tp: int, fp: int, fn: int)-> float:
    """
    Calculer la moyenne harmonique entre la précision et le rappel.

    :param tp: True Positive
    :param fp: False Positive
    :param fn: False Negative
    :return: Le taux de précision.
    """

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)

    if (precision + recall) > 0:
        return 2 * ((precision * recall) / (precision + recall))
    else:
        return 0.0

def calculate_roc_auc(y_true: list[float], y_pred_probs: list[float])-> float:
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

    return auc