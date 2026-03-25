# neural_network/formula.py
import math
from enum import Enum

# >@<----------------------------------->@<
#          Type de données
# >@<----------------------------------->@<

class ActivationFunction(Enum):
    """
    Enumération des fonctions d'activation.
    """
    STEP = 'step'
    IDENTITY = 'identity'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'

# >@<----------------------------------->@<
#          Fonctions d'activation
# >@<----------------------------------->@<

def step(x: float)-> float:
    return 1.0 if x>=0 else 0.0

def identity(x: float)-> float:
    return x

def sigmoid(x: float, c: float=1.0)-> float:
    x = max(-700.0, min(700.0, x))
    return 1.0 / (1.0 + math.exp(-c * x))

def tanh(x: float)-> float:
    return math.tanh(x)

def activate(func_type: ActivationFunction, x: float, c=1.0)-> float:
    """
    Fonction d'activation.

    :param func_type: Choix de la fonction activation.
    :param x: Valeur d'entrée.
    :param c: Uniquement utilisé pour la fonction sigmoïde. Par defaut 1.0.
    :raise ValueError: Si la fonction n'est pas reconnue.
    :return: Valeur de sortie de la fonction d'activation.
    """
    if func_type == ActivationFunction.STEP:
        return step(x)
    elif func_type == ActivationFunction.IDENTITY:
        return identity(x)
    elif func_type == ActivationFunction.SIGMOID:
        return sigmoid(x, c)
    elif func_type == ActivationFunction.TANH:
        return tanh(x)
    else:
        raise ValueError("Fonction non reconnue")

# >@<----------------------------------->@<
#                Dérivées
# >@<----------------------------------->@<

def step_derivative()-> float:
    return 1.0

def identity_derivative()-> float:
    return 1.0

def sigmoid_derivative(y: float, c: float)-> float:
    return c * y * (1.0 - y)

def tanh_derivative(y: float)-> float:
    return 1.0 - y**2

def derivative(func_type: ActivationFunction, y: float, c=1.0)-> float:
    """
    Dérivées calculées à partir de la sortie 'y' de la fonction d'activation.
    Pour l'identité et le step, on utilise 1.0 (ou l'erreur brute) selon les règles.

    :param func_type: Choix de la fonction activation.
    :param y: Valeur de sortie de la fonction d'activation.
    :param c: Uniquement utilisé pour la fonction sigmoïde. Par defaut 1.0.
    :raise ValueError: Si la fonction n'est pas reconnue.
    :return: Valeur de dérivée de la fonction d'activation.
    """
    if func_type == ActivationFunction.STEP:
        return step_derivative()
    elif func_type == ActivationFunction.IDENTITY:
        return identity_derivative()
    elif func_type == ActivationFunction.SIGMOID:
        return sigmoid_derivative(y, c)
    elif func_type == ActivationFunction.TANH:
        return tanh_derivative(y)

    raise ValueError("Fonction non reconnue")

# >@<----------------------------------->@<
#                Vecteurs
# >@<----------------------------------->@<

def dot_product(v1: list[float], v2: list[float])-> float:
    """
    Produit scalaire de deux vecteurs.
    :param v1: Premier vecteur. Format : (x1, x2, ..., xn)
    :param v2: Deuxième vecteur. Format : (y1, y2, ..., yn)
    :return: Produit scalaire des deux vecteurs.
    """
    return sum(x * y for x, y in zip(v1, v2))

def matrix_transpose(matrix: list[list[float]])-> list[list[float]]:
    """
    Transposée d'une matrice.
    :param matrix: Matrice à transposer. Format : [[x1, x2, ..., xn],[y1,y2,...,yn],...,[z1,z2,...,zn]]
    :return: Matrice transposée. Format : [[x1,y1,...,z1],[x2,y2,...,z2],...,[xn,yn,...,zn]]
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_confusion(y_true: list[float], y_pred_probs: list[float], threshold=0.5):
    """
    Calcule TP, TN, FP, FN en appliquant un seuil sur les probabilités.
    TP => True Positive | TN => True Negative | FP => False Positive | FN => False Negative

    :param y_true: Les valeurs réelles.
    :param y_pred_probs: La probabilité que la valeur appartient à la classe.
    :param threshold: La valeur limite pour être dans une catégorie
    :raise ValueError: Si les listes ne sont pas de même taille.
    """
    if len(y_true) != len(y_pred_probs):
        raise ValueError("Les listes doivent avoir la même taille.")

    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred_probs):
        # Binarisation de la prédiction selon le seuil
        yp = 1 if yp >= threshold else 0

        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    return tp, tn, fp, fn