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
        return sigmoid(c)
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

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def matrix_transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]