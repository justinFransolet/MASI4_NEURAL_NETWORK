import csv
import matplotlib.pyplot as plt
from .metrics import MetricType


def load_csv_dataset(path: str, target_columns: int = 1, has_header: bool = False):
    """
    Charge un dataset CSV.

    :param path: chemin vers le fichier CSV
    :param target_columns: nombre de colonnes de sortie à la fin de chaque ligne
    :param has_header: True si le CSV contient une ligne d'en-tête
    :return: tuple (x_data, y_data)
    """
    x_data = []
    y_data = []

    with open(path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)

        if has_header:
            next(reader)

        for row in reader:
            if not row:
                continue

            values = [float(v) for v in row]
            x_data.append(values[:-target_columns])
            y_data.append(values[-target_columns:])

    return x_data, y_data


def plot_metric(history, metric_type: MetricType, title: str):
    """
    Affiche l'évolution d'une métrique sur les époques.
    """
    values = history.metrics[metric_type.value]

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(values) + 1), values, label=metric_type.value.upper())
    plt.title(title)
    plt.xlabel("Époque")
    plt.ylabel(metric_type.value.upper())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_binary_classification_points(x_data, y_data, title: str = "Classification"):
    """
    Affiche un nuage de points pour une classification binaire 2D.
    """
    points_0_x = []
    points_0_y = []
    points_1_x = []
    points_1_y = []

    for x, y in zip(x_data, y_data):
        if y[0] == 0:
            points_0_x.append(x[0])
            points_0_y.append(x[1])
        else:
            points_1_x.append(x[0])
            points_1_y.append(x[1])

    plt.figure(figsize=(7, 5))
    plt.scatter(points_0_x, points_0_y, label="Classe 0")
    plt.scatter(points_1_x, points_1_y, label="Classe 1")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, x_data, y_data, title: str = "Frontière de décision", threshold: float = 0.5):
    """
    Affiche les points d'un dataset binaire 2D et la frontière de décision
    d'un réseau à une seule couche contenant un seul neurone.

    Pour une activation IDENTITY, la frontière correcte est :
        w1*x1 + w2*x2 + b = threshold
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    points_0_x = []
    points_0_y = []
    points_1_x = []
    points_1_y = []

    for x, y in zip(x_data, y_data):
        if y[0] == 0:
            points_0_x.append(x[0])
            points_0_y.append(x[1])
        else:
            points_1_x.append(x[0])
            points_1_y.append(x[1])

    plt.scatter(points_0_x, points_0_y, label="Classe 0")
    plt.scatter(points_1_x, points_1_y, label="Classe 1")

    layer = model._NeuralNetwork__layers[0]
    bias = layer.biases[0]
    w1 = layer.weights[0][0]
    w2 = layer.weights[0][1]

    x1_values = [min(x[0] for x in x_data) - 1, max(x[0] for x in x_data) + 1]

    if abs(w2) > 1e-12:
        x2_values = [(threshold - bias - w1 * x1) / w2 for x1 in x1_values]
        plt.plot(x1_values, x2_values, label="Frontière")
    elif abs(w1) > 1e-12:
        x_vertical = (threshold - bias) / w1
        plt.axvline(x=x_vertical, label="Frontière")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regression_result(x_data, y_data, y_pred, title: str = "Régression"):
    """
    Affiche les valeurs réelles et les prédictions pour un problème de régression.
    """
    x_values = [x[0] for x in x_data]
    y_true_values = [y[0] for y in y_data]
    y_pred_values = [y[0] for y in y_pred]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_true_values, label="Valeurs réelles")
    plt.plot(x_values, y_pred_values, label="Prédictions")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_predictions(model, x_data, y_data=None, classification: bool = True, threshold: float = 0.5):
    """
    Affiche les prédictions du modèle.
    """
    for i, x in enumerate(x_data):
        output = model.predict(x)

        if classification:
            predicted_class = [1 if v >= threshold else 0 for v in output]
            if y_data is not None:
                print(f"x={x} -> y_pred={output} -> classe={predicted_class} | attendu={y_data[i]}")
            else:
                print(f"x={x} -> y_pred={output} -> classe={predicted_class}")
        else:
            if y_data is not None:
                print(f"x={x} -> y_pred={output} | attendu={y_data[i]}")
            else:
                print(f"x={x} -> y_pred={output}")
