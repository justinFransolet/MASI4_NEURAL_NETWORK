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


def print_multiclass_predictions(model, x_data, y_data):
    """
    Affiche les prédictions d'un modèle multiclasses.
    La classe prédite et la classe attendue sont obtenues avec argmax.
    """
    for x, y_true in zip(x_data, y_data):
        y_pred = model.predict(x)
        pred_class = y_pred.index(max(y_pred))
        true_class = y_true.index(max(y_true))
        print(f"x={x} -> y_pred={y_pred} -> classe_pred={pred_class} | classe_attendue={true_class}")


def multiclass_accuracy(model, x_data, y_data):
    """
    Calcule l'accuracy pour un problème multiclasses.
    """
    correct = 0
    total = len(x_data)

    for x, y_true in zip(x_data, y_data):
        y_pred = model.predict(x)
        pred_class = y_pred.index(max(y_pred))
        true_class = y_true.index(max(y_true))

        if pred_class == true_class:
            correct += 1

    return correct / total if total > 0 else 0.0


def plot_multiclass_decision_boundaries_2d(model, x_data, y_data, title: str = "Frontières de décision multiclasses", threshold: float = 0.5):
    """
    Affiche un dataset 2D multiclasses et les droites de décision
    d'un perceptron monocouche.
    """
    plt.figure(figsize=(8, 6))

    n_classes = len(y_data[0])
    markers = ["o", "s", "^", "D", "x", "*"]

    class_points_x = [[] for _ in range(n_classes)]
    class_points_y = [[] for _ in range(n_classes)]

    for x, y in zip(x_data, y_data):
        true_class = y.index(max(y))
        class_points_x[true_class].append(x[0])
        class_points_y[true_class].append(x[1])

    for c in range(n_classes):
        plt.scatter(
            class_points_x[c],
            class_points_y[c],
            label=f"Classe {c + 1}",
            marker=markers[c % len(markers)]
        )

    layer = model._NeuralNetwork__layers[0]

    x1_min = min(x[0] for x in x_data) - 1
    x1_max = max(x[0] for x in x_data) + 1
    x1_values = [x1_min, x1_max]

    for j in range(len(layer.weights)):
        bias = layer.biases[j]
        w1 = layer.weights[j][0]
        w2 = layer.weights[j][1]

        if abs(w2) > 1e-12:
            x2_values = [(threshold - bias - w1 * x1) / w2 for x1 in x1_values]
            plt.plot(x1_values, x2_values, label=f"P{j + 1}")
        elif abs(w1) > 1e-12:
            x_vertical = (threshold - bias) / w1
            plt.axvline(x=x_vertical, label=f"P{j + 1}")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_5x5_samples_with_predictions(model, x_data, y_data, max_samples: int = 12):
    """
    Affiche des échantillons 5x5 avec leur classe vraie et leur classe prédite.
    Adapté à la table 3.5.
    """
    n_samples = min(max_samples, len(x_data))
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows))
    axes = axes.flatten()

    for i in range(n_samples):
        x = x_data[i]
        y_true = y_data[i]
        y_pred = model.predict(x)

        true_class = y_true.index(max(y_true))
        pred_class = y_pred.index(max(y_pred))

        image = []
        for r in range(5):
            row = []
            for c in range(5):
                row.append(x[r * 5 + c])
            image.append(row)

        axes[i].imshow(image, cmap="gray_r")
        axes[i].set_title(f"Vraie: {true_class} / Préd: {pred_class}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_decision_boundary_history(history, x_data, y_data, title: str = "Évolution de la frontière de décision"):
    """
    Affiche l'évolution de la frontière de décision au fil des époques
    pour un modèle à une seule couche et un seul neurone.
    """
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

    x1_values = [min(x[0] for x in x_data) - 1, max(x[0] for x in x_data) + 1]

    for i, params in enumerate(history.params_history, start=1):
        bias = params["bias"]
        w1 = params["weights"][0]
        w2 = params["weights"][1]

        if abs(w2) > 1e-12:
            x2_values = [-(bias + w1 * x1) / w2 for x1 in x1_values]
            plt.plot(x1_values, x2_values, label=f"epoch {i}")
        elif abs(w1) > 1e-12:
            x_vertical = -bias / w1
            plt.axvline(x=x_vertical, label=f"epoch {i}")

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_binary_decision_regions_2d(model, x_data, y_data, title: str = "Régions de décision binaires", threshold: float = 0.5, step: float = 0.02):
    """
    Affiche les régions de décision pour une classification binaire 2D.
    Adapté aux perceptrons multicouches.
    """
    import matplotlib.pyplot as plt

    x1_min = min(x[0] for x in x_data) - 1
    x1_max = max(x[0] for x in x_data) + 1
    x2_min = min(x[1] for x in x_data) - 1
    x2_max = max(x[1] for x in x_data) + 1

    xx = []
    yy = []
    x1 = x1_min
    while x1 <= x1_max:
        row_x = []
        row_y = []
        x2 = x2_min
        while x2 <= x2_max:
            row_x.append(x1)
            row_y.append(x2)
            x2 += step
        xx.append(row_x)
        yy.append(row_y)
        x1 += step

    zz = []
    for i in range(len(xx)):
        row_z = []
        for j in range(len(xx[i])):
            pred = model.predict([xx[i][j], yy[i][j]])[0]
            row_z.append(1 if pred >= threshold else 0)
        zz.append(row_z)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=0.3)

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

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiclass_decision_regions_2d(model, x_data, y_data, title: str = "Régions de décision multiclasses", step: float = 0.02):
    """
    Affiche les régions de décision pour une classification multiclasses 2D.
    La classe prédite est obtenue avec argmax.
    """
    import matplotlib.pyplot as plt

    x1_min = min(x[0] for x in x_data) - 1
    x1_max = max(x[0] for x in x_data) + 1
    x2_min = min(x[1] for x in x_data) - 1
    x2_max = max(x[1] for x in x_data) + 1

    xx = []
    yy = []
    x1 = x1_min
    while x1 <= x1_max:
        row_x = []
        row_y = []
        x2 = x2_min
        while x2 <= x2_max:
            row_x.append(x1)
            row_y.append(x2)
            x2 += step
        xx.append(row_x)
        yy.append(row_y)
        x1 += step

    zz = []
    for i in range(len(xx)):
        row_z = []
        for j in range(len(xx[i])):
            pred = model.predict([xx[i][j], yy[i][j]])
            pred_class = pred.index(max(pred))
            row_z.append(pred_class)
        zz.append(row_z)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=0.3)

    n_classes = len(y_data[0])
    markers = ["o", "s", "^", "D", "x", "*"]

    class_points_x = [[] for _ in range(n_classes)]
    class_points_y = [[] for _ in range(n_classes)]

    for x, y in zip(x_data, y_data):
        true_class = y.index(max(y))
        class_points_x[true_class].append(x[0])
        class_points_y[true_class].append(x[1])

    for c in range(n_classes):
        plt.scatter(
            class_points_x[c],
            class_points_y[c],
            label=f"Classe {c + 1}",
            marker=markers[c % len(markers)]
        )

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regression_curve(model, x_data, y_data, title: str = "Régression non linéaire"):
    """
    Affiche les valeurs réelles et la courbe prédite triée selon x.
    Adapté aux problèmes de régression 1D.
    """
    import matplotlib.pyplot as plt

    data = []
    for x, y in zip(x_data, y_data):
        pred = model.predict(x)
        data.append((x[0], y[0], pred[0]))

    data.sort(key=lambda item: item[0])

    x_values = [item[0] for item in data]
    y_true_values = [item[1] for item in data]
    y_pred_values = [item[2] for item in data]

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



def plot_multiclass_decision_regions_2d_with_uncertainty(
    model,
    x_data,
    y_data,
    title: str = "Régions de décision multiclasses",
    step: float = 0.0025,
    margin: float = 0.15
):
    """
    Affiche les régions de décision pour une classification multiclasses 2D,
    avec une zone blanche d'incertitude lorsque les deux meilleures sorties
    du réseau sont trop proches.

    - step : finesse du quadrillage
    - margin : différence minimale entre les 2 plus grandes sorties pour
               attribuer clairement une classe
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    x1_min = min(x[0] for x in x_data) - 1
    x1_max = max(x[0] for x in x_data) + 1
    x2_min = min(x[1] for x in x_data) - 1
    x2_max = max(x[1] for x in x_data) + 1

    xx = []
    yy = []

    x1 = x1_min
    while x1 <= x1_max:
        row_x = []
        row_y = []
        x2 = x2_min
        while x2 <= x2_max:
            row_x.append(x1)
            row_y.append(x2)
            x2 += step
        xx.append(row_x)
        yy.append(row_y)
        x1 += step

    zz = []
    for i in range(len(xx)):
        row_z = []
        for j in range(len(xx[i])):
            pred = model.predict([xx[i][j], yy[i][j]])

            # indices triés par valeur décroissante
            sorted_indices = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
            best_idx = sorted_indices[0]
            second_idx = sorted_indices[1]

            best_value = pred[best_idx]
            second_value = pred[second_idx]

            # zone blanche si incertitude
            if best_value - second_value < margin:
                row_z.append(-1)
            else:
                row_z.append(best_idx)
        zz.append(row_z)

    # -1 = blanc, puis classes 0,1,2
    cmap = ListedColormap(["white", "#440154", "#21918c", "#fde725"])

    # contourf attend des indices 0..N, donc on décale de +1
    zz_shifted = []
    for row in zz:
        zz_shifted.append([v + 1 for v in row])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz_shifted, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap, alpha=0.75)

    n_classes = len(y_data[0])
    markers = ["o", "s", "^", "D", "x", "*"]

    class_points_x = [[] for _ in range(n_classes)]
    class_points_y = [[] for _ in range(n_classes)]

    for x, y in zip(x_data, y_data):
        true_class = y.index(max(y))
        class_points_x[true_class].append(x[0])
        class_points_y[true_class].append(x[1])

    for c in range(n_classes):
        plt.scatter(
            class_points_x[c],
            class_points_y[c],
            label=f"Classe {c + 1}",
            marker=markers[c % len(markers)],
            s=30
        )

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()