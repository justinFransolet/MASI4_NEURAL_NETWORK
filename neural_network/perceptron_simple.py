from .formula import activate, ActivationFunction, matrix_confusion
from .history import History
from .metrics import MetricType


class PerceptronSimple:
    """
    Implémentation du perceptron simple de Rosenblatt.

    - Classification binaire uniquement
    - Fonction d'activation : seuil (step)
    - Apprentissage supervisé par correction directe
    - Pas de dérivée, pas de rétropropagation
    """

    def __init__(self, n_inputs: int, learning_rate: float = 1.0):
        """
        :param n_inputs: nombre d'entrées réelles x1..xn
        :param learning_rate: taux d'apprentissage η
        """
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate

        # Poids synaptiques w1..wn
        self.weights = [0.0 for _ in range(n_inputs)]

        # Biais = poids w0 associé à l'entrée fictive x0 = 1
        self.bias = 0.0

    def net_input(self, inputs: list[float]) -> float:
        """
        Calcule le potentiel p = w0 + w1*x1 + ... + wn*xn
        """
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Le perceptron attend {self.n_inputs} entrées, reçu {len(inputs)}.")

        z = self.bias
        for i in range(self.n_inputs):
            z += self.weights[i] * inputs[i]
        return z

    def predict(self, inputs: list[float]) -> int:
        """
        Prédiction binaire du perceptron :
        y = 1 si p >= 0, sinon 0
        """
        z = self.net_input(inputs)
        return int(activate(ActivationFunction.STEP, z))

    def train(
        self,
        x_train: list[list[float]],
        y_train: list[int] | list[list[int]],
        epochs: int = 100,
        metrics: tuple = (),
        verbose: bool = True
    ) -> History:
        """
        Entraîne le perceptron selon la règle de Rosenblatt.

        Algorithme :
        1. Pour chaque exemple
        2. Calculer la sortie y
        3. Calculer l'erreur e = d - y
        4. Si e != 0, corriger les poids :
              wi <- wi + η * e * xi
              b  <- b  + η * e
        5. Arrêt si nbErreurs = 0 sur une époque complète

        :param x_train: liste des entrées
        :param y_train: liste des sorties attendues (0/1), ex: [0,0,0,1]
                        ou format [[0],[0],[0],[1]]
        :param epochs: nombre max d'époques
        :param metrics: métriques à suivre
        :param verbose: affiche le détail de l'apprentissage
        :return: History
        """
        if len(x_train) != len(y_train):
            raise ValueError("x_train et y_train doivent avoir la même taille.")

        # Uniformiser y_train au format plat : [0,1,0,...]
        y_train_flat = []
        for y in y_train:
            if isinstance(y, list):
                if len(y) != 1:
                    raise ValueError("Le perceptron simple attend une seule sortie par échantillon.")
                y_train_flat.append(int(y[0]))
            else:
                y_train_flat.append(int(y))

        history = History(metrics)

        for epoch in range(1, epochs + 1):
            nb_errors = 0
            epoch_predictions = []

            if verbose:
                print(f"\n---- Epoch {epoch} ----")

            for x, d in zip(x_train, y_train_flat):
                y = self.predict(x)
                error = d - y

                if error != 0:
                    nb_errors += 1

                    # Mise à jour des poids w1..wn
                    for i in range(self.n_inputs):
                        self.weights[i] += self.learning_rate * error * x[i]

                    # Mise à jour du biais (équivaut à x0 = 1)
                    self.bias += self.learning_rate * error

                # On stocke la sortie après traitement de l'exemple
                epoch_predictions.append([float(y)])

                if verbose:
                    weights_str = ", ".join(f"{w:.2f}" for w in self.weights)
                    x_str = ", ".join(f"{v:.0f}" for v in x)
                    print(
                        f"x=[{x_str}], d={d}, y={y}, error={error}, "
                        f"bias={self.bias:.2f}, w=[{weights_str}]"
                    )

            if verbose:
                print(f"Errors this epoch: {nb_errors}")

            # Logging des métriques
            y_true_for_history = [[float(v)] for v in y_train_flat]

            if MetricType.MSE in metrics:
                history.log_mse(y_true_for_history, epoch_predictions)

            y_true_flat = [float(v) for v in y_train_flat]
            y_pred_flat = [pred[0] for pred in epoch_predictions]
            tp, tn, fp, fn = matrix_confusion(y_true_flat, y_pred_flat, threshold=0.5)

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

            # Critère d'arrêt du cours : nbErreurs == 0
            if nb_errors == 0:
                if verbose:
                    print("Training finished: no errors.")
                break

        return history

    def evaluate(self, x_test: list[list[float]], y_test: list[int] | list[list[int]]) -> dict:
        """
        Évalue le perceptron sur un jeu de test.
        Retourne accuracy + prédictions.
        """
        if len(x_test) != len(y_test):
            raise ValueError("x_test et y_test doivent avoir la même taille.")

        y_test_flat = []
        for y in y_test:
            if isinstance(y, list):
                if len(y) != 1:
                    raise ValueError("Le perceptron simple attend une seule sortie par échantillon.")
                y_test_flat.append(int(y[0]))
            else:
                y_test_flat.append(int(y))

        predictions = [self.predict(x) for x in x_test]
        correct = sum(1 for yt, yp in zip(y_test_flat, predictions) if yt == yp)
        accuracy = correct / len(y_test_flat) if y_test_flat else 0.0

        return {
            "accuracy": accuracy,
            "predictions": predictions
        }

    def decision_boundary(self) -> tuple[float, list[float]]:
        """
        Retourne le biais et les poids.
        Utile pour interprétation géométrique.
        """
        return self.bias, self.weights[:]

    def print_parameters(self) -> None:
        """
        Affiche les paramètres finaux du modèle.
        """
        print("Final parameters:")
        print(f"bias = {self.bias}")
        for i, w in enumerate(self.weights, start=1):
            print(f"w{i} = {w}")