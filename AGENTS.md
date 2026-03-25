# Fiche Agent : Projet NN Scratch

Ce document définit les spécifications, contraintes et objectifs pour l'implémentation de la bibliothèque de réseaux de neurones dans le cadre du cours de **Réseaux de neurones(2025-2026)**.

---

## 🎯 Objectif Principal
Développer une bibliothèque complète de neurones artificiels, du **Perceptron simple** au **Perceptron Multicouche (PMC)**, sans utiliser de bibliothèques de calcul avancées, afin de valider des modèles de classification et de régression.

## 🛠️ Stack Technique & Contraintes
L'aspect critique de ce projet est l'interdiction de bibliothèques tierces pour le calcul matriciel.

* **Langage :** Python 3.14 (via Notebooks Jupyter).
* **Calcul :** **INTERDIT** : NumPy, TensorFlow, PyTorch, Scikit-learn.
    * *Note : Les opérations mathématiques et la gestion des matrices doivent être codées en Python pur (listes, boucles, etc.).*
* **Autorisé :** Bibliothèques standards pour la lecture de fichiers CSV et bibliothèques de visualisation (Matplotlib/Seaborn).
* **Qualité :** Code orienté objet (Classes `Layer` et `Network`), documenté en Markdown pour expliquer les points critiques.

---

## 🧠 Modèles à Implémenter

| Modèle                    | Fonction d'Activation  | Règle d'Apprentissage / Stratégie                                   |
|:--------------------------|:-----------------------|:--------------------------------------------------------------------|
| **Perceptron Simple**     | Seuillage binaire      | Règle de Rosenblatt : $w_i(t+1) = w_i(t) + \eta(d(k) - y(k))x_i(k)$ |
| **Perceptron (Gradiant)** | Identité               | Full-Batch (Mise à jour globale)                                    |
| **ADALINE**               | Identité               | Stochastique (Règle Delta de Widrow-Hoff)                           |
| **Monocouche**            | Parallèle (M neurones) | Classification multi-classes (Linéairement séparables)              |
| **PMC (Multicouche)**     | Sigmoïde / Tanh        | **Rétropropagation du gradient**                                    |

---

## 🧪 Plan de Validation

### 1. Tests Logiques et Fondamentaux
* **Portes Logiques :** Validation sur le ET (linéaire) et le XOR (non-linéaire, spécifique au PMC).
* **Classification :** Tests sur données linéairement séparables (3 et 4 classes).
* **Régression :** Modèles linéaires et non-linéaires.

### 2. Cas Réel : Langage des Signes
* **Données :** 300 photos (Lettres A à E).
* **Input :** 42 entrées (21 points de coordonnées x,y par image).
* **Split :** 250 images pour l'entraînement / 50 pour la validation.

---

## 📊 Livrables Attendus
* **Visualisations :** * Graphiques des frontières de décision.
    * Courbes d'évolution de l'erreur (Loss curves).
    * Résultats de régression.
* **Code :** Une archive structurée contenant les notebooks et les interprétations des résultats.

---