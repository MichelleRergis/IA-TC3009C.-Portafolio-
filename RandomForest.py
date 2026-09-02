"""
RandomForest.py
=================
Implementacion de un algoritmo de Random Forest sin usar ninguna
biblioteca o framework de aprendizaje maquina ni de estadistica avanzada
para el algoritmo en si.

El nucleo del modelo (arbol de decision, impureza Gini, bagging, votacion,
importancia de variables y metricas de precision/recall/F1) esta programado
a mano con la biblioteca estandar de Python (math, random, collections).

Para todo lo que NO es el algoritmo -carga de datos, limpieza, analisis
exploratorio (EDA), tablas descriptivas y graficas- si se usan librerias
(pandas, numpy, matplotlib, seaborn) para que el resultado sea mas claro
y profesional. La UNICA parte que usa scikit-learn es el calculo/grafica
de la matriz de confusion (sklearn.metrics.confusion_matrix y
ConfusionMatrixDisplay), solicitada explicitamente para tener una
matriz de confusion con el formato estandar de la libreria; el modelo
en si (el Random Forest) sigue sin usar sklearn.

Ademas de la evaluacion en el conjunto de prueba, el script genera dos
figuras de "curvas de entrenamiento" (estilo accuracy/loss por epoca,
con paneles lado a lado y lineas Train/Validation):
    1. learning_curves_trees.png      -> accuracy y log-loss vs.
       numero de arboles del bosque (el "numero de arboles" hace aqui
       el papel de las epocas: a mas arboles, mas "entrenado" esta el
       bosque).
    2. learning_curves_train_size.png -> accuracy y log-loss vs.
       tamano del conjunto de entrenamiento usado.
Ambas comparan el desempenio en entrenamiento (Train) contra el de
prueba (Validation), para poder ver si el modelo esta sobreajustando
(overfitting) o si le faltarian mas datos/arboles. Como el Random
Forest no tiene una "loss" nativa (no es un modelo que se entrena
minimizando una funcion de perdida por epoca), la loss se calcula a
mano como log-loss (entropia cruzada) a partir de la proporcion de
votos que el bosque le da a la clase real de cada ejemplo. Con pocos
arboles/pocos datos el bagging tiene bastante varianza de una corrida
a otra -y el conjunto de prueba tambien es chico, asi que un solo
ejemplo dificil pesa bastante-, asi que cada punto de ambas curvas es
el PROMEDIO de N_CURVE_RUNS corridas con semillas distintas, donde
CADA corrida rehace tambien el split train/test completo (no solo el
bosque; ver compute_trees_learning_curve / compute_train_size_learning_curve),
en vez de una sola corrida atada a un unico split fijo. Los
hiperparametros del bosque (numero de arboles, profundidad maxima,
min_samples_split/leaf, max_features) estan centralizados en las
constantes FOREST_* al inicio del script, para poder ajustar la
regularizacion (y con ella la brecha train/validation) desde un solo
lugar.

Problema: clasificar la dieta de un dinosaurio (carnivorous / herbivorous
/ omnivorous) a partir de variables morfologicas, temporales y de contexto
geografico/taxonomico.

Dataset: dinosaurs.csv (Natural History Museum "Dino Directory", via
Kaggle "Jurassic Park - The Exhaustive Dinosaur Dataset":
https://www.kaggle.com/datasets/kjanjua/jurassic-park-the-exhaustive-dinosaur-dataset)

Columnas originales del CSV (texto libre, NO limpio):
    name, diet, period, lived_in, type, length, taxonomy, named_by,
    species, link

Ninguna de estas columnas viene lista para usarse directamente en el
modelo: 'length' es texto como "8.0m", 'period' es texto como
"Early Jurassic 199-189 million years ago" (el rango de millones de
anios esta embebido en la frase), y no existe una columna de "clase"
taxonomica ni de coordenadas: 'taxonomy' es una cadena con el arbol
taxonomico completo (ej. "Dinosauria Saurischia Theropoda ...") y
'lived_in' es el pais/region.

Por eso, antes de entrenar, este script DERIVA las variables que usa
el algoritmo a partir de esas columnas de texto:
    - length_m  <- parseado de 'length'            (ej. "8.0m" -> 8.0)
    - max_ma    <- parseado de 'period'             (millones de anios,
    - min_ma    <-    inicio y fin del rango)
    - class     <- segundo token de 'taxonomy'      (Saurischia /
                     Ornithischia: la gran division de los dinosaurios)
    - type      <- tal cual (sauropod, large theropod, ceratopsian, ...)
    - lived_in  <- tal cual (pais/region, como variable categorica)

Autor: Michelle Rergis Novelo
Materia: TC3006C - IA Modulo 2.

Como ejecutar:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python3 RandomForest.py
"""

import math
import os
import random
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: funciona en cualquier consola/servidor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

sns.set_theme(style="whitegrid")

# Configuracion inicial

RANDOM_SEED = 42
DATA_PATH = "dinosaurs.csv"
TARGET = "diet"
TEST_RATIO = 0.2  # proporcion del dataset reservada para prueba

# Hiperparametros del Random Forest, centralizados aqui para poder
# ajustar la regularizacion (y por lo tanto la brecha train/validation)
# desde un solo lugar en vez de tener numeros sueltos repetidos por
# todo el script.
FOREST_N_TREES = 60
FOREST_MAX_DEPTH = 10
FOREST_MIN_SAMPLES_SPLIT = 10
FOREST_MIN_SAMPLES_LEAF = 4
FOREST_MAX_FEATURES = "sqrt"  # "sqrt", "log2", o "all"

# Numero de corridas (con distinta semilla) que se promedian en cada
# punto de las curvas de entrenamiento (accuracy/loss vs. numero de
# arboles y vs. tamano de entrenamiento), para suavizar el ruido propio
# del bagging cuando hay pocos arboles/pocos datos. Cada corrida usa
# ademas su propio split train/test (no solo su propio bosque), para
# tambien promediar el ruido que viene de que el conjunto de prueba es
# chico.
N_CURVE_RUNS = 5

# Columnas que debe traer el CSV crudo (texto libre, sin procesar).
RAW_COLUMNS = ["name", "diet", "period", "lived_in", "type", "length", "taxonomy"]

# Variables ya derivadas/limpias que usa el algoritmo.
NUMERIC_FEATURES = ["length_m", "max_ma", "min_ma"]
CATEGORICAL_FEATURES = ["type", "class", "lived_in"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_DIETS = {"carnivorous", "herbivorous", "omnivorous"}

RESULTS_DIR = "results"
IMG_DIR = os.path.join(RESULTS_DIR, "img")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# 1. Carga, limpieza y analisis exploratorio

def parse_length_m(value):
    """Convierte texto como '8.0m' o '1.5 m' en un float (metros).
    Devuelve None si no se encuentra un numero."""
    if pd.isna(value):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def parse_period_ma(value):
    """Extrae el rango de millones de anios (max_ma, min_ma) de textos
    como 'Early Jurassic 199-189 million years ago'. Si solo hay un
    numero (ej. 'Early Jurassic 190 million years ago'), usa el mismo
    valor para max_ma y min_ma. Devuelve (None, None) si no hay numeros."""
    if pd.isna(value):
        return (None, None)
    numbers = re.findall(r"(\d+(?:\.\d+)?)", str(value))
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    if len(numbers) == 1:
        return float(numbers[0]), float(numbers[0])
    return (None, None)


def parse_class(taxonomy_value):
    """Extrae la gran division taxonomica (Saurischia / Ornithischia)
    a partir de la cadena completa de 'taxonomy', p. ej.
    'Dinosauria Saurischia Theropoda ...' -> 'Saurischia'."""
    if pd.isna(taxonomy_value):
        return None
    tokens = str(taxonomy_value).split()
    if len(tokens) > 1:
        return tokens[1]
    return tokens[0] if tokens else None


def load_dataframe(path):
    df = pd.read_csv(path)  # Carga el CSV crudo con pandas

    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Al archivo '{path}' le faltan columnas esperadas: {missing_cols}.\n"
            f"Columnas encontradas: {list(df.columns)}"
        )

    # Normaliza y filtra la variable objetivo: solo las 3 dietas validas
    # (se descartan 'unknown', combinaciones como 'herbivorous/omnivorous', etc.)
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df = df[df[TARGET].isin(VALID_DIETS)]

    # Deriva las variables numericas/categoricas limpias a partir de
    # las columnas de texto libre del CSV original.
    df["length_m"] = df["length"].apply(parse_length_m)
    period_ranges = df["period"].apply(parse_period_ma)
    df["max_ma"] = period_ranges.apply(lambda t: t[0])
    df["min_ma"] = period_ranges.apply(lambda t: t[1])
    df["class"] = df["taxonomy"].apply(parse_class)

    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["lived_in"] = df["lived_in"].astype(str).str.strip()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Descarta filas donde no se pudo derivar alguna variable necesaria
    # (longitud/periodo faltante o mal formado, taxonomia vacia, etc.)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]
    df = df.reset_index(drop=True)
    return df


def run_eda(df):
    """Genera tablas descriptivas y graficas exploratorias del dataset,
    guardandolas en results/."""

    os.makedirs(IMG_DIR, exist_ok=True)

    print(f"      Registros validos y limpios (listos para el modelo): {len(df)}")

    # Tabla describe() de variables numericas
    desc = df[NUMERIC_FEATURES].describe().T
    desc.to_csv(os.path.join(RESULTS_DIR, "eda_describe_numericas.csv"))
    print("\n      Estadisticas descriptivas (variables numericas):")
    print(desc.round(2).to_string())

    # Distribucion de la variable objetivo
    dist = df[TARGET].value_counts()
    dist_pct = (dist / len(df) * 100).round(1)
    print("\n      Distribucion de la dieta:")
    for label in dist.index:
        print(f"        - {label:12s}: {dist[label]:4d} ({dist_pct[label]}%)")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    palette = {"herbivorous": "#55A868", "carnivorous": "#C44E52", "omnivorous": "#8172B2"}
    sns.barplot(x=dist.index, y=dist.values, hue=dist.index, legend=False,
                palette=[palette.get(l, "#4C72B0") for l in dist.index], ax=ax)
    ax.set_ylabel("Numero de registros")
    ax.set_xlabel("")
    ax.set_title(f"Distribucion de la dieta en el dataset (n={len(df)})")
    for i, v in enumerate(dist.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "class_distribution.png"), dpi=160)
    plt.close(fig)

    # Distribucion del tipo anatomico
    type_dist = df["type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=type_dist.values, y=type_dist.index, color="#4C72B0", ax=ax)
    ax.set_xlabel("Numero de registros")
    ax.set_title("Distribucion por tipo anatomico")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "type_distribution.png"), dpi=160)
    plt.close(fig)

    # Longitud corporal por dieta (boxplot)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    diet_order = [d for d in ["carnivorous", "herbivorous", "omnivorous"]
                  if d in df[TARGET].unique()]
    sns.boxplot(data=df, x=TARGET, y="length_m", order=diet_order, hue=TARGET,
                legend=False, palette=palette, ax=ax)
    ax.set_title("Longitud corporal por tipo de dieta")
    ax.set_xlabel("")
    ax.set_ylabel("Longitud (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "length_by_diet_boxplot.png"), dpi=160)
    plt.close(fig)

    # Correlacion entre variables numericas
    corr = df[NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(5, 4.3))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlacion entre variables numericas")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"), dpi=160)
    plt.close(fig)

    # Top paises por numero de registros
    top_countries = df["lived_in"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=top_countries.values, y=top_countries.index, color="#4C72B0", ax=ax)
    ax.set_xlabel("Numero de registros")
    ax.set_title("Top 10 paises/regiones con mas hallazgos")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "top_countries.png"), dpi=160)
    plt.close(fig)

    # Tabla cruzada tipo anatomico x dieta
    cross = pd.crosstab(df["type"], df[TARGET])
    cross.to_csv(os.path.join(RESULTS_DIR, "eda_type_vs_diet.csv"))

    print("\n      Graficas de EDA guardadas en results/img/:")
    print("        - class_distribution.png")
    print("        - type_distribution.png")
    print("        - length_by_diet_boxplot.png")
    print("        - correlation_heatmap.png")
    print("        - top_countries.png")


# 2. Particion entrenamiento / prueba (estratificada)

def dataframe_to_records(df):
    """Convierte el DataFrame a una lista de diccionarios: a partir de
    aqui el algoritmo trabaja solo con estructuras nativas de Python."""
    return df.to_dict(orient="records")


def train_test_split_stratified(data, test_ratio=0.2, seed=RANDOM_SEED):
    """Separa los datos en entrenamiento/prueba manteniendo la misma
    proporcion de clases en ambos conjuntos (split estratificado).
    Implementado a mano (sin sklearn.train_test_split)."""
    rnd = random.Random(seed)
    by_class = {}
    for row in data:
        by_class.setdefault(row[TARGET], []).append(row)

    train, test = [], []
    for label, items in by_class.items():
        items = items[:]
        rnd.shuffle(items)
        n_test = max(1, round(len(items) * test_ratio))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


# 3. Arbol de decision (CART simplificado, impureza Gini)

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    impurity = 1.0
    for c in counts.values():
        p = c / n
        impurity -= p * p
    return impurity


def weighted_gini(left_labels, right_labels):
    n = len(left_labels) + len(right_labels)
    if n == 0:
        return 0.0
    g_left = gini_impurity(left_labels)
    g_right = gini_impurity(right_labels)
    return (len(left_labels) * g_left + len(right_labels) * g_right) / n


class Node:
    __slots__ = (
        "feature", "is_numeric", "threshold", "category",
        "left", "right", "prediction", "n_samples",
    )

    def __init__(self):
        self.feature = None
        self.is_numeric = None
        self.threshold = None
        self.category = None
        self.left = None
        self.right = None
        self.prediction = None
        self.n_samples = 0

    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    """Arbol de clasificacion binario construido con feature + punto de corte
    que maximiza la reduccion de impureza Gini.
    Para el Random Forest, en cada division solo se
    evalua un subconjunto aleatorio de variables (n_features_sample)."""

    def __init__(self, max_depth=10, min_samples_split=10,
                 min_samples_leaf=4, n_features_sample=None,
                 max_thresholds=20, rnd=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features_sample = n_features_sample or len(FEATURES)
        self.max_thresholds = max_thresholds
        self.rnd = rnd or random
        self.root = None
        self.feature_importance_ = {f: 0.0 for f in FEATURES}

    def fit(self, data):
        self.root = self._build(data, depth=0)
        return self

    @staticmethod
    def _majority(labels):
        return Counter(labels).most_common(1)[0][0]

    def _build(self, data, depth):
        node = Node()
        node.n_samples = len(data)
        labels = [r[TARGET] for r in data]
        node.prediction = self._majority(labels)

        if (depth >= self.max_depth
                or len(data) < self.min_samples_split
                or gini_impurity(labels) == 0.0):
            return node

        split = self._best_split(data)
        if split is None:
            return node

        feature, is_numeric, threshold, category, left_data, right_data, gain = split
        if len(left_data) < self.min_samples_leaf or len(right_data) < self.min_samples_leaf:
            return node

        self.feature_importance_[feature] += gain * len(data)

        node.feature = feature
        node.is_numeric = is_numeric
        node.threshold = threshold
        node.category = category
        node.left = self._build(left_data, depth + 1)
        node.right = self._build(right_data, depth + 1)
        return node

    def _best_split(self, data):
        candidate_features = self.rnd.sample(
            FEATURES, min(self.n_features_sample, len(FEATURES))
        )
        labels = [r[TARGET] for r in data]
        base_impurity = gini_impurity(labels)

        best = None
        best_gain = 1e-12  # solo aceptar divisiones que mejoren algo

        for feature in candidate_features:
            if feature in NUMERIC_FEATURES:
                values = sorted(set(r[feature] for r in data))
                if len(values) < 2:
                    continue
                thresholds = [
                    (values[i] + values[i + 1]) / 2
                    for i in range(len(values) - 1)
                ]
                if len(thresholds) > self.max_thresholds:
                    step = len(thresholds) / self.max_thresholds
                    thresholds = [
                        thresholds[int(i * step)] for i in range(self.max_thresholds)
                    ]
                for t in thresholds:
                    left = [r for r in data if r[feature] <= t]
                    right = [r for r in data if r[feature] > t]
                    if not left or not right:
                        continue
                    wg = weighted_gini(
                        [r[TARGET] for r in left], [r[TARGET] for r in right]
                    )
                    gain = base_impurity - wg
                    if gain > best_gain:
                        best_gain = gain
                        best = (feature, True, t, None, left, right, gain)
            else:
                categories = set(r[feature] for r in data)
                for cat in categories:
                    left = [r for r in data if r[feature] == cat]
                    right = [r for r in data if r[feature] != cat]
                    if not left or not right:
                        continue
                    wg = weighted_gini(
                        [r[TARGET] for r in left], [r[TARGET] for r in right]
                    )
                    gain = base_impurity - wg
                    if gain > best_gain:
                        best_gain = gain
                        best = (feature, False, None, cat, left, right, gain)

        return best

    def predict_one(self, row):
        node = self.root
        while not node.is_leaf():
            if node.is_numeric:
                node = node.left if row[node.feature] <= node.threshold else node.right
            else:
                node = node.left if row[node.feature] == node.category else node.right
        return node.prediction

    def predict(self, data):
        return [self.predict_one(r) for r in data]


# 4. Random Forest (bagging + seleccion aleatoria de variables)

class RandomForest:
    """Ensamble de arboles de decision con bagging y seleccion aleatoria
    de variables."""

    def __init__(self, n_trees=FOREST_N_TREES, max_depth=FOREST_MAX_DEPTH,
                 min_samples_split=FOREST_MIN_SAMPLES_SPLIT,
                 min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
                 max_features=FOREST_MAX_FEATURES, seed=RANDOM_SEED):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.feature_importance_ = {f: 0.0 for f in FEATURES}
        self.oob_score_ = None

    def _n_features_sample(self):
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(len(FEATURES))))
        if self.max_features == "log2":
            return max(1, int(math.log2(len(FEATURES))))
        return len(FEATURES)

    @staticmethod
    def _bootstrap(data, rnd):
        n = len(data)
        idxs = [rnd.randrange(n) for _ in range(n)]
        sample = [data[i] for i in idxs]
        oob_idx = set(range(n)) - set(idxs)
        oob = [data[i] for i in oob_idx]
        return sample, oob

    def fit(self, data, verbose=True):
        rnd = random.Random(self.seed)
        n_feat = self._n_features_sample()
        self.trees = []

        oob_votes = {}  # id(row) -- Counter de predicciones
        oob_true = {}

        for i in range(self.n_trees):
            sample, oob = self._bootstrap(data, rnd)
            tree_rnd = random.Random(rnd.randrange(10 ** 9))
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_features_sample=n_feat,
                rnd=tree_rnd,
            )
            tree.fit(sample)
            self.trees.append(tree)

            for f, v in tree.feature_importance_.items():
                self.feature_importance_[f] += v

            # Registrar votos "out-of-bag" para estimar el error sin usar
            # el conjunto de prueba (equivalente a una validacion interna).
            if oob:
                preds = tree.predict(oob)
                for row, pred in zip(oob, preds):
                    key = id(row)
                    oob_votes.setdefault(key, Counter())[pred] += 1
                    oob_true[key] = row[TARGET]

            if verbose and (i + 1) % 10 == 0:
                print(f"  Arbol {i + 1}/{self.n_trees} entrenado")

        total_importance = sum(self.feature_importance_.values()) or 1.0
        self.feature_importance_ = {
            f: v / total_importance for f, v in self.feature_importance_.items()
        }

        if oob_votes:
            correct = 0
            for key, votes in oob_votes.items():
                pred = votes.most_common(1)[0][0]
                if pred == oob_true[key]:
                    correct += 1
            self.oob_score_ = correct / len(oob_votes)

        return self

    def predict_one(self, row):
        votes = Counter(tree.predict_one(row) for tree in self.trees)
        return votes.most_common(1)[0][0]

    def predict(self, data):
        return [self.predict_one(r) for r in data]

    def predict_proba_one(self, row):
        votes = Counter(tree.predict_one(row) for tree in self.trees)
        total = sum(votes.values())
        return {label: count / total for label, count in votes.items()}


# 5. Metricas de evaluacion (matriz de confusion, precision, recall, F1)
#    La matriz de confusion se calcula con sklearn.metrics.confusion_matrix
#    (unica parte del script que usa scikit-learn); precision/recall/F1 se
#    siguen calculando a mano a partir de esa matriz.

def accuracy_score_manual(y_true, y_pred):
    """Exactitud (accuracy) calculada a mano: proporcion de aciertos."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def log_loss_manual(data, forest, eps=1e-15):
    """Log-loss (entropia cruzada) calculada a mano: para cada ejemplo
    se usa la proporcion de votos que el bosque le dio a la clase real
    (forest.predict_proba_one) como si fuera su 'probabilidad'. Sirve
    como equivalente de la 'loss' que se grafica por epoca en modelos
    entrenados de forma iterativa (aqui, por numero de arboles / tamano
    de entrenamiento en vez de epocas)."""
    if not data:
        return 0.0
    total = 0.0
    for row in data:
        proba = forest.predict_proba_one(row)
        p = proba.get(row[TARGET], 0.0)
        p = min(max(p, eps), 1.0 - eps)
        total += -math.log(p)
    return total / len(data)


def classification_report(y_true, y_pred, labels):
    matrix = sk_confusion_matrix(y_true, y_pred, labels=labels)
    n = len(labels)
    total = len(y_true)
    correct = sum(matrix[i][i] for i in range(n))
    accuracy = correct / total if total else 0.0

    report = {}
    macro_p = macro_r = macro_f1 = 0.0
    weighted_p = weighted_r = weighted_f1 = 0.0

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n) if r != i)
        fn = sum(matrix[i][c] for c in range(n) if c != i)
        support = sum(matrix[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        report[label] = {
            "precision": precision, "recall": recall,
            "f1": f1, "support": support,
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support

    report["accuracy"] = accuracy
    report["macro avg"] = {
        "precision": macro_p / n, "recall": macro_r / n,
        "f1": macro_f1 / n, "support": total,
    }
    report["weighted avg"] = {
        "precision": weighted_p / total if total else 0.0,
        "recall": weighted_r / total if total else 0.0,
        "f1": weighted_f1 / total if total else 0.0,
        "support": total,
    }
    return report, matrix


def report_to_dataframe(report, labels):
    """Convierte el reporte (dict calculado a mano) en un DataFrame de
    pandas, solo para mostrarlo/guardarlo de forma legible."""
    rows = []
    for label in labels:
        r = report[label]
        rows.append([label, r["precision"], r["recall"], r["f1"], r["support"]])
    for key in ("macro avg", "weighted avg"):
        r = report[key]
        rows.append([key, r["precision"], r["recall"], r["f1"], r["support"]])
    df = pd.DataFrame(rows, columns=["clase", "precision", "recall", "f1", "support"])
    return df.set_index("clase")


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    """Grafica la matriz de confusion usando sklearn.metrics
    (ConfusionMatrixDisplay), con el formato estandar de la libreria."""
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=labels,
        cmap="Blues", colorbar=True, ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion (sklearn) - Random Forest (prueba)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(feature_importance, out_path):
    fi = pd.Series(feature_importance).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    fi.plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_xlabel("Importancia relativa (reduccion de impureza Gini)")
    ax.set_title("Importancia de variables - Random Forest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_results(report_df, matrix, labels, feature_importance, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    cm_df = pd.DataFrame(matrix, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    report_df.round(4).to_csv(os.path.join(out_dir, "metrics.csv"))

    fi_df = pd.Series(feature_importance, name="importance").sort_values(ascending=False)
    fi_df.round(4).to_csv(os.path.join(out_dir, "feature_importance.csv"))


# 6. Curvas de entrenamiento (accuracy vs. numero de arboles y vs.
#    tamano del conjunto de entrenamiento), para revisar overfitting.
#    Se grafican juntas, en dos paneles lado a lado, siguiendo el mismo
#    estilo (subplots 1x2, grid con alpha=0.3) que se usa habitualmente
#    para graficar accuracy/loss por epoca en modelos entrenados de forma
#    iterativa.

def stratified_subsample(data, fraction, seed=RANDOM_SEED):
    """Toma una fraccion (0-1] de 'data' manteniendo la proporcion de
    clases (igual idea que train_test_split_stratified, pero para
    submuestrear el propio conjunto de entrenamiento)."""
    rnd = random.Random(seed)
    by_class = {}
    for row in data:
        by_class.setdefault(row[TARGET], []).append(row)

    subset = []
    for label, items in by_class.items():
        items = items[:]
        rnd.shuffle(items)
        n_take = max(1, round(len(items) * fraction))
        subset.extend(items[:n_take])
    rnd.shuffle(subset)
    return subset


def compute_trees_learning_curve(data, tree_counts, n_runs=N_CURVE_RUNS,
                                  base_seed=RANDOM_SEED, test_ratio=TEST_RATIO,
                                  max_depth=FOREST_MAX_DEPTH,
                                  min_samples_split=FOREST_MIN_SAMPLES_SPLIT,
                                  min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
                                  max_features=FOREST_MAX_FEATURES):
    """Entrena bosques con distinto numero de arboles y mide accuracy y
    log-loss en entrenamiento y prueba para cada uno (equivalente a la
    curva de accuracy/loss por epoca, pero usando el numero de arboles
    como eje de progreso del entrenamiento).

    Con pocos arboles el bagging tiene bastante varianza (una sola
    corrida puede subir/bajar solo por la suerte del bootstrap), y como
    el conjunto de prueba tambien es chico, un solo ejemplo dificil mal
    votado puede mover bastante la log-loss. Por eso, para cada valor
    de n_trees, se repite 'n_runs' veces con semillas distintas TODO el
    proceso -incluyendo un split train/test nuevo cada vez (no solo un
    bosque nuevo)- y se promedia el resultado: eso da una curva mas
    representativa del comportamiento tipico del modelo, en vez de una
    sola corrida ruidosa atada a un unico split fijo."""
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []
    for n_trees in tree_counts:
        run_train_acc, run_test_acc = [], []
        run_train_loss, run_test_loss = [], []
        for run in range(n_runs):
            seed = base_seed + run
            train, test = train_test_split_stratified(data, test_ratio=test_ratio, seed=seed)
            y_train_true = [r[TARGET] for r in train]
            y_test_true = [r[TARGET] for r in test]

            forest = RandomForest(
                n_trees=n_trees, max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf, max_features=max_features,
                seed=seed,
            )
            forest.fit(train, verbose=False)

            train_pred = forest.predict(train)
            test_pred = forest.predict(test)

            run_train_acc.append(accuracy_score_manual(y_train_true, train_pred))
            run_test_acc.append(accuracy_score_manual(y_test_true, test_pred))
            run_train_loss.append(log_loss_manual(train, forest))
            run_test_loss.append(log_loss_manual(test, forest))

        train_acc.append(sum(run_train_acc) / n_runs)
        test_acc.append(sum(run_test_acc) / n_runs)
        train_loss.append(sum(run_train_loss) / n_runs)
        test_loss.append(sum(run_test_loss) / n_runs)

    return train_acc, test_acc, train_loss, test_loss


def compute_train_size_learning_curve(data, fractions, n_runs=N_CURVE_RUNS,
                                       base_seed=RANDOM_SEED, test_ratio=TEST_RATIO,
                                       n_trees=FOREST_N_TREES,
                                       max_depth=FOREST_MAX_DEPTH,
                                       min_samples_split=FOREST_MIN_SAMPLES_SPLIT,
                                       min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
                                       max_features=FOREST_MAX_FEATURES):
    """Entrena el bosque usando fracciones crecientes del conjunto de
    entrenamiento y mide accuracy y log-loss en ese subconjunto y en el
    conjunto de prueba, para ver como mejora el modelo conforme hay mas
    datos disponibles.

    Igual que en compute_trees_learning_curve, cada fraccion se repite
    con 'n_runs' semillas distintas que cambian el split train/test
    completo (ademas del submuestreo estratificado y el bagging del
    bosque), y se promedia el resultado para suavizar el ruido de un
    solo conjunto de prueba fijo."""
    sizes, train_acc, test_acc = [], [], []
    train_loss, test_loss = [], []
    for frac in fractions:
        run_sizes = []
        run_train_acc, run_test_acc = [], []
        run_train_loss, run_test_loss = [], []
        for run in range(n_runs):
            seed = base_seed + run
            train, test = train_test_split_stratified(data, test_ratio=test_ratio, seed=seed)
            y_test_true = [r[TARGET] for r in test]

            subset = stratified_subsample(train, frac, seed=seed)
            forest = RandomForest(
                n_trees=n_trees, max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf, max_features=max_features,
                seed=seed,
            )
            forest.fit(subset, verbose=False)

            y_subset_true = [r[TARGET] for r in subset]
            subset_pred = forest.predict(subset)
            test_pred = forest.predict(test)

            run_sizes.append(len(subset))
            run_train_acc.append(accuracy_score_manual(y_subset_true, subset_pred))
            run_test_acc.append(accuracy_score_manual(y_test_true, test_pred))
            run_train_loss.append(log_loss_manual(subset, forest))
            run_test_loss.append(log_loss_manual(test, forest))

        sizes.append(round(sum(run_sizes) / n_runs))
        train_acc.append(sum(run_train_acc) / n_runs)
        test_acc.append(sum(run_test_acc) / n_runs)
        train_loss.append(sum(run_train_loss) / n_runs)
        test_loss.append(sum(run_test_loss) / n_runs)

    return sizes, train_acc, test_acc, train_loss, test_loss


def plot_accuracy_loss_curves(x_values, train_acc, val_acc, train_loss, val_loss,
                               x_label, out_path):
    """Grafica accuracy y loss lado a lado (1 fila x 2 columnas), en el
    mismo estilo que se usa para graficar accuracy/loss por epoca en
    modelos entrenados de forma iterativa: sin marcadores, grid con
    alpha=0.3, figsize=(10, 4), dpi=150. Aqui el eje x no son epocas
    sino la variable que se va incrementando en cada corrida (numero de
    arboles o tamano del conjunto de entrenamiento).

    Se ordena por x_values antes de graficar: si la lista de valores de
    x (p. ej. tree_counts) no viene en orden ascendente, matplotlib
    conectaria los puntos en el orden en que llegan (no en orden
    numerico), lo que produce picos/zigzags falsos en la curva. Ordenar
    aqui evita ese problema sin importar el orden en que se haya
    construido la lista de x_values."""
    order = sorted(range(len(x_values)), key=lambda i: x_values[i])
    x_sorted = [x_values[i] for i in order]
    train_acc = [train_acc[i] for i in order]
    val_acc = [val_acc[i] for i in order]
    train_loss = [train_loss[i] for i in order]
    val_loss = [val_loss[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_sorted, train_acc, label="Train")
    axes[0].plot(x_sorted, val_acc, label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x_sorted, train_loss, label="Train")
    axes[1].plot(x_sorted, val_loss, label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# 7. Programa principal

def main():
    print("=" * 78)
    print(" Random Forest implementado desde cero - Dieta de dinosaurios")
    print("=" * 78)

    print(f"\n[1/7] Cargando y limpiando dataset '{DATA_PATH}' (pandas) ...")
    df = load_dataframe(DATA_PATH)

    print("\n[2/7] Analisis exploratorio de datos (EDA) ...")
    run_eda(df)

    print("\n[3/7] Separando en entrenamiento (80%) y prueba (20%), estratificado ...")
    data = dataframe_to_records(df)
    train, test = train_test_split_stratified(data, test_ratio=TEST_RATIO)
    n_total = len(data)
    print(f"      Dataset total (limpio): {n_total} dinosaurios")
    print(f"      Entrenamiento: {len(train)} ({len(train) / n_total:.1%})  |  "
          f"Prueba: {len(test)} ({len(test) / n_total:.1%})")
    print("      (el split es estratificado: cada dieta guarda la misma "
          "proporcion en train y en test)")

    print(f"\n[4/7] Entrenando Random Forest ({FOREST_N_TREES} arboles, "
          f"profundidad max={FOREST_MAX_DEPTH}) ...")
    forest = RandomForest(
        n_trees=FOREST_N_TREES, max_depth=FOREST_MAX_DEPTH,
        min_samples_split=FOREST_MIN_SAMPLES_SPLIT,
        min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
        max_features=FOREST_MAX_FEATURES, seed=RANDOM_SEED,
    )
    forest.fit(train)
    if forest.oob_score_ is not None:
        print(f"      Accuracy Out-of-Bag (estimada durante el entrenamiento): "
              f"{forest.oob_score_:.4f}")

    print("\n[5/7] Evaluando en el conjunto de PRUEBA (datos nunca vistos) ...")
    y_pred = forest.predict(test)
    y_true = [r[TARGET] for r in test]
    labels = sorted(set(r[TARGET] for r in data))
    report, matrix = classification_report(y_true, y_pred, labels)
    report_df = report_to_dataframe(report, labels)

    print("\nMatriz de confusion (filas = real, columnas = predicho):")
    print(pd.DataFrame(matrix, index=labels, columns=labels).to_string())

    print(f"\nAccuracy global en prueba: {report['accuracy']:.4f}")
    print(report_df.round(3).to_string())

    print("\nImportancia de variables (proporcion de reduccion de impureza Gini):")
    fi_series = pd.Series(forest.feature_importance_).sort_values(ascending=False)
    print(fi_series.round(3).to_string())

    print(f"\n[6/7] Generando curvas de entrenamiento (accuracy / loss, "
          f"promedio de {N_CURVE_RUNS} corridas por punto -cada corrida con su "
          f"propio split train/test-) ...")
    tree_counts = [1, 5, 10, 15, 20, 30, 40, 50, 60]
    train_acc_t, test_acc_t, train_loss_t, test_loss_t = compute_trees_learning_curve(
        data, tree_counts
    )
    plot_accuracy_loss_curves(
        tree_counts, train_acc_t, test_acc_t, train_loss_t, test_loss_t,
        "Número de árboles",
        os.path.join(IMG_DIR, "learning_curves_trees.png"),
    )
    print("      -> learning_curves_trees.png (accuracy/loss vs. número de árboles)")

    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sizes, train_acc_s, test_acc_s, train_loss_s, test_loss_s = compute_train_size_learning_curve(
        data, fractions
    )
    plot_accuracy_loss_curves(
        sizes, train_acc_s, test_acc_s, train_loss_s, test_loss_s,
        "Ejemplos de entrenamiento usados",
        os.path.join(IMG_DIR, "learning_curves_train_size.png"),
    )
    print("      -> learning_curves_train_size.png (accuracy/loss vs. tamaño de entrenamiento)")

    print("\n[7/7] Guardando resultados y graficas en 'results/' ...")
    save_results(report_df, matrix, labels, forest.feature_importance_)
    plot_confusion_matrix(y_true, y_pred, labels, os.path.join(IMG_DIR, "confusion_matrix.png"))
    plot_feature_importance(forest.feature_importance_, os.path.join(IMG_DIR, "feature_importance.png"))
    print("      -> results/confusion_matrix.csv")
    print("      -> results/metrics.csv")
    print("      -> results/feature_importance.csv")
    print("      -> results/eda_describe_numericas.csv")
    print("      -> results/eda_type_vs_diet.csv")
    print("      -> results/img/*.png")

    print("\nEjemplos de predicciones individuales (5 dinosaurios de prueba):")
    for row in test[:5]:
        pred = forest.predict_one(row)
        proba = forest.predict_proba_one(row)
        proba_str = ", ".join(f"{k}={v:.2f}" for k, v in proba.items())
        status = "OK " if pred == row[TARGET] else "ERR"
        print(f"  [{status}] real={row[TARGET]:<12s} prediccion={pred:<12s} "
              f"(longitud={row['length_m']}m, tipo={row['type']}) | votos: {proba_str}")

    print("\nListo. Ejecucion terminada.")


if __name__ == "__main__":
    main()