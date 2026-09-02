# IA-TC3009C.-Portafolio

<div align="center">

# 🌿🦖 Random Forest algoritmo desde cero 
### Clasificación de la dieta de dinosaurios (carnívoro / herbívoro / omnívoro) con un Random Forest implementado en Python

<img src="https://img.shields.io/badge/Python-3.9%2B-FFD8B1?style=for-the-badge&logo=python&logoColor=594936&labelColor=FFEEDD" alt="Python"/>
<img src="https://img.shields.io/badge/Random%20Forest-desde%20cero-C9E4DE?style=for-the-badge&labelColor=E4F4EE&logoColor=2E4E4A" alt="Random Forest"/>
<img src="https://img.shields.io/badge/Dataset-Dino%20Directory-FCE1E4?style=for-the-badge&labelColor=FDEEF0&logoColor=6E3B4E" alt="Dataset"/>
<img src="https://img.shields.io/badge/Licencia-MIT-D6E2F0?style=for-the-badge&labelColor=EAF1FA&logoColor=3A5A80" alt="Licencia"/>

<br/>

<img src="https://img.shields.io/badge/pandas-B4E1D6?style=flat-square&logo=pandas&logoColor=2E5C4D&labelColor=E3F4EE" alt="pandas"/>
<img src="https://img.shields.io/badge/numpy-C9D9F4?style=flat-square&logo=numpy&logoColor=2E4370&labelColor=E9EFFB" alt="numpy"/>
<img src="https://img.shields.io/badge/matplotlib-F6D9C4?style=flat-square&logo=plotly&logoColor=7A4A20&labelColor=FCEDE1" alt="matplotlib"/>
<img src="https://img.shields.io/badge/seaborn-D9C9F0?style=flat-square&logoColor=4E2E70&labelColor=F0E9FA" alt="seaborn"/>
<img src="https://img.shields.io/badge/scikit--learn-F9D8DE?style=flat-square&logo=scikitlearn&logoColor=7A2E3E&labelColor=FCEBEE" alt="scikit-learn"/>

</div>

<br/>

## 🌸 Descripción

Este proyecto implementa un algoritmo de **Random Forest (bosque aleatorio)** completamente desde cero — árbol de decisión CART, impureza de Gini, bagging, selección aleatoria de variables, votación por mayoría, importancia de variables y evaluación Out-of-Bag — usando **únicamente la biblioteca estándar de Python** (`math`, `random`, `collections`) para el núcleo del algoritmo.

El modelo clasifica la **dieta de un dinosaurio** (`carnivorous`, `herbivorous` u `omnivorous`) a partir de variables morfológicas, temporales y taxonómicas/geográficas derivadas del dataset **Dino Directory** del Natural History Museum.

> 🔎 `pandas`, `numpy`, `matplotlib` y `seaborn` se usan solo para carga, limpieza, EDA y graficación. La única excepción es `scikit-learn`, usado exclusivamente para calcular y graficar la **matriz de confusión** con su formato estándar — el modelo en sí no usa ningún framework de ML.

<br/>

## 📋 Tabla de contenido

- [Descripción](#-descripción)
- [Diagrama de flujo](#-diagrama-de-flujo)
- [Dataset](#-dataset)
- [Instalación](#️-instalación)
- [Uso](#️-uso)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Conceptos clave del algoritmo](#-conceptos-clave-del-algoritmo)
- [Resultados](#-resultados)
- [Referencias](#-referencias)
- [Autor](#-autor)

<br/>

## 🧭 Diagrama de flujo

<div align="center">
<img src="results/img/flowchart.png" alt="Diagrama de flujo del algoritmo" width="480"/>
</div>

El script sigue 7 etapas: carga/limpieza de datos → EDA → split estratificado 80/20 → entrenamiento del bosque (bagging + CART recursivo por árbol) → evaluación en prueba → curvas de entrenamiento → guardado de resultados. Ver la sección [Conceptos clave](#-conceptos-clave-del-algoritmo) para el detalle de cada paso.

<br/>

## 🦕 Dataset

| | |
|---|---|
| **Fuente** | [Jurassic Park - The Exhaustive Dinosaur Dataset](https://www.kaggle.com/datasets/kjanjua/jurassic-park-the-exhaustive-dinosaur-dataset) (Kaggle), derivado del *Dino Directory* del Natural History Museum |
| **Archivo** | `dinosaurs.csv` |
| **Target** | `diet` → `carnivorous` \| `herbivorous` \| `omnivorous` |
| **Variables usadas** | `length_m`, `max_ma`, `min_ma` (numéricas) · `type`, `class`, `lived_in` (categóricas) |
| **Split** | 80% entrenamiento / 20% prueba, **estratificado** por clase |

Las variables numéricas/categóricas se **derivan** de columnas de texto libre del CSV original (`length`, `period`, `taxonomy`) mediante parseo con expresiones regulares — ver `parse_length_m()`, `parse_period_ma()` y `parse_class()` en el script.

<br/>

## ⚙️ Instalación

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

<br/>

## ▶️ Uso

```bash
python3 RandomForest.py
```

El script imprime en consola el progreso (carga, EDA, entrenamiento, evaluación) y genera automáticamente todos los resultados y gráficas en la carpeta `results/`.

<br/>

## 🗂️ Estructura del proyecto

```
.
├── RandomFores.pdf                    # Reporte del algoritmo
├── RandomForest.py                    # Script principal (todo el pipeline)
├── dinosaurs.csv                      # Dataset crudo
└── results/
    ├── confusion_matrix.csv
    ├── metrics.csv
    ├── feature_importance.csv
    ├── eda_describe_numericas.csv
    ├── eda_type_vs_diet.csv
    └── img/
        ├── class_distribution.png
        ├── type_distribution.png
        ├── length_by_diet_boxplot.png
        ├── correlation_heatmap.png
        ├── top_countries.png
        ├── confusion_matrix.png
        ├── feature_importance.png
        ├── learning_curves_trees.png
        └── learning_curves_train_size.png
```

<br/>

## 🌱 Conceptos clave del algoritmo

<table>
<tr><td width="26%"><b>🌳 CART</b></td><td>Árbol de decisión binario y recursivo (Breiman et al., 1984); en cada nodo evalúa una sola variable/punto de corte y elige el que más reduce la impureza.</td></tr>
<tr><td><b>🎯 Impureza de Gini</b></td><td><code>Gini = 1 - Σ p_c²</code>. Se elige la división que maximiza la reducción de impureza respecto al nodo padre.</td></tr>
<tr><td><b>🎒 Bagging</b></td><td>Cada árbol se entrena con una muestra <i>bootstrap</i> (con reemplazo) distinta del conjunto de entrenamiento.</td></tr>
<tr><td><b>🎲 Variables aleatorias</b></td><td>En cada división solo se evalúa un subconjunto aleatorio de variables (<code>max_features="sqrt"</code>), decorrelacionando los árboles entre sí.</td></tr>
<tr><td><b>🗳️ Votación</b></td><td>La predicción final del bosque es la clase con más votos entre todos sus árboles.</td></tr>
<tr><td><b>📦 Out-of-Bag (OOB)</b></td><td>~37% de las muestras quedan fuera de cada bootstrap; se usan como validación interna gratuita, sin tocar el conjunto de prueba.</td></tr>
<tr><td><b>📊 Importancia de variables</b></td><td>Reducción de impureza acumulada y ponderada de cada variable, a lo largo de todos los árboles del bosque.</td></tr>
<tr><td><b>📈 Curvas de entrenamiento</b></td><td>Accuracy/log-loss (train vs. test) vs. número de árboles y tamaño de entrenamiento, para diagnosticar over/underfitting.</td></tr>
</table>

<br/>

## 🏆 Resultados

**Hiperparámetros del bosque:** 60 árboles · profundidad máx. 10 · min. muestras para dividir = 10 · min. muestras por hoja = 4 · `max_features="sqrt"`

### Matriz de confusión (conjunto de prueba, n=57)

<div align="center">

<img width="880" height="768" alt="confusion_matrix" src="https://github.com/user-attachments/assets/6023fc16-bfaf-4826-b1b6-20864482a09f" />

</div>

|  Real \ Predicho | carnivorous | herbivorous | omnivorous |
|---|:---:|:---:|:---:|
| **carnivorous** | 17 | 0 | 1 |
| **herbivorous** | 1 | 33 | 0 |
| **omnivorous** | 3 | 0 | 2 |

### Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|---|:---:|:---:|:---:|:---:|
| carnivorous | 0.8095 | 0.9444 | 0.8718 | 18 |
| herbivorous | 1.0000 | 0.9706 | 0.9851 | 34 |
| omnivorous | 0.6667 | 0.4000 | 0.5000 | 5 |
| **Accuracy** | | | **0.9123** | 57 |
| **Macro avg** | 0.8254 | 0.7717 | 0.7856 | 57 |
| **Weighted avg** | 0.9106 | 0.9123 | 0.9066 | 57 |

### Curvas de entrenamiento



📌 **Resumen:** el modelo alcanza **91.2% de accuracy** en prueba, con muy buen desempeño en `herbivorous` y `carnivorous` (clases mayoritarias); el reto principal es la clase minoritaria `omnivorous` (solo 5 ejemplos de prueba), reflejado en su recall más bajo (0.40). La brecha train/validation en las curvas es moderada y estable, indicando una regularización razonable (sin overfitting severo).

**Para más información checar el reporte RandomForest.pdf**

<br/>

## 📚 Referencias

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and regression trees*. Chapman & Hall/CRC.
- kjanjua. (s.f.). *Jurassic Park - The Exhaustive Dinosaur Dataset* [Conjunto de datos]. Kaggle.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.

<br/>

## 👩‍💻 Autor

**Michelle Rergis Novelo**
TC3006C — Inteligencia Artificial, Módulo 2 · Tecnológico de Monterrey



