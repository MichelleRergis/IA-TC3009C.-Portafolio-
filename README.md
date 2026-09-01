# IA-TC3009C.-Portafolio


Implementación manual (sin frameworks de Machine Learning) de un algoritmo
**Random Forest** para predecir la **dieta** (`carnivorous`, `herbivorous`,
`omnivorous`) de un dinosaurio a partir de sus datos morfológicos, taxonómicos
y de contexto geológico.

> Entregable del módulo **TC3006C — Aprendizaje Máquina**: implementación de
> una técnica de aprendizaje automático sin usar bibliotecas de ML ni de
> estadística avanzada (no scikit-learn, no numpy, no pandas, no scipy).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `random_forest.py` | Implementación completa del algoritmo (árboles de decisión + bagging + votación) y script principal que entrena, evalúa e imprime resultados. **Solo usa la biblioteca estándar de Python** (`csv`, `math`, `random`, `collections`). |
| `dinosaurs.csv` | Dataset utilizado (ver sección Dataset). |
| `Reporte_Resultados_RandomForest.pdf` | Reporte con la metodología, matriz de confusión, métricas y análisis de resultados. |
| `results/` | Carpeta generada automáticamente al ejecutar el script, con la matriz de confusión, métricas e importancia de variables en formato `.csv`. |

## Dataset

**Dinosaur Dataset** (derivado de la Paleobiology Database, distribuido en
Kaggle como *"Jurassic Park - The Exhaustive Dinosaur Dataset"*). Contiene
4,951 registros de ocurrencias fósiles con las columnas:

`occurrence_no, name, diet, type, length_m, max_ma, min_ma, region, lng, lat, class, family`

Para este proyecto se usó `diet` como variable objetivo (clasificación
multiclase) y como variables predictoras: `length_m`, `max_ma`, `min_ma`,
`lat`, `lng`, `type` y `class`. Se descartaron los registros sin dieta
documentada, quedando **3,568 registros válidos**.

## Cómo ejecutar

Requiere únicamente **Python 3.8+**, sin instalar ningún paquete adicional:

```bash
python3 random_forest.py
```

El script:
1. Carga y limpia `dinosaurs.csv`.
2. Separa los datos en 80% entrenamiento / 20% prueba (estratificado).
3. Entrena un Random Forest de 60 árboles.
4. Evalúa el modelo en el conjunto de prueba (matriz de confusión, precision,
   recall, F1-score).
5. Imprime la importancia de variables y ejemplos de predicciones
   individuales.
6. Guarda los resultados en la carpeta `results/`.

## Resumen de resultados

- **Accuracy en prueba:** 97.6%
- **Accuracy Out-of-Bag (validación interna):** 97.4%
- Variable más importante: `type` (grupo anatómico del dinosaurio)

Ver el detalle completo, la matriz de confusión y el análisis en
[`Reporte_Resultados_RandomForest.pdf`](./Reporte_Resultados_RandomForest.pdf).

## Algoritmo implementado

- **Árbol de decisión (CART simplificado)** con impureza **Gini** como
  criterio de división, soportando variables numéricas (umbral) y
  categóricas (partición binaria por categoría).
- **Bagging:** cada árbol se entrena con una muestra bootstrap del conjunto
  de entrenamiento.
- **Selección aleatoria de variables** en cada división (`sqrt(n_features)`),
  para decorrelacionar los árboles.
- **Votación por mayoría** entre todos los árboles para la predicción final.
- **Validación Out-of-Bag (OOB)** para estimar el error del modelo sin usar
  el conjunto de prueba.
- **Importancia de variables** vía reducción acumulada de impureza Gini.

## Licencia

Proyecto académico. Dataset de uso educativo (Paleobiology Database / Kaggle).


