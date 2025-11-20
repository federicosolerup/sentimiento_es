# 🎧 Clasificación Multiclase - Análisis de Sentimiento en Voz Humana — Pipeline K-NN (Python + Librosa)

Este proyecto implementa un pipeline **end-to-end** para clasificar emociones en audios cortos (Negativo / Neutro / Positivo) utilizando extracción de features acústicas, ingeniería de features, normalización y modelado con **K-Nearest Neighbors (KNN)**.
Incluye entrenamiento, predicción sobre datos nuevos y evaluación final sobre el conjunto de test.

---

## 📁 Estructura del proyecto

```
project/
│── TRAIN/                         # Audios de entrenamiento
│── TEST/                          # Audios de test
│── 1_feature_extraction.py        # Extracción de features crudas
│── 2_feature_engineering.py       # Agregaciones + transforms logarítmicas
│── 3_modeling_knn.py              # Entrenamiento + validación + gráficos
│── 4_predict_test.py              # Pipeline completo de predicción en TEST
│── 5_evaluate_test.py             # Métricas y PCA 3D sobre TEST
│── features_raw_train.csv
│── features_engineered_train.csv
│── knn_model.pkl
│── scaler.pkl
│── label_encoder.pkl
│── predictions.csv
│── README.md
```

---

## 📌 1. Extracción de Features (Script 1)

**Archivo:** `1_feature_extraction.py`
Fuente: 

Este script convierte cada `.wav` en un set de series temporales:

* RMS (energía)
* Zero Crossing Rate
* MFCCs (13 coeficientes)
* Centroid, Rolloff, Bandwidth
* Pitch (pYIN)
* Entropía de Shannon
* Exponente de Hurst

**Output generado:**

```
features_raw_train.csv
```

---

## 📌 2. Ingeniería de Features (Script 2)

**Archivo:** `2_feature_engineering.py`
Fuente: 

Transforma las series temporales crudas en **features numéricas agregadas**:

* mean, std, var, max, min, median
* percentiles 25 y 75
* skewness y kurtosis
* RMS → dB
* Pitch → semitonos
* ZCR + spectral → log(1+x)

**Output generado:**

```
features_engineered_train.csv
```

---

## 📌 3. Entrenamiento con KNN (Script 3)

**Archivo:** `3_modeling_knn.py`
Fuente: 

El pipeline de entrenamiento incluye:

* Normalización con **StandardScaler**
* Train/validation estratificado 80/20
* Búsqueda del mejor **k**
* Métricas: accuracy, F1 macro/weighted
* Gráficos:

  * Accuracy vs k
  * Matriz de confusión
  * PCA 2D
  * Top features por correlación

**Outputs generados:**

```
knn_model.pkl
scaler.pkl
label_encoder.pkl
plot_accuracy_vs_k.png
plot_confusion_matrix.png
plot_pca_2d.png
plot_top_features.png
model_metrics.txt
```

---

## 📌 4. Predicción en TEST (Script 4)

**Archivo:** `4_predict_test.py`
Fuente: 

Este script ejecuta automáticamente:

1. Extracción de features de TEST
2. Ingeniería de features
3. Normalización usando el scaler del entrenamiento
4. Predicción con el modelo KNN cargado
5. Generación del archivo final de predicciones

**Outputs generados:**

```
features_raw_test.csv
features_engineered_test.csv
predictions.csv
```

---

## 📌 5. Evaluación Final en TEST (Script 5)

**Archivo:** `5_evaluate_test.py`
Fuente: 

Produce el análisis final del modelo:

* Métricas completas sobre TEST
* Matriz de confusión en PNG
* PCA 3D coloreado por clase real
* Comparación entre etiquetas reales y predichas

**Outputs generados:**

```
plot_confusion_matrix_test.png
plot_pca_3d_test.png
test_metrics.txt
```

---

## ▶️ Cómo ejecutar el proyecto

### 1. Preparar entorno

```bash
pip install numpy pandas librosa scikit-learn seaborn matplotlib tqdm joblib
```

### 2. Colocar audios en:

```
TRAIN/
TEST/
```

### 3. Ejecutar cada script en orden

```bash
python 1_feature_extraction.py
python 2_feature_engineering.py
python 3_modeling_knn.py
python 4_predict_test.py
python 5_evaluate_test.py
```

---

## 📊 Resultados esperados

* Modelo KNN entrenado con normalización
* Visualizaciones completas
* Archivo `predictions.csv` listo para entregar
* Informe final de performance sobre TEST

---

## 📝 Licencia

Este proyecto puede ser reutilizado y modificado libremente para fines académicos.
