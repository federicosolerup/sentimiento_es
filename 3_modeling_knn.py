"""
Script 3: Modelado con K-Nearest Neighbors (KNN)

Este script:
1. Carga features engineered
2. Divide train/validation (80/20 estratificado)
3. Normaliza features con StandardScaler (crítico para KNN)
4. Entrena KNN con diferentes valores de k
5. Evalúa y selecciona mejor modelo
6. Genera visualizaciones:
   - Matriz de confusión
   - Accuracy vs k
   - PCA 2D (separabilidad de clases)
   - Top features por correlación

Output: knn_model.pkl, scaler.pkl, métricas y gráficos
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def cargar_y_preparar_datos(csv_path):
    """
    Carga el CSV de features y prepara datos para entrenamiento
    
    Returns:
        X: Features (numpy array)
        y: Labels (numpy array)
        feature_names: Nombres de las features
        label_encoder: Encoder para las clases
        df: DataFrame original
    """
    print(f"Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"  Total muestras: {len(df)}")
    print(f"  Total features: {len(df.columns) - 2}")  # -2 por filename y label
    
    # Separar features de labels
    X = df.drop(['filename', 'label'], axis=1)
    y = df['label']
    
    # Convertir labels a numéricos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nDistribución de clases:")
    for i, clase in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {clase}: {count} ({count/len(y)*100:.1f}%)")
    
    # Manejar NaN e infinitos
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Imputar NaN con mediana
    if X.isna().sum().sum() > 0:
        print(f"\nAdvertencia: {X.isna().sum().sum()} valores NaN encontrados")
        print("  Imputando con mediana...")
        X = X.fillna(X.median())
    
    feature_names = X.columns.tolist()
    X = X.values
    
    return X, y_encoded, feature_names, label_encoder, df


def entrenar_knn_con_validacion(X_train, y_train, k_values=[3, 5, 7, 9, 11, 13, 15]):
    """
    Entrena KNN con diferentes valores de k y retorna el mejor
    
    Returns:
        mejor_k: Mejor valor de k
        resultados: Dict con accuracy de cada k
    """
    print("\nEntrenando KNN con validación cruzada...")
    print(f"Probando k = {k_values}")
    
    resultados = {}
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        # Validación cruzada (5 folds)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        resultados[k] = {
            'mean': mean_score,
            'std': std_score,
            'scores': scores
        }
        
        print(f"  k={k:2d}: Accuracy = {mean_score:.4f} (±{std_score:.4f})")
    
    # Seleccionar mejor k
    mejor_k = max(resultados.keys(), key=lambda k: resultados[k]['mean'])
    print(f"\n✓ Mejor k seleccionado: {mejor_k} (Accuracy: {resultados[mejor_k]['mean']:.4f})")
    
    return mejor_k, resultados


def evaluar_modelo(modelo, X_test, y_test, label_encoder):
    """
    Evalúa el modelo en el conjunto de validación
    
    Returns:
        y_pred: Predicciones
        metricas: Dict con métricas de evaluación
    """
    print("\nEvaluando modelo en conjunto de validación...")
    
    y_pred = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")
    print(f"  F1-Score (weighted): {f1_weighted:.4f}")
    
    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    metricas = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    }
    
    return y_pred, metricas


def plot_accuracy_vs_k(resultados, output_path):
    """
    Grafica accuracy vs k
    """
    k_values = sorted(resultados.keys())
    means = [resultados[k]['mean'] for k in k_values]
    stds = [resultados[k]['std'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, means, yerr=stds, marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Valor de k (número de vecinos)', fontsize=12)
    plt.ylabel('Accuracy (Validación Cruzada)', fontsize=12)
    plt.title('Accuracy vs k en K-Nearest Neighbors', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Marcar el mejor k
    mejor_k = max(resultados.keys(), key=lambda k: resultados[k]['mean'])
    plt.axvline(x=mejor_k, color='r', linestyle='--', alpha=0.5, label=f'Mejor k = {mejor_k}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico guardado: {output_path}")
    plt.close()


def plot_confusion_matrix(cm, label_encoder, output_path):
    """
    Grafica matriz de confusión
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cbar_kws={'label': 'Cantidad'})
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.title('Matriz de Confusión - KNN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Matriz de confusión guardada: {output_path}")
    plt.close()


def plot_pca_2d(X, y, label_encoder, output_path):
    """
    Proyecta features a 2D con PCA y visualiza separabilidad de clases
    """
    print("\nGenerando visualización PCA 2D...")
    
    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Varianza explicada
    var_exp = pca.explained_variance_ratio_
    print(f"  Varianza explicada: PC1={var_exp[0]:.2%}, PC2={var_exp[1]:.2%}")
    print(f"  Total: {sum(var_exp):.2%}")
    
    # Graficar
    plt.figure(figsize=(10, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']
    
    for i, clase in enumerate(label_encoder.classes_):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], marker=markers[i], 
                   label=clase, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    plt.xlabel(f'PC1 ({var_exp[0]:.1%} varianza)', fontsize=12)
    plt.ylabel(f'PC2 ({var_exp[1]:.1%} varianza)', fontsize=12)
    plt.title('Espacio de Features - Proyección PCA 2D', fontsize=14, fontweight='bold')
    plt.legend(title='Clase', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gráfico PCA guardado: {output_path}")
    plt.close()


def analizar_top_features(df, feature_names, n_top=10, output_path=None):
    """
    Analiza correlación de features con labels y muestra las más importantes
    """
    print(f"\nAnalizando top {n_top} features por correlación...")
    
    # Crear copia del dataframe
    df_analysis = df.copy()
    
    # Codificar labels numéricamente
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df_analysis['label_encoded'] = df_analysis['label'].map(label_map)
    
    # Calcular correlaciones
    correlaciones = {}
    for feature in feature_names:
        if feature in df_analysis.columns:
            corr = abs(df_analysis[feature].corr(df_analysis['label_encoded']))
            if not np.isnan(corr):
                correlaciones[feature] = corr
    
    # Ordenar por correlación
    top_features = sorted(correlaciones.items(), key=lambda x: x[1], reverse=True)[:n_top]
    
    print("\nTop features por correlación con labels:")
    for i, (feature, corr) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feature:40s}: {corr:.4f}")
    
    # Graficar si se especifica output_path
    if output_path:
        features, corrs = zip(*top_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), corrs, color='steelblue')
        plt.yticks(range(len(features)), features, fontsize=10)
        plt.xlabel('Correlación Absoluta con Label', fontsize=12)
        plt.title(f'Top {n_top} Features por Importancia', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Gráfico de features guardado: {output_path}")
        plt.close()
    
    return top_features


def guardar_metricas(metricas, mejor_k, output_path):
    """
    Guarda métricas en archivo de texto
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MÉTRICAS DEL MODELO KNN\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Mejor k seleccionado: {mejor_k}\n\n")
        
        f.write(f"Accuracy: {metricas['accuracy']:.4f}\n")
        f.write(f"F1-Score (macro): {metricas['f1_macro']:.4f}\n")
        f.write(f"F1-Score (weighted): {metricas['f1_weighted']:.4f}\n\n")
        
        f.write("Matriz de Confusión:\n")
        f.write(str(metricas['confusion_matrix']) + "\n\n")
        
        f.write("Reporte de Clasificación:\n")
        f.write(metricas['classification_report'] + "\n")
    
    print(f"  ✓ Métricas guardadas: {output_path}")


if __name__ == "__main__":
    # Configuración de paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_CSV = os.path.join(BASE_DIR, "features_engineered_train.csv")
    
    # Outputs
    MODEL_PATH = os.path.join(BASE_DIR, "knn_model.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
    METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.txt")
    PLOT_ACCURACY_PATH = os.path.join(BASE_DIR, "plot_accuracy_vs_k.png")
    PLOT_CM_PATH = os.path.join(BASE_DIR, "plot_confusion_matrix.png")
    PLOT_PCA_PATH = os.path.join(BASE_DIR, "plot_pca_2d.png")
    PLOT_FEATURES_PATH = os.path.join(BASE_DIR, "plot_top_features.png")
    
    print("="*60)
    print("SCRIPT 3: MODELADO CON K-NEAREST NEIGHBORS")
    print("="*60)
    
    # Verificar archivo de entrada
    if not os.path.exists(INPUT_CSV):
        print(f"Error: No se encuentra el archivo {INPUT_CSV}")
        print("Por favor ejecuta primero el script 2_feature_engineering.py")
        exit(1)
    
    # 1. Cargar y preparar datos
    X, y, feature_names, label_encoder, df = cargar_y_preparar_datos(INPUT_CSV)
    
    # 2. Split train/validation (80/20 estratificado)
    print("\n" + "-"*60)
    print("Dividiendo datos en train/validation (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} muestras")
    print(f"  Validation: {len(X_val)} muestras")
    
    # 3. Normalización (StandardScaler)
    print("\n" + "-"*60)
    print("Normalizando features con StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print("  ✓ Features normalizadas (media=0, std=1)")
    
    # 4. Entrenar KNN con diferentes k
    print("\n" + "-"*60)
    mejor_k, resultados_cv = entrenar_knn_con_validacion(X_train_scaled, y_train)
    
    # 5. Entrenar modelo final con mejor k
    print("\n" + "-"*60)
    print(f"Entrenando modelo final con k={mejor_k}...")
    knn_final = KNeighborsClassifier(n_neighbors=mejor_k, metric='euclidean')
    knn_final.fit(X_train_scaled, y_train)
    print("  ✓ Modelo entrenado")
    
    # 6. Evaluar en validation set
    print("\n" + "-"*60)
    y_pred, metricas = evaluar_modelo(knn_final, X_val_scaled, y_val, label_encoder)
    
    # 7. Guardar modelo y scaler
    print("\n" + "-"*60)
    print("Guardando modelo y scaler...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(knn_final, f)
    print(f"  ✓ Modelo guardado: {MODEL_PATH}")
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler guardado: {SCALER_PATH}")
    
    # Guardar label encoder para uso en predicción
    LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  ✓ Label encoder guardado: {LABEL_ENCODER_PATH}")
    
    # 8. Guardar métricas
    print("\n" + "-"*60)
    print("Guardando métricas...")
    guardar_metricas(metricas, mejor_k, METRICS_PATH)
    
    # 9. Generar visualizaciones
    print("\n" + "-"*60)
    print("Generando visualizaciones...")
    
    # Accuracy vs k
    plot_accuracy_vs_k(resultados_cv, PLOT_ACCURACY_PATH)
    
    # Matriz de confusión
    plot_confusion_matrix(metricas['confusion_matrix'], label_encoder, PLOT_CM_PATH)
    
    # PCA 2D (usar datos escalados completos)
    X_scaled = scaler.transform(X)
    plot_pca_2d(X_scaled, y, label_encoder, PLOT_PCA_PATH)
    
    # Top features
    analizar_top_features(df, feature_names, n_top=15, output_path=PLOT_FEATURES_PATH)
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
    print(f"\nResumen:")
    print(f"  - Modelo KNN (k={mejor_k}) entrenado y guardado")
    print(f"  - Accuracy en validación: {metricas['accuracy']:.4f}")
    print(f"  - F1-Score (macro): {metricas['f1_macro']:.4f}")
    print(f"  - Visualizaciones generadas: 4 gráficos")

