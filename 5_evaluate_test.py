import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D  # necesario para 3D
from sklearn.decomposition import PCA
import joblib
import numpy as np

# ============================================================
# CONFIGURACIÓN BÁSICA
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(BASE_DIR, "TEST")
TEST_CSV = os.path.join(BASE_DIR, "TEST.csv")
PRED_CSV = os.path.join(BASE_DIR, "predictions.csv")

FEATURES_TEST_CSV = os.path.join(BASE_DIR, "features_engineered_test.csv")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

OUTPUT_CM_PNG = os.path.join(BASE_DIR, "plot_confusion_matrix_test.png")
OUTPUT_METRICS_TXT = os.path.join(BASE_DIR, "test_metrics.txt")
OUTPUT_PCA3D_PNG = os.path.join(BASE_DIR, "plot_pca_3d_test.png")

LABELS_ORDEN = ["negativo", "neutro", "positivo"]  # orden fijo para matriz de confusión / gráficos


# ============================================================
# FUNCIÓN 1: Generar TEST.csv a partir de la carpeta TEST/
# ============================================================
def generar_test_csv_desde_carpeta(test_dir, output_csv):
    """
    Genera un TEST.csv con columnas Filename, Class
    deduciendo la etiqueta a partir del nombre del archivo:
    - empieza con "negativo"  -> clase negativo
    - empieza con "neutro"    -> clase neutro
    - empieza con "positivo"  -> clase positivo
    """
    print(f"Generando TEST.csv a partir de archivos en: {test_dir}")

    rows = []

    for fname in os.listdir(test_dir):
        if not fname.lower().endswith(".wav"):
            continue

        lower = fname.lower().strip()

        if lower.startswith("negativo"):
            label = "negativo"
        elif lower.startswith("neutro"):
            label = "neutro"
        elif lower.startswith("positivo"):
            label = "positivo"
        else:
            print(f"  ⚠ No se reconoce la clase para: {fname} (se omite)")
            continue

        rows.append({"Filename": fname, "Class": label})

    if not rows:
        raise RuntimeError("No se pudo generar ninguna fila para TEST.csv. "
                           "Revisa los nombres de los archivos en TEST/.")

    df = pd.DataFrame(rows)
    df = df.sort_values("Filename")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"✓ TEST.csv generado con {len(df)} filas en: {output_csv}")
    print(df.head())
    print()
    return df


# ============================================================
# FUNCIÓN 2: PCA 3D sobre TEST
# ============================================================
def plot_pca_3d_test(features_csv, scaler_path, merged_df, output_path):
    """
    Genera un PCA 3D usando las features engineered de TEST
    y el mismo scaler del entrenamiento. Colorea por clase real.
    """
    print("\nGenerando PCA 3D sobre TEST...")

    if not os.path.exists(features_csv):
        print(f"⚠ No se encontró {features_csv}. No se generará PCA 3D.")
        return

    if not os.path.exists(scaler_path):
        print(f"⚠ No se encontró scaler.pkl ({scaler_path}). No se generará PCA 3D.")
        return

    # Cargar features engineered de TEST
    feats_df = pd.read_csv(features_csv)

    # ---- Normalizar nombres de columna para evitar KeyError por capitalización/espacios ----
    feats_df.columns = feats_df.columns.str.strip()
    merged_df = merged_df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    # Buscar la columna que corresponde al nombre de archivo en ambos dataframes
    def find_filename_col(df):
        for c in df.columns:
            if isinstance(c, str) and c.lower() == "filename":
                return c
        for c in df.columns:
            if isinstance(c, str) and "file" in c.lower():
                return c
        return None

    merged_fname_col = find_filename_col(merged_df)
    feats_fname_col = find_filename_col(feats_df)

    if merged_fname_col is None or feats_fname_col is None:
        print("⚠ No se encontró columna de nombre de archivo en merged_df o en features_engineered_test.csv.")
        return

    # Renombrar a 'Filename' para hacer merge seguro
    if merged_fname_col != "Filename":
        merged_df = merged_df.rename(columns={merged_fname_col: "Filename"})
    if feats_fname_col != "Filename":
        feats_df = feats_df.rename(columns={feats_fname_col: "Filename"})
    # ---------------------------------------------------------------------------------------

    # Unir features con las etiquetas reales por Filename
    # merged_df tiene columnas: Filename, Class, Predicted_Class
    merged_features = pd.merge(merged_df[["Filename", "Class", "Predicted_Class"]],
                               feats_df,
                               on="Filename",
                               how="inner")

    if merged_features.empty:
        print("⚠ No hay intersección entre features_engineered_test y merged (labels+pred). "
              "No se generará PCA 3D.")
        return

    # Extraer X y labels
    # Eliminamos columnas no numéricas:
    X = merged_features.drop(columns=["Filename", "Class", "Predicted_Class"])
    y = merged_features["Class"].values

    # Cargar scaler entrenado
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # PCA 3D
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_scaled)

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    pc3_var = pca.explained_variance_ratio_[2] * 100

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colores = {"negativo": "red", "neutro": "teal", "positivo": "skyblue"}
    marcadores = {"negativo": "o", "neutro": "s", "positivo": "^"}

    for clase in np.unique(y):
        idx = (y == clase)
        ax.scatter(
            pcs[idx, 0],
            pcs[idx, 1],
            pcs[idx, 2],
            label=clase,
            c=colores.get(clase, "gray"),
            marker=marcadores.get(clase, "o"),
            s=60,
            alpha=0.8
        )

    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)")
    ax.set_zlabel(f"PC3 ({pc3_var:.1f}% var)")
    ax.set_title("PCA 3D - Features TEST (clase real)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✓ PCA 3D de TEST guardado en: {output_path}")


# ============================================================
# FUNCIÓN 3: Evaluar métricas en TEST
# ============================================================
def evaluar_test(test_csv, pred_csv, cm_png_path, metrics_txt_path):
    print("=" * 60)
    print("SCRIPT 5: EVALUACIÓN EN CONJUNTO DE TEST")
    print("=" * 60)

    # 1. Cargar labels reales y predicciones
    print(f"Cargando TEST.csv desde: {test_csv}")
    test_df = pd.read_csv(test_csv)

    print(f"Cargando predictions.csv desde: {pred_csv}")
    pred_df = pd.read_csv(pred_csv)

    # Asegurar nombres de columnas
    pred_df.rename(columns={
        'filename': 'Filename',
        'Predicted_Class': 'Predicted_Class'
    }, inplace=True)

    if "Filename" not in pred_df.columns:
        raise RuntimeError("predictions.csv no contiene columna 'Filename'. Revisa el script 4.")

    if "Predicted_Class" not in pred_df.columns:
        raise RuntimeError("predictions.csv no contiene columna 'Predicted_Class'. "
                           "Revisa cómo se guardan las predicciones en el script 4.")

    # 2. Merge por Filename
    print("\nUniendo labels reales y predicciones por Filename...")
    merged = pd.merge(test_df, pred_df, on="Filename", how="inner")

    print(f"Filas en TEST.csv:                  {len(test_df)}")
    print(f"Filas en predictions.csv:           {len(pred_df)}")
    print(f"Filas tras el merge (TEST efectivo): {len(merged)}")

    if len(merged) == 0:
        raise RuntimeError("No se pudo unir nada. Revisa que 'Filename' coincida en ambos CSV.")

    if len(merged) < len(test_df):
        print("⚠ Advertencia: Hay filas en TEST.csv que no tienen predicción asociada.")
        print("  Asegúrate de que todos los archivos de TEST/ hayan sido procesados en el script 4.\n")

    y_true = merged["Class"]
    y_pred = merged["Predicted_Class"]

    print("\nDistribución de clases reales en TEST:")
    print(y_true.value_counts())
    print("\nDistribución de predicciones en TEST:")
    print(y_pred.value_counts())

    # 3. Métricas
    print("\n" + "=" * 60)
    print("MÉTRICAS EN CONJUNTO DE TEST")
    print("=" * 60 + "\n")

    # Usamos target_names para forzar el orden deseado en el reporte
    report = classification_report(
        y_true, y_pred,
        digits=4,
        labels=LABELS_ORDEN,
        target_names=LABELS_ORDEN
    )
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(report)
    print(f"Accuracy (TEST):      {acc:.4f}")
    print(f"F1 macro (TEST):      {f1_macro:.4f}")
    print(f"F1 weighted (TEST):   {f1_weighted:.4f}")

    # Guardar métricas en un txt para el informe
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("MÉTRICAS EN CONJUNTO DE TEST\n")
        f.write("=" * 40 + "\n\n")
        f.write(report + "\n")
        f.write(f"Accuracy (TEST):    {acc:.4f}\n")
        f.write(f"F1 macro (TEST):    {f1_macro:.4f}\n")
        f.write(f"F1 weighted (TEST): {f1_weighted:.4f}\n")

    print(f"\n✓ Métricas de TEST guardadas en: {metrics_txt_path}")

    # 4. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDEN)

    print("\nMatriz de confusión (filas = real, columnas = pred):")
    print(pd.DataFrame(cm, index=LABELS_ORDEN, columns=LABELS_ORDEN))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABELS_ORDEN, yticklabels=LABELS_ORDEN,
                cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión - TEST")
    plt.tight_layout()
    plt.savefig(cm_png_path)
    plt.close()

    print(f"\n✓ Matriz de confusión de TEST guardada en: {cm_png_path}")

    # 5. PCA 3D sobre TEST
    plot_pca_3d_test(FEATURES_TEST_CSV, SCALER_PATH, merged, OUTPUT_PCA3D_PNG)

    print("\nPROCESO COMPLETADO.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1) Si no existe TEST.csv, lo generamos automáticamente
    if not os.path.exists(TEST_CSV):
        if not os.path.isdir(TEST_DIR):
            raise RuntimeError(f"No existe la carpeta TEST/ en {TEST_DIR}")
        generar_test_csv_desde_carpeta(TEST_DIR, TEST_CSV)
    else:
        print(f"TEST.csv ya existe en: {TEST_CSV} (se usará tal cual).\n")

    # 2) Evaluar TEST usando predictions.csv y generar PCA 3D
    if not os.path.exists(PRED_CSV):
        raise RuntimeError(f"No se encontró predictions.csv en {PRED_CSV}. "
                           "Primero ejecuta el script 4_predict_test.py.")
    evaluar_test(TEST_CSV, PRED_CSV, OUTPUT_CM_PNG, OUTPUT_METRICS_TXT)
