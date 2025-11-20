"""
Script 4: Predicción en Conjunto de Test

Este script:
1. Procesa audios de TEST/ con el mismo pipeline de extracción
2. Aplica las mismas agregaciones y transformaciones logarítmicas
3. Carga el modelo KNN y scaler entrenados
4. Normaliza features con el scaler del entrenamiento
5. Genera predicciones
6. Guarda resultados en predictions.csv

Output: features_raw_test.csv, features_engineered_test.csv, predictions.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from scipy.stats import skew, kurtosis, entropy
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FUNCIONES REUTILIZADAS DEL SCRIPT 1 (Feature Extraction)
# ============================================================================

def calcular_exponente_hurst(serie_temporal):
    """Calcula el exponente de Hurst usando el método R/S"""
    if len(serie_temporal) < 20:
        return np.nan
    
    serie_temporal = serie_temporal[np.isfinite(serie_temporal)]
    if len(serie_temporal) < 20:
        return np.nan
    
    mean_serie = np.mean(serie_temporal)
    desviaciones_acum = np.cumsum(serie_temporal - mean_serie)
    
    R = np.max(desviaciones_acum) - np.min(desviaciones_acum)
    S = np.std(serie_temporal)
    
    if S == 0 or R == 0:
        return np.nan
    
    RS = R / S
    n = len(serie_temporal)
    H = np.log(RS) / np.log(n / 2)
    
    return H


def calcular_entropia_shannon(y, sr):
    """Calcula la entropía de Shannon del espectrograma"""
    S = np.abs(librosa.stft(y))
    S_norm = S / np.sum(S)
    
    entropias = []
    for i in range(S_norm.shape[1]):
        frame = S_norm[:, i]
        frame = frame[frame > 0]
        if len(frame) > 0:
            ent = entropy(frame)
            entropias.append(ent)
    
    return np.array(entropias)


def extraer_pitch(y, sr):
    """Extrae el pitch usando pYIN"""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0


def extraer_features_audio(filepath):
    """Extrae todas las features de un archivo de audio"""
    try:
        y, sr = librosa.load(filepath, sr=None)
        features = {}
        
        # RMS
        features['rms'] = librosa.feature.rms(y=y)[0]
        
        # Zero Crossing Rate
        features['zcr'] = librosa.feature.zero_crossing_rate(y)[0]
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = mfccs[i]
        
        # Spectral Features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # Pitch
        features['pitch'] = extraer_pitch(y, sr)
        
        # Entropía
        features['entropia_shannon'] = calcular_entropia_shannon(y, sr)
        
        # Hurst
        features['hurst_exponent'] = np.array([calcular_exponente_hurst(y)])
        
        return features
    
    except Exception as e:
        print(f"Error procesando {filepath}: {str(e)}")
        return None


# ============================================================================
# FUNCIONES REUTILIZADAS DEL SCRIPT 2 (Feature Engineering)
# ============================================================================

def calcular_agregaciones(array, prefix):
    """Calcula estadísticos agregados de un array"""
    array = array[np.isfinite(array)]
    
    if len(array) == 0:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_var': np.nan,
            f'{prefix}_max': np.nan,
            f'{prefix}_min': np.nan,
            f'{prefix}_median': np.nan,
            f'{prefix}_q25': np.nan,
            f'{prefix}_q75': np.nan,
            f'{prefix}_skew': np.nan,
            f'{prefix}_kurtosis': np.nan
        }
    
    return {
        f'{prefix}_mean': np.mean(array),
        f'{prefix}_std': np.std(array),
        f'{prefix}_var': np.var(array),
        f'{prefix}_max': np.max(array),
        f'{prefix}_min': np.min(array),
        f'{prefix}_median': np.median(array),
        f'{prefix}_q25': np.percentile(array, 25),
        f'{prefix}_q75': np.percentile(array, 75),
        f'{prefix}_skew': skew(array) if len(array) > 1 else 0,
        f'{prefix}_kurtosis': kurtosis(array) if len(array) > 1 else 0
    }


def transformacion_rms_a_decibeles(rms_array, epsilon=1e-10):
    """Convierte RMS a decibeles"""
    rms_safe = np.maximum(rms_array, epsilon)
    return 20 * np.log10(rms_safe)


def transformacion_pitch_a_semitonos(pitch_array, ref_freq=440.0):
    """Convierte pitch a semitonos"""
    pitch_array = pitch_array[pitch_array > 0]
    if len(pitch_array) == 0:
        return np.array([0])
    return 12 * np.log2(pitch_array / ref_freq)


def transformacion_logaritmica_general(array, epsilon=1e-10):
    """Aplica log(1+x)"""
    array_safe = np.maximum(array, 0)
    return np.log1p(array_safe)


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def procesar_test_extraccion(audio_dir, output_csv):
    """
    Paso 1: Extrae features crudas de archivos TEST/
    """
    print("="*60)
    print("PASO 1: EXTRACCIÓN DE FEATURES CRUDAS (TEST)")
    print("="*60)
    
    # Listar archivos .wav en TEST/
    archivos = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    
    print(f"Encontrados {len(archivos)} archivos de audio en TEST/")
    
    resultados = []
    
    for filename in tqdm(archivos, desc="Extrayendo features"):
        filepath = os.path.join(audio_dir, filename)
        
        # Extraer features
        features = extraer_features_audio(filepath)
        
        if features is None:
            continue
        
        # Crear registro
        registro = {'filename': filename}
        
        # Convertir arrays a strings
        for feature_name, feature_array in features.items():
            registro[feature_name] = ','.join(map(str, feature_array))
        
        resultados.append(registro)
    
    # Guardar
    df = pd.DataFrame(resultados)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Features crudas guardadas: {output_csv}")
    print(f"  Total archivos procesados: {len(resultados)}")
    
    return df


def procesar_test_engineering(raw_csv, output_csv):
    """
    Paso 2: Aplica feature engineering (agregaciones + transformaciones)
    """
    print("\n" + "="*60)
    print("PASO 2: FEATURE ENGINEERING (TEST)")
    print("="*60)
    
    df_raw = pd.read_csv(raw_csv)
    print(f"Procesando {len(df_raw)} archivos...")
    
    resultados = []
    
    for idx, row in df_raw.iterrows():
        if idx % 20 == 0:
            print(f"  Procesando {idx}/{len(df_raw)}...")
        
        registro = {'filename': row['filename']}
        
        # Procesar cada feature
        for col in df_raw.columns:
            if col == 'filename':
                continue
            
            try:
                values_str = str(row[col])
                array = np.array([float(x) for x in values_str.split(',')])
            except:
                continue
            
            # Agregaciones estadísticas
            agregaciones = calcular_agregaciones(array, col)
            registro.update(agregaciones)
            
            # Transformaciones logarítmicas
            if col == 'rms':
                db_array = transformacion_rms_a_decibeles(array)
                agregaciones_db = calcular_agregaciones(db_array, 'rms_db')
                registro.update(agregaciones_db)
            
            elif col == 'pitch':
                semitonos_array = transformacion_pitch_a_semitonos(array)
                agregaciones_semitonos = calcular_agregaciones(semitonos_array, 'pitch_semitonos')
                registro.update(agregaciones_semitonos)
            
            elif col in ['zcr', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']:
                log_array = transformacion_logaritmica_general(array)
                agregaciones_log = calcular_agregaciones(log_array, f'{col}_log')
                registro.update(agregaciones_log)
            
            elif col == 'entropia_shannon':
                log_array = transformacion_logaritmica_general(array)
                agregaciones_log = calcular_agregaciones(log_array, 'entropia_log')
                registro.update(agregaciones_log)
        
        resultados.append(registro)
    
    # Guardar
    df_engineered = pd.DataFrame(resultados)
    df_engineered.to_csv(output_csv, index=False)
    
    print(f"\n✓ Features engineered guardadas: {output_csv}")
    print(f"  Total features: {len(df_engineered.columns) - 1}")  # -1 por filename
    
    return df_engineered


def generar_predicciones(features_csv, model_path, scaler_path, output_csv):
    """
    Paso 3: Carga modelo, normaliza features y genera predicciones
    """
    print("\n" + "="*60)
    print("PASO 3: GENERACIÓN DE PREDICCIONES")
    print("="*60)
    
    # Cargar features
    print(f"Cargando features desde: {features_csv}")
    df_features = pd.read_csv(features_csv)
    filenames = df_features['filename'].values
    
    # Separar X
    X = df_features.drop(['filename'], axis=1)
    
    # Manejar NaN e infinitos (igual que en entrenamiento)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"  Muestras: {len(X)}")
    print(f"  Features: {len(X.columns)}")
    
    # Cargar modelo y scaler
    print("\nCargando modelo y scaler...")
    with open(model_path, 'rb') as f:
        modelo = pickle.load(f)
    print(f"  ✓ Modelo cargado: {model_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✓ Scaler cargado: {scaler_path}")
    
    # Normalizar
    print("\nNormalizando features con scaler entrenado...")
    X_scaled = scaler.transform(X.values)
    print("  ✓ Features normalizadas")
    
    # Predecir
    print("\nGenerando predicciones...")
    y_pred_encoded = modelo.predict(X_scaled)
    
    # Decodificar labels: intentar usar label_encoder guardado (si existe junto al modelo)
    label_encoder_path = os.path.join(os.path.dirname(model_path), "label_encoder.pkl")
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        try:
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        except Exception:
            # Fallback en caso de incompatibilidad
            label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            y_pred = [label_map.get(int(p), str(p)) for p in y_pred_encoded]
    else:
        # Fallback si no existe label_encoder.pkl
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        y_pred = [label_map.get(int(p), str(p)) for p in y_pred_encoded]
    
    # Crear DataFrame de resultados
    df_predictions = pd.DataFrame({
        'Filename': filenames,
        'Predicted_Class': y_pred
    })
    
    # Guardar
    df_predictions.to_csv(output_csv, index=False)
    
    print(f"\n✓ Predicciones guardadas: {output_csv}")
    print(f"\nDistribución de predicciones:")
    print(df_predictions['Predicted_Class'].value_counts())
    
    return df_predictions


if __name__ == "__main__":
    # Configuración de paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Inputs
    TEST_AUDIO_DIR = os.path.join(BASE_DIR, "TEST")
    MODEL_PATH = os.path.join(BASE_DIR, "knn_model.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
    
    # Outputs
    RAW_TEST_CSV = os.path.join(BASE_DIR, "features_raw_test.csv")
    ENGINEERED_TEST_CSV = os.path.join(BASE_DIR, "features_engineered_test.csv")
    PREDICTIONS_CSV = os.path.join(BASE_DIR, "predictions.csv")
    
    print("="*60)
    print("SCRIPT 4: PREDICCIÓN EN CONJUNTO DE TEST")
    print("="*60)
    
    # Verificar que existen los archivos necesarios
    if not os.path.exists(TEST_AUDIO_DIR):
        print(f"Error: No se encuentra el directorio {TEST_AUDIO_DIR}")
        exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encuentra el modelo {MODEL_PATH}")
        print("Por favor ejecuta primero el script 3_modeling_knn.py")
        exit(1)
    
    if not os.path.exists(SCALER_PATH):
        print(f"Error: No se encuentra el scaler {SCALER_PATH}")
        print("Por favor ejecuta primero el script 3_modeling_knn.py")
        exit(1)
    
    # Pipeline completo
    try:
        # Paso 1: Extracción de features crudas
        df_raw = procesar_test_extraccion(TEST_AUDIO_DIR, RAW_TEST_CSV)
        
        # Paso 2: Feature engineering
        df_engineered = procesar_test_engineering(RAW_TEST_CSV, ENGINEERED_TEST_CSV)
        
        # Paso 3: Predicciones
        df_predictions = generar_predicciones(
            ENGINEERED_TEST_CSV, 
            MODEL_PATH, 
            SCALER_PATH, 
            PREDICTIONS_CSV
        )
        
        print("\n" + "="*60)
        print("PROCESO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"\nArchivos generados:")
        print(f"  1. {RAW_TEST_CSV}")
        print(f"  2. {ENGINEERED_TEST_CSV}")
        print(f"  3. {PREDICTIONS_CSV}")
        
    except Exception as e:
        print(f"\n❌ Error durante el proceso: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

