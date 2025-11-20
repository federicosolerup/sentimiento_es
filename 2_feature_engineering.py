"""
Script 2: Ingeniería de Características (Feature Engineering)

Este script toma las features crudas (series temporales) y genera:
1. Agregaciones estadísticas: mean, std, var, max, min, median, skewness, kurtosis, percentiles
2. Transformaciones logarítmicas:
   - RMS → Decibeles (20*log10)
   - Pitch → Semitonos (12*log2)
   - Zero Crossing Rate y Spectral Features → log(1+x)

Output: features_engineered_train.csv con ~100-150 features agregadas
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


def calcular_agregaciones(array, prefix):
    """
    Calcula estadísticos agregados de un array de valores
    
    Args:
        array: numpy array con valores de la serie temporal
        prefix: string con el nombre de la feature (ej: 'rms', 'mfcc_0')
    
    Returns:
        dict: Diccionario con todas las agregaciones
    """
    # Eliminar NaN e infinitos
    array = array[np.isfinite(array)]
    
    if len(array) == 0:
        # Retornar NaN si no hay valores válidos
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
    
    agregaciones = {
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
    
    return agregaciones


def transformacion_rms_a_decibeles(rms_array, epsilon=1e-10):
    """
    Convierte RMS (amplitud) a decibeles
    dB = 20 * log10(rms)
    
    Justificación: El oído humano percibe el volumen logarítmicamente (Ley de Weber-Fechner).
    Los decibeles son la escala natural de percepción de intensidad sonora.
    """
    # Evitar log(0) sumando epsilon
    rms_safe = np.maximum(rms_array, epsilon)
    decibeles = 20 * np.log10(rms_safe)
    return decibeles


def transformacion_pitch_a_semitonos(pitch_array, ref_freq=440.0):
    """
    Convierte frecuencia (Hz) a semitonos relativos a A4 (440 Hz)
    semitonos = 12 * log2(freq / 440)
    
    Justificación: Las notas musicales están espaciadas logarítmicamente.
    Cada octava duplica la frecuencia.
    """
    # Filtrar frecuencias no válidas (silencios = 0)
    pitch_array = pitch_array[pitch_array > 0]
    
    if len(pitch_array) == 0:
        return np.array([0])
    
    semitonos = 12 * np.log2(pitch_array / ref_freq)
    return semitonos


def transformacion_logaritmica_general(array, epsilon=1e-10):
    """
    Aplica log(1 + x) para normalizar distribuciones sesgadas
    
    Justificación: Comprime rangos dinámicos grandes y hace features más robustas.
    Útil para zero_crossing_rate y spectral features que pueden tener outliers.
    """
    array_safe = np.maximum(array, 0)  # Asegurar valores no negativos
    return np.log1p(array_safe)


def procesar_features_crudas(raw_csv, output_csv):
    """
    Procesa el CSV de features crudas y genera features engineered
    
    Args:
        raw_csv: Path al CSV con features crudas (series temporales)
        output_csv: Path de salida con features agregadas
    """
    print(f"Cargando features crudas desde: {raw_csv}")
    df_raw = pd.read_csv(raw_csv)
    
    print(f"  Archivos: {len(df_raw)}")
    print(f"  Columnas: {len(df_raw.columns)}")
    
    resultados = []
    
    print("\nProcesando features...")
    
    for idx, row in df_raw.iterrows():
        if idx % 50 == 0:
            print(f"  Procesando {idx}/{len(df_raw)}...")
        
        registro = {
            'filename': row['filename'],
            'label': row['label']
        }
        
        # Procesar cada feature
        for col in df_raw.columns:
            if col in ['filename', 'label']:
                continue
            
            # Convertir string de valores separados por coma a array numpy
            try:
                values_str = str(row[col])
                array = np.array([float(x) for x in values_str.split(',')])
            except:
                print(f"  Warning: No se pudo parsear {col} en fila {idx}")
                continue
            
            # 1. Calcular agregaciones estadísticas
            agregaciones = calcular_agregaciones(array, col)
            registro.update(agregaciones)
            
            # 2. Aplicar transformaciones logarítmicas específicas
            
            # RMS → Decibeles
            if col == 'rms':
                db_array = transformacion_rms_a_decibeles(array)
                agregaciones_db = calcular_agregaciones(db_array, 'rms_db')
                registro.update(agregaciones_db)
            
            # Pitch → Semitonos
            elif col == 'pitch':
                semitonos_array = transformacion_pitch_a_semitonos(array)
                agregaciones_semitonos = calcular_agregaciones(semitonos_array, 'pitch_semitonos')
                registro.update(agregaciones_semitonos)
            
            # Zero Crossing Rate y Spectral Features → log(1+x)
            elif col in ['zcr', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']:
                log_array = transformacion_logaritmica_general(array)
                agregaciones_log = calcular_agregaciones(log_array, f'{col}_log')
                registro.update(agregaciones_log)
            
            # Entropía Shannon → log(1+x)
            elif col == 'entropia_shannon':
                log_array = transformacion_logaritmica_general(array)
                agregaciones_log = calcular_agregaciones(log_array, 'entropia_log')
                registro.update(agregaciones_log)
        
        resultados.append(registro)
    
    # Crear DataFrame final
    df_engineered = pd.DataFrame(resultados)
    
    # Guardar
    df_engineered.to_csv(output_csv, index=False)
    
    print(f"\n✓ Features engineered guardadas en: {output_csv}")
    print(f"  Total archivos: {len(df_engineered)}")
    print(f"  Total features: {len(df_engineered.columns) - 2}")  # -2 por filename y label
    print(f"\nDistribución de clases:")
    print(df_engineered['label'].value_counts())
    
    # Estadísticas de las features
    print(f"\nEstadísticas de features:")
    print(f"  Features numéricas: {len(df_engineered.select_dtypes(include=[np.number]).columns)}")
    print(f"  NaN totales: {df_engineered.isna().sum().sum()}")
    
    return df_engineered


def mostrar_ejemplos_transformaciones():
    """
    Muestra ejemplos de las transformaciones logarítmicas para el informe
    """
    print("\n" + "="*60)
    print("EJEMPLOS DE TRANSFORMACIONES LOGARÍTMICAS")
    print("="*60)
    
    # Ejemplo 1: RMS → Decibeles
    print("\n1. RMS → Decibeles (20 * log10(rms))")
    print("   Justificación: Ley de Weber-Fechner - percepción logarítmica del volumen")
    rms_ejemplos = [0.01, 0.1, 0.5, 1.0]
    print(f"   RMS:       {rms_ejemplos}")
    db_ejemplos = [20 * np.log10(x) for x in rms_ejemplos]
    print(f"   Decibeles: {[f'{x:.2f}' for x in db_ejemplos]}")
    
    # Ejemplo 2: Pitch → Semitonos
    print("\n2. Pitch → Semitonos (12 * log2(freq / 440))")
    print("   Justificación: Notas musicales espaciadas logarítmicamente")
    pitch_ejemplos = [220, 440, 880, 1760]  # A3, A4, A5, A6
    print(f"   Freq (Hz): {pitch_ejemplos}")
    semitonos_ejemplos = [12 * np.log2(x / 440.0) for x in pitch_ejemplos]
    print(f"   Semitonos: {[f'{x:.1f}' for x in semitonos_ejemplos]}")
    print("   (0 = A4, -12 = A3, +12 = A5, +24 = A6)")
    
    # Ejemplo 3: log(1+x)
    print("\n3. Zero Crossing Rate → log(1 + x)")
    print("   Justificación: Comprime rangos dinámicos, reduce impacto de outliers")
    zcr_ejemplos = [0.01, 0.1, 1.0, 10.0]
    print(f"   ZCR:       {zcr_ejemplos}")
    log_ejemplos = [np.log1p(x) for x in zcr_ejemplos]
    print(f"   log(1+x):  {[f'{x:.3f}' for x in log_ejemplos]}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Configuración de paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_CSV = os.path.join(BASE_DIR, "features_raw_train.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "features_engineered_train.csv")
    
    print("="*60)
    print("SCRIPT 2: INGENIERÍA DE CARACTERÍSTICAS")
    print("="*60)
    
    # Verificar que existe el archivo de entrada
    if not os.path.exists(RAW_CSV):
        print(f"Error: No se encuentra el archivo {RAW_CSV}")
        print("Por favor ejecuta primero el script 1_feature_extraction.py")
        exit(1)
    
    # Mostrar ejemplos de transformaciones
    mostrar_ejemplos_transformaciones()
    
    # Procesar features
    df_engineered = procesar_features_crudas(RAW_CSV, OUTPUT_CSV)
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)

