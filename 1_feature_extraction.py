"""
Script 1: Extracción de Features Globales de Audio

Este script procesa archivos .wav y extrae características acústicas
usando Librosa:
- RMS (Root Mean Square): Energía/volumen
- Zero Crossing Rate: Cruces por cero
- MFCCs: Coeficientes cepstrales mel-frequency (timbre)
- Spectral Features: Centroid, Rolloff, Bandwidth
- Pitch/F0: Tono fundamental
- Entropía de Shannon: Complejidad de la señal
- Exponente de Hurst: Auto-correlación temporal

Output: features_raw_train.csv con series temporales completas
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import entropy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def calcular_exponente_hurst(serie_temporal):
    """
    Calcula el exponente de Hurst usando el método R/S (Rescaled Range)
    H ~ 0.5: Random walk (sin memoria)
    H > 0.5: Persistencia (tendencias se mantienen)
    H < 0.5: Anti-persistencia (reversión a la media)
    """
    if len(serie_temporal) < 20:
        return np.nan
    
    # Eliminar valores infinitos o NaN
    serie_temporal = serie_temporal[np.isfinite(serie_temporal)]
    if len(serie_temporal) < 20:
        return np.nan
    
    # Calcular desviaciones acumuladas
    mean_serie = np.mean(serie_temporal)
    desviaciones_acum = np.cumsum(serie_temporal - mean_serie)
    
    # Rango
    R = np.max(desviaciones_acum) - np.min(desviaciones_acum)
    
    # Desviación estándar
    S = np.std(serie_temporal)
    
    if S == 0 or R == 0:
        return np.nan
    
    # R/S ratio
    RS = R / S
    
    # H = log(R/S) / log(n/2)
    n = len(serie_temporal)
    H = np.log(RS) / np.log(n / 2)
    
    return H


def calcular_entropia_shannon(y, sr):
    """
    Calcula la entropía de Shannon del espectrograma de potencia
    Mide la complejidad/desorden de la señal de audio
    """
    # Calcular espectrograma de potencia
    S = np.abs(librosa.stft(y))
    
    # Normalizar para obtener distribución de probabilidad
    S_norm = S / np.sum(S)
    
    # Calcular entropía para cada frame temporal
    entropias = []
    for i in range(S_norm.shape[1]):
        frame = S_norm[:, i]
        frame = frame[frame > 0]  # Evitar log(0)
        if len(frame) > 0:
            ent = entropy(frame)
            entropias.append(ent)
    
    return np.array(entropias)


def extraer_pitch(y, sr):
    """
    Extrae el pitch (frecuencia fundamental) usando pYIN
    Retorna array de frecuencias (Hz)
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz (voz humana baja)
        fmax=librosa.note_to_hz('C7')   # ~2093 Hz (voz humana alta)
    )
    
    # Reemplazar NaN con 0 (silencio)
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return f0


def extraer_features_audio(filepath):
    """
    Extrae todas las features de un archivo de audio .wav
    
    Returns:
        dict: Diccionario con todas las features extraídas
    """
    try:
        # Cargar audio
        y, sr = librosa.load(filepath, sr=None)
        
        # Diccionario para almacenar features
        features = {}
        
        # 1. RMS (Root Mean Square) - Energía/Volumen
        rms = librosa.feature.rms(y=y)[0]
        features['rms'] = rms
        
        # 2. Zero Crossing Rate - Cruces por cero
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr'] = zcr
        
        # 3. MFCCs (13 coeficientes) - Timbre
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = mfccs[i]
        
        # 4. Spectral Centroid - "Brillo" del sonido
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid'] = spectral_centroid
        
        # 5. Spectral Rolloff - Límite frecuencia (85% energía)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr
        )[0]
        features['spectral_rolloff'] = spectral_rolloff
        
        # 6. Spectral Bandwidth - Ancho de banda espectral
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth'] = spectral_bandwidth
        
        # 7. Pitch/F0 - Tono fundamental
        pitch = extraer_pitch(y, sr)
        features['pitch'] = pitch
        
        # 8. Entropía de Shannon - Complejidad de la señal
        entropia = calcular_entropia_shannon(y, sr)
        features['entropia_shannon'] = entropia
        
        # 9. Exponente de Hurst - Auto-correlación
        # Calculado sobre la señal de audio completa
        hurst = calcular_exponente_hurst(y)
        features['hurst_exponent'] = np.array([hurst])  # Valor único, convertir a array
        
        return features
    
    except Exception as e:
        print(f"Error procesando {filepath}: {str(e)}")
        return None


# python
def procesar_dataset(audio_dir, labels_csv, output_csv):
    """
    Procesa todo el dataset y guarda features en CSV
    Ahora filtra `labels_csv` para quedarse solo con archivos existentes en `audio_dir`.
    """
    # Cargar labels
    labels_df = pd.read_csv(labels_csv)

    # Comprobar existencia de archivos y filtrar
    labels_df['exists'] = labels_df['Filename'].apply(
        lambda fn: os.path.exists(os.path.join(audio_dir, fn))
    )
    total_listed = len(labels_df)
    total_existing = int(labels_df['exists'].sum())

    print(f"Archivos listados en {labels_csv}: {total_listed}")
    print(f"Archivos encontrados en {audio_dir}: {total_existing}")

    # Warn about missing files (optional concise message)
    if total_existing < total_listed:
        missing = total_listed - total_existing
        print(f"Advertencia: {missing} archivos listados no se encontraron y serán omitidos.")

    # Filtrar para procesar solo los existentes
    existing_df = labels_df[labels_df['exists']].reset_index(drop=True)

    resultados = []

    print(f"Procesando {len(existing_df)} archivos de audio...")

    for idx, row in tqdm(existing_df.iterrows(), total=len(existing_df)):
        filename = row['Filename']
        label = row['Class']
        filepath = os.path.join(audio_dir, filename)

        # Extraer features
        features = extraer_features_audio(filepath)
        if features is None:
            continue

        registro = {'filename': filename, 'label': label}
        for feature_name, feature_array in features.items():
            registro[feature_name] = ','.join(map(str, feature_array))

        resultados.append(registro)

    df = pd.DataFrame(resultados)
    df.to_csv(output_csv, index=False)

    print(f"\n✓ Features guardadas en: {output_csv}")
    print(f"  Total archivos procesados: {len(resultados)}")
    print(f"  Total features extraídas: {len(df.columns) - 2}")

if __name__ == "__main__":
    # Configuración de paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_AUDIO_DIR = os.path.join(BASE_DIR, "TRAIN")
    TRAIN_LABELS_CSV = os.path.join(BASE_DIR, "TRAIN.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "features_raw_train.csv")
    
    print("="*60)
    print("SCRIPT 1: EXTRACCIÓN DE FEATURES GLOBALES")
    print("="*60)
    
    # Verificar que existen los directorios
    if not os.path.exists(TRAIN_AUDIO_DIR):
        print(f"Error: No se encuentra el directorio {TRAIN_AUDIO_DIR}")
        exit(1)
    
    if not os.path.exists(TRAIN_LABELS_CSV):
        print(f"Advertencia: No se encuentra el archivo {TRAIN_LABELS_CSV}. Generando desde los archivos en {TRAIN_AUDIO_DIR}...")
        wavs = sorted([f for f in os.listdir(TRAIN_AUDIO_DIR) if f.lower().endswith('.wav')])
        rows = []
        for w in wavs:
            # Extraer etiqueta desde el nombre de archivo: toma el primer token antes del espacio o guion bajo
            label_token = os.path.splitext(w)[0].split()[0].split('_')[0].lower()
            rows.append({'Filename': w, 'Class': label_token})
        df_gen = pd.DataFrame(rows)
        df_gen.to_csv(TRAIN_LABELS_CSV, index=False)
        print(f"  ✓ TRAIN.csv generado con {len(df_gen)} entradas. Revisa y ajusta etiquetas si es necesario.")
    
    # Procesar dataset
    procesar_dataset(TRAIN_AUDIO_DIR, TRAIN_LABELS_CSV, OUTPUT_CSV)
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)

