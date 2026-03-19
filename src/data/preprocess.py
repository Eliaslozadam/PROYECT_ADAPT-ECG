"""
preprocess.py
=============
Modulo de preprocesamiento de senales ECG - Fase 1 del pipeline CARDIA-RT ISRA.

Responsabilidades:
    1. Filtrado pasa-banda (0.5-40 Hz) para eliminar ruido de baja frecuencia
       (deriva de linea base) y alta frecuencia (interferencia electrica).
    2. Segmentacion de latidos: extraer ventanas de longitud fija centradas
       en cada anotacion R proporcionada por MIT-BIH.
    3. Normalizacion z-score por latido para estandarizar amplitudes entre
       registros y pacientes distintos.
    4. Mapeo de etiquetas al esquema AAMI de 5 clases (N, S, V, F, Q).
    5. Guardado del dataset procesado como arrays NumPy (.npy).

Autor: Elias Bejarano Lozada
Institucion: Instituto Tecnologico de Tijuana
Residencia Profesional - Ingenieria Biomedica
"""

import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm

from src.config.settings import (
    FS,
    FILTER_LOW_HZ,
    FILTER_HIGH_HZ,
    FILTER_ORDER,
    BEAT_BEFORE,
    BEAT_AFTER,
    BEAT_LEN,
    AAMI_MAP,
    AAMI_CLASSES,
    DATA_PROCESSED_DIR,
)


def bandpass_filter(signal, fs=FS, low=FILTER_LOW_HZ, high=FILTER_HIGH_HZ, order=FILTER_ORDER):
    """
    Aplica un filtro pasa-banda Butterworth a la senal ECG.

    El filtro elimina:
    - Componentes < 0.5 Hz: deriva de linea base (respiracion, movimiento).
    - Componentes > 40 Hz: interferencia electrica y ruido de alta frecuencia.

    Se usa filtfilt (zero-phase) para evitar distorsion de fase.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Senal ECG cruda.
    fs : float
        Frecuencia de muestreo en Hz.
    low : float
        Frecuencia de corte inferior en Hz.
    high : float
        Frecuencia de corte superior en Hz.
    order : int
        Orden del filtro Butterworth.

    Returns
    -------
    np.ndarray
        Senal filtrada.

    Examples
    --------
    >>> signal_f = bandpass_filter(signal_raw, fs=360)
    """
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = high / nyq

    if high_n >= 1.0:
        raise ValueError(
            f"La frecuencia alta ({high} Hz) debe ser menor que "
            f"la frecuencia de Nyquist ({nyq} Hz)."
        )

    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal.astype(np.float64)).astype(signal.dtype)


def segment_beats(signal, r_peaks, symbols, before=BEAT_BEFORE, after=BEAT_AFTER):
    """
    Extrae ventanas de latido centradas en los picos R anotados.

    Cada latido se representa como un vector de longitud fija
    (before + after) muestras, centrado en el pico R.
    Los latidos fuera de los limites de la senal se descartan.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Senal ECG filtrada.
    r_peaks : np.ndarray, shape (n_beats,)
        Indices de muestra de cada pico R.
    symbols : list of str
        Simbolos de anotacion correspondientes a cada pico R.
    before : int
        Muestras a incluir antes del pico R.
    after : int
        Muestras a incluir despues del pico R.

    Returns
    -------
    beats : np.ndarray, shape (n_valid, beat_len)
        Latidos segmentados.
    labels_aami : list of str
        Etiquetas AAMI (N, S, V, F, Q) de cada latido valido.

    Examples
    --------
    >>> beats, labels = segment_beats(signal_f, ann.sample, ann.symbol)
    """
    beats = []
    labels_aami = []
    n = len(signal)

    for peak, sym in zip(r_peaks, symbols):
        if sym not in AAMI_MAP:
            continue

        start = peak - before
        end = peak + after

        if start < 0 or end > n:
            continue

        beats.append(signal[start:end])
        labels_aami.append(AAMI_MAP[sym])

    return np.array(beats, dtype=np.float32), labels_aami


def normalize_beats(beats):
    """
    Aplica normalizacion z-score a cada latido de forma independiente.

    Formula: x_norm = (x - mean(x)) / std(x)

    La normalizacion por latido permite comparar senales de distintos
    pacientes sin depender de la amplitud absoluta.
    Si std == 0 el latido se devuelve sin modificar.

    Parameters
    ----------
    beats : np.ndarray, shape (n_beats, beat_len)
        Latidos crudos o filtrados.

    Returns
    -------
    np.ndarray, shape (n_beats, beat_len)
        Latidos normalizados.

    Examples
    --------
    >>> beats_norm = normalize_beats(beats)
    """
    means = beats.mean(axis=1, keepdims=True)
    stds = beats.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (beats - means) / stds


def process_record(signal, annotation, apply_filter=True, apply_norm=True):
    """
    Ejecuta el pipeline completo de preprocesamiento para un registro.

    Pasos:
        1. Filtrado pasa-banda (opcional).
        2. Segmentacion de latidos centrados en picos R.
        3. Normalizacion z-score por latido (opcional).

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
        Senal ECG cruda.
    annotation : wfdb.Annotation
        Anotaciones con sample y symbol.
    apply_filter : bool
        Aplicar filtro pasa-banda.
    apply_norm : bool
        Aplicar normalizacion z-score.

    Returns
    -------
    beats : np.ndarray, shape (n_beats, beat_len)
        Latidos listos para entrenamiento.
    labels : list of str
        Etiquetas AAMI.

    Examples
    --------
    >>> beats, labels = process_record(signal, ann)
    """
    sig = signal.copy()

    if apply_filter:
        sig = bandpass_filter(sig)

    beats, labels = segment_beats(sig, annotation.sample, annotation.symbol)

    if apply_norm and len(beats) > 0:
        beats = normalize_beats(beats)

    return beats, labels


def process_and_save(records_data, out_dir=DATA_PROCESSED_DIR,
                     apply_filter=True, apply_norm=True):
    """
    Procesa todos los registros y guarda el dataset final como archivos .npy.

    Concatena los latidos de todos los registros:
    - X.npy : latidos preprocesados, shape (N_total, BEAT_LEN)
    - y.npy : etiquetas numericas AAMI, shape (N_total,)

    Codificacion numerica: N=0, S=1, V=2, F=3, Q=4.

    Parameters
    ----------
    records_data : dict
        {record_name: (signal, annotation)} como devuelve load_all_records().
    out_dir : Path
        Directorio de salida.
    apply_filter : bool
        Aplicar filtro pasa-banda.
    apply_norm : bool
        Aplicar normalizacion z-score.

    Returns
    -------
    X : np.ndarray, shape (N_total, BEAT_LEN)
    y : np.ndarray, shape (N_total,)

    Examples
    --------
    >>> X, y = process_and_save(records_data)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_to_idx = {c: i for i, c in enumerate(AAMI_CLASSES)}

    all_beats = []
    all_labels = []

    for rec_name, (signal, ann) in tqdm(records_data.items(),
                                        desc="Preprocesando",
                                        unit="registro"):
        beats, labels = process_record(signal, ann, apply_filter, apply_norm)

        if len(beats) == 0:
            continue

        all_beats.append(beats)
        all_labels.extend([label_to_idx[l] for l in labels])

    X = np.concatenate(all_beats, axis=0)
    y = np.array(all_labels, dtype=np.int64)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    print(f"\n[preprocess] Dataset guardado en {out_dir}")
    print(f"  Total latidos : {len(X)}")
    print(f"  Shape X       : {X.shape}")
    for cls, idx in label_to_idx.items():
        count = int((y == idx).sum())
        pct = 100 * count / len(y)
        print(f"  Clase {cls} ({idx})  : {count:>7,} latidos  ({pct:.1f}%)")

    return X, y


def load_processed(data_dir=DATA_PROCESSED_DIR):
    """
    Carga el dataset preprocesado guardado por process_and_save.

    Parameters
    ----------
    data_dir : Path
        Directorio con X.npy e y.npy.

    Returns
    -------
    X : np.ndarray, shape (N, BEAT_LEN)
    y : np.ndarray, shape (N,)

    Raises
    ------
    FileNotFoundError
        Si X.npy o y.npy no existen.

    Examples
    --------
    >>> X, y = load_processed()
    """
    data_dir = Path(data_dir)
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {data_dir}. "
            "Ejecuta primero process_and_save()."
        )

    return np.load(x_path), np.load(y_path)
