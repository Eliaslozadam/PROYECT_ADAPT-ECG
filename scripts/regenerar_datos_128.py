"""
regenerar_datos_128.py
======================
Re-segmenta todos los registros MIT-BIH con ventana de 128 muestras
(50 antes del pico R + 78 despues = 355ms a 360 Hz).

La ventana de 71 muestras (anterior) no captura bien la onda P,
que es clave para distinguir latidos supraventriculares (clase S).
Con 128 muestras se captura el ciclo cardiaco completo: P-QRS-T.

Salida:
    data/processed/X_128.npy  shape (N, 128)
    data/processed/y_128.npy  shape (N,)

Uso:
    python scripts/regenerar_datos_128.py

Tiempo estimado: ~3 min en CPU
"""

import numpy as np
import wfdb
from pathlib import Path
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
BEAT_BEFORE = 50    # muestras antes del pico R  (~139ms)
BEAT_AFTER  = 78    # muestras despues del pico R (~217ms)
BEAT_LEN    = BEAT_BEFORE + BEAT_AFTER  # 128 muestras = 355ms
FS          = 360   # Hz
LEAD        = 0     # canal MLII

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

RECORDS = [
    "100","101","102","103","104","105","106","107",
    "108","109","111","112","113","114","115","116",
    "117","118","119","121","122","123","124","200",
    "201","202","203","205","207","208","209","210",
    "212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234",
]

# Mapeo simbolos MIT-BIH -> indice AAMI (0-4)
AAMI_MAP = {
    "N": 0, ".": 0,                          # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,         # Supraventricular
    "e": 1, "j": 1,
    "V": 2, "E": 2,                          # Ventricular
    "F": 3,                                  # Fusion
    "Q": 4, "?": 4, "/": 4, "f": 4, "!": 4, # No clasificable
}

# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def bandpass(signal, low=0.5, high=40.0, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)


def zscore(beat):
    mu, sd = beat.mean(), beat.std()
    return (beat - mu) / sd if sd > 1e-8 else beat


def procesar_registro(rec_name):
    path = str(RAW_DIR / rec_name)
    if not (RAW_DIR / f"{rec_name}.dat").exists():
        return [], []

    record = wfdb.rdrecord(path)
    ann    = wfdb.rdann(path, "atr")
    sig    = bandpass(record.p_signal[:, LEAD])
    n      = len(sig)

    beats, labels = [], []
    for peak, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI_MAP:
            continue
        s, e = peak - BEAT_BEFORE, peak + BEAT_AFTER
        if s < 0 or e > n:
            continue
        beats.append(zscore(sig[s:e]))
        labels.append(AAMI_MAP[sym])

    return beats, labels


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_X, all_y = [], []
    saltados = []

    print(f"Procesando {len(RECORDS)} registros MIT-BIH")
    print(f"Ventana: {BEAT_BEFORE}+{BEAT_AFTER} = {BEAT_LEN} samples ({BEAT_LEN/FS*1000:.0f}ms)\n")

    for rec in tqdm(RECORDS, desc="Registros", unit="rec"):
        beats, labels = procesar_registro(rec)
        if not beats:
            saltados.append(rec)
            continue
        all_X.extend(beats)
        all_y.extend(labels)

    if saltados:
        print(f"\nRegistros no encontrados: {saltados}")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    np.save(OUT_DIR / "X_128.npy", X)
    np.save(OUT_DIR / "y_128.npy", y)

    clases = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
    print(f"\nDataset guardado en {OUT_DIR}/")
    print(f"  X_128.npy : {X.shape}  ({X.nbytes/1e6:.1f} MB)")
    print(f"  y_128.npy : {y.shape}")
    print("\nDistribucion de clases:")
    for idx, nombre in clases.items():
        c = int((y == idx).sum())
        print(f"  Clase {idx} ({nombre}): {c:6,}  ({100*c/len(y):.1f}%)")


if __name__ == "__main__":
    main()
