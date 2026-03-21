"""
settings.py
===========
Parámetros globales del sistema ADAPT-ECG.
Sistema inteligente con reentrenamiento continuo para la detección
adaptativa de patologías cardiovasculares basado en señales ECG.

Centraliza rutas, constantes de preprocesamiento y configuración del modelo
para que todos los módulos compartan los mismos valores sin hardcodear.

Autor: Elias Bejarano Lozada
Institución: Instituto Tecnológico de Tijuana
Residencia Profesional — Ingeniería Biomédica
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base del proyecto
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]   # raíz del repositorio

DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"        # señales MIT-BIH descargadas
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"  # latidos segmentados (numpy)
MODELS_DIR         = ROOT_DIR / "models"               # pesos del modelo guardados
LOGS_DIR           = ROOT_DIR / "docs" / "logs"        # registros de entrenamiento

# Crear directorios si no existen (útil en primera ejecución y en Colab)
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MIT-BIH Arrhythmia Database
# ---------------------------------------------------------------------------
MITBIH_DB_NAME = "mitdb"   # nombre de la base en PhysioNet

# Los 48 registros disponibles en MIT-BIH
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# ---------------------------------------------------------------------------
# Preprocesamiento de señal
# ---------------------------------------------------------------------------
FS = 360          # frecuencia de muestreo (Hz) — estándar MIT-BIH
LEAD = 0          # canal a usar: 0 = MLII (canal principal)

# Filtro pasa-banda Butterworth
FILTER_LOW_HZ  = 0.5    # frecuencia de corte inferior (Hz)
FILTER_HIGH_HZ = 40.0   # frecuencia de corte superior (Hz)
FILTER_ORDER   = 4      # orden del filtro

# Segmentación de latidos (ventana centrada en la anotación R)
BEAT_BEFORE_MS = 90     # milisegundos antes del pico R
BEAT_AFTER_MS  = 110    # milisegundos después del pico R

# Convertir ms a muestras
BEAT_BEFORE = int(BEAT_BEFORE_MS * FS / 1000)   # 32 muestras
BEAT_AFTER  = int(BEAT_AFTER_MS  * FS / 1000)   # 40 muestras
BEAT_LEN    = BEAT_BEFORE + BEAT_AFTER           # 72 muestras por latido

# ---------------------------------------------------------------------------
# Clases AAMI (estándar de clasificación de arritmias)
# ---------------------------------------------------------------------------
# Mapeo de símbolos MIT-BIH a las 5 clases AAMI recomendadas
AAMI_MAP = {
    # Normal
    "N": "N", ".": "N",
    # Supraventricular
    "A": "S", "a": "S", "J": "S", "S": "S", "e": "S", "j": "S",
    # Ventricular
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Desconocido / artefacto
    "Q": "Q", "?": "Q", "/": "Q", "f": "Q", "!": "Q",
}

AAMI_CLASSES   = ["N", "S", "V", "F", "Q"]  # orden de etiquetas
AAMI_N_CLASSES = len(AAMI_CLASSES)           # 5 clases

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
EPOCHS        = 30
VAL_SPLIT     = 0.2    # fracción de datos para validación
RANDOM_SEED   = 42
