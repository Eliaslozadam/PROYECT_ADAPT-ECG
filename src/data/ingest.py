"""
ingest.py
=========
Módulo de adquisición de datos — Fase 1 del pipeline ADAPT-ECG.
Sistema inteligente con reentrenamiento continuo para la detección
adaptativa de patologías cardiovasculares basado en señales ECG.

Responsabilidades:
    - Descargar registros de la MIT-BIH Arrhythmia Database desde PhysioNet
      usando la librería ``wfdb``.
    - Cargar señales (.dat) y anotaciones (.atr) de registros ya descargados.
    - Exponer funciones reutilizables para los módulos de preprocesamiento
      y entrenamiento.

Base de datos:
    MIT-BIH Arrhythmia Database (Moody & Mark, 2001).
    48 registros de ECG ambulatorio de 30 minutos, 360 Hz, 2 canales.
    Fuente: https://physionet.org/content/mitdb/

Autor: Elias Bejarano Lozada
Institucion: Instituto Tecnologico de Tijuana
Residencia Profesional - Ingenieria Biomedica
"""

import wfdb
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config.settings import (
    DATA_RAW_DIR,
    MITBIH_DB_NAME,
    MITBIH_RECORDS,
    FS,
    LEAD,
)


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def download_record(record_name: str, dest_dir: Path = DATA_RAW_DIR) -> None:
    """
    Descarga un registro individual de MIT-BIH desde PhysioNet.

    Si los archivos ya existen en dest_dir la descarga se omite
    para evitar trafico innecesario.

    Parameters
    ----------
    record_name : str
        Numero de registro, p. ej. "100".
    dest_dir : Path
        Carpeta de destino para los archivos descargados.

    Raises
    ------
    ConnectionError
        Si wfdb no puede conectarse a PhysioNet.

    Examples
    --------
    >>> download_record("100")
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    signal_file = dest_dir / f"{record_name}.dat"

    if signal_file.exists():
        return  # ya descargado, no repetir

    wfdb.dl_database(
        db_dir=MITBIH_DB_NAME,
        dl_dir=str(dest_dir),
        records=[record_name],
        annotators=["atr"],
        keep_subdirs=False,
    )


def download_all(records=None, dest_dir: Path = DATA_RAW_DIR) -> None:
    """
    Descarga todos los registros de MIT-BIH listados en records.

    Muestra una barra de progreso e informa cuantos registros fueron
    descargados vs. omitidos (ya existentes).

    Parameters
    ----------
    records : list of str, optional
        Lista de nombres de registro. Por defecto usa MITBIH_RECORDS (48).
    dest_dir : Path
        Carpeta de destino.

    Examples
    --------
    >>> download_all(records=["100", "101"])
    """
    if records is None:
        records = MITBIH_RECORDS

    print(f"[ingest] Descargando {len(records)} registros MIT-BIH -> {dest_dir}")
    for rec in tqdm(records, desc="Descargando", unit="registro"):
        download_record(rec, dest_dir)
    print("[ingest] Descarga completada.")


# ---------------------------------------------------------------------------
# Carga de senal y anotaciones
# ---------------------------------------------------------------------------

def load_record(record_name: str, data_dir: Path = DATA_RAW_DIR):
    """
    Carga la senal ECG y las anotaciones de un registro MIT-BIH.

    Parameters
    ----------
    record_name : str
        Numero de registro, p. ej. "100".
    data_dir : Path
        Directorio donde estan los archivos .dat / .hea / .atr.

    Returns
    -------
    signal : np.ndarray, shape (n_samples,)
        Senal ECG del canal LEAD (por defecto MLII).
    annotation : wfdb.Annotation
        Objeto con los arrays sample (posiciones R) y symbol
        (etiquetas de cada latido).

    Raises
    ------
    FileNotFoundError
        Si el registro no existe en data_dir.

    Examples
    --------
    >>> signal, ann = load_record("100")
    >>> print(signal.shape, ann.symbol[:5])
    """
    record_path = str(data_dir / record_name)

    record = wfdb.rdrecord(record_path, channels=[LEAD])
    annotation = wfdb.rdann(record_path, "atr")

    # p_signal tiene shape (n_samples, n_channels); tomamos el canal seleccionado
    signal = record.p_signal[:, 0].astype(np.float32)

    return signal, annotation


def load_all_records(records=None, data_dir: Path = DATA_RAW_DIR) -> dict:
    """
    Carga todos los registros especificados y los devuelve en un diccionario.

    Parameters
    ----------
    records : list of str, optional
        Lista de nombres de registro. Por defecto usa MITBIH_RECORDS.
    data_dir : Path
        Directorio con los archivos descargados.

    Returns
    -------
    dict
        {record_name: (signal, annotation)} para cada registro cargado.
        Los registros que fallen se omiten con un aviso.

    Examples
    --------
    >>> data = load_all_records(records=["100", "101"])
    >>> signal, ann = data["100"]
    """
    if records is None:
        records = MITBIH_RECORDS

    data = {}
    for rec in tqdm(records, desc="Cargando registros", unit="registro"):
        try:
            signal, ann = load_record(rec, data_dir)
            data[rec] = (signal, ann)
        except FileNotFoundError:
            print(f"  [!] Registro {rec} no encontrado.")
    return data


# ---------------------------------------------------------------------------
# Informacion de un registro
# ---------------------------------------------------------------------------

def record_info(record_name: str, data_dir: Path = DATA_RAW_DIR) -> dict:
    """
    Devuelve un resumen informativo de un registro MIT-BIH.

    Util para verificar la correcta descarga y exploracion inicial.

    Parameters
    ----------
    record_name : str
        Numero de registro.
    data_dir : Path
        Directorio con los archivos.

    Returns
    -------
    dict
        Diccionario con claves: record, n_samples, duration_s,
        fs, n_beats, symbols.

    Examples
    --------
    >>> info = record_info("100")
    >>> print(info)
    """
    signal, ann = load_record(record_name, data_dir)

    from collections import Counter
    symbol_counts = dict(Counter(ann.symbol))

    return {
        "record":     record_name,
        "n_samples":  len(signal),
        "duration_s": len(signal) / FS,
        "fs":         FS,
        "n_beats":    len(ann.sample),
        "symbols":    symbol_counts,
    }
