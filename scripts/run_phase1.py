"""
run_phase1.py
=============
Script de ejecucion de la Fase 1: Adquisicion y Preprocesamiento de Datos.

Uso desde VSCode (terminal integrada):
    python scripts/run_phase1.py

Uso con subset de prueba (rapido, 3 registros):
    python scripts/run_phase1.py --demo

Que hace este script:
    1. Descarga los registros MIT-BIH desde PhysioNet (si no existen localmente).
    2. Carga las senales y anotaciones.
    3. Aplica filtrado pasa-banda (0.5-40 Hz) y segmentacion de latidos.
    4. Normaliza cada latido con z-score.
    5. Guarda el dataset procesado en data/processed/ (X.npy, y.npy).
    6. Genera una figura de verificacion visual (signal cruda vs. filtrada).

Autor: Elias Bejarano Lozada
Institucion: Instituto Tecnologico de Tijuana
Residencia Profesional - Ingenieria Biomedica
Sistema inteligente con reentrenamiento continuo para la deteccion
adaptativa de patologias cardiovasculares basado en senales ECG
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Asegurar que la raiz del proyecto este en el path de Python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import download_all, load_all_records, record_info
from src.data.preprocess import bandpass_filter, process_and_save
from src.config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR, FS, AAMI_CLASSES


DEMO_RECORDS = ["100", "101", "102"]   # subset rapido para pruebas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase 1: Adquisicion y Preprocesamiento MIT-BIH"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Usar solo 3 registros para prueba rapida (~1 min)",
    )
    return parser.parse_args()


def plot_signal_comparison(record_name="100"):
    """
    Genera una figura comparando la senal cruda vs. la senal filtrada.
    Guarda el resultado en docs/ para documentacion.
    """
    import wfdb
    record_path = str(DATA_RAW_DIR / record_name)
    rec = wfdb.rdrecord(record_path, channels=[0])
    ann = wfdb.rdann(record_path, "atr")

    signal_raw = rec.p_signal[:, 0].astype(float)
    signal_filt = bandpass_filter(signal_raw, fs=FS)

    # Mostrar 5 segundos centrados en el primer latido
    start = max(0, ann.sample[10] - int(2.5 * FS))
    end   = start + 5 * FS
    t     = np.arange(end - start) / FS

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    axes[0].plot(t, signal_raw[start:end], color="steelblue", linewidth=0.8)
    axes[0].set_title(f"Registro {record_name} - Senal cruda", fontsize=11)
    axes[0].set_ylabel("Amplitud (mV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, signal_filt[start:end], color="tomato", linewidth=0.8)
    axes[1].set_title(f"Registro {record_name} - Senal filtrada (0.5-40 Hz)", fontsize=11)
    axes[1].set_ylabel("Amplitud (mV)")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = ROOT / "docs" / "fase1_verificacion.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\n[fase1] Figura guardada en {out_path}")
    plt.show()


def main():
    args = parse_args()
    records = DEMO_RECORDS if args.demo else None

    mode = "DEMO (3 registros)" if args.demo else "COMPLETO (48 registros)"
    print("=" * 60)
    print(f"  FASE 1 - Adquisicion y Preprocesamiento")
    print(f"  Modo: {mode}")
    print("=" * 60)

    # 1. Descarga
    print("\n[1/4] Descargando registros MIT-BIH...")
    download_all(records=records)

    # 2. Informacion de un registro de ejemplo
    print("\n[2/4] Inspeccionando registro de ejemplo (100)...")
    info = record_info("100")
    print(f"  Duracion     : {info['duration_s']:.0f} s ({info['duration_s']/60:.1f} min)")
    print(f"  Muestras     : {info['n_samples']:,}")
    print(f"  Latidos      : {info['n_beats']:,}")
    print(f"  Distribucion : {info['symbols']}")

    # 3. Cargar todos los registros en memoria
    print("\n[3/4] Cargando senales y anotaciones...")
    records_data = load_all_records(records=records)

    # 4. Preprocesar y guardar
    print("\n[4/4] Preprocesando y guardando dataset...")
    X, y = process_and_save(records_data)

    # 5. Verificacion visual
    print("\nGenerando figura de verificacion...")
    plot_signal_comparison(record_name="100")

    print("\n" + "=" * 60)
    print("  FASE 1 COMPLETADA")
    print(f"  Dataset: {X.shape[0]:,} latidos x {X.shape[1]} muestras")
    print(f"  Archivos: {DATA_PROCESSED_DIR}/X.npy")
    print(f"            {DATA_PROCESSED_DIR}/y.npy")
    print("=" * 60)


if __name__ == "__main__":
    main()
