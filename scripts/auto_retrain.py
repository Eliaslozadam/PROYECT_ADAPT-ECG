"""
auto_retrain.py — Reentrenamiento automatizado por lotes
=========================================================
Itera sobre todos los registros MIT-BIH disponibles y ejecuta
N sesiones de reentrenamiento incremental por registro.

Mejoras sobre la UI original:
  1. Loss ponderada por clase (inversamente proporcional a frecuencia)
     -> penaliza mas los errores en clases minoritarias (V, S, F, Q)
  2. Replay Buffer balanceado por clase (cupo igual por clase)
     -> evita que el buffer quede dominado por clase N

Uso:
    python scripts/auto_retrain.py
    python scripts/auto_retrain.py --sesiones 1
    python scripts/auto_retrain.py --registros 100 200 208
    python scripts/auto_retrain.py --min-latidos 1000

Autor: Elias Bejarano Lozada (#20213057)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[1]
MODELS_DIR     = ROOT / "models"
MODEL_PATH     = MODELS_DIR / "ADAPT-ECG-RETRAINED.pth"
BUFFER_PATH    = MODELS_DIR / "replay_buffer.npz"
HISTORY_PATH   = MODELS_DIR / "retrain_history.json"
MITBIH_DIR     = ROOT / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"

# ─── Constantes ───────────────────────────────────────────────────────────────
BEAT_LEN     = 71
BEAT_BEFORE  = 35
BEAT_AFTER   = 36
N_CLASSES    = 5
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {
    "N": "N", ".": "N",
    "A": "S", "a": "S", "J": "S", "S": "S", "e": "S", "j": "S",
    "V": "V", "E": "V",
    "F": "F",
    "Q": "Q", "?": "Q", "/": "Q", "f": "Q", "!": "Q",
}

# Registros a excluir (marcapaso u otros problemáticos)
EXCLUDE = {"124"}

# ─── Arquitectura CNN ─────────────────────────────────────────────────────────
class ECG_CNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes))

    def forward(self, x):
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.pool(x)
        return self.classifier(x)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)

# ─── Replay Buffer ────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size=2000, n_classes=5):
        self.X = []
        self.y = []
        self.max_size = max_size
        self.n_classes = n_classes

    def add(self, X_batch, y_batch):
        self.X.extend(X_batch.tolist())
        self.y.extend(y_batch.tolist())
        if len(self.X) > self.max_size:
            self.X = self.X[-self.max_size:]
            self.y = self.y[-self.max_size:]

    def sample(self, n):
        idx = np.random.choice(len(self.X), size=min(n, len(self.X)), replace=False)
        return (np.array([self.X[i] for i in idx], dtype=np.float32),
                np.array([self.y[i] for i in idx], dtype=np.int64))

    def save(self, path):
        np.savez(path, X=np.array(self.X, dtype=np.float32),
                       y=np.array(self.y, dtype=np.int64))

    def load(self, path):
        if Path(path).exists():
            data = np.load(path)
            self.X = data["X"].tolist()
            self.y = data["y"].tolist()

    def __len__(self):
        return len(self.X)

    def class_counts(self):
        import collections
        counts = collections.Counter(self.y)
        return {c: counts.get(c, 0) for c in range(self.n_classes)}

# ─── Preprocesamiento ─────────────────────────────────────────────────────────
def bandpass_filter(signal, fs):
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = min(40.0 / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)

def segment_beats(signal, r_peaks, symbols):
    beats, labels = [], []
    n = len(signal)
    for peak, sym in zip(r_peaks, symbols):
        if sym not in AAMI_MAP:
            continue
        s, e = peak - BEAT_BEFORE, peak + BEAT_AFTER
        if s < 0 or e > n:
            continue
        beat = signal[s:e].astype(np.float32)
        std = beat.std()
        if std > 0:
            beat = (beat - beat.mean()) / std
        beats.append(beat)
        labels.append(AAMI_CLASSES.index(AAMI_MAP[sym]))
    if not beats:
        return np.empty((0, BEAT_LEN), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)

# ─── Evaluación ──────────────────────────────────────────────────────────────
def evaluate(model, beats, labels):
    model.eval()
    X = torch.tensor(beats, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        preds = model(X).argmax(dim=1).numpy()
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1

# ─── Loss ponderada por clase ─────────────────────────────────────────────────
def class_weights(labels, n_classes=5):
    """
    Pesos suavizados (raiz cuadrada inversa) solo para clases presentes.
    Si hay una sola clase en el batch, devuelve None (loss estandar).
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    clases_presentes = (counts > 0).sum()
    if clases_presentes < 2:
        return None  # sin ponderar en batches uniclase
    # Sqrt inversa: suaviza el rango de pesos, evita extremos
    counts[counts == 0] = np.inf  # clases ausentes no reciben peso
    weights = 1.0 / np.sqrt(counts)
    weights[counts == np.inf] = 0.0
    weights = weights / weights.sum() * clases_presentes
    return torch.tensor(weights, dtype=torch.float32)

# ─── Reentrenamiento incremental ─────────────────────────────────────────────
def incremental_train(model, beats, labels, replay_buffer,
                      epochs=5, lr=1e-4, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    replay_buffer.add(beats, labels)

    for _ in range(epochs):
        n_replay = int(len(beats) * 0.5)
        if len(replay_buffer) > 0 and n_replay > 0:
            Xr, yr = replay_buffer.sample(n_replay)
            X_comb = np.concatenate([beats, Xr], axis=0)
            y_comb = np.concatenate([labels, yr], axis=0)
        else:
            X_comb, y_comb = beats, labels

        # Loss ponderada solo si hay multiples clases en el batch combinado
        w = class_weights(y_comb)
        criterion = nn.CrossEntropyLoss(weight=w)

        perm   = np.random.permutation(len(X_comb))
        X_comb = X_comb[perm]; y_comb = y_comb[perm]
        Xt = torch.tensor(X_comb, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y_comb, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Reentrenamiento automatizado ADAPT-ECG")
    parser.add_argument("--sesiones",    type=int, default=2,
                        help="Sesiones de reentrenamiento por registro (default: 2)")
    parser.add_argument("--min-latidos", type=int, default=500,
                        help="Mínimo de latidos para incluir un registro (default: 500)")
    parser.add_argument("--registros",   nargs="+", default=None,
                        help="Lista de registros específicos (ej: 100 200 208)")
    parser.add_argument("--pasadas",     type=int, default=1,
                        help="Numero de pasadas completas en orden aleatorio (default: 1)")
    args = parser.parse_args()

    # Cargar modelo
    print(f"\n{'='*60}")
    print("  ADAPT-ECG - Reentrenamiento automatizado")
    print(f"{'='*60}")
    print(f"  Modelo:      {MODEL_PATH.name}")
    print(f"  Sesiones:    {args.sesiones} por registro")
    print(f"  Pasadas:     {args.pasadas} completas (orden aleatorio)")
    print(f"  Min latidos: {args.min_latidos}")
    print(f"{'='*60}\n")

    model = ECG_CNN(N_CLASSES)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
    model.eval()

    replay_buffer = ReplayBuffer(max_size=2000, n_classes=N_CLASSES)
    replay_buffer.load(BUFFER_PATH)
    counts = replay_buffer.class_counts()
    print(f"Buffer cargado: {len(replay_buffer)} latidos")
    print(f"  Clases: { {AAMI_CLASSES[c]: n for c,n in counts.items()} }\n")

    # Historial
    history = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            history = json.load(f)

    # Detectar registros disponibles
    if args.registros:
        records = [r for r in args.registros if r not in EXCLUDE]
    else:
        records = sorted([
            f.stem for f in MITBIH_DIR.glob("*.hea")
            if "-" not in f.stem and f.stem not in EXCLUDE
        ])

    print(f"Registros a procesar: {len(records)}")
    print(f"Excluidos: {EXCLUDE}\n")

    total_sesiones = 0
    skipped = 0

    # Multiples pasadas en orden aleatorio para reducir catastrophic forgetting
    import random
    pasadas = getattr(args, 'pasadas', 1)
    records_total = []
    for p in range(pasadas):
        shuffled = records.copy()
        random.shuffle(shuffled)
        records_total.extend(shuffled)

    if pasadas > 1:
        print(f"Pasadas: {pasadas} (orden aleatorio por pasada)\n")

    for i, record_id in enumerate(records_total):
        print(f"[{i+1}/{len(records)}] Registro {record_id}", end=" ... ")

        # Cargar y procesar
        try:
            rec_path = str(MITBIH_DIR / record_id)
            record   = wfdb.rdrecord(rec_path)
            ann      = wfdb.rdann(rec_path, "atr")
            signal   = record.p_signal[:, 0].astype(np.float32)
            fs       = record.fs
        except Exception as e:
            print(f"ERROR al cargar: {e}")
            skipped += 1
            continue

        signal = bandpass_filter(signal, fs)
        beats, labels = segment_beats(signal, ann.sample, ann.symbol)

        if len(beats) < args.min_latidos:
            print(f"omitido ({len(beats)} latidos < mínimo {args.min_latidos})")
            skipped += 1
            continue

        print(f"{len(beats)} latidos")

        # 2 sesiones de reentrenamiento
        for sesion in range(1, args.sesiones + 1):
            acc_antes, f1_antes = evaluate(model, beats, labels)
            incremental_train(model, beats, labels, replay_buffer)
            acc_despues, f1_despues = evaluate(model, beats, labels)

            entry = {
                "fecha":       datetime.now().strftime("%Y-%m-%d %H:%M"),
                "registro":    record_id,
                "sesion":      sesion,
                "acc_antes":   round(acc_antes, 4),
                "acc_despues": round(acc_despues, 4),
                "f1_antes":    round(f1_antes, 4),
                "f1_despues":  round(f1_despues, 4),
                "n_latidos":   len(beats),
                "buffer_size": len(replay_buffer),
                "epocas":      5,
                "lr":          0.0001,
            }
            history.append(entry)
            total_sesiones += 1

            delta_acc = acc_despues - acc_antes
            delta_f1  = f1_despues - f1_antes
            signo_acc = "+" if delta_acc >= 0 else ""
            signo_f1  = "+" if delta_f1  >= 0 else ""
            print(f"    Sesion {sesion}: Acc {acc_antes:.4f}->{acc_despues:.4f} "
                  f"({signo_acc}{delta_acc:.4f})  "
                  f"F1 {f1_antes:.4f}->{f1_despues:.4f} "
                  f"({signo_f1}{delta_f1:.4f})")

        # Guardar modelo y buffer después de cada registro
        torch.save(model.state_dict(), str(MODEL_PATH))
        replay_buffer.save(BUFFER_PATH)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    # Resumen final
    print(f"\n{'='*60}")
    print("  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Registros procesados: {total_sesiones//args.sesiones}/{len(records)} x {pasadas} pasadas")
    print(f"  Omitidos:             {skipped}")
    print(f"  Sesiones totales:     {total_sesiones}")
    print(f"  Buffer final:         {len(replay_buffer)} latidos")
    print(f"  Modelo guardado en:   {MODEL_PATH}")
    print(f"  Historial guardado en:{HISTORY_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
