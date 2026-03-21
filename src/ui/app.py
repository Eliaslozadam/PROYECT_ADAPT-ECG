"""
app.py — UI Streamlit
=====================
Sistema inteligente con reentrenamiento continuo para la detección
adaptativa de patologías cardiovasculares basado en señales ECG
Modos:
  1. MIT-BIH / carpeta personalizada: inferencia + reentrenamiento incremental.
  2. CSV propio: inferencia sobre señal subida.

Autor: Elias Bejarano Lozada (#20213057)
Institución: Instituto Tecnológico de Tijuana
Residencia Profesional — Ingeniería Biomédica
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import wfdb
import seaborn as sns
import matplotlib.pyplot as plt
import io, json, copy
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix
)

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[2]
MODELS_DIR     = ROOT / "models"
MODEL_STATIC   = MODELS_DIR / "ecg_cnn_base.pth"
MODEL_ADAPTIVE = MODELS_DIR / "ADAPT-ECG-RETRAINED.pth"
BUFFER_PATH    = MODELS_DIR / "replay_buffer.npz"
HISTORY_PATH   = MODELS_DIR / "retrain_history.json"
MITBIH_DEFAULT = ROOT / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"

# ─── Constantes ───────────────────────────────────────────────────────────────
BEAT_LEN     = 71
BEAT_BEFORE  = 35
BEAT_AFTER   = 36
N_CLASSES    = 5
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_NAMES   = {"N": "Normal", "S": "Supraventricular",
                 "V": "Ventricular", "F": "Fusión", "Q": "Desconocido"}
AAMI_COLORS  = {"N": "#2196F3", "S": "#FF9800",
                 "V": "#F44336", "F": "#9C27B0", "Q": "#607D8B"}
AAMI_MAP = {
    "N": "N", ".": "N",
    "A": "S", "a": "S", "J": "S", "S": "S", "e": "S", "j": "S",
    "V": "V", "E": "V",
    "F": "F",
    "Q": "Q", "?": "Q", "/": "Q", "f": "Q", "!": "Q",
}

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
    def __init__(self, max_size=2000):
        self.X = []
        self.y = []
        self.max_size = max_size

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

# ─── Funciones de carga ───────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached(path_str: str) -> ECG_CNN:
    model = ECG_CNN(N_CLASSES)
    model.load_state_dict(torch.load(path_str, map_location="cpu"))
    model.eval()
    return model

def load_model_fresh(path: Path) -> ECG_CNN:
    """Carga sin cache — para usar después de reentrenar."""
    model = ECG_CNN(N_CLASSES)
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    return model

def scan_db_folder(folder: Path):
    """Detecta registros disponibles en una carpeta (archivos .hea)."""
    hea_files = sorted(folder.glob("*.hea"))
    records = []
    for f in hea_files:
        name = f.stem
        # Excluir archivos auxiliares con sufijos como -0, -1
        if "-" not in name:
            records.append(name)
    return records

@st.cache_data
def load_record(folder_str: str, record_id: str):
    path   = str(Path(folder_str) / record_id)
    record = wfdb.rdrecord(path)
    ann    = wfdb.rdann(path, "atr")
    signal = record.p_signal[:, 0].astype(np.float32)
    return signal, ann.sample, ann.symbol, record.fs

# ─── Procesamiento ────────────────────────────────────────────────────────────
def bandpass_filter(signal, fs):
    nyq = 0.5 * fs
    low, high = 0.5 / nyq, min(40.0 / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal.astype(np.float64)).astype(np.float32)

def segment_beats_annotated(signal, r_peaks, symbols):
    beats, valid_peaks, labels = [], [], []
    n = len(signal)
    for peak, sym in zip(r_peaks, symbols):
        if sym not in AAMI_MAP:
            continue
        s, e = peak - BEAT_BEFORE, peak + BEAT_AFTER
        if s < 0 or e > n:
            continue
        beat = signal[s:e].astype(np.float32)
        std  = beat.std()
        if std > 0:
            beat = (beat - beat.mean()) / std
        beats.append(beat)
        valid_peaks.append(peak)
        labels.append(AAMI_CLASSES.index(AAMI_MAP[sym]))
    return (np.array(beats, dtype=np.float32),
            np.array(valid_peaks),
            np.array(labels, dtype=np.int64))

def segment_beats_csv(signal, fs):
    min_dist  = int(0.3 * fs)
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    r_peaks, _ = find_peaks(signal, distance=min_dist, height=threshold)
    beats, valid_peaks = [], []
    n = len(signal)
    for peak in r_peaks:
        s, e = peak - BEAT_BEFORE, peak + BEAT_AFTER
        if s < 0 or e > n:
            continue
        beat = signal[s:e].astype(np.float32)
        std  = beat.std()
        if std > 0:
            beat = (beat - beat.mean()) / std
        beats.append(beat)
        valid_peaks.append(peak)
    return np.array(beats, dtype=np.float32), np.array(valid_peaks)

def run_inference(model, beats, batch_size=128):
    model.eval()
    X = torch.tensor(beats, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for (xb,) in loader:
            all_probs.append(model.predict_proba(xb).numpy())
    probs = np.concatenate(all_probs, axis=0)
    return probs.argmax(axis=1), probs

def incremental_train(model, beats, labels, replay_buffer,
                      epochs=5, lr=1e-4, batch_size=64):
    """Reentrenamiento incremental con Replay Buffer."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    replay_buffer.add(beats, labels)

    losses = []
    for _ in range(epochs):
        n_replay = int(len(beats) * 0.5)
        if len(replay_buffer) > 0 and n_replay > 0:
            Xr, yr = replay_buffer.sample(n_replay)
            X_comb = np.concatenate([beats, Xr], axis=0)
            y_comb = np.concatenate([labels, yr], axis=0)
        else:
            X_comb, y_comb = beats, labels

        perm    = np.random.permutation(len(X_comb))
        X_comb  = X_comb[perm]; y_comb = y_comb[perm]
        Xt = torch.tensor(X_comb, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y_comb, dtype=torch.long)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    model.eval()
    return losses

# ─── Gráficas ─────────────────────────────────────────────────────────────────
def plot_ecg_with_peaks(signal, r_peaks, preds, fs, max_samples=3600):
    sig  = signal[:max_samples]
    t    = np.arange(len(sig)) / fs
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sig, mode="lines", name="ECG",
                              line=dict(color="#1565C0", width=1), opacity=0.85))
    mask = r_peaks < max_samples
    for i, cls in enumerate(AAMI_CLASSES):
        m = (preds == i) & mask
        if not m.any(): continue
        pk = r_peaks[m]
        fig.add_trace(go.Scatter(
            x=pk/fs, y=signal[pk], mode="markers",
            name=f"{cls} — {AAMI_NAMES[cls]}",
            marker=dict(color=AAMI_COLORS[cls], size=10,
                        symbol="circle-open", line=dict(width=2.5))))
    fig.update_layout(
        title=f"ECG — primeros {len(sig)/fs:.0f} s con picos R clasificados",
        xaxis_title="Tiempo (s)", yaxis_title="Amplitud (mV)",
        height=360, margin=dict(l=10,r=10,t=45,b=10),
        legend=dict(orientation="h", y=-0.28))
    return fig

def plot_distribution(preds):
    counts = {cls: int((preds==i).sum()) for i,cls in enumerate(AAMI_CLASSES) if (preds==i).any()}
    fig = go.Figure(go.Pie(
        labels=[f"{k} — {AAMI_NAMES[k]}" for k in counts],
        values=list(counts.values()),
        marker_colors=[AAMI_COLORS[k] for k in counts],
        hole=0.45, textinfo="label+percent"))
    fig.update_layout(title="Distribución de clases",
                      height=300, margin=dict(l=10,r=10,t=40,b=10), showlegend=False)
    return fig

def plot_confusion(y_true, y_pred):
    classes = sorted(set(y_true)|set(y_pred))
    labels  = [AAMI_CLASSES[i] for i in classes]
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta Real")
    ax.set_title("Matriz de Confusión\n(normalizada por fila)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0); plt.close()
    return buf

def plot_per_class_bars(y_true, y_pred):
    rows = []
    for i in sorted(set(y_true)):
        mask = y_true == i
        if not mask.any(): continue
        tp = int(((y_pred==i) & mask).sum())
        fn = int(((y_pred!=i) & mask).sum())
        fp = int(((y_pred==i) & (y_true!=i)).sum())
        tn = int(((y_pred!=i) & (y_true!=i)).sum())
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1   = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
        rows.append({"Clase":AAMI_CLASSES[i], "Sensibilidad":round(sens,3),
                     "Especificidad":round(spec,3), "Precisión":round(prec,3),
                     "F1-Score":round(f1,3)})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for color, metric in zip(["#1976D2","#388E3C","#F57C00","#7B1FA2"],
                              ["Sensibilidad","Especificidad","Precisión","F1-Score"]):
        fig.add_trace(go.Bar(name=metric, x=df["Clase"], y=df[metric],
                              marker_color=color, opacity=0.85))
    fig.update_layout(barmode="group", title="Métricas por Clase",
                      yaxis=dict(range=[0,1.1]), height=320,
                      margin=dict(l=10,r=10,t=40,b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig, df

# ─── Historial de reentrenamiento ────────────────────────────────────────────
def load_history() -> list:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(entry: dict):
    history = load_history()
    history.append(entry)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def plot_history(history: list):
    df = pd.DataFrame(history)
    df["sesión"] = range(1, len(df) + 1)
    df["label"]  = df["sesión"].astype(str) + " — " + df["registro"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["acc_antes"] * 100,
        mode="lines+markers", name="Accuracy ANTES",
        line=dict(color="#EF5350", width=2, dash="dot"),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["acc_despues"] * 100,
        mode="lines+markers", name="Accuracy DESPUÉS",
        line=dict(color="#1976D2", width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="Evolución de Accuracy — Reentrenamiento Continuo",
        xaxis_title="Sesión de reentrenamiento",
        yaxis_title="Accuracy (%)",
        height=350, margin=dict(l=10, r=10, t=45, b=80),
        legend=dict(orientation="h", y=-0.35),
        xaxis=dict(tickangle=-30)
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["label"], y=df["f1_antes"],
        mode="lines+markers", name="F1 Macro ANTES",
        line=dict(color="#EF5350", width=2, dash="dot"),
        marker=dict(size=8)
    ))
    fig2.add_trace(go.Scatter(
        x=df["label"], y=df["f1_despues"],
        mode="lines+markers", name="F1 Macro DESPUÉS",
        line=dict(color="#388E3C", width=2),
        marker=dict(size=8)
    ))
    fig2.update_layout(
        title="Evolución de F1 Macro — Reentrenamiento Continuo",
        xaxis_title="Sesión de reentrenamiento",
        yaxis_title="F1 Macro",
        height=350, margin=dict(l=10, r=10, t=45, b=80),
        legend=dict(orientation="h", y=-0.35),
        xaxis=dict(tickangle=-30)
    )

    fig3 = go.Bar(
        x=df["label"],
        y=df["n_latidos"],
        marker_color="#7B1FA2",
        name="Latidos usados"
    )
    fig_bar = go.Figure(fig3)
    fig_bar.update_layout(
        title="Latidos por sesión de reentrenamiento",
        xaxis_title="Sesión", yaxis_title="Latidos",
        height=280, margin=dict(l=10, r=10, t=45, b=80),
        xaxis=dict(tickangle=-30)
    )

    return fig, fig2, fig_bar, df

def plot_loss_curve(losses):
    fig = go.Figure(go.Scatter(y=losses, mode="lines+markers",
                                line=dict(color="#1976D2", width=2),
                                marker=dict(size=6)))
    fig.update_layout(title="Pérdida por época (reentrenamiento)",
                      xaxis_title="Época", yaxis_title="Loss",
                      height=260, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Reentrenamiento-ECG", page_icon="🫀", layout="wide")
st.markdown("## 🫀 Reentrenamiento-ECG")
st.caption("Sistema inteligente con reentrenamiento continuo para la detección adaptativa de patologías cardiovasculares basado en señales ECG — Residencia Profesional ITT · Elias Bejarano Lozada")
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    input_mode = st.radio("Fuente de datos",
                           ["📂 Base de datos ECG (carpeta)", "📄 CSV propio"])

    st.divider()

    if "carpeta" in input_mode:

        # ── Selector de carpeta ────────────────────────────────────────────
        st.markdown("**Carpeta de registros**")

        default_path = str(MITBIH_DEFAULT) if MITBIH_DEFAULT.exists() else ""
        db_path_str  = st.text_input(
            "Ruta de la carpeta",
            value=default_path,
            placeholder="C:/ruta/a/tu/base-de-datos",
            help="Carpeta que contiene archivos .hea, .dat y .atr"
        )

        db_path = Path(db_path_str) if db_path_str else None

        if db_path and db_path.exists():
            records_found = scan_db_folder(db_path)
            if records_found:
                st.success(f"✅ {len(records_found)} registros encontrados")
                record_id = st.selectbox("Registro", records_found)
            else:
                st.warning("No se encontraron registros .hea en esa carpeta.")
                record_id = None
        else:
            if db_path_str:
                st.error("Carpeta no encontrada.")
            record_id = None

        apply_filter = st.checkbox("Filtro pasa-banda (0.5–40 Hz)", value=True)
        uploaded = None

    else:
        uploaded    = st.file_uploader("Sube señal ECG (.csv)", type=["csv"])
        fs_csv      = st.number_input("Frecuencia de muestreo (Hz)",
                                       min_value=50, max_value=2000, value=360, step=10)
        apply_filter = st.checkbox("Filtro pasa-banda (0.5–40 Hz)", value=True)
        record_id   = None
        db_path     = None

    st.divider()
    model_choice = st.radio("Modelo",
                             ["Adaptativo (Reentrenado)", "Estático (Base)"], index=0)
    run_btn = st.button("▶ Analizar", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Clases AAMI**")
    for cls, name in AAMI_NAMES.items():
        st.markdown(
            f'<span style="color:{AAMI_COLORS[cls]};font-weight:bold">{cls}</span> — {name}',
            unsafe_allow_html=True)

# ─── Verificar modelos ────────────────────────────────────────────────────────
if not MODEL_ADAPTIVE.exists() or not MODEL_STATIC.exists():
    st.error("No se encontraron los modelos en `models/`.")
    st.stop()

model_path = MODEL_ADAPTIVE if "Adaptativo" in model_choice else MODEL_STATIC

# ─── Pestañas principales ─────────────────────────────────────────────────────
tab_analysis, tab_history = st.tabs(["🔬 Análisis e Inferencia", "📈 Historial de Reentrenamiento"])

with tab_history:
    history = load_history()
    if not history:
        st.info("Aún no hay sesiones de reentrenamiento registradas. "
                "Analiza un registro y presiona **🔄 Reentrenar** para comenzar.")
    else:
        st.subheader(f"Evolución del modelo adaptativo — {len(history)} sesiones")

        fig_acc, fig_f1, fig_beats, df_hist = plot_history(history)

        st.plotly_chart(fig_acc,   use_container_width=True, key="hist_acc")
        st.plotly_chart(fig_f1,    use_container_width=True, key="hist_f1")
        st.plotly_chart(fig_beats, use_container_width=True, key="hist_beats")

        st.divider()
        st.markdown("**Tabla de sesiones**")

        df_show = df_hist[["sesión","fecha","registro","n_latidos","buffer_size",
                            "acc_antes","acc_despues","f1_antes","f1_despues",
                            "epocas","lr"]].copy()
        df_show["Δ Accuracy"] = ((df_show["acc_despues"] - df_show["acc_antes"]) * 100).round(2)
        df_show["Δ F1 Macro"] = (df_show["f1_despues"] - df_show["f1_antes"]).round(4)
        df_show.columns = ["#","Fecha","Registro","Latidos","Buffer",
                            "Acc Antes","Acc Después","F1 Antes","F1 Después",
                            "Épocas","LR","Δ Acc (%)","Δ F1"]
        st.dataframe(df_show.set_index("#"), use_container_width=True)

        col_dl, col_rst = st.columns([3,1])
        with col_dl:
            st.download_button(
                "⬇️ Exportar historial (.csv)",
                data=df_show.to_csv(index=False).encode("utf-8"),
                file_name="adapt_ecg_historial.csv", mime="text/csv"
            )
        with col_rst:
            if st.button("🗑️ Borrar historial", type="secondary"):
                HISTORY_PATH.unlink(missing_ok=True)
                st.rerun()

with tab_analysis:
    # ── MODO BASE DE DATOS (carpeta) ──────────────────────────────────────
    if "carpeta" in input_mode:

        if not db_path or not db_path.exists():
            st.info("👈 Ingresa la ruta de tu carpeta de registros ECG en el panel lateral.")
            st.stop()

        if not record_id:
            st.warning("No se encontraron registros en la carpeta seleccionada.")
            st.stop()

        # Clave única para este registro en session_state
        state_key = f"analysis_{record_id}"

        # Ejecutar análisis si se presionó Analizar O si ya hay resultados guardados
        if run_btn:
            with st.spinner(f"Cargando registro {record_id}..."):
                try:
                    signal, r_peaks, symbols, fs = load_record(str(db_path), record_id)
                except Exception as e:
                    st.error(f"Error al cargar el registro: {e}")
                    st.stop()
                if apply_filter:
                    signal = bandpass_filter(signal, fs)
                beats, valid_peaks, y_true = segment_beats_annotated(signal, r_peaks, symbols)

            if len(beats) == 0:
                st.error("No se encontraron latidos válidos.")
                st.stop()

            with st.spinner("Ejecutando inferencia..."):
                model = load_model_cached(str(model_path))
                preds_before, probs = run_inference(model, beats)

            # Guardar en session_state para que persista al presionar Reentrenar
            st.session_state[state_key] = {
                "beats": beats, "valid_peaks": valid_peaks,
                "y_true": y_true, "preds_before": preds_before,
                "probs": probs, "fs": fs, "signal": signal,
            }

        elif state_key in st.session_state:
            # Resultados ya calculados — recuperar sin reejecutar análisis
            s = st.session_state[state_key]
            beats       = s["beats"];       valid_peaks = s["valid_peaks"]
            y_true      = s["y_true"];      preds_before = s["preds_before"]
            probs       = s["probs"];       fs = s["fs"]
            signal      = s["signal"]

        else:
            st.info(f"📂 Registro: **{record_id}** — presiona **▶ Analizar** para comenzar.")
            st.markdown("""
            ### ¿Qué sucede al presionar Analizar?
            1. **Carga** la señal y anotaciones reales del registro seleccionado
            2. **Segmenta** latidos usando los picos R anotados (71 muestras c/u)
            3. **Inferencia** con la CNN — predicción de clase por latido
            4. **Comparación** predicción vs etiqueta real del cardiólogo
            5. **Métricas**: Accuracy, F1, Sensibilidad, Especificidad
            """)
            st.stop()

        with st.spinner("Ejecutando inferencia..."):
            model = load_model_cached(str(model_path))
            preds_before, probs = run_inference(model, beats)

        acc = accuracy_score(y_true, preds_before)
        f1m = f1_score(y_true, preds_before, average="macro",    zero_division=0)
        f1w = f1_score(y_true, preds_before, average="weighted", zero_division=0)
        conf = probs.max(axis=1).mean() * 100
        n_correct = int((preds_before == y_true).sum())

        st.subheader(f"Resultados — Registro {record_id}")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Latidos evaluados", f"{len(preds_before):,}")
        c2.metric("Accuracy",          f"{acc*100:.2f}%")
        c3.metric("F1 Macro",          f"{f1m:.4f}")
        c4.metric("F1 Weighted",       f"{f1w:.4f}")
        c5.metric("Confianza media",   f"{conf:.1f}%")
        st.divider()

        st.plotly_chart(plot_ecg_with_peaks(signal, valid_peaks, preds_before, fs),
                        use_container_width=True, key="ecg_main")
        st.divider()

        st.plotly_chart(plot_distribution(preds_before), use_container_width=True, key="dist_before")
        # col_d, col_cm = st.columns(2)
        # with col_d:
        #     st.plotly_chart(plot_distribution(preds_before), use_container_width=True, key="dist_before")
        # with col_cm:
        #     st.image(plot_confusion(y_true, preds_before), use_container_width=True)

        fig_bars, df_metrics = plot_per_class_bars(y_true, preds_before)
        col_bar, col_tbl = st.columns([3, 2])
        with col_bar:
            st.plotly_chart(fig_bars, use_container_width=True, key="bars_class")
        with col_tbl:
            st.markdown("**Métricas por clase**")
            st.dataframe(df_metrics.set_index("Clase"), use_container_width=True)
        st.divider()

        st.subheader("Detalle por latido")
        df_detail = pd.DataFrame({
            "Latido #":      range(1, len(preds_before)+1),
            "Pico R":        valid_peaks,
            "Tiempo (s)":    (valid_peaks/fs).round(3),
            "Etiqueta real": [AAMI_CLASSES[i] for i in y_true],
            "Predicción":    [AAMI_CLASSES[i] for i in preds_before],
            "Correcto":      ["✅" if p==r else "❌" for p,r in zip(preds_before, y_true)],
            "Confianza (%)": (probs.max(axis=1)*100).round(1),
        })
        for cls in AAMI_CLASSES:
            df_detail[f"P({cls}) %"] = (probs[:, AAMI_CLASSES.index(cls)]*100).round(1)
        st.info(f"Predicciones correctas: **{n_correct:,} / {len(preds_before):,}** "
                f"({n_correct/len(preds_before)*100:.2f}%)")
        st.dataframe(df_detail, use_container_width=True, height=350)
        st.download_button("⬇️ Descargar resultados (.csv)",
                            data=df_detail.to_csv(index=False).encode("utf-8"),
                            file_name=f"adapt_ecg_{record_id}.csv", mime="text/csv")

        # ── Reentrenamiento ───────────────────────────────────────────────
        st.divider()
        st.subheader("🔄 Reentrenamiento Continuo (Replay Buffer)")
        st.markdown(f"""
        Actualiza el modelo con los **{len(beats):,} latidos** de este registro.
        Los pesos anteriores se preservan mezclando datos nuevos con el buffer de memoria.
        """)

        col_rt1, col_rt2 = st.columns([2, 1])
        with col_rt1:
            rt_epochs = st.slider("Épocas", 1, 15, 5)
            rt_lr     = st.select_slider("Learning rate",
                                          options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                                          value=1e-4,
                                          format_func=lambda x: f"{x:.0e}")
            rt_buffer = st.slider("Tamaño Replay Buffer", 500, 5000, 2000, step=500)
        with col_rt2:
            st.metric("Accuracy ANTES", f"{acc*100:.2f}%")
            st.metric("F1 Macro ANTES", f"{f1m:.4f}")

        retrain_btn = st.button("🔄 Reentrenar modelo adaptativo",
                                 use_container_width=True, type="primary")

        if retrain_btn:
            if "Estático" in model_choice:
                st.warning("⚠️ Selecciona el **Modelo Adaptativo** para reentrenar.")
                st.stop()

            with st.spinner("Reentrenando..."):
                model_rt = load_model_fresh(MODEL_ADAPTIVE)
                rb = ReplayBuffer(max_size=rt_buffer)
                rb.load(BUFFER_PATH)
                losses = incremental_train(model_rt, beats, y_true, rb,
                                           epochs=rt_epochs, lr=rt_lr)
                torch.save(model_rt.state_dict(), str(MODEL_ADAPTIVE))
                rb.save(BUFFER_PATH)
                load_model_cached.clear()

            preds_after, probs_after = run_inference(model_rt, beats)
            acc_after = accuracy_score(y_true, preds_after)
            f1m_after = f1_score(y_true, preds_after, average="macro", zero_division=0)

            from datetime import datetime
            save_history({
                "fecha":       datetime.now().strftime("%Y-%m-%d %H:%M"),
                "registro":    record_id,
                "acc_antes":   round(float(acc), 4),
                "acc_despues": round(float(acc_after), 4),
                "f1_antes":    round(float(f1m), 4),
                "f1_despues":  round(float(f1m_after), 4),
                "n_latidos":   int(len(beats)),
                "buffer_size": int(len(rb)),
                "epocas":      rt_epochs,
                "lr":          rt_lr,
            })

            st.success("✅ Modelo reentrenado y guardado en `models/ADAPT-ECG-RETRAINED.pth`")
            st.info(f"Buffer acumulado: **{len(rb):,}** latidos en memoria.")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Accuracy DESPUÉS", f"{acc_after*100:.2f}%",
                          delta=f"{(acc_after-acc)*100:+.2f}%")
            col_b.metric("F1 Macro DESPUÉS", f"{f1m_after:.4f}",
                          delta=f"{f1m_after-f1m:+.4f}")
            col_c.metric("Buffer acumulado", f"{len(rb):,} latidos")

            st.plotly_chart(plot_loss_curve(losses), use_container_width=True, key="rt_loss")

            col_ba, col_dep = st.columns(2)
            with col_ba:
                st.markdown("**Distribución ANTES**")
                st.plotly_chart(plot_distribution(preds_before), use_container_width=True, key="rt_dist_before")
            with col_dep:
                st.markdown("**Distribución DESPUÉS**")
                st.plotly_chart(plot_distribution(preds_after), use_container_width=True, key="rt_dist_after")

            st.info("💡 Ve a la pestaña **📈 Historial** para ver la evolución acumulada.")

    # ── MODO CSV ──────────────────────────────────────────────────────────
    else:
        if uploaded is None:
            st.info("👈 Sube un archivo CSV con una columna numérica de señal ECG.")
            c1,c2,c3 = st.columns(3)
            c1.metric("Modelo adaptativo", "97.41% Acc")
            c2.metric("Modelo estático",   "88.72% Acc")
            c3.metric("Mejora",            "+8.69%")
            st.markdown("""
            ### Formato del CSV esperado
            ```
            ecg_signal
            -0.145
             0.032
             0.210
            ...
            ```
            """)
            st.stop()

        try:
            df = pd.read_csv(uploaded)
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            if not numeric_cols:
                st.error("El CSV no tiene columnas numéricas.")
                st.stop()
            selected_col = st.sidebar.selectbox("Columna ECG", numeric_cols)
            ecg_raw = df[selected_col].dropna().to_numpy(dtype=np.float32)
            fs = fs_csv
        except Exception as e:
            st.error(f"Error al leer CSV: {e}")
            st.stop()

        if not run_btn:
            t = np.arange(len(ecg_raw)) / fs
            fig_raw = go.Figure(go.Scatter(x=t, y=ecg_raw, mode="lines",
                                            line=dict(color="#1976D2", width=1)))
            fig_raw.update_layout(title="Señal ECG cargada",
                                   xaxis_title="Tiempo (s)", yaxis_title="Amplitud",
                                   height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_raw, use_container_width=True, key="csv_raw")
            st.info("Presiona **▶ Analizar** para ejecutar inferencia.")
            st.stop()

        with st.spinner("Procesando..."):
            ecg_filt = bandpass_filter(ecg_raw, fs) if apply_filter else ecg_raw.copy()
            beats, valid_peaks = segment_beats_csv(ecg_filt, fs)

        if len(beats) == 0:
            st.error("No se detectaron latidos.")
            st.stop()

        with st.spinner("Ejecutando inferencia..."):
            model = load_model_cached(str(model_path))
            preds, probs = run_inference(model, beats)

        dominant = AAMI_CLASSES[int(np.bincount(preds).argmax())]
        n_anom   = int((preds != 0).sum())

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Latidos detectados", f"{len(preds)}")
        c2.metric("Clase dominante",    f"{dominant} — {AAMI_NAMES[dominant]}")
        c3.metric("Confianza media",    f"{probs.max(axis=1).mean()*100:.1f}%")
        c4.metric("Latidos anómalos",   f"{n_anom} ({n_anom/len(preds)*100:.1f}%)")
        st.divider()

        st.plotly_chart(plot_ecg_with_peaks(ecg_filt, valid_peaks, preds, fs),
                        use_container_width=True, key="csv_ecg")

        col_d, col_conf = st.columns(2)
        with col_d:
            st.plotly_chart(plot_distribution(preds), use_container_width=True, key="csv_dist")
        with col_conf:
            fig_conf = px.histogram(x=probs.max(axis=1)*100, nbins=20,
                                     color_discrete_sequence=["#1976D2"],
                                     labels={"x": "Confianza (%)"},
                                     title="Distribución de Confianza")
            fig_conf.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10),
                                    yaxis_title="Latidos")
            st.plotly_chart(fig_conf, use_container_width=True, key="csv_conf")

        st.divider()
        st.subheader("Detalle por latido")
        df_detail = pd.DataFrame({
            "Latido #":      range(1, len(preds)+1),
            "Pico R":        valid_peaks,
            "Tiempo (s)":    (valid_peaks/fs).round(3),
            "Clase":         [AAMI_CLASSES[p] for p in preds],
            "Descripción":   [AAMI_NAMES[AAMI_CLASSES[p]] for p in preds],
            "Confianza (%)": (probs.max(axis=1)*100).round(1),
        })
        for cls in AAMI_CLASSES:
            df_detail[f"P({cls}) %"] = (probs[:,AAMI_CLASSES.index(cls)]*100).round(1)
        st.dataframe(df_detail, use_container_width=True, height=350)
        st.download_button("⬇️ Descargar resultados (.csv)",
                            data=df_detail.to_csv(index=False).encode("utf-8"),
                            file_name="adapt_ecg_resultados.csv", mime="text/csv")
