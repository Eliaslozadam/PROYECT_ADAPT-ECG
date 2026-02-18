import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# ============================
# CONFIG
# ============================
st.set_page_config(page_title="CARDIA-RT ISRA", layout="wide")

st.markdown("## CARDIA-RT ISRA")
st.caption("Adaptive ECG Intelligence with Continuous Retraining")
st.divider()

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.header("Controles")

    file = st.file_uploader("Sube ECG (.csv)", type=["csv"])

    fs = st.number_input(
        "Frecuencia de muestreo (Hz)",
        min_value=50, max_value=2000,
        value=360, step=10
    )

    apply_filter = st.checkbox("Aplicar filtro pasa-banda (0.5–40 Hz)", value=True)
    apply_norm = st.checkbox("Normalizar (z-score)", value=True)

    mode = st.radio("Modo", ["Inferencia", "Reentrenamiento (demo)"], index=0)

    run = st.button("Analizar", use_container_width=True)

# ============================
# FUNCTIONS
# ============================
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def get_numeric_columns(df: pd.DataFrame):
    return list(df.select_dtypes(include=[np.number]).columns)

def bandpass_filter(x: np.ndarray, fs: float, low=0.5, high=40.0, order=4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    lowc = low / nyq
    highc = high / nyq

    # Seguridad básica por si fs es muy bajo
    if highc >= 1.0:
        highc = 0.999

    b, a = butter(order, [lowc, highc], btype="band")
    return filtfilt(b, a, x)

def normalize_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s == 0:
        return x
    return (x - np.mean(x)) / s

def plot_ecg(x: np.ndarray, title: str):
    t = np.arange(len(x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", name=title))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="muestras",
        yaxis_title="amplitud",
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================
# MAIN
# ============================
if file is None:
    st.info("Sube un CSV con una columna numérica para comenzar.")
    st.stop()

try:
    df = load_csv(file)
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) == 0:
        st.error("El CSV no tiene columnas numéricas.")
        st.stop()

    # Selector de columna en sidebar (solo aparece si ya hay archivo)
    selected_col = st.sidebar.selectbox("Columna ECG", numeric_cols, index=0)

    ecg_raw = df[selected_col].to_numpy(dtype=float)

    # Procesamiento
    ecg_proc = ecg_raw.copy()
    if apply_filter:
        ecg_proc = bandpass_filter(ecg_proc, fs=fs)
    if apply_norm:
        ecg_proc = normalize_zscore(ecg_proc)

    # Layout: 2 columnas (crudo vs procesado) y resultados abajo
    colA, colB = st.columns([2, 2], gap="large")

    with colA:
        st.subheader("ECG crudo")
        plot_ecg(ecg_raw, "ECG crudo")

    with colB:
        st.subheader("ECG procesado")
        plot_ecg(ecg_proc, "ECG procesado")

    st.divider()

    # Resultados
    col1, col2, col3 = st.columns(3)
    col1.metric("Muestras", f"{len(ecg_raw)}")
    col2.metric("FS (Hz)", f"{fs}")
    col3.metric("Modo", mode)

    if run:
        st.success("✅ Listo para inferencia (siguiente paso: conectar modelo PyTorch)")
        # Aquí conectaremos tu pipeline real:
        # - features: extracción de características / segmentación
        # - models: inferencia (PyTorch)
        # - retraining: guardar muestra + métricas
        st.write("🧠 Predicción (demo): **Normal**")
        st.progress(0.78)
    else:
        st.info("Presiona **Analizar** para obtener predicción.")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
