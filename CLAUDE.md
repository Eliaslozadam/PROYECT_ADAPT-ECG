# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Identidad del proyecto

**Título oficial:** "Sistema inteligente con reentrenamiento continuo para la detección adaptativa de patologías cardiovasculares basado en señales ECG"

**Alumno:** Elias Bejarano Lozada (#20213057) — Ingeniería Biomédica, ITT
**Asesor:** M.C. Fortunato Ramírez Arzate
**Tipo:** Residencia Profesional (alcance académico, no producción clínica)

---

## Comandos de desarrollo

```bash
# Activar entorno virtual (Windows)
.venv\Scripts\activate

# Ejecutar la UI Streamlit
streamlit run src/ui/app.py

# Fase 1: descarga y preprocesamiento completo (requiere internet)
python scripts/run_phase1.py

# Fase 1: demo rápido con 3 registros
python scripts/run_phase1.py --demo

# Reentrenamiento automatizado (sin UI)
python scripts/auto_retrain.py
```

No hay suite de tests automatizados. Las verificaciones se hicieron en notebooks de Jupyter.

---

## Arquitectura del sistema

### Flujo de datos

```
MIT-BIH PhysioNet
      ↓ wfdb download
data/raw/*.dat, *.hea, *.atr
      ↓ scripts/regenerar_datos_128.py
  bandpass_filter() → segment_beats(BEAT_BEFORE=50, BEAT_AFTER=78) → normalize_beats()
      ↓
data/processed/X_128.npy  (94,618 × 128)
data/processed/y_128.npy  (94,618,) etiquetas AAMI 0-4
      ↓ notebooks/fase2b_entrenamiento_mejorado.ipynb  (Google Colab)
models/ecg_resnet_se_base.pth
      ↓ notebooks/fase3b_reentrenamiento.ipynb  (Google Colab)
models/ecg_resnet_se_retrained.pth  +  models/replay_buffer_128.npz
```

### Modelo activo: ECG_ResNet_SE — definido inline en `src/ui/app.py`

```python
Input: (batch, 1, 128)
Stem:   Conv1d(1→32, k=7) + BN + ReLU                          → (batch, 32, 128)
Layer1: ResBlock1D(32→32)  + MaxPool(2)                         → (batch, 32,  64)
Layer2: ResBlock1D(32→64)  + MaxPool(2)                         → (batch, 64,  32)
Layer3: ResBlock1D(64→128) + MaxPool(2)                         → (batch,128,  16)
SE:     SEBlock1D(128, ratio=8) — recalibración por canal       → (batch,128,  16)
Pool:   AdaptiveAvgPool1d(1)                                    → (batch,128)
self.classifier: Flatten → Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→5)
```

**188,469 parámetros entrenables.** ResBlock1D usa conexión residual (shortcut Conv 1×1 cuando cambian canales). SEBlock1D aprende un vector de importancia por canal (squeeze-and-excitation).

### Replay Buffer (ReplayBuffer en `app.py`)

- FIFO circular con `max_size=2000`
- En cada sesión de reentrenamiento: 50% datos nuevos + 50% muestras del buffer
- Persiste entre sesiones en `models/replay_buffer_128.npz`
- Evita catastrophic forgetting sin almacenar todo el historial

### UI Streamlit (`src/ui/app.py`, ~990 líneas)

Tres tabs: **Historial** (log de sesiones), **Comparación** (estático vs adaptativo), **Análisis** (inferencia + reentrenamiento). Dos modos de entrada en Análisis: carpeta MIT-BIH local o CSV propio. El estado entre tabs se gestiona con `st.session_state`.

---

## Notas técnicas críticas

1. **BEAT_LEN = 128** — el modelo activo (ECG_ResNet_SE) fue entrenado con ventana 128. `settings.py` tiene 72 (error histórico, ignorar). El modelo original ECG_CNN usaba 71 — solo relevante como referencia.

2. **Nombre de la capa FC:** `self.classifier` (NO `self.fc`). Keys del state_dict: `classifier.1.weight`, `classifier.4.weight`. El índice 0 de `self.classifier` es `nn.Flatten()`.

3. **scipy.stats.mcnemar no existe** — calcular manualmente: `chi2 = (abs(b-c) - 1)**2 / (b+c)`, luego `p = 1 - chi2.cdf(chi2_stat, df=1)`.

4. **MIT-BIH local** está en doble carpeta: `mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/`.

5. **Registro 124** tiene solo 88 latidos (marcapaso) — malo para demo. Usar 100, 200, 208 (+2000 latidos, variedad de clases).

6. **Focal Loss alpha:** usar `sqrt(1/count)` normalizado. Con `1/count` directo la clase F (0.8%) queda con peso 3.38 y se sobreestima masivamente.

7. **Matriz de confusión en UI** — código comentado (no eliminado) en `app.py` ~líneas 616–620. No eliminar.

---

## Clasificación AAMI (5 clases)

| Índice | Código | Descripción |
|--------|--------|-------------|
| 0 | N | Normal y unión |
| 1 | S | Ectópicos supraventriculares |
| 2 | V | Ectópicos ventriculares |
| 3 | F | Fusión ventricular/normal |
| 4 | Q | No clasificable / artefactos |

---

## Parámetros clave

```python
FS = 360           # Hz — MIT-BIH
BEAT_LEN = 128     # muestras por latido — modelo ECG_ResNet_SE (ACTIVO)
BEAT_BEFORE = 50   # muestras antes del pico R
BEAT_AFTER  = 78   # muestras después del pico R
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0
# Entrenamiento base (Fase 2B)
LR_BASE = 1e-3
EPOCHS_BASE = 50
# Reentrenamiento incremental
LR_RETRAIN = 1e-4
EPOCHS_PER_BATCH = 5
REPLAY_MAX_SIZE = 2000
```

---

## Estado del informe de residencia (LaTeX)

| Capítulo | Estado | Archivo |
|----------|--------|---------|
| Cap. 1 — Introducción | ❌ Pendiente | `Cap1_Introduccion.tex` |
| Cap. 2 — Marco Teórico | ✅ Completado | `Cap2_Marco_Teorico.tex` |
| Cap. 3 — Desarrollo y Resultados | ✅ Actualizado | `Cap3_Resultados.tex` |
| Cap. 4 — Discusión y Conclusiones | ❌ Pendiente | `Cap4_Conclusiones.tex` |

Formato ITT: Times New Roman 12pt, interlineado 1.5, márgenes 3/2.5 cm. 12 referencias IEEE.

---

## Resultados de evaluación (modelo ECG_ResNet_SE — Fase 3B, 94,618 latidos)

| Modelo | Accuracy | F1 Macro | F1 S | F1 F |
|--------|----------|----------|------|------|
| Estático (ResNet-SE base) | 97.26% | 0.8955 | 0.7983 | 0.7575 |
| Adaptativo (Replay Buffer) | **99.06%** | **0.9592** | **0.9166** | **0.9087** |

McNemar: χ²=1327.51, p≈0 → diferencia estadísticamente significativa.

**Referencia histórica (modelo original CNN-71):**
| Modelo | Accuracy | F1 Macro |
|--------|----------|----------|
| Estático CNN | 88.72% | 0.7381 |
| Adaptativo CNN | 97.41% | 0.8954 |

---

## Convenciones

- Responder siempre en **español**
- Entorno local: Windows 11, VSCode, Python `.venv/`
- Entrenamiento pesado: Google Colab con GPU T4
- Los archivos `.pth`, `data/raw/` y `data/processed/` están excluidos de git
