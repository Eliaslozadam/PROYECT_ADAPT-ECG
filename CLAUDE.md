# CLAUDE.md — ADAPT-ECG

Archivo de contexto persistente para Claude Code. Se carga automáticamente en cada sesión sin importar la cuenta.

---

## Identidad del proyecto

**Título oficial:**
"Sistema inteligente con reentrenamiento continuo para la detección adaptativa de patologías cardiovasculares basado en señales ECG"

**Alumno:** Elias Bejarano Lozada (#20213057)
**Institución:** Instituto Tecnológico de Tijuana (ITT) — Ingeniería Biomédica
**Asesor:** M.C. Fortunato Ramírez Arzate
**Tipo:** Residencia Profesional (alcance académico/investigación, no producción clínica)
**Duración:** 6 meses

---

## Problema y solución

**Problema:** El _concept drift_ degrada los modelos estáticos de clasificación de ECG con el tiempo porque los patrones cardíacos varían entre pacientes y dispositivos.

**Solución:** Red neuronal convolucional 1D que se reentrenar incrementalmente con un Replay Buffer, evitando el _catastrophic forgetting_ sin reiniciar el entrenamiento desde cero.

---

## Stack tecnológico

- Python, PyTorch, scikit-learn, NumPy, SciPy, pandas
- Streamlit (UI web interactiva)
- wfdb (acceso a MIT-BIH / PhysioNet)
- Google Colab + GPU T4 (entrenamiento), VSCode (desarrollo local)
- Google Drive (respaldo)

---

## Estado del proyecto — TODAS LAS FASES COMPLETADAS

### Fase 1 — Datos ✅
- 48 registros MIT-BIH descargados en `data/raw/`
- Dataset procesado: **94,627 latidos × 71 muestras**
- Archivos: `data/processed/X.npy` (26 MB), `data/processed/y.npy`
- Distribución AAMI: N=79.3%, V=7.6%, Q=9.0%, S=3.2%, F=0.8%
- Script: `scripts/run_phase1.py`

### Fase 2 — Modelo base CNN ✅
- Modelo CNN 1D, **44,229 parámetros entrenables**
- 30 épocas, batch=64, lr=1e-3, weighted NLLLoss
- **Accuracy validación: 94.00%**
- Modelo guardado: `models/ecg_cnn_base.pth`
- Notebook: `notebooks/fase2_entrenamiento.ipynb`

### Fase 3 — Reentrenamiento continuo ✅
- Técnica: Replay Buffer (max_size=2000) + fine-tuning incremental
- Dataset dividido en 5 lotes secuenciales (~18,925 latidos c/u)
- lr=1e-4, 5 épocas por lote
- **Resultados:**
  - Modelo estático:   Accuracy=93.54%, F1 Macro=0.7849, F1 Weighted=0.9447
  - Modelo adaptativo: Accuracy=97.83%, F1 Macro=0.9071, F1 Weighted=0.9773
  - Mejora: **+4.29% accuracy, +0.1222 F1 Macro**
- Modelo guardado: `models/ADAPT-ECG-RETRAINED.pth`
- Notebook: `notebooks/fase3_reentrenamiento.ipynb`

### Fase 4 — Evaluación comparativa ✅
- 75,702 latidos (lotes 1-4)
- **Resultados formales:**
  - Estático:   Accuracy=88.72%, F1 Macro=0.7381
  - Adaptativo: Accuracy=97.41%, F1 Macro=0.8954
  - Mejora: **+8.69% accuracy, +0.1573 F1 Macro**
- McNemar: χ²=5342, p≈0 → diferencia estadísticamente significativa
- Resultados: `models/fase4_resultados.json`
- Notebook: `notebooks/fase4_evaluacion.ipynb`

### UI Streamlit ✅
- Archivo: `src/ui/app.py`
- Título: "Reentrenamiento-ECG" (cambiado de "ADAPT-ECG" en sesión 2026-03-20)
- Dos modos de entrada: MIT-BIH (carpeta local) y CSV propio
- Inferencia real con ECG_CNN usando modelos .pth locales
- Reentrenamiento incremental con Replay Buffer desde la UI
- Historial persistente en `models/retrain_history.json`
- Iniciar: `.venv\Scripts\streamlit run src\ui\app.py`

---

## Arquitectura CNN — ECG_CNN

```
Input: (batch, 1, 71)   ← BEAT_LEN=71, NO 72

Block 1: Conv1d(1→32, k=5, pad=2) + BN + ReLU + MaxPool(2)  → (batch, 32, 35)
Block 2: Conv1d(32→64, k=5, pad=2) + BN + ReLU + MaxPool(2) → (batch, 64, 17)
Block 3: Conv1d(64→128, k=3, pad=1) + BN + ReLU + MaxPool(2)→ (batch, 128, 8)
AdaptiveAvgPool1d(1)                                          → (batch, 128, 1)
self.classifier: [Flatten, Linear(128→64), ReLU, Dropout(0.5), Linear(64→5)]
Output: (batch, 5)  ← logits para 5 clases AAMI
```

**CRÍTICO:** La capa FC se llama `self.classifier` (NO `self.fc`). Keys del state_dict: `classifier.1.weight`, `classifier.4.weight`, etc. El índice 0 de `self.classifier` es `nn.Flatten()`.

---

## NOTAS TÉCNICAS CRÍTICAS

1. **BEAT_LEN = 71** — usar este valor, no 72. `settings.py` tiene 72 pero `app.py` usa 71. Los modelos `.pth` guardados fueron entrenados con 71.

2. **Clase FC:** `self.classifier` con `nn.Flatten()` en índice 0. Si cargas pesos y falla, verifica que la arquitectura use exactamente `self.classifier`.

3. **scipy.stats.mcnemar no existe** — usar chi2 manual: `chi2 = (|b-c| - 1)^2 / (b+c)`, `p = 1 - chi2.cdf(chi2_stat, df=1)`.

4. **Registro 124** — solo 88 latidos (marcapaso), mal ejemplo para demo de reentrenamiento. Usar registros 100, 200, 208 (>2000 latidos, variedad de clases).

5. **MIT-BIH local** está en: `mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/` (doble carpeta).

6. **Fase 3 "Simulación":** Los 5 lotes son datos reales divididos artificialmente para imitar llegada progresiva de pacientes. Es metodología estándar de Continual Learning, no simulación inventada.

7. **Matriz de confusión en UI:** Código comentado (no eliminado) en `app.py` ~líneas 616-620 por decisión de diseño (sesión 2026-03-20).

---

## Estructura de archivos clave

```
PROYECT_ADAPT-ECG/
├── CLAUDE.md                          ← este archivo
├── README.md
├── requirements.txt
├── src/
│   ├── config/settings.py             ← parámetros globales (FS=360, BEAT_LEN=72*)
│   ├── data/ingest.py                 ← descarga y carga de registros MIT-BIH
│   ├── data/preprocess.py             ← filtrado, segmentación, normalización
│   └── ui/app.py                      ← UI Streamlit (826 líneas)
├── scripts/
│   ├── run_phase1.py                  ← ejecuta Fase 1 completa
│   ├── generar_marco_teorico.py       ← generó Marco Teórico Word
│   ├── generar_informe_cnn.py
│   └── generar_residencia_docx.py
├── notebooks/
│   ├── fase2_entrenamiento.ipynb      ← entrenamiento CNN (Google Colab)
│   ├── fase3_reentrenamiento.ipynb    ← reentrenamiento incremental
│   └── fase4_evaluacion.ipynb         ← evaluación comparativa (439 KB)
├── models/
│   ├── ecg_cnn_base.pth               ← modelo base estático (184 KB)
│   ├── ADAPT-ECG-RETRAINED.pth        ← modelo adaptativo (184 KB)
│   ├── replay_buffer.npz              ← buffer de memoria (571 KB)
│   ├── retrain_history.json           ← historial de sesiones de reentrenamiento
│   ├── fase3_resultados.json
│   └── fase4_resultados.json
├── data/
│   ├── raw/                           ← 48 registros MIT-BIH (excluido de git)
│   ├── processed/X.npy, y.npy        ← dataset procesado (excluido de git)
│   └── sample_ecg.csv
├── docs/
│   ├── ADAPT-ECG_Documentacion_Tecnica.md/docx/pdf
│   ├── ADAPT-ECG_Marco_Teorico.docx   ← Cap. 2 informe residencia
│   ├── ADAPT-ECG_Informe_Tecnico_CNN.docx
│   ├── ADAPT-ECG_Notas_Sesion.docx
│   ├── diagrama_cnn_arquitectura.drawio/.md
│   ├── diagrama_flujo_cnn.drawio/.md
│   ├── flujo_cnn.png
│   └── fase1/2/3/4_*.png              ← figuras de evaluación
└── Residencia_Profesional (1)/        ← carpeta con LaTeX del informe
    ├── ADAPT-ECG_Residencia_Profesional.docx
    ├── Cap1_Introduccion.tex
    ├── Cap2_Marco_Teorico.tex
    ├── Cap3_Resultados.tex
    ├── Cap4_Conclusiones.tex
    ├── Portada.tex
    ├── Referencias_RP.bib
    └── Residencia_Main.pdf
```

---

## Informe de Residencia Profesional — Estado

| Capítulo | Estado | Archivo |
|----------|--------|---------|
| Cap. 1 — Introducción | ❌ Pendiente | `Cap1_Introduccion.tex` |
| Cap. 2 — Marco Teórico | ✅ Completado | `docs/ADAPT-ECG_Marco_Teorico.docx` + `Cap2_Marco_Teorico.tex` |
| Cap. 3 — Desarrollo y Resultados | ❌ Pendiente | `Cap3_Resultados.tex` |
| Cap. 4 — Discusión y Conclusiones | ❌ Pendiente | `Cap4_Conclusiones.tex` |

**Marco Teórico tiene 11 subsecciones (2.1–2.11):**
Señal ECG, Patologías cardiovasculares, AAMI EC57, MIT-BIH, Preprocesamiento, Aprendizaje supervisado, CNN, Concept Drift, Aprendizaje incremental, Replay Buffer, Métricas de evaluación. Incluye 12 refs IEEE. Formato ITT: Times New Roman 12pt, interlineado 1.5, márgenes 3/2.5 cm.

---

## Clasificación AAMI (5 clases)

| Código | Nombre | Descripción | Color UI |
|--------|--------|-------------|----------|
| N (0) | Normal | Latidos normales y de unión | #2196F3 azul |
| S (1) | Supraventricular | Ectópicos supraventriculares (A, a, J, S) | #FF9800 naranja |
| V (2) | Ventricular | Ectópicos ventriculares (V, E) | #F44336 rojo |
| F (3) | Fusión | Fusión ventricular/normal (F) | #9C27B0 púrpura |
| Q (4) | Desconocido | No clasificable / artefactos | #607D8B gris |

---

## Parámetros de preprocesamiento

```python
FS = 360           # Hz — frecuencia de muestreo MIT-BIH
LEAD = 0           # Canal MLII
FILTER_LOW_HZ = 0.5    # Filtro pasa-banda inferior
FILTER_HIGH_HZ = 40.0  # Filtro pasa-banda superior
FILTER_ORDER = 4        # Butterworth
BEAT_LEN = 71           # muestras por latido (usar este, no 72 de settings.py)
```

Normalización: z-score por latido individual (`x_norm = (x - mean) / std`).

---

## Parámetros de entrenamiento

| Parámetro | Fase 2 (base) | Fase 3 (reentrenamiento) |
|-----------|--------------|--------------------------|
| Optimizer | Adam | Adam |
| LR | 1e-3 | 1e-4 |
| Batch size | 64 | 64 |
| Épocas | 30 | 5 por lote |
| Loss | weighted NLLLoss | NLLLoss |
| Replay Buffer | — | max_size=2000 |

---

## Replay Buffer — funcionamiento

1. Al llegar nuevos datos, se agregan al buffer (FIFO, descarta los más antiguos si supera max_size).
2. En cada época de reentrenamiento: 50% datos nuevos + 50% muestreo aleatorio del buffer.
3. Persiste entre sesiones en `models/replay_buffer.npz`.
4. Evita catastrophic forgetting sin almacenar todo el historial.

---

## Cómo ejecutar

```bash
# Activar entorno virtual
.venv\Scripts\activate

# Ejecutar UI
streamlit run src\ui\app.py

# Fase 1 completa (requiere internet para descargar MIT-BIH)
python scripts\run_phase1.py

# Fase 1 demo (solo 3 registros)
python scripts\run_phase1.py --demo
```

---

## Convenciones del proyecto

- Responder siempre en **español**
- El usuario es estudiante de Ingeniería Biomédica, nivel académico (no producción)
- Entorno local: Windows 11, VSCode, Python venv en `.venv/`
- Entrenamiento pesado: Google Colab con GPU T4
- No agregar características o refactoring no solicitado
- Los archivos `.pth`, `data/raw/` y `data/processed/` están excluidos de git (son grandes)

---

## Historial de cambios importantes

### 2026-03-19
- Eliminados 8 archivos `.py` vacíos (stubs nunca implementados) de `src/`
- Eliminadas carpetas vacías, `Archivos de Google Colab/`, `.tmp.driveupload/`, `app.ipynb`
- Título oficial actualizado en todos los archivos fuente
- Generados: `docs/ADAPT-ECG_Documentacion_Tecnica.md/docx/pdf`

### 2026-03-20
- Título UI: "ADAPT-ECG" → "Reentrenamiento-ECG" (`app.py` líneas 417-418)
- Matriz de confusión ocultada en UI (código comentado, no eliminado, ~líneas 616-620)
- Generados: `docs/ADAPT-ECG_Marco_Teorico.docx`, `docs/ADAPT-ECG_Notas_Sesion.docx`
- Carpeta `.claude` marcada como oculta: `attrib +h .claude`
