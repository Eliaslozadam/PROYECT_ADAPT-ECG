# Sesión de trabajo — 22 de abril 2026
# ADAPT-ECG: Mejora del modelo y reentrenamiento

**Alumno:** Elias Bejarano Lozada (#20213057)
**Proyecto:** Sistema inteligente con reentrenamiento continuo para la detección adaptativa de patologías cardiovasculares basado en señales ECG

---

## 1. Punto de partida

Al inicio de la sesión el proyecto contaba con:

| Componente | Estado |
|------------|--------|
| Modelo base (ECG_CNN) | Entrenado — 44,229 params, ventana 71 samples |
| Modelo adaptativo (ADAPT-ECG-RETRAINED) | Entrenado — reentrenamiento continuo completado |
| UI Streamlit | Funcional |
| Dataset | 94,627 latidos × 71 samples |

**Métricas del modelo adaptativo original:**

| Métrica | Valor |
|---------|-------|
| Accuracy | 97.41% |
| F1 Macro | 0.8954 |
| F1 clase S (Supraventricular) | 0.7737 |
| F1 clase F (Fusión) | 0.7932 |
| F1 clase V (Ventricular) | 0.9470 |

---

## 2. Diagnóstico y motivación del cambio

Se identificaron tres cuellos de botella en el modelo original:

**a) Datos insuficientes en entrenamiento base**
El notebook `fase2_entrenamiento.ipynb` solo entrenó con ~6,324 latidos (6.7% del dataset disponible). El dataset completo tiene 94,627 latidos.

**b) Ventana de tiempo limitada (71 samples = 197 ms)**
La onda P del ECG aparece ~100-120 ms antes del pico R. Con 71 muestras (35 antes del pico) la onda P quedaba prácticamente fuera de la ventana, afectando la clasificación de latidos supraventriculares (clase S), que se distinguen precisamente por morfología anormal de la onda P.

**c) Función de pérdida no óptima para desbalance**
Con NLLLoss + pesos de clase simple, las clases minoritarias (F=0.8%, S=3.2%) seguían recibiendo atención insuficiente.

---

## 3. Decisión de diseño — Opción B (Escalada)

Se decidió **reiniciar el entrenamiento desde cero** con las siguientes mejoras:

| Componente | Original | Nuevo |
|------------|----------|-------|
| Ventana | 71 samples (197 ms) | **128 samples (355 ms)** |
| Arquitectura | CNN simple (3 bloques Conv) | **ResNet1D + SE-Attention** |
| Parámetros | 44,229 | **188,469** |
| Función de pérdida | NLLLoss + class weights | **Focal Loss (γ=2, α=√(1/count))** |
| Muestreo | Aleatorio | **WeightedRandomSampler** |
| Scheduler | StepLR | **OneCycleLR (warmup 10% + cosine decay)** |
| Augmentación | Ninguna | **Ruido gaussiano + scaling de amplitud** |

---

## 4. Fase 1 complementaria — Regeneración de datos (Paso 1)

Se creó el script `scripts/regenerar_datos_128.py` que re-segmenta los 48 registros MIT-BIH con ventana de 128 samples:

- **BEAT_BEFORE = 50** muestras (139 ms antes del pico R)
- **BEAT_AFTER = 78** muestras (217 ms después del pico R)
- Misma normalización z-score por latido
- Mismo mapeo AAMI (5 clases)

**Resultado:** `data/processed/X_128.npy` (94,618 × 128) y `y_128.npy`

Nota: 9 latidos menos que el dataset de 71 samples (bordes de registro descartados al ampliar la ventana).

---

## 5. Nueva arquitectura — ECG_ResNet_SE

### Diagrama de bloques

```
Input: (batch, 1, 128)
       │
   ┌───▼────────────────────────────────┐
   │ Stem: Conv1d(1→32, k=7) + BN + ReLU│  → (batch, 32, 128)
   └───────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────────────────┐
   │ ResBlock1D(32→32): Conv+BN+ReLU+Conv+BN + shortcut(Id)   │
   │ MaxPool1d(2)                                              │  → (batch, 32, 64)
   └──────────────────────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────────────────┐
   │ ResBlock1D(32→64): Conv+BN+ReLU+Conv+BN + shortcut(1×1)  │
   │ MaxPool1d(2)                                              │  → (batch, 64, 32)
   └──────────────────────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────────────────┐
   │ ResBlock1D(64→128): Conv+BN+ReLU+Conv+BN + shortcut(1×1) │
   │ MaxPool1d(2)                                              │  → (batch, 128, 16)
   └──────────────────────────────────────────────────────────┘
       │
   ┌───▼────────────────────────────────────────────────────────┐
   │ SE-Attention: AvgPool → Linear(128→16) → ReLU              │
   │               → Linear(16→128) → Sigmoid → recalibración  │  → (batch, 128, 16)
   └────────────────────────────────────────────────────────────┘
       │
   AdaptiveAvgPool1d(1)                                          → (batch, 128)
       │
   ┌───▼──────────────────────────────────────────┐
   │ Clasificador: Flatten → Linear(128→64)        │
   │               → ReLU → Dropout(0.4)           │
   │               → Linear(64→5)                  │  → (batch, 5)
   └──────────────────────────────────────────────┘
```

**Total parámetros entrenables: 188,469**

### Componentes clave

**ResBlock1D:** Conexión residual que permite gradientes más limpios en capas profundas. El shortcut usa proyección Conv 1×1 cuando los canales cambian, e identidad cuando son iguales.

**SEBlock1D (Squeeze-and-Excitation):** Recalibra los 128 canales aprendiendo un vector de importancia por canal. Permite que el modelo enfatice las características más relevantes para cada tipo de latido.

**Focal Loss:** `FL = -α(1-pₜ)^γ · log(pₜ)` donde γ=2 hace que los ejemplos fáciles (ya bien clasificados) contribuyan menos al gradiente, enfocando el aprendizaje en los difíciles.

---

## 6. Fase 2B — Entrenamiento del modelo base mejorado

### Configuración

| Parámetro | Valor |
|-----------|-------|
| Dataset | 94,618 latidos × 128 samples |
| Split | 70% train / 15% val / 15% test (estratificado) |
| Train set | 66,270 latidos |
| Batch size | 128 |
| Épocas | 50 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=1e-3, warmup=10%) |
| Early stopping | Patience=8 épocas por F1 macro |

### Iteraciones de entrenamiento

**Run 1 — alpha = 1/count (fallido)**

Pesos clase F = 3.38 → demasiado agresivo → sobrepredicción masiva de clase F.

| Métrica | Resultado |
|---------|-----------|
| Accuracy | 95.79% |
| F1 Macro | 0.8470 |
| F1 clase S | 0.6884 |
| F1 clase F | 0.6231 (precision=0.49, recall=0.87) |

Diagnóstico: recall alto pero precisión baja en F → el modelo predice F en exceso.

**Run 2 — alpha = sqrt(1/count) (final)**

Pesos recalibrados: F=2.21 (vs 3.38 anterior), más balanceados.

| Clase | Alpha Run 1 | Alpha Run 2 |
|-------|------------|------------|
| N | 0.036 | 0.229 |
| S | 0.895 | 1.140 |
| V | 0.374 | 0.737 |
| F | 3.377 | **2.214** |
| Q | 0.318 | 0.680 |

### Resultados finales — Fase 2B (Run 2)

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| N | 0.9936 | 0.9713 | **0.9823** |
| S | 0.7054 | 0.8965 | **0.7895** |
| V | 0.9506 | 0.9585 | **0.9546** |
| F | 0.5866 | 0.8678 | **0.7000** |
| Q | 0.9448 | 0.9914 | **0.9675** |
| **Macro avg** | **0.8362** | **0.9371** | **0.8788** |

**Accuracy: 96.89%** sobre 14,193 latidos del test set.

### Comparación con modelo original

| Métrica | Adaptativo original | Nuevo base (2B) | Diferencia |
|---------|---------------------|-----------------|------------|
| Accuracy | 97.41% | 96.89% | -0.52% |
| F1 Macro | 0.8954 | 0.8788 | -0.0166 |
| F1 clase S | 0.7737 | **0.7895** | **+0.0158** ✓ |
| F1 clase V | 0.9470 | **0.9546** | **+0.0076** ✓ |
| F1 clase F | 0.7932 | 0.7000 | -0.0932 |

**Observación importante:** La comparación es asimétrica. El "adaptativo original" es el resultado tras miles de sesiones de reentrenamiento continuo. El nuevo modelo es solo el punto de partida (base). Una vez completada la Fase 3B, se espera que supere al adaptativo original.

**La clase S ya supera al modelo adaptativo anterior siendo solo el modelo base.** Esto confirma que la ventana de 128 samples beneficia directamente la detección de ectópicos supraventriculares.

---

## 7. Archivos generados en esta sesión

| Archivo | Descripción |
|---------|-------------|
| `scripts/regenerar_datos_128.py` | Regenera dataset con ventana 128 samples |
| `notebooks/fase2b_entrenamiento_mejorado.ipynb` | Notebook de entrenamiento ResNet-SE |
| `notebooks/fase3b_reentrenamiento.ipynb` | Notebook de reentrenamiento continuo |
| `models/ecg_resnet_se_base.pth` | Pesos del modelo base nuevo |
| `models/ecg_resnet_se_meta.json` | Metadatos del modelo nuevo |
| `data/processed/X_128.npy` | Dataset 94,618 × 128 samples |
| `data/processed/y_128.npy` | Etiquetas AAMI correspondientes |
| `src/ui/app.py` | Actualizado con ECG_ResNet_SE y BEAT_LEN=128 |

---

## 8. Cambios en app.py

| Parámetro | Antes | Después |
|-----------|-------|---------|
| `BEAT_LEN` | 71 | 128 |
| `BEAT_BEFORE` | 35 | 50 |
| `BEAT_AFTER` | 36 | 78 |
| Arquitectura | `ECG_CNN` | `ECG_ResNet_SE` |
| Modelo estático | `ecg_cnn_base.pth` | `ecg_resnet_se_base.pth` |
| Modelo adaptativo | `ADAPT-ECG-RETRAINED.pth` | `ecg_resnet_se_retrained.pth` |

---

## 9. Estado actual y próximos pasos

### Completado hoy ✅
- [x] Diagnóstico del modelo original
- [x] Script de regeneración de datos (ventana 128)
- [x] Dataset X_128.npy generado y subido a Google Drive
- [x] Notebook Fase 2B creado y ejecutado (2 iteraciones)
- [x] Modelo ecg_resnet_se_base.pth descargado y colocado en models/
- [x] app.py actualizado con nueva arquitectura
- [x] Notebook Fase 3B creado

### Pendiente ⏳
- [ ] **Fase 3B:** Ejecutar notebook de reentrenamiento continuo en Colab
  - Subir ecg_resnet_se_base.pth a Google Drive
  - Abrir fase3b_reentrenamiento.ipynb en Colab (GPU T4)
  - Descargar: ecg_resnet_se_retrained.pth, replay_buffer_128.npz, fase3b_resultados.json
  - Copiar a models/

- [ ] **Fase 4B:** Evaluación comparativa formal del modelo nuevo
  - Comparar: ECG_ResNet_SE estático vs ECG_ResNet_SE adaptativo
  - Comparar también contra resultados originales (CNN vs ResNet)

- [ ] **Actualizar informe LaTeX** (Cap3_Resultados.tex) con nuevos resultados

---

## 10. Notas técnicas para referencia futura

```python
# Constantes del modelo nuevo
BEAT_LEN    = 128   # NO usar 71 (ese era el anterior)
BEAT_BEFORE = 50    # muestras antes del pico R
BEAT_AFTER  = 78    # muestras después del pico R
FS          = 360   # Hz

# Focal Loss — alpha correcto
pesos_clase = 1.0 / np.sqrt(conteos_arr)   # sqrt, NO 1/count directo
# Con 1/count la clase F (0.8%) queda con peso 3.38 → sobrepredicción

# Reentrenamiento incremental
LR_RETRAIN      = 1e-4   # más bajo que entrenamiento base (1e-3)
EPOCHS_PER_BATCH = 5
REPLAY_MAX_SIZE  = 2000  # buffer FIFO

# Carga del modelo nuevo en Python
from torch import nn
model = ECG_ResNet_SE(n_classes=5)
model.load_state_dict(torch.load('models/ecg_resnet_se_base.pth', map_location='cpu'))
```
