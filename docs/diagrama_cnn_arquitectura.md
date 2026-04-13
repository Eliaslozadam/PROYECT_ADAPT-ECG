# ADAPT-ECG — Arquitectura CNN para Detección de Arritmias
**Elias Bejarano Lozada #20213057 — Residencia Profesional — ITT**

---

## Diagrama 1 — Operación de Convolución 1D (Matemáticas)

```mermaid
flowchart TD
    subgraph FORMULA["📐 Fórmula de Convolución 1D"]
        F1["y[n] = Σ w[k] · x[n + k - pad]   (k = 0 a K-1)"]
        F2["y[n] = valor de salida en posición n\nx[ ] = señal ECG de entrada (71 muestras)\nw[k] = pesos del filtro (APRENDIDOS con backprop)\nK   = tamaño del kernel (5 en Block 1-2 | 3 en Block 3)\npad = ceros en los bordes (2 en Block 1-2 | 1 en Block 3)"]
        F1 --> F2
    end

    subgraph SEÑAL["📡 Señal ECG — latido segmentado (71 muestras)"]
        direction LR
        X0["x[0]\n0.12"]
        X1["x[1]\n-0.34"]
        X2["x[2]\n0.87 ◀"]
        X3["x[3]\n1.25 ◀"]
        X4["x[4]\n0.95 ◀"]
        X5["x[5]\n0.43 ◀"]
        X6["x[6]\n-0.12 ◀"]
        X7["x[7]\n-0.55"]
        X8["x[8]\n..."]
        X0 --- X1 --- X2 --- X3 --- X4 --- X5 --- X6 --- X7 --- X8
    end

    subgraph KERNEL["🔧 Kernel w[k] — 5 pesos entrenables (K=5)\nSe desliza sobre la señal calculando productos punto"]
        direction LR
        W0["w[0]\n0.21"]
        W1["w[1]\n-0.45"]
        W2["w[2]\n0.78"]
        W3["w[3]\n0.33"]
        W4["w[4]\n-0.67"]
        W0 --- W1 --- W2 --- W3 --- W4
    end

    subgraph CALCULO["🧮 Cálculo del valor de salida y[2]"]
        C1["y[2] = x[2]·w[0] + x[3]·w[1] + x[4]·w[2] + x[5]·w[3] + x[6]·w[4]"]
        C2["y[2] = 0.87·(0.21) + 1.25·(-0.45) + 0.95·(0.78) + 0.43·(0.33) + (-0.12)·(-0.67)"]
        C3["y[2] = 0.183 - 0.563 + 0.741 + 0.142 + 0.080 = 0.583"]
        C1 --> C2 --> C3
    end

    subgraph OPERACIONES["⚙️ Operaciones por capa"]
        BN["BatchNorm1d\nNormaliza activaciones de cada canal.\nEstabiliza el entrenamiento."]
        RE["ReLU\nf(x) = max(0, x)\nActiva solo valores positivos.\nIntroduce no-linealidad."]
        MP["MaxPool1d(2)\nReduce la longitud a la mitad.\nToma el valor máximo de cada par.\nHace el modelo más robusto."]
        DO["Dropout(p=0.5)\nDesactiva 50% de neuronas al azar.\nSolo activo en entrenamiento.\nPreviene sobreajuste."]
        BN --> RE --> MP --> DO
    end

    FORMULA --> SEÑAL
    SEÑAL --> KERNEL
    KERNEL --> CALCULO
    CALCULO --> OPERACIONES
```

---

## Diagrama 2 — Pipeline Completo CNN (Forward Pass / Inferencia)

```mermaid
flowchart LR
    subgraph INPUT["📥 INPUT"]
        I["Latido ECG\nsegmentado\ny normalizado\n\nTensor:\nbatch × 1 × 71\n\n71 muestras:\n35 antes del pico R\n+ pico R\n+ 36 después"]
    end

    subgraph B1["🔷 BLOQUE 1"]
        B1C["Conv1d\nin=1 → out=32\nkernel=5, pad=2\n→ batch×32×71\n32 filtros distintos"]
        B1N["BatchNorm1d(32)\n+ ReLU\nNormaliza y activa"]
        B1P["MaxPool1d(2)\n71 → 35 muestras"]
        B1C --> B1N --> B1P
    end

    subgraph B2["🔷 BLOQUE 2"]
        B2C["Conv1d\nin=32 → out=64\nkernel=5, pad=2\n→ batch×64×35\nCombina patrones B1"]
        B2N["BatchNorm1d(64)\n+ ReLU\nPatrones más complejos"]
        B2P["MaxPool1d(2)\n35 → 17 muestras"]
        B2C --> B2N --> B2P
    end

    subgraph B3["🔷 BLOQUE 3"]
        B3C["Conv1d\nin=64 → out=128\nkernel=3, pad=1\n→ batch×128×17\nRepresentación alto nivel"]
        B3N["BatchNorm1d(128)\n+ ReLU\nMorfología completa del latido"]
        B3P["MaxPool1d(2)\n17 → 8 muestras"]
        B3C --> B3N --> B3P
    end

    subgraph POOL["🔹 POOLING GLOBAL"]
        AP["AdaptiveAvgPool1d(1)\n8 → 1 muestra\nPromedia cada canal\nbatch×128×1"]
        FL["Flatten\nbatch×128×1\n→ batch×128\nVector 1D"]
        AP --> FL
    end

    subgraph CLAS["🔸 CLASIFICADOR (self.classifier)"]
        L1["Linear(128 → 64)\n8,192 pesos + 64 bias\nComprime la representación"]
        RU["ReLU\nf(x) = max(0, x)"]
        DR["Dropout(p=0.5)\n50% neuronas desactivadas\nSolo en entrenamiento"]
        L2["Linear(64 → 5)\n320 pesos + 5 bias\n5 logits — uno por clase"]
        L1 --> RU --> DR --> L2
    end

    subgraph OUTPUT["📤 SALIDA"]
        SM["Softmax\npredict_proba()\n5 probabilidades\nSuma = 1.0"]
        N["🔵 N — Normal\n79.3% del dataset"]
        S["🟠 S — Supraventricular\n3.2% del dataset"]
        V["🔴 V — Ventricular\n7.6% del dataset"]
        F["🟣 F — Fusión\n0.8% del dataset"]
        Q["⚫ Q — Desconocido\n9.0% del dataset"]
        SM --> N
        SM --> S
        SM --> V
        SM --> F
        SM --> Q
    end

    INPUT -->|"batch×1×71"| B1
    B1 -->|"batch×32×35"| B2
    B2 -->|"batch×64×17"| B3
    B3 -->|"batch×128×8"| POOL
    POOL -->|"batch×128"| CLAS
    CLAS -->|"5 logits"| OUTPUT
```

### Resumen de dimensiones del tensor

```mermaid
flowchart LR
    D0["Input\n1 × 71"]
    D1["Block 1\n32 × 35"]
    D2["Block 2\n64 × 17"]
    D3["Block 3\n128 × 8"]
    D4["AvgPool\n128 × 1"]
    D5["Flatten\n128"]
    D6["Linear 1\n64"]
    D7["Linear 2\n5"]
    D8["Softmax\nN/S/V/F/Q"]

    D0 -->|"Conv+BN+ReLU+MaxPool"| D1
    D1 -->|"Conv+BN+ReLU+MaxPool"| D2
    D2 -->|"Conv+BN+ReLU+MaxPool"| D3
    D3 -->|"AdaptiveAvgPool"| D4
    D4 -->|"Flatten"| D5
    D5 -->|"Linear+ReLU+Dropout"| D6
    D6 -->|"Linear"| D7
    D7 -->|"Softmax"| D8
```

### Conteo de parámetros

| Capa | Fórmula | Parámetros |
|------|---------|-----------|
| Conv1d Block 1 | 1 × 32 × 5 + 32 | 192 |
| Conv1d Block 2 | 32 × 64 × 5 + 64 | 10,304 |
| Conv1d Block 3 | 64 × 128 × 3 + 128 | 24,704 |
| BatchNorm ×3 | 2 × (32 + 64 + 128) | 448 |
| Linear(128→64) | 128 × 64 + 64 | 8,256 |
| Linear(64→5) | 64 × 5 + 5 | 325 |
| **TOTAL** | | **44,229** |

---

## Diagrama 3 — Módulo de Reentrenamiento Continuo

```mermaid
flowchart TD
    subgraph STATIC["💾 Modelo Base — ecg_cnn_base.pth"]
        MB["Entrenamiento único\n• 94,627 latidos\n• 30 épocas\n• lr = 1e-3\n• Weighted NLLLoss\n\nResultado:\nAccuracy: 93.54%\nF1 Macro: 0.7849\n\nNO cambia con nuevos datos"]
    end

    subgraph BUFFER["🗄️ Replay Buffer — replay_buffer.npz"]
        RB["Capacidad: 2,000 latidos\nAlmacena muestras pasadas\nSelección aleatoria\nPersistente entre sesiones\n\nPropósito:\nEvitar Catastrophic Forgetting.\nEl modelo sigue viendo\nmuestras de pacientes\naneriores en cada sesión."]
    end

    subgraph NUEVOS["📂 Nuevos datos de paciente"]
        ND["Registro MIT-BIH seleccionado\n• Segmentación de latidos\n• Etiquetas reales del .atr\n• Mapeo AAMI aplicado\n\nEj: Registro 208\n(N, V, S — más de 2,000 latidos)"]
    end

    subgraph MEZCLA["🔀 Mezcla de datos"]
        MX["50% Buffer + 50% Nuevos\nnp.random.permutation()\n\nGarantiza que el modelo vea\ndatos de TODOS los pacientes\nen cada sesión.\nEvita sesgo hacia datos nuevos."]
    end

    subgraph RETRAIN["🔁 incremental_train()"]
        RT["• Optimizer: Adam\n• lr = 1e-4  (10x menor)\n• Loss: CrossEntropyLoss\n• Épocas por sesión: 5\n• Batch: 64\n\nPor qué lr más pequeño:\nCambios pequeños conservan\nel conocimiento base.\nSolo ajusta ligeramente los\npesos hacia los nuevos datos."]
    end

    subgraph ADAPTIVE["✅ Modelo Adaptativo — ADAPT-ECG-RETRAINED.pth"]
        MA["Misma arquitectura CNN\nPesos ACTUALIZADOS sesión a sesión\n\nResultado Fase 3:\nAccuracy:  97.83%  (+4.29%)\nF1 Macro:  0.9071  (+0.1222)\n\nResultado Fase 4 (75,702 latidos):\nAccuracy:  97.41%  (+8.69%)\nF1 Macro:  0.8954  (+0.1573)\nMcNemar χ²=5342, p≈0\n(diferencia estadísticamente significativa)"]
    end

    subgraph HISTORIAL["📝 retrain_history.json"]
        HI["Registra por sesión:\n• Accuracy antes / después\n• F1-Score antes / después\n• Latidos procesados\n• Loss por época\n• Fecha y hora"]
    end

    MB -->|"Carga pesos iniciales"| RETRAIN
    RB -->|"Muestras del pasado"| MEZCLA
    NUEVOS -->|"Datos nuevos"| MEZCLA
    MEZCLA -->|"Dataset combinado"| RETRAIN
    RETRAIN -->|"Guarda pesos actualizados"| ADAPTIVE
    RETRAIN -->|"Registra métricas"| HISTORIAL
    ADAPTIVE -->|"Actualiza buffer\nCola FIFO — máx 2,000"| BUFFER
```

---

## Diagrama 4 — Diferencia entre Modelo Base y Adaptativo

```mermaid
flowchart LR
    subgraph ARCH["⚙️ Arquitectura (IDÉNTICA en ambos modelos)"]
        direction TB
        A1["ECG_CNN\n44,229 parámetros\n\nBlock1: Conv1d(1→32,k=5) + BN + ReLU + MaxPool\nBlock2: Conv1d(32→64,k=5) + BN + ReLU + MaxPool\nBlock3: Conv1d(64→128,k=3) + BN + ReLU + MaxPool\npool: AdaptiveAvgPool1d(1)\nclassifier: Flatten→Linear(128→64)→ReLU→Dropout→Linear(64→5)"]
    end

    subgraph BASE["💙 Modelo Base"]
        B1T["Entrenamiento: una sola vez\nDatos: todos los 94,627 latidos\nÉpocas: 30 | lr = 1e-3\nLoss: Weighted NLLLoss\n\nAccuracy: 93.54%\nF1 Macro: 0.7849"]
    end

    subgraph ADAPT["💚 Modelo Adaptativo"]
        A2T["Entrenamiento: sesión a sesión\nDatos: nuevos + buffer (50/50)\nÉpocas: 5 por sesión | lr = 1e-4\nLoss: CrossEntropyLoss\n\nAccuracy: 97.83%\nF1 Macro: 0.9071"]
    end

    ARCH -->|"Misma estructura\nDiferentes PESOS w[k]"| BASE
    ARCH -->|"Misma estructura\nPesos refinados\nprogresivamente"| ADAPT

    BASE -->|"Los pesos no cambian\ncon nuevos datos"| STATIC2["❌ No aprende de\nnuevos pacientes"]
    ADAPT -->|"Los pesos se actualizan\ncon cada sesión"| DYNAMIC["✅ Se adapta a\nnuevos pacientes\nsin olvidar los anteriores"]
```

---

*Archivo generado para: Residencia Profesional — Ingeniería Biomédica — ITT — 2026*
