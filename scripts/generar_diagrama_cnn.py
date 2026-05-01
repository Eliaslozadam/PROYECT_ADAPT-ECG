"""
Genera docs/diagrama_arquitectura_cnn.png
Diagrama visual de la arquitectura ECG_CNN — fondo blanco, estilo publicación.
Optimizado para inserción en Google Docs / Word.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrama_arquitectura_cnn.png"

# ── Paleta ───────────────────────────────────────────────────────────────────
C = {
    "input"  : "#1B2A3B",
    "conv1"  : "#1565C0",
    "bn1"    : "#1976D2",
    "pool1"  : "#6A1B9A",
    "conv2"  : "#0D47A1",
    "bn2"    : "#1565C0",
    "pool2"  : "#4A148C",
    "conv3"  : "#0A2342",
    "bn3"    : "#0D3056",
    "pool3"  : "#380E6A",
    "avg"    : "#4527A0",
    "flat"   : "#311B92",
    "fc1"    : "#E65100",
    "relu"   : "#2E7D32",
    "fc2"    : "#BF360C",
    "N"      : "#1565C0",
    "S"      : "#E65100",
    "V"      : "#C62828",
    "F"      : "#6A1B9A",
    "Q"      : "#37474F",
}
WHITE  = "#FFFFFF"
DARK   = "#1C2833"
GRAY   = "#717D7E"
BG     = "#FFFFFF"
BORDER = "#ECEFF1"

# ── Figura ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 11), dpi=160, facecolor=BG)
ax  = fig.add_axes([0.01, 0.08, 0.98, 0.84], facecolor=BG)
ax.set_xlim(0, 26)
ax.set_ylim(0, 9)
ax.axis("off")

# ── Helpers ──────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, label1, label2="", label3="",
         dim="", fs=9.5, radius=0.22, alpha=1.0):
    """Caja redondeada con hasta 3 líneas de texto + dimensión al pie."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5, edgecolor=WHITE,
        facecolor=fc, alpha=alpha, zorder=3,
    )
    ax.add_patch(patch)

    lines = [l for l in [label1, label2, label3] if l]
    n = len(lines)
    # centro vertical (deja espacio para 'dim' abajo)
    cy = y + h * (0.58 if dim else 0.50)
    step = fs * 0.017
    for i, txt in enumerate(lines):
        offset = (i - (n - 1) / 2) * step
        ax.text(x + w / 2, cy + offset, txt,
                ha="center", va="center",
                fontsize=fs, color=WHITE, fontweight="bold",
                zorder=4, clip_on=False)
    if dim:
        ax.text(x + w / 2, y + h * 0.14, dim,
                ha="center", va="center",
                fontsize=7.2, color=WHITE, alpha=0.82,
                zorder=4, clip_on=False)


def arr(ax, x1, x2, y, dim_lbl="", lw=2.2, color="#455A64"):
    """Flecha horizontal entre x1 y x2."""
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=16),
        zorder=5,
    )
    if dim_lbl:
        mx = (x1 + x2) / 2
        ax.text(mx, y + 0.22, dim_lbl,
                ha="center", va="bottom",
                fontsize=7.5, color=color,
                fontstyle="italic", zorder=6)


def grp(ax, x, y, w, h, title, fc="#F0F4F8", ec="#B0BEC5", fs=9):
    """Rectángulo de fondo para agrupar capas."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.28",
        linewidth=1.2, edgecolor=ec,
        facecolor=fc, alpha=0.50, zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h + 0.04, title,
            ha="center", va="bottom",
            fontsize=fs, color=ec, fontweight="bold", zorder=2)


def mini_arr(ax, x1, x2, y):
    """Flecha pequeña entre capas del mismo bloque."""
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color="#90A4AE",
                        lw=1.0, mutation_scale=10),
        zorder=5,
    )


# ── Constantes de layout ─────────────────────────────────────────────────────
BH   = 6.4          # alto de cada bloque
BY   = 1.0          # y base
CY   = BY + BH / 2  # centro vertical
BW   = 0.90         # ancho de cada caja
GAP  = 0.16         # gap entre cajas del mismo bloque
GARR = 0.40         # espacio de flecha entre grupos

# Inicio x de cada grupo
xi = {}
xi["inp"]  = 0.25
xi["b1"]   = xi["inp"] + 1.10 + GARR
xi["b2"]   = xi["b1"]  + BW * 3 + GAP * 2 + GARR + 0.55
xi["b3"]   = xi["b2"]  + BW * 3 + GAP * 2 + GARR + 0.55
xi["pool"] = xi["b3"]  + BW * 3 + GAP * 2 + GARR + 0.50
xi["cls"]  = xi["pool"] + BW * 2 + GAP + GARR + 0.42
xi["out"]  = xi["cls"]  + BW * 3 + GAP * 2 + GARR + 0.38

# ── TÍTULOS PRINCIPALES ───────────────────────────────────────────────────────
ax.text(13, 8.82,
        "Arquitectura ECG_CNN — Red Neuronal Convolucional 1D",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=DARK)
ax.text(13, 8.45,
        "Sistema ADAPT-ECG  ·  Clasificación de señales ECG en 5 clases AAMI EC57  ·  44,229 parámetros entrenables",
        ha="center", va="center",
        fontsize=10, color=GRAY)

# ═════════════════════════════════════════════════════════════════════════════
# ENTRADA
# ═════════════════════════════════════════════════════════════════════════════
grp(ax, xi["inp"] - 0.12, BY - 0.15, 1.10, BH + 0.30,
    "ENTRADA", fc="#EAF2FF", ec="#1565C0", fs=9.5)

rbox(ax, xi["inp"], BY + 0.55, 0.85, BH - 1.10,
     C["input"],
     "Latido ECG", "segmentado", "z-score",
     dim="batch × 1 × 71", fs=9.0)

ax.text(xi["inp"] + 0.425, BY + 0.25,
        "71 muestras · 360 Hz",
        ha="center", va="center", fontsize=7.0, color=GRAY)

arr(ax, xi["inp"] + 0.85, xi["b1"] - 0.05, CY,
    "1 × 71", color=C["conv1"])

# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 1
# ═════════════════════════════════════════════════════════════════════════════
bx = xi["b1"]
grp(ax, bx - 0.14, BY - 0.15, BW * 3 + GAP * 2 + 0.28, BH + 0.30,
    "BLOQUE 1", fc="#EBF5FB", ec="#1565C0", fs=9.5)

rbox(ax, bx,              BY + 0.30, BW, BH - 0.60,
     C["conv1"], "Conv1d", "1 → 32", "k=5  pad=2", dim="32 × 71", fs=9.2)
rbox(ax, bx + BW + GAP,   BY + 0.30, BW, BH - 0.60,
     C["bn1"],   "Batch", "Norm", "+ ReLU",         dim="32 × 71", fs=9.2)
rbox(ax, bx + BW*2+GAP*2, BY + 0.30, BW, BH - 0.60,
     C["pool1"], "Max", "Pool", "(kernel 2)",        dim="32 × 35", fs=9.2)

mini_arr(ax, bx + BW, bx + BW + GAP, CY)
mini_arr(ax, bx + BW*2 + GAP, bx + BW*2 + GAP*2, CY)

arr(ax, bx + BW*3 + GAP*2, xi["b2"] - 0.05, CY,
    "32 × 35", color=C["conv2"])

# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 2
# ═════════════════════════════════════════════════════════════════════════════
bx = xi["b2"]
grp(ax, bx - 0.14, BY - 0.15, BW * 3 + GAP * 2 + 0.28, BH + 0.30,
    "BLOQUE 2", fc="#EBF5FB", ec="#0D47A1", fs=9.5)

rbox(ax, bx,              BY + 0.30, BW, BH - 0.60,
     C["conv2"], "Conv1d", "32 → 64", "k=5  pad=2", dim="64 × 35", fs=9.2)
rbox(ax, bx + BW + GAP,   BY + 0.30, BW, BH - 0.60,
     C["bn2"],   "Batch", "Norm", "+ ReLU",          dim="64 × 35", fs=9.2)
rbox(ax, bx + BW*2+GAP*2, BY + 0.30, BW, BH - 0.60,
     C["pool2"], "Max", "Pool", "(kernel 2)",         dim="64 × 17", fs=9.2)

mini_arr(ax, bx + BW, bx + BW + GAP, CY)
mini_arr(ax, bx + BW*2 + GAP, bx + BW*2 + GAP*2, CY)

arr(ax, bx + BW*3 + GAP*2, xi["b3"] - 0.05, CY,
    "64 × 17", color=C["conv3"])

# ═════════════════════════════════════════════════════════════════════════════
# BLOQUE 3
# ═════════════════════════════════════════════════════════════════════════════
bx = xi["b3"]
grp(ax, bx - 0.14, BY - 0.15, BW * 3 + GAP * 2 + 0.28, BH + 0.30,
    "BLOQUE 3", fc="#EBF5FB", ec="#0A2342", fs=9.5)

rbox(ax, bx,              BY + 0.30, BW, BH - 0.60,
     C["conv3"], "Conv1d", "64 → 128", "k=3  pad=1", dim="128 × 17", fs=9.2)
rbox(ax, bx + BW + GAP,   BY + 0.30, BW, BH - 0.60,
     C["bn3"],   "Batch", "Norm", "+ ReLU",           dim="128 × 17", fs=9.2)
rbox(ax, bx + BW*2+GAP*2, BY + 0.30, BW, BH - 0.60,
     C["pool3"], "Max", "Pool", "(kernel 2)",          dim="128 × 8",  fs=9.2)

mini_arr(ax, bx + BW, bx + BW + GAP, CY)
mini_arr(ax, bx + BW*2 + GAP, bx + BW*2 + GAP*2, CY)

arr(ax, bx + BW*3 + GAP*2, xi["pool"] - 0.05, CY,
    "128 × 8", color=C["avg"])

# ═════════════════════════════════════════════════════════════════════════════
# POOLING GLOBAL
# ═════════════════════════════════════════════════════════════════════════════
bx = xi["pool"]
grp(ax, bx - 0.14, BY - 0.15, BW * 2 + GAP + 0.28, BH + 0.30,
    "POOLING GLOBAL", fc="#F3E5F5", ec="#4527A0", fs=9.5)

rbox(ax, bx,            BY + 0.30, BW, BH - 0.60,
     C["avg"],  "Adaptive", "AvgPool", "(out=1)", dim="128 × 1", fs=9.2)
rbox(ax, bx + BW + GAP, BY + 0.30, BW, BH - 0.60,
     C["flat"], "Flatten", "", "",               dim="128",     fs=9.5)

mini_arr(ax, bx + BW, bx + BW + GAP, CY)

arr(ax, bx + BW*2 + GAP, xi["cls"] - 0.05, CY,
    "128", color=C["fc1"])

# ═════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR
# ═════════════════════════════════════════════════════════════════════════════
bx = xi["cls"]
grp(ax, bx - 0.14, BY - 0.15, BW * 3 + GAP * 2 + 0.28, BH + 0.30,
    "CLASIFICADOR   (self.classifier)", fc="#FFF8E1", ec="#E65100", fs=9.5)

rbox(ax, bx,              BY + 0.30, BW, BH - 0.60,
     C["fc1"],  "Linear", "128 → 64", "", dim="64", fs=9.2)
rbox(ax, bx + BW + GAP,   BY + 0.30, BW, BH - 0.60,
     C["relu"], "ReLU", "Dropout", "(p = 0.5)", dim="64", fs=9.2)
rbox(ax, bx + BW*2+GAP*2, BY + 0.30, BW, BH - 0.60,
     C["fc2"],  "Linear", "64 → 5", "logits", dim="5", fs=9.2)

mini_arr(ax, bx + BW, bx + BW + GAP, CY)
mini_arr(ax, bx + BW*2 + GAP, bx + BW*2 + GAP*2, CY)

arr(ax, bx + BW*3 + GAP*2, xi["out"] - 0.05, CY,
    "5 logits", color="#C62828")

# ═════════════════════════════════════════════════════════════════════════════
# SALIDA — 5 clases
# ═════════════════════════════════════════════════════════════════════════════
bx = xi["out"]
GRP_W = 1.72
grp(ax, bx - 0.12, BY - 0.15, GRP_W, BH + 0.30,
    "SALIDA  —  AAMI EC57", fc="#E8F8F5", ec="#00796B", fs=9.5)

# Softmax
ax.text(bx + GRP_W / 2, BY + BH - 0.05,
        "Softmax  →  probabilidades",
        ha="center", va="top",
        fontsize=8.2, color="#00796B", fontweight="bold")

classes = [
    ("N", "Normal",                  C["N"]),
    ("S", "Supraventricular",        C["S"]),
    ("V", "Ventricular",             C["V"]),
    ("F", "Fusión",                  C["F"]),
    ("Q", "Desconocido",             C["Q"]),
]
n_cls = len(classes)
cls_h = (BH - 1.10) / n_cls
cls_w = GRP_W - 0.26

for i, (code, name, col) in enumerate(classes):
    cy_i = BY + 0.42 + (n_cls - 1 - i) * cls_h
    patch = FancyBboxPatch(
        (bx + 0.10, cy_i + 0.04), cls_w, cls_h - 0.09,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=1.0, edgecolor=WHITE,
        facecolor=col, alpha=0.93, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(bx + 0.10 + cls_w * 0.28,
            cy_i + 0.04 + (cls_h - 0.09) / 2,
            code,
            ha="center", va="center",
            fontsize=11.5, color=WHITE, fontweight="bold", zorder=4)
    ax.text(bx + 0.10 + cls_w * 0.62,
            cy_i + 0.04 + (cls_h - 0.09) / 2,
            name,
            ha="center", va="center",
            fontsize=8.0, color=WHITE, zorder=4)

# ═════════════════════════════════════════════════════════════════════════════
# LEYENDA
# ═════════════════════════════════════════════════════════════════════════════
leg_items = [
    (C["conv1"], "Convolución 1D"),
    (C["bn1"],   "Batch Norm + ReLU"),
    (C["pool1"], "MaxPool"),
    (C["avg"],   "AvgPool + Flatten"),
    (C["fc1"],   "Fully Connected"),
    (C["relu"],  "ReLU + Dropout"),
    ("#00796B",  "Clases de salida"),
]
n_leg  = len(leg_items)
sq     = 0.32
gap_l  = 2.60
total  = n_leg * gap_l
start  = (26 - total) / 2 + 0.20
leg_y  = 0.30

for i, (col, lbl) in enumerate(leg_items):
    lx = start + i * gap_l
    patch = FancyBboxPatch(
        (lx, leg_y), sq, sq,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=0, facecolor=col, zorder=4,
    )
    ax.add_patch(patch)
    ax.text(lx + sq + 0.12, leg_y + sq / 2, lbl,
            ha="left", va="center",
            fontsize=8.2, color=DARK, zorder=5)

# ── Pie ──────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.012,
         "Residencia Profesional  ·  Ingeniería Biomédica  ·  "
         "Instituto Tecnológico de Tijuana  ·  "
         "Elias Bejarano Lozada  #20213057  ·  2026",
         ha="center", va="bottom",
         fontsize=7.5, color="#B2BEC3", fontstyle="italic")

# ── Guardar ──────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none", format="png")
plt.close(fig)
print(f"Diagrama generado: {OUT}")
