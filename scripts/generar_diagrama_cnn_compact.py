"""
Genera docs/diagrama_arquitectura_cnn_compact.png
Versión compacta del diagrama ECG_CNN — optimizada para Word/Google Docs.
Diseño horizontal ajustado a ancho de página A4.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrama_arquitectura_cnn_compact.png"

# ── Paleta ───────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
GRAY_LT = "#F8F9FA"
GRAY_BD = "#DEE2E6"

# Colores por tipo de capa
COL = {
    "input"  : "#263238",
    "conv"   : "#1565C0",
    "bn"     : "#0288D1",
    "pool"   : "#6A1B9A",
    "avg"    : "#4527A0",
    "flat"   : "#283593",
    "fc"     : "#E65100",
    "drop"   : "#2E7D32",
    "out_N"  : "#1565C0",
    "out_S"  : "#E65100",
    "out_V"  : "#B71C1C",
    "out_F"  : "#6A1B9A",
    "out_Q"  : "#37474F",
}
WHITE = "#FFFFFF"
DARK  = "#212121"
DIM_C = "#546E7A"   # color dimensiones

# ── Figura ───────────────────────────────────────────────────────────────────
FW, FH = 20, 7.2
fig = plt.figure(figsize=(FW, FH), dpi=180, facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

# ── Helpers ──────────────────────────────────────────────────────────────────

def rbox(x, y, w, h, fc, lines, fs=8.5, dim=None, r=0.18):
    """Caja redondeada. lines = lista de strings."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=1.4, edgecolor=WHITE,
        facecolor=fc, zorder=3))

    n   = len(lines)
    cy  = y + h * (0.60 if dim else 0.50)
    stp = fs * 0.018
    for i, ln in enumerate(lines):
        ax.text(x + w/2, cy + (i - (n-1)/2)*stp, ln,
                ha="center", va="center", fontsize=fs,
                color=WHITE, fontweight="bold", zorder=4)
    if dim:
        ax.text(x + w/2, y + h*0.17, dim,
                ha="center", va="center",
                fontsize=6.6, color=WHITE, alpha=0.85, zorder=4)


def grp_bg(x, y, w, h, title, fc="#EBF5FB", ec="#90CAF9", fs=8.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.22",
        linewidth=1.1, edgecolor=ec,
        facecolor=fc, alpha=0.55, zorder=1))
    ax.text(x + w/2, y + h + 0.03, title,
            ha="center", va="bottom",
            fontsize=fs, color=ec, fontweight="bold", zorder=2)


def harrow(x1, x2, y, lbl="", lw=2.0, col="#455A64"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=13), zorder=5)
    if lbl:
        ax.text((x1+x2)/2, y + 0.18, lbl,
                ha="center", va="bottom",
                fontsize=6.6, color=col, fontstyle="italic", zorder=6)


def miniarr(x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#B0BEC5",
                                lw=0.9, mutation_scale=9), zorder=5)


# ── Dimensiones del layout ────────────────────────────────────────────────────
BY  = 1.20          # y base de las cajas
BH  = 4.50          # alto de cada caja
CY  = BY + BH/2     # centro vertical

BW  = 0.78          # ancho de una caja
GAP = 0.13          # gap interno entre cajas
ARW = 0.32          # espacio de flecha entre grupos

# Posición x de inicio de cada grupo
P = {}
P["inp"]  = 0.28
P["b1"]   = P["inp"] + 0.95 + ARW
P["b2"]   = P["b1"]  + BW*3 + GAP*2 + ARW + 0.42
P["b3"]   = P["b2"]  + BW*3 + GAP*2 + ARW + 0.42
P["pool"] = P["b3"]  + BW*3 + GAP*2 + ARW + 0.38
P["cls"]  = P["pool"] + BW*2 + GAP   + ARW + 0.30
P["out"]  = P["cls"]  + BW*3 + GAP*2 + ARW + 0.25

# ══════════════════════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════════════════════
ax.text(FW/2, FH - 0.28,
        "ECG_CNN — Arquitectura de Red Neuronal Convolucional 1D  ·  44,229 parámetros  ·  5 clases AAMI EC57",
        ha="center", va="top",
        fontsize=11.5, fontweight="bold", color=DARK)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRADA
# ══════════════════════════════════════════════════════════════════════════════
grp_bg(P["inp"]-0.10, BY-0.12, 0.92, BH+0.24,
       "ENTRADA", fc="#E3F2FD", ec="#1565C0", fs=8)

rbox(P["inp"], BY+0.38, 0.72, BH-0.76,
     COL["input"], ["Latido", "ECG", "z-score"],
     fs=8.8, dim="1 × 71")

ax.text(P["inp"]+0.36, BY+0.18,
        "71 muestras", ha="center", va="center",
        fontsize=6.4, color=DIM_C)

harrow(P["inp"]+0.72, P["b1"]-0.04, CY, "1×71",
       col=COL["conv"])

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 1
# ══════════════════════════════════════════════════════════════════════════════
bx = P["b1"]
GW1 = BW*3 + GAP*2
grp_bg(bx-0.10, BY-0.12, GW1+0.20, BH+0.24,
       "BLOQUE  1", fc="#E3F2FD", ec="#1565C0", fs=8)

rbox(bx,             BY+0.28, BW, BH-0.56, COL["conv"],
     ["Conv1d", "1→32", "k=5"], fs=8.5, dim="32×71")
rbox(bx+BW+GAP,      BY+0.28, BW, BH-0.56, COL["bn"],
     ["BN", "+", "ReLU"], fs=8.5, dim="32×71")
rbox(bx+BW*2+GAP*2,  BY+0.28, BW, BH-0.56, COL["pool"],
     ["Max", "Pool", "÷2"], fs=8.5, dim="32×35")

miniarr(bx+BW, bx+BW+GAP, CY)
miniarr(bx+BW*2+GAP, bx+BW*2+GAP*2, CY)

harrow(bx+GW1, P["b2"]-0.04, CY, "32×35", col=COL["conv"])

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 2
# ══════════════════════════════════════════════════════════════════════════════
bx = P["b2"]
GW2 = BW*3 + GAP*2
grp_bg(bx-0.10, BY-0.12, GW2+0.20, BH+0.24,
       "BLOQUE  2", fc="#E8EAF6", ec="#283593", fs=8)

rbox(bx,             BY+0.28, BW, BH-0.56, "#0D47A1",
     ["Conv1d", "32→64", "k=5"], fs=8.5, dim="64×35")
rbox(bx+BW+GAP,      BY+0.28, BW, BH-0.56, "#1565C0",
     ["BN", "+", "ReLU"], fs=8.5, dim="64×35")
rbox(bx+BW*2+GAP*2,  BY+0.28, BW, BH-0.56, "#4A148C",
     ["Max", "Pool", "÷2"], fs=8.5, dim="64×17")

miniarr(bx+BW, bx+BW+GAP, CY)
miniarr(bx+BW*2+GAP, bx+BW*2+GAP*2, CY)

harrow(bx+GW2, P["b3"]-0.04, CY, "64×17", col="#283593")

# ══════════════════════════════════════════════════════════════════════════════
# BLOQUE 3
# ══════════════════════════════════════════════════════════════════════════════
bx = P["b3"]
GW3 = BW*3 + GAP*2
grp_bg(bx-0.10, BY-0.12, GW3+0.20, BH+0.24,
       "BLOQUE  3", fc="#EDE7F6", ec="#311B92", fs=8)

rbox(bx,             BY+0.28, BW, BH-0.56, "#1A237E",
     ["Conv1d", "64→128", "k=3"], fs=8.5, dim="128×17")
rbox(bx+BW+GAP,      BY+0.28, BW, BH-0.56, "#283593",
     ["BN", "+", "ReLU"], fs=8.5, dim="128×17")
rbox(bx+BW*2+GAP*2,  BY+0.28, BW, BH-0.56, "#4527A0",
     ["Max", "Pool", "÷2"], fs=8.5, dim="128×8")

miniarr(bx+BW, bx+BW+GAP, CY)
miniarr(bx+BW*2+GAP, bx+BW*2+GAP*2, CY)

harrow(bx+GW3, P["pool"]-0.04, CY, "128×8", col="#311B92")

# ══════════════════════════════════════════════════════════════════════════════
# POOLING GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
bx = P["pool"]
GWP = BW*2 + GAP
grp_bg(bx-0.10, BY-0.12, GWP+0.20, BH+0.24,
       "POOL GLOBAL", fc="#F3E5F5", ec="#4527A0", fs=8)

rbox(bx,          BY+0.28, BW, BH-0.56, COL["avg"],
     ["Avg", "Pool", "(1)"], fs=8.5, dim="128×1")
rbox(bx+BW+GAP,   BY+0.28, BW, BH-0.56, COL["flat"],
     ["Flat", "ten", ""], fs=8.5, dim="128")

miniarr(bx+BW, bx+BW+GAP, CY)

harrow(bx+GWP, P["cls"]-0.04, CY, "128", col=COL["avg"])

# ══════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR
# ══════════════════════════════════════════════════════════════════════════════
bx = P["cls"]
GWC = BW*3 + GAP*2
grp_bg(bx-0.10, BY-0.12, GWC+0.20, BH+0.24,
       "CLASIFICADOR  (self.classifier)", fc="#FFF3E0", ec="#E65100", fs=8)

rbox(bx,             BY+0.28, BW, BH-0.56, COL["fc"],
     ["Linear", "128→64", ""], fs=8.5, dim="64")
rbox(bx+BW+GAP,      BY+0.28, BW, BH-0.56, COL["drop"],
     ["ReLU", "Drop", "p=0.5"], fs=8.5, dim="64")
rbox(bx+BW*2+GAP*2,  BY+0.28, BW, BH-0.56, "#BF360C",
     ["Linear", "64→5", "logits"], fs=8.5, dim="5")

miniarr(bx+BW, bx+BW+GAP, CY)
miniarr(bx+BW*2+GAP, bx+BW*2+GAP*2, CY)

harrow(bx+GWC, P["out"]-0.04, CY, "5 logits", col=COL["fc"])

# ══════════════════════════════════════════════════════════════════════════════
# SALIDA — 5 clases
# ══════════════════════════════════════════════════════════════════════════════
bx = P["out"]
OW = 1.55
grp_bg(bx-0.08, BY-0.12, OW+0.16, BH+0.24,
       "SALIDA  AAMI EC57", fc="#E0F2F1", ec="#00695C", fs=8)

ax.text(bx + OW/2, BY + BH - 0.08,
        "Softmax", ha="center", va="top",
        fontsize=7.8, color="#00695C", fontweight="bold")

classes = [
    ("N", "Normal",           COL["out_N"]),
    ("S", "Supraventricular", COL["out_S"]),
    ("V", "Ventricular",      COL["out_V"]),
    ("F", "Fusión",           COL["out_F"]),
    ("Q", "Desconocido",      COL["out_Q"]),
]
ch   = (BH - 1.05) / 5
cw   = OW - 0.22

for i, (code, name, col) in enumerate(classes):
    cy_i = BY + 0.38 + (4 - i) * ch
    ax.add_patch(FancyBboxPatch(
        (bx+0.09, cy_i+0.04), cw, ch-0.09,
        boxstyle="round,pad=0,rounding_size=0.10",
        linewidth=0.8, edgecolor=WHITE,
        facecolor=col, alpha=0.93, zorder=3))
    # código en negrita
    ax.text(bx + 0.09 + cw*0.24, cy_i + 0.04 + (ch-0.09)/2,
            code, ha="center", va="center",
            fontsize=10.5, color=WHITE, fontweight="bold", zorder=4)
    # nombre
    ax.text(bx + 0.09 + cw*0.65, cy_i + 0.04 + (ch-0.09)/2,
            name, ha="center", va="center",
            fontsize=7.2, color=WHITE, zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# LEYENDA compacta (fila inferior)
# ══════════════════════════════════════════════════════════════════════════════
LEG = [
    (COL["input"], "Entrada"),
    (COL["conv"],  "Conv1d"),
    (COL["bn"],    "BN + ReLU"),
    (COL["pool"],  "MaxPool"),
    (COL["avg"],   "AvgPool"),
    (COL["fc"],    "Fully Connected"),
    (COL["drop"],  "ReLU + Dropout"),
    ("#00695C",    "Softmax / Salida"),
]
sq   = 0.25
step = FW / len(LEG)
LY   = 0.34

for i, (col, lbl) in enumerate(LEG):
    lx = i * step + (step - sq)/2 - 0.55
    ax.add_patch(FancyBboxPatch(
        (lx, LY), sq, sq,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=0, facecolor=col, zorder=4))
    ax.text(lx + sq + 0.10, LY + sq/2, lbl,
            ha="left", va="center",
            fontsize=7.4, color=DARK, zorder=5)

# ── línea separadora leyenda ──────────────────────────────────────────────────
ax.add_line(Line2D([0.5, FW-0.5], [LY+sq+0.12, LY+sq+0.12],
                   color=GRAY_BD, lw=0.8, zorder=2))

# ── pie ───────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.006,
         "Residencia Profesional  ·  Ingeniería Biomédica  ·  Instituto Tecnológico de Tijuana  ·  "
         "Elias Bejarano Lozada  #20213057  ·  2026",
         ha="center", va="bottom",
         fontsize=6.8, color="#90A4AE", fontstyle="italic")

# ── guardar ───────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none", format="png")
plt.close(fig)
print(f"Diagrama compacto generado: {OUT}")
