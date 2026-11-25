"""Figura 15b: F0.5, precisión y recall de BART-large por época en dev_seleccion.

Los valores salen de modelos/experimentos/large-3e-5/checkpoint-10548/trainer_state.json
(log_history, entradas eval_*). Uso, desde proyecto-final/: python scripts/figura_epocas_large.py
"""
import sys; sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graficas import aplicar_estilo, AZUL, NARANJA, AQUA, TINTA_SECUNDARIA
aplicar_estilo()
ep = [0,1,2,3,4,5,6]
f05 = [0.0142,0.5002,0.5163,0.5314,0.5287,0.5222,0.5215]
p = [0.0775,0.5888,0.5910,0.5883,0.5640,0.5565,0.5631]
r = [0.0033,0.3124,0.3430,0.3832,0.4229,0.4189,0.4025]
base_ep = [0,1,2,3,4,5]; base_f = [0.0074,0.4538,0.5029,0.4988,0.4951,0.4860]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(ep, p, color=NARANJA, marker="o", ms=4, label="precisión")
ax.plot(ep, r, color=AQUA, marker="o", ms=4, label="recall")
ax.plot(ep, f05, color=AZUL, marker="o", ms=5, lw=2.2, label="F0.5 (BART-large)")
ax.plot(base_ep, base_f, color=AZUL, ls=":", lw=1.4, alpha=.6, label="F0.5 de BART-base, referencia")
ax.axhline(0.855, color=TINTA_SECUNDARIA, ls="--", lw=1)
ax.text(6, 0.835, "techo 0,855", ha="right", va="top", color=TINTA_SECUNDARIA, fontsize=8.5)
ax.scatter([3], [0.5314], s=140, facecolors="none", edgecolors=AZUL, lw=1.5, zorder=5)
ax.annotate("época 3: 0,531\n(checkpoint elegido)", (3, 0.5314), (3.3, 0.66), fontsize=8.5,
            color=AZUL, arrowprops=dict(arrowstyle="-", color=AZUL, lw=.8))
ax.set_xlabel("época"); ax.set_ylim(0, 0.9); ax.set_xticks(ep)
ax.set_title("BART-large en dev_seleccion, por época")
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.savefig("figuras/15b_f05_por_epoca_large.png")
