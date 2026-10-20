"""Estilo común de las figuras (paleta validada de la skill dataviz)."""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categórica en orden fijo: el color sigue a la entidad, nunca al rango
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR_CORPUS = {"fce": AZUL, "wi": NARANJA, "locness": AQUA}
NOMBRE_CORPUS = {"fce": "FCE", "wi": "W&I", "locness": "LOCNESS"}
# Secuencial de un solo azul, de claro a oscuro
SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
TINTA_SECUNDARIA, LINEA_BASE = "#52514e", "#c3c2b7"


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "text.color": "#0b0b0b", "axes.labelcolor": TINTA_SECUNDARIA, "axes.edgecolor": LINEA_BASE,
        "xtick.color": "#898781", "ytick.color": "#898781", "grid.color": "#e1e0d9",
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlesize": 11, "font.size": 9.5,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
