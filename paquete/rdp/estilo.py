"""Paleta y estilo de matplotlib del curso, alineados con el sitio y las presentaciones.

Colores: Manual de Identidad Institucional UNISON 2025–2030 (p. 8–9).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

_FUENTES = Path(__file__).with_name("fuentes")

#: Paleta institucional (nombre → HEX). Debe coincidir con las hojas de estilo (tests/test_marca.py).
PALETA: dict[str, str] = {
    "azul": "#24398A",
    "dorado": "#E1B73D",
    "ambar": "#EBA93B",
    "rojo": "#D5433C",
    "verde": "#5CA847",
    "cafe": "#946721",
    "azul-marino": "#20305A",
    "azul-medio": "#496EA5",
    "azul-claro": "#8DA3C9",
    "azul-tenue": "#C0CEE3",
    "gris": "#C5C5C5",
    "gris-claro": "#F1F1F1",
}

AZUL = PALETA["azul"]
DORADO = PALETA["dorado"]
MARINO = PALETA["azul-marino"]
ROJO = PALETA["rojo"]
VERDE = PALETA["verde"]
GRIS = "#6B7385"
#: Dorado más oscuro, legible sobre blanco como línea o texto.
DORADO_OSC = "#C9971C"

# Neutros (los mismos del sitio y las presentaciones)
TINTA = "#1A2133"
TENUE = "#5E6678"
LINEA = "#E2E5EC"

#: Orden fijo de colores para series. Los primeros cuatro están validados para
#: daltonismo (azul, rojo, dorado, gris); el verde no va junto al rojo.
CICLO: list[str] = [AZUL, ROJO, DORADO_OSC, GRIS, PALETA["azul-medio"], PALETA["cafe"], MARINO]

#: Mapa secuencial blanco → azul institucional, y divergente rojo ↔ azul.
SECUENCIAL = mpl.colors.LinearSegmentedColormap.from_list("rdp_azul", ["#FFFFFF", PALETA["azul-tenue"], AZUL, MARINO])
DIVERGENTE = mpl.colors.LinearSegmentedColormap.from_list("rdp_div", [ROJO, "#F7F7F7", AZUL])
#: Dos clases (regiones de decisión): rojo claro / azul claro
CLASES2 = mpl.colors.ListedColormap(["#F4C7C3", "#C9D3EC"])

_RC = {
    "axes.prop_cycle": cycler(color=CICLO),
    "axes.edgecolor": "#8A91A0",
    "axes.titlecolor": MARINO,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12,
    "axes.labelcolor": TENUE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": LINEA,
    "grid.linewidth": 0.8,
    "xtick.color": TENUE,
    "ytick.color": TENUE,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "text.color": TINTA,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "scatter.edgecolors": "white",
    "patch.edgecolor": "white",
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.figsize": (7.0, 4.2),
    "figure.facecolor": "white",
    "image.cmap": "rdp_azul",
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans 3", "DejaVu Sans"],
    "font.size": 12,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "svg.fonttype": "path",
}


def _registrar_fuentes() -> None:
    """Registra Source Sans 3 (incluida en el paquete, licencia OFL) en matplotlib."""
    for ttf in _FUENTES.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))


def usar(contexto: str = "libreta") -> None:
    """Activa el estilo del curso.

    contexto="presentacion" agranda los textos para diapositivas.
    """
    _registrar_fuentes()
    for nombre, cmap in (("rdp_azul", SECUENCIAL), ("rdp_div", DIVERGENTE)):
        if nombre not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=nombre)
    mpl.rcParams.update(_RC)
    if contexto == "presentacion":
        mpl.rcParams.update({
            "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
            "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
            "figure.figsize": (9.0, 5.0), "lines.linewidth": 2.8, "lines.markersize": 8,
        })


def etiquetar(ax, x, valores: dict[str, float], separacion: float | None = None, **kwargs) -> None:
    """Etiqueta series directamente a la derecha de su último punto, sin que se encimen.

    `valores` asocia el nombre de cada serie con la altura de su último punto. Las
    etiquetas se reparten verticalmente para quedar al menos a `separacion`
    (por omisión, 5 % del rango del eje y) unas de otras.
    """
    if separacion is None:
        y0, y1 = ax.get_ylim()
        separacion = 0.055 * (y1 - y0)
    orden = sorted(valores.items(), key=lambda kv: kv[1])
    ys = [v for _, v in orden]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + separacion)
    # si se salieron por arriba, se recorren hacia abajo conservando la separación
    exceso = ys[-1] - max(v for _, v in orden) if ys else 0
    if exceso > 0:
        ys = [y - exceso / 2 for y in ys]
    opciones = {"va": "center", "fontsize": 11, "color": TINTA} | kwargs
    for (nombre, _), y in zip(orden, ys):
        ax.text(x, y, nombre, **opciones)
