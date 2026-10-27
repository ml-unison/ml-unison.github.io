"""Paleta y estilo de matplotlib del curso, alineados con _brand.yml.

Colores: Manual de Identidad Institucional UNISON 2025–2030 (p. 8–9).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

_FUENTES = Path(__file__).with_name("fuentes")

#: Paleta institucional (nombre → HEX). Debe coincidir con _brand.yml.
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

#: Orden de colores para series categóricas (contraste alto entre vecinos).
CICLO: list[str] = [AZUL, DORADO, ROJO, VERDE, PALETA["azul-medio"], PALETA["cafe"], PALETA["ambar"], MARINO]

#: Mapa secuencial blanco → azul institucional, y divergente rojo ↔ azul.
SECUENCIAL = mpl.colors.LinearSegmentedColormap.from_list("rdp_azul", ["#FFFFFF", PALETA["azul-tenue"], AZUL, MARINO])
DIVERGENTE = mpl.colors.LinearSegmentedColormap.from_list("rdp_div", [ROJO, "#F7F7F7", AZUL])
#: Dos clases (regiones de decisión): rojo claro / azul claro
CLASES2 = mpl.colors.ListedColormap(["#F4C7C3", "#C9D3EC"])

_RC = {
    "axes.prop_cycle": cycler(color=CICLO),
    "axes.edgecolor": "#5B6475",
    "axes.labelcolor": "#1E2433",
    "axes.titlecolor": MARINO,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E3E7EE",
    "grid.linewidth": 0.8,
    "xtick.color": "#5B6475",
    "ytick.color": "#5B6475",
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
    "font.sans-serif": ["Lato", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "svg.fonttype": "path",
}


def _registrar_fuentes() -> None:
    """Registra Lato (incluida en el paquete, licencia OFL) en matplotlib."""
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
