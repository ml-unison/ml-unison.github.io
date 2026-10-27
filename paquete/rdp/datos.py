"""Acceso a los conjuntos de datos del curso (carpeta datos/ del repositorio)."""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import pandas as pd

REPO = "ml-unison/ml-unison.github.io"
#: Rama de la que se descargan los datos. Se puede cambiar con la variable RDP_RAMA.
RAMA = os.environ.get("RDP_RAMA", "main")

_LECTORES = {
    ".csv": pd.read_csv,
    ".txt": lambda f, **kw: pd.read_csv(f, **({"sep": ","} | kw)),
    ".xlsx": pd.read_excel,
    ".parquet": pd.read_parquet,
}


def url(archivo: str, rama: str | None = None) -> str:
    """URL 'raw' de un archivo de la carpeta datos/."""
    return f"https://raw.githubusercontent.com/{REPO}/{rama or RAMA}/datos/{archivo}"


def _local(archivo: str) -> Path | None:
    for base in (Path.cwd(), *Path.cwd().parents):
        p = base / "datos" / archivo
        if p.exists():
            return p
    return None


def cargar(nombre: str, rama: str | None = None, **kwargs) -> pd.DataFrame:
    """Carga datos/<nombre> (con o sin extensión) como DataFrame.

    Si la libreta corre dentro de una copia del repositorio usa el archivo
    local; si no (Colab), lo descarga de GitHub.
    """
    candidatos = [nombre] if Path(nombre).suffix else [nombre + ext for ext in _LECTORES]
    for archivo in candidatos:
        lector = _LECTORES[Path(archivo).suffix]
        local = _local(archivo)
        if local is not None:
            return lector(local, **kwargs)
        try:
            with urllib.request.urlopen(url(archivo, rama), timeout=30) as r:
                return lector(io.BytesIO(r.read()), **kwargs)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    raise FileNotFoundError(f"No encuentro '{nombre}' en datos/ (rama {rama or RAMA}).")
