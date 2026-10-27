"""Utilidades comunes de los módulos de datos de las láminas."""

from __future__ import annotations

import numpy as np


def r(a, d: int = 4):
    """Redondea a `d` decimales y convierte a listas de Python (JSON compacto)."""
    return np.round(np.asarray(a, dtype=float), d).tolist()
