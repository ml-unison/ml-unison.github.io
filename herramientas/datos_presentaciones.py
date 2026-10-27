"""Calcula con Python los datos de las figuras de las presentaciones.

Las presentaciones dibujan las gráficas en el navegador (con la tipografía y la
escala de la lámina), pero los números salen de aquí, con el mismo NumPy que las
libretas. Cada unidad escribe `presentaciones/datos/uNN.json`.

    uv run python herramientas/datos_presentaciones.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DESTINO = Path(__file__).resolve().parents[1] / "presentaciones" / "datos"


def r(a, d=4):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def u01() -> dict:
    rng = np.random.default_rng(1942)
    f = lambda x: np.sin(2 * np.pi * x)
    n, ruido = 15, 0.25
    x_tr = np.sort(rng.uniform(0, 1, n))
    y_tr = f(x_tr) + rng.normal(0, ruido, n)
    x_te = rng.uniform(0, 1, 5000)
    y_te = f(x_te) + rng.normal(0, ruido, 5000)
    malla = np.linspace(0, 1, 241)

    curvas, ein, eout = [], [], []
    for g in range(15):
        # mínimos cuadrados en base de Legendre (bien condicionada en [-1, 1])
        V = np.polynomial.legendre.legvander(2 * x_tr - 1, g)
        c, *_ = np.linalg.lstsq(V, y_tr, rcond=None)
        pred = lambda x: np.polynomial.legendre.legvander(2 * x - 1, g) @ c
        curvas.append(r(np.clip(pred(malla), -3, 3), 3))
        ein.append(float(np.mean((pred(x_tr) - y_tr) ** 2)))
        eout.append(float(np.mean((pred(x_te) - y_te) ** 2)))

    return {"polinomios": {
        "ruido": ruido, "x": r(malla, 4), "f": r(f(malla), 3),
        "xtr": r(x_tr), "ytr": r(y_tr), "curvas": curvas,
        "ein": r(ein, 5), "eout": r(eout, 5),
    }}


if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, fn in {"u01": u01}.items():
        ruta = DESTINO / f"{nombre}.json"
        ruta.write_text(json.dumps(fn(), separators=(",", ":")), encoding="utf-8")
        print(f"✓ {ruta.relative_to(DESTINO.parents[1])}")
