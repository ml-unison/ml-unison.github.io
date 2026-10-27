"""Láminas de la Unidad 6, tema 2: optimización y regularización de redes.

Para que las cifras de las láminas sean exactamente las de la libreta, este módulo
**ejecuta la libreta** `u06-e02-optimizacion.py` (el mismo código, las mismas
semillas, sin el modo rápido) y toma de ella los resultados. Tarda lo mismo que la
libreta: unos dos minutos en CPU. Si cambias nombres de variables en la libreta,
actualiza este módulo.
"""

from __future__ import annotations

import base64
import io
import os
import runpy
from pathlib import Path

import numpy as np

from _comun import r

LIBRETA = Path(__file__).resolve().parents[2] / "unidades/06-redes-neuronales/ejemplos/u06-e02-optimizacion.py"


def _png(img) -> str:
    """Imagen 28×28 en [0, 1] → PNG en base64 con tinta azul marino sobre fondo transparente."""
    from PIL import Image

    a = np.zeros((28, 28, 4), dtype=np.uint8)
    a[..., :3] = (0x20, 0x30, 0x5A)
    a[..., 3] = np.clip(np.asarray(img) * 255 * 1.05, 0, 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ejecutar_libreta() -> dict:
    os.environ.pop("RDP_RAPIDO", None)
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mostrar = plt.show
    plt.show = lambda *a, **k: plt.close("all")
    try:
        return runpy.run_path(str(LIBRETA), run_name="__libreta__")
    finally:
        plt.show = mostrar


def datos() -> dict:
    ns = _ejecutar_libreta()
    import torch

    out: dict = {}

    # --- mosaico: seis ejemplos por clase (los mismos primeros de la libreta)
    X_todo, y_todo, i_ent = ns["X_todo"], ns["y_todo"], ns["i_ent"]
    out["clases"] = ns["CLASES"]
    out["mosaico"] = [[_png(X_todo[i_ent][i].reshape(28, 28).numpy())
                       for i in torch.nonzero(y_todo[i_ent] == k).flatten()[:6]] for k in range(10)]

    # --- ruido del gradiente del minilote
    ruido = ns["ruido"]
    out["ruido"] = {"B": ruido["B"].tolist(), "e": r(ruido["E‖g_B − g‖²"], 4), "Be": r(ruido["B · E‖g_B − g‖²"], 1),
                    "g2": round(float(ns["g"].pow(2).sum()), 3), "sigma2": round(float(ns["sigma2"]), 2)}

    # --- barrido de la tasa de aprendizaje (promedio móvil de 10 pasos, como en la libreta)
    import pandas as pd

    barrido = []
    for eta, p in ns["barrido"].items():
        if np.isfinite(p[-1]):
            barrido.append({"eta": eta, "perdida": r(pd.Series(p).rolling(10, min_periods=1).mean(), 3), "diverge": False,
                            "final": round(float(np.mean(p[-20:])), 3)})
        else:
            k = int(np.argmax(~np.isfinite(p) | (p > 10)))
            barrido.append({"eta": eta, "perdida": r(np.minimum(p[:k], 3.5), 3), "diverge": True,
                            "nan": int(np.argmax(~np.isfinite(p))) + 1})
    out["barrido"] = barrido

    # --- el valle con κ = 50: caminos con la misma η y varios β; error relativo
    caminos, w0 = ns["caminos"], ns["w0"]
    it = ns["iteraciones"]
    error = lambda c: np.maximum(np.linalg.norm(c, axis=1) / np.linalg.norm(w0), 1e-12)
    out["valle"] = {
        "kappa": ns["kappa"], "eta": round(float(ns["eta"]), 4),
        "caminos": [{"beta": b, "camino": r(c[:60], 4), "iter": it(c), "error": r(np.log10(error(c)), 3)}
                    for b, c in caminos.items()],
        "optimo": {"eta": round(float(ns["eta_opt"]), 4), "beta": round(float(ns["beta_opt"]), 3),
                   "iter": it(ns["optimo"]), "error": r(np.log10(error(ns["optimo"])), 3)},
    }

    # --- Adam: el factor del paso sin corrección de sesgo
    t, factor = ns["t"], ns["factor"]
    idx = np.unique(np.round(np.logspace(0, np.log10(len(t)), 160)).astype(int)) - 1
    out["adam"] = {"t": t[idx].tolist(), "f": r(factor[idx], 3), "max": round(float(factor.max()), 2),
                   "tmax": int(t[factor.argmax()]), "t1": round(float(factor[0]), 2), "t1000": round(float(factor[999]), 2)}

    # --- calendarios (en épocas)
    pasos = ns["pasos"]
    sel = np.arange(0, len(pasos), 5)
    out["calendarios"] = {"epoca": r(pasos[sel] / 79, 3), **{k: r(v[sel], 4) for k, v in ns["calendarios"].items()}}

    # --- corridas: historias por época y exactitudes de prueba
    corridas = []
    for nombre, c in ns["CORRIDAS"].items():
        h = c["historia"]
        corridas.append({"nombre": nombre, "pe": r(h["pérdida ent."], 4), "pv": r(h["pérdida val."], 4),
                         "ae": r(h["exactitud ent."], 4), "av": r(h["exactitud val."], 4),
                         "final": round(c["prueba final"], 3), "temprana": round(c["prueba temprana"], 3),
                         "epoca": c["época temprana"], "pv_min": round(float(h["pérdida val."].min()), 3),
                         "pv_fin": round(float(h["pérdida val."].iloc[-1]), 3)})
    out["corridas"] = corridas

    # --- dropout como ensamble
    p_det, p_ens, y_val = ns["p_det"], ns["p_ens"], ns["y_val"]
    out["ensamble"] = {"det": round(float((p_det.argmax(1) == y_val).float().mean()), 4),
                       "ens": round(float((p_ens.argmax(1) == y_val).float().mean()), 4),
                       "coinciden": round(float((p_det.argmax(1) == p_ens.argmax(1)).float().mean()), 3)}
    return out
