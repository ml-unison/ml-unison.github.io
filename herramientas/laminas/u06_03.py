"""Láminas de la Unidad 6, tema 3: redes convolucionales y transferencia.

Para que las cifras de las láminas sean exactamente las de la libreta, este módulo
**ejecuta la libreta** `u06-e03-convolucionales.py` (el mismo código, las mismas
semillas, sin el modo rápido) y toma de ella los resultados. Tarda lo mismo que la
libreta: un par de minutos en CPU. Si cambias nombres de variables en la libreta,
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

LIBRETA = Path(__file__).resolve().parents[2] / "unidades/06-redes-neuronales/ejemplos/u06-e03-convolucionales.py"


def _png(t) -> str:
    """Tensor (3, H, W) en [0, 1] → PNG en base64 (data URI)."""
    from PIL import Image

    a = (np.clip(t.permute(1, 2, 0).numpy(), 0, 1) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
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

    gris = lambda X: np.round(np.asarray(X) * 255).astype(int).tolist()   # imágenes 0–255 enteras
    out: dict = {}

    # --- convolución: ejemplo numérico, filtros de Sobel, equivariancia
    out["conv"] = {"X": ns["X_chica"].astype(int).tolist(), "K": ns["K_vertical"].astype(int).tolist(),
                   "Y": ns["Y_chica"].astype(int).tolist()}
    prenda, lienzo = ns["prenda"], ns["lienzo"]
    out["sobel"] = {"prenda": gris(prenda), "v": r(ns["mapa_v"], 2), "h": r(ns["mapa_h"], 2),
                    "idx": ns["IDX_PRENDA"]}
    a, b = ns["a"], ns["b"]
    out["equivariancia"] = {
        "a": a, "b": b, "dif": float(np.abs(ns["izq"] - ns["der"]).max()),
        "x": gris(lienzo), "xt": gris(ns["trasladar"](lienzo, a, b)),
        "y": r(ns["correlacion2d"](lienzo, ns["SOBEL_V"]), 2), "yt": r(ns["izq"], 2),
    }

    # --- arquitectura, campo receptivo, invariancia
    out["capas"] = [{"capa": f, "forma": list(s), "params": p} for (f, s), p in zip(ns["filas"], ns["params_capa"])]
    out["campo"] = {"teorico": ns["campo_receptivo"]([(3, 1), (2, 2), (3, 1), (2, 2)]),
                    "filas": [int(ns["filas_ok"].min()), int(ns["filas_ok"].max())],
                    "cols": [int(ns["cols_ok"].min()), int(ns["cols_ok"].max())]}
    out["invariancia"] = [{"etapa": k, "cambio": round(v, 3)} for k, v in ns["invariancia"].items()]

    # --- red convolucional contra perceptrón multicapa
    h_mlp, h_cnn = ns["hist_mlp"], ns["hist_cnn"]
    out["fashion"] = {
        "params": {"mlp": ns["contar"](ns["mlp"]), "cnn": ns["contar"](ns["cnn"])},
        "mlp": r(h_mlp["exactitud de prueba"], 4), "cnn": r(h_cnn["exactitud de prueba"], 4),
        "segundos": {"mlp": round(ns["t_mlp"]), "cnn": round(ns["t_cnn"])},
        "media": round(ns["MEDIA"], 4), "de": round(ns["DE"], 4),
    }
    # mosaico: dos ejemplos de cada clase (los primeros del conjunto de entrenamiento)
    X_ent, y_ent = ns["X_ent"], ns["y_ent"]
    out["mosaico"] = [[gris(X_ent[int(i), 0]) for i in torch.where(y_ent == c)[0][:2]] for c in range(10)]
    out["clases"] = ns["CLASES"]

    # filtros aprendidos y mapas de activación en el botín
    out["filtros"] = r(ns["filtros"].numpy(), 3)
    out["mapas"] = r(ns["mapas"].numpy(), 2)

    # errores
    mc = ns["mc"]
    out["confusion"] = mc.astype(int).tolist()
    X_pru = ns["X_pru"]
    out["camisas"] = [gris(X_pru[int(i), 0]) for i in ns["camisas"][:6]]

    # --- conexiones residuales
    out["residual"] = {k: r(np.log10(v), 3) for k, v in ns["grad"].items()}

    # --- transferencia
    S_ent, ys_ent = ns["S_ent"], ns["ys_ent"]
    out["eurosat"] = {
        "clases": ns["CLASES_SAT"],
        "imagenes": [[_png(S_ent[np.where(ys_ent == c)[0][f]]) for f in range(2)] for c in range(10)],
        "k": list(ns["POR_CLASE"]),
        "transfer": [round(float(ns["exact_transfer"][k]), 3) for k in ns["POR_CLASE"]],
        "cero": [round(float(ns["exact_cero"][k]), 3) for k in ns["POR_CLASE"]],
        "params_resnet": ns["contar"](ns["resnet"]) + 512 * 1000 + 1000,   # se le quitó la capa fc
        "params_cero": ns["contar"](ns["red_satelite"]()),
        "pasos": ns["PASOS"],
        "n_prueba": len(ns["ys_pru"]),
    }
    return out
