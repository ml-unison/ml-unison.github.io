"""Láminas de la Unidad 6, tema 3: redes convolucionales (de LeNet a EfficientNet).

Para que las cifras de las láminas sean exactamente las de la libreta, este módulo
**ejecuta la libreta** `u06-e03-convolucionales.py` (el mismo código, las mismas
semillas, sin el modo rápido) y toma de ella los resultados. Tarda lo mismo que la
libreta: un par de minutos en CPU. Si cambias nombres de variables en la libreta,
actualiza este módulo.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np

from _comun import r

LIBRETA = Path(__file__).resolve().parents[2] / "unidades/06-redes-neuronales/ejemplos/u06-e03-convolucionales.py"


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

    # --- la parte convolucional como preprocesamiento: mismos clasificadores, distintos atributos
    out["extractor"] = {
        "filas": [{"nombre": n, "exactitud": round(float(e), 4), "segundos": round(float(t), 1)}
                  for n, (e, t) in ns["extractor"].items()],
        "n_logistica": ns["N_LOG"], "n_atributos": int(ns["A_pru"].shape[1]),
    }

    # --- VGG: dos 3×3 contra una 5×5; VGG-16 completa
    vgg = ns["vgg"]
    out["vgg"] = {
        "C": ns["C_VGG"], "dos3x3": ns["vgg_cuenta"]["dos 3×3"], "una5x5": ns["vgg_cuenta"]["una 5×5"],
        "campo": {"dos3x3": ns["campo_receptivo"]([(3, 1), (3, 1)])[-1],
                  "tres3x3": ns["campo_receptivo"]([(3, 1)] * 3)[-1]},
        "params": vgg["parámetros"], "params_densas": vgg["en las capas densas"],
        "params_primera_densa": vgg["primera densa (25,088 → 4,096)"],
        "frac_densas": round(vgg["fracción en las densas"], 4),
        "mult": ns["mult_vgg"], "frac_mult_conv": round(vgg["fracción de las multiplicaciones en las convoluciones"], 4),
    }

    # --- ResNet: bloque básico contra bottleneck (256 canales, mapa de 56×56)
    br = ns["bloques_resnet"]
    out["bottleneck"] = {
        "C": 256, "C_m": 64,
        "basico": {"pesos": int(br.iloc[0, 0]), "mult": int(br.iloc[0, 1])},
        "cuello": {"pesos": int(br.iloc[1, 0]), "mult": int(br.iloc[1, 1])},
    }

    # --- Inception 3a: con y sin reducción 1×1
    ic = ns["inception_cuenta"]
    out["inception"] = {
        "entrada": 192, "salidas": [64, 128, 32, 32], "reducciones": [96, 16], "lado": 28,
        "sin": {"params": int(ic.iloc[0, 0]), "mult": int(ic.iloc[0, 1])},
        "con": {"params": int(ic.iloc[1, 0]), "mult": int(ic.iloc[1, 1])},
    }

    # --- convolución separable en profundidad
    out["separable"] = {
        "tabla": [{"C": int(f["C"]), "C2": int(f["C'"]), "k": int(f["k"]), "normal": int(f["normal: C C' k²"]),
                   "separable": int(f["separable: C k² + C C'"]), "razon": round(float(f["razón"]), 4),
                   "aprox": round(float(f["1/C' + 1/k²"]), 4)} for _, f in ns["separable_cuenta"].iterrows()],
        "mult": {k: int(v) for k, v in ns["mult_sep"].items()}, "lado": 28,
    }

    # --- EfficientNet: escalamiento compuesto
    esc = ns["escalamiento"]
    out["efficientnet"] = {
        "alfa": ns["ALFA"], "beta": ns["BETA"], "gamma": ns["GAMMA"],
        "producto": round(ns["ALFA"] * ns["BETA"] ** 2 * ns["GAMMA"] ** 2, 4),
        "phi": [{"phi": int(p), "profundidad": round(float(f.iloc[0]), 3), "ancho": round(float(f.iloc[1]), 3),
                 "resolucion": int(f.iloc[2]), "costo": round(float(f.iloc[3]), 3)} for p, f in esc.iterrows()],
    }

    # --- tabla de arquitecturas: parámetros (torchvision, sin pesos) y exactitud top-1 en ImageNet
    out["arquitecturas"] = [
        {"nombre": n, "anio": int(f["año"]), "params_m": round(float(f["parámetros (millones)"]), 2),
         "top1": round(float(f["exactitud top-1 (%)"]), 2), "fuente": f["fuente de la exactitud"]}
        for n, f in ns["arquitecturas"].iterrows()
    ]
    return out
