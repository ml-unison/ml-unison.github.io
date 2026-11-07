"""Láminas de la Unidad 6, tema 4: transferencia y detección de objetos.

Para que las cifras de las láminas sean exactamente las de la libreta, este módulo
**ejecuta la libreta** `u06-e04-transferencia.py` (el mismo código, las mismas
semillas, sin el modo rápido) y toma de ella los resultados. Tarda lo mismo que la
libreta: de 10 a 20 minutos en CPU (más la parte de detección). Si cambias nombres de variables en la libreta,
actualiza este módulo.

La parte de detección de objetos la calcula `_u06_04_deteccion.py` (si existe) y se
guarda en la clave "deteccion".
"""

from __future__ import annotations

import base64
import io
import os
import runpy
from pathlib import Path

import numpy as np

from _comun import r

LIBRETA = Path(__file__).resolve().parents[2] / "unidades/06-redes-neuronales/ejemplos/u06-e04-transferencia.py"


def _png(t, colores: int | None = None) -> str:
    """Tensor (3, H, W) en [0, 1] → PNG en base64 (data URI). Con `colores`, PNG de paleta (más chico)."""
    from PIL import Image

    a = (np.clip(t.permute(1, 2, 0).numpy(), 0, 1) * 255).round().astype(np.uint8)
    im = Image.fromarray(a)
    if colores:
        im = im.quantize(colors=colores, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
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
    out: dict = {}

    # --- qué aprende cada capa: parches que más activan y la imagen que maximiza cada canal
    capas, campo, CAPAS = [], ns["campo"], ns["CAPAS"]
    for nombre, canales in ns["CANALES"].items():
        rf = campo[CAPAS[nombre]]
        capas.append({
            "nombre": nombre,
            "campo": rf,                                  # campo receptivo teórico (pixeles)
            "lado": min(rf, ns["LADO_MAX"]),              # lado del parche recortado
            "canales": [{"canal": c, "parches": _png(m, 96), "maxima": _png(im, 96)}
                        for c, m, im in zip(canales, ns["mosaicos"][nombre], ns["imagenes_max"][nombre])],
        })
    out["capas"] = capas
    out["visualizacion"] = {
        "n_imagenes": ns["N_IMAGENES"], "lado_foto": ns["LADO"], "top": ns["TOP"],
        "pasos": ns["PASOS_MAX"], "tam": ns["TAM_MAX"], "clases": ns["CLASES_INET"],
    }

    # --- catálogo de modelos preentrenados de torchvision
    cat = ns["catalogo"].reset_index()
    out["catalogo"] = [{"modelo": f["modelo"], "pesos": f["pesos"], "params_M": f["parámetros (M)"],
                        "top1": f["exactitud top-1 en ImageNet (%)"], "gflops": f["GFLOPs por imagen"],
                        "mb": f["archivo (MB)"]} for f in cat.to_dict("records")]

    # --- EuroSAT: tres formas de transferir contra entrenar desde cero
    S_ent, ys_ent = ns["S_ent"], ns["ys_ent"]
    ks = list(ns["POR_CLASE"])
    serie = lambda d: [round(float(d[k]), 3) for k in ks]
    mc = ns["mc_sat"]
    out["eurosat"] = {
        "clases": ns["CLASES_SAT"],
        "imagenes": [[_png(S_ent[np.where(ys_ent == c)[0][f]]) for f in range(2)] for c in range(10)],
        "k": ks,
        "transfer": serie(ns["exact_transfer"]),     # (a) atributos congelados + logística
        "ultima": serie(ns["exact_ultima"]),         # (b) solo la última capa
        "fino": serie(ns["exact_fino"]),             # (c) ajuste fino
        "cero": serie(ns["exact_cero"]),             # (d) red pequeña desde cero
        "params_resnet": ns["contar"](ns["modelo"]) - (512 * 10 + 10) + (512 * 1000 + 1000),   # con su capa original
        "params_fc": 512 * 10 + 10,
        "params_cero": ns["contar"](ns["red_satelite"]()),
        "pasos": ns["PASOS"], "pasos_fc": ns["PASOS_FC"], "pasos_fino": ns["PASOS_FINO"],
        "segundos": {k: round(v) for k, v in ns["tiempos"].items()},
        "n_prueba": len(ns["ys_pru"]),
        "confusion": mc.astype(int).tolist(),        # ajuste fino con el mayor k
        "por_clase": r(mc.diagonal() / mc.sum(1), 3),
        "errores": [{"casos": int(f["casos"]), "verdadera": f["verdadera"], "predicha": f["predicha"]}
                    for f in ns["errores_comunes"].to_dict("records")],
    }

    # --- detección de objetos (otro módulo, si ya existe)
    try:
        from _u06_04_deteccion import datos as datos_deteccion
    except ImportError:
        pass
    else:
        out["deteccion"] = datos_deteccion()
    return out
