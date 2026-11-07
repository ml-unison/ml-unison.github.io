"""Láminas de la Unidad 6, tema 4 (segunda parte): detección de objetos y YOLO.

Lo importa `u06_04.py` y guarda el resultado de `datos()` bajo la clave "deteccion" de
`presentaciones/datos/u06-04.json`.

Para que las cifras de las láminas sean exactamente las de la libreta, este módulo
**ejecuta la libreta** `u06-e05-deteccion-yolo.py` (sin el modo rápido) y toma de ella
los resultados. Tarda lo mismo que la libreta (uno o dos minutos en CPU, más la descarga
de los modelos la primera vez). Si cambias nombres de variables en la libreta, actualiza
este módulo.

Coordenadas: todas las cajas están en pixeles de la foto **reducida** que se exporta
(`ancho` × `alto`), con el origen en la esquina superior izquierda y formato (x1, y1, x2, y2).
"""

from __future__ import annotations

import base64
import io
import os
import runpy
from pathlib import Path

import numpy as np

LIBRETA = Path(__file__).resolve().parents[2] / "unidades/06-redes-neuronales/ejemplos/u06-e05-deteccion-yolo.py"
ANCHO = 800              # ancho de las fotos de las láminas, en pixeles
CALIDAD_JPEG = 62
MAX_CRUDAS = 100         # tope de cajas antes de la NMS que se exportan (las de mayor confianza)


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


def _jpeg(im) -> str:
    """Imagen de PIL → JPEG en base64 (data URI)."""
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=CALIDAD_JPEG, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _cajas(cajas, factor: float) -> list:
    return np.round(np.asarray(cajas, dtype=float) * factor, 1).tolist()


def _detecciones(cajas, clases, confs, factor: float, es: dict) -> list:
    return [{"caja": c, "clase": es.get(k, k), "clase_en": k, "conf": round(float(p), 3)}
            for c, k, p in zip(_cajas(cajas, factor), clases, confs)]


def datos() -> dict:
    ns = _ejecutar_libreta()
    from PIL import Image

    es, nombres = ns["ES"], ns["NOMBRES"]
    out: dict = {}

    # --- fotos reducidas, con su crédito y las detecciones de YOLO26n (confianza ≥ 0.25)
    factores = {}
    out["fotos"] = []
    for nombre, im in ns["fotos"].items():
        f = ns["FOTOS"][nombre]
        factor = ANCHO / im.width
        factores[nombre] = factor
        chica = im.resize((ANCHO, round(im.height * factor)), Image.LANCZOS)
        d = ns["det_yolo"][nombre]
        out["fotos"].append({
            "nombre": nombre, "titulo": f["titulo"], "autor": f["autor"], "licencia": f["licencia"], "url": f["url"],
            "ancho": chica.width, "alto": chica.height, "imagen": _jpeg(chica),
            "ms_yolo": round(ns["tiempo_yolo"][nombre] * 1000),
            "detecciones": _detecciones(d["cajas"], d["clases"], d["conf"], factor, es),
        })

    # --- supresión de no máximos: cajas crudas (las MAX_CRUDAS de mayor confianza) y cuáles sobreviven.
    # Si una caja sobrevive o no depende solo de las cajas con mayor confianza que ella, así que
    # quedarse con las primeras MAX_CRUDAS no cambia la decisión sobre ninguna de ellas.
    foto_nms = ns["FOTO_NMS"]
    conf = ns["conf_crudas"].numpy()
    orden = np.argsort(-conf)[:MAX_CRUDAS]
    clases = [nombres[int(c)] for c in ns["clases_crudas"]]
    quedan = set(int(i) for i in ns["quedan"])
    pos = {int(i): k for k, i in enumerate(orden)}          # índice original → posición en la lista exportada
    out["nms"] = {
        "foto": foto_nms, "conf_minima": ns["CONF_CRUDAS"], "umbral_iou": ns["UMBRAL_NMS"],
        "total_crudas": len(conf), "total_despues": len(quedan),
        "sin_nms_uno_a_uno": len(ns["r_sin_nms"].boxes),
        # cada caja cruda: si sobrevive a la NMS y `grupo`, la posición (en esta misma lista) de la
        # caja conservada de su clase con la que tiene mayor IoU (la que la eliminó, si fue eliminada)
        "crudas": [dict(d, conservada=int(i) in quedan, grupo=pos.get(int(ns["quedan"][int(ns["grupo_crudas"][i])]), -1))
                   for d, i in zip(_detecciones(ns["cajas_crudas"].numpy()[orden], [clases[i] for i in orden],
                                                conf[orden], factores[foto_nms], es), orden)],
    }

    # --- IoU: la caja verdadera y cuatro predichas, en las unidades del ejemplo de la libreta
    verdadera = ns["caja_verdadera"]
    out["iou"] = {
        "verdadera": verdadera[0].tolist(),
        "ejemplos": [{"nombre": k, "caja": c[0].tolist(), "iou": round(ns["iou_matriz"](verdadera, c).item(), 4)}
                     for k, c in ns["ejemplos_iou"].items()],
    }

    # --- tiempos y detecciones: YOLO26n contra Faster R-CNN (CPU de la máquina donde se corrió)
    res = ns["resumen"]
    out["comparacion"] = [
        {"modelo": m, "parametros": int(fila["parámetros"]), "segundos_por_imagen": round(float(fila["segundos_por_imagen"]), 3),
         "detecciones_conf_05": int(fila["detecciones"]), "map_coco": float(fila["mAP50-95 en COCO"]),
         "veces_yolo": round(float(fila["veces más lento que YOLO"]), 1)}
        for m, fila in res.iterrows()
    ]
    out["comparacion_por_foto"] = [
        {"modelo": f["modelo"], "foto": f["foto"], "segundos": round(float(f["segundos"]), 3),
         "detecciones_conf_05": int(f["detecciones > 0.5"])}
        for f in ns["comparacion"].to_dict("records")
    ]
    out["umbral_conf_comparacion"] = ns["UMBRAL_CONF"]

    # --- ventanas deslizantes: foto de 640 × 480, ventana de 64 × 128, paso 8, escalas × 1.05
    v = ns["ventanas"]
    out["ventanas"] = {
        "imagen": [640, 480], "ventana": [64, 128], "paso": 8, "factor": 1.05,
        "niveles": len(v), "total": ns["total_ventanas"], "en_escala_original": int(v["ventanas"].iloc[0]),
        "por_nivel": v["ventanas"].astype(int).tolist(),
        "mult_hog_millones": round(ns["total_ventanas"] * 3780 / 1e6),
        "mult_alexnet_billones": round(ns["total_ventanas"] * 720e6 / 1e12),
    }

    # --- YOLO: la rejilla de 2016 y la salida de YOLO26n
    out["yolo"] = {
        "S": ns["S"], "B": 2, "C": 20, "numeros_v1": ns["S"] ** 2 * (2 * 5 + 20),
        "salida_yolo26": list(ns["salida"].shape), "posiciones_yolo26": 80 * 80 + 40 * 40 + 20 * 20,
        "parametros_yolo26n": ns["params_yolo"],
    }
    return out


if __name__ == "__main__":
    import json

    d = datos()
    texto = json.dumps(d, ensure_ascii=False)
    print(f"{len(texto) / 1024:.0f} KB en JSON; fotos: {sum(len(f['imagen']) for f in d['fotos']) / 1024:.0f} KB")
