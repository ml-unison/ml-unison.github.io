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

    # quince puntos nuevos, para mostrar en clase qué pasa fuera de la muestra
    x_nu = np.sort(rng.uniform(0, 1, n))
    y_nu = f(x_nu) + rng.normal(0, ruido, n)

    curvas, ein, eout, enu, pred_tr, pred_nu = [], [], [], [], [], []
    for g in range(15):
        # mínimos cuadrados en base de Legendre (bien condicionada en [-1, 1])
        V = np.polynomial.legendre.legvander(2 * x_tr - 1, g)
        c, *_ = np.linalg.lstsq(V, y_tr, rcond=None)
        pred = lambda x: np.polynomial.legendre.legvander(2 * x - 1, g) @ c
        curvas.append(r(np.clip(pred(malla), -3, 3), 3))
        ein.append(float(np.mean((pred(x_tr) - y_tr) ** 2)))
        eout.append(float(np.mean((pred(x_te) - y_te) ** 2)))
        enu.append(float(np.mean((pred(x_nu) - y_nu) ** 2)))
        pred_tr.append(r(pred(x_tr), 3))
        pred_nu.append(r(pred(x_nu), 3))

    return {"polinomios": {
        "ruido": ruido, "x": r(malla, 4), "f": r(f(malla), 3),
        "xtr": r(x_tr), "ytr": r(y_tr), "curvas": curvas,
        "ein": r(ein, 5), "eout": r(eout, 5),
        "xnu": r(x_nu), "ynu": r(y_nu), "enu": r(enu, 5),
        "pred_tr": pred_tr, "pred_nu": pred_nu,
    }, "digitos": digitos(), "presas": presas()}


def presas() -> dict:
    """Almacenamiento de las presas de Sonora (CONAGUA), resumido para las láminas de pandas."""
    import pandas as pd

    raiz = DESTINO.parents[1] / "datos"
    reg = pd.read_excel(raiz / "almac_reciente.xlsx")
    cat = pd.read_excel(raiz / "catalogo_estatal.xlsx", sheet_name="Catálogo_estatal")
    reg["fecha"] = pd.to_datetime(reg["fecha"], format="%Y/%m/%d")
    df = reg.merge(cat[["Clave", "Nombre_comun", "cap_namo"]].rename(columns={"Clave": "clave", "Nombre_comun": "presa"}),
                   on="clave", validate="many_to_one")
    df["llenado"] = 100 * df["almacenamiento"] / df["cap_namo"]
    ancho = df.pivot(index="fecha", columns="presa", values="llenado")
    todas = pd.date_range(ancho.index.min(), ancho.index.max(), freq="D")
    ancho = ancho.reindex(todas)

    # Cobertura: tramos continuos con datos, por presa
    cobertura = {}
    for presa in ancho.columns:
        hay = ancho[presa].notna().to_numpy()
        tramos, ini = [], None
        for i, v in enumerate(hay):
            if v and ini is None:
                ini = i
            if (not v or i == len(hay) - 1) and ini is not None:
                fin = i if v else i - 1
                tramos.append([todas[ini].strftime("%Y-%m-%d"), todas[fin].strftime("%Y-%m-%d")])
                ini = None
        cobertura[presa] = tramos

    yaqui = ["La Angostura", "El Novillo", "Oviachic"]
    llenas = ancho.drop(columns="Los Pilares").interpolate(limit=10)
    mensual = llenas[yaqui].resample("MS").mean()
    anual = llenas[yaqui].groupby(llenas.index.month).mean()
    muestra = (df.sort_values(["fecha", "clave"]).loc[lambda t: t["fecha"] == pd.Timestamp("2022-09-15")]
               [["clave", "fecha", "almacenamiento", "presa", "cap_namo", "llenado"]])
    return {
        "inicio": todas[0].strftime("%Y-%m-%d"), "fin": todas[-1].strftime("%Y-%m-%d"),
        "cobertura": cobertura,
        "mensual": {"fechas": [d.strftime("%Y-%m-%d") for d in mensual.index],
                    "series": {k: r(mensual[k], 1) for k in yaqui}},
        "anual": {k: r(anual[k], 1) for k in yaqui},
        "promedio": {k: round(float(v), 1) for k, v in df.groupby("presa")["llenado"].mean().sort_values().items()},
        "muestra": [{"clave": f.clave, "fecha": f.fecha.strftime("%Y-%m-%d"), "alm": round(f.almacenamiento, 1),
                     "presa": f.presa, "cap": int(f.cap_namo), "llenado": round(f.llenado, 1)} for f in muestra.itertuples()],
        "n_registros": int(len(reg)),
    }


def digitos() -> dict:
    """Dígitos escritos a mano (sklearn, 8×8, valores 0–16): ejemplos reales para la clase."""
    from sklearn.datasets import load_digits

    d = load_digits()
    rng = np.random.default_rng(7)
    por_clase = {k: rng.permutation(np.flatnonzero(d.target == k)) for k in range(10)}
    mosaico = [[d.images[por_clase[k][j]].astype(int).ravel().tolist() for k in range(10)] for j in range(4)]
    return {
        "mosaico": mosaico,                                                   # 4 filas × 10 clases
        "sietes": [d.images[i].astype(int).ravel().tolist() for i in por_clase[7][4:16]],
        "unos": [d.images[i].astype(int).ravel().tolist() for i in por_clase[1][4:16]],
    }


if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, fn in {"u01": u01}.items():
        ruta = DESTINO / f"{nombre}.json"
        ruta.write_text(json.dumps(fn(), separators=(",", ":")), encoding="utf-8")
        print(f"✓ {ruta.relative_to(DESTINO.parents[1])}")
