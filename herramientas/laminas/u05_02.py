"""Datos de las láminas de u05-02-bosques («Bagging y bosques aleatorios»).

Para que las cifras de las láminas coincidan exactamente con las de la libreta
`u05-e02-bosques`, este módulo ejecuta el código de la libreta (su fuente .py, sin
gráficas) y toma de ahí los resultados. Tarda alrededor de un minuto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def r(a, d: int = 4):
    """Redondea a `d` decimales y convierte a listas de Python (JSON compacto)."""
    return np.round(np.asarray(a, dtype=float), d).tolist()

LIBRETA = Path(__file__).resolve().parents[2] / "unidades/05-arboles-ensambles/ejemplos/u05-e02-bosques.py"


def _ejecuta_libreta() -> dict:
    import contextlib
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns: dict = {"__name__": "__libreta__"}
    codigo = compile(LIBRETA.read_text(encoding="utf-8"), str(LIBRETA), "exec")
    mostrar = plt.show
    plt.show = lambda *a, **k: plt.close("all")
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # la libreta imprime sus resultados
            exec(codigo, ns)
    finally:
        plt.show = mostrar
        plt.close("all")
    return ns


def datos() -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score

    v = _ejecuta_libreta()
    y_te, y_tr = v["y_te"], v["y_tr"]
    out: dict = {}

    # Un árbol solo y dos árboles en remuestras
    arbol, A_tr = v["arbol"], v["A_tr"]
    out["arbol"] = {
        "profundidad": int(arbol.get_depth()), "hojas": int(arbol.get_n_leaves()),
        "exactitud_tr": round(float(accuracy_score(y_tr, arbol.predict(A_tr))), 3),
        "exactitud_te": round(float(accuracy_score(y_te, v["p_arbol"] >= 0.5)), 3),
        "auc_te": round(float(roc_auc_score(y_te, v["p_arbol"])), 3),
        "discrepan": round(float(v["discrepan"]), 3),
        "dos": [{"profundidad": int(t.get_depth()), "hojas": int(t.get_n_leaves()),
                 "exactitud": round(float(accuracy_score(y_te, t.predict(v["A_te"]))), 3)} for t in v["dos"]],
    }

    # Varianza del promedio: simulación (la fórmula se dibuja en el navegador)
    out["promedio"] = {"B": v["Bs"].tolist(), "rhos": v["rhos"],
                       "simulada": [r(v["simulada"][rho], 4) for rho in v["rhos"]]}

    # Curvas de error contra número de árboles (prueba y fuera de la bolsa)
    out["curvas"] = {k: r(v[k], 4) for k in ("error_bag", "oob_error_bag", "error_rf", "oob_error_rf")}
    out["curvas"]["error_arbol"] = round(1 - float(accuracy_score(y_te, v["p_arbol"] >= 0.5)), 4)

    # Fuera de la bolsa
    fuera, N = v["fuera"], v["N"]
    out["oob"] = {"N": int(N), "teorica": round((1 - 1 / N) ** N, 4), "media": round(float(fuera.mean()), 4),
                  "min": round(float(fuera.min()), 4), "max": round(float(fuera.max()), 4),
                  "votos": round(float(v["cuenta_bag"].mean()), 1)}

    # Bagging contra bosque: correlación entre árboles
    out["decorrelar"] = {
        nombre: {"corr": round(float(v["correlacion_media"](P)), 3),
                 "auc_uno": round(float(np.mean([roc_auc_score(y_te, p) for p in P])), 3),
                 "auc": round(float(auc[-1]), 3), "error": round(float(err[-1]), 3), "error_oob": round(float(oob[-1]), 3)}
        for nombre, P, err, auc, oob in [("bagging", v["P_bag"], v["error_bag"], v["auc_bag"], v["oob_error_bag"]),
                                         ("bosque", v["P_rf"], v["error_rf"], v["auc_rf"], v["oob_error_rf"])]
    }

    # Hiperparámetros con OOB
    out["hiper"] = {"m": v["ms"], "auc_m": r(v["oob_m"], 4), "hojas": v["hojas"], "auc_hoja": r(v["oob_hoja"], 4),
                    "m_mejor": int(v["m_mejor"]), "hoja_mejor": int(v["hoja_mejor"])}

    # OOB contra validación cruzada
    out["oob_cv"] = {"oob": round(float(v["auc_oob"](v["elegido"])), 4), "cv": round(float(v["auc_cv"].mean()), 4),
                     "prueba": round(float(roc_auc_score(y_te, v["p_elegido"])), 4),
                     "t_oob": round(float(v["t_oob"]), 1), "t_cv": round(float(v["t_cv"]), 1)}

    # Bicicletas: RECM, descomposición y ρσ²
    out["bicis_recm"] = [[k, round(float(x), 1)] for k, x in v["recm_b"].items()]
    tb = v["tabla_bv"]
    out["bicis_bv"] = [{"modelo": k, "varianza": round(float(tb.loc[k, "varianza"]), 1),
                        "resto": round(float(tb.loc[k, "sesgo² + ruido"]), 1), "ecm": round(float(tb.loc[k, "ECM"]), 1),
                        "recm": round(float(tb.loc[k, "RECM"]), 1)} for k in tb.index]
    rs = {}
    for nombre, T in v["pred_arboles"].items():
        s2, cov = v["rho_sigma"](T.astype(float))
        rs[nombre] = {"sigma2": round(float(s2)), "cov": round(float(cov)), "rho": round(float(cov / s2), 3),
                      "medida": round(float(T.mean(axis=1).var(axis=0).mean()))}
    out["bicis_rho"] = rs
    dia = v["dia"]
    out["dia"] = {"hora": v["horas"].tolist(), "real": r(v["yb_te"][dia], 0),
                  "arboles": r(v["pred_arbol"][:8, dia], 1),
                  "bagging": r(v["pred_arboles"]["bagging"].mean(axis=1)[:8, dia], 1)}

    # Importancias
    imp = v["imp"]
    out["importancias"] = {"nombres": imp.index.tolist(), "mdi": r(imp["MDI"], 4),
                           "perm": r(imp["permutación (prueba)"], 4), "distintos": imp["valores distintos"].astype(int).tolist()}

    # Comparación final
    c = v["comparacion"]
    out["comparacion"] = [{"modelo": k, "auc": round(float(c.loc[k, "AUC"]), 3), "exactitud": round(float(c.loc[k, "exactitud"]), 3),
                           "tiempo": round(float(c.loc[k, "tiempo (s)"]), 2)} for k in c.index]
    return out
