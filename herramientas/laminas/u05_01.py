"""Datos de las láminas de u05-01-arboles (árboles de decisión).

Mismo código y semillas que la libreta unidades/05-arboles-ensambles/ejemplos/u05-e01-arboles.py.
"""

from __future__ import annotations

import numpy as np

SEMILLA = 2011
DOM = (31.0, 61.0, 12.5, 22.0)   # largo × profundidad del pico (mm)


def r(a, d: int = 4):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def _rectangulos(arbol, dom=DOM):
    """Hojas de un árbol de 2 atributos como rectángulos [x0, x1, y0, y1, clase] dentro de `dom`."""
    t = arbol.tree_
    out = []

    def bajar(nodo, x0, x1, y0, y1):
        if t.children_left[nodo] == -1:
            out.append([round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3), int(np.argmax(t.value[nodo][0]))])
            return
        u = float(t.threshold[nodo])
        if t.feature[nodo] == 0:
            bajar(t.children_left[nodo], x0, min(x1, max(x0, u)), y0, y1)
            bajar(t.children_right[nodo], max(x0, min(x1, u)), x1, y0, y1)
        else:
            bajar(t.children_left[nodo], x0, x1, y0, min(y1, max(y0, u)))
            bajar(t.children_right[nodo], x0, x1, max(y0, min(y1, u)), y1)

    bajar(0, *dom)
    return [q for q in out if q[1] > q[0] and q[3] > q[2]]


def _franjas(etiq):
    out = []
    for i in range(etiq.shape[0]):
        j = 0
        while j < etiq.shape[1]:
            k = j
            while k + 1 < etiq.shape[1] and etiq[i, k + 1] == etiq[i, j]:
                k += 1
            out.append([i, j, k, int(etiq[i, j])])
            j = k + 1
    return out


def _gini(conteos):
    conteos = np.asarray(conteos, dtype=float)
    p = conteos / np.maximum(conteos.sum(-1, keepdims=True), 1)
    return 1 - (p**2).sum(-1)


def datos() -> dict:
    import pandas as pd
    import rdp
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, cross_validate
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    out: dict = {}

    # ===== pingüinos
    ping = rdp.datos.cargar("pinguinos").dropna(subset=["bill_length_mm", "bill_depth_mm"])
    ESPECIES = ["Adelie", "Chinstrap", "Gentoo"]
    Xp = ping[["bill_length_mm", "bill_depth_mm"]].to_numpy()
    yp = ping["species"].map({e: k for k, e in enumerate(ESPECIES)}).to_numpy()
    pliegues = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
    niveles = []
    for prof in [1, 2, 3, 4, 6, None]:
        m = DecisionTreeClassifier(max_depth=prof, random_state=SEMILLA).fit(Xp, yp)
        cv = cross_validate(DecisionTreeClassifier(max_depth=prof, random_state=SEMILLA), Xp, yp, cv=pliegues,
                            return_train_score=True)
        niveles.append({"prof": "sin límite" if prof is None else str(prof), "hojas": int(m.get_n_leaves()),
                        "exact": round(float(m.score(Xp, yp)), 3), "cv": round(float(cv["test_score"].mean()), 3),
                        "rect": _rectangulos(m)})
    out["pinguinos"] = {"x": r(Xp[:, 0], 1), "y": r(Xp[:, 1], 1), "g": yp.tolist(), "niveles": niveles}

    # barrido de umbrales en la raíz (largo del pico)
    orden = np.argsort(Xp[:, 0], kind="stable")
    xs, ys = Xp[orden, 0], yp[orden]
    uno = np.eye(3)[ys]
    izq = np.cumsum(uno, axis=0)[:-1]
    der = uno.sum(axis=0) - izq
    n_izq = np.arange(1, len(ys))
    imp = (n_izq * _gini(izq) + (len(ys) - n_izq) * _gini(der)) / len(ys)
    ok = xs[1:] > xs[:-1]
    umbrales = (xs[1:][ok] + xs[:-1][ok]) / 2
    out["barrido"] = {"umbral": r(umbrales, 3), "gan": r(_gini(np.bincount(yp)) - imp[ok], 4),
                      "izq": izq[ok].astype(int).tolist(), "total": np.bincount(yp).tolist(),
                      "padre": round(float(_gini(np.bincount(yp))), 4)}

    # impurezas para dos clases y el ejemplo 80/20
    p = np.linspace(0, 1, 101)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(p > 0, p * np.log2(p), 0) - np.where(p < 1, (1 - p) * np.log2(1 - p), 0)
    out["impureza"] = {"p": r(p, 2), "gini": r(2 * p * (1 - p)), "entropia": r(H / 2), "error": r(np.minimum(p, 1 - p))}

    # ===== bicicletas
    b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
    b["anio"] = b.fecha.dt.year - 2011
    b["clima"] = b.clima.clip(upper=3)
    entrena, prueba = b[b.fecha.dt.day <= 19], b[b.fecha.dt.day >= 20]
    atributos = ["hora", "laboral", "clima", "anio", "temp", "humedad", "viento"]
    Xb, yb, Xt, yt = entrena[atributos], entrena["total"], prueba[atributos], prueba["total"]

    filas = []
    for prof in [2, 3, 4, 6, 8, 10, 12, 14, 16, 20, None]:
        m = DecisionTreeRegressor(max_depth=prof, random_state=SEMILLA).fit(Xb, yb)
        filas.append([40 if prof is None else prof, int(m.get_n_leaves()),
                      round(root_mean_squared_error(yb, m.predict(Xb)), 1), round(root_mean_squared_error(yt, m.predict(Xt)), 1)])
    out["profundidad"] = {"prof": [f[0] for f in filas], "hojas": [f[1] for f in filas],
                          "entrena": [f[2] for f in filas], "prueba": [f[3] for f in filas]}

    alfas = np.geomspace(1, 3000, 36)
    dias = entrena["fecha"].dt.date
    cvs, ees, hs = [], [], []
    for a in alfas:
        sc = -cross_val_score(DecisionTreeRegressor(ccp_alpha=a, random_state=SEMILLA), Xb, yb, groups=dias,
                              cv=GroupKFold(5), scoring="neg_root_mean_squared_error")
        cvs.append(sc.mean())
        ees.append(sc.std(ddof=1) / np.sqrt(len(sc)))
        hs.append(int(DecisionTreeRegressor(ccp_alpha=a, random_state=SEMILLA).fit(Xb, yb).get_n_leaves()))
    i_min = int(np.argmin(cvs))
    podado = DecisionTreeRegressor(ccp_alpha=alfas[i_min], random_state=SEMILLA).fit(Xb, yb)
    arbol3 = DecisionTreeRegressor(max_depth=3, random_state=SEMILLA).fit(Xb, yb)
    ruta = DecisionTreeRegressor(random_state=SEMILLA).cost_complexity_pruning_path(Xb, yb)
    out["poda"] = {"alfa": r(alfas, 2), "cv": r(cvs, 2), "ee": r(ees, 2), "hojas": hs, "i": i_min,
                   "alfa_cv": round(float(alfas[i_min]), 1), "hojas_cv": int(podado.get_n_leaves()),
                   "prof_cv": int(podado.get_depth()), "cv_min": round(float(cvs[i_min]), 1),
                   "prueba": round(root_mean_squared_error(yt, podado.predict(Xt)), 1),
                   "n_sub": len(ruta.ccp_alphas)}
    out["resumen"] = [["M1 lineal (U4)", 165.0], ["M2 lineal (U4)", 153.5], ["M3 lineal (U4)", 107.4],
                      ["M4 lineal, interacción a mano", 86.4], ["M4 Poisson (U4)", 72.2],
                      ["árbol de profundidad 3", round(root_mean_squared_error(yt, arbol3.predict(Xt)), 1)],
                      ["árbol sin límite", filas[-1][3]], ["árbol podado", out["poda"]["prueba"]]]

    def escenario(**cambios):
        base = {"hora": 17, "laboral": 1, "clima": 1, "anio": 1, "temp": 25.0, "humedad": 50, "viento": 10.0}
        n = max(np.size(v) for v in cambios.values())
        return pd.DataFrame({k: np.broadcast_to(cambios.get(k, v), n) for k, v in base.items()})[atributos]

    horas = np.arange(24)
    temps = np.linspace(0, 40, 401)
    obs = {lab: entrena[(entrena.laboral == lab) & (entrena.anio == 1)].groupby("hora").total.mean() for lab in (0, 1)}
    out["escalones"] = {"laboral": r(podado.predict(escenario(hora=horas, laboral=1)), 1),
                        "descanso": r(podado.predict(escenario(hora=horas, laboral=0)), 1),
                        "obs_laboral": r(obs[1].values, 1), "obs_descanso": r(obs[0].values, 1),
                        "temp": r(temps, 1), "pred_temp": r(podado.predict(escenario(temp=temps)), 1)}

    # ===== inestabilidad: árboles en remuestras bootstrap (mismo orden de números aleatorios que la libreta)
    g1, g2 = np.meshgrid(np.linspace(31, 61, 400), np.linspace(12.5, 22, 300))
    malla = np.column_stack([g1.ravel(), g2.ravel()])
    rng = np.random.default_rng(SEMILLA)
    remuestras, raices, preds = [], [], []
    for k in range(200):
        i = rng.integers(0, len(yp), len(yp))
        m = DecisionTreeClassifier(random_state=SEMILLA).fit(Xp[i], yp[i])
        raices.append(int(m.tree_.feature[0]))
        preds.append(m.predict(malla))
        if k < 4:
            remuestras.append({"rect": _rectangulos(m), "hojas": int(m.get_n_leaves()),
                               "raiz": ["largo", "profundidad"][m.tree_.feature[0]],
                               "umbral": round(float(m.tree_.threshold[0]), 2)})
    preds = np.array(preds)
    moda = np.array([np.bincount(c, minlength=3).argmax() for c in preds.T])
    desacuerdo = (preds != moda).mean(axis=0)
    # votación en una malla más gruesa (para las láminas)
    gx, gy = np.linspace(31, 61, 151), np.linspace(12.5, 22, 96)
    sub = np.ix_(np.searchsorted(np.linspace(12.5, 22, 300), gy).clip(0, 299),
                 np.searchsorted(np.linspace(31, 61, 400), gx).clip(0, 399))
    voto = moda.reshape(g1.shape)[sub]
    out["inestabilidad"] = {"remuestras": remuestras, "raiz_prof": int(np.sum(np.array(raices) == 1)),
                            "desacuerdo10": round(float((desacuerdo >= 0.10).mean()), 3),
                            "voto": {"x": r(gx, 3), "y": r(gy, 3), "franjas": _franjas(voto)}}
    return out
