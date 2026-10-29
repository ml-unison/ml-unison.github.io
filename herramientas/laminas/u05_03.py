"""Datos de las láminas de la presentación u05-03-boosting.

Mismo código y semillas que la libreta unidades/05-arboles-ensambles/ejemplos/u05-e03-boosting.py.
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from _comun import r


class Tocon:
    """Árbol de una pregunta, h(x) = s si x_j > u y −s si no, que minimiza el error ponderado."""

    def fit(self, X, y, sample_weight):
        w = sample_weight
        self.error = np.inf
        for j in range(X.shape[1]):
            orden = np.argsort(X[:, j], kind="stable")
            xs, ys, ws = X[orden, j], y[orden], w[orden]
            pos_izq = np.r_[0, np.cumsum(ws * (ys == 1))]
            neg_der = w[y == -1].sum() - np.r_[0, np.cumsum(ws * (ys == -1))]
            err = pos_izq + neg_der
            valido = np.r_[False, xs[1:] != xs[:-1], False]
            for s, e in ((1, err), (-1, w.sum() - err)):
                k = np.flatnonzero(valido)[np.argmin(e[valido])]
                if e[k] < self.error:
                    self.error, self.j, self.u, self.s = e[k], j, (xs[k - 1] + xs[k]) / 2, s
        return self

    def predict(self, X):
        return self.s * np.where(X[:, self.j] > self.u, 1, -1)


def _adaboost(X, y, T):
    N = len(y)
    w = np.full(N, 1 / N)
    tocones, alfas, pesos, eps = [], [], [w], []
    for _ in range(T):
        h = Tocon().fit(X, y, sample_weight=w)
        pred = h.predict(X)
        e = w[pred != y].sum()
        a = 0.5 * np.log((1 - e) / e)
        w = w * np.exp(-a * y * pred)
        w = w / w.sum()
        tocones.append(h); alfas.append(a); pesos.append(w); eps.append(e)
    return tocones, np.array(alfas), np.array(pesos), np.array(eps)


def _franjas(etiq, xs, ys):
    out = []
    for i in range(etiq.shape[0]):
        j = 0
        while j < etiq.shape[1]:
            k = j
            while k + 1 < etiq.shape[1] and etiq[i, k + 1] == etiq[i, j]:
                k += 1
            out.append([i, j, k, int(etiq[i, j])])
            j = k + 1
    return {"x": r(xs, 3), "y": r(ys, 3), "franjas": out}


def _gradient_boosting(X, y, M, eta, profundidad, semilla=0):
    from sklearn.tree import DecisionTreeRegressor

    azar = np.random.RandomState(semilla)
    f0 = y.mean()
    F = np.full(len(y), f0)
    arboles = []
    for _ in range(M):
        h = DecisionTreeRegressor(max_depth=profundidad, random_state=azar).fit(X, y - F)
        F += eta * h.predict(X)
        arboles.append(h)
    return f0, arboles


def _predecir_gb(X, f0, arboles, eta):
    return f0 + eta * sum(h.predict(X) for h in arboles) if arboles else np.full(len(X), f0)


def datos() -> dict:
    import pandas as pd
    import rdp
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    from threadpoolctl import threadpool_limits
    from sklearn.model_selection import train_test_split
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor, RandomForestClassifier)
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score, root_mean_squared_error

    warnings.filterwarnings("ignore", message=".*eval_set.*")
    SEMILLA = 2011
    out = {}

    # ---------- AdaBoost en los pingüinos ----------
    ping = rdp.datos.cargar("pinguinos").dropna(subset=["bill_length_mm", "bill_depth_mm"])
    Xp = ping[["bill_length_mm", "bill_depth_mm"]].to_numpy()
    yp = np.where(ping["species"] == "Chinstrap", 1, -1)
    T = 200
    tocones, alfas, pesos, eps = _adaboost(Xp, yp, T)
    F_hist = np.cumsum(alfas[:, None] * np.array([h.predict(Xp) for h in tocones]), axis=0)
    err_t = np.mean(np.sign(F_hist) != yp, axis=1)
    cota_t = np.cumprod(2 * np.sqrt(eps * (1 - eps)))
    gx, gy = np.linspace(31, 61, 151), np.linspace(12.5, 22, 96)
    G1, G2 = np.meshgrid(gx, gy)
    malla = np.column_stack([G1.ravel(), G2.ravel()])
    rondas = {}
    for t in [1, 2, 3, 10, 50, 200]:
        F = sum(a * h.predict(malla) for h, a in zip(tocones[:t], alfas[:t]))
        h = tocones[t - 1]
        rondas[str(t)] = {"regiones": _franjas((F > 0).astype(int).reshape(G1.shape), gx, gy),
                          "w": r(pesos[t] * len(yp), 3),  # peso relativo al inicial (1 = 1/N)
                          "j": int(h.j), "u": round(float(h.u), 2), "s": int(h.s),
                          "error": round(float(err_t[t - 1]), 3), "eps": round(float(eps[t - 1]), 3),
                          "alfa": round(float(alfas[t - 1]), 3)}
    # peso total de los mal clasificados por h_1, después de la actualización
    mal1 = tocones[0].predict(Xp) != yp
    out["adaboost"] = {"x": r(Xp[:, 0], 1), "y": r(Xp[:, 1], 1), "g": (yp == 1).astype(int).tolist(),
                       "rondas": rondas, "peso_errores_r2": round(float(pesos[1][mal1].sum()), 3),
                       "tocones": [{"j": int(h.j), "u": round(float(h.u), 2), "s": int(h.s), "eps": round(float(e), 3),
                                    "alfa": round(float(a), 3)} for h, e, a in zip(tocones[:3], eps[:3], alfas[:3])]}
    out["cota"] = {"t": list(range(1, T + 1)), "error": r(err_t, 4), "cota": r(cota_t, 4)}
    # alfa óptimo de la ronda 5 por minimización numérica
    from scipy.optimize import minimize_scalar
    F_ant, h5 = F_hist[3], tocones[4].predict(Xp)
    opt = minimize_scalar(lambda a: np.mean(np.exp(-yp * (F_ant + a * h5))), bounds=(0, 3), method="bounded",
                          options={"xatol": 1e-10})
    out["alfa5"] = {"numerico": round(float(opt.x), 6), "formula": round(float(alfas[4]), 6), "eps": round(float(eps[4]), 4)}

    # ---------- Gradient boosting en las bicicletas ----------
    b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
    b["anio"] = b.fecha.dt.year - 2011
    b["clima"] = b.clima.clip(upper=3)
    entrena = b[b.fecha.dt.day <= 19]
    prueba = b[b.fecha.dt.day >= 20]
    atributos = ["hora", "laboral", "clima", "anio", "temp", "humedad", "viento"]
    Xb, yb = entrena[atributos].to_numpy(float), entrena["total"].to_numpy(float)
    Xb_te, yb_te = prueba[atributos].to_numpy(float), prueba["total"].to_numpy(float)
    M, ETA, PROF = 300, 0.1, 4
    f0, arboles = _gradient_boosting(Xb, yb, M, ETA, PROF)
    pred_nuestro = _predecir_gb(Xb_te, f0, arboles, ETA)
    pred_sk = GradientBoostingRegressor(n_estimators=M, learning_rate=ETA, max_depth=PROF, random_state=0).fit(Xb, yb).predict(Xb_te)
    hgb_b = HistGradientBoostingRegressor(loss="poisson", random_state=0).fit(Xb, yb)
    out["bicis"] = {"dif_max": float(f"{np.abs(pred_nuestro - pred_sk).max():.1e}"),
                    "recm_nuestro": round(float(root_mean_squared_error(yb_te, pred_nuestro)), 1),
                    "recm_sk": round(float(root_mean_squared_error(yb_te, pred_sk)), 1),
                    "recm_poisson_hgb": round(float(root_mean_squared_error(yb_te, hgb_b.predict(Xb_te))), 1),
                    # referencias con los mismos atributos y la misma partición
                    "recm_promedio": round(float(root_mean_squared_error(yb_te, np.full(len(yb_te), yb.mean()))), 1),
                    "recm_lineal": round(float(root_mean_squared_error(yb_te, LinearRegression().fit(Xb, yb).predict(Xb_te))), 1),
                    "recm_arbol10": round(float(root_mean_squared_error(
                        yb_te, DecisionTreeRegressor(max_depth=10, random_state=0).fit(Xb, yb).predict(Xb_te))), 1),
                    "n_entrena": len(entrena), "n_prueba": len(prueba), "n_horas": len(b)}

    # etapas con tocones sobre la hora (días laborales)
    lab = entrena[entrena.laboral == 1]
    xh, yh = lab[["hora"]].to_numpy(float), lab["total"].to_numpy(float)
    f0_h, tocones_h = _gradient_boosting(xh, yh, 300, 0.5, 1)
    horas = np.arange(24.0)[:, None]
    etapas = {}
    for m in [0, 1, 2, 3, 5, 10, 20, 50, 300]:
        Fm = _predecir_gb(horas, f0_h, tocones_h[:m], 0.5)
        sig = tocones_h[m].predict(horas) if m < len(tocones_h) else np.zeros(24)
        etapas[str(m)] = {"F": r(Fm, 1), "h": r(sig, 1)}
    rng = np.random.default_rng(0)
    i = rng.choice(len(yh), 900, replace=False)
    out["etapas"] = {"x": r(xh[i, 0] + rng.uniform(-0.3, 0.3, len(i)), 2), "y": r(yh[i], 0),
                     "promedio": r(lab.groupby("hora").total.mean().to_numpy(), 1), "eta": 0.5, "m": etapas}

    # ---------- Mercadeo bancario ----------
    banco = rdp.datos.cargar("mercadeo_banco")
    categoricas = ["trabajo", "estado_civil", "escolaridad", "incumplimiento", "hipoteca", "prestamo",
                   "contacto", "mes", "resultado_previo"]
    for c in categoricas:
        banco[c] = banco[c].astype("category")
    y_banco = banco["suscribe"]
    X_banco = banco.drop(columns="suscribe")
    X_tr, X_te, y_tr, y_te = train_test_split(X_banco, y_banco, test_size=0.25, stratify=y_banco, random_state=SEMILLA)

    def logistica(columnas):
        numericas = [c for c in columnas if c not in categoricas]
        return make_pipeline(
            make_column_transformer((OneHotEncoder(handle_unknown="ignore"), categoricas), (StandardScaler(), numericas)),
            LogisticRegression(max_iter=2000))

    todas = list(X_banco.columns)
    fuga = {}
    for nombre, modelo in [("logística", logistica(todas)),
                           ("HistGradientBoosting", HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEMILLA))]:
        modelo.fit(X_tr[todas], y_tr)
        fuga[nombre] = round(float(roc_auc_score(y_te, modelo.predict_proba(X_te[todas])[:, 1])), 3)
    fuga["solo la duración"] = round(float(roc_auc_score(y_te, X_te.duracion)), 3)
    bins = np.arange(0, 1500, 30)
    hist = {str(v): r(np.histogram(banco.duracion[banco.suscribe == v], bins=bins, density=True)[0] * 1000, 3) for v in (0, 1)}
    out["banco"] = {"n": len(banco), "prevalencia": round(float(y_banco.mean()), 3), "n_tr": len(X_tr), "n_te": len(X_te),
                    "fuga": fuga, "hist": {"bordes": bins.tolist(), "no": hist["0"], "si": hist["1"]}}

    cols = [c for c in todas if c != "duracion"]
    Xtr, Xte = X_tr[cols], X_te[cols]
    X_a, X_v, y_a, y_v = train_test_split(Xtr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEMILLA)

    def xgb_modelo(n, eta=0.05, **kw):
        return xgb.XGBClassifier(n_estimators=n, learning_rate=eta, max_depth=4, enable_categorical=True,
                                 tree_method="hist", eval_metric="logloss", random_state=SEMILLA, n_jobs=4, **kw)

    def lgb_modelo(n, eta=0.05):
        return lgb.LGBMClassifier(n_estimators=n, learning_rate=eta, num_leaves=15, random_state=SEMILLA,
                                  n_jobs=4, verbose=-1)

    paro_xgb = xgb_modelo(3000, early_stopping_rounds=100).fit(X_a, y_a, eval_set=[(X_v, y_v)], verbose=False)
    paro_lgb = lgb_modelo(3000).fit(X_a, y_a, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(100, verbose=False)])
    n_xgb, n_lgb = paro_xgb.best_iteration + 1, paro_lgb.best_iteration_

    def cb_modelo(n, **kw):
        return CatBoostClassifier(iterations=n, learning_rate=0.05, depth=6, cat_features=categoricas,
                                  random_seed=SEMILLA, thread_count=4, verbose=0, allow_writing_files=False, **kw)

    paro_cb = cb_modelo(3000, early_stopping_rounds=100).fit(X_a, y_a, eval_set=(X_v, y_v))
    n_cb = paro_cb.get_best_iteration() + 1
    modelos = {
        "logística": logistica(cols),
        "bosque aleatorio": make_pipeline(
            make_column_transformer((OrdinalEncoder(), categoricas), remainder="passthrough"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=4, random_state=SEMILLA)),
        "HistGradientBoosting": HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEMILLA),
        "XGBoost": xgb_modelo(n_xgb),
        "LightGBM": lgb_modelo(n_lgb),
        "CatBoost": cb_modelo(n_cb),
    }

    def captura(p, y, fraccion):
        k = int(round(fraccion * len(p)))
        return np.asarray(y)[np.argsort(-p)[:k]].sum() / np.sum(y)

    comp = []
    for nombre, modelo in modelos.items():
        with threadpool_limits(4):
            t0 = time.perf_counter()
            modelo.fit(Xtr, y_tr)
            seg = time.perf_counter() - t0
        p = modelo.predict_proba(Xte)[:, 1]
        comp.append({"modelo": nombre, "auc": round(float(roc_auc_score(y_te, p)), 3), "segundos": round(seg, 2),
                     "captura": [round(float(captura(p, y_te, q)), 3) for q in (0.1, 0.2, 0.3)]})
    imp = permutation_importance(modelos["LightGBM"], Xte, y_te, scoring="roc_auc", n_repeats=5, random_state=SEMILLA, n_jobs=4)
    imp = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).head(5)
    out["comparacion"] = {"modelos": comp, "n_xgb": int(n_xgb), "n_lgb": int(n_lgb), "n_cb": int(n_cb),
                          "importancia": [[k, round(float(v), 3)] for k, v in imp.items()]}

    # curvas de tasa de aprendizaje
    curvas = []
    for eta in [0.3, 0.1, 0.03]:
        m = xgb_modelo(1500, eta=eta).fit(X_a, y_a, eval_set=[(X_a, y_a), (X_v, y_v)], verbose=False)
        ev = m.evals_result()
        e, v = np.array(ev["validation_0"]["logloss"]), np.array(ev["validation_1"]["logloss"])
        idx = np.r_[np.arange(0, 60), np.arange(60, 1500, 5)]
        curvas.append({"eta": eta, "n": (idx + 1).tolist(), "entrena": r(e[idx], 4), "valida": r(v[idx], 4),
                       "mejor_n": int(v.argmin() + 1), "mejor": round(float(v.min()), 4), "final": round(float(v[-1]), 4)})
    out["curvas"] = curvas

    # verificación de la ganancia de XGBoost
    uno = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=1.0, gamma=0, min_child_weight=0,
                            base_score=0.5, tree_method="exact").fit(Xtr[["edad"]], y_tr)
    arbol_xgb = uno.get_booster().trees_to_dataframe()
    umbral = float(arbol_xgb.loc[0, "Split"])
    g = 0.5 - y_tr.to_numpy()
    h = np.full(len(g), 0.25)
    izq = Xtr["edad"].to_numpy() < umbral
    lam = 1.0
    G_I, H_I, G_D, H_D = g[izq].sum(), h[izq].sum(), g[~izq].sum(), h[~izq].sum()
    out["xgb"] = {"umbral": umbral, "G_I": round(float(G_I), 1), "H_I": round(float(H_I), 2), "G_D": round(float(G_D), 1),
                  "H_D": round(float(H_D), 2), "N_I": int(izq.sum()), "N_D": int((~izq).sum()),
                  "w_I": round(float(-G_I / (H_I + lam)), 4), "w_D": round(float(-G_D / (H_D + lam)), 4),
                  "w_I_xgb": round(float(arbol_xgb.loc[1, "Gain"]), 4), "w_D_xgb": round(float(arbol_xgb.loc[2, "Gain"]), 4),
                  "ganancia": round(float(0.5 * (G_I**2 / (H_I + lam) + G_D**2 / (H_D + lam) - (G_I + G_D) ** 2 / (H_I + H_D + lam))), 1),
                  "ganancia_xgb": round(float(arbol_xgb.loc[0, "Gain"]), 1)}

    # histogramas contra exacto
    X_num = make_column_transformer((OrdinalEncoder(), categoricas), remainder="passthrough").fit_transform(Xtr)
    tiempos = {}
    for nombre, modelo in [("exacto", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEMILLA)),
                           ("histogramas", HistGradientBoostingClassifier(max_iter=100, max_depth=4, early_stopping=False, random_state=SEMILLA))]:
        with threadpool_limits(4):
            t0 = time.perf_counter()
            modelo.fit(X_num, y_tr)
            tiempos[nombre] = round(time.perf_counter() - t0, 2)
    out["histogramas"] = {"saldo_distintos": int(Xtr.saldo.nunique()), "tiempos": tiempos}

    # codificar una categoría con la respuesta: un identificador al azar, sin información
    azar = np.random.default_rng(0)
    id_tr = pd.Series(azar.integers(0, 8000, len(Xtr)), index=Xtr.index).astype(str)
    id_te = pd.Series(azar.integers(0, 8000, len(Xte)), index=Xte.index).astype(str)
    previa, a = float(y_tr.mean()), 1.0

    def ingenua(c):  # con todos los datos de entrenamiento, incluido el propio
        g = pd.DataFrame({"c": id_tr, "y": y_tr}).groupby("c").y.agg(["sum", "count"])
        return ((g["sum"].reindex(c).fillna(0) + a * previa) / (g["count"].reindex(c).fillna(0) + a)).to_numpy()

    def ordenada(c, semilla=0):  # solo con los que van antes en un orden al azar
        orden = np.random.default_rng(semilla).permutation(len(c))
        suma, cuenta, cod = {}, {}, np.empty(len(c))
        for k in orden:
            ck, yk = c.iloc[k], y_tr.iloc[k]
            cod[k] = (suma.get(ck, 0) + a * previa) / (cuenta.get(ck, 0) + a)
            suma[ck] = suma.get(ck, 0) + yk
            cuenta[ck] = cuenta.get(ck, 0) + 1
        return cod

    codif = {}
    for nombre, (c_tr, c_te) in {"sin": (None, None), "ingenua": (ingenua(id_tr), ingenua(id_te)),
                                 "ordenada": (ordenada(id_tr), ingenua(id_te))}.items():
        A, B = Xtr.copy(), Xte.copy()
        if c_tr is not None:
            A["id"], B["id"] = c_tr, c_te
        m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEMILLA).fit(A, y_tr)
        codif[nombre] = {"auc_tr": round(float(roc_auc_score(y_tr, m.predict_proba(A)[:, 1])), 3),
                         "auc_te": round(float(roc_auc_score(y_te, m.predict_proba(B)[:, 1])), 3)}
    A, B = Xtr.copy(), Xte.copy()
    A["id"], B["id"] = id_tr, id_te
    m = CatBoostClassifier(iterations=n_cb, learning_rate=0.05, depth=6, cat_features=categoricas + ["id"],
                           random_seed=SEMILLA, thread_count=4, verbose=0, allow_writing_files=False).fit(A, y_tr)
    codif["catboost"] = {"auc_te": round(float(roc_auc_score(y_te, m.predict_proba(B)[:, 1])), 3)}
    codif["n_id"] = int(id_tr.nunique())
    out["codificacion"] = codif
    return out
