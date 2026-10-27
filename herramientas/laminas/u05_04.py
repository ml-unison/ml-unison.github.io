"""Datos de las láminas de u05-04-interpretabilidad.

Mismo código, datos y semillas que la libreta
unidades/05-arboles-ensambles/ejemplos/u05-e04-interpretabilidad.py.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


def r(a, d: int = 4):
    """Redondea a `d` decimales y convierte a listas de Python (como _comun.r)."""
    return np.round(np.asarray(a, dtype=float), d).tolist()


RAIZ = Path(__file__).resolve().parents[2]
SEMILLA = 2011


def _boosting(semilla=SEMILLA):
    import lightgbm as lgb

    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                             random_state=semilla, n_jobs=1, verbose=-1)


def _importancia_permutacion(modelo, X, y, K=10, semilla=SEMILLA):
    from sklearn.metrics import root_mean_squared_error

    rng = np.random.default_rng(semilla)
    base = root_mean_squared_error(y, modelo.predict(X))
    imp = {}
    for c in X.columns:
        perdidas = []
        for _ in range(K):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            perdidas.append(root_mean_squared_error(y, modelo.predict(Xp)))
        imp[c] = np.mean(perdidas) - base
    return pd.Series(imp).sort_values(ascending=False)


def _dependencia_parcial(modelo, X, atributo, malla):
    ice = np.empty((len(X), len(malla)))
    for k, v in enumerate(malla):
        ice[:, k] = modelo.predict(X.assign(**{atributo: v}))
    return ice, ice.mean(axis=0)


def _shap_por_enumeracion(modelo, fondo, x):
    d = fondo.shape[1]
    mascaras = np.array(list(itertools.product([False, True], repeat=d)))
    sintetico = pd.concat([fondo] * len(mascaras), ignore_index=True)
    presente = np.repeat(mascaras, len(fondo), axis=0)
    for j, c in enumerate(fondo.columns):
        sintetico.loc[presente[:, j], c] = x[c].iloc[0]
    v = modelo.predict(sintetico).reshape(len(mascaras), len(fondo)).mean(axis=1)
    tam = mascaras.sum(axis=1)
    indice = {tuple(m): k for k, m in enumerate(mascaras)}
    phi = np.zeros(d)
    for k, m in enumerate(mascaras):
        for j in np.flatnonzero(~m):
            con_j = m.copy()
            con_j[j] = True
            peso = math.factorial(tam[k]) * math.factorial(d - tam[k] - 1) / math.factorial(d)
            phi[j] += peso * (v[indice[tuple(con_j)]] - v[k])
    return phi, v[0]


def _enjambre_y(valores, ancho=0.4, semilla=0):
    h = 0.9 * valores.std() * len(valores) ** -0.2 + 1e-9
    dens = np.exp(-0.5 * ((valores[:, None] - valores[None, :]) / h) ** 2).sum(axis=1)
    return np.random.default_rng(semilla).uniform(-1, 1, len(valores)) * ancho * dens / dens.max()


def datos() -> dict:
    import shap
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import root_mean_squared_error
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    out: dict = {}
    casas = pd.read_csv(RAIZ / "datos" / "casas_ames.csv")
    casas = casas[casas["GrLivArea"] <= 4000].reset_index(drop=True)
    M2 = 0.092903
    CALIDAD = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    X = pd.DataFrame({
        "area": (casas["GrLivArea"] * M2).round(1), "calidad": casas["OverallQual"], "anio": casas["YearBuilt"],
        "remodelacion": casas["YearRemodAdd"], "sotano": (casas["TotalBsmtSF"] * M2).round(1),
        "cochera": casas["GarageCars"], "terreno": (casas["LotArea"] * M2).round(0),
        "banos": casas["FullBath"] + 0.5 * casas["HalfBath"], "chimeneas": casas["Fireplaces"],
        "cocina": casas["KitchenQual"].map(CALIDAD), "condicion": casas["OverallCond"],
    })
    barrio = casas["Neighborhood"]
    y = np.log(casas["SalePrice"]).to_numpy()
    i_tr, i_te = train_test_split(np.arange(len(casas)), test_size=0.2, random_state=SEMILLA)
    mediana_barrio = casas["SalePrice"].iloc[i_tr].groupby(barrio.iloc[i_tr]).median() / 1000
    X["barrio"] = barrio.map(mediana_barrio).fillna(mediana_barrio.median()).round(1)
    ATR = list(X.columns)
    X_tr, X_te, y_tr, y_te = X.iloc[i_tr], X.iloc[i_te], y[i_tr], y[i_te]

    # ===== modelos
    def lineal():
        logs = ["area", "sotano", "terreno"]
        otros = [c for c in ATR if c not in logs + ["barrio"]]
        return make_pipeline(make_column_transformer(
            (make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), logs),
            (StandardScaler(), otros), (OneHotEncoder(handle_unknown="ignore"), ["barrio_nombre"])), Ridge(alpha=10.0))

    X_lin = X.assign(barrio_nombre=barrio)
    cv = KFold(5, shuffle=True, random_state=SEMILLA)
    modelos = []
    for nombre, modelo, XX in [("ridge", lineal(), X_lin), ("lightgbm", _boosting(), X)]:
        e_cv = -cross_val_score(modelo, XX.iloc[i_tr], y_tr, cv=cv, scoring="neg_root_mean_squared_error")
        modelo.fit(XX.iloc[i_tr], y_tr)
        modelos.append({"modelo": nombre, "cv": round(float(e_cv.mean()), 3), "ee": round(float(e_cv.std() / np.sqrt(5)), 3),
                        "prueba": round(float(root_mean_squared_error(y_te, modelo.predict(XX.iloc[i_te]))), 3)})
        if nombre == "lightgbm":
            gb = modelo
        else:
            lin = modelo
    out["modelos"] = modelos
    # efectos del modelo lineal en unidades naturales
    coef = pd.Series(lin[-1].coef_, index=lin[0].get_feature_names_out())
    escalas = np.r_[lin[0].named_transformers_["pipeline"][-1].scale_, lin[0].named_transformers_["standardscaler"].scale_]
    efecto = coef.iloc[:len(escalas)].to_numpy() / escalas
    pasos = np.r_[[np.log(1.1)] * 3, np.ones(len(efecto) - 3)]
    nombres_lin = [n.split("__")[1] for n in coef.index[:len(escalas)]]
    out["efectos_lineal"] = {n: round(float(100 * (np.exp(e * p) - 1)), 1) for n, e, p in zip(nombres_lin, efecto, pasos)}

    # ===== importancia por permutación (original y con la gemela)
    imp = _importancia_permutacion(gb, X_te, y_te)
    out["base_recm"] = round(float(root_mean_squared_error(y_te, gb.predict(X_te))), 3)
    out["permutacion"] = [[k, round(float(v), 5)] for k, v in imp.items()]
    rng = np.random.default_rng(SEMILLA)
    X2 = X.assign(area_vendedor=(X["area"] * np.exp(rng.normal(0, 0.03, len(X)))).round(1))
    X2_tr, X2_te = X2.iloc[i_tr], X2.iloc[i_te]
    gemela = _boosting().fit(X2_tr, y_tr)
    imp2 = _importancia_permutacion(gemela, X2_te, y_te)
    out["permutacion_gemela"] = [[k, round(float(v), 5)] for k, v in imp2.items()]
    out["gemela"] = {"corr": round(float(X2["area"].corr(X2["area_vendedor"])), 3),
                     "recm": round(float(root_mean_squared_error(y_te, gemela.predict(X2_te))), 3)}

    # ===== PDP e ICE del área y del año
    muestra = np.random.default_rng(1).choice(len(X_tr), 60, replace=False)
    for atr in ("area", "anio"):
        malla = np.linspace(*np.quantile(X_tr[atr], [0.01, 0.99]), 40)
        ice, pdp = _dependencia_parcial(gb, X_tr, atr, malla)
        out[f"pdp_{atr}"] = {"malla": r(malla, 1), "pdp": r(np.exp(pdp) / 1000, 1),
                              "ice": [r(np.exp(f) / 1000, 1) for f in ice[muestra]],
                              "factor": round(float(np.exp(pdp[-1] - pdp[0])), 2)}
        if atr == "area":
            malla_area, pdp_area = malla, pdp
    # casas reales contra sintéticas (área, calidad)
    grandes = X_tr["area"] >= 250
    out["extrapolacion"] = {"area": r(X_tr["area"], 1), "calidad": X_tr["calidad"].tolist(),
                            "n_grandes": int(grandes.sum()), "n_grandes_baja": int((grandes & (X_tr["calidad"] <= 5)).sum()),
                            "frac_sintetica_baja": round(float((X_tr["calidad"] <= 5).mean()), 3)}

    # ===== modelo de juguete: Shapley a mano
    fondo = X_tr.sample(100, random_state=SEMILLA)
    JUG = ["area", "calidad", "anio"]
    juguete = _boosting().fit(X_tr[JUG], casas["SalePrice"].iloc[i_tr] / 1000)
    casa = X_te.loc[[X_te.index[np.argmin(np.abs(X_te["area"] - 180) + 10 * np.abs(X_te["calidad"] - 8) + np.abs(X_te["anio"] - 2005))]]]

    def v_coal(S):
        sint = fondo[JUG].copy()
        for c in S:
            sint[c] = casa[c].iloc[0]
        return float(juguete.predict(sint).mean())

    v = {frozenset(S): v_coal(S) for k in range(4) for S in itertools.combinations(JUG, k)}
    ordenes = []
    for orden in itertools.permutations(JUG):
        llegados, contrib = frozenset(), {}
        for j in orden:
            contrib[j] = round(v[llegados | {j}] - v[llegados], 1)
            llegados = llegados | {j}
        ordenes.append({"orden": list(orden), "contrib": contrib})
    ex_j = shap.TreeExplainer(juguete, data=fondo[JUG], feature_perturbation="interventional")
    out["juguete"] = {"casa": {c: float(casa[c].iloc[0]) for c in JUG},
                      "v": [[sorted(S, key=JUG.index), round(val, 1)] for S, val in v.items()],
                      "ordenes": ordenes,
                      "shapley": dict(zip(JUG, r(ex_j.shap_values(casa[JUG])[0], 1)))}

    # ===== SHAP del modelo completo
    k_casa = 45
    casa = X_te.iloc[[k_casa]]
    t0 = time.perf_counter()
    phi_enum, _ = _shap_por_enumeracion(gb, fondo, casa)
    t_enum = time.perf_counter() - t0
    explicador = shap.TreeExplainer(gb, data=fondo, feature_perturbation="interventional")
    t0 = time.perf_counter()
    phi_shap = explicador.shap_values(casa)[0]
    t_shap = time.perf_counter() - t0
    out["enumeracion"] = {"t_enum": round(t_enum, 1), "t_shap_ms": round(1000 * t_shap, 1),
                          "dif": float(f"{np.abs(phi_enum - phi_shap).max():.1e}")}
    sv = explicador.shap_values(X_te)
    phi0 = float(explicador.expected_value)
    out["cascada"] = {"phi0": round(phi0, 4), "phi": dict(zip(ATR, r(sv[k_casa], 4))),
                      "x": {c: float(X_te.iloc[k_casa][c]) for c in ATR},
                      "pred": round(float(np.exp(gb.predict(casa)[0]) / 1000), 1), "real": round(float(np.exp(y_te[k_casa]) / 1000), 1)}
    # enjambre
    imp_shap = pd.Series(np.abs(sv).mean(axis=0), index=ATR).sort_values(ascending=False)
    filas = []
    for c in imp_shap.index:
        j = ATR.index(c)
        val = X_te[c].to_numpy(dtype=float)
        rango = np.clip((val - np.quantile(val, 0.05)) / (np.quantile(val, 0.95) - np.quantile(val, 0.05) + 1e-9), 0, 1)
        filas.append({"atr": c, "media": round(float(imp_shap[c]), 4), "phi": r(sv[:, j], 3),
                      "dy": r(_enjambre_y(sv[:, j]), 3), "color": r(rango, 2)})
    out["enjambre"] = filas
    # dependencia del área
    j_area = ATR.index("area")
    out["dependencia"] = {"area": r(X_te["area"], 1), "phi": r(sv[:, j_area], 3), "calidad": X_te["calidad"].tolist(),
                          "malla": r(malla_area, 1),
                          "pdp": r(pdp_area - np.interp(X_te["area"], malla_area, pdp_area).mean() + sv[:, j_area].mean(), 3)}
    # gemela con SHAP
    ex2 = shap.TreeExplainer(gemela, data=X2_tr.sample(100, random_state=SEMILLA), feature_perturbation="interventional")
    sv2 = ex2.shap_values(X2_te)
    j2 = X2.columns.get_loc("area_vendedor")
    out["gemela_shap"] = {"original": round(float(np.abs(sv[:, j_area]).mean()), 4),
                          "area": round(float(np.abs(sv2[:, j_area]).mean()), 4),
                          "vendedor": round(float(np.abs(sv2[:, j2]).mean()), 4),
                          "suma": round(float(np.abs(sv2[:, [j_area, j2]].sum(axis=1)).mean()), 4)}
    return out


if __name__ == "__main__":
    import json

    d = datos()
    print(json.dumps({k: v for k, v in d.items() if k not in ("enjambre", "extrapolacion", "dependencia", "pdp_area", "pdp_anio")}, ensure_ascii=False, indent=1))
