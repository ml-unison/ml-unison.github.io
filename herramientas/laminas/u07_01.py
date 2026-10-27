"""Datos de las láminas de u07-01-series-validacion (Series de tiempo: estructura y validación).

Mismo código, partición y semillas que la libreta
unidades/07-series-tiempo/ejemplos/u07-e01-series-validacion.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _comun import r

RAIZ = Path(__file__).resolve().parents[2]
SEMILLA = 2027
ENTRENA, PRUEBA = slice("1992", "2020"), slice("2021", "2024")


def armonicos(fechas, K=3):
    t = 2 * np.pi * (fechas.day_of_year - 1) / 365.25
    columnas = [np.ones(len(fechas))]
    for k in range(1, K + 1):
        columnas += [np.cos(k * t), np.sin(k * t)]
    return np.column_stack(columnas)


def datos() -> dict:
    import statsmodels.api as sm
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf

    out: dict = {}
    d = pd.read_csv(RAIZ / "datos" / "temperatura_hermosillo.csv", parse_dates=["fecha"])
    y_crudo = d.set_index("fecha")["tmed"].asfreq("D")
    y = y_crudo.interpolate()

    # ===== serie completa (cada día) y un año de cerca
    out["serie"] = {"y": r(y, 1),  # un valor por día desde el 1 de enero de 1992
                    "n": int(len(y)), "faltan": int(y_crudo.isna().sum()),
                    "registros": int(len(d))}

    # ===== climatología con armónicos (1992–2020)
    y_tr = y[ENTRENA]
    beta, *_ = np.linalg.lstsq(armonicos(y_tr.index), y_tr.to_numpy(), rcond=None)
    clim = pd.Series(armonicos(y.index) @ beta, index=y.index)
    clim_cruda = y_tr.groupby(y_tr.index.day_of_year).mean()
    dias = pd.date_range("2001-01-01", "2001-12-31")
    muestra = np.sort(np.random.default_rng(1).choice(len(y_tr), 3000, replace=False))  # solo para dibujar
    anom = y - clim
    out["clima"] = {"dia": list(range(1, 366)), "armonicos": r(armonicos(dias) @ beta, 2),
                    "cruda": r(clim_cruda.loc[1:365], 2),
                    "pd": y_tr.index.day_of_year.to_numpy()[muestra].tolist(), "py": r(y_tr.to_numpy()[muestra], 1),
                    "min": round(float(clim.min()), 1), "max": round(float(clim.max()), 1),
                    "sd_y": round(float(y[ENTRENA].std()), 2), "sd_a": round(float(anom[ENTRENA].std()), 2),
                    "r2": round(float(1 - anom[ENTRENA].var() / y[ENTRENA].var()), 3)}
    a23 = y["2023"]
    out["anio2023"] = {"dia": a23.index.day_of_year.tolist(), "y": r(a23, 1), "clim": r(clim["2023"], 2)}

    # ===== STL (cada 3 días para dibujar)
    stl = STL(y, period=365).fit()
    paso = slice(None, None, 3)
    out["stl"] = {"paso": 3, "serie": r(y.to_numpy()[paso], 1), "tendencia": r(stl.trend.to_numpy()[paso], 2),
                  "estacional": r(stl.seasonal.to_numpy()[paso], 2), "residuo": r(stl.resid.to_numpy()[paso], 2),
                  "sd_resid": round(float(stl.resid.std()), 2)}

    # ===== ACF
    a_tr = anom[ENTRENA].to_numpy()
    T = len(a_tr)
    out["acf"] = {"cruda": r(acf(y[ENTRENA], nlags=730), 3), "anom": r(acf(a_tr, nlags=30), 3),
                  "banda": round(1.96 / np.sqrt(T), 3), "T": T}
    # mismo generador y mismo orden de llamadas que la libreta: ruido blanco y luego caminatas
    rng_c = np.random.default_rng(SEMILLA)
    ruido = rng_c.normal(size=T)
    r_ruido = acf(ruido, nlags=40)[1:]
    out["acf"]["ruido"] = r(acf(ruido, nlags=30), 3)
    out["acf"]["ruido_fuera"] = int(np.sum(np.abs(r_ruido) > 1.96 / np.sqrt(T)))

    # ===== caminata aleatoria
    caminos = rng_c.normal(size=(2000, 500)).cumsum(axis=1)
    var_emp = caminos.var(axis=0)
    out["caminata"] = {"t": list(range(2, 501, 2)), "caminos": [r(c[1::2], 2) for c in caminos[:25]],
                       "var": r(var_emp[1::2], 1), "var_100_250_500": r(var_emp[[99, 249, 499]], 1),
                       "acf_caminata": r(acf(caminos[0], nlags=60), 3), "acf_dif": r(acf(np.diff(caminos[0]), nlags=60), 3),
                       "banda": round(1.96 / np.sqrt(500), 3)}

    # ===== tendencia
    anual = anom.groupby(anom.index.year).mean()
    dec = (anual.index.to_numpy() + 0.5 - 1992) / 10
    ols = sm.OLS(anual.to_numpy(), sm.add_constant(dec)).fit()
    malla = np.linspace(anual.index.min() - 0.5, anual.index.max() + 0.5, 34)
    pr = ols.get_prediction(sm.add_constant((malla + 0.5 - 1992) / 10)).summary_frame(alpha=0.05)
    dec_d = ((y.index.year + (y.index.day_of_year - 1) / 365.25) - 1992) / 10
    Xd = sm.add_constant(dec_d.to_numpy())
    ols_d = sm.OLS(anom.to_numpy(), Xd).fit()
    hac_d = sm.OLS(anom.to_numpy(), Xd).fit(cov_type="HAC", cov_kwds={"maxlags": 365})
    lo, hi = ols.conf_int()[1]
    out["tendencia"] = {"anio": anual.index.tolist(), "anom": r(anual, 3), "malla": r(malla, 2), "recta": r(pr["mean"], 3),
                        "lo": r(pr.mean_ci_lower, 3), "hi": r(pr.mean_ci_upper, 3),
                        "b": round(float(ols.params[1]), 3), "ee": round(float(ols.bse[1]), 3),
                        "ic": [round(float(lo), 2), round(float(hi), 2)],
                        "ee_ols_diario": round(float(ols_d.bse[1]), 3), "ee_hac": round(float(hac_d.bse[1]), 3),
                        "b_diario": round(float(ols_d.params[1]), 3),
                        "t_ols": round(float(ols_d.params[1] / ols_d.bse[1]), 2), "t_hac": round(float(hac_d.params[1] / hac_d.bse[1]), 2)}

    # ===== pronósticos de referencia con origen móvil (re-ajuste cada 1 de enero)
    def ajustar_referencia(hasta):
        previo = y[: pd.Timestamp(hasta) - pd.Timedelta(days=1)]
        coef, *_ = np.linalg.lstsq(armonicos(previo.index), previo.to_numpy(), rcond=None)
        a = previo.to_numpy() - armonicos(previo.index) @ coef
        return coef, acf(a, nlags=1)[1]

    def pronosticos(h):
        filas = []
        for anio_ in range(2021, 2025):
            coef, phi = ajustar_referencia(f"{anio_}-01-01")
            dias_ = y[str(anio_)].index
            c = armonicos(dias_) @ coef
            origen = dias_ - pd.Timedelta(days=h)
            y_origen = y.reindex(origen).to_numpy()
            a_origen = y_origen - armonicos(origen) @ coef
            filas.append(pd.DataFrame({
                "real": y[str(anio_)].to_numpy(), "ingenuo": y_origen,
                "estacional ingenuo": y.reindex(dias_ - pd.Timedelta(days=365)).to_numpy(),
                "climatología": c, "AR(1) de anomalías": c + phi**h * a_origen}, index=dias_))
        return pd.concat(filas)

    metodos = ["ingenuo", "estacional ingenuo", "climatología", "AR(1) de anomalías"]
    v_tr = y[ENTRENA].to_numpy()
    q = np.mean(np.abs(v_tr[365:] - v_tr[:-365]))
    p1 = pronosticos(1)
    tabla = []
    for m in metodos:
        e = p1["real"] - p1[m]
        tabla.append({"metodo": m, "mae": round(float(np.mean(np.abs(e))), 2), "recm": round(float(np.sqrt(np.mean(e**2))), 2),
                      "mape_c": round(float(100 * np.mean(np.abs(e / p1["real"]))), 2),
                      "mape_k": round(float(100 * np.mean(np.abs(e / (p1["real"] + 273.15)))), 2),
                      "mase": round(float(np.mean(np.abs(e)) / q), 2)})
    _, phi21 = ajustar_referencia("2021-01-01")
    mae_h = {m: [] for m in metodos}
    for h in range(1, 31):
        p_h = pronosticos(h)
        for m in metodos:
            mae_h[m].append(float(np.mean(np.abs(p_h["real"] - p_h[m]))))
    out["referencias"] = {"tabla": tabla, "q": round(float(q), 2), "phi": round(float(phi21), 3), "h": list(range(1, 31)),
                          "mae_h": {m: r(v, 3) for m, v in mae_h.items()}, "n_prueba": int(len(p1))}

    # ejemplo desde un origen fijo
    origen = pd.Timestamp("2023-01-15")
    coef, phi = ajustar_referencia("2023-01-01")
    futuro = pd.date_range(origen + pd.Timedelta(days=1), periods=30)
    hs = np.arange(1, 31)
    a_T = y[origen] - (armonicos(pd.DatetimeIndex([origen])) @ coef)[0]
    c_fut = armonicos(futuro) @ coef
    pasado = y[origen - pd.Timedelta(days=45): origen]
    out["ejemplo"] = {"pasado": r(pasado, 1), "futuro": r(y[futuro], 1), "n_pasado": len(pasado),
                      "ingenuo": r(np.full(30, y[origen]), 2),
                      "estacional": r(y.reindex(futuro - pd.Timedelta(days=365)).to_numpy(), 2),
                      "clima": r(c_fut, 2), "ar1": r(c_fut + phi**hs * a_T, 2)}

    # ===== la fuga: KFold aleatorio contra origen móvil contra el futuro, h = 7
    H = 7
    X_rez = pd.DataFrame({
        "rezago_7": y.shift(H), "rezago_8": y.shift(H + 1),
        "media_7": y.shift(H).rolling(7).mean(), "media_30": y.shift(H).rolling(30).mean(),
        "cos_dia": np.cos(2 * np.pi * (y.index.day_of_year - 1) / 365.25),
        "sen_dia": np.sin(2 * np.pi * (y.index.day_of_year - 1) / 365.25),
        "anio": y.index.year}, index=y.index)
    tabla_rez = X_rez.assign(y=y).dropna()
    X_tr, y_tr_ = tabla_rez.loc[ENTRENA].drop(columns="y"), tabla_rez.loc[ENTRENA, "y"]
    X_te, y_te_ = tabla_rez.loc[PRUEBA].drop(columns="y"), tabla_rez.loc[PRUEBA, "y"]
    modelos = {"regresión lineal": LinearRegression(),
               "bosque aleatorio": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.5,
                                                         n_jobs=4, random_state=SEMILLA)}
    esquemas = {"kfold": KFold(5, shuffle=True, random_state=SEMILLA), "movil": TimeSeriesSplit(n_splits=5, test_size=365)}
    fuga = []
    for nombre, modelo in modelos.items():
        fila = {"modelo": nombre}
        for esq, cv in esquemas.items():
            fila[esq] = round(float(-cross_val_score(modelo, X_tr, y_tr_, cv=cv, scoring="neg_mean_absolute_error").mean()), 2)
        fila["futuro"] = round(float(np.mean(np.abs(modelo.fit(X_tr, y_tr_).predict(X_te) - y_te_))), 2)
        fuga.append(fila)
    out["fuga"] = {"tabla": fuga, "clima_h7": round(mae_h["climatología"][H - 1], 2),
                   "acf1": round(float(acf(a_tr, nlags=1)[1]), 2)}

    # ===== pliegos para el diagrama (120 días)
    n = 120
    idx = np.arange(n)
    kf = np.zeros(n, dtype=int)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=SEMILLA).split(idx)):
        kf[te] = k
    crec = [[int(a[0]), int(a[-1]), int(b[0]), int(b[-1])] for a, b in TimeSeriesSplit(5, test_size=15).split(idx)]
    desl = [[int(a[0]), int(a[-1]), int(b[0]), int(b[-1])] for a, b in TimeSeriesSplit(5, test_size=15, max_train_size=40, gap=3).split(idx)]
    out["pliegos"] = {"n": n, "kfold": kf.tolist(), "creciente": crec, "deslizante": desl}
    return out
