"""Datos de las láminas de u07-02-arima (Modelos ARIMA).

Mismo código y semillas que la libreta unidades/07-series-tiempo/ejemplos/u07-e02-arima.py:
las cifras de las láminas deben coincidir con las de la libreta.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd


def r(a, d: int = 4):
    """Redondea a `d` decimales y convierte a listas de Python (como _comun.r)."""
    return np.round(np.asarray(a, dtype=float), d).tolist()


SEMILLA = 1958


def simular_arma(phi=(), theta=(), n=500, sigma=1.0, rng=None, quemado=300):
    """Igual que en la libreta: y_t = Σ φ_i y_{t-i} + ε_t + Σ θ_j ε_{t-j}."""
    phi, theta = np.atleast_1d(phi), np.atleast_1d(theta)
    e = rng.normal(0, sigma, n + quemado)
    y = np.zeros(n + quemado)
    for t in range(n + quemado):
        ar = sum(phi[i] * y[t - 1 - i] for i in range(len(phi)) if t - 1 - i >= 0)
        ma = sum(theta[j] * e[t - 1 - j] for j in range(len(theta)) if t - 1 - j >= 0)
        y[t] = ar + e[t] + ma
    return y[quemado:]


def datos() -> dict:
    import rdp
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import acf, pacf

    warnings.filterwarnings("ignore", module="statsmodels")
    out: dict = {}

    # ---- datos
    co2 = rdp.datos.cargar("co2_mauna_loa")
    co2.index = pd.period_range(f"{co2.anio[0]}-{co2.mes[0]:02d}", periods=len(co2), freq="M")
    y = co2["co2"]
    entrena, prueba = y[:"2014-12"], y["2015-01":]
    n_ent = len(entrena)
    fx = co2.fecha_decimal.to_numpy()
    anual = y.groupby(y.index.year).mean()
    completo = y.groupby(y.index.year).size() == 12
    crec = anual.diff()[completo & completo.shift(1, fill_value=False)]
    out["keeling"] = {
        "t": r(fx, 3), "y": r(y, 2), "n_ent": n_ent,
        "primero": float(y.iloc[0]), "maximo": float(y.max()), "mes_maximo": str(y.idxmax()),
        "ultimo": float(y.iloc[-1]), "mes_ultimo": str(y.index[-1]),
        "crec_decada": {str(k): v for k, v in crec.groupby(crec.index // 10 * 10).mean().round(2).items()},
    }

    # ---- procesos simulados (mismo orden de uso del generador que la libreta)
    rng = np.random.default_rng(SEMILLA)
    T, K = 500, 20
    procesos = {
        "ruido": rng.normal(0, 1, T),
        "ar09": simular_arma(phi=0.9, rng=rng),
        "ar07": simular_arma(phi=-0.7, rng=rng),
        "ma08": simular_arma(theta=0.8, rng=rng),
        "caminata": np.cumsum(rng.normal(0, 1, T)),
    }
    k = np.arange(1, K + 1)
    teoricas = {"ruido": 0 * k, "ar09": 0.9**k, "ar07": (-0.7) ** k,
                "ma08": np.where(k == 1, 0.8 / 1.64, 0.0), "caminata": None}
    out["procesos"] = {
        nombre: {"serie": r(s[:200], 3), "acf": r(acf(s, nlags=K)[1:], 3), "pacf": r(pacf(s, nlags=K)[1:], 3),
                 "teorica": None if teoricas[nombre] is None else r(teoricas[nombre], 3)}
        for nombre, s in procesos.items()
    }
    out["banda"] = round(1.96 / np.sqrt(T), 4)

    # comprobación del AR(1) (consume el generador como la libreta)
    filas = []
    for phi in (0.5, 0.9, -0.7):
        s = simular_arma(phi=phi, n=50_000, rng=rng)
        rr = acf(s, nlags=3)
        filas.append({"phi": phi, "var": round(float(s.var()), 3), "teo": round(1 / (1 - phi**2), 3), "rho1": round(float(rr[1]), 3)})
    out["ar1"] = filas

    # AR(2) por mínimos cuadrados
    phi_verd, c_verd = np.array([0.75, -0.5]), 2.0
    s = simular_arma(phi=phi_verd, n=1_000, rng=rng) + c_verd / (1 - phi_verd.sum())
    Tn = len(s)
    X = np.column_stack([np.ones(Tn - 2), s[1:Tn - 1], s[0:Tn - 2]])
    w, *_ = np.linalg.lstsq(X, s[2:], rcond=None)
    ar = AutoReg(s, lags=2, trend="c").fit()
    raiz = np.roots([-phi_verd[1], -phi_verd[0], 1])[0]
    out["ar2"] = {"mc": r(w, 4), "autoreg": r(ar.params, 4), "raiz": [round(float(raiz.real), 2), round(float(abs(raiz.imag)), 2)],
                  "modulo": round(float(abs(raiz)), 3), "serie": r(s[:120], 3), "acf": r(acf(s, nlags=K)[1:], 3)}

    # error de pronóstico contra horizonte: teoría y simulación
    H, replicas, y_T = 24, 20_000, 2.0
    hs = np.arange(1, H + 1)
    horiz = {}
    for nombre, phi in (("ar05", 0.5), ("ar09", 0.9), ("caminata", 1.0)):
        tray = np.empty((replicas, H))
        actual = np.full(replicas, y_T)
        for h in range(H):
            actual = phi * actual + rng.normal(0, 1, replicas)
            tray[:, h] = actual
        err = tray - (phi**hs) * y_T
        teo = hs if phi == 1 else (1 - phi ** (2 * hs)) / (1 - phi**2)
        horiz[nombre] = {"teoria": r(np.sqrt(teo), 3), "simulacion": r(err.std(axis=0), 3)}
    out["horizonte"] = horiz

    # ---- diferencias del CO2
    versiones = {"y": entrena, "d1": entrena.diff(), "d12": entrena.diff(12), "d1d12": entrena.diff(12).diff()}
    out["diferencias"] = {}
    for nombre, serie in versiones.items():
        serie = serie.dropna()
        out["diferencias"][nombre] = {
            "t": r(serie.index.year + (serie.index.month - 0.5) / 12, 3), "y": r(serie, 3),
            "acf": r(acf(serie, nlags=48)[1:], 3), "banda": round(1.96 / np.sqrt(len(serie)), 4),
        }

    # ---- malla de SARIMA y AIC
    malla = []
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
        aj = SARIMAX(entrena, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
        malla.append({"modelo": f"({p},1,{q})({P},1,{Q})", "k": p + q + P + Q + 1, "aic": round(float(aj.aic), 2)})
    malla.sort(key=lambda d: d["aic"])
    out["aic"] = malla

    modelo = SARIMAX(entrena, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    out["parametros"] = {k: round(float(v), 4) for k, v in modelo.params.items()}
    out["errores_est"] = {k: round(float(v), 4) for k, v in modelo.bse.items()}
    residuos = modelo.resid[13:]
    lb = acorr_ljungbox(residuos, lags=[12, 24, 36], model_df=3)
    out["residuos"] = {"t": r(residuos.index.year + (residuos.index.month - 0.5) / 12, 3), "y": r(residuos, 3),
                       "acf": r(acf(residuos, nlags=36)[1:], 3), "banda": round(1.96 / np.sqrt(len(residuos)), 4),
                       "lb": [{"K": int(k), "Q": round(float(q), 2), "p": round(float(pv), 3)} for k, q, pv in zip(lb.index, lb.lb_stat, lb.lb_pvalue)]}

    # ---- pronóstico de largo plazo
    pron = modelo.get_forecast(len(prueba))
    media = pron.predicted_mean
    i80, i95 = pron.conf_int(alpha=0.20).to_numpy(), pron.conf_int(alpha=0.05).to_numpy()
    t = np.arange(len(y), dtype=float)
    meses = pd.get_dummies(co2.mes).to_numpy(dtype=float)
    X_lin = np.column_stack([t, meses])
    X_cuad = np.column_stack([t, (t / 100) ** 2, meses])

    def regresion(X, hasta):
        w, *_ = np.linalg.lstsq(X[:hasta], y.to_numpy()[:hasta], rcond=None)
        return X @ w

    pronos = {
        "sarima": media.to_numpy(),
        "ingenuo": np.resize(entrena.to_numpy()[-12:], len(prueba)),
        "lineal": regresion(X_lin, n_ent)[n_ent:],
        "cuadratica": regresion(X_cuad, n_ent)[n_ent:],
    }
    escala = float(np.mean(np.abs(entrena.to_numpy()[12:] - entrena.to_numpy()[:-12])))
    obs = prueba.to_numpy()
    desde = n_ent - 60
    out["pronostico"] = {
        "t": r(fx[desde:], 3), "obs": r(y.to_numpy()[desde:], 2), "inicio": n_ent - desde,
        "media": r(media, 2), "lo80": r(i80[:, 0], 2), "hi80": r(i80[:, 1], 2), "lo95": r(i95[:, 0], 2), "hi95": r(i95[:, 1], 2),
        "otros": {k: r(v, 2) for k, v in pronos.items() if k != "sarima"},
        "metricas": {k: {"mae": round(float(np.mean(np.abs(obs - v))), 2), "recm": round(float(np.sqrt(np.mean((obs - v) ** 2))), 2),
                         "mase": round(float(np.mean(np.abs(obs - v)) / escala), 2)} for k, v in pronos.items()},
        "escala": round(escala, 3),
        "cobertura80": round(float(np.mean((obs >= i80[:, 0]) & (obs <= i80[:, 1]))), 3),
        "cobertura95": round(float(np.mean((obs >= i95[:, 0]) & (obs <= i95[:, 1]))), 3),
        "semiancho": [round(float((i95[h, 1] - i95[h, 0]) / 2), 2) for h in (0, 11, len(prueba) - 1)],
    }

    # ---- origen móvil a 12 meses
    H = 12
    yv = y.to_numpy()
    err = {k: [] for k in pronos}
    c80, c95 = [], []
    for o in range(n_ent, len(y) - H + 1):
        real = yv[o:o + H]
        p = modelo.apply(y.iloc[:o]).get_forecast(H)
        err["sarima"].append(real - p.predicted_mean.to_numpy())
        lo, hi = p.conf_int(alpha=0.20).to_numpy().T
        c80.append((real >= lo) & (real <= hi))
        lo, hi = p.conf_int(alpha=0.05).to_numpy().T
        c95.append((real >= lo) & (real <= hi))
        err["ingenuo"].append(real - yv[o - 12:o - 12 + H])
        err["lineal"].append(real - regresion(X_lin, o)[o:o + H])
        err["cuadratica"].append(real - regresion(X_cuad, o)[o:o + H])
    out["movil"] = {
        "n_origenes": len(c80),
        "mae": {k: r(np.abs(np.array(v)).mean(axis=0), 3) for k, v in err.items()},
        "mae_prom": {k: round(float(np.abs(np.array(v)).mean()), 2) for k, v in err.items()},
        "cobertura80": r(np.mean(c80, axis=0), 3), "cobertura95": r(np.mean(c95, axis=0), 3),
        "cob80_prom": round(float(np.mean(c80)), 3), "cob95_prom": round(float(np.mean(c95)), 3),
    }
    return out


if __name__ == "__main__":
    import json

    d = datos()
    print(json.dumps({k: d[k] for k in ("ar1", "ar2", "parametros", "residuos")}, default=str)[:1500])
    print(d["aic"][:6], d["pronostico"]["metricas"], d["movil"]["mae_prom"])
