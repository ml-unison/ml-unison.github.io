"""Datos de las láminas de u08-02-despliegue-monitoreo (Unidad 8 · Despliegue y monitoreo).

Reproduce, con el mismo código y las mismas semillas, las cifras de la libreta
unidades/08-mlops/ejemplos/u08-e02-despliegue-monitoreo.py: un modelo de rentas
de bicicletas entrenado con 2011 y vigilado durante 2012 (PSI, KS, error mensual
y políticas de reentrenamiento).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

from _comun import r

SEMILLA = 2011
ATRIBUTOS = ["hora", "laboral", "dia_semana", "clima", "temp", "humedad", "viento"]


def nuevo_modelo():
    return HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.1, random_state=SEMILLA)


def sesgo(y, pred):
    return np.sum(pred) / np.sum(y) - 1


def psi(ref, act, k=10, eps=1e-4):
    ref, act = np.asarray(ref, float), np.asarray(act, float)
    valores = np.unique(ref)
    if len(valores) <= k:
        cortes = (valores[:-1] + valores[1:]) / 2
    else:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, k + 1)[1:-1]))
    p = np.bincount(np.searchsorted(cortes, ref, side="right"), minlength=len(cortes) + 1) / len(ref)
    q = np.bincount(np.searchsorted(cortes, act, side="right"), minlength=len(cortes) + 1) / len(act)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks(a, c):
    a, c = np.sort(a), np.sort(c)
    x = np.concatenate([a, c])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(c, x, side="right") / len(c))))


def datos() -> dict:
    import rdp

    b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
    b["clima"] = b.clima.clip(upper=3)
    b["anio"] = b.fecha.dt.year
    b["periodo"] = (b.anio - 2011) * 12 + b.mes
    b11, b12 = b[b.anio == 2011].copy(), b[b.anio == 2012].copy()
    out: dict = {}

    # Rentas por mes (promedio por hora)
    mensual = b.groupby(["anio", "mes"]).total.mean().unstack(0)
    out["mensual"] = {"y2011": r(mensual[2011], 1), "y2012": r(mensual[2012], 1),
                      "crecimiento": round(b12.total.sum() / b11.total.sum() - 1, 3)}

    # Validación y modelo de 2011
    ent, val = b11[b11.fecha.dt.day <= 19], b11[b11.fecha.dt.day >= 20]
    pred_val = nuevo_modelo().fit(ent[ATRIBUTOS], ent.total).predict(val[ATRIBUTOS])
    recm_val = root_mean_squared_error(val.total, pred_val)
    modelo = nuevo_modelo().fit(b11[ATRIBUTOS], b11.total)
    b11["prediccion"] = modelo.predict(b11[ATRIBUTOS])
    b12["prediccion"] = modelo.predict(b12[ATRIBUTOS])
    out["validacion"] = {"recm": round(recm_val, 1), "sesgo": round(100 * sesgo(val.total, pred_val), 1)}

    # Las tres maneras de cambiar
    bt = np.arange(-2, 36, 2)
    by = np.arange(0, 1000, 50)
    bins_t = np.arange(-6, 42, 3)
    curvas = {}
    for anio, d in [(2011, b11), (2012, b12)]:
        g = d.groupby(pd.cut(d.temp, bins_t), observed=True).total.agg(["mean", "size"])
        g = g[g["size"] >= 50]
        curvas[str(anio)] = {"x": r([i.mid for i in g.index], 1), "y": r(g["mean"], 1)}
    out["cambios"] = {
        "temp": {"bordes": r(bt, 0), "a": r(np.histogram(b11[b11.mes == 3].temp, bt)[0] / (b11.mes == 3).sum()),
                 "b": r(np.histogram(b12[b12.mes == 3].temp, bt)[0] / (b12.mes == 3).sum()),
                 "media_a": round(b11[b11.mes == 3].temp.mean(), 1), "media_b": round(b12[b12.mes == 3].temp.mean(), 1)},
        "curva": curvas,
        "rentas": {"bordes": r(by, 0), "a": r(np.histogram(b11[b11.mes == 6].total, by)[0] / (b11.mes == 6).sum()),
                   "b": r(np.histogram(b12[b12.mes == 6].total, by)[0] / (b12.mes == 6).sum())},
    }

    # Piso de ruido del PSI y KS contra tamaño de muestra (mismo orden de números aleatorios que la libreta)
    rng = np.random.default_rng(SEMILLA)
    temps = b11.temp.to_numpy() + rng.normal(0, 0.05, len(b11))
    piso = []
    for m in [24, 72, 168, 720, 2000, 8000]:
        sim = np.array([psi(rng.choice(temps, m), rng.choice(temps, m)) for _ in range(300)])
        piso.append({"m": m, "psi": round(sim.mean(), 4), "formula": round(18 / m, 4), "alarmas": round(100 * np.mean(sim > 0.1), 1)})
    out["piso"] = piso
    meses = range(1, 13)
    out["psi_temp"] = {
        "ingenuo2011": r([psi(b11.temp, b11[b11.mes == m].temp) for m in meses], 2),
        "ingenuo2012": r([psi(b11.temp, b12[b12.mes == m].temp) for m in meses], 2),
        "mismo_mes": r([psi(b11[b11.mes == m].temp, b12[b12.mes == m].temp) for m in meses], 2),
    }
    a, c = b11[b11.mes == 3].temp, b12[b12.mes == 3].temp
    rejilla = np.arange(0, 32.01, 0.25)
    Fa = np.searchsorted(np.sort(a), rejilla, side="right") / len(a)
    Fc = np.searchsorted(np.sort(c), rejilla, side="right") / len(c)
    j = int(np.argmax(np.abs(Fa - Fc)))
    out["ks_marzo"] = {"D": round(ks(a, c), 3), "p": f"{ks_2samp(a, c).pvalue:.0e}", "x": r(rejilla, 2),
                       "F2011": r(Fa, 4), "F2012": r(Fc, 4), "xmax": float(rejilla[j]), "n2011": int(len(a)), "n2012": int(len(c))}
    filas = []
    for n in [100, 300, 1000, 3000, 10_000, 30_000, 100_000]:
        res = [ks_2samp(rng.choice(temps, n), rng.choice(temps, n) + 0.5) for _ in range(20)]
        filas.append({"n": n, "D": round(float(np.median([x.statistic for x in res])), 4),
                      "log10p": round(float(np.log10(max(np.median([x.pvalue for x in res]), 1e-300))), 1),
                      "umbral": round(1.358 * np.sqrt(2 / n), 4)})
    out["ks_n"] = filas

    # Tablero de PSI (mismo mes de 2011) y error mensual
    filas = {"temperatura": "temp", "humedad": "humedad", "viento": "viento", "clima": "clima",
             "predicción": "prediccion", "rentas observadas": "total"}
    out["tablero"] = [{"fila": k, "psi": r([psi(b11[b11.mes == m][v], b12[b12.mes == m][v]) for m in meses], 2)} for k, v in filas.items()]
    err = b12.groupby("mes")[["total", "prediccion"]].apply(
        lambda d: pd.Series({"recm": root_mean_squared_error(d.total, d.prediccion), "sesgo": 100 * sesgo(d.total, d.prediccion)}))
    factor = b12.total.sum() / b12.prediccion.sum()
    out["error"] = {"recm": r(err.recm, 1), "sesgo": r(err.sesgo, 1),
                    "recm_2012": round(root_mean_squared_error(b12.total, b12.prediccion), 1),
                    "sesgo_2012": round(100 * sesgo(b12.total, b12.prediccion), 1),
                    "factor": round(factor, 2),
                    "recm_escalado": round(root_mean_squared_error(b12.total, factor * b12.prediccion), 1)}

    # Políticas de reentrenamiento
    def reentrenado(periodo, ventana, retraso):
        ultimo = periodo - 1 - retraso
        d = b[(b.periodo <= ultimo) & (b.periodo > ultimo - ventana)]
        return nuevo_modelo().fit(d[ATRIBUTOS], d.total)

    pol = {"sin reentrenar": err.recm.to_numpy()}
    for nombre, (ventana, retraso) in {"ventana de 12 meses": (12, 0), "ventana de 3 meses": (3, 0),
                                       "3 meses, etiquetas 3 meses tarde": (3, 3)}.items():
        pol[nombre] = np.array([root_mean_squared_error(b12[b12.mes == m].total,
                                                        reentrenado(12 + m, ventana, retraso).predict(b12[b12.mes == m][ATRIBUTOS]))
                                for m in meses])
    out["politicas"] = [{"nombre": k, "recm": r(v, 1), "promedio": round(float(v.mean()), 1)} for k, v in pol.items()]
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(datos(), ensure_ascii=False)[:3000])
