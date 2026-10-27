"""Datos de las láminas de u07-03-aprendizaje-series (pronosticar con aprendizaje automático).

Reproduce exactamente los cálculos de la libreta
unidades/07-series-tiempo/ejemplos/u07-e03-aprendizaje-series.py (mismos atributos,
hiperparámetros y semillas), y guarda solo lo que dibujan las láminas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
SEMILLA = 2012
VENTANA, H = 336, 24
HORIZONTES = np.arange(1, H + 1)
NOMBRES = (["y(t)"] + [f"y(t-{k})" for k in range(1, 24)] + ["y(t+h-24)", "y(t+h-168)", "media 24 h", "media 168 h",
           "hora", "día de la semana", "laboral", "feriado"])


def r(a, d: int = 1):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def datos() -> dict:
    import lightgbm as lgb
    import torch
    from torch import nn
    from sklearn.linear_model import LinearRegression

    torch.set_num_threads(1)  # como la libreta: evita el conflicto de OpenMP entre LightGBM y PyTorch
    out: dict = {}

    # ---------- datos (sección 1 de la libreta)
    b = pd.read_csv(RAIZ / "datos" / "bicis_washington.csv", parse_dates=["fecha"])
    b.index = b.fecha + pd.to_timedelta(b.hora, unit="h")
    horas = pd.date_range("2011-01-01 00:00", "2012-12-31 23:00", freq="h")
    faltan = horas.difference(b.index)
    por_dia = pd.Series(1, index=faltan).groupby(faltan.normalize()).sum()
    out["faltantes"] = {"n": int(len(faltan)), "total": int(len(horas)), "minimo": int(b.total.min()),
                        "por_hora": pd.Series(faltan.hour).value_counts().reindex(range(24), fill_value=0).tolist(),
                        "dias": [{"fecha": str(d.date()), "n": int(n)} for d, n in por_dia[por_dia >= 6].items()]}

    s = b.reindex(horas)
    s["total"] = s.total.fillna(0)
    s["hora"] = horas.hour
    calendario = b.groupby("fecha")[["dia_semana", "feriado", "laboral"]].first()
    for c in calendario:
        s[c] = calendario[c].reindex(horas.normalize()).to_numpy()
    s[["temp", "humedad", "clima"]] = s[["temp", "humedad", "clima"]].interpolate().ffill()
    y = s.total.to_numpy(dtype=float)
    hora, dsem = s.hora.to_numpy(), s.dia_semana.to_numpy().astype(int)
    laboral, feriado = s.laboral.to_numpy().astype(int), s.feriado.to_numpy().astype(int)
    temp, clima = s.temp.to_numpy(), s.clima.to_numpy()
    INICIO_PRUEBA = horas.get_loc(pd.Timestamp("2012-10-01 00:00"))

    diario = s.total.resample("D").sum()
    media = diario.rolling(28, center=True).mean()
    out["serie"] = {"inicio": "2011-01-01", "diario": [int(v) for v in diario.values],
                    "media": [None if np.isnan(v) else round(float(v), 0) for v in media.values],
                    "prueba": int((diario.index < "2012-10-01").sum()), "sandy": int((diario.index < "2012-10-29").sum()),
                    "media_prueba": round(float(y[INICIO_PRUEBA:].mean()), 0),
                    "semanas": {"inicio": "2012-09-10", "y": [int(v) for v in s.loc["2012-09-10":"2012-09-23", "total"]]}}
    # ventana para la lámina interactiva: 9 días, origen el domingo 16 de septiembre a las 20 h
    tramo = s.loc["2012-09-09":"2012-09-17", "total"]
    out["ventana"] = {"inicio": "2012-09-09", "y": [int(v) for v in tramo], "origen": 7 * 24 + 20}

    # ---------- atributos (sección 2)
    def ventanas(o):
        return y[o[:, None] + np.arange(-VENTANA + 1, 1)]

    def atributos(V, o, h, con_clima=False):
        t = o + h
        cols = [V[:, -1 - k] for k in range(24)]
        cols += [V[:, h - 25], V[:, h - 169]]
        cols += [V[:, -24:].mean(1), V[:, -168:].mean(1)]
        cols += [hora[t], dsem[t], laboral[t], feriado[t]]
        if con_clima:
            cols += [temp[t], clima[t]]
        return np.column_stack(cols)

    ULTIMO_ENTRENA = INICIO_PRUEBA - 1
    o_tr = np.arange(VENTANA - 1, ULTIMO_ENTRENA - H + 1)
    o_te = np.arange(ULTIMO_ENTRENA, len(y) - H)
    V_tr, V_te = ventanas(o_tr), ventanas(o_te)
    Y_tr, Y_te = y[o_tr[:, None] + HORIZONTES], y[o_te[:, None] + HORIZONTES]
    out["particion"] = {"n_tr": int(len(o_tr)), "n_te": int(len(o_te))}

    mae_por_h = lambda P: np.abs(P - Y_te).mean(axis=0)
    resumen = lambda P: {"h1": round(float(mae_por_h(P)[0]), 1), "h24": round(float(mae_por_h(P)[-1]), 1),
                         "mae": round(float(np.abs(P - Y_te).mean()), 1), "recm": round(float(np.sqrt(((P - Y_te) ** 2).mean())), 1)}
    P: dict = {}
    P["ingenuo: ayer"] = np.column_stack([V_te[:, h - 25] for h in HORIZONTES])
    P["ingenuo: semana pasada"] = np.column_stack([V_te[:, h - 169] for h in HORIZONTES])

    # ---------- origen móvil para el número de árboles (sección 4)
    PARAMETROS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=SEMILLA, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
    MALLA = np.arange(50, 801, 50)
    curvas = {}
    h = 24
    for mes in ["2012-07", "2012-08", "2012-09"]:
        inicio = horas.get_loc(pd.Timestamp(f"{mes}-01"))
        fin = horas.get_loc(pd.Timestamp(f"{mes}-01") + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23))
        entrena = o_tr + h < inicio
        valida = (o_tr + h >= inicio) & (o_tr + h <= fin)
        m = lgb.LGBMRegressor(n_estimators=MALLA[-1], **PARAMETROS)
        m.fit(atributos(V_tr[entrena], o_tr[entrena], h), Y_tr[entrena, h - 1])
        Xv = atributos(V_tr[valida], o_tr[valida], h)
        curvas[mes] = [np.abs(m.predict(Xv, num_iteration=k) - Y_tr[valida, h - 1]).mean() for k in MALLA]
    curvas = pd.DataFrame(curvas, index=MALLA)
    ARBOLES = int(curvas.mean(axis=1).idxmin())
    out["cv"] = {"arboles": MALLA.tolist(), "julio": r(curvas["2012-07"]), "agosto": r(curvas["2012-08"]),
                 "septiembre": r(curvas["2012-09"]), "promedio": r(curvas.mean(axis=1)), "elegido": ARBOLES,
                 "mae_elegido": round(float(curvas.mean(axis=1).min()), 1)}

    # ---------- modelos directos
    def lineal_atributos(X, o, h):
        t = o + h
        return np.column_stack([X[:, :28], np.eye(48)[2 * hora[t] + laboral[t]], feriado[t]])

    directos, P_lin, P_dir = {}, [], []
    for h in HORIZONTES:
        Xa, Xb = atributos(V_tr, o_tr, h), atributos(V_te, o_te, h)
        P_lin.append(LinearRegression().fit(lineal_atributos(Xa, o_tr, h), Y_tr[:, h - 1]).predict(lineal_atributos(Xb, o_te, h)))
        directos[h] = lgb.LGBMRegressor(n_estimators=ARBOLES, **PARAMETROS).fit(Xa, Y_tr[:, h - 1])
        P_dir.append(directos[h].predict(Xb))
    P["lineal directa"] = np.clip(np.column_stack(P_lin), 0, None)
    P["LightGBM directa"] = np.clip(np.column_stack(P_dir), 0, None)
    imp = {}
    for h in (1, 24):
        g = pd.Series(directos[h].booster_.feature_importance("gain"), index=NOMBRES)
        g = (g / g.sum()).sort_values(ascending=False).head(6)
        imp[f"h{h}"] = [{"nombre": k, "valor": round(float(v), 3)} for k, v in g.items()]
    out["importancia"] = imp

    # ---------- fuga: clima observado en t+h (sección 5)
    fuga = []
    for h in [1, 6, 12, 24]:
        m = lgb.LGBMRegressor(n_estimators=ARBOLES, **PARAMETROS)
        m.fit(atributos(V_tr, o_tr, h, con_clima=True), Y_tr[:, h - 1])
        p = np.clip(m.predict(atributos(V_te, o_te, h, con_clima=True)), 0, None)
        sin, con = float(mae_por_h(P["LightGBM directa"])[h - 1]), float(np.abs(p - Y_te[:, h - 1]).mean())
        fuga.append({"h": h, "sin": round(sin, 1), "con": round(con, 1), "mejora": round(100 * (1 - con / sin), 1)})
    out["fuga"] = fuga

    # ---------- recursiva (sección 6)
    P_rec, V = [], V_te.copy()
    for j in range(H):
        p = np.clip(directos[1].predict(atributos(V, o_te + j, 1)), 0, None)
        P_rec.append(p)
        V = np.column_stack([V[:, 1:], p])
    P["LightGBM recursiva"] = np.column_stack(P_rec)

    # ---------- red multisalida
    def entradas_red(V, o):
        semana = np.log1p(V[:, -168:]) / np.log1p(1000)
        return np.hstack([semana, np.eye(24)[hora[o]], np.eye(7)[dsem[o]], laboral[o[:, None] + HORIZONTES]]).astype(np.float32)

    ESCALA = 100.0
    torch.manual_seed(SEMILLA)
    red = nn.Sequential(nn.Linear(168 + 24 + 7 + 24, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, H))
    X_red = torch.tensor(entradas_red(V_tr, o_tr))
    Y_red = torch.tensor(Y_tr / ESCALA, dtype=torch.float32)
    opt = torch.optim.Adam(red.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(SEMILLA)
    for _ in range(60):
        orden = torch.randperm(len(X_red), generator=gen)
        for i in range(0, len(orden), 128):
            lote = orden[i:i + 128]
            perdida = ((red(X_red[lote]) - Y_red[lote]) ** 2).mean()
            opt.zero_grad()
            perdida.backward()
            opt.step()
    with torch.no_grad():
        P["red multisalida"] = np.clip(ESCALA * red(torch.tensor(entradas_red(V_te, o_te))).numpy(), 0, None)

    # trayectorias de 24 horas desde un origen: recursiva, directa y red, contra lo observado
    tray = []
    for f in ["2012-11-13 06:00", "2012-11-14 06:00", "2012-12-04 06:00", "2012-12-05 06:00", "2012-11-27 06:00", "2012-11-28 06:00"]:
        i = int(np.flatnonzero(horas[o_te] == pd.Timestamp(f))[0])
        o = o_te[i]
        tray.append({"origen": f, "antes": [int(v) for v in y[o - 23:o + 1]], "real": [int(v) for v in Y_te[i]],
                     "recursiva": r(P["LightGBM recursiva"][i]), "directa": r(P["LightGBM directa"][i]), "red": r(P["red multisalida"][i]),
                     "err": {k: round(float(np.abs(P[k][i] - Y_te[i]).mean()), 1) for k in ("LightGBM recursiva", "LightGBM directa", "red multisalida")}})
    out["trayectorias"] = tray

    out["tabla"] = [{"modelo": k, **resumen(v)} for k, v in P.items()]
    out["horizonte"] = {"h": HORIZONTES.tolist(), **{k: r(mae_por_h(v)) for k, v in P.items()}}

    # ---------- pérdida cuantil (sección 7)
    ocho = y[:INICIO_PRUEBA][(hora[:INICIO_PRUEBA] == 8) & (laboral[:INICIO_PRUEBA] == 1)]
    fina = np.arange(0, 1001, 1.0)
    gruesa = np.arange(0, 1001, 10.0)
    taus = np.round(np.arange(0.05, 0.951, 0.05), 2)
    def perdida_media(tau, q):
        u = ocho[None, :] - q[:, None]
        return (u * (tau - (u < 0))).mean(axis=1)
    out["pinball"] = {"n": int(len(ocho)), "hist": np.histogram(ocho, bins=np.arange(0, 1001, 40))[0].tolist(), "ancho_hist": 40, "q": gruesa.tolist(), "tau": taus.tolist(),
                      "curvas": [r(perdida_media(t, gruesa)) for t in taus],
                      "minimo": [float(fina[perdida_media(t, fina).argmin()]) for t in taus],
                      "cuantil": [round(float(np.quantile(ocho, t)), 0) for t in taus],
                      "valor_min": [round(float(perdida_media(t, fina).min()), 1) for t in taus]}

    cuantiles, cob = {}, []
    for h in [1, 6, 12, 24]:
        Xa, Xb = atributos(V_tr, o_tr, h), atributos(V_te, o_te, h)
        for tau in [0.1, 0.9]:
            m = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=ARBOLES, **PARAMETROS)
            cuantiles[h, tau] = np.clip(m.fit(Xa, Y_tr[:, h - 1]).predict(Xb), 0, None)
        bajo, alto, real = cuantiles[h, 0.1], cuantiles[h, 0.9], Y_te[:, h - 1]
        cob.append({"h": h, "cobertura": round(100 * float(np.mean((real >= bajo) & (real <= alto))), 1),
                    "debajo": round(100 * float(np.mean(real < bajo)), 1), "arriba": round(100 * float(np.mean(real > alto)), 1),
                    "ancho": round(float(np.mean(alto - bajo)), 0)})
    out["cobertura"] = cob

    def semana(desde, hasta):
        idx = horas[o_te + 24]
        sel = lambda v: pd.Series(v, index=idx).loc[desde:hasta]
        return {"inicio": desde, "real": [int(v) for v in s.total.loc[desde:hasta]],
                "pred": r(sel(P["LightGBM directa"][:, 23])), "q10": r(sel(cuantiles[24, 0.1])), "q90": r(sel(cuantiles[24, 0.9]))}
    out["noviembre"] = semana("2012-11-12", "2012-11-18 23:00")
    out["sandy"] = semana("2012-10-26", "2012-11-01 23:00")
    sandy = (horas[o_te + 24] >= "2012-10-29") & (horas[o_te + 24] < "2012-10-31")
    err = np.abs(P["LightGBM directa"][:, 23] - Y_te[:, 23])
    out["sandy"] |= {"mae": round(float(err.mean()), 1), "mae_sin": round(float(err[~sandy].mean()), 1),
                     "fraccion": round(100 * float(err[sandy].sum() / err.sum()), 1)}
    return out


if __name__ == "__main__":
    import json
    d = datos()
    print(json.dumps({k: d[k] for k in ("tabla", "fuga", "cobertura", "cv", "importancia")}, ensure_ascii=False, indent=1)[:4000])
