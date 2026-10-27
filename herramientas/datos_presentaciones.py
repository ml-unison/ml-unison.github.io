"""Calcula con Python los datos de las figuras de las presentaciones.

Las presentaciones dibujan las gráficas en el navegador (con la tipografía y la
escala de la lámina), pero los números salen de aquí, con el mismo NumPy que las
libretas. Cada unidad escribe `presentaciones/datos/uNN.json`.

    uv run python herramientas/datos_presentaciones.py
"""

from __future__ import annotations

import json
import sys
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
    }, "digitos": digitos(), "presas": presas(), "casas": casas()}


def presas() -> dict:
    """Almacenamiento de las presas de Sonora (datos.sonora.gob.mx), resumido para las láminas de pandas."""
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


def casas() -> dict:
    """Casas de Ames (De Cock, 2011) para la presentación de atributos y pipelines.

    Usa la misma separación y los mismos resultados que la libreta u01-e02.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    d = pd.read_csv(DESTINO.parents[1] / "datos" / "casas_ames.csv")
    tr, _ = train_test_split(d, test_size=0.2, random_state=2011)
    precio = tr["SalePrice"]
    h1, b1 = np.histogram(precio / 1000, bins=np.arange(0, 780, 20))
    h2, b2 = np.histogram(np.log10(precio), bins=np.arange(4.5, 5.95, 0.04))
    orden = ["Fa", "TA", "Gd", "Ex"]
    cocina = tr.groupby("KitchenQual")["SalePrice"].median().reindex(orden) / 1000
    dos = tr[["GrLivArea", "OverallQual", "SalePrice"]].iloc[:3]
    return {
        "hist_precio": {"cuentas": h1.tolist(), "bordes": r(b1, 2)},
        "hist_log": {"cuentas": h2.tolist(), "bordes": r(b2, 3)},
        "cocina": {"niveles": orden, "mediana": r(cocina, 1)},
        "dispersion": {"area": tr["GrLivArea"].tolist(), "precio": r(precio / 1000, 1)},
        "ejemplo_distancia": dos.to_dict(orient="records"),
        "area_media": round(float(tr["GrLivArea"].mean()), 1), "area_sd": round(float(tr["GrLivArea"].std()), 1),
        "calidad_media": round(float(tr["OverallQual"].mean()), 2), "calidad_sd": round(float(tr["OverallQual"].std()), 2),
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


# ---------------------------------------------------------------- Unidad 2

def _rfm() -> "pd.DataFrame":
    """Tabla RFM (recencia, frecuencia, monto) de los clientes de la tienda en línea."""
    import pandas as pd

    f = pd.read_csv(DESTINO.parents[1] / "datos" / "tienda_facturas.csv", parse_dates=["fecha"])
    f = f.dropna(subset=["cliente"])
    corte = f.fecha.max().normalize() + pd.Timedelta(days=1)
    compras = f[~f.factura.astype(str).str.startswith("C")]
    t = compras.groupby("cliente").agg(ultima=("fecha", "max"), frecuencia=("factura", "nunique"))
    t["recencia"] = (corte - t.ultima).dt.days
    t["monto"] = f.groupby("cliente").importe.sum()
    return t[t.monto > 0][["recencia", "frecuencia", "monto"]]


def _elipse(mu, S, k=2.0, n=48):
    """Contorno de la elipse de k desviaciones de una normal bivariada."""
    val, vec = np.linalg.eigh(np.asarray(S))
    t = np.linspace(0, 2 * np.pi, n)
    pts = (vec @ (k * np.sqrt(np.maximum(val, 1e-12))[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T + mu
    return r(pts, 3)


SEMILLA_EM = 1


def u02() -> dict:
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.datasets import load_digits, make_blobs, make_moons
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    raiz = DESTINO.parents[1] / "datos"
    out: dict = {}

    # --- pingüinos
    pg = pd.read_csv(raiz / "pinguinos.csv").dropna(subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
    especies = ["Adelie", "Chinstrap", "Gentoo"]
    esp = pg.species.map({e: i for i, e in enumerate(especies)}).to_numpy()
    out["pinguinos"] = {"largo": r(pg.bill_length_mm, 1), "prof": r(pg.bill_depth_mm, 1),
                        "aleta": r(pg.flipper_length_mm, 0), "masa": r(pg.body_mass_g, 0),
                        "especie": esp.tolist(), "nombres": especies}

    # --- clientes: Lloyd paso a paso (muestra de 400, dos variables en log, estandarizadas)
    from sklearn.datasets import make_blobs
    rfm = _rfm()
    lrfm = pd.DataFrame({"recencia": np.log1p(rfm.recencia), "frecuencia": np.log(rfm.frecuencia), "monto": np.log(rfm.monto)})
    Z2 = StandardScaler().fit_transform(lrfm[["frecuencia", "monto"]])
    rng = np.random.default_rng(3)
    m = Z2[rng.choice(len(Z2), 400, replace=False)]
    C = m[[5, 11, 17, 23]].copy()                     # inicio al azar (cuatro clientes)
    C = np.array([[-1.2, -1.6], [-0.9, -1.0], [-0.6, -1.5], [2.2, 2.0]])  # inicio malo, a propósito
    pasos = []
    for _ in range(30):
        d = ((m[:, None, :] - C[None]) ** 2).sum(-1)
        e = d.argmin(1)
        J = float(d[np.arange(len(m)), e].sum())
        pasos.append({"C": r(C, 3), "e": e.tolist(), "J": round(J, 1)})
        Cn = np.array([m[e == j].mean(0) if (e == j).any() else C[j] for j in range(4)])
        if np.allclose(Cn, C):
            break
        C = Cn
    out["lloyd"] = {"x": r(m[:, 0], 3), "y": r(m[:, 1], 3), "pasos": pasos}

    # --- inercias de muchas inicializaciones, k y silueta (RFM completa, log + estandarizar)
    Z3 = StandardScaler().fit_transform(lrfm)
    # mínimos locales: nueve grupos en rejilla, k = 9, una sola inicialización cada vez
    Xr, _ = make_blobs(n_samples=450, centers=[[i, j] for i in range(3) for j in range(3)], cluster_std=.12, random_state=0)
    corr = [KMeans(9, init="random", n_init=1, random_state=s).fit(Xr) for s in range(60)]
    inits = [float(c.inertia_) for c in corr]
    pp = [float(KMeans(9, init="k-means++", n_init=1, random_state=s).fit(Xr).inertia_) for s in range(60)]
    peor = corr[int(np.argmax(inits))]
    out["rejilla"] = {"x": r(Xr[:, 0], 3), "y": r(Xr[:, 1], 3), "e_peor": peor.labels_.tolist(), "C_peor": r(peor.cluster_centers_, 3),
                      "e_mejor": corr[int(np.argmin(inits))].labels_.tolist(), "C_mejor": r(corr[int(np.argmin(inits))].cluster_centers_, 3)}
    ks = list(range(2, 9))
    km = {k: KMeans(k, n_init=10, random_state=0).fit(Z3) for k in ks}
    out["clientes"] = {
        "n": int(len(rfm)), "inits_azar": r(inits, 1), "inits_pp": r(pp, 1),
        "k": ks, "inercia": r([km[k].inertia_ for k in ks], 1),
        "silueta": r([silhouette_score(Z3, km[k].labels_) for k in ks], 3),
    }

    # --- dónde falla k-medias y qué hace DBSCAN
    Xa, _ = make_blobs(n_samples=[150, 150, 150], centers=[[0, 0], [0, 3.2], [0, 6.4]], cluster_std=[[2.6, .45]] * 3, random_state=4)
    Xb, _ = make_moons(300, noise=.07, random_state=1)
    Xc, _ = make_blobs(n_samples=[400, 40], centers=[[0, 0], [4.2, 0]], cluster_std=[1.6, .35], random_state=2)
    fallas = []
    for X, k, eps in [(Xa, 3, .55), (Xb, 2, .2), (Xc, 2, .45)]:
        fallas.append({"x": r(X[:, 0], 3), "y": r(X[:, 1], 3),
                       "km": KMeans(k, n_init=10, random_state=0).fit_predict(X).tolist(),
                       "db": DBSCAN(eps=eps, min_samples=6).fit_predict(X).tolist()})
    out["fallas"] = fallas

    # --- dendrograma de las presas por su perfil de llenado (% del NAMO, promedio mensual)
    reg = pd.read_excel(raiz / "almac_reciente.xlsx")
    cat = pd.read_excel(raiz / "catalogo_estatal.xlsx", sheet_name="Catálogo_estatal")
    reg["fecha"] = pd.to_datetime(reg.fecha)
    reg = reg.merge(cat[["Clave", "Nombre_comun", "cap_namo", "Río"]], left_on="clave", right_on="Clave")
    reg["llenado"] = 100 * reg.almacenamiento / reg.cap_namo
    perfil = reg.pivot_table(index="Nombre_comun", columns=reg.fecha.dt.to_period("M"), values="llenado").dropna(axis=1)
    L = linkage(perfil.to_numpy(), method="average")
    dg = dendrogram(L, labels=perfil.index.tolist(), no_plot=True)
    rio = cat.set_index("Nombre_comun")["Río"].str.replace("Río ", "").to_dict()
    out["presas"] = {"Z": r(L, 3), "orden": dg["leaves"], "icoord": dg["icoord"], "dcoord": r(dg["dcoord"], 2), "hojas": dg["ivl"],
                     "rio": [rio[h] for h in dg["ivl"]],
                     "series": {n: r(perfil.loc[n].to_numpy(), 1) for n in perfil.index},
                     "meses": [str(p) for p in perfil.columns]}

    # --- mezclas: 1D (aleta) y EM en 2D (pico, estandarizado)
    aleta = pg.flipper_length_mm.to_numpy().reshape(-1, 1)
    g1 = GaussianMixture(2, random_state=0).fit(aleta)
    xs = np.linspace(165, 240, 151)
    comp = [g1.weights_[j] * np.exp(-(xs - g1.means_[j, 0]) ** 2 / (2 * g1.covariances_[j, 0, 0])) / np.sqrt(2 * np.pi * g1.covariances_[j, 0, 0]) for j in range(2)]
    orden = np.argsort(g1.means_[:, 0])
    out["aleta"] = {"datos": r(aleta[:, 0], 0), "xs": r(xs, 1), "comp": [r(comp[j], 5) for j in orden],
                    "mu": r(g1.means_[orden, 0], 1), "sd": r(np.sqrt(g1.covariances_[orden, 0, 0]), 1), "pi": r(g1.weights_[orden], 3)}

    X = StandardScaler().fit_transform(pg[["bill_length_mm", "bill_depth_mm"]])
    kmp = KMeans(3, n_init=10, random_state=0).fit_predict(X)
    from scipy.stats import multivariate_normal as mvn
    K = 3
    # inicio: tres pingüinos al azar, covarianzas identidad (como en la libreta)
    mu = X[np.random.default_rng(SEMILLA_EM).choice(len(X), K, replace=False)]; S = np.array([np.eye(2)] * K); pi = np.ones(K) / K
    hist, em, previo = [], [], -np.inf
    for it in range(1000):
        dens = np.column_stack([pi[j] * mvn(mu[j], S[j]).pdf(X) for j in range(K)])
        ll = float(np.log(dens.sum(1)).sum())
        hist.append(round(ll, 2))
        fin = it > 0 and abs(ll - previo) < 1e-6
        previo = ll
        if it in (0, 1, 2, 5, 10, 20, 50) or fin:
            em.append({"it": it, "mu": r(mu, 3), "elipses": [_elipse(mu[j], S[j], 1.5) for j in range(K)], "pi": r(pi, 3),
                       "resp": r((dens / dens.sum(1, keepdims=True)).max(1), 2), "e": dens.argmax(1).tolist()})
        if fin:
            break
        R = dens / dens.sum(1, keepdims=True)
        Nk = R.sum(0); pi = Nk / len(X); mu = (R.T @ X) / Nk[:, None]
        S = np.array([((R[:, j, None] * (X - mu[j])).T @ (X - mu[j])) / Nk[j] + 1e-6 * np.eye(2) for j in range(K)])
    out["em"] = {"x": r(X[:, 0], 3), "y": r(X[:, 1], 3), "pasos": em, "ll": hist,
                 "km": kmp.tolist(), "ari_km": round(adjusted_rand_score(esp, kmp), 3),
                 "ari_gmm": round(adjusted_rand_score(esp, em[-1]["e"]), 3)}

    tipos = {}
    for t in ["spherical", "diag", "tied", "full"]:
        g = GaussianMixture(3, covariance_type=t, random_state=0, n_init=5).fit(X)
        cov = {"spherical": lambda c: [np.eye(2) * v for v in c], "diag": lambda c: [np.diag(v) for v in c],
               "tied": lambda c: [c] * 3, "full": lambda c: list(c)}[t](g.covariances_)
        tipos[t] = {"elipses": [_elipse(g.means_[j], cov[j], 1.5) for j in range(3)], "ari": round(adjusted_rand_score(esp, g.predict(X)), 3)}
    out["covarianzas"] = tipos

    X4 = StandardScaler().fit_transform(pg[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
    bic = {t: [round(float(GaussianMixture(k, covariance_type=t, random_state=0, n_init=5).fit(X4).bic(X4)), 1) for k in range(1, 8)]
           for t in ["spherical", "diag", "tied", "full"]}
    out["bic"] = {"k": list(range(1, 8)), "bic": bic}

    # --- reducción de dimensión
    rng = np.random.default_rng(0)
    razones = {}
    for d in [2, 10, 100, 1000]:
        P = rng.uniform(size=(500, d)); q = rng.uniform(size=d)
        dist = np.linalg.norm(P - q, axis=1)
        razones[d] = round(float((dist.max() - dist.min()) / dist.min()), 3)
    out["dimension"] = {"d": list(razones), "razon": list(razones.values())}

    A = pg[["flipper_length_mm", "body_mass_g"]].to_numpy()
    As = StandardScaler().fit_transform(A)
    p2 = PCA(2).fit(As)
    idx = rng.choice(len(As), 90, replace=False)
    out["pca2d"] = {"x": r(As[:, 0], 3), "y": r(As[:, 1], 3), "comp": r(p2.components_, 4), "var": r(p2.explained_variance_ratio_, 3), "muestra": idx.tolist()}
    p_raw = PCA().fit(pg[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
    out["escala_pca"] = {"crudo": r(p_raw.explained_variance_ratio_, 4), "estandar": r(PCA().fit(X4).explained_variance_ratio_, 4)}

    dg_ = load_digits()
    D, y = dg_.data, dg_.target
    pc = PCA().fit(D)
    acum = np.cumsum(pc.explained_variance_ratio_)
    i0 = int(np.flatnonzero(y == 3)[2])
    recon = {}
    for mm in [2, 5, 10, 20, 40]:
        p = PCA(mm).fit(D)
        recon[mm] = r(p.inverse_transform(p.transform(D[i0:i0 + 1]))[0], 2)
    err = [round(float(np.mean((D - PCA(mm).fit(D).inverse_transform(PCA(mm).fit(D).transform(D))) ** 2)), 3) for mm in range(1, 65, 3)]
    Z = PCA(2).fit_transform(D)
    t0 = __import__("time").perf_counter()
    T = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(D)
    t_tsne = __import__("time").perf_counter() - t0
    sub = rng.choice(len(D), 600, replace=False)
    perp = {p: r(TSNE(2, perplexity=p, init="pca", random_state=0).fit_transform(D[sub]), 2) for p in [5, 30, 100]}
    import umap
    t0 = __import__("time").perf_counter()
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(D)
    t_umap = __import__("time").perf_counter() - t0
    Xg = np.vstack([rng.normal(0, 1, (200, 10)), rng.normal(0, 8, (200, 10)) + 40])
    Tg = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Xg)
    out["digitos"] = {
        "acum": r(acum, 4), "n80": int(np.searchsorted(acum, .8) + 1), "n90": int(np.searchsorted(acum, .9) + 1), "n95": int(np.searchsorted(acum, .95) + 1),
        "media": r(pc.mean_, 2), "comps": r(pc.components_[:8], 3), "original": D[i0].astype(int).tolist(), "recon": recon,
        "err_m": list(range(1, 65, 3)), "err": err,
        "y": y.tolist(), "pca": r(Z, 2), "tsne": r(T, 2), "umap": r(U, 2),
        "sub_y": y[sub].tolist(), "perp": perp, "t_tsne": round(t_tsne, 1), "t_umap": round(t_umap, 1),
        "trampa": {"orig": r(Xg[:, :2], 2), "tsne": r(Tg, 2)},
    }
    return out


# ---------------------------------------------------------------- Unidad 3

def u03() -> dict:
    import pandas as pd
    from scipy.optimize import linprog
    from scipy.stats import binom
    from sklearn.calibration import calibration_curve
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, lasso_path
    from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    out: dict = {}
    rng = np.random.default_rng(2027)

    # --- Hoeffding exacto: P[|nu - mu| > eps] con nu ~ Bin(N, mu)/N, contra 2 exp(-2 eps^2 N)
    Ns = list(range(10, 401, 10))
    hoef = {}
    for eps in (0.05, 0.1):
        exacto = []
        for N in Ns:
            k = np.arange(N + 1)
            exacto.append(float(binom.pmf(k, N, 0.5)[np.abs(k / N - 0.5) > eps].sum()))
        hoef[str(eps)] = {"exacto": r(exacto, 6), "cota": r([min(1, 2 * np.exp(-2 * eps**2 * N)) for N in Ns], 6)}
    out["hoeffding"] = {"N": Ns, "eps": hoef}

    # --- las 1,000 monedas, 10 lanzamientos cada una, 20,000 repeticiones
    caras = rng.binomial(10, 0.5, size=(20000, 1000))
    nu1, nurand, numin = caras[:, 0] / 10, caras[np.arange(20000), rng.integers(0, 1000, 20000)] / 10, caras.min(1) / 10
    epss = np.round(np.arange(0, 0.55, 0.05), 2)
    frec = lambda v: [float(np.mean(np.abs(v - 0.5) > e)) for e in epss]
    out["monedas"] = {"hist": {n: [int(np.sum(np.isclose(v, j / 10))) for j in range(11)] for n, v in [("primera", nu1), ("azar", nurand), ("minima", numin)]},
                      "eps": epss.tolist(), "primera": r(frec(nu1), 4), "azar": r(frec(nurand), 4), "minima": r(frec(numin), 4),
                      "cota": r([min(1, 2 * np.exp(-2 * e**2 * 10)) for e in epss], 4)}

    # --- dicotomías de 3 puntos con un perceptrón: una recta separadora para cada etiquetado
    P3 = np.array([[0.2, 0.25], [0.8, 0.3], [0.45, 0.8]])
    rectas = []
    for m in range(8):
        y = np.array([1 if (m >> i) & 1 else -1 for i in range(3)])
        # máximo margen aproximado: max t s.a. y_i (w·x_i + b) >= t, |w|_inf <= 1
        c = np.array([0, 0, 0, -1.0])
        A = np.column_stack([-y[:, None] * P3, -y, np.ones(3)])
        res = linprog(c, A_ub=A, b_ub=np.zeros(3), bounds=[(-1, 1), (-1, 1), (-5, 5), (0, None)])
        w = res.x[:3]
        rectas.append({"y": y.tolist(), "w": r(w, 4)})
    out["dicotomias"] = {"puntos": P3.tolist(), "rectas": rectas,
                         "xor": [[0.25, 0.25, 1], [0.75, 0.75, 1], [0.25, 0.75, -1], [0.75, 0.25, -1]]}
    Nm = list(range(1, 13))
    out["crecimiento"] = {"N": Nm, "todas": [2**n for n in Nm], "rayos": [n + 1 for n in Nm],
                          "intervalos": [n * (n + 1) // 2 + 1 for n in Nm], "perceptron": [n * (n - 1) + 2 for n in Nm]}

    # --- cota VC (forma de Learning From Data) para d_VC = 3, delta = 0.05; m_H(2N) <= (2N)^d + 1
    NV = np.unique(np.round(np.logspace(2, 6, 60))).astype(float)
    omega = lambda N, d=3, delta=0.05: np.sqrt(8 / N * np.log(4 * ((2 * N) ** d + 1) / delta))
    out["cota_vc"] = {"N": NV.tolist(), "omega": r(omega(NV), 4),
                      "N_para_01": int(next(n for n in range(1000, 10**7, 100) if omega(n) <= 0.1))}

    # --- sesgo y varianza: sin(2 pi x) + ruido 0.25, N = 15 (como en la Unidad 1)
    f = lambda x: np.sin(2 * np.pi * x)
    xg = np.linspace(0, 1, 101)
    # x estratificada (una por cada N-ésimo de [0, 1]), como en la libreta u03-e02: con x libres a veces
    # quedan huecos en los extremos y la varianza de los polinomios de grado alto es infinita en la práctica
    estrat = lambda N: (np.arange(N) + rng.uniform(0, 1, N)) / N
    def ajustes(grado, N=15, R=2000, lam=0.0):
        G = np.empty((R, len(xg)))
        for t in range(R):
            x = estrat(N); y = f(x) + rng.normal(0, 0.25, N)
            V = np.vander(2 * x - 1, grado + 1); Vg = np.vander(2 * xg - 1, grado + 1)
            w = np.linalg.solve(V.T @ V + lam * np.eye(grado + 1) + 1e-10 * np.eye(grado + 1), V.T @ y)
            G[t] = Vg @ w
        return G
    sv = {"grado": list(range(0, 11)), "sesgo2": [], "var": []}
    muestras = {}
    for gdo in range(0, 11):
        G = ajustes(gdo)
        gbar = G.mean(0)
        sv["sesgo2"].append(float(np.mean((gbar - f(xg)) ** 2)))
        sv["var"].append(float(np.mean(G.var(0))))
        if gdo in (1, 3, 9):
            muestras[gdo] = {"curvas": r(np.clip(G[:25], -2.5, 2.5), 3), "media": r(gbar, 3)}
    sv["sesgo2"] = r(sv["sesgo2"], 4); sv["var"] = r(np.minimum(sv["var"], 50), 4)
    out["sesgo_varianza"] = {**sv, "x": r(xg, 3), "f": r(f(xg), 3), "muestras": muestras, "ruido": 0.0625}

    # curvas de aprendizaje: E_in y E_out esperados contra N para grados 1 y 5
    curvas = {}
    for gdo in (1, 5):
        Ein, Eout = [], []
        for N in range(gdo + 2, 61, 2):
            ei = eo = 0.0
            for _ in range(300):
                x = estrat(N); y = f(x) + rng.normal(0, 0.25, N)
                V = np.vander(2 * x - 1, gdo + 1); w = np.linalg.lstsq(V, y, rcond=None)[0]
                xt = rng.uniform(0, 1, 400); yt = f(xt) + rng.normal(0, 0.25, 400)
                ei += np.mean((V @ w - y) ** 2); eo += np.mean((np.vander(2 * xt - 1, gdo + 1) @ w - yt) ** 2)
            Ein.append(ei / 300); Eout.append(min(eo / 300, 3))
        curvas[gdo] = {"N": list(range(gdo + 2, 61, 2)), "ein": r(Ein, 4), "eout": r(Eout, 4)}
    out["aprendizaje"] = curvas

    # regularización: grado 12 con distintos lambda (un mismo conjunto) y sesgo/varianza contra lambda
    x = estrat(15); y = f(x) + rng.normal(0, 0.25, 15)
    V = np.vander(2 * x - 1, 13); Vg = np.vander(2 * xg - 1, 13)
    reg = {}
    for lam in (0.0, 1e-4, 1e-2, 1.0):
        w = np.linalg.solve(V.T @ V + lam * np.eye(13) + 1e-12 * np.eye(13), V.T @ y)
        reg[str(lam)] = r(np.clip(Vg @ w, -3, 3), 3)
    lams = np.logspace(-6, 1, 15)
    bl, vl = [], []
    for lam in lams:
        G = ajustes(12, lam=lam, R=800); gbar = G.mean(0)
        bl.append(float(np.mean((gbar - f(xg)) ** 2))); vl.append(float(np.mean(G.var(0))))
    out["regularizacion"] = {"xtr": r(x, 3), "ytr": r(y, 3), "curvas": reg, "lam": r(lams, 8), "sesgo2": r(bl, 4), "var": r(np.minimum(vl, 50), 4)}

    # caminos de ridge y lasso en diabetes (estandarizada)
    db = load_diabetes()
    Xd = StandardScaler().fit_transform(db.data); yd = db.target - db.target.mean()
    lr = np.logspace(-2, 5, 50)
    U, sd_, Vt = np.linalg.svd(Xd, full_matrices=False)
    ridge = np.array([Vt.T @ np.diag(sd_ / (sd_**2 + l)) @ U.T @ yd for l in lr])
    la, coefs, _ = lasso_path(Xd, yd, alphas=np.logspace(-1.5, 1.7, 50))
    out["caminos"] = {"nombres": ["edad", "sexo", "IMC", "presión", "s1", "s2", "s3", "s4", "s5", "s6"],
                      "lam_ridge": r(lr, 4), "ridge": r(ridge.T, 2), "lam_lasso": r(la, 4), "lasso": r(coefs, 2)}

    # --- validación: optimismo de quedarse con el mejor de M modelos sin poder predictivo
    Ms = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    K = 100
    mejor = [float(np.mean(rng.binomial(K, 0.5, size=(3000, M)).min(1) / K)) for M in Ms]
    out["optimismo"] = {"M": Ms, "mejor": r(mejor, 4), "K": K}

    # --- métricas con el crédito de Taiwán
    cr = pd.read_csv(DESTINO.parents[1] / "datos" / "credito_taiwan.csv")
    y = cr.pop("incumple").to_numpy()
    cat = ["sexo", "escolaridad", "estado_civil"]
    # misma partición y modelos que la libreta u03-e04
    Xtr, Xte, ytr, yte = train_test_split(cr, y, test_size=0.25, stratify=y, random_state=2011)
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder=StandardScaler())
    log = make_pipeline(prep, LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    bosque = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=2011)
    hgb = bosque.fit(Xtr, ytr)
    pl, ph = log.predict_proba(Xte)[:, 1], hgb.predict_proba(Xte)[:, 1]
    piso = CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=2011),
                                  method="isotonic", cv=5).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    def roc_red(p):
        fpr, tpr, thr = roc_curve(yte, p)
        idx = np.unique(np.linspace(0, len(fpr) - 1, 150).astype(int))
        return {"fpr": r(fpr[idx], 4), "tpr": r(tpr[idx], 4), "auc": round(float(roc_auc_score(yte, p)), 3)}
    def pr_red(p):
        pr, rc, _ = precision_recall_curve(yte, p)
        idx = np.unique(np.linspace(0, len(pr) - 1, 150).astype(int))
        return {"precision": r(pr[idx], 4), "exhaustividad": r(rc[idx], 4)}
    bins = np.linspace(0, 1, 41)
    out["credito"] = {
        "n_prueba": int(len(yte)), "prevalencia": round(float(yte.mean()), 4), "exactitud_base": round(float(1 - yte.mean()), 4),
        "hist_pos": np.histogram(ph[yte == 1], bins)[0].tolist(), "hist_neg": np.histogram(ph[yte == 0], bins)[0].tolist(),
        "puntajes_pos": r(np.sort(ph[yte == 1]), 4), "puntajes_neg": r(np.sort(ph[yte == 0]), 4),
        "roc": {"logistica": roc_red(pl), "bosque": roc_red(ph)}, "pr": {"logistica": pr_red(pl), "bosque": pr_red(ph)},
    }
    # costos: c_FP = 1 (llamar o reducir el límite a quien sí paga), c_FN = 5 (pérdida esperada por incumplimiento)
    ts = np.round(np.linspace(0.02, 0.9, 89), 3)
    costo = lambda p, t, cfp=1, cfn=5: float((cfp * np.sum((p >= t) & (yte == 0)) + cfn * np.sum((p < t) & (yte == 1))) / len(yte))
    out["costos"] = {"t": ts.tolist(), "logistica": r([costo(pl, t) for t in ts], 4), "bosque": r([costo(ph, t) for t in ts], 4),
                     "t_bayes": round(1 / 6, 4), "cfp": 1, "cfn": 5}
    cal = {}
    for nombre, p in (("logistica", pl), ("bosque", ph), ("bosque_iso", piso)):
        pv, pp = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        cal[nombre] = {"predicha": r(pp, 4), "observada": r(pv, 4), "brier": round(float(brier_score_loss(yte, p)), 4)}
    out["calibracion"] = cal
    t = 1 / 6
    grupos = {}
    sexo = Xte.sexo.to_numpy()
    for g, nombre in ((1, "hombres"), (2, "mujeres")):
        m = sexo == g; yy, pp = yte[m], piso[m] >= t
        grupos[nombre] = {"n": int(m.sum()), "prevalencia": round(float(yy.mean()), 4),
                          "seleccion": round(float(pp.mean()), 4), "tpr": round(float(pp[yy == 1].mean()), 4),
                          "fpr": round(float(pp[yy == 0].mean()), 4), "vpp": round(float(yy[pp].mean()), 4)}
    out["equidad"] = grupos
    return out


if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, fn in {"u01": u01, "u02": u02, "u03": u03}.items():
        if len(sys.argv) > 1 and nombre not in sys.argv[1:]:
            continue
        ruta = DESTINO / f"{nombre}.json"
        ruta.write_text(json.dumps(fn(), separators=(",", ":")), encoding="utf-8")
        print(f"✓ {ruta.relative_to(DESTINO.parents[1])}")
