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


if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre, fn in {"u01": u01, "u02": u02}.items():
        if len(sys.argv) > 1 and nombre not in sys.argv[1:]:
            continue
        ruta = DESTINO / f"{nombre}.json"
        ruta.write_text(json.dumps(fn(), separators=(",", ":")), encoding="utf-8")
        print(f"✓ {ruta.relative_to(DESTINO.parents[1])}")
