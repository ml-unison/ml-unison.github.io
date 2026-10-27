"""Datos de las láminas de u06-01-mlp-backprop (perceptrón multicapa y retropropagación).

Mismo código, datos y semillas que la libreta unidades/06-redes-neuronales/ejemplos/u06-e01-mlp-backprop.py,
para que las cifras de las láminas coincidan con las de la libreta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
SEMILLA = 2011


def r(a, d: int = 4):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def _franjas(etiq, xs, ys):
    """Malla de clases → franjas horizontales [fila, j0, j1, clase] (formato del componente Regiones)."""
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


# ---------- la red de la libreta, en NumPy ----------
ACTIVACIONES = {
    "tanh": (np.tanh, lambda Z, A: 1 - A**2),
    "relu": (lambda Z: np.maximum(Z, 0), lambda Z, A: (Z > 0).astype(float)),
    "identidad": (lambda Z: Z, lambda Z, A: np.ones_like(Z)),
}


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def inicializar(d, H, K, rng, act="tanh"):
    var1 = 2 / d if act == "relu" else 2 / (d + H)
    return {"W1": rng.normal(0, np.sqrt(var1), (H, d)), "b1": np.zeros(H),
            "W2": rng.normal(0, np.sqrt(2 / (H + K)), (K, H)), "b2": np.zeros(K)}


def adelante(p, X, act="tanh"):
    Z1 = X @ p["W1"].T + p["b1"]
    A1 = ACTIVACIONES[act][0](Z1)
    Z2 = A1 @ p["W2"].T + p["b2"]
    return softmax(Z2), (X, Z1, A1)


def entropia_cruzada(P, y):
    return -np.mean(np.log(P[np.arange(len(y)), y]))


def atras(p, cache, P, y, act="tanh"):
    X, Z1, A1 = cache
    N = len(y)
    D2 = P.copy()
    D2[np.arange(N), y] -= 1
    D2 /= N
    D1 = (D2 @ p["W2"]) * ACTIVACIONES[act][1](Z1, A1)
    return {"W1": D1.T @ X, "b1": D1.sum(0), "W2": D2.T @ A1, "b2": D2.sum(0)}


def gradiente_numerico(p, X, y, act, h=1e-5):
    g = {}
    for nombre, M in p.items():
        g[nombre] = np.zeros_like(M)
        for i in np.ndindex(M.shape):
            original = M[i]
            M[i] = original + h; e_mas = entropia_cruzada(adelante(p, X, act)[0], y)
            M[i] = original - h; e_menos = entropia_cruzada(adelante(p, X, act)[0], y)
            M[i] = original
            g[nombre][i] = (e_mas - e_menos) / (2 * h)
    return g


def entrenar(p, X, y, act, eta, epocas, tam_lote, rng, guardar_en=()):
    historia, copias = [], {}
    if 0 in guardar_en:
        copias[0] = {k: M.copy() for k, M in p.items()}
    for epoca in range(1, epocas + 1):
        orden = rng.permutation(len(y))
        for i in range(0, len(y), tam_lote):
            lote = orden[i:i + tam_lote]
            P, cache = adelante(p, X[lote], act)
            g = atras(p, cache, P, y[lote], act)
            for k in p:
                p[k] -= eta * g[k]
        historia.append(entropia_cruzada(adelante(p, X, act)[0], y))
        if epoca in guardar_en:
            copias[epoca] = {k: M.copy() for k, M in p.items()}
    return historia, copias


def exactitud(p, X, y, act):
    return float(np.mean(adelante(p, X, act)[0].argmax(1) == y))


def nubes_xor(n_por_nube, rng, desv=0.35):
    centros = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    X = np.concatenate([c + desv * rng.standard_normal((n_por_nube, 2)) for c in centros])
    y = np.repeat([0, 0, 1, 1], n_por_nube)
    return X, y


def datos() -> dict:
    import pandas as pd
    import torch, torch.nn as nn, torch.nn.functional as F
    from sklearn.datasets import load_digits, make_moons
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    torch.manual_seed(SEMILLA)
    out: dict = {}

    # ---------- dígitos y logística (sección 1)
    Xd, yd = load_digits(return_X_y=True)
    Xd = Xd / 16
    Xd_ent, Xd_pru, yd_ent, yd_pru = train_test_split(Xd, yd, test_size=0.25, stratify=yd, random_state=SEMILLA)
    logistica = LogisticRegression(C=10, max_iter=5000).fit(Xd_ent, yd_ent)
    exac_log = float(logistica.score(Xd_pru, yd_pru))

    # ---------- XOR (secciones 2 y 6)
    Xx, yx = nubes_xor(50, np.random.default_rng(SEMILLA))
    log_xor = float(LogisticRegression().fit(Xx, yx).score(Xx, yx))

    # ---------- lunas: verificación del gradiente (sección 5)
    X_l, y_l = make_moons(400, noise=0.25, random_state=SEMILLA)
    X_lp, y_lp = make_moons(2000, noise=0.25, random_state=SEMILLA + 1)
    p = inicializar(2, 8, 2, np.random.default_rng(SEMILLA), "tanh")
    P, cache = adelante(p, X_l, "tanh")
    g_atras = atras(p, cache, P, y_l, "tanh")
    g_num = gradiente_numerico(p, X_l, y_l, "tanh")
    rel = {k: float(np.max(np.abs(g_atras[k] - g_num[k]) / np.maximum(np.maximum(np.abs(g_atras[k]), np.abs(g_num[k])), 1e-12)))
           for k in p}

    def error_derivada(h, centrada):
        M = p["W1"]; original = M[0, 0]
        M[0, 0] = original + h; e_mas = entropia_cruzada(adelante(p, X_l, "tanh")[0], y_l)
        M[0, 0] = original - h if centrada else original
        e_otro = entropia_cruzada(adelante(p, X_l, "tanh")[0], y_l)
        M[0, 0] = original
        aprox = (e_mas - e_otro) / (2 * h if centrada else h)
        return abs(aprox - g_atras["W1"][0, 0]) / abs(g_atras["W1"][0, 0])

    hs = np.logspace(-1, -13, 49)
    out["difs"] = {"h": [float(f"{h:.4g}") for h in hs],
                   "adelante": [float(f"{error_derivada(h, False):.3g}") for h in hs],
                   "centrada": [float(f"{error_derivada(h, True):.3g}") for h in hs],
                   "rel": {k: float(f"{v:.2g}") for k, v in rel.items()}, "n_param": int(sum(M.size for M in p.values()))}

    # costo (sección 5): solo conteos, los tiempos dependen de la máquina
    p_dig = inicializar(64, 64, 10, np.random.default_rng(SEMILLA), "relu")
    out["costo"] = {"n_param": int(sum(M.size for M in p_dig.values()))}

    # ---------- lunas: entrenamiento (sección 6)
    rng = np.random.default_rng(SEMILLA)
    p_lunas = inicializar(2, 16, 2, rng, "tanh")
    epocas_foto = [0, 5, 20, 100, 300]
    hist, fotos = entrenar(p_lunas, X_l, y_l, "tanh", 0.3, 300, 20, rng, guardar_en=epocas_foto)
    log_l = LogisticRegression().fit(X_l, y_l)
    gx, gy = np.linspace(-2, 3, 126), np.linspace(-1.6, 2.1, 94)
    malla = np.array([[a, b] for b in gy for a in gx])
    reg = lambda q, act: _franjas((adelante(q, malla, act)[0][:, 1] > 0.5).astype(int).reshape(len(gy), len(gx)), gx, gy)

    rng = np.random.default_rng(SEMILLA)
    p_lin = inicializar(2, 16, 2, rng, "identidad")
    entrenar(p_lin, X_l, y_l, "identidad", 0.3, 300, 20, rng)
    out["lunas"] = {
        "x": r(X_l[:, 0], 3), "y": r(X_l[:, 1], 3), "g": y_l.tolist(),
        "epocas": epocas_foto, "regiones": [reg(fotos[e], "tanh") for e in epocas_foto],
        "perdida": r(hist, 4), "perdida_log": round(float(entropia_cruzada(log_l.predict_proba(X_l), y_l)), 4),
        "exac_red": exactitud(p_lunas, X_l, y_l, "tanh"), "exac_red_nuevos": exactitud(p_lunas, X_lp, y_lp, "tanh"),
        "exac_log": float(log_l.score(X_l, y_l)), "exac_log_nuevos": float(log_l.score(X_lp, y_lp)),
        "lineal": reg(p_lin, "identidad"), "exac_lineal": exactitud(p_lin, X_l, y_l, "identidad"),
    }

    # XOR con dos neuronas ocultas
    rng = np.random.default_rng(SEMILLA)
    p_xor = inicializar(2, 2, 2, rng, "tanh")
    entrenar(p_xor, Xx, yx, "tanh", 0.3, 200, 20, rng)
    _, (_, _, A) = adelante(p_xor, Xx, "tanh")
    gx2 = np.linspace(-2.2, 2.2, 111)
    malla2 = np.array([[a, b] for b in gx2 for a in gx2])
    out["xor"] = {"x": r(Xx[:, 0], 3), "y": r(Xx[:, 1], 3), "g": yx.tolist(), "a1": r(A[:, 0], 4), "a2": r(A[:, 1], 4),
                  "W1": r(p_xor["W1"], 4), "b1": r(p_xor["b1"], 4), "W2": r(p_xor["W2"], 4), "b2": r(p_xor["b2"], 4),
                  "regiones": _franjas((adelante(p_xor, malla2, "tanh")[0][:, 1] > 0.5).astype(int).reshape(111, 111), gx2, gx2),
                  "exac": exactitud(p_xor, Xx, yx, "tanh"), "exac_log": log_xor}

    # ---------- Hermosillo (sección 7)
    temp = pd.read_csv(RAIZ / "datos" / "temperatura_hermosillo.csv", parse_dates=["fecha"])
    temp = temp[~((temp.fecha.dt.month == 2) & (temp.fecha.dt.day == 29))]
    temp["dia"] = temp.fecha.dt.day_of_year - ((temp.fecha.dt.is_leap_year) & (temp.fecha.dt.month > 2))
    clima = temp.groupby("dia").tmed.mean()
    dia, f_dia = clima.index.to_numpy(dtype=float), clima.to_numpy()
    redes = {}
    for H in [1, 2, 3, 6, 12, 24, 48, 96]:
        t = np.linspace(dia.min(), dia.max(), H + 1)
        v = np.interp(t, dia, f_dia)
        g = np.interp(dia, t, v)                       # = salida de la red ReLU interpolante
        pend = np.diff(v) / np.diff(t)
        redes[str(H)] = {"t": r(t, 2), "v": r(v, 2), "s": r(np.diff(np.concatenate([[0], pend])), 4),
                         "recm": round(float(np.sqrt(np.mean((g - f_dia) ** 2))), 2)}
    out["hermosillo"] = {"dia": dia.astype(int).tolist(), "t": r(f_dia, 2), "redes": redes,
                         "anios": [int(temp.fecha.dt.year.min()), int(temp.fecha.dt.year.max())]}

    # ---------- dígitos: NumPy (sección 8) y PyTorch (sección 9)
    rng = np.random.default_rng(SEMILLA)
    p_dig = inicializar(64, 64, 10, rng, "relu")
    entrenar(p_dig, Xd_ent, yd_ent, "relu", 0.1, 40, 32, rng)
    exac_np = exactitud(p_dig, Xd_pru, yd_pru, "relu")

    p0 = inicializar(2, 8, 2, np.random.default_rng(SEMILLA), "tanh")
    P, cache = adelante(p0, X_l, "tanh")
    g_mano = atras(p0, cache, P, y_l, "tanh")
    pt = {k: torch.tensor(M, dtype=torch.float64, requires_grad=True) for k, M in p0.items()}
    logits = torch.tanh(torch.tensor(X_l) @ pt["W1"].T + pt["b1"]) @ pt["W2"].T + pt["b2"]
    F.cross_entropy(logits, torch.tensor(y_l)).backward()
    dif_autograd = max(float(np.max(np.abs(pt[k].grad.numpy() - g_mano[k]))) for k in pt)

    class MLP(nn.Module):
        def __init__(self, d, H, K):
            super().__init__()
            self.oculta = nn.Linear(d, H)
            self.salida = nn.Linear(H, K)

        def forward(self, x):
            return self.salida(F.relu(self.oculta(x)))

    Xe = torch.tensor(Xd_ent, dtype=torch.float32); ye = torch.tensor(yd_ent)
    Xp = torch.tensor(Xd_pru, dtype=torch.float32)
    cargador = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xe, ye), batch_size=32, shuffle=True,
                                           generator=torch.Generator().manual_seed(SEMILLA))
    torch.manual_seed(SEMILLA)
    mlp = MLP(64, 64, 10)
    opt = torch.optim.SGD(mlp.parameters(), lr=0.1)
    perdida_fn = nn.CrossEntropyLoss()
    for _ in range(40):
        mlp.train()
        for xb, yb in cargador:
            opt.zero_grad(); perdida_fn(mlp(xb), yb).backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        pred = mlp(Xp).argmax(1).numpy()
    exac_t = float(np.mean(pred == yd_pru))
    malos = np.flatnonzero(pred != yd_pru)
    imgs = load_digits().images
    _, idx_pru = train_test_split(np.arange(len(yd)), test_size=0.25, stratify=yd, random_state=SEMILLA)
    out["digitos"] = {
        "exac_log": exac_log, "exac_np": exac_np, "exac_torch": exac_t,
        "err_log": int(np.sum(logistica.predict(Xd_pru) != yd_pru)),
        "err_np": int(np.sum(adelante(p_dig, Xd_pru, "relu")[0].argmax(1) != yd_pru)), "err_torch": int(len(malos)),
        "dif_autograd": float(f"{dif_autograd:.2g}"),
        "errores": [{"px": imgs[idx_pru[i]].astype(int).ravel().tolist(), "real": int(yd_pru[i]), "pred": int(pred[i])} for i in malos],
    }

    # ---------- inicialización (sección 10)
    Xd_est = (Xd_ent - Xd_ent.mean(0)) / (Xd_ent.std(0) + 1e-8)
    inits = {"pequena": lambda n_in, n_out: 0.01, "xavier": lambda n_in, n_out: np.sqrt(2 / (n_in + n_out)),
             "he": lambda n_in, n_out: np.sqrt(2 / n_in), "grande": lambda n_in, n_out: 1.0}

    def activaciones_por_capa(X, capas, ancho, desv_w, act, rng):
        a, desv = X, []
        for _ in range(capas):
            W = rng.normal(0, desv_w(a.shape[1], ancho), (ancho, a.shape[1]))
            a = ACTIVACIONES[act][0](a @ W.T)
            desv.append(a.std())
        return np.array(desv)

    out["activaciones"] = {k: [float(f"{v:.3g}") for v in activaciones_por_capa(Xd_est, 20, 256, f, "relu", np.random.default_rng(SEMILLA))]
                           for k, f in inits.items()}

    def normas_gradiente(activacion, init, capas=10, ancho=64):
        torch.manual_seed(SEMILLA)
        lineales = [nn.Linear(64 if i == 0 else ancho, ancho) for i in range(capas)] + [nn.Linear(ancho, 10)]
        for capa in lineales:
            init(capa.weight); nn.init.zeros_(capa.bias)
        partes = []
        for capa in lineales[:-1]:
            partes += [capa, activacion()]
        red = nn.Sequential(*partes, lineales[-1])
        F.cross_entropy(red(Xe[:256]), ye[:256]).backward()
        return [float(f"{capa.weight.grad.norm().item():.3g}") for capa in lineales[:-1]]

    out["gradientes"] = {"sigmoide": normas_gradiente(nn.Sigmoid, nn.init.xavier_normal_),
                         "relu": normas_gradiente(nn.ReLU, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))}
    return out


if __name__ == "__main__":
    import json
    d = datos()
    print(json.dumps({k: (v if k in ("costo",) else "…") for k, v in d.items()}))
    print({k: d["digitos"][k] for k in d["digitos"] if k != "errores"})
    print(d["difs"]["rel"], d["xor"]["exac"], d["lunas"]["exac_red_nuevos"], d["lunas"]["exac_log_nuevos"], d["lunas"]["exac_lineal"])
    print({k: v["recm"] for k, v in d["hermosillo"]["redes"].items()})
    print({k: (v[0], v[-1]) for k, v in d["activaciones"].items()}, d["gradientes"]["sigmoide"][-1] / d["gradientes"]["sigmoide"][0])
