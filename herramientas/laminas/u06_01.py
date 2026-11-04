"""Datos de las láminas de u06-01-mlp-backprop (perceptrón multicapa y diferenciación automática).

Para que las cifras de las láminas sean exactamente las de la libreta, este módulo ejecuta el código de
unidades/06-redes-neuronales/ejemplos/u06-e01-mlp-backprop.py (sin gráficas ni cuestionario) y toma los
resultados de su espacio de nombres.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
LIBRETA = RAIZ / "unidades" / "06-redes-neuronales" / "ejemplos" / "u06-e01-mlp-backprop.py"


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


def _ejecutar_libreta() -> dict:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    fuente = LIBRETA.read_text(encoding="utf-8").replace("rdp.quiz.mostrar(", "(lambda *a, **k: None)(")
    ns: dict = {"__name__": "libreta"}
    mostrar = plt.show
    plt.show = lambda *a, **k: plt.close("all")
    try:
        exec(compile(fuente, str(LIBRETA), "exec"), ns)
    finally:
        plt.show = mostrar
    return ns


def datos() -> dict:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    L = _ejecutar_libreta()
    nn, torch = L["nn"], L["torch"]
    probabilidad, exactitud, red_mlp = L["probabilidad"], L["exactitud"], L["red_mlp"]
    out: dict = {}

    # ---------- grafo de cómputo de una neurona logística (sección 4)
    w, b, x = -0.8, 0.4, 1.5
    u = w * x; z = u + b; yg = 1 / (1 + np.exp(-z)); l = -np.log(yg)
    zbar = yg - 1
    out["grafo"] = {"w": w, "b": b, "x": x, "u": u, "z": z, "yg": round(float(yg), 4), "l": round(float(l), 4),
                    "barl": 1.0, "baryg": round(float(-1 / yg), 4), "barz": round(float(zbar), 4),
                    "baru": round(float(zbar), 4), "barb": round(float(zbar), 4), "barw": round(float(zbar * x), 4)}

    # ---------- costo (sección 5)
    costo = L["costo"]
    out["costo"] = {"n_param": int(L["n_param"]), "P": [int(p) for p in costo.index],
                    "directo": [float(f"{v:.3g}") for v in costo["modo directo (ms)"]],
                    "inverso": [float(f"{v:.3g}") for v in costo["modo inverso (ms)"]],
                    "t_inverso_ms": round(1000 * L["t_inverso"], 2), "t_numerico_s": round(L["t_numerico"], 2),
                    "dif": float(f"{L['dif']:.2g}")}

    # ---------- lunas (sección 6)
    X_l, y_l = L["X_lunas"], L["y_lunas"]
    X_lp, y_lp = L["X_lunas_pru"], L["y_lunas_pru"]
    gx, gy = np.linspace(-2, 3, 126), np.linspace(-1.6, 2.1, 94)
    malla = np.array([[a, c] for c in gy for a in gx])
    reg = lambda red: _franjas((probabilidad(red, malla)[:, 1] > 0.5).astype(int).reshape(len(gy), len(gx)), gx, gy)
    copia = red_mlp(2, 16, 2, nn.Tanh)
    regiones = []
    for e in L["EPOCAS_FOTO"]:
        copia.load_state_dict(L["fotos"][e])
        regiones.append(reg(copia))
    log_l = L["log_lunas"]
    out["lunas"] = {
        "x": r(X_l[:, 0], 3), "y": r(X_l[:, 1], 3), "g": y_l.tolist(),
        "epocas": L["EPOCAS_FOTO"], "regiones": regiones,
        "perdida": r(L["hist_lunas"], 4), "perdida_log": round(float(L["perdida_log_lunas"]), 4),
        "exac_red": exactitud(L["red_lunas"], X_l, y_l), "exac_red_nuevos": exactitud(L["red_lunas"], X_lp, y_lp),
        "exac_log": float(log_l.score(X_l, y_l)), "exac_log_nuevos": float(log_l.score(X_lp, y_lp)),
        "exac_log_a_nuevos": float(L["log_sobre_a"].score(L["A_lunas_pru"], y_lp)),
        "lineal": reg(L["red_lineal"]), "exac_lineal": exactitud(L["red_lineal"], X_l, y_l),
    }

    # ---------- XOR con dos neuronas ocultas
    Xx, yx, red_xor = L["Xx"], L["yx"], L["red_xor"]
    gx2 = np.linspace(-2.2, 2.2, 111)
    malla2 = np.array([[a, c] for c in gx2 for a in gx2])
    A = L["A_xor"]
    out["xor"] = {"x": r(Xx[:, 0], 3), "y": r(Xx[:, 1], 3), "g": yx.tolist(), "a1": r(A[:, 0], 4), "a2": r(A[:, 1], 4),
                  "W1": r(L["W1_xor"], 4), "b1": r(L["b1_xor"], 4), "W2": r(L["W2_xor"], 4), "b2": r(L["b2_xor"], 4),
                  "regiones": _franjas((probabilidad(red_xor, malla2)[:, 1] > 0.5).astype(int).reshape(111, 111), gx2, gx2),
                  "exac": exactitud(red_xor, Xx, yx), "exac_log": float(L["log_xor"].score(Xx, yx)),
                  "semillas": [round(float(v), 3) for v in L["exactitud_por_semilla"].values()], "exitos": int(L["exitos"])}

    # ---------- Hermosillo (sección 7)
    dia, f_dia, temp = L["dia"], L["f_dia"], L["temp"]
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

    # ---------- dígitos (sección 8)
    yd, yd_pru, pred = L["yd"], L["yd_pru"], L["pred_t"]
    malos = L["malos"]
    imgs = load_digits().images
    _, idx_pru = train_test_split(np.arange(len(yd)), test_size=0.25, stratify=yd, random_state=L["SEMILLA"])
    Zx, Za = L["pca_x"].transform(L["Xd_pru"]), L["pca_a"].transform(L["A_pru"])
    out["digitos"] = {
        "exac_log": float(L["exac_logistica"]), "exac_torch": float(L["exac_torch"]), "exac_log_a": float(L["exac_log_a"]),
        "err_log": int(np.sum(L["logistica_digitos"].predict(L["Xd_pru"]) != yd_pru)), "err_torch": int(len(malos)),
        "err_log_a": int(round((1 - L["exac_log_a"]) * len(yd_pru))),
        "errores": [{"px": imgs[idx_pru[i]].astype(int).ravel().tolist(), "real": int(yd_pru[i]), "pred": int(pred[i])} for i in malos],
        "pca": {"g": yd_pru.tolist(), "x1": r(Zx[:, 0], 3), "x2": r(Zx[:, 1], 3), "a1": r(Za[:, 0], 3), "a2": r(Za[:, 1], 3)},
        "silueta_x": round(float(L["silhouette_score"](L["Xd_pru"], yd_pru)), 2),
        "silueta_a": round(float(L["silhouette_score"](L["A_pru"], yd_pru)), 2),
    }

    # ---------- inicialización (sección 10)
    bordes = np.linspace(-4, 4, 41)
    hist = {}
    for nombre, clave in [("N(0, 1)", "grande"), ("N(0, 0.01²)", "pequena"), ("N(0, 1/n_in)", "justa")]:
        zs = L["z_por_escala"][nombre]
        hist[clave] = {"desv": round(float(zs.std()), 2),
                       "frec": r(np.histogram(np.clip(zs, -4, 4), bins=bordes)[0] / len(zs), 4)}
    out["escala"] = {"bordes": r(bordes, 2), **hist}
    out["simetria"] = {"distintos": int(L["torch"].unique(L["W1_sim"].round(decimals=6), dim=0).shape[0]),
                       "exac": round(L["evaluar"](L["simetrica"], L["Xd_pru_t"], L["yd_pru_t"]), 3)}
    claves = {"N(0, 0.01²)": "pequena", "Xavier: 2/(n_in + n_out)": "xavier", "He: 2/n_in": "he", "N(0, 1)": "grande"}
    out["activaciones"] = {claves[k]: [float(f"{v:.3g}") for v in d] for k, d in L["desviaciones"].items()}
    out["gradientes"] = {"sigmoide": [float(f"{v:.3g}") for v in L["grad_sigmoide"]],
                         "relu": [float(f"{v:.3g}") for v in L["grad_relu"]]}
    del torch
    return out


if __name__ == "__main__":
    d = datos()
    print(d["grafo"], d["costo"])
    print({k: d["digitos"][k] for k in d["digitos"] if k not in ("errores", "pca")})
    print(d["xor"]["exac"], d["xor"]["exitos"], d["lunas"]["exac_red_nuevos"], d["lunas"]["exac_log_nuevos"],
          d["lunas"]["exac_log_a_nuevos"], d["lunas"]["exac_lineal"])
    print({k: v["recm"] for k, v in d["hermosillo"]["redes"].items()})
    print(d["simetria"], {k: d["escala"][k]["desv"] for k in ("grande", "pequena", "justa")})
    print({k: (v[0], v[-1]) for k, v in d["activaciones"].items()}, d["gradientes"]["sigmoide"][-1] / d["gradientes"]["sigmoide"][0])
