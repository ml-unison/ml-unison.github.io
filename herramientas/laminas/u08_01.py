"""Datos de las láminas de u08-01-experimentos (mismo código y semillas que la libreta u08-e01-mlflow)."""

from __future__ import annotations

import numpy as np


def _r(a, d: int = 4):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def datos() -> dict:
    import lightgbm as lgb
    import rdp
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SEMILLA = 2011
    credito = rdp.datos.cargar("credito_taiwan")
    categoricas = ["sexo", "escolaridad", "estado_civil"]
    X, y = credito.drop(columns="incumple"), credito["incumple"]
    numericas = [c for c in X.columns if c not in categoricas]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)

    def pre():
        return ColumnTransformer([("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
                                  ("numericas", StandardScaler(), numericas)])

    # 1. El mismo código, diez semillas
    X_a, X_v, y_a, y_v = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEMILLA)
    bosque = [roc_auc_score(y_v, RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=s)
                            .fit(X_a, y_a).predict_proba(X_v)[:, 1]) for s in range(10)]
    log = make_pipeline(pre(), LogisticRegression(max_iter=2000))
    pliegues = [cross_validate(log, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=s),
                               scoring="roc_auc")["test_score"].mean() for s in range(10)]

    # 2. Siete configuraciones con validación cruzada
    configs = [
        ("logística C=1", "logistica", {"C": 1.0}),
        ("logística C=0.01", "logistica", {"C": 0.01}),
        ("bosque hoja≥50", "bosque", {"n_estimators": 200, "min_samples_leaf": 50}),
        ("bosque hoja≥5", "bosque", {"n_estimators": 200, "min_samples_leaf": 5}),
        ("boosting η=0.05, 15 hojas", "boosting", {"learning_rate": 0.05, "num_leaves": 15, "n_estimators": 300}),
        ("boosting η=0.1, 31 hojas", "boosting", {"learning_rate": 0.1, "num_leaves": 31, "n_estimators": 100}),
        ("boosting η=0.3, 63 hojas", "boosting", {"learning_rate": 0.3, "num_leaves": 63, "n_estimators": 300}),
    ]

    def construir(fam, p):
        if fam == "logistica":
            return make_pipeline(pre(), LogisticRegression(max_iter=2000, **p))
        if fam == "bosque":
            return RandomForestClassifier(n_jobs=-1, random_state=SEMILLA, **p)
        return lgb.LGBMClassifier(random_state=SEMILLA, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1, **p)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    corridas = []
    for nombre, fam, p in configs:
        r = cross_validate(construir(fam, p), X_tr, y_tr, cv=cv, scoring=["roc_auc", "neg_log_loss"])
        corridas.append({"nombre": nombre, "familia": fam, "auc": round(float(r["test_roc_auc"].mean()), 4),
                         "auc_de": round(float(r["test_roc_auc"].std()), 4),
                         "logloss": round(float(-r["test_neg_log_loss"].mean()), 4),
                         "pliegues": _r(r["test_roc_auc"])})
    corridas.sort(key=lambda c: -c["auc"])

    # 3. Campeón (logística) contra retador (bosque) en prueba
    v1 = construir("logistica", {"C": 1.0}).fit(X_tr, y_tr)
    v2 = construir("bosque", {"n_estimators": 200, "min_samples_leaf": 50}).fit(X_tr, y_tr)
    p1, p2 = v1.predict_proba(X_te)[:, 1], v2.predict_proba(X_te)[:, 1]
    duelo = {"v1": {"auc": round(roc_auc_score(y_te, p1), 4), "logloss": round(log_loss(y_te, p1), 4)},
             "v2": {"auc": round(roc_auc_score(y_te, p2), 4), "logloss": round(log_loss(y_te, p2), 4)}}

    # 4. Ficha: métricas por grupo con el umbral 1/6
    yt = y_te.to_numpy()
    grupos = {"todos": np.ones(len(X_te), bool), "hombres": (X_te.sexo == 1).to_numpy(), "mujeres": (X_te.sexo == 2).to_numpy(),
              "menores de 30": (X_te.edad < 30).to_numpy(), "30 a 45": X_te.edad.between(30, 45).to_numpy(),
              "mayores de 45": (X_te.edad > 45).to_numpy()}
    ficha = []
    for g, m in grupos.items():
        s = p2[m] >= 1 / 6
        ficha.append({"grupo": g, "n": int(m.sum()), "prev": round(float(yt[m].mean()), 3),
                      "auc": round(float(roc_auc_score(yt[m], p2[m])), 3), "sel": round(float(s.mean()), 3),
                      "tvp": round(float(s[yt[m] == 1].mean()), 3), "tfp": round(float(s[yt[m] == 0].mean()), 3)})

    return {"semillas": {"bosque": _r(bosque), "pliegues": _r(pliegues)}, "corridas": corridas,
            "duelo": duelo, "ficha": ficha}


if __name__ == "__main__":
    import json
    print(json.dumps(datos(), ensure_ascii=False, indent=1))
