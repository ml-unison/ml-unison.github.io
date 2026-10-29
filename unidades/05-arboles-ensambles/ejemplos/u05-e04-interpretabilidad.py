# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <small>Reconocimiento de Patrones · Licenciatura en Ciencias de la Computación · Universidad de Sonora<br>
# Julio Waissman · <a href="https://ml-unison.github.io">ml-unison.github.io</a> · Licencia CC BY-SA 4.0</small>
#
# ---
#
# # Interpretar un modelo de precios de casas
#
# **Unidad 5 · Ejemplo 4**
#
# A mediados de los años noventa, un proyecto financiado para reducir costos hospitalarios entrenó varios modelos para estimar el riesgo de muerte de pacientes con neumonía; la idea era decidir a quién hospitalizar y a quién mandar a casa con antibióticos (Cooper et al., *Artificial Intelligence in Medicine*, 1997). Uno de los modelos, un sistema de reglas, aprendió que **tener asma reduce el riesgo**. Los datos no mentían: a los pacientes con asma y neumonía se les hospitalizaba de inmediato, muchas veces en terapia intensiva, y ese tratamiento agresivo hacía que murieran menos que el promedio. Pero usar el modelo para decidir habría mandado a casa justo a los pacientes que más atención necesitan. Las redes neuronales del mismo proyecto eran más exactas, y no se usaron porque nadie podía revisar si habían aprendido lo mismo. Veinte años después, R. Caruana y sus colaboradores volvieron a esos datos con un modelo aditivo interpretable y encontraron la misma regla del asma, y otras parecidas (*KDD*, 2015).
#
# La lección: un modelo puede ser **exacto y peligroso a la vez**, y la exactitud medida en datos de prueba no lo delata, porque los datos de prueba tienen el mismo sesgo que los de entrenamiento. Para saberlo hay que **mirar qué aprendió**. En esta libreta vamos a:
#
# 1. entrenar un modelo de *gradient boosting* para el precio de las casas de Ames y compararlo con un modelo lineal, que es interpretable por diseño;
# 2. medir qué atributos importan con la **importancia por permutación**, y ver cómo la engañan los atributos correlacionados;
# 3. ver **cómo** influye cada atributo con las curvas de **dependencia parcial** (PDP) y de **expectativa condicional individual** (ICE), y su problema de extrapolación;
# 4. repartir cada predicción entre los atributos con los **valores de Shapley**: primero a mano con un modelo de juguete, luego exactos por enumeración y verificados contra **SHAP**;
# 5. leer las gráficas de SHAP (cascada, enjambre y dependencia) y discutir lo que una explicación **no** dice.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios) y `shap`. En tu máquina, si usas el entorno del repositorio, ya están instalados.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q shap "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import itertools
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import root_mean_squared_error

import rdp
from rdp.estilo import AZUL, ROJO, DORADO_OSC, GRIS, TENUE, TINTA

rdp.estilo.usar()
SEMILLA = 2011

# %% [markdown]
# ## 1. El problema y los datos
#
# **Datos.** Las ventas de casas en Ames, Iowa (EE. UU.), de 2006 a 2010, que ya usamos en las unidades 1 y 3: 1,460 casas reunidas por Dean De Cock a partir de los registros de la oficina de avalúos de la ciudad (*Journal of Statistics Education*, 2011), en la versión de OpenML. Como en la Unidad 3, quitamos las cuatro casas de más de 4,000 pies² de área habitable, que De Cock recomienda excluir.
#
# **Problema.** Predecir el precio de venta y, sobre todo, **entender qué hace el modelo**: qué atributos usa, cómo cambia la predicción con cada uno y por qué le pone a una casa concreta el precio que le pone. Es lo que preguntaría un valuador, un comprador al que le negaron un crédito hipotecario por el avalúo o un regulador.
#
# Para que las explicaciones se lean fácilmente, usamos solo 12 atributos con nombres en español y unidades métricas (las áreas en m²). El barrio es una categoría con 25 valores; lo convertimos en un número con significado: el **precio mediano de las casas de ese barrio en el conjunto de entrenamiento**, en miles de dólares. Es una codificación por la respuesta (*target encoding*): se calcula solo con entrenamiento, para no filtrar información de prueba.

# %%
casas = rdp.datos.cargar("casas_ames")
casas = casas[casas["GrLivArea"] <= 4000].reset_index(drop=True)
M2 = 0.092903  # m² por pie²
CALIDAD = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

X = pd.DataFrame({
    "area": (casas["GrLivArea"] * M2).round(1),           # área habitable sobre el suelo (m²)
    "calidad": casas["OverallQual"],                       # calidad general de materiales y acabados, 1–10
    "anio": casas["YearBuilt"],                            # año de construcción
    "remodelacion": casas["YearRemodAdd"],                 # año de la última remodelación (= construcción si no hubo)
    "sotano": (casas["TotalBsmtSF"] * M2).round(1),        # área del sótano (m²)
    "cochera": casas["GarageCars"],                        # capacidad de la cochera (autos)
    "terreno": (casas["LotArea"] * M2).round(0),           # área del terreno (m²)
    "banos": casas["FullBath"] + 0.5 * casas["HalfBath"],  # baños sobre el suelo (medio baño = 0.5)
    "chimeneas": casas["Fireplaces"],
    "cocina": casas["KitchenQual"].map(CALIDAD),           # calidad de la cocina, 1–5
    "condicion": casas["OverallCond"],                     # estado de conservación, 1–10
})
barrio = casas["Neighborhood"]
y = np.log(casas["SalePrice"]).to_numpy()

# la misma partición de la Unidad 3: 80 % para entrenar, 20 % de prueba
i_tr, i_te = train_test_split(np.arange(len(casas)), test_size=0.2, random_state=SEMILLA)
mediana_barrio = casas["SalePrice"].iloc[i_tr].groupby(barrio.iloc[i_tr]).median() / 1000
X["barrio"] = barrio.map(mediana_barrio).fillna(mediana_barrio.median()).round(1)

ATRIBUTOS = list(X.columns)
X_tr, X_te, y_tr, y_te = X.iloc[i_tr], X.iloc[i_te], y[i_tr], y[i_te]
print(f"entrenamiento: {len(X_tr):,} casas · prueba: {len(X_te)} casas · {len(ATRIBUTOS)} atributos")
X_tr.describe().T[["mean", "min", "50%", "max"]].round(1)

# %% [markdown]
# Trabajamos, como en la Unidad 3, con el **logaritmo del precio**: los errores se leen como errores relativos y, como veremos, las explicaciones se leen como **factores** que multiplican el precio.

# %% [markdown]
# ## 2. Dos modelos: uno transparente y una «caja negra»
#
# **El modelo lineal.** Una regresión ridge (Unidad 3) con el logaritmo de las áreas y el barrio en *one-hot*. Es interpretable por diseño: cada coeficiente dice cuánto cambia el logaritmo del precio por unidad del atributo, con todo lo demás fijo.
#
# **El modelo de *boosting*.** LightGBM (tema 3 de esta unidad): 600 árboles de 8 hojas con tasa de aprendizaje 0.03. Puede aprender relaciones no lineales e interacciones sin que se las construyamos, pero ya no se puede leer: son 600 árboles.
#
# Comparamos los dos con validación cruzada de 5 particiones en entrenamiento y con el conjunto de prueba.

# %%
def lineal():
    """Ridge con log de las áreas y el barrio original en one-hot."""
    logs = ["area", "sotano", "terreno"]
    otros = [c for c in ATRIBUTOS if c not in logs + ["barrio"]]
    return make_pipeline(
        make_column_transformer(
            (make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), logs),
            (StandardScaler(), otros),
            (OneHotEncoder(handle_unknown="ignore"), ["barrio_nombre"])),
        Ridge(alpha=10.0))


def boosting(semilla=SEMILLA):
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                             random_state=semilla, n_jobs=1, verbose=-1)


X_lin = X.assign(barrio_nombre=barrio)          # el modelo lineal usa el nombre del barrio
cv = KFold(5, shuffle=True, random_state=SEMILLA)
filas = []
for nombre, modelo, XX in [("ridge (lineal)", lineal(), X_lin), ("LightGBM", boosting(), X)]:
    e_cv = -cross_val_score(modelo, XX.iloc[i_tr], y_tr, cv=cv, scoring="neg_root_mean_squared_error")
    modelo.fit(XX.iloc[i_tr], y_tr)
    filas.append({"modelo": nombre, "RECM CV": e_cv.mean(), "EE CV": e_cv.std() / np.sqrt(5),
                  "RECM prueba": root_mean_squared_error(y_te, modelo.predict(XX.iloc[i_te]))})
    if nombre == "LightGBM":
        modelo_gb = modelo
    else:
        modelo_lin = modelo
pd.DataFrame(filas).set_index("modelo").round(4)

# %% [markdown]
# Los dos se equivocan típicamente en un 12 a 14 % del precio, y la diferencia entre ellos es menor que la incertidumbre de la estimación. Con estos datos, **el modelo lineal empata con el de *boosting***.
#
# Éste es justamente el argumento de Cynthia Rudin (*Nature Machine Intelligence*, 2019): en muchos problemas con atributos bien construidos, un modelo interpretable por diseño es igual de exacto que una caja negra, y entonces no hay razón para usar la caja negra y luego tratar de explicarla. Vale la pena comprobarlo **siempre** antes de decidir. Los coeficientes de ridge, llevados a unidades naturales, ya son una explicación completa:

# %%
nombres_lin = modelo_lin[0].get_feature_names_out()
coef = pd.Series(modelo_lin[-1].coef_, index=nombres_lin)
escalas = np.r_[modelo_lin[0].named_transformers_["pipeline"][-1].scale_, modelo_lin[0].named_transformers_["standardscaler"].scale_]
efecto = (coef.iloc[:len(escalas)] / escalas)        # cambio en log(precio) por unidad (o por unidad de log para las áreas)
efecto.index = [n.split("__")[1] for n in efecto.index]
cambio = ["10 % más de área habitable", "10 % más de sótano", "10 % más de terreno", "un punto más de calidad general",
          "un año más reciente", "una remodelación un año más reciente", "lugar para un auto más en la cochera",
          "un baño completo más", "una chimenea más", "un punto más de calidad de la cocina", "un punto más de condición"]
pasos = np.r_[[np.log(1.1)] * 3, np.ones(len(efecto) - 3)]
pd.DataFrame({"cambio": cambio, "efecto en el precio (%)": 100 * (np.exp(efecto.to_numpy() * pasos) - 1)},
             index=efecto.index).round(1)

# %% [markdown]
# Con todo lo demás fijo, cada punto de calidad general sube el precio un 6.7 %; un 10 % más de área habitable, un 3.6 %; un lugar más en la cochera, un 5.2 %; cada año más reciente de construcción, un 0.3 %. (El efecto de los baños sale casi nulo porque, dada el área, más baños no agregan mucho: los baños y el área están muy correlacionados.) No hay que explicar nada más: **el modelo es la explicación**.
#
# Aun así, seguiremos con el modelo de *boosting*, por dos razones. Primera: en muchos problemas (imágenes, textos, datos tabulares con interacciones fuertes) la caja negra sí gana por mucho, y entonces necesitamos herramientas para entenderla. Segunda: las herramientas que veremos sirven para **cualquier** modelo, también para el lineal, y compararlas con lo que ya sabemos del modelo lineal nos ayuda a calibrar qué dicen.
#
# Dos distinciones que conviene tener presentes:
#
# - **Global contra local.** Una explicación global describe el modelo completo («la calidad es lo que más pesa»); una local explica una predicción («esta casa vale 250 mil porque…»).
# - **Por diseño contra *post hoc*.** El modelo lineal es interpretable por diseño. Todo lo que sigue son explicaciones ***post hoc***: se calculan después, mirando el modelo desde afuera, y son **aproximaciones** a lo que hace.

# %% [markdown]
# ## 3. Importancia por permutación
#
# **La pregunta.** ¿Qué tanto **depende** el modelo de cada atributo? Una forma directa de averiguarlo: romper la relación entre el atributo y la respuesta y ver cuánto empeora el modelo. Para romperla sin cambiar la distribución del atributo, **barajamos** su columna en el conjunto de prueba (Breiman, 2001, para los bosques aleatorios; Fisher, Rudin y Dominici, *JMLR*, 2019, para cualquier modelo):
#
# $$I_j = \frac{1}{K}\sum_{k=1}^{K} L\big(\mathbf{y},\, f(X^{\pi_k}_{(j)})\big) - L\big(\mathbf{y},\, f(X)\big),$$
#
# donde $X^{\pi}_{(j)}$ es $X$ con la columna $j$ permutada al azar y $L$ es la pérdida (aquí, la RECM). Si el modelo no usa el atributo, $I_j = 0$. Se calcula **en prueba** porque nos interesa lo que el modelo usa para generalizar, no lo que memorizó (en entrenamiento, un atributo de puro ruido que el modelo memorizó aparece como importante).
#
# Son unas cuantas líneas:

# %%
def importancia_permutacion(modelo, X, y, K=10, semilla=SEMILLA):
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
    return pd.Series(imp).sort_values()


imp = importancia_permutacion(modelo_gb, X_te, y_te)
imp_sk = permutation_importance(modelo_gb, X_te, y_te, n_repeats=10, random_state=SEMILLA,
                                scoring="neg_root_mean_squared_error")
pd.DataFrame({"desde cero": imp, "scikit-learn": pd.Series(imp_sk.importances_mean, index=ATRIBUTOS),
              "± desv. est. (sklearn)": pd.Series(imp_sk.importances_std, index=ATRIBUTOS)}).sort_values("desde cero", ascending=False).round(4)

# %% [markdown]
# Los números de las dos columnas difieren un poco porque las permutaciones al azar son otras, pero el orden es el mismo. Gráficamente:

# %%
def barras_importancia(ax, serie, titulo, color=AZUL, resaltar=()):
    serie = serie.sort_values()
    colores = [ROJO if n in resaltar else color for n in serie.index]
    ax.barh(serie.index, serie.values, color=colores, height=0.65)
    for k, v in enumerate(serie.values):
        ax.text(v + 0.002, k, f"{v:.3f}", va="center", fontsize=10, color=TINTA)
    ax.set_title(titulo)
    ax.grid(axis="y", visible=False)


fig, ax = plt.subplots(figsize=(8.5, 5))
barras_importancia(ax, imp, "Importancia por permutación en prueba (aumento de la RECM)")
ax.set_xlabel("aumento de la RECM en log(precio) al barajar el atributo")
ax.set_xlim(0, imp.max() * 1.15)
plt.show()

# %% [markdown]
# El área habitable y la calidad general dominan: barajar el área sube la RECM de 0.141 a 0.212, un 50 % más de error. Les siguen el sótano, el barrio y el año de construcción. Los baños no importan nada **dado el resto**: el modelo obtiene esa información del área.
#
# ### El problema de los atributos correlacionados
#
# Supongamos que, además del área que registró el valuador, tenemos la que declaró el vendedor: la misma área con un error de medición de alrededor del 3 %. Agregamos esa «gemela» y reentrenamos.

# %%
rng = np.random.default_rng(SEMILLA)
X2 = X.assign(area_vendedor=(X["area"] * np.exp(rng.normal(0, 0.03, len(X)))).round(1))
X2_tr, X2_te = X2.iloc[i_tr], X2.iloc[i_te]
modelo_gemela = boosting().fit(X2_tr, y_tr)
print(f"correlación entre las dos áreas: {X2['area'].corr(X2['area_vendedor']):.3f}")
print(f"RECM de prueba: original {root_mean_squared_error(y_te, modelo_gb.predict(X_te)):.4f}, "
      f"con la gemela {root_mean_squared_error(y_te, modelo_gemela.predict(X2_te)):.4f}")
imp_gemela = importancia_permutacion(modelo_gemela, X2_te, y_te)

fig, ejes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
barras_importancia(ejes[0], imp, "Modelo original")
barras_importancia(ejes[1], imp_gemela, "Con el área declarada por el vendedor", resaltar=("area", "area_vendedor"))
for ax in ejes:
    ax.set_xlabel("aumento de la RECM al barajar")
ejes[0].set_xlim(0, imp.max() * 1.18)
fig.tight_layout()
plt.show()

# %% [markdown]
# El modelo es igual de bueno, pero la importancia del área **se desplomó**: al barajar una de las dos áreas, el modelo todavía tiene la otra, así que pierde poco. Ninguna de las dos parece imprescindible, aunque la información que comparten es la más importante del modelo. La importancia por permutación mide **cuánto se pierde al quitar un atributo dado todos los demás**, no cuánta información aporta.
#
# Hay un segundo problema, más sutil: al barajar el área, se crean casas que no existen (300 m² con un baño, o 60 m² con cochera para tres autos). El modelo nunca vio casas así, y su respuesta en esa región es arbitraria. Lo veremos con más claridad en la siguiente sección.
#
# Remedios: agrupar los atributos correlacionados y permutarlos juntos, o reentrenar sin el atributo (*drop-column importance*, más caro pero sin casas imposibles).

# %% [markdown]
# ## 4. Dependencia parcial y curvas ICE
#
# La importancia dice **cuánto** pesa un atributo, no **cómo**: ¿el precio sube con el área? ¿Siempre al mismo ritmo? La **dependencia parcial** (Friedman, *Annals of Statistics*, 2001) responde fijando el atributo en un valor $x_S$ para **todas** las casas y promediando las predicciones:
#
# $$\hat f_S(x_S) = \frac{1}{N}\sum_{i=1}^{N} \hat f\big(x_S,\, \mathbf{x}_C^{(i)}\big),$$
#
# donde $\mathbf{x}_C^{(i)}$ son los demás atributos de la casa $i$. Es una estimación de $\mathbb{E}_{X_C}\big[\hat f(x_S, X_C)\big]$: el efecto de $x_S$ promediado sobre la distribución **marginal** del resto.
#
# Cada término de la suma, visto como función de $x_S$, es una curva **ICE** (*individual conditional expectation*; Goldstein et al., *JCGS*, 2015): qué le pasaría a la predicción de **esa** casa si solo cambiáramos su área. La PDP es el promedio de las curvas ICE. Si las curvas ICE son paralelas, el atributo no interactúa con los demás; si se cruzan o abren, hay interacciones que el promedio esconde.

# %%
def dependencia_parcial(modelo, X, atributo, malla):
    """Curvas ICE (una fila por casa) y su promedio, la PDP."""
    ice = np.empty((len(X), len(malla)))
    for k, v in enumerate(malla):
        ice[:, k] = modelo.predict(X.assign(**{atributo: v}))
    return ice, ice.mean(axis=0)


malla_area = np.linspace(*np.quantile(X_tr["area"], [0.01, 0.99]), 40)
ice_area, pdp_area = dependencia_parcial(modelo_gb, X_tr, "area", malla_area)
sk = partial_dependence(modelo_gb, X_tr, ["area"], grid_resolution=40, percentiles=(0.01, 0.99), kind="average", method="brute")
print("máxima diferencia con scikit-learn:", np.abs(sk["average"][0] - dependencia_parcial(modelo_gb, X_tr, "area", sk["grid_values"][0])[1]).max())

# %% [markdown]
# Dibujamos la PDP y 60 curvas ICE para el área y para el año de construcción. El eje vertical está en dólares (en escala logarítmica, porque el modelo predice el logaritmo del precio).

# %%
malla_anio = np.linspace(*np.quantile(X_tr["anio"], [0.01, 0.99]), 40)
ice_anio, pdp_anio = dependencia_parcial(modelo_gb, X_tr, "anio", malla_anio)
muestra = np.random.default_rng(1).choice(len(X_tr), 60, replace=False)

fig, ejes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, malla, ice, pdp, nombre in [(ejes[0], malla_area, ice_area, pdp_area, "área habitable (m²)"),
                                    (ejes[1], malla_anio, ice_anio, pdp_anio, "año de construcción")]:
    ax.plot(malla, np.exp(ice[muestra].T) / 1000, color=GRIS, lw=0.8, alpha=0.45)
    ax.plot(malla, np.exp(pdp) / 1000, color=AZUL, lw=3.5)
    ax.set_yscale("log")
    ax.set_xlabel(nombre)
ejes[0].set_yticks([50, 75, 100, 150, 200, 300, 400], ["50", "75", "100", "150", "200", "300", "400"])
ejes[0].yaxis.set_minor_locator(plt.NullLocator())
ejes[0].set_ylim(45, 560)
ejes[0].set_ylabel("precio predicho (miles de dólares)")
ejes[0].text(malla_area[-1], np.exp(pdp_area[-1]) / 1000 * 1.05, "PDP", color=AZUL, ha="right", va="bottom", fontsize=12, weight="semibold")
ejes[0].set_title("Área habitable: 60 curvas ICE (gris) y su promedio")
ejes[1].set_title("Año de construcción")
fig.tight_layout()
plt.show()
f_area = np.exp(pdp_area[-1] - pdp_area[0])
print(f"PDP del área: de {malla_area[0]:.0f} a {malla_area[-1]:.0f} m², el precio se multiplica por {f_area:.2f}")
print(f"PDP del año: de {malla_anio[0]:.0f} a {malla_anio[-1]:.0f}, el precio se multiplica por {np.exp(pdp_anio[-1] - pdp_anio[0]):.2f}")

# %% [markdown]
# Según el modelo, pasar de 64 a 286 m² multiplica el precio por 1.63, con todo lo demás igual. La subida es en escalones (la huella de los árboles, que solo aprenden funciones escalonadas) y se detiene arriba de 260 m²: más allá casi no hay casas, y ningún árbol tiene datos para seguir subiendo. El año de construcción pesa mucho menos: entre 1896 y 2009, el precio sube un factor de 1.23.
#
# Las curvas ICE son casi paralelas en escala logarítmica: el efecto del área es aproximadamente un **factor** que multiplica el precio, más o menos el mismo para todas las casas. Eso es justo lo que supone el modelo lineal en logaritmo, y explica por qué empata.
#
# ### El precio de promediar: casas imposibles
#
# Para calcular $\hat f_S(250\text{ m}^2)$, la PDP le pone 250 m² a **todas** las casas, incluidas las de calidad 4 con un solo baño. ¿Existen casas así?

# %%
grandes = X_tr["area"] >= 250
print(f"casas reales con área ≥ 250 m²: {grandes.sum()}, de ellas con calidad ≤ 5: {(grandes & (X_tr['calidad'] <= 5)).sum()}")
print(f"casas sintéticas que la PDP usa en 250 m² con calidad ≤ 5: {(X_tr['calidad'] <= 5).mean():.0%}")

fig, ax = plt.subplots(figsize=(10, 5))
sacudida = np.random.default_rng(2).uniform(-0.25, 0.25, len(X_tr))
ax.scatter(X_tr["area"], X_tr["calidad"] + sacudida, s=14, color=AZUL, alpha=0.55, lw=0)
for v in [250, 300]:
    ax.scatter(np.full(len(X_tr), v) + np.random.default_rng(v).uniform(-3, 3, len(X_tr)), X_tr["calidad"] + sacudida,
               s=10, color=ROJO, alpha=0.25, lw=0)
ax.text(312, 2.2, "casas sintéticas\nde la PDP", color=ROJO, ha="left", fontsize=11)
ax.set_xlim(20, 370)
ax.set_xlabel("área habitable (m²)")
ax.set_ylabel("calidad general")
ax.set_yticks(range(1, 11))
ax.set_title("Las casas reales (azul) y las que inventa la dependencia parcial (rojo)")
plt.show()

# %% [markdown]
# De las 31 casas reales de 250 m² o más, solo una tiene calidad 5 o menos; pero el 36 % de los puntos que la PDP promedia en 250 m² son así. El modelo **extrapola** en esa región, y la PDP mezcla esas predicciones inventadas con las razonables. Cuando los atributos están correlacionados (área con calidad, con baños, con cochera), la PDP puede describir el comportamiento del modelo en lugares donde no hay datos. Las gráficas de **efectos locales acumulados** (ALE; Apley y Zhu, *JRSS B*, 2020) corrigen esto promediando solo sobre casas cercanas al valor $x_S$.

# %% [markdown]
# ## 5. Valores de Shapley
#
# ### Un juego cooperativo
#
# Ahora la pregunta es local: **¿por qué esta casa vale lo que el modelo dice?** El modelo predice para una casa concreta $\hat f(\mathbf{x})$, y el precio «de referencia», sin saber nada de la casa, es el promedio $\mathbb{E}[\hat f(X)]$. Queremos repartir la diferencia entre los atributos.
#
# Ésa es una pregunta clásica de teoría de juegos. Un **juego cooperativo** es un conjunto de $d$ jugadores y una función $v$ que a cada **coalición** $S \subseteq \{1, \dots, d\}$ le asigna lo que gana, con $v(\varnothing) = 0$ o un valor de referencia. ¿Cómo repartir $v(\{1,\dots,d\}) - v(\varnothing)$ de forma justa? Lloyd Shapley (1953) propuso cuatro propiedades que un reparto $\phi_1, \dots, \phi_d$ debería cumplir:
#
# 1. **Eficiencia:** se reparte todo, $\sum_j \phi_j = v(\{1,\dots,d\}) - v(\varnothing)$.
# 2. **Simetría:** si $j$ y $k$ aportan lo mismo a toda coalición, $\phi_j = \phi_k$.
# 3. **Jugador nulo:** si $j$ no aporta nada a ninguna coalición, $\phi_j = 0$.
# 4. **Aditividad:** el reparto del juego $v + w$ es la suma de los repartos de $v$ y de $w$.
#
# y demostró que **hay un único reparto** que las cumple:
#
# $$\phi_j = \sum_{S \subseteq \{1,\dots,d\} \setminus \{j\}} \frac{|S|!\,(d - |S| - 1)!}{d!}\,\big[v(S \cup \{j\}) - v(S)\big].$$
#
# La fórmula tiene una lectura sencilla. Imaginemos que los jugadores llegan **uno por uno**, en un orden al azar, y que a cada uno le pagamos lo que su llegada aumenta el valor de la coalición que ya estaba: su **contribución marginal**. El valor de Shapley es el pago **promedio sobre los $d!$ órdenes**. El peso $|S|!(d-|S|-1)!/d!$ es la fracción de órdenes en los que, justo antes de $j$, llegaron exactamente los jugadores de $S$.
#
# ### Un modelo de juguete, a mano
#
# Para ver los números, entrenemos un modelo con solo tres atributos (área, calidad y año), esta vez directamente sobre el precio en dólares para que las cuentas se lean en miles de dólares. Los jugadores son los atributos. ¿Qué gana una coalición? Definimos
#
# $$v(S) = \mathbb{E}\big[\hat f(\mathbf{x}_S, X_C)\big] \approx \frac{1}{N}\sum_{i=1}^{N} \hat f\big(\mathbf{x}_S,\, \mathbf{x}_C^{(i)}\big):$$
#
# la predicción promedio cuando conocemos los atributos de $S$ de **nuestra** casa y los demás los tomamos de un conjunto de casas de referencia, el **fondo**: aquí, 100 casas de entrenamiento elegidas al azar. Es la misma idea de la dependencia parcial, ahora con varios atributos a la vez. Con $S = \varnothing$ obtenemos la predicción promedio; con todos los atributos, la predicción de nuestra casa.

# %%
fondo = X_tr.sample(100, random_state=SEMILLA)   # 100 casas de entrenamiento como «fondo»
JUGUETE = ["area", "calidad", "anio"]
modelo_juguete = boosting().fit(X_tr[JUGUETE], casas["SalePrice"].iloc[i_tr] / 1000)
casa = X_te.loc[[X_te.index[np.argmin(np.abs(X_te["area"] - 180) + 10 * np.abs(X_te["calidad"] - 8) + np.abs(X_te["anio"] - 2005))]]]
print(casa[JUGUETE])


def valor_coalicion(modelo, fondo, x, S):
    """v(S): promedio de las predicciones con los atributos de S tomados de x y el resto del fondo."""
    sintetico = fondo.copy()
    for c in S:
        sintetico[c] = x[c].iloc[0]
    return modelo.predict(sintetico).mean()


coaliciones = [S for k in range(4) for S in itertools.combinations(JUGUETE, k)]
v = {frozenset(S): valor_coalicion(modelo_juguete, fondo[JUGUETE], casa[JUGUETE], S) for S in coaliciones}
pd.Series({("{" + ", ".join(S) + "}" if S else "∅"): v[frozenset(S)] for S in coaliciones}, name="v(S), miles de dólares").round(1)

# %% [markdown]
# Ahora los $3! = 6$ órdenes de llegada, y la contribución marginal de cada atributo en cada orden:

# %%
ordenes = []
for orden in itertools.permutations(JUGUETE):
    llegados, fila = frozenset(), {}
    for j in orden:
        fila[j] = v[llegados | {j}] - v[llegados]
        llegados = llegados | {j}
    ordenes.append(pd.Series(fila, name=" → ".join(orden)))
tabla_ordenes = pd.DataFrame(ordenes)[JUGUETE]
tabla_ordenes.loc["promedio = Shapley"] = tabla_ordenes.mean()
tabla_ordenes.round(1)

# %% [markdown]
# Cada renglón reparte la misma cantidad, $v(\{\text{todos}\}) - v(\varnothing)$, pero de forma distinta: el atributo que llega primero se lleva su efecto «solo», y el que llega después se lleva lo que agrega dado lo que ya se sabía. El área aporta 20 mil dólares si llega primero y 41 mil si llega al último: en dólares, el efecto del área crece con la calidad (un metro cuadrado de calidad 8 vale más que uno de calidad 5), así que saber el área agrega más cuando ya se sabe que la casa es buena. Ninguno de los seis órdenes es «el correcto»; el valor de Shapley los promedia.
#
# Verificamos la eficiencia, la fórmula con pesos y el resultado de la biblioteca `shap`, que calcula lo mismo con un algoritmo para árboles (más adelante explicamos cómo):

# %%
def shapley_exacto(v, jugadores):
    d = len(jugadores)
    phi = {}
    for j in jugadores:
        otros = [k for k in jugadores if k != j]
        phi[j] = sum(math.factorial(len(S)) * math.factorial(d - len(S) - 1) / math.factorial(d)
                     * (v[frozenset(S) | {j}] - v[frozenset(S)])
                     for r in range(d) for S in itertools.combinations(otros, r))
    return pd.Series(phi)


phi_juguete = shapley_exacto(v, JUGUETE)
exp_juguete = shap.TreeExplainer(modelo_juguete, data=fondo[JUGUETE], feature_perturbation="interventional")
print("fórmula:      ", phi_juguete.round(2).to_dict())
print("shap:         ", dict(zip(JUGUETE, exp_juguete.shap_values(casa[JUGUETE])[0].round(2))))
print(f"eficiencia:    {v[frozenset()]:.1f} + {phi_juguete.sum():.1f} = {v[frozenset()] + phi_juguete.sum():.1f}"
      f" = predicción {modelo_juguete.predict(casa[JUGUETE])[0]:.1f}")

# %% [markdown]
# ## 6. SHAP: Shapley para explicar predicciones
#
# Lundberg y Lee (*NeurIPS*, 2017) propusieron usar exactamente este juego para explicar cualquier modelo, con el nombre de **SHAP** (*SHapley Additive exPlanations*). La explicación de la predicción de $\mathbf{x}$ es aditiva:
#
# $$\hat f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{d} \phi_j(\mathbf{x}), \qquad \phi_0 = \mathbb{E}[\hat f(X)],$$
#
# y es la **única** explicación aditiva que cumple los axiomas de Shapley para el juego $v_{\mathbf{x}}(S) = \mathbb{E}\big[\hat f(\mathbf{x}_S, X_C)\big]$. Esta versión, que toma $X_C$ de su distribución marginal (de un conjunto de casas de **fondo**), se llama **intervencional**: es lo que pasaría si **interviniéramos** los atributos de $S$, fijándolos, sin cambiar los demás.
#
# **Un caso que se puede hacer a mano: el modelo lineal.** Si $\hat f(\mathbf{x}) = w_0 + \sum_k w_k x_k$, entonces $v(S) = w_0 + \sum_{k \in S} w_k x_k + \sum_{k \notin S} w_k\,\mathbb{E}[X_k]$, y la contribución marginal de $j$ a **cualquier** coalición es la misma, $w_j(x_j - \mathbb{E}[X_j])$. Promediar números iguales da ese número:
#
# $$\phi_j = w_j\,\big(x_j - \mathbb{E}[X_j]\big).$$
#
# Para un modelo lineal, SHAP es «coeficiente por desviación respecto del promedio», que es lo que cualquiera habría propuesto. Lo valioso es que la misma definición se extiende a modelos no lineales sin cambiar de idea.
#
# ### El costo: $2^d$ coaliciones
#
# Con los 12 atributos del modelo completo hay $2^{12} = 4{,}096$ coaliciones. Con un fondo de 100 casas, la enumeración exacta necesita 409,600 predicciones por casa explicada. Es factible aquí, y nos sirve para verificar a `shap`:

# %%
def shap_por_enumeracion(modelo, fondo, x):
    """Valores de Shapley exactos del juego intervencional, enumerando las 2^d coaliciones."""
    d = fondo.shape[1]
    mascaras = np.array(list(itertools.product([False, True], repeat=d)))   # una fila por coalición
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


k_casa = 45                       # una casa de prueba con razones a favor y en contra
casa = X_te.iloc[[k_casa]]
t0 = time.perf_counter()
phi_enum, phi0_enum = shap_por_enumeracion(modelo_gb, fondo, casa)
t_enum = time.perf_counter() - t0

explicador = shap.TreeExplainer(modelo_gb, data=fondo, feature_perturbation="interventional")
t0 = time.perf_counter()
phi_shap = explicador.shap_values(casa)[0]
t_shap = time.perf_counter() - t0
print(f"enumeración: {t_enum:.2f} s · TreeSHAP: {1000 * t_shap:.1f} ms · máxima diferencia: {np.abs(phi_enum - phi_shap).max():.1e}")
print(f"φ0 = {phi0_enum:.4f} (enumeración) y {explicador.expected_value:.4f} (shap)")

# %% [markdown]
# Coinciden hasta el redondeo de la máquina, y `shap` es cientos de veces más rápido. El truco es que no necesita enumerar coaliciones: para cada pareja (casa explicada, casa del fondo) recorre cada árbol una vez y, como cada camino de la raíz a una hoja solo usa unos pocos atributos, puede contar con fórmulas combinatorias en cuántas coaliciones se llega a cada hoja. El algoritmo **TreeSHAP** (Lundberg et al., *Nature Machine Intelligence*, 2020) calcula así los valores exactos en tiempo polinomial, proporcional al número de árboles, de hojas y de casas del fondo. Para modelos que no son árboles, `shap` usa aproximaciones por muestreo de coaliciones (KernelSHAP), mucho más lentas.
#
# Una nota técnica: si a `TreeExplainer` no se le da un fondo, usa la versión `tree_path_dependent`, que en lugar de promediar sobre casas del fondo usa cuántas casas de entrenamiento pasaron por cada rama. Es más rápida pero aproxima otro juego (más parecido a condicionar en $X_S = \mathbf{x}_S$), así que sus valores difieren un poco. Aquí usamos siempre la intervencional.
#
# Calculamos los valores SHAP de las 292 casas de prueba y verificamos la eficiencia, $\phi_0 + \sum_j \phi_j = \hat f(\mathbf{x})$:

# %%
sv = explicador.shap_values(X_te)
phi0 = explicador.expected_value
error_eficiencia = np.abs(phi0 + sv.sum(axis=1) - modelo_gb.predict(X_te))
print(f"φ0 = {phi0:.3f}, es decir, un precio de referencia de {np.exp(phi0) / 1000:.1f} mil dólares")
print(f"casas con |φ0 + Σφj − f(x)| < 1e-6: {(error_eficiencia < 1e-6).sum()} de {len(X_te)}; máximo: {error_eficiencia.max():.1e}")

# %% [markdown]
# La suma cuadra en todas las casas menos una, donde la diferencia es de 0.008 en el logaritmo (menos de 1 % del precio). Para esa casa, nuestra enumeración exacta sí cumple la eficiencia: es un detalle numérico de la implementación de `shap`, no del método. Conviene saberlo: **verificar la eficiencia es una prueba barata** de que las explicaciones están bien calculadas.
#
# ### Una casa: la gráfica de cascada
#
# Como el modelo predice el **logaritmo** del precio, los $\phi_j$ se suman en logaritmo y, en dólares, **multiplican**: $\text{precio} = e^{\phi_0}\, e^{\phi_1} \cdots e^{\phi_d}$. Un $\phi_j = 0.10$ quiere decir que ese atributo sube el precio predicho un factor $e^{0.10} = 1.105$, un 10.5 %. Dibujamos la cascada en un eje logarítmico rotulado en dólares, así las barras se suman y el eje se lee en dinero.

# %%
NOMBRES = {"area": "área habitable", "calidad": "calidad general", "anio": "año de construcción",
           "remodelacion": "año de remodelación", "sotano": "sótano", "cochera": "cochera", "terreno": "terreno",
           "banos": "baños", "chimeneas": "chimeneas", "cocina": "cocina", "condicion": "condición",
           "barrio": "barrio", "area_vendedor": "área (vendedor)"}
UNIDADES = {"area": " m²", "sotano": " m²", "terreno": " m²", "cochera": " autos", "barrio": " mil USD (mediana)"}


def cascada(ax, phi, x, phi0, k=7):
    orden = np.argsort(-np.abs(phi))
    principales, resto = orden[:k], orden[k:]
    pasos = [(f"{NOMBRES[c]} = {x[c]:g}{UNIDADES.get(c, '')}", phi[j]) for j, c in ((j, ATRIBUTOS[j]) for j in principales)]
    pasos.append((f"otros {len(resto)} atributos", phi[resto].sum()))
    nivel = phi0
    for fila, (texto, p) in enumerate(pasos):
        ax.barh(fila, p, left=nivel, color=ROJO if p > 0 else AZUL, height=0.6)
        ax.text(nivel + p + (0.008 if p > 0 else -0.008), fila, f"×{np.exp(p):.2f}", va="center",
                ha="left" if p > 0 else "right", fontsize=10, color=TINTA)
        nivel += p
    ax.set_yticks(range(len(pasos)), [t for t, _ in pasos])
    ax.invert_yaxis()
    ax.axvline(phi0, color=TENUE, ls="--", lw=1)
    ax.axvline(nivel, color=TINTA, lw=1.2)
    marcas = np.array([100, 125, 150, 200, 250, 300, 400])
    ax.set_xlim(np.log(1000 * marcas[0]), np.log(1000 * marcas[-1]))
    ax.set_xticks(np.log(marcas * 1000), [f"{m}" for m in marcas])
    ax.set_xlabel("precio (miles de dólares, escala logarítmica)")
    ax.grid(axis="y", visible=False)
    return nivel


fig, ax = plt.subplots(figsize=(10, 5))
final = cascada(ax, sv[k_casa], X_te.iloc[k_casa], phi0)
ax.set_title(f"Por qué el modelo le pone {np.exp(final) / 1000:.0f} mil dólares a esta casa "
             f"(referencia: {np.exp(phi0) / 1000:.0f} mil; precio real: {np.exp(y_te[k_casa]) / 1000:.0f} mil)")
plt.show()

# %% [markdown]
# Se lee de arriba abajo: partiendo del precio de referencia (línea punteada), cada atributo multiplica la predicción por el factor anotado, hasta llegar a la predicción (línea continua). Es una casa grande (245 m²) y muy antigua (1880) en un barrio barato. El área la encarece un 34 % respecto de la casa de referencia y la calidad 7 un 11 %; la antigüedad la abarata un 11 % y el barrio un 5 %. Una explicación así es la que se le podría dar a un comprador o a un valuador, y ellos podrían discutirla: «la casa se remodeló en 2003; ¿por qué eso casi no cuenta?» (la remodelación aporta un factor de 1.02 y quedó entre los «otros»).
#
# Los factores son exactos y no dependen del orden de las barras; en dólares, en cambio, cuánto «suma» cada atributo sí dependería del orden en que se apliquen los factores, y por eso preferimos el eje logarítmico.
#
# ### Todas las casas: el enjambre
#
# Una explicación global sale de juntar las locales. En la gráfica de **enjambre** (*beeswarm*), cada punto es una casa de prueba, su posición horizontal es el $\phi_j$ de esa casa y su color, el valor del atributo (de bajo, azul, a alto, rojo). Los atributos se ordenan por $\frac{1}{N}\sum_i |\phi_j(\mathbf{x}_i)|$, la importancia SHAP media.

# %%
def enjambre_y(valores, ancho=0.4, semilla=0):
    """Desplazamiento vertical al azar, proporcional a la densidad local: los puntos se abren donde hay muchos."""
    h = 0.9 * valores.std() * len(valores) ** -0.2 + 1e-9                     # ancho de banda de Silverman
    dens = np.exp(-0.5 * ((valores[:, None] - valores[None, :]) / h) ** 2).sum(axis=1)
    return np.random.default_rng(semilla).uniform(-1, 1, len(valores)) * ancho * dens / dens.max()


importancia_shap = pd.Series(np.abs(sv).mean(axis=0), index=ATRIBUTOS).sort_values()
fig, ax = plt.subplots(figsize=(10, 6.2))
for fila, c in enumerate(importancia_shap.index):
    j = ATRIBUTOS.index(c)
    valores = X_te[c].to_numpy(dtype=float)
    rango = (valores - np.quantile(valores, 0.05)) / (np.quantile(valores, 0.95) - np.quantile(valores, 0.05) + 1e-9)
    ax.scatter(sv[:, j], fila + enjambre_y(sv[:, j]), c=np.clip(rango, 0, 1), cmap=rdp.estilo.DIVERGENTE.reversed(),
               s=12, lw=0)
ax.axvline(0, color=TENUE, lw=1)
ax.set_yticks(range(len(importancia_shap)), [NOMBRES[c] for c in importancia_shap.index])
ax.set_xlabel("valor SHAP (cambio en log(precio))")
ax.set_title("Valores SHAP de las 292 casas de prueba (color: valor del atributo, azul bajo · rojo alto)")
ax.grid(axis="y", visible=False)
plt.show()

# %% [markdown]
# Se ven tres cosas que ninguna tabla de importancias muestra:
#
# - **La dirección.** En el área y la calidad, los puntos rojos (valores altos) están a la derecha: suben el precio. En el año de construcción también, aunque con menos fuerza.
# - **La forma.** El área tiene una cola larga a la derecha: unas pocas casas muy grandes reciben factores de hasta $e^{0.4} \approx 1.5$, y las chicas, castigos parecidos. El terreno tiene una cola a la izquierda: un terreno muy chico castiga más de lo que premia uno grande.
# - **La dispersión.** En la cochera, el mismo valor del atributo recibe valores SHAP distintos según la casa: es la huella de las **interacciones**.
#
# El orden se parece al de la importancia por permutación, pero no es el mismo: una mide cuánto **empeora el error** sin el atributo, la otra cuánto **mueve la predicción**.

# %%
comparacion = pd.DataFrame({"SHAP media |φ|": importancia_shap, "permutación": imp}).sort_values("SHAP media |φ|", ascending=False)
comparacion.round(4)

# %% [markdown]
# ### Un atributo: la gráfica de dependencia
#
# Si dibujamos $\phi_{\text{área}}$ contra el área, una casa por punto, obtenemos algo parecido a la dependencia parcial, pero con datos **reales** (no inventados) y con la dispersión vertical a la vista. La coloreamos por la calidad, para buscar una interacción.

# %%
j_area = ATRIBUTOS.index("area")
fig, ax = plt.subplots(figsize=(10, 5))
puntos = ax.scatter(X_te["area"], sv[:, j_area], c=X_te["calidad"], cmap=rdp.estilo.DIVERGENTE.reversed(), s=26, lw=0)
# la PDP se desplaza para que su promedio sobre las casas de prueba coincida con el de los valores SHAP
ax.plot(malla_area, pdp_area - np.interp(X_te["area"], malla_area, pdp_area).mean() + sv[:, j_area].mean(), color=TINTA, lw=2, ls="--")
ax.axhline(0, color=TENUE, lw=1)
fig.colorbar(puntos, ax=ax, label="calidad general", pad=0.01)
ax.set_xlabel("área habitable (m²)")
ax.set_ylabel("valor SHAP del área")
ax.set_title("Dependencia SHAP del área (punteada: dependencia parcial, desplazada)")
plt.show()

# %% [markdown]
# La nube sigue a la dependencia parcial: el área empuja el precio hacia abajo por debajo de unos 140 m² (cerca del área promedio) y hacia arriba por encima. Pero arriba de 180 m² aparece algo que la PDP no muestra: las casas de mejor calidad (rojas) quedan **por encima** de la curva y las de calidad media, por debajo. El modelo aprendió una **interacción** leve: en una casa de buena calidad, cada metro cuadrado vale un poco más. Por debajo de 150 m², en cambio, los colores se mezclan y los dos efectos son casi independientes, como en el modelo lineal en logaritmo.

# %% [markdown]
# ## 7. Lo que una explicación no dice
#
# ### Explicar el modelo no es explicar el mundo
#
# Los valores SHAP describen **la función $\hat f$**, no el mecanismo que produce los precios. Que $\phi_{\text{chimeneas}} > 0$ para las casas con chimenea no quiere decir que construir una chimenea suba el precio en ese factor: tener chimenea va junto con otras cosas que el modelo no ve (el estilo de la casa, el cuidado de los dueños). Es el mismo error que habría cometido el hospital del asma si hubiera leído «el asma protege» como un hecho médico. Para afirmaciones causales se necesitan experimentos o un modelo causal explícito, no un modelo predictivo.
#
# ### Dos modelos igual de buenos, dos explicaciones distintas
#
# Volvamos al modelo con el área declarada por el vendedor, que era igual de exacto. ¿Qué dice SHAP?

# %%
explicador2 = shap.TreeExplainer(modelo_gemela, data=X2_tr.sample(100, random_state=SEMILLA), feature_perturbation="interventional")
sv2 = explicador2.shap_values(X2_te)
gemelas = pd.DataFrame({
    "modelo original": pd.Series(np.abs(sv).mean(axis=0), index=ATRIBUTOS),
    "con la gemela": pd.Series(np.abs(sv2).mean(axis=0), index=X2.columns),
}).loc[["area", "area_vendedor", "calidad", "barrio"]]
gemelas.loc["área + gemela"] = [np.abs(sv[:, j_area]).mean(), np.abs(sv2[:, [ATRIBUTOS.index("area"), X2.columns.get_loc("area_vendedor")]].sum(axis=1)).mean()]
gemelas.round(4)

# %% [markdown]
# El segundo modelo reparte el efecto del área entre las dos columnas, de manera más o menos arbitraria (depende de qué columna eligió cada árbol al dividir). Sumadas, recuperan el efecto original. Una explicación es de **un** modelo: si hay muchos modelos igual de buenos (lo que Breiman llamó el **efecto Rashomon**), sus explicaciones pueden ser muy distintas, y ninguna es «la» explicación de los datos.
#
# ### Las explicaciones se pueden manipular
#
# SHAP intervencional, PDP e importancia por permutación evalúan el modelo en puntos **sintéticos**, que pueden no parecerse a ningún dato real. Slack et al. (*AIES*, 2020) aprovecharon esto: construyeron clasificadores que discriminan abiertamente por raza en los datos reales, pero que se comportan de forma inocente en los puntos sintéticos que usan LIME y SHAP para explicar. Las explicaciones resultantes no muestran la raza entre los atributos importantes. Una explicación *post hoc* no es una auditoría; si lo que está en juego es alto (crédito, justicia, salud), Rudin tiene razón: mejor un modelo que se pueda leer directamente.
#
# ## 8. Cierre
#
# | Pregunta | Herramienta | Cuidado |
# |---|---|---|
# | ¿Qué atributos usa el modelo? | importancia por permutación (en prueba), media de $\lvert\phi_j\rvert$ | los atributos correlacionados se reparten (o se esconden) la importancia |
# | ¿Cómo cambia la predicción con un atributo? | dependencia parcial y curvas ICE; dependencia SHAP | la PDP promedia sobre casas imposibles cuando hay correlación |
# | ¿Por qué esta predicción? | valores SHAP, cascada | explica el modelo, no el mundo; hay muchos modelos igual de buenos |
# | ¿Necesito una caja negra? | comparar con un modelo interpretable por diseño | si empatan, usar el interpretable |
#
# *Referencias:* L. S. Shapley, «A value for n-person games», en *Contributions to the Theory of Games II* (1953). J. H. Friedman, «Greedy function approximation: a gradient boosting machine», *Annals of Statistics* 29 (2001). S. M. Lundberg y S.-I. Lee, «A unified approach to interpreting model predictions», *NeurIPS* (2017). S. M. Lundberg et al., «From local explanations to global understanding with explainable AI for trees», *Nature Machine Intelligence* 2 (2020). C. Rudin, «Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead», *Nature Machine Intelligence* 1 (2019). R. Caruana et al., «Intelligible models for healthcare», *KDD* (2015). D. Slack et al., «Fooling LIME and SHAP», *AIES* (2020). C. Molnar, *Interpretable Machine Learning*, 2.ª ed. (2022), en línea.

# %% [markdown]
# ## 9. Autoevaluación

# %%
rdp.quiz.mostrar("u05", ids=["u05-p13", "u05-p14", "u05-p15", "u05-p16"])

# %% [markdown]
# ## 10. Ejercicios
#
# 1. **Shapley de un juego de tres jugadores.** En el juego de los guantes, el jugador 1 tiene un guante izquierdo y los jugadores 2 y 3 tienen uno derecho cada uno; una coalición gana 1 si puede formar un par y 0 si no. Calcula los valores de Shapley con la fórmula y con los seis órdenes. ¿Por qué el jugador 1 recibe más que los otros dos? Verifica los cuatro axiomas.
# 2. **SHAP de un modelo lineal.** Demuestra que, para $\hat f(\mathbf{x}) = w_0 + \mathbf{w}^\top\mathbf{x}$ y el juego intervencional, $\phi_j = w_j(x_j - \mathbb{E}[X_j])$ (completa el argumento de la sección 6). Después, verifícalo numéricamente con el modelo ridge de la sección 2: `shap.LinearExplainer` o tu propia función de enumeración, aplicada al modelo completo (con su `Pipeline`) sobre las columnas originales. ¿Qué pasa con el logaritmo de las áreas?
# 3. **Importancia por grupos.** Modifica `importancia_permutacion` para que reciba grupos de columnas y las baraje **juntas** (con la misma permutación). Aplícala al modelo con la gemela, con el grupo {área, área del vendedor}. ¿Recuperas la importancia del modelo original?
# 4. **ALE contra PDP.** Implementa las gráficas de efectos locales acumulados para el área: divide el rango en 20 intervalos por cuantiles, y en cada intervalo promedia, **solo sobre las casas que caen en él**, la diferencia de predicción entre el extremo derecho y el izquierdo; luego acumula. Compárala con la PDP. ¿Dónde difieren más y por qué?
# 5. **Rashomon.** Entrena 10 modelos LightGBM con semillas distintas y `subsample=0.8, subsample_freq=1, colsample_bytree=0.8`. Para la casa de la cascada, ¿cuánto varían el $\phi_j$ de cada atributo entre modelos? ¿Y la RECM de prueba? Compara la variabilidad de las explicaciones con la de las predicciones.
