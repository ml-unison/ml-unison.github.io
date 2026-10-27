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
# # Validación: estimar $E_{\text{out}}$ sin hacer trampa
#
# **Unidad 3 · Ejemplo 3**
#
# La cota VC nos dice que $E_{\text{in}}$ se acerca a $E_{\text{out}}$ cuando hay suficientes datos, pero la cota es tan holgada que casi nunca sirve para dar un número. En la práctica, la pregunta «¿qué tan bien le irá a mi modelo con datos nuevos?» se contesta de otra manera: **apartando datos que el modelo no vio** y midiendo ahí.
#
# La idea es sencilla; lo difícil es no hacer trampa sin darse cuenta. En esta libreta vamos a:
#
# 1. ver por qué el error en un conjunto de validación es un buen estimador de $E_{\text{out}}$, y cuánto varía según su tamaño;
# 2. comprobar que en cuanto usamos la validación para **elegir**, el estimado deja de ser honesto;
# 3. estudiar la validación cruzada y dejar-uno-fuera, incluido un atajo exacto para la regresión ridge;
# 4. elegir un hiperparámetro con validación cruzada, con la regla de una desviación estándar, y estimar honestamente el error de todo el procedimiento con validación cruzada anidada;
# 5. ver cómo los datos **agrupados** engañan a la validación cruzada común.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado.

# %%
import os, sys

RAMA = "rediseno-2027"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (train_test_split, cross_val_score, KFold, GroupKFold,
                                     LeaveOneOut, GridSearchCV)

import rdp
from rdp.estilo import AZUL, ROJO, DORADO_OSC, GRIS, TENUE, TINTA

rdp.estilo.usar()
SEMILLA = 2011
rng = np.random.default_rng(SEMILLA)

# %% [markdown]
# ## 1. El problema y los datos
#
# **Problema.** El mismo de la Unidad 1: estimar el precio de venta de una casa a partir de sus características. Pero ahora la pregunta no es cómo construir el modelo, sino **cuánto se va a equivocar** con casas que todavía no se venden, y cómo saberlo sin engañarnos.
#
# **Datos.** Ventas de casas en Ames, Iowa (EE. UU.) entre 2006 y 2010, reunidas por Dean De Cock a partir de los registros de la oficina de avalúos de la ciudad (*Journal of Statistics Education*, 2011), en la versión de 1,460 casas de OpenML. Como recomienda De Cock, quitamos las casas de más de 4,000 pies² de área habitable (son cuatro, y dos de ellas son ventas parciales que no representan el mercado).

# %%
casas = rdp.datos.cargar("casas_ames")
casas = casas[casas["GrLivArea"] <= 4000].reset_index(drop=True)
print(casas.shape)

# %% [markdown]
# Para que el modelo no distraiga, usamos una versión sencilla de lo que hicimos en la Unidad 1: unas 15 variables numéricas (con logaritmo en las áreas y las edades de la casa en lugar de los años), las calidades en escala ordinal (0 si no hay sótano o chimenea) y tres variables categóricas en *one-hot*. Todas estas transformaciones son **fijas**: no aprenden nada de los datos (ni medias, ni medianas, ni vocabularios que dependan de la muestra), así que podemos aplicarlas a toda la tabla de una vez sin contaminar la evaluación. Lo único que sí aprende de los datos, la estandarización, va dentro del `Pipeline` del modelo.
#
# El objetivo es el **logaritmo del precio**, y medimos el error con el error cuadrático:
#
# $$e\big(h(\mathbf{x}), y\big) = \big(h(\mathbf{x}) - y\big)^2, \qquad y = \ln(\text{precio}).$$
#
# Una ventaja de usar el logaritmo: la raíz del error cuadrático medio se lee aproximadamente como un error **relativo**. Una raíz de 0.12 quiere decir que nos equivocamos típicamente en un 12 % del precio.

# %%
CALIDAD = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

X = pd.DataFrame({
    "log_area": np.log(casas["GrLivArea"]),
    "log_terreno": np.log(casas["LotArea"]),
    "sotano": casas["TotalBsmtSF"] / 1000,
    "cochera": casas["GarageArea"] / 1000,
    "calidad": casas["OverallQual"],
    "condicion": casas["OverallCond"],
    "recamaras": casas["BedroomAbvGr"],
    "banos": casas["FullBath"] + 0.5 * casas["HalfBath"],
    "chimeneas": casas["Fireplaces"],
    "edad": casas["YrSold"] - casas["YearBuilt"],
    "desde_remodelacion": casas["YrSold"] - casas["YearRemodAdd"],
    "aire": (casas["CentralAir"] == "Y").astype(int),
})
for c in ["ExterQual", "KitchenQual", "BsmtQual", "FireplaceQu"]:
    X["cal_" + c] = casas[c].map(CALIDAD).fillna(0)
NUMERICAS = list(X.columns)
X = pd.concat([X, pd.get_dummies(casas[["Neighborhood", "MSZoning", "BldgType"]], dtype=float)], axis=1)
y = np.log(casas["SalePrice"]).to_numpy()
print(f"{X.shape[1]} columnas ({len(NUMERICAS)} numéricas y {X.shape[1] - len(NUMERICAS)} indicadoras)")

# %% [markdown]
# ### El conjunto de prueba se aparta primero
#
# Antes de hacer nada, apartamos el 20 % de las casas como **conjunto de prueba**. No lo vamos a mirar hasta la sección 5, cuando tengamos un modelo final. Todo lo demás (los experimentos, las simulaciones, la elección de hiperparámetros) se hace con el 80 % restante, al que llamaremos $\mathcal{D}$.

# %%
X_d, X_prueba, y_d, y_prueba = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
N = len(y_d)
print(f"D: {N} casas · prueba: {len(y_prueba)} casas (guardadas)")


def ridge(alpha=10.0):
    """El modelo de las secciones 2 a 4: estandarizar y regresión ridge."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def ecm(modelo, X, y):
    """Error cuadrático medio de un modelo ya entrenado."""
    return mean_squared_error(y, modelo.predict(X))


# %% [markdown]
# ## 2. Un conjunto de validación
#
# ### La idea
#
# Partimos $\mathcal{D}$ (con $N$ ejemplos) en dos: un conjunto de **entrenamiento** $\mathcal{D}_{\text{ent}}$ con $N-K$ ejemplos y uno de **validación** $\mathcal{D}_{\text{val}}$ con $K$. Entrenamos con el primero y obtenemos una hipótesis que llamaremos $g^-$ (el signo menos recuerda que se entrenó con menos datos que los disponibles). Luego medimos:
#
# $$E_{\text{val}}(g^-) = \frac{1}{K}\sum_{(\mathbf{x}_n, y_n)\in \mathcal{D}_{\text{val}}} e\big(g^-(\mathbf{x}_n), y_n\big).$$
#
# **¿Por qué funciona?** Porque los ejemplos de validación no participaron en la construcción de $g^-$. Condicionado a $g^-$, cada término $e_n = e(g^-(\mathbf{x}_n), y_n)$ es una variable aleatoria independiente con esperanza
#
# $$\mathbb{E}[e_n] = \mathbb{E}_{\mathbf{x}, y}\big[e(g^-(\mathbf{x}), y)\big] = E_{\text{out}}(g^-).$$
#
# Por linealidad de la esperanza, $\mathbb{E}[E_{\text{val}}(g^-)] = E_{\text{out}}(g^-)$: el estimador es **insesgado**. Y como es un promedio de $K$ variables independientes con varianza $\sigma^2 = \operatorname{Var}[e_n]$,
#
# $$\operatorname{Var}\big[E_{\text{val}}(g^-)\big] = \frac{\sigma^2}{K}, \qquad \text{es decir, su error típico es } \frac{\sigma}{\sqrt{K}}.$$
#
# Es exactamente el mismo razonamiento que la desigualdad de Hoeffding con **una sola** hipótesis: no hay unión de eventos, porque $g^-$ quedó fija antes de mirar la validación.
#
# ### Verlo con las casas
#
# Para comprobarlo necesitamos conocer $E_{\text{out}}$, que en la vida real nunca conocemos. Hacemos un pequeño laboratorio dentro de $\mathcal{D}$: entrenamos $g^-$ con 400 casas y tratamos a las demás como si fueran **la población completa**. El error promedio en ellas es, por definición, el $E_{\text{out}}$ de nuestro laboratorio. Después sacamos de esa población muchos conjuntos de validación de tamaño $K$ (con reemplazo, como si muestreáramos de la población) y vemos cuánto varía $E_{\text{val}}$.

# %%
orden = rng.permutation(N)
ent, poblacion = orden[:400], orden[400:]
g_menos = ridge().fit(X_d.iloc[ent], y_d[ent])
errores = (g_menos.predict(X_d.iloc[poblacion]) - y_d[poblacion]) ** 2
E_out, sigma = errores.mean(), errores.std()
print(f"E_out(g-) = {E_out:.4f}   (raíz: {np.sqrt(E_out):.3f})   sigma = {sigma:.4f}")

tamanos = np.array([10, 20, 50, 100, 200, 500, 1000])
E_val = {K: rng.choice(errores, size=(4000, K)).mean(axis=1) for K in tamanos}
resumen = pd.DataFrame({
    "media de E_val": [E_val[K].mean() for K in tamanos],
    "desv. est. de E_val": [E_val[K].std() for K in tamanos],
    "σ/√K (teoría)": sigma / np.sqrt(tamanos),
}, index=pd.Index(tamanos, name="K"))
resumen.round(4)

# %% [markdown]
# La media de $E_{\text{val}}$ coincide con $E_{\text{out}}$ para todo $K$ (es insesgado), y su desviación estándar sigue a $\sigma/\sqrt{K}$ casi al dígito. Veámoslo en una gráfica: cada franja abarca el 90 % central de los 4,000 conjuntos de validación simulados.

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
bajo = [np.quantile(E_val[K], 0.05) for K in tamanos]
alto = [np.quantile(E_val[K], 0.95) for K in tamanos]
ax.fill_between(tamanos, bajo, alto, color=AZUL, alpha=0.25, lw=0)
ax.plot(tamanos, [E_val[K].mean() for K in tamanos], "o-", color=AZUL)
ax.plot(tamanos, E_out + 1.645 * sigma / np.sqrt(tamanos), "--", color=ROJO, lw=1.5)
ax.plot(tamanos, E_out - 1.645 * sigma / np.sqrt(tamanos), "--", color=ROJO, lw=1.5)
ax.axhline(E_out, color=TINTA, lw=1)
ax.set_xscale("log")
ax.set(title="Cuánto varía $E_{val}$ según el tamaño de la validación",
       xlabel="K (casas de validación, escala logarítmica)", ylabel="error cuadrático medio",
       xlim=(8, 2500))
ax.text(1100, E_out, "$E_{out}(g^-)$", va="bottom", color=TINTA)
ax.text(1100, alto[-1] + 0.001, "90 % central\nsimulado", va="bottom", color=AZUL, fontsize=10)
ax.text(12, E_out + 1.645 * sigma / np.sqrt(12), "$\\pm 1.645\\,\\sigma/\\sqrt{K}$", color=ROJO, fontsize=10, va="bottom")
plt.show()

# %% [markdown]
# Con 10 casas de validación el estimado puede salir en cualquier lado; con 50, el error relativo típico de la estimación todavía es grande; para que la franja sea angosta hacen falta cientos. La varianza baja como $1/K$, y esa es una ley que no se puede esquivar: **para estimar el doble de preciso, hacen falta cuatro veces más datos de validación**.
#
# Observa también que $\sigma$ es grande comparada con $E_{\text{out}}$: los errores al cuadrado tienen una distribución muy sesgada (la mayoría de las casas se predicen bien y unas pocas muy mal), y eso hace al estimado más ruidoso que si los errores fueran parecidos entre sí.

# %% [markdown]
# ### El dilema: ¿cuántos datos apartar?
#
# Parecería que conviene un $K$ grande. Pero los datos que apartamos no se usan para entrenar, así que $g^-$ empeora. Hay dos cantidades en juego:
#
# - $E_{\text{out}}(g^-)$, lo que **de verdad** vale la hipótesis entrenada con $N - K$ datos, que crece con $K$;
# - qué tan bien estima $E_{\text{val}}$ a esa cantidad, que mejora con $K$.
#
# Y en realidad, al final no vamos a entregar $g^-$: vamos a reentrenar con los $N$ datos y entregar $g$, que es mejor. Entonces $E_{\text{val}}(g^-)$ estima el error de una hipótesis que no es la que entregamos; si $K$ es grande, lo sobrestima. En resumen:
#
# $$E_{\text{out}}(g) \;\underset{\text{$K$ chico}}{\approx}\; E_{\text{out}}(g^-) \;\underset{\text{$K$ grande}}{\approx}\; E_{\text{val}}(g^-).$$
#
# Lo vemos con una simulación: tomamos 600 casas de $\mathcal{D}$ como «nuestros datos» y las otras como población para medir $E_{\text{out}}$. Para cada $K$ repetimos el experimento 100 veces, sorteando cada vez cuáles son las 600 casas y cuáles de ellas van a validación. En cada repetición calculamos tres cosas: $E_{\text{val}}(g^-)$, $E_{\text{out}}(g^-)$ y $E_{\text{out}}(g)$, con $g$ entrenada con las 600.

# %%
valores_K = [25, 50, 100, 200, 300, 400, 500]
dilema = []
for K in valores_K:
    for r in range(100):
        p = rng.permutation(N)
        nuestras, referencia = p[:600], p[600:]
        val, entr = nuestras[:K], nuestras[K:]
        g_m = ridge().fit(X_d.iloc[entr], y_d[entr])
        g = ridge().fit(X_d.iloc[nuestras], y_d[nuestras])
        dilema.append({"K": K, "E_val": ecm(g_m, X_d.iloc[val], y_d[val]),
                       "E_out_g_menos": ecm(g_m, X_d.iloc[referencia], y_d[referencia]),
                       "E_out_g": ecm(g, X_d.iloc[referencia], y_d[referencia])})
dilema = pd.DataFrame(dilema)
tabla_dilema = dilema.groupby("K").agg(E_out_g=("E_out_g", "mean"), E_out_g_menos=("E_out_g_menos", "mean"),
                                       E_val_media=("E_val", "mean"), E_val_desv=("E_val", "std"))
E_out_g = dilema["E_out_g"].mean()
tabla_dilema.round(4)

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
t = tabla_dilema
ax.fill_between(t.index, t["E_val_media"] - t["E_val_desv"], t["E_val_media"] + t["E_val_desv"],
                color=AZUL, alpha=0.2, lw=0)
ax.plot(t.index, t["E_val_media"], "o-", color=AZUL)
ax.plot(t.index, t["E_out_g_menos"], "s-", color=ROJO)
ax.hlines(E_out_g, 0, 500, color=TINTA, lw=1, ls="--")
ax.set(title="El dilema del tamaño de la validación (N = 600)", xlabel="K (casas apartadas para validar)",
       ylabel="error cuadrático medio", xlim=(0, 680))
ax.text(512, t["E_val_media"].iloc[-1] + t["E_val_desv"].iloc[-1], "$E_{val}(g^-)$\n± 1 desv. est.", color=AZUL, va="center")
ax.text(512, t["E_out_g_menos"].iloc[-1] - 0.0008, "$E_{out}(g^-)$", color=ROJO, va="center")
ax.text(512, E_out_g, "$E_{out}(g)$", color=TINTA, va="center")
plt.show()

# %% [markdown]
# La curva roja es la curva de aprendizaje vista al revés: entre más apartamos, peor la hipótesis $g^-$. La curva azul casi se encima con la roja: en promedio, $E_{\text{val}}$ sí sigue a $E_{\text{out}}(g^-)$ (es insesgado para $g^-$). Pero la banda muestra que con $K$ chico es muy ruidoso, y con $K$ grande, aunque es más estable, estima algo bastante peor que lo que de verdad entregamos ($E_{\text{out}}(g)$, la línea punteada). La regla práctica que se usa mucho es apartar alrededor de $K = N/5$. La validación cruzada (sección 4) es la forma de salir del dilema.

# %% [markdown]
# ## 3. Usar la validación para elegir la contamina
#
# Casi nunca validamos un solo modelo. Probamos varios y nos quedamos con el de menor $E_{\text{val}}$. ¿Sigue siendo $E_{\text{val}}$ del elegido un estimado insesgado de su $E_{\text{out}}$?
#
# No. Es la misma historia de las **1,000 monedas** de la primera presentación: si lanzamos 1,000 monedas justas 10 veces cada una, la que salió con menos águilas parece cargada, aunque todas son iguales. El mínimo de muchos promedios ruidosos es sistemáticamente menor que su valor esperado.
#
# ### Muchos modelos inútiles
#
# Llevémoslo al extremo con un problema de clasificación: $M$ «modelos» que adivinan al azar, así que todos tienen $E_{\text{out}} = 0.5$. Los evaluamos en una validación de $K = 100$ ejemplos y elegimos el de menor $E_{\text{val}}$. El número de errores de cada uno es $\text{Binomial}(K, 1/2)$, así que no hace falta simular los ejemplos; basta simular los conteos. Y la distribución del mínimo tiene fórmula:
#
# $$P\Big(\min_{m} E_{\text{val}}(h_m) \le t\Big) = 1 - \big(1 - F_K(t)\big)^M,$$
#
# donde $F_K(t) = P(\text{Binomial}(K, \tfrac12) \le Kt)$.

# %%
K = 100
modelos_M = np.array([1, 10, 100, 1000, 10000])
mejor = {M: rng.binomial(K, 0.5, size=(2000, M)).min(axis=1) / K for M in modelos_M}


def cuantil_teorico(q, M, K=K):
    """Cuantil q del mínimo de M proporciones Binomial(K, 1/2)/K."""
    t = np.arange(K + 1)
    acumulada = 1 - (1 - binom.cdf(t, K, 0.5)) ** M
    return t[np.searchsorted(acumulada, q)] / K


pd.DataFrame({
    "mediana simulada": [np.median(mejor[M]) for M in modelos_M],
    "mediana teórica": [cuantil_teorico(0.5, M) for M in modelos_M],
    "5 % simulado": [np.quantile(mejor[M], 0.05) for M in modelos_M],
}, index=pd.Index(modelos_M, name="M"))

# %% [markdown]
# Con 1,000 modelos que no sirven para nada, el mejor tiene típicamente $E_{\text{val}} \approx 0.35$: ¡parece que acierta el 65 %! Y es puro azar.
#
# La teoría de la primera presentación nos dice cuánto puede llegar a engañarnos. Con Hoeffding y la unión de eventos sobre los $M$ modelos (ahora $\mathcal{H}$ es finito: son los $M$ candidatos), con probabilidad al menos $1-\delta$:
#
# $$E_{\text{out}}(g^{*}) \le E_{\text{val}}(g^{*}) + \sqrt{\frac{1}{2K}\ln\frac{2M}{\delta}}.$$
#
# El castigo crece con $\ln M$: elegir entre pocos modelos cuesta poco, entre muchos cuesta más. Comparemos la simulación con esa cota.

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
eje = np.logspace(0, 4, 50)
ax.fill_between(modelos_M, [np.quantile(mejor[M], 0.05) for M in modelos_M],
                [np.quantile(mejor[M], 0.95) for M in modelos_M], color=AZUL, alpha=0.25, lw=0)
ax.plot(modelos_M, [np.median(mejor[M]) for M in modelos_M], "o-", color=AZUL)
ax.plot(eje, 0.5 - np.sqrt(np.log(2 * eje / 0.05) / (2 * K)), "--", color=ROJO)
ax.axhline(0.5, color=TINTA, lw=1)
ax.set_xscale("log")
ax.set(title="El mejor de M modelos que adivinan al azar (K = 100)", xlabel="M (modelos comparados)",
       ylabel="$E_{val}$ del mejor", ylim=(0.15, 0.56), xlim=(0.8, 40000))
ax.text(1.2, 0.505, "$E_{out}$ de todos = 0.5", va="bottom", color=TINTA)
ax.text(150, 0.43, "mediana del mejor\n(franja: 90 % central)", color=AZUL, fontsize=10, va="bottom")
ax.text(1500, 0.205, "límite de Hoeffding\n+ unión (δ = 0.05)", color=ROJO, fontsize=10)
plt.show()

# %% [markdown]
# La franja nunca cruza la línea roja, como garantiza la teoría (con 95 % de probabilidad), y la cota es bastante pesimista: el mejor real queda lejos del límite. Pero la lección es la tendencia: **entre más modelos comparamos, más optimista es el $E_{\text{val}}$ del ganador.**
#
# ### Lo mismo, con modelos de verdad
#
# Los modelos al azar exageran. ¿Pasa con modelos razonables? Volvemos a las casas: entrenamos $M = 100$ regresiones ridge con 300 casas, cada una con **cuatro columnas elegidas al azar**, y las comparamos en una validación de $K = 50$ casas. Elegimos la de menor $E_{\text{val}}$ y medimos su error en el resto de $\mathcal{D}$ (nuestra aproximación a $E_{\text{out}}$). Repetimos el experimento 30 veces.

# %%
columnas = X.columns.to_numpy()
optimismo = []
for r in range(30):
    p = rng.permutation(N)
    entr, val, resto = p[:300], p[300:350], p[350:]
    candidatos = []
    for m in range(100):
        cols = rng.choice(columnas, size=4, replace=False)
        h = ridge(1.0).fit(X_d.iloc[entr][cols], y_d[entr])
        candidatos.append((ecm(h, X_d.iloc[val][cols], y_d[val]), h, cols))
    e_val, h, cols = min(candidatos, key=lambda c: c[0])
    optimismo.append({"E_val del elegido": e_val, "E_out del elegido": ecm(h, X_d.iloc[resto][cols], y_d[resto])})
optimismo = pd.DataFrame(optimismo)
optimismo.describe().loc[["mean", "std"]].round(4)

# %% [markdown]
# En promedio, el $E_{\text{val}}$ del ganador es notablemente menor que su error real. No es un error de programación ni mala suerte: es lo que pasa siempre que **el mismo conjunto de datos se usa para elegir y para reportar**. Después de elegir, la validación se volvió parte del entrenamiento (entrenamos «el selector»), y su error ya no es un estimado insesgado.
#
# Por eso existe el **conjunto de prueba**: un conjunto que no se usa para elegir nada y se mira **una sola vez**, al final. Si se mira varias veces y se corrige el modelo según lo que dice, se convierte en otra validación más.

# %% [markdown]
# ## 4. Validación cruzada
#
# ### La idea
#
# El dilema de la sección 2 viene de que cada dato sirve para entrenar **o** para validar. La validación cruzada hace que cada dato sirva para las dos cosas, en turnos distintos.
#
# **Validación cruzada de $k$ particiones** (*k-fold cross validation*). Partimos $\mathcal{D}$ al azar en $k$ pedazos (*folds*) de tamaño $N/k$: $\mathcal{D}_1, \dots, \mathcal{D}_k$. Para cada $j$, entrenamos con todo menos $\mathcal{D}_j$, obtenemos $g_j^-$ y medimos su error $E_j$ en $\mathcal{D}_j$. El estimado es el promedio:
#
# $$E_{\text{cv}} = \frac{1}{k}\sum_{j=1}^{k} E_j, \qquad E_j = \frac{1}{|\mathcal{D}_j|}\sum_{n \in \mathcal{D}_j} e\big(g_j^-(\mathbf{x}_n), y_n\big).$$
#
# Cada $E_j$ es un estimado insesgado del $E_{\text{out}}$ de una hipótesis entrenada con $N - N/k$ datos, así que $\mathbb{E}[E_{\text{cv}}] = \bar{E}_{\text{out}}(N - N/k)$, el error esperado con ese tamaño de muestra. Ahora **todos** los $N$ datos se usaron para validar, lo que reduce la varianza; y cada $g_j^-$ se entrenó con casi todos los datos, lo que reduce el sesgo pesimista. Los $E_j$ no son independientes (comparten datos de entrenamiento), así que la varianza no baja tanto como $\sigma^2/N$, pero baja.
#
# **Dejar-uno-fuera** (*leave-one-out*, LOO) es el caso extremo $k = N$: cada pedazo es un solo dato. Entrenamos $N$ veces, cada vez sin el dato $n$, y
#
# $$E_{\text{LOO}} = \frac{1}{N}\sum_{n=1}^{N} e\big(g_n^-(\mathbf{x}_n), y_n\big).$$
#
# Es casi insesgado para $\bar{E}_{\text{out}}(N-1)$, pero cuesta $N$ entrenamientos.
#
# ### ¿Cuánto importa $k$, y cuánto la partición?
#
# Con las $N$ casas de $\mathcal{D}$ calculamos $E_{\text{cv}}$ con $k = 2, 5, 10$, cada uno con 30 particiones al azar distintas (30 semillas), y el LOO, que no depende del azar.

# %%
estimados = {}
for k in [2, 5, 10]:
    estimados[k] = [-cross_val_score(ridge(), X_d, y_d, cv=KFold(k, shuffle=True, random_state=s),
                                     scoring="neg_mean_squared_error").mean() for s in range(30)]
inicio = time.perf_counter()
E_loo = -cross_val_score(ridge(), X_d, y_d, cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=-1).mean()
tiempo_loo = time.perf_counter() - inicio
pd.DataFrame({"media": {k: np.mean(v) for k, v in estimados.items()} | {f"N = {N}": E_loo},
              "desv. est. entre particiones": {k: np.std(v) for k, v in estimados.items()} | {f"N = {N}": 0.0}},
             ).rename_axis("k").round(5)

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, k in enumerate([2, 5, 10]):
    ax.scatter(i + rng.uniform(-0.12, 0.12, 30), estimados[k], s=22, color=AZUL, alpha=0.7, edgecolors="none")
    ax.plot([i - 0.22, i + 0.22], [np.mean(estimados[k])] * 2, color=TINTA, lw=2)
ax.scatter([3], [E_loo], s=60, color=ROJO, zorder=3)
ax.set(xticks=range(4), xticklabels=["k = 2", "k = 5", "k = 10", f"LOO (k = {N})"],
       title="Estimados de validación cruzada con 30 particiones distintas",
       ylabel="$E_{cv}$ (error cuadrático medio)", xlim=(-0.6, 3.6))
plt.show()
print(f"LOO por fuerza bruta: {N} entrenamientos, {tiempo_loo:.1f} s")

# %% [markdown]
# Tres observaciones:
#
# 1. Con $k = 2$ el estimado es más **alto**: cada $g_j^-$ se entrenó con la mitad de los datos, así que es pesimista. Entre $k = 5$, $k = 10$ y LOO la diferencia en promedio ya es pequeña.
# 2. Con $k = 2$ el estimado también **cambia más** de una partición a otra. Con $k = 10$ casi no importa la semilla.
# 3. Cambiar la semilla mueve el estimado en la cuarta cifra decimal, mientras que la diferencia entre modelos que comparemos suele ser de ese orden. Por eso, al comparar modelos conviene usar **las mismas particiones** para todos (fijar `random_state`) y, si hay tiempo, repetir con varias semillas.
#
# La costumbre de usar $k = 5$ o $k = 10$ es un buen compromiso entre sesgo, varianza y costo.

# %% [markdown]
# ### Dejar-uno-fuera sin reentrenar
#
# Para la regresión lineal y ridge hay un resultado precioso: el LOO se puede calcular **exactamente con un solo entrenamiento**.
#
# Sea $Z$ la matriz $N \times (d+1)$ de datos con una columna de unos, y $\Lambda = \operatorname{diag}(0, \lambda, \dots, \lambda)$ (no castigamos el intercepto). La solución de ridge es $\mathbf{w} = A^{-1}Z^{\mathsf T}\mathbf{y}$ con $A = Z^{\mathsf T}Z + \Lambda$, y las predicciones son
#
# $$\hat{\mathbf{y}} = Z\mathbf{w} = H\mathbf{y}, \qquad H = Z A^{-1} Z^{\mathsf T}.$$
#
# A $H$ se le llama **matriz sombrero** (*hat matrix*) porque «le pone el sombrero» a $\mathbf{y}$. El resultado es:
#
# $$\boxed{\;y_n - g_n^-(\mathbf{x}_n) = \frac{y_n - \hat{y}_n}{1 - H_{nn}}\;}
# \qquad\Longrightarrow\qquad
# E_{\text{LOO}} = \frac{1}{N}\sum_{n=1}^{N}\left(\frac{y_n - \hat{y}_n}{1 - H_{nn}}\right)^2.$$
#
# **Demostración.** Sea $\mathbf{z}$ el renglón $n$ de $Z$ y $h = H_{nn} = \mathbf{z}^{\mathsf T}A^{-1}\mathbf{z}$. Sin el dato $n$, la matriz y el vector del sistema pierden un término: $A_{-n} = A - \mathbf{z}\mathbf{z}^{\mathsf T}$ y $\mathbf{b}_{-n} = Z^{\mathsf T}\mathbf{y} - \mathbf{z}\,y_n$. La fórmula de Sherman–Morrison da la inversa de una matriz modificada por un término de rango uno:
#
# $$A_{-n}^{-1} = A^{-1} + \frac{A^{-1}\mathbf{z}\mathbf{z}^{\mathsf T}A^{-1}}{1 - h}
# \quad\Longrightarrow\quad
# \mathbf{z}^{\mathsf T}A_{-n}^{-1} = \mathbf{z}^{\mathsf T}A^{-1}\Big(1 + \frac{h}{1-h}\Big) = \frac{\mathbf{z}^{\mathsf T}A^{-1}}{1-h}.$$
#
# Entonces la predicción sin el dato $n$ es
#
# $$g_n^-(\mathbf{x}_n) = \mathbf{z}^{\mathsf T}A_{-n}^{-1}\mathbf{b}_{-n} = \frac{\mathbf{z}^{\mathsf T}A^{-1}Z^{\mathsf T}\mathbf{y} - h\,y_n}{1-h} = \frac{\hat{y}_n - h\,y_n}{1-h},$$
#
# y restando de $y_n$: $\;y_n - g_n^-(\mathbf{x}_n) = \dfrac{y_n(1-h) - \hat{y}_n + h\,y_n}{1-h} = \dfrac{y_n - \hat{y}_n}{1-h}.$ $\blacksquare$
#
# **Intuición.** $H_{nn}$ mide cuánto «jala» el dato $n$ a su propia predicción (se llama la **palanca** o *leverage* del dato). Si es cercano a 0, quitarlo casi no cambia nada; si es cercano a 1, el modelo se dobló para pasar por él, y su residuo dentro de la muestra subestima mucho el error real. La fórmula infla cada residuo justo en esa medida.
#
# **Verificación numérica.** Tomamos 200 casas, estandarizamos las columnas una sola vez (la fórmula supone que las variables están fijas; si el escalador se reentrenara en cada vuelta, la igualdad ya no sería exacta, aunque la diferencia sería minúscula) y comparamos con el LOO por fuerza bruta.

# %%
def loo_ridge(Xm, y, lam):
    """E_LOO exacto de ridge (intercepto sin castigo) con la matriz sombrero: un solo entrenamiento."""
    Z = np.column_stack([np.ones(len(y)), Xm])
    castigo = lam * np.eye(Z.shape[1])
    castigo[0, 0] = 0.0
    H = Z @ np.linalg.solve(Z.T @ Z + castigo, Z.T)
    residuos = (y - H @ y) / (1 - np.diag(H))
    return np.mean(residuos ** 2), np.diag(H)


LAM = 10.0
sub = rng.choice(N, size=200, replace=False)
X_sub = StandardScaler().fit_transform(X_d.iloc[sub])
y_sub = y_d[sub]

inicio = time.perf_counter()
fuerza_bruta = np.mean([(Ridge(alpha=LAM).fit(np.delete(X_sub, n, axis=0), np.delete(y_sub, n))
                         .predict(X_sub[n:n + 1])[0] - y_sub[n]) ** 2 for n in range(len(y_sub))])
t_bruta = time.perf_counter() - inicio
inicio = time.perf_counter()
formula, palancas = loo_ridge(X_sub, y_sub, LAM)
t_formula = time.perf_counter() - inicio
print(f"LOO por fuerza bruta (200 entrenamientos): {fuerza_bruta:.10f}   ({t_bruta * 1000:.0f} ms)")
print(f"LOO con la matriz sombrero (1 cálculo):    {formula:.10f}   ({t_formula * 1000:.1f} ms)")
print(f"diferencia relativa: {abs(formula - fuerza_bruta) / fuerza_bruta:.1e}")

# %% [markdown]
# Coinciden hasta el error de redondeo de la máquina. scikit-learn usa este mismo atajo en `RidgeCV` cuando no se le da el parámetro `cv`: por eso puede probar decenas de valores de $\lambda$ casi gratis.
#
# Un detalle útil: la traza de $H$ es la suma de las palancas y funciona como el **número efectivo de parámetros** del modelo. Para mínimos cuadrados sin castigo vale exactamente el rango de $Z$ (que es $d + 1$ si no hay columnas redundantes); ridge la reduce.

# %%
for lam in [0.001, 10.0, 100.0, 1000.0]:
    _, h = loo_ridge(X_sub, y_sub, lam)
    print(f"λ = {lam:>6g}:  traza(H) = {h.sum():5.1f}   (columnas + intercepto: {X_sub.shape[1] + 1})")

# %% [markdown]
# Con $\lambda$ casi cero la traza es menor que el número de columnas porque en estas 200 casas algunas indicadoras son redundantes o están vacías (un barrio que no aparece, por ejemplo); por eso mismo no usamos $\lambda = 0$, con el que $A$ no sería invertible. Al crecer $\lambda$, el modelo «gasta» cada vez menos parámetros: esa es otra forma de ver que la regularización reduce la complejidad.

# %% [markdown]
# ## 5. Elegir un hiperparámetro y estimar el error del procedimiento completo
#
# ### Un modelo donde $\lambda$ importa
#
# Con 51 columnas y más de mil casas, ridge casi no necesita regularización. Para que la elección de $\lambda$ sea interesante, le damos al modelo mucha más libertad: agregamos todos los **productos de pares** de las 16 variables numéricas (y sus cuadrados), lo que da 152 columnas numéricas más las 35 indicadoras. Con tantos parámetros, sin castigo el modelo sobreajusta, y con demasiado castigo se queda corto.

# %%
def ridge_cuadratico(alpha=1.0):
    pares = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler())
    preparar = ColumnTransformer([("pares", pares, NUMERICAS)], remainder=StandardScaler())
    return make_pipeline(preparar, Ridge(alpha=alpha))


lambdas = np.logspace(-2, 4, 31)
k = 10
busqueda = GridSearchCV(ridge_cuadratico(), {"ridge__alpha": lambdas}, cv=KFold(k, shuffle=True, random_state=SEMILLA),
                        scoring="neg_mean_squared_error").fit(X_d, y_d)
cv_media = -busqueda.cv_results_["mean_test_score"]
cv_ee = busqueda.cv_results_["std_test_score"] / np.sqrt(k)

# %% [markdown]
# ### La regla de una desviación estándar
#
# Cada punto de la curva de validación cruzada es un promedio de $k = 10$ errores, así que tiene incertidumbre. Su **error estándar** se estima con la desviación estándar de los $E_j$ entre particiones:
#
# $$\operatorname{EE}(\lambda) = \frac{\operatorname{desv}\big(E_1(\lambda), \dots, E_k(\lambda)\big)}{\sqrt{k}}.$$
#
# (Es una estimación aproximada, porque los $E_j$ no son independientes, pero sirve como guía.)
#
# Si dos valores de $\lambda$ difieren en menos que eso, la validación cruzada no puede distinguirlos. La **regla de una desviación estándar** (Breiman y colaboradores, 1984; popularizada por Hastie, Tibshirani y Friedman) dice: entre todos los modelos cuyo error de CV está a menos de un error estándar del mejor, **elige el más simple**. En ridge, «más simple» significa **mayor** $\lambda$:
#
# $$\lambda_{1\text{EE}} = \max\big\{\lambda : E_{\text{cv}}(\lambda) \le E_{\text{cv}}(\lambda^{*}) + \operatorname{EE}(\lambda^{*})\big\}, \qquad \lambda^{*} = \arg\min_\lambda E_{\text{cv}}(\lambda).$$
#
# El argumento: el mínimo de la curva es en parte ruido (sección 3), y ante la duda, el modelo más simple suele generalizar igual o mejor y es más estable.

# %%
def regla_1ee(resultados):
    """Índice del λ más grande cuyo error de CV está a menos de un error estándar del mínimo.

    Sirve como `refit` de GridSearchCV; supone que la rejilla de λ va en orden creciente."""
    media = -resultados["mean_test_score"]
    ee = resultados["std_test_score"] / np.sqrt(k)
    mejor = np.argmin(media)
    return int(np.flatnonzero(media <= media[mejor] + ee[mejor]).max())


i_mejor = int(np.argmin(cv_media))
i_1ee = regla_1ee(busqueda.cv_results_)
print(f"λ* (mínimo de CV) = {lambdas[i_mejor]:7.1f}   E_cv = {cv_media[i_mejor]:.4f} ± {cv_ee[i_mejor]:.4f}")
print(f"λ con la regla 1-EE = {lambdas[i_1ee]:5.1f}   E_cv = {cv_media[i_1ee]:.4f}")

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.fill_between(lambdas, cv_media - cv_ee, cv_media + cv_ee, color=AZUL, alpha=0.2, lw=0)
ax.plot(lambdas, cv_media, "o-", color=AZUL, ms=4)
ax.axhline(cv_media[i_mejor] + cv_ee[i_mejor], color=GRIS, lw=1, ls=":")
ax.scatter(lambdas[i_mejor], cv_media[i_mejor], s=90, color=ROJO, zorder=3)
ax.scatter(lambdas[i_1ee], cv_media[i_1ee], s=90, color=DORADO_OSC, zorder=3)
ax.set_xscale("log")
ax.set(title="Validación cruzada (k = 10) para elegir λ en ridge con pares de variables",
       xlabel="λ (escala logarítmica)", ylabel="$E_{cv}$ ± 1 error estándar", ylim=(0.012, 0.032), xlim=(0.006, 4e4))
ax.annotate(f"mínimo: λ* ≈ {lambdas[i_mejor]:.0f}", (lambdas[i_mejor], cv_media[i_mejor]), xytext=(0.6, 0.0135),
            color=ROJO, arrowprops=dict(arrowstyle="-", color=ROJO))
ax.annotate(f"regla 1-EE: λ ≈ {lambdas[i_1ee]:.0f}", (lambdas[i_1ee], cv_media[i_1ee]), xytext=(1500, 0.0145),
            color=DORADO_OSC, arrowprops=dict(arrowstyle="-", color=DORADO_OSC))
ax.text(0.012, cv_media[i_mejor] + cv_ee[i_mejor] - 0.0004, "mínimo + 1 EE", color=GRIS, fontsize=10, va="top")
ax.text(0.012, 0.0185, "λ chico: sobreajuste", color=TENUE, fontsize=10)
ax.text(5000, 0.022, "λ grande:\nmodelo\ndemasiado\nrígido", color=TENUE, fontsize=10)
plt.show()

# %% [markdown]
# La curva tiene la forma de U que predice la descomposición sesgo-varianza: a la izquierda domina la varianza y a la derecha el sesgo. El fondo es plano y ancho: muchos valores de $\lambda$ son indistinguibles con estos datos, y la regla 1-EE se va al extremo derecho de esa meseta.
#
# ### ¿Cuánto se equivocará el modelo elegido?
#
# Es tentador reportar $E_{\text{cv}}(\lambda^{*})$ como el error esperado del modelo final. Pero $\lambda^{*}$ se eligió precisamente porque su $E_{\text{cv}}$ era el más bajo entre 31: es la situación de la sección 3, en pequeño. $E_{\text{cv}}(\lambda^*)$ es **optimista**.
#
# La forma honesta de estimar el error es pensar que lo que estamos evaluando no es un modelo, sino un **procedimiento**: «hacer CV de 10 particiones sobre la rejilla de $\lambda$, elegir con tal regla y reentrenar con todo». Ese procedimiento se evalúa como cualquier otro algoritmo de aprendizaje: con validación cruzada **por fuera**. Eso es la **validación cruzada anidada**:
#
# - el ciclo **externo** parte $\mathcal{D}$ en 5 pedazos; para cada uno,
# - el ciclo **interno** corre toda la búsqueda de $\lambda$ (con sus 10 particiones) usando solo los otros 4 pedazos,
# - el modelo elegido se mide en el pedazo externo, que nunca participó en la elección.
#
# En scikit-learn es tan sencillo como pasarle el `GridSearchCV` a `cross_val_score`: el objeto de búsqueda **es** un estimador. Para la regla 1-EE usamos `refit=regla_1ee`, que le dice a `GridSearchCV` qué índice elegir.

# %%
def procedimiento(regla):
    return GridSearchCV(ridge_cuadratico(), {"ridge__alpha": lambdas}, cv=KFold(k, shuffle=True, random_state=SEMILLA),
                        scoring="neg_mean_squared_error", refit=regla)


externa = KFold(5, shuffle=True, random_state=SEMILLA + 1)
anidada = {nombre: -cross_val_score(procedimiento(regla), X_d, y_d, cv=externa, scoring="neg_mean_squared_error")
           for nombre, regla in [("mínimo", True), ("regla 1-EE", regla_1ee)]}

# %% [markdown]
# Ahora, por fin, abrimos el conjunto de prueba que apartamos al principio. Entrenamos los dos modelos finales con todo $\mathcal{D}$ y los medimos una sola vez.

# %%
final_min = ridge_cuadratico(lambdas[i_mejor]).fit(X_d, y_d)
final_1ee = ridge_cuadratico(lambdas[i_1ee]).fit(X_d, y_d)
errores_prueba = {"mínimo": (final_min.predict(X_prueba) - y_prueba) ** 2,
                  "regla 1-EE": (final_1ee.predict(X_prueba) - y_prueba) ** 2}

comparacion = pd.DataFrame({
    "λ elegido": [lambdas[i_mejor], lambdas[i_1ee]],
    "E_cv del elegido (optimista)": [cv_media[i_mejor], cv_media[i_1ee]],
    "CV anidada": [anidada["mínimo"].mean(), anidada["regla 1-EE"].mean()],
    "prueba": [errores_prueba["mínimo"].mean(), errores_prueba["regla 1-EE"].mean()],
    "EE de la prueba": [errores_prueba[r].std() / np.sqrt(len(y_prueba)) for r in ["mínimo", "regla 1-EE"]],
}, index=["mínimo", "regla 1-EE"])
comparacion.round(4)

# %% [markdown]
# Cómo leer esta tabla:
#
# - **$E_{\text{cv}}$ del elegido** es el número que da `busqueda.best_score_`. Para el mínimo es optimista por construcción; aquí el optimismo es pequeño porque solo elegimos un número (un solo hiperparámetro continuo, con una curva suave), pero con búsquedas grandes (muchos hiperparámetros, muchos tipos de modelo) crece, como en la sección 3.
# - **La CV anidada** estima el error del procedimiento completo, incluida la elección. Es el número honesto para reportar si no tenemos conjunto de prueba.
# - **La prueba** es el único número calculado con datos que no tocaron ninguna decisión. Pero fíjate en su error estándar: con 292 casas, la incertidumbre de la prueba es del mismo orden que las diferencias que estamos mirando (es la ley $\sigma/\sqrt{K}$ de la sección 2). Un conjunto de prueba no da la verdad; da un estimado insesgado **y ruidoso**.
#
# Por eso no hay que leer demasiado en cuál de los dos modelos ganó en la prueba: las dos reglas dan modelos prácticamente equivalentes, y la regla 1-EE lo logra con un modelo más regularizado.
#
# Una nota honesta: este modelo con pares de variables, aun con el mejor $\lambda$, **no** supera al modelo lineal de la sección 4, cuyo error de dejar-uno-fuera era de 0.0141. Con estos datos los productos de pares no aportan, y la validación cruzada nos lo dice. Elegir entre el modelo lineal y el cuadrático sería un hiperparámetro más de la búsqueda, y habría que incluirlo dentro de la CV anidada (ejercicio 5).

# %% [markdown]
# ## 6. Datos agrupados: cuando la validación cruzada común engaña
#
# Todo lo anterior supone que los ejemplos son **independientes**: que conocer una casa no dice nada sobre otra. Muchas veces no es así. Los datos vienen en **grupos** de ejemplos parecidos entre sí: varias consultas del mismo paciente, varias mediciones del mismo sensor, varios alumnos de la misma escuela, varios años de la misma presa, varias compras del mismo cliente. Si los ejemplos de un grupo caen tanto en entrenamiento como en validación, el modelo puede **reconocer al grupo** en lugar de aprender el patrón general, y la validación lo premia.
#
# ### Los datos y el problema
#
# **Datos.** El conjunto *Online Retail* que Daqing Chen publicó en el UCI Machine Learning Repository (2015, licencia CC BY 4.0), el mismo que usamos para segmentar clientes en la Unidad 2: todas las transacciones, de diciembre de 2010 a diciembre de 2011, de una tienda en línea del Reino Unido que vende artículos de regalo, con muchos clientes mayoristas. Para el curso lo agregamos a una fila por **factura**: número de factura, cliente, fecha y hora, número de artículos distintos (`lineas`), número de piezas, importe en libras y país.
#
# **Problema.** Predecir el **importe** de una factura (en escala logarítmica) a partir de lo que se sabe de ella: cuántos artículos y cuántas piezas lleva, el país, la hora, el día de la semana y el mes. Es el tipo de estimación que una tienda usa para planear inventario o detectar facturas raras. Cada cliente compra cosas de precios parecidos (un mayorista compra mucho barato; un particular, poco y más caro), así que **las facturas de un mismo cliente se parecen**: son un grupo.
#
# Usamos solo compras (sin cancelaciones) de clientes registrados, y para que corra rápido, 1,500 clientes al azar con **todas** sus facturas.

# %%
facturas = rdp.datos.cargar("tienda_facturas", parse_dates=["fecha"])
compras = facturas[facturas["cliente"].notna() & ~facturas["factura"].astype(str).str.startswith("C")
                   & (facturas["importe"] > 0)].copy()
compras["cliente"] = compras["cliente"].astype(int)
elegidos = compras["cliente"].drop_duplicates().sample(1500, random_state=SEMILLA)
compras = compras[compras["cliente"].isin(elegidos)].reset_index(drop=True)
print(f"{len(compras):,} facturas de {compras['cliente'].nunique():,} clientes")
compras.groupby("cliente").size().describe().round(1)

# %% [markdown]
# La mitad de los clientes tiene 2 facturas o menos, pero algunos tienen decenas: esos clientes frecuentes acaparan buena parte de las filas.
#
# Construimos dos versiones de las variables: una **sin** el identificador del cliente y otra **con** él, como una columna numérica más. Agregar el identificador es una tentación común («a lo mejor ayuda»), y un bosque aleatorio puede aprovecharlo: con suficientes cortes, aísla a cada cliente y aprende su precio típico.

# %%
X_fact = pd.DataFrame({
    "lineas": compras["lineas"],
    "log_piezas": np.log(compras["piezas"].clip(lower=1)),
    "pais": compras["pais"].astype("category").cat.codes,
    "hora": compras["fecha"].dt.hour + compras["fecha"].dt.minute / 60,
    "dia": compras["fecha"].dt.day_of_week,
    "mes": compras["fecha"].dt.month,
})
X_fact_id = X_fact.assign(cliente=compras["cliente"])
y_fact = np.log(compras["importe"]).to_numpy()
grupos = compras["cliente"].to_numpy()
print(f"varianza de log(importe): {y_fact.var():.3f}")

# %% [markdown]
# Comparamos dos formas de partir para la validación cruzada:
#
# - **`KFold` aleatorio**: las facturas se reparten al azar; las de un mismo cliente quedan regadas entre entrenamiento y validación.
# - **`GroupKFold` por cliente**: cada cliente queda completo de un solo lado. La validación simula lo que pasa con un **cliente nuevo**.

# %%
bosque = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEMILLA)
esquemas = {"KFold aleatorio": KFold(5, shuffle=True, random_state=SEMILLA),
            "GroupKFold por cliente": GroupKFold(5, shuffle=True, random_state=SEMILLA)}
agrupados = {}
for nombre_X, XX in [("sin id de cliente", X_fact), ("con id de cliente", X_fact_id)]:
    for nombre_cv, cv in esquemas.items():
        e = -cross_val_score(bosque, XX, y_fact, cv=cv, scoring="neg_mean_squared_error",
                             groups=grupos if "Group" in nombre_cv else None)
        agrupados[(nombre_X, nombre_cv)] = e.mean()
tabla_grupos = pd.Series(agrupados).unstack().loc[["sin id de cliente", "con id de cliente"], list(esquemas)]
tabla_grupos.round(3)

# %%
fig, ax = plt.subplots(figsize=(8, 4.2))
posiciones = np.arange(2)
for j, (nombre_cv, color) in enumerate(zip(esquemas, [AZUL, ROJO])):
    valores = tabla_grupos[nombre_cv].loc[["sin id de cliente", "con id de cliente"]]
    barras = ax.bar(posiciones + (j - 0.5) * 0.36, valores, width=0.34, color=color, label=nombre_cv)
    ax.bar_label(barras, fmt="%.3f", padding=3, fontsize=10, color=TINTA)
ax.set(xticks=posiciones, xticklabels=["sin id de cliente", "con id de cliente"], ylim=(0, 0.42),
       title="Error de validación cruzada del bosque aleatorio", ylabel="error cuadrático medio de log(importe)")
ax.legend(loc="upper left", frameon=False, ncols=2)
ax.grid(axis="x", visible=False)
plt.show()

# %% [markdown]
# Si solo hubiéramos usado `KFold` aleatorio, habríamos concluido que agregar el identificador del cliente **mejora** el modelo de forma clara. `GroupKFold` dice otra cosa: para clientes que el modelo no ha visto, el identificador no sirve de nada (un número de cliente nuevo no dice nada de cómo compra). La mejora era el bosque **reconociendo clientes** que también estaban en la validación. Incluso sin el identificador, `KFold` aleatorio es algo optimista, porque la combinación de país, tamaño de la compra y horario a veces basta para reconocer a un cliente frecuente.
#
# La moraleja no es «siempre usar `GroupKFold`», sino **validar como se va a usar el modelo**:
#
# - Si el modelo se usará con **clientes nuevos** (o pacientes nuevos, sensores nuevos, escuelas nuevas, presas que no estaban en los datos), la validación debe dejar fuera grupos completos.
# - Si se usará con **los mismos clientes** en el futuro, lo que hay que respetar es el **tiempo**: entrenar con el pasado y validar con el futuro. Con series de tiempo, partir al azar deja que el modelo «vea el futuro»; en la Unidad 7 veremos cómo validar respetando el orden (`TimeSeriesSplit` y validación con ventana móvil).
#
# En los dos casos el principio es el mismo que en todo el resto de la libreta: **los datos de validación deben ser tan nuevos para el modelo como lo serán los datos reales.**

# %% [markdown]
# ## 7. Cierre: reglas prácticas
#
# | Situación | Qué hacer | Por qué |
# |---|---|---|
# | Quieres saber cuánto se equivocará un modelo ya fijo | un conjunto de validación o prueba; reporta su error estándar $\sigma/\sqrt{K}$ | $E_{\text{val}}$ es insesgado, pero ruidoso si $K$ es chico |
# | Tienes pocos datos | validación cruzada con $k = 5$ o $10$ en lugar de un solo conjunto de validación | cada dato valida una vez y entrena $k-1$ veces |
# | Regresión lineal o ridge | LOO exacto con la matriz sombrero (`RidgeCV`) | cuesta un solo entrenamiento |
# | Comparas varios modelos o hiperparámetros | mismas particiones para todos; desconfía del $E_{\text{cv}}$ del ganador | elegir el mínimo de muchos estimados ruidosos es optimista |
# | Varios candidatos casi empatados | regla de una desviación estándar: el más simple dentro de 1 EE del mejor | la CV no puede distinguirlos; el más simple es más estable |
# | Quieres reportar el error de lo que elegiste | CV anidada, o un conjunto de prueba apartado al inicio y mirado **una vez** | la elección es parte del entrenamiento |
# | Preprocesamiento que aprende de los datos (medias, imputación, selección de variables) | dentro del `Pipeline`, para que se reaprenda en cada partición | si no, la validación ve información del entrenamiento |
# | Datos en grupos (clientes, pacientes, sensores, escuelas) | `GroupKFold` si el modelo se usará con grupos nuevos | el modelo puede reconocer al grupo en vez de aprender |
# | Datos en el tiempo | entrenar con el pasado y validar con el futuro (Unidad 7) | partir al azar deja ver el futuro |

# %% [markdown]
# ## 8. Autoevaluación

# %%
rdp.quiz.mostrar("u03", ids=["u03-p10", "u03-p11"])

# %% [markdown]
# ## 9. Ejercicios
#
# 1. **Varianza de la validación.** En la sección 2, cambia el modelo por uno con $\lambda = 1000$ (mucho más rígido). ¿Cambia $\sigma$? ¿Cuántas casas de validación harían falta para que el error típico de $E_{\text{val}}$ sea el 5 % de $E_{\text{out}}$?
# 2. **Monedas.** Repite la simulación de la sección 3 con $K = 1000$ en lugar de $K = 100$. ¿Cuánto baja el optimismo del mejor de 1,000 modelos? Compáralo con lo que predice el término $\sqrt{\ln(2M/\delta)/(2K)}$.
# 3. **Dejar-uno-fuera.** Encuentra las cinco casas con mayor palanca $H_{nn}$ en el subconjunto de 200 casas. ¿Qué tienen de especial? (Pista: mira el barrio y el tipo de construcción.) ¿Cuánto cambia $E_{\text{LOO}}$ si las quitas?
# 4. **LOO para clasificación.** Para k vecinos más cercanos también hay un atajo de LOO: basta calcular los vecinos de cada punto **excluyéndose a sí mismo**. Prográmalo para el conjunto de pingüinos de la Unidad 2 (predecir la especie con las cuatro medidas) y verifícalo contra `LeaveOneOut`.
# 5. **CV anidada con más candidatos.** Agrega a la búsqueda de la sección 5 el grado del polinomio (1, 2 o 3) además de $\lambda$. ¿Crece la distancia entre $E_{\text{cv}}$ del mejor y la CV anidada? ¿Por qué?
# 6. **Datos agrupados.** En la sección 6, cambia el bosque por k vecinos más cercanos (con las variables estandarizadas y $k = 5$) y repite la comparación. ¿Es más o menos sensible que el bosque al identificador del cliente?
# 7. **Validación en el tiempo.** Ordena las facturas por fecha, entrena con las de 2010 y los primeros nueve meses de 2011, y valida con las de octubre a diciembre de 2011. ¿El error se parece más al de `KFold` o al de `GroupKFold`? ¿Qué pregunta de negocio contesta cada esquema?
