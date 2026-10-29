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
# # Bosques aleatorios: la sabiduría de muchos árboles
#
# **Unidad 5 · Ejemplo 2**
#
# Un árbol de decisión profundo se ajusta casi perfectamente a sus datos de entrenamiento, y por eso mismo cambia mucho si los datos cambian un poco: tiene poco sesgo y mucha **varianza**. La idea de esta libreta es tan vieja como la estadística: si una estimación es ruidosa, **promedia muchas**. El problema es que solo tenemos un conjunto de entrenamiento. Vamos a:
#
# 1. ver con datos del censo de Estados Unidos qué tan mal generaliza un árbol solo;
# 2. derivar la varianza de un promedio de estimaciones **correlacionadas** y verla en una simulación;
# 3. fabricar «muchos conjuntos de entrenamiento» con el *bootstrap* y programar *bagging* desde cero;
# 4. obtener casi gratis una estimación del error de prueba con los datos **fuera de la bolsa** (OOB);
# 5. bajar la correlación entre árboles sorteando atributos: los **bosques aleatorios**;
# 6. medir el sesgo y la varianza de un árbol y de un promedio de árboles con las bicicletas de Washington;
# 7. preguntarle al bosque qué atributos usa, y descubrir que una de las dos formas de hacerlo tiene trampa.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score, root_mean_squared_error

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Datos.** Usamos otra vez la muestra del censo de Estados Unidos de 1994 del Laboratorio 1: 12,000 personas, tomadas del conjunto *Adult* que prepararon Ronny Kohavi y Barry Becker y que distribuye el repositorio de aprendizaje automático de la UCI (licencia CC BY 4.0). De cada persona sabemos su edad, tipo de empleador (`workclass`), escolaridad (`education-num`, de 1 = preescolar a 16 = doctorado), estado civil, ocupación, relación dentro del hogar (`relationship`), raza, sexo, ganancias y pérdidas de capital, horas trabajadas a la semana y país de origen. Los faltantes vienen escritos como `?`.
#
# **Problema.** Predecir si la persona gana **más de 50,000 dólares al año** (columna `ingreso`). Como en el laboratorio, descartamos `education` (es la misma información que `education-num`) y `fnlwgt` (un peso muestral del censo, no un atributo de la persona).
#
# Separamos un 25 % para prueba, estratificado por la clase. Esa parte no se toca hasta el final de cada comparación.

# %%
censo = rdp.datos.cargar("censo_adult", na_values="?")
y = (censo["ingreso"] == ">50K").astype(int).to_numpy()
X = censo.drop(columns=["ingreso", "education", "fnlwgt"])
categoricas = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
numericas = [c for c in X.columns if c not in categoricas]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
print(f"{len(X):,} personas, {X.shape[1]} atributos; ganan más de 50,000: {y.mean():.1%}")
print(f"entrenamiento: {len(X_tr):,} · prueba: {len(X_te):,}")

# %% [markdown]
# **Codificación para los árboles.** Un árbol no necesita estandarizar (solo compara un atributo contra un umbral, y eso no cambia con transformaciones monótonas) ni *one-hot*. Basta con convertir cada categoría en un número entero con `OrdinalEncoder`. El orden de esos enteros es arbitrario (alfabético) y el árbol lo usa como si significara algo, pero como puede hacer varias divisiones seguidas sobre el mismo atributo, termina aislando las categorías que le importan. Los faltantes y las categorías que no aparecen en entrenamiento se codifican como −1, un valor más que el árbol puede separar.

# %%
codificador = ColumnTransformer(
    [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), categoricas)],
    remainder="passthrough", verbose_feature_names_out=False,
).set_output(transform="pandas")
A_tr = codificador.fit_transform(X_tr)
A_te = codificador.transform(X_te)
atributos = A_tr.columns.tolist()
A_tr, A_te = A_tr.to_numpy(float), A_te.to_numpy(float)
N, d = A_tr.shape
print(f"N = {N:,}, d = {d}")

# %% [markdown]
# ### Un árbol solo
#
# Empecemos con un árbol de clasificación que crece hasta que cada hoja es pura (el comportamiento por omisión de `DecisionTreeClassifier`).

# %%
arbol = DecisionTreeClassifier(random_state=SEMILLA).fit(A_tr, y_tr)
p_arbol = arbol.predict_proba(A_te)[:, 1]
print(f"profundidad {arbol.get_depth()}, {arbol.get_n_leaves():,} hojas")
print(f"exactitud en entrenamiento: {accuracy_score(y_tr, arbol.predict(A_tr)):.3f}")
print(f"exactitud en prueba:        {accuracy_score(y_te, p_arbol >= 0.5):.3f}")
print(f"AUC en prueba:              {roc_auc_score(y_te, p_arbol):.3f}")

# %% [markdown]
# El árbol memorizó: en entrenamiento acierta el 98.9 % (no llega al 100 % solo porque hay personas con atributos idénticos y distinto ingreso) y en prueba falla en casi una de cada cinco personas. Su AUC es pobre porque sus hojas son puras: casi todas sus «probabilidades» son 0 o 1, y no sabe ordenar a las personas por grado de certeza. Podríamos podarlo (como en el ejemplo anterior), pero vamos a tomar otro camino: dejar los árboles profundos, con poco sesgo, y **quitarles la varianza promediando**.

# %% [markdown]
# ## 2. La sabiduría de la multitud
#
# En la exposición de ganado y aves de corral del oeste de Inglaterra, en Plymouth, se organizó un concurso: por seis peniques, cualquiera podía apostar cuánto pesaría un buey ya sacrificado y limpiado. Francis Galton consiguió los boletos y publicó el análisis en *Nature* («Vox populi», 1907): de 787 estimaciones legibles, la **mediana** fue de 1,207 libras; el peso real resultó de 1,198. La multitud se equivocó por 9 libras, menos del 1 %, mucho menos que la mayoría de los concursantes. (Galton usó la mediana, y no el promedio, porque así cada boleto es «un voto» y un disparate no la mueve.)
#
# ¿Por qué funciona promediar? ¿Y cuándo no funciona? Modelemos cada estimación como una variable aleatoria $Z_b$, $b = 1, \dots, B$, todas con la misma media $\mu$, la misma varianza $\sigma^2$ y la misma correlación $\rho$ entre cada par.
#
# **Proposición.** La varianza del promedio $\bar Z = \frac1B \sum_b Z_b$ es
#
# $$
# \operatorname{Var}(\bar Z) = \rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2 .
# $$
#
# *Demostración.* Por bilinealidad de la covarianza,
#
# $$
# \operatorname{Var}\Big(\frac1B \sum_b Z_b\Big) = \frac{1}{B^2} \sum_{b} \sum_{c} \operatorname{Cov}(Z_b, Z_c)
# = \frac{1}{B^2}\Big[ B\,\sigma^2 + B(B-1)\,\rho\,\sigma^2 \Big]
# = \frac{\sigma^2}{B} + \frac{B-1}{B}\,\rho\,\sigma^2 ,
# $$
#
# que reacomodado es la fórmula. $\square$
#
# Dos lecturas:
#
# - si las estimaciones son **independientes** ($\rho = 0$), la varianza baja como $\sigma^2/B$ y se puede hacer tan chica como se quiera;
# - si están **correlacionadas**, por muchos que promediemos queda un piso de $\rho\sigma^2$. Si todos los visitantes de la feria hubieran leído el mismo periódico con la misma estimación equivocada, la mediana heredaría ese error.
#
# Verifiquémoslo por simulación. Una manera de fabricar variables con correlación $\rho$ es $Z_b = \sqrt{\rho}\,U + \sqrt{1-\rho}\,E_b$, con $U, E_1, \dots, E_B$ normales estándar independientes: $U$ es el error que todos comparten.

# %%
def var_promedio(rho, B, sigma2=1.0):
    """Varianza del promedio de B variables con varianza sigma2 y correlación rho entre pares."""
    return rho * sigma2 + (1 - rho) * sigma2 / B


rng = np.random.default_rng(SEMILLA)
REPETICIONES = 4000
Bs = np.array([1, 3, 10, 30, 100, 300, 1000])
rhos = [0.0, 0.05, 0.2, 0.5]
simulada = {}
for rho in rhos:
    U = rng.standard_normal((REPETICIONES, 1))
    E = rng.standard_normal((REPETICIONES, Bs.max()))
    Z = np.sqrt(rho) * U + np.sqrt(1 - rho) * E
    simulada[rho] = [Z[:, :B].mean(axis=1).var() for B in Bs]

pd.DataFrame({f"ρ = {rho}": [f"{s:.4f} ({var_promedio(rho, B):.4f})" for s, B in zip(simulada[rho], Bs)] for rho in rhos},
             index=pd.Index(Bs, name="B")).rename_axis(columns="simulada (fórmula)")

# %%
fig, ax = plt.subplots(figsize=(9, 5))
B_fino = np.unique(np.round(np.logspace(0, 3, 200)).astype(int))
for rho, color in zip(rhos, [AZUL, ROJO, DORADO_OSC, GRIS]):
    ax.plot(B_fino, var_promedio(rho, B_fino), color=color, lw=2.2)
    ax.plot(Bs, simulada[rho], "o", color=color, ms=6)
    if rho > 0:
        ax.axhline(rho, color=color, lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(5e-4, 1.3)
ax.set(xlabel="número de estimaciones promediadas, B", ylabel="Var(promedio) / σ²",
       title="El promedio de estimaciones correlacionadas tiene un piso: ρσ²")
ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
rdp.estilo.etiquetar(ax, 1300, {f"ρ = {rho:g}": var_promedio(rho, 1000) for rho in rhos}, separacion=0)
fig.tight_layout()
plt.show()

# %% [markdown]
# Los puntos (simulación) caen sobre las curvas (fórmula). Con $\rho = 0.2$, promediar 10 estimaciones ya baja la varianza a 0.28 de la original, pero pasar de 100 a 1,000 casi no ayuda: estamos pegados al piso de 0.2.
#
# Para los árboles, esto dice dos cosas: promediar muchos árboles reduce la varianza, y la reduce más mientras **menos correlacionados** estén entre sí. Pero primero necesitamos muchos árboles distintos, y solo tenemos un conjunto de entrenamiento.

# %% [markdown]
# ## 3. *Bootstrap* y *bagging*
#
# ### El *bootstrap*
#
# Bradley Efron (1979) propuso imitar el muestreo de la población remuestreando **la muestra misma**: se sacan $N$ datos **con reemplazo** de los $N$ originales. Cada remuestra *bootstrap* tiene el mismo tamaño que la original, pero algunos datos aparecen repetidos y otros no aparecen. Ya lo usamos en la Unidad 4 para medir la incertidumbre de la regresión logística.
#
# ### Un árbol cambia mucho con los datos
#
# Entrenemos árboles profundos en dos remuestras distintas del mismo conjunto de entrenamiento y comparemos.

# %%
def remuestra(n, rng):
    """Índices de una remuestra bootstrap: n enteros de 0 a n−1, con reemplazo."""
    return rng.integers(0, n, n)


rng = np.random.default_rng(SEMILLA)
dos = []
for _ in range(2):
    idx = remuestra(N, rng)
    dos.append(DecisionTreeClassifier(random_state=SEMILLA).fit(A_tr[idx], y_tr[idx]))
for k, t in enumerate(dos, 1):
    print(f"árbol {k}: profundidad {t.get_depth()}, {t.get_n_leaves():,} hojas, "
          f"exactitud de prueba {accuracy_score(y_te, t.predict(A_te)):.3f}")
discrepan = (dos[0].predict(A_te) != dos[1].predict(A_te)).mean()
print(f"las dos predicciones discrepan en el {discrepan:.1%} de las personas de prueba")

# %% [markdown]
# Los dos árboles tienen exactitudes parecidas, pero **no se equivocan en las mismas personas**: discrepan en una de cada seis. Esa es la varianza del árbol, y es justo lo que un promedio puede quitar.
#
# ### *Bagging* desde cero
#
# Leo Breiman (1996) llamó *bagging* (de *bootstrap aggregating*) a la receta:
#
# 1. para $b = 1, \dots, B$: saca una remuestra *bootstrap* $\mathcal{D}_b$ y entrena un árbol profundo $T_b$ con ella;
# 2. predice con el promedio: $\hat f(\mathbf{x}) = \frac1B\sum_b T_b(\mathbf{x})$.
#
# En clasificación, $T_b(\mathbf{x})$ es la proporción de positivos en la hoja (casi siempre 0 o 1), así que $\hat f(\mathbf{x})$ es casi la fracción de árboles que votan por la clase positiva. Guardamos también qué datos entraron a cada remuestra, porque los vamos a necesitar en la sección 4.
#
# La función recibe además `max_features`, el número de atributos que el árbol sortea en cada división. Con `None` (todos) es *bagging*; en la sección 5 le daremos otro valor.

# %%
def ensamble(A, y, B, max_features=None, semilla=SEMILLA):
    """B árboles profundos, cada uno entrenado en una remuestra bootstrap. Regresa (árboles, remuestras)."""
    rng = np.random.default_rng(semilla)
    arboles, bolsas = [], []
    for _ in range(B):
        idx = remuestra(len(y), rng)
        t = DecisionTreeClassifier(max_features=max_features, random_state=int(rng.integers(2**31)))
        arboles.append(t.fit(A[idx], y[idx]))
        bolsas.append(idx)
    return arboles, bolsas


B = 300
inicio = time.perf_counter()
arboles_bag, bolsas_bag = ensamble(A_tr, y_tr, B)
print(f"{B} árboles en {time.perf_counter() - inicio:.1f} s")

# predicciones de cada árbol en prueba: matriz B × N_prueba
P_bag = np.array([t.predict_proba(A_te)[:, 1] for t in arboles_bag])
p_bag = P_bag.mean(axis=0)
print(f"bagging: exactitud de prueba {accuracy_score(y_te, p_bag >= 0.5):.3f}, AUC {roc_auc_score(y_te, p_bag):.3f}")

# %% [markdown]
# Con el mismo tipo de árbol, sin cambiar nada más, la AUC de prueba sube de 0.76 a 0.88 y la exactitud de 0.815 a 0.841. Ahora el ensamble sí sabe ordenar: una persona por la que votan 280 de 300 árboles está más claramente arriba de 50,000 dólares que una por la que votan 160.
#
# ¿Cuántos árboles hacen falta? Como ya tenemos las predicciones de cada uno, el error del ensamble con los primeros $b$ árboles sale de promedios acumulados.

# %%
def curva_prueba(P, y):
    """Error y AUC de prueba del promedio de los primeros b árboles, para b = 1, …, B."""
    acumulado = np.cumsum(P, axis=0) / np.arange(1, len(P) + 1)[:, None]
    error = np.array([np.mean((p >= 0.5) != y) for p in acumulado])
    auc = np.array([roc_auc_score(y, p) for p in acumulado])
    return error, auc


error_bag, auc_bag = curva_prueba(P_bag, y_te)
for b in (1, 10, 50, 100, 300):
    print(f"B = {b:>3}: error de prueba {error_bag[b - 1]:.3f}, AUC {auc_bag[b - 1]:.3f}")

# %% [markdown]
# La mejora llega rápido: con 10 árboles ya se tiene buena parte, y con 50 casi todo. Agregar árboles **no sobreajusta**: cada árbol nuevo solo hace que el promedio se acerque a su límite ($B \to \infty$), y lo que se puede ganar ya está descrito por la proposición de la sección 2: la parte $\frac{1-\rho}{B}\sigma^2$ desaparece, la parte $\rho\sigma^2$ no.

# %% [markdown]
# ## 4. El error fuera de la bolsa
#
# ¿Qué fracción de los datos queda fuera de una remuestra? Un dato dado no sale en un sorteo con probabilidad $1 - 1/N$; los $N$ sorteos son independientes, así que
#
# $$
# P(\text{el dato } n \text{ no está en } \mathcal{D}_b) = \Big(1 - \frac1N\Big)^N \xrightarrow[N\to\infty]{} e^{-1} \approx 0.368 .
# $$
#
# (El límite sale de $\ln(1 - 1/N)^N = N\ln(1 - 1/N) = N\big(-\tfrac1N - \tfrac{1}{2N^2} - \cdots\big) \to -1$.) Ya con $N = 10$ la probabilidad es 0.349. Cada árbol deja fuera más o menos un tercio de los datos: su **bolsa** contiene alrededor del 63.2 % de los datos distintos.

# %%
fuera = np.array([1 - len(np.unique(idx)) / N for idx in bolsas_bag])
print(f"(1 − 1/N)^N con N = {N:,}: {(1 - 1 / N) ** N:.4f}   ·   e^(−1) = {np.exp(-1):.4f}")
print(f"fracción fuera de la bolsa en nuestros {B} árboles: media {fuera.mean():.4f}, "
      f"de {fuera.min():.4f} a {fuera.max():.4f}")

# %% [markdown]
# Esos datos que un árbol no vio son, para ese árbol, datos de prueba. La **estimación fuera de la bolsa** (*out-of-bag*, OOB) de Breiman aprovecha esto: para cada dato de entrenamiento $\mathbf{x}_n$, promediamos solo los árboles que **no** lo usaron,
#
# $$
# \hat f_{\text{OOB}}(\mathbf{x}_n) = \frac{1}{|\{b : n \notin \mathcal{D}_b\}|} \sum_{b\,:\, n \notin \mathcal{D}_b} T_b(\mathbf{x}_n),
# $$
#
# y medimos el error de esas predicciones. Cada dato recibe el voto de alrededor de $0.368\,B$ árboles: es un ensamble más chico que el real, así que con pocos árboles el estimado es pesimista; con cientos, la diferencia ya es menor que la variación por muestreo. Y no cuesta entrenar nada más.

# %%
def curva_oob(arboles, bolsas, A, y):
    """Error y AUC fuera de la bolsa del ensamble con los primeros b árboles, para b = 1, …, B."""
    suma, cuenta = np.zeros(len(y)), np.zeros(len(y))
    error, auc = [], []
    for t, idx in zip(arboles, bolsas):
        fuera = np.ones(len(y), dtype=bool)
        fuera[idx] = False
        suma[fuera] += t.predict_proba(A[fuera])[:, 1]
        cuenta[fuera] += 1
        con = cuenta > 0  # al principio, algunos datos todavía no tienen votos fuera de la bolsa
        p = suma[con] / cuenta[con]
        error.append(np.mean((p >= 0.5) != y[con]))
        auc.append(roc_auc_score(y[con], p))
    return np.array(error), np.array(auc), cuenta


oob_error_bag, oob_auc_bag, cuenta_bag = curva_oob(arboles_bag, bolsas_bag, A_tr, y_tr)
print(f"cada dato quedó fuera de la bolsa en {cuenta_bag.mean():.1f} árboles en promedio (0.368·B = {0.368 * B:.1f})")
print(f"OOB:    error {oob_error_bag[-1]:.3f}, AUC {oob_auc_bag[-1]:.3f}")
print(f"prueba: error {error_bag[-1]:.3f}, AUC {auc_bag[-1]:.3f}")

# %% [markdown]
# El estimado OOB, calculado **solo con el conjunto de entrenamiento**, da un error de 0.157 contra 0.159 en prueba, y una AUC de 0.892 contra 0.884: diferencias del tamaño de la variación que esperaríamos entre dos muestras de 3,000 personas. Lo comparamos con validación cruzada en la sección 5, ya con el bosque.

# %% [markdown]
# ## 5. Bosques aleatorios
#
# ### Árboles menos parecidos
#
# Los árboles del *bagging* se parecen mucho entre sí: todos ven casi los mismos datos y todos eligen primero las divisiones más fuertes (en el censo, el estado civil y la escolaridad). Su correlación $\rho$ es alta y el promedio choca pronto con el piso $\rho\sigma^2$.
#
# Breiman (2001) agregó una segunda fuente de azar: en **cada división** de cada árbol se sortean $m$ de los $d$ atributos y solo se busca la mejor división entre esos $m$. Con $m < d$, a veces el atributo dominante no está disponible y el árbol tiene que usar otro: los árboles se diversifican, $\rho$ baja, a cambio de que cada árbol sea un poco peor ($\sigma^2$ y el sesgo suben un poco). Ese es un **bosque aleatorio** (*random forest*). Los valores habituales son $m = \lfloor\sqrt d\rfloor$ en clasificación y $m = d/3$ en regresión; con $m = d$ se recupera *bagging*.
#
# En el censo, $d = 12$ y $\lfloor\sqrt{12}\rfloor = 3$. Usamos la misma función con `max_features=3`.

# %%
arboles_rf, bolsas_rf = ensamble(A_tr, y_tr, B, max_features=3)
P_rf = np.array([t.predict_proba(A_te)[:, 1] for t in arboles_rf])
error_rf, auc_rf = curva_prueba(P_rf, y_te)
oob_error_rf, oob_auc_rf, _ = curva_oob(arboles_rf, bolsas_rf, A_tr, y_tr)

# correlación media entre las predicciones de dos árboles distintos, en las personas de prueba
def correlacion_media(P):
    C = np.corrcoef(P)
    return C[np.triu_indices_from(C, k=1)].mean()

pd.DataFrame({
    nombre: {"correlación entre árboles": correlacion_media(P),
             "AUC media de un árbol": np.mean([roc_auc_score(y_te, p) for p in P]),
             "AUC del ensamble": auc[-1], "error de prueba": err[-1], "error OOB": oob[-1]}
    for nombre, P, err, auc, oob in [("bagging (m = 12)", P_bag, error_bag, auc_bag, oob_error_bag),
                                     ("bosque (m = 3)", P_rf, error_rf, auc_rf, oob_error_rf)]
}).T.round(3)

# %% [markdown]
# Cada árbol del bosque es individualmente un poco **peor** que uno del *bagging* (le quitamos opciones en cada división), pero se parecen menos entre sí, y el promedio sale mejor: el error de prueba baja de 0.159 a 0.151. Es la proposición de la sección 2 en acción: $\sigma^2$ sube un poco, $\rho$ baja más.
#
# (La correlación de la tabla es entre las predicciones de dos árboles **a lo largo de las personas de prueba**, una medida práctica de qué tanto se parecen. La $\rho$ de la proposición es otra cosa: la correlación, en un punto fijo $\mathbf{x}$, a lo largo de distintos conjuntos de entrenamiento. La estimaremos con las bicicletas en la sección 6.)
#
# Veamos las curvas completas, de prueba y fuera de la bolsa.

# %%
b_ = np.arange(1, B + 1)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(b_, error_bag, color=GRIS, lw=2, label="bagging, prueba")
ax.plot(b_, oob_error_bag, color=GRIS, lw=1.5, ls="--", label="bagging, OOB")
ax.plot(b_, error_rf, color=AZUL, lw=2, label="bosque (m = 3), prueba")
ax.plot(b_, oob_error_rf, color=AZUL, lw=1.5, ls="--", label="bosque (m = 3), OOB")
ax.axhline(1 - accuracy_score(y_te, p_arbol >= 0.5), color=ROJO, lw=1.2, ls=":")
ax.text(B, 1 - accuracy_score(y_te, p_arbol >= 0.5) + 0.001, "un árbol profundo solo (prueba)", color=ROJO, ha="right", va="bottom")
ax.set_ylim(0.14, 0.20)
ax.set_xlim(0, B)
ax.set(xlabel="número de árboles, B", ylabel="error de clasificación",
       title="Agregar árboles no sobreajusta; el bosque baja más que el bagging")
rdp.estilo.etiquetar(ax, B + 6, {"bagging, prueba": error_bag[-1], "bagging, OOB": oob_error_bag[-1],
                                "bosque, prueba": error_rf[-1], "bosque, OOB": oob_error_rf[-1]})
fig.tight_layout()
plt.show()

# %% [markdown]
# Tres cosas en la gráfica:
#
# - Con un solo árbol el error es muy alto (y el OOB, con pocos votos por dato, todavía más); en unas decenas de árboles se estabiliza.
# - Ninguna curva vuelve a subir: $B$ no es un hiperparámetro que haya que afinar, solo hay que poner «suficientes» árboles (el costo es tiempo).
# - Las curvas OOB (punteadas) siguen de cerca a las de prueba sin haber usado ningún dato de prueba. Los vaivenes de la curva de prueba (por ejemplo, alrededor de 120 árboles) son ruido: 0.003 de error son 9 personas de 3,000.
#
# ### El número de atributos por división, $m$
#
# $m$ sí es un hiperparámetro: controla el compromiso entre árboles buenos y árboles distintos. Usemos ahora `RandomForestClassifier`, que hace lo mismo que nuestra función (en paralelo y en C) y calcula el estimado OOB con `oob_score=True`; con `oob_decision_function_` obtenemos las probabilidades OOB y de ahí la AUC. Elegimos $m$ **sin tocar la prueba**.

# %%
def bosque(m="sqrt", hoja=1, B=300, **kw):
    return RandomForestClassifier(n_estimators=B, max_features=m, min_samples_leaf=hoja, oob_score=True,
                                  n_jobs=-1, random_state=SEMILLA, **kw)


def auc_oob(modelo):
    return roc_auc_score(y_tr, modelo.oob_decision_function_[:, 1])


ms = list(range(1, d + 1))
oob_m = [auc_oob(bosque(m).fit(A_tr, y_tr)) for m in ms]
pd.Series(oob_m, index=pd.Index(ms, name="m"), name="AUC OOB").round(4).to_frame().T

# %% [markdown]
# Con $m = 1$ cada división se hace sobre un atributo elegido completamente al azar y los árboles son peores; con $m = d = 12$ es *bagging*. El mejor valor es $m = 3$, justo $\lfloor\sqrt{d}\rfloor$, aunque las diferencias son pequeñas (de 0.892 a 0.897). Los árboles profundos tienen otro hiperparámetro que en la práctica pesa más en la AUC: el **tamaño mínimo de las hojas**. Con hojas puras de un dato, cada árbol vota 0 o 1; con hojas más grandes, cada árbol da una proporción, y el promedio estima mejor la probabilidad.

# %%
hojas = [1, 2, 5, 10, 20, 50]
m_mejor = ms[int(np.argmax(oob_m))]
oob_hoja = [auc_oob(bosque(m_mejor, h).fit(A_tr, y_tr)) for h in hojas]
hoja_mejor = hojas[int(np.argmax(oob_hoja))]
print(f"m = {m_mejor}")
pd.Series(oob_hoja, index=pd.Index(hojas, name="tamaño mínimo de hoja"), name="AUC OOB").round(4).to_frame().T

# %%
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
ejes[0].plot(ms, oob_m, "o-", color=AZUL, lw=2)
ejes[0].axvline(d, color=GRIS, lw=1, ls=":")
ejes[0].text(d - 0.2, min(oob_m) + 0.001, "m = d:\nbagging", color=TENUE, ha="right", va="bottom")
ejes[0].set(xlabel="atributos sorteados por división, m", ylabel="AUC fuera de la bolsa",
            title="Hojas de un dato", xticks=ms)
ejes[1].plot(hojas, oob_hoja, "o-", color=AZUL, lw=2)
ejes[1].set_xscale("log")
ejes[1].set_xticks(hojas, [str(h) for h in hojas])
ejes[1].set(xlabel="tamaño mínimo de hoja", title=f"m = {m_mejor}")
fig.suptitle("Elegir hiperparámetros con el estimado fuera de la bolsa (sin tocar la prueba)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### OOB contra validación cruzada
#
# Para el bosque elegido, comparemos tres estimaciones de su AUC: fuera de la bolsa (un solo ajuste), validación cruzada de 5 pliegues (cinco ajustes con el 80 % de los datos) y, ahora sí, la prueba.

# %%
inicio = time.perf_counter()
elegido = bosque(m_mejor, hoja_mejor, B=500).fit(A_tr, y_tr)
t_oob = time.perf_counter() - inicio

inicio = time.perf_counter()
pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
auc_cv = cross_val_score(bosque(m_mejor, hoja_mejor, B=500), A_tr, y_tr, cv=pliegues, scoring="roc_auc")
t_cv = time.perf_counter() - inicio

p_elegido = elegido.predict_proba(A_te)[:, 1]
pd.DataFrame({
    "AUC": [auc_oob(elegido), auc_cv.mean(), roc_auc_score(y_te, p_elegido)],
    "tiempo (s)": [t_oob, t_cv, np.nan],
}, index=["fuera de la bolsa", "validación cruzada (5)", "prueba"]).round(4)

# %% [markdown]
# Las tres difieren en menos de 0.002, y el OOB costó la quinta parte: salió del mismo ajuste que produce el modelo, mientras que la validación cruzada tuvo que entrenar cinco bosques. Por eso con bosques se suele usar OOB en lugar de validación cruzada. (Ojo: los tiempos dependen de la máquina; la proporción, alrededor de 1 a 5, no.)

# %% [markdown]
# ## 6. Sesgo y varianza, a la vista: las bicicletas
#
# Volvamos a las rentas de bicicletas de Washington de la Unidad 4: rentas por hora en 2011–2012, con entrenamiento en los días 1 a 19 de cada mes (10,886 horas) y prueba en los días 20 en adelante (6,493 horas). Allí, la escalera de modelos lineales llegó a una RECM de prueba de 86.4 con M4, que necesitó construir **a mano** la interacción hora × día laboral, y la regresión de Poisson con los mismos atributos bajó a 72.2.
#
# Aquí le damos a los árboles los mismos atributos de M4, pero sin la interacción: hora, día laboral, clima, año, temperatura, humedad y viento. En regresión, la hoja predice el promedio de sus datos, y el ensamble promedia esas predicciones.

# %%
b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
b["anio"] = b.fecha.dt.year - 2011
b["clima"] = b.clima.clip(upper=3)
entrena = b[b.fecha.dt.day <= 19]
prueba = b[b.fecha.dt.day >= 20]
F = ["hora", "laboral", "clima", "anio", "temp", "humedad", "viento"]
Xb_tr, yb_tr = entrena[F].to_numpy(float), entrena.total.to_numpy(float)
Xb_te, yb_te = prueba[F].to_numpy(float), prueba.total.to_numpy(float)
print(f"entrenamiento: {len(yb_tr):,} horas · prueba: {len(yb_te):,} horas")

arbol_b = DecisionTreeRegressor(random_state=SEMILLA).fit(Xb_tr, yb_tr)
bag_b = RandomForestRegressor(n_estimators=300, max_features=None, n_jobs=-1, random_state=SEMILLA).fit(Xb_tr, yb_tr)
rf_b = RandomForestRegressor(n_estimators=300, max_features=3, n_jobs=-1, random_state=SEMILLA).fit(Xb_tr, yb_tr)
recm_b = {
    "M4 lineal (Unidad 4)": 86.4,
    "un árbol profundo": root_mean_squared_error(yb_te, arbol_b.predict(Xb_te)),
    "M4 Poisson (Unidad 4)": 72.2,
    "bosque, m = 3": root_mean_squared_error(yb_te, rf_b.predict(Xb_te)),
    "bagging (m = 7)": root_mean_squared_error(yb_te, bag_b.predict(Xb_te)),
}
pd.Series(recm_b, name="RECM de prueba").round(1).to_frame()

# %% [markdown]
# El árbol profundo (80.5) ya le gana a M4 lineal sin que nadie le diga que la hora importa distinto en días laborales y en fines de semana: lo descubre al partir. El promedio de 300 árboles baja la RECM a 61.7, muy por debajo de Poisson.
#
# Aquí, a diferencia del censo, sortear atributos **empeora**: hay solo 7, y la hora domina todo. Con $m = 3$, en más de la mitad de las divisiones la hora no está disponible y el árbol tiene que conformarse con algo peor. Los bosques aleatorios brillan cuando hay muchos atributos útiles; cuando hay pocos y uno manda, $m = d$ (*bagging*) puede ser lo mejor. Por eso $m$ se elige con OOB o validación cruzada, no se fija de antemano.
#
# ### Descomposición del error, con remuestras
#
# Como en la Unidad 3, para ver sesgo y varianza necesitamos **muchos conjuntos de entrenamiento**. Los fabricamos con el *bootstrap*: $R = 20$ remuestras del entrenamiento de las bicicletas. En cada una entrenamos un árbol profundo, un *bagging* de 50 árboles y un bosque de 50 árboles con $m = 3$, y guardamos las predicciones de cada árbol en las 6,493 horas de prueba. Para cada hora de prueba $\mathbf{x}$ y cada modelo $\hat f$:
#
# - la **varianza** es $\operatorname{Var}_r\big(\hat f_r(\mathbf{x})\big)$, cuánto cambia la predicción de una remuestra a otra;
# - el resto del error cuadrático, $\big(\overline{f}(\mathbf{x}) - y\big)^2$ con $\overline f$ el promedio sobre las remuestras, es **sesgo² + ruido** (no conocemos la verdadera $f$, así que no los podemos separar).
#
# Promediando sobre las horas de prueba, error cuadrático medio = varianza + (sesgo² + ruido), exactamente.

# %%
R, B_b = 20, 50
rng = np.random.default_rng(SEMILLA)
pred_arbol = np.empty((R, len(yb_te)))
pred_arboles = {"bagging": np.empty((R, B_b, len(yb_te)), dtype=np.float32),
                "bosque": np.empty((R, B_b, len(yb_te)), dtype=np.float32)}
inicio = time.perf_counter()
for r in range(R):
    idx = remuestra(len(yb_tr), rng)
    Xr, yr = Xb_tr[idx], yb_tr[idx]
    pred_arbol[r] = DecisionTreeRegressor(random_state=r).fit(Xr, yr).predict(Xb_te)
    for nombre, m in [("bagging", None), ("bosque", 3)]:
        f = RandomForestRegressor(n_estimators=B_b, max_features=m, n_jobs=-1, random_state=r).fit(Xr, yr)
        pred_arboles[nombre][r] = [t.predict(Xb_te) for t in f.estimators_]
print(f"{R} remuestras en {time.perf_counter() - inicio:.0f} s")


def descompone(P):
    """P: R × N_prueba. Regresa (varianza, sesgo² + ruido, ECM), promediados sobre la prueba."""
    varianza = P.var(axis=0).mean()
    resto = ((P.mean(axis=0) - yb_te) ** 2).mean()
    return varianza, resto, ((P - yb_te) ** 2).mean()


filas = {"un árbol profundo": descompone(pred_arbol)}
for nombre in pred_arboles:
    filas[f"{nombre} de {B_b} árboles"] = descompone(pred_arboles[nombre].mean(axis=1))
tabla_bv = pd.DataFrame(filas, index=["varianza", "sesgo² + ruido", "ECM"]).T
tabla_bv["RECM"] = np.sqrt(tabla_bv["ECM"])
tabla_bv.round(1)

# %% [markdown]
# El *bagging* divide la varianza entre más de cinco (de 2,619 a 463) y casi toda la mejora del ECM viene de ahí. El término de sesgo² + ruido también baja un poco (8 %): el promedio de muchas funciones escalonadas es más suave que cualquiera de ellas, y aproxima mejor una función suave. El bosque con $m = 3$ tiene todavía menos varianza, pero su sesgo es mayor (árboles peores, porque a veces no pueden usar la hora), y en la suma pierde.
#
# ### $\rho$ y $\sigma^2$ en datos reales
#
# Con las predicciones de cada árbol podemos estimar las dos cantidades de la proposición en cada hora de prueba $\mathbf{x}$:
#
# - $\sigma^2(\mathbf{x})$, la varianza de **un árbol** del ensamble, sobre todas las remuestras y todos los árboles;
# - $\rho(\mathbf{x})\sigma^2(\mathbf{x})$, la covarianza entre **dos árboles distintos del mismo ensamble**: comparten el conjunto de entrenamiento y solo difieren en su remuestra interna y en el sorteo de atributos.
#
# Esa covarianza es justo la varianza que el ensamble no puede quitar: si $B \to \infty$, la varianza del promedio tiende a $\rho\sigma^2$.

# %%
def rho_sigma(T):
    """T: R × B × N. Regresa (σ², ρσ²) promediados sobre los puntos de prueba."""
    R, B, _ = T.shape
    media = T.mean(axis=(0, 1))
    sigma2 = ((T - media) ** 2).mean(axis=(0, 1))
    S = (T - media).sum(axis=1)                      # suma sobre los árboles de cada remuestra
    cov = ((S ** 2 - ((T - media) ** 2).sum(axis=1)) / (B * (B - 1))).mean(axis=0)  # pares b ≠ c
    return sigma2.mean(), cov.mean()


filas = {}
for nombre, T in pred_arboles.items():
    s2, cov = rho_sigma(T.astype(float))
    filas[nombre] = {"σ²": s2, "ρσ²": cov, "ρ": cov / s2, f"fórmula, B = {B_b}": var_promedio(cov / s2, B_b, s2),
                     "varianza medida": T.mean(axis=1).var(axis=0).mean()}
pd.DataFrame(filas).T.round({"σ²": 0, "ρσ²": 0, "ρ": 3, f"fórmula, B = {B_b}": 0, "varianza medida": 0})

# %% [markdown]
# Un árbol del *bagging* tiene varianza $\sigma^2 \approx$ 3,160 y correlación $\rho \approx 0.13$ con sus compañeros: el promedio de 50 queda en $\rho\sigma^2 + (1-\rho)\sigma^2/50 \approx$ 463, y aunque usáramos infinitos árboles no bajaría de $\rho\sigma^2 \approx$ 408. En el bosque, cada árbol es el doble de variable ($\sigma^2 \approx$ 6,450), pero la correlación cae a 0.03 y el piso a 195. La columna «medida» es la varianza observada de los ensambles en la tabla anterior: coincide con la fórmula. (Coincide **exactamente**, porque con medias muestrales la proposición es una identidad algebraica; lo valioso es la separación en las dos partes.)
#
# Una imagen vale más que las tablas. Tomemos el martes 29 de marzo de 2011, un día del conjunto de prueba, y dibujemos lo que predicen, hora por hora, los árboles profundos de 8 remuestras distintas y los *bagging* de esas mismas remuestras.

# %%
dia = (prueba.fecha.dt.date == pd.Timestamp("2011-03-29").date()).to_numpy()
horas = prueba.hora.to_numpy()[dia]
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, P, color, titulo in [(ejes[0], pred_arbol, ROJO, "Árboles profundos de 8 remuestras"),
                             (ejes[1], pred_arboles["bagging"].mean(axis=1), AZUL, f"Bagging de {B_b} árboles, mismas 8 remuestras")]:
    for r in range(8):
        ax.plot(horas, P[r, dia], color=color, lw=1.4, alpha=0.6)
    ax.plot(horas, yb_te[dia], "o", color=TINTA, ms=5, zorder=3)
    ax.set(xlabel="hora", title=titulo, xticks=range(0, 24, 4))
ejes[0].set_ylabel("bicicletas rentadas")
ejes[0].text(0.5, 330, "puntos: rentas observadas", color=TENUE, fontsize=10)
fig.suptitle("Martes 29 de marzo de 2011 (prueba)")
fig.tight_layout()
plt.show()

# %% [markdown]
# A la izquierda, cada árbol sigue la forma del día (picos a las 8 y a las 17–18 h), pero con saltos que dependen de qué horas cayeron en su remuestra: en las horas pico, las predicciones de distintas remuestras se separan más de cien bicicletas. A la derecha, los promedios casi se enciman. Eso es reducir varianza.

# %% [markdown]
# ## 7. ¿Qué atributos usa el bosque?
#
# Un bosque de 500 árboles no se puede leer como un árbol. Hay dos maneras comunes de resumir qué tanto pesa cada atributo:
#
# - **Disminución media de la impureza** (MDI, *mean decrease in impurity*): cada vez que un árbol divide un nodo con el atributo $j$, la impureza (Gini) baja; se suman esas bajadas, ponderadas por la fracción de datos que llegan al nodo, y se promedian sobre los árboles. Es lo que da `feature_importances_`, y sale gratis del entrenamiento.
# - **Importancia por permutación**: en un conjunto de datos que el modelo **no** usó para entrenar, se revuelve al azar la columna $j$ (rompiendo su relación con $y$ sin cambiar su distribución) y se mide cuánto empeora la métrica. Si el modelo no usa el atributo, revolverlo no cambia nada.
#
# Para ver la diferencia, agreguemos dos columnas **inútiles** por construcción: `ruido`, un número normal al azar, y `id`, un número de identificación distinto para cada persona. Ninguna tiene relación con el ingreso.

# %%
rng = np.random.default_rng(SEMILLA)
def con_inutiles(A):
    return np.column_stack([A, rng.standard_normal(len(A)), rng.permutation(len(A))])

nombres = atributos + ["ruido", "id"]
A2_tr, A2_te = con_inutiles(A_tr), con_inutiles(A_te)
bosque_imp = bosque(B=300).fit(A2_tr, y_tr)
perm = permutation_importance(bosque_imp, A2_te, y_te, scoring="roc_auc", n_repeats=5, random_state=SEMILLA)
imp = pd.DataFrame({"MDI": bosque_imp.feature_importances_, "permutación (prueba)": perm.importances_mean,
                    "valores distintos": [len(np.unique(A2_tr[:, j])) for j in range(len(nombres))]},
                   index=nombres).sort_values("MDI", ascending=False)
imp.round(3)

# %%
orden = imp.index[::-1]
fig, ejes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
colores = [ROJO if n in ("ruido", "id") else AZUL for n in orden]
ejes[0].barh(orden, imp.loc[orden, "MDI"], color=colores)
ejes[0].set(title="Disminución media de la impureza (entrenamiento)", xlabel="importancia (suman 1)")
ejes[1].barh(orden, imp.loc[orden, "permutación (prueba)"], color=colores)
ejes[1].axvline(0, color=TENUE, lw=0.8)
ejes[1].set(title="Permutación (prueba)", xlabel="caída de la AUC al revolver la columna")
fig.tight_layout()
plt.show()

# %% [markdown]
# Según MDI, `id` y `ruido` son el tercer y cuarto atributos **más importantes** del bosque, por encima de `relationship` y `capital-gain`, que son los dos más importantes según la permutación. Según la permutación en prueba, `id` y `ruido` valen cero (−0.001: revolver una columna inútil a veces mejora la AUC por azar), como debe ser.
#
# El sesgo de MDI tiene una explicación sencilla. Los árboles crecen hasta hojas puras, y en los nodos profundos, con pocos datos, cualquier atributo con **muchos valores distintos** ofrece muchísimos umbrales posibles, y alguno separa por casualidad los pocos datos que quedan. Cada una de esas divisiones baja la impureza **en entrenamiento**, y MDI la cuenta como mérito. `id` tiene 9,000 valores distintos y `ruido` también; `sex` tiene 2. Strobl, Boulesteix, Zeileis y Hothorn (*BMC Bioinformatics*, 2007) documentaron este sesgo y propusieron alternativas. La importancia por permutación en datos de prueba no lo tiene, porque mide desempeño fuera de la muestra; su propio problema aparece con atributos **correlacionados** (lo veremos en el tema de interpretabilidad).
#
# Regla práctica: para decir qué usa un modelo, **permutación en prueba**; MDI solo como primera mirada, y nunca para comparar atributos con números de valores muy distintos.
#
# **Una nota sobre estos datos.** La permutación dice que el bosque se apoya sobre todo en `relationship`, `capital-gain` y `education-num`, y casi nada en `sex` (0.002). No hay que leer eso como «el sexo no importa para el ingreso»: como vimos en el Laboratorio 1, `relationship` (esposo / esposa) contiene el sexo casi entero, y si revolvemos `sex` el bosque sigue sabiéndolo por otro lado. Además, lo que aprende el modelo describe la desigualdad de ingresos en Estados Unidos en 1994, no una relación causal ni un criterio aceptable para decidir sobre personas.

# %% [markdown]
# ## 8. Comparación final y *extra trees*
#
# Una variante más: los *extremely randomized trees* o *extra trees* (Geurts, Ernst y Wehenkel, 2006) llevan el azar un paso más allá: además de sortear los $m$ atributos, sortean el **umbral** de cada uno (uniforme entre el mínimo y el máximo del nodo) y eligen el mejor de esos $m$ cortes al azar. Por omisión no usan *bootstrap*. Son más rápidos de entrenar (no hay que ordenar) y sus árboles todavía menos correlacionados.
#
# Comparemos todo en la prueba, incluida la regresión logística de la Unidad 4 con la preparación del Laboratorio 1 (*one-hot* para las categóricas, logaritmo para las ganancias de capital y estandarización).

# %%
sesgadas = ["capital-gain", "capital-loss"]
otras = [c for c in numericas if c not in sesgadas]
logistica = make_pipeline(
    ColumnTransformer([
        ("numericas", StandardScaler(), otras),
        ("sesgadas", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), sesgadas),
        ("nominales", make_pipeline(SimpleImputer(strategy="constant", fill_value="desconocido"),
                                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)), categoricas),
    ]),
    LogisticRegression(max_iter=2000),
)
modelos = {
    "regresión logística": (logistica, X_tr, X_te),
    "un árbol profundo": (DecisionTreeClassifier(random_state=SEMILLA), A_tr, A_te),
    "bagging": (bosque(None, B=500), A_tr, A_te),
    "bosque (por omisión)": (bosque("sqrt", B=500), A_tr, A_te),
    f"bosque (m = {m_mejor}, hoja = {hoja_mejor})": (bosque(m_mejor, hoja_mejor, B=500), A_tr, A_te),
    f"extra trees (m = {m_mejor}, hoja = {hoja_mejor})": (
        ExtraTreesClassifier(n_estimators=500, max_features=m_mejor, min_samples_leaf=hoja_mejor,
                             n_jobs=-1, random_state=SEMILLA), A_tr, A_te),
}
filas = {}
for nombre, (modelo, Xa, Xb) in modelos.items():
    inicio = time.perf_counter()
    modelo.fit(Xa, y_tr)
    segundos = time.perf_counter() - inicio
    p = modelo.predict_proba(Xb)[:, 1]
    filas[nombre] = {"AUC": roc_auc_score(y_te, p), "exactitud": accuracy_score(y_te, p >= 0.5), "tiempo (s)": segundos}
comparacion = pd.DataFrame(filas).T
comparacion.round(3)

# %% [markdown]
# - El árbol solo es el peor por mucho (AUC 0.763). El ensamble de esos mismos árboles da un salto enorme (0.886).
# - El bosque con los valores por omisión (0.891) **no** le gana en AUC a la regresión logística (0.903): con hojas de un dato, sus probabilidades son votos y ordenan peor. Ajustando $m$ y el tamaño de hoja con OOB, el bosque supera a la logística en las dos métricas (AUC 0.915, exactitud 0.860 contra 0.846).
# - *Extra trees* con los mismos hiperparámetros queda en medio (0.906) y entrena en la mitad del tiempo que el bosque.
# - La logística sigue siendo una línea base muy digna: entrena en una fracción del tiempo y se puede interpretar coeficiente por coeficiente. En la próxima libreta veremos el *boosting*, que en datos tabulares como estos suele ganarle a los dos.

# %% [markdown]
# ## 9. Cierre
#
# Un árbol profundo tiene poco sesgo y mucha varianza; un promedio de árboles conserva el sesgo y se lleva buena parte de la varianza, tanto más cuanto menos se parezcan los árboles entre sí. El *bootstrap* fabrica los conjuntos de entrenamiento que no tenemos; el sorteo de atributos en cada división (bosques aleatorios) los hace más distintos; y los datos que cada árbol no vio dan, gratis, una estimación honesta del error. Es difícil encontrar otro método que funcione tan bien con tan poco ajuste: por eso el bosque aleatorio es, desde 2001, la primera opción razonable para casi cualquier tabla de datos.
#
# *Referencias:* F. Galton, «Vox populi», *Nature* 75 (1907), 450–451. B. Efron, «Bootstrap methods: another look at the jackknife», *Annals of Statistics* 7 (1979). L. Breiman, «Bagging predictors», *Machine Learning* 24 (1996). L. Breiman, «Random forests», *Machine Learning* 45 (2001). P. Geurts, D. Ernst y L. Wehenkel, «Extremely randomized trees», *Machine Learning* 63 (2006). C. Strobl, A.-L. Boulesteix, A. Zeileis y T. Hothorn, «Bias in random forest variable importance measures», *BMC Bioinformatics* 8 (2007). T. Hastie, R. Tibshirani y J. Friedman, *The Elements of Statistical Learning*, 2.ª ed., cap. 8.7 y 15.

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u05", ids=["u05-p05", "u05-p06", "u05-p07", "u05-p08"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. **Correlación negativa.** En la proposición de la sección 2, ¿qué pasa si $\rho < 0$? Demuestra que, para $B$ variables con la misma correlación entre todos los pares, necesariamente $\rho \ge -1/(B-1)$ (pista: la varianza de la suma no puede ser negativa). ¿Qué varianza tiene el promedio en el caso extremo?
# 2. **Cuántos datos distintos.** Demuestra que el número esperado de datos **distintos** en una remuestra *bootstrap* de tamaño $N$ es $N\big[1 - (1 - 1/N)^N\big]$ (pista: suma de indicadoras). Calcula la varianza de esa cantidad y compárala con la de los 300 árboles de la sección 4.
# 3. **Votos o probabilidades.** Nuestro ensamble promedia probabilidades. Cambia `ensamble` para que cada árbol vote 0/1 (`predict` en lugar de `predict_proba`) con hojas de tamaño mínimo 10. ¿Cambian la exactitud y la AUC del ensamble? ¿Por qué con hojas de un dato las dos versiones son casi idénticas?
# 4. **MDI con hojas más grandes.** Repite la sección 7 con `min_samples_leaf=20`. ¿Qué le pasa a la importancia MDI de `id` y `ruido`? Explica por qué, con el argumento de los nodos profundos.
# 5. **El piso del bagging.** Con los números de la sección 6, ¿cuántos árboles hacen falta para que la varianza del *bagging* quede a menos del 5 % de su piso $\rho\sigma^2$? Verifícalo entrenando *bagging* de 10, 50, 200 y 500 árboles en las bicicletas y graficando la RECM de prueba contra $B$.
