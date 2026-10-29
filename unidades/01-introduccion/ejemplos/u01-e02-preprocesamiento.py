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
# # Preprocesamiento: preparar datos reales para un modelo
#
# **Unidad 1 · Ejemplo 2**
#
# Los modelos de scikit-learn esperan una matriz de números, sin huecos y en escalas razonables. Los datos reales casi nunca llegan así. En esta libreta vamos a recorrer, uno por uno, los problemas más comunes y la transformación que corresponde a cada uno:
#
# | Problema | Transformación |
# |---|---|
# | Faltantes que significan «no tiene» | una categoría propia |
# | Faltantes de verdad | imputar, y marcar que se imputó |
# | Categorías sin orden | codificación *one-hot* |
# | Categorías con orden | codificación ordinal |
# | Variables muy sesgadas | logaritmo |
# | Fechas | convertirlas en cantidades con sentido (edades) |
# | Escalas muy distintas | estandarizar |
# | Valores atípicos | entender antes de quitar |
#
# Al final juntamos todo en un `Pipeline` y medimos cuánto ayudó. El modelo en sí (una regresión lineal regularizada) es solo el instrumento de medición; lo estudiaremos en la Unidad 4.

# %% [markdown]
# ## 0. Preparación del entorno

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.linear_model import RidgeCV

import rdp

rdp.estilo.usar()
SEMILLA = 2011

# %% [markdown]
# ## 1. El problema y los datos
#
# **Problema.** Estimar el precio de venta de una casa a partir de sus características. Es lo que hace un valuador, y lo que hacen los portales inmobiliarios cuando sugieren un precio.
#
# **Datos.** Ventas de casas en Ames, Iowa (EE. UU.) entre 2006 y 2010, reunidas por Dean De Cock a partir de los registros de la oficina de avalúos de la ciudad para usarlas en la enseñanza (*Journal of Statistics Education*, 2011). Usamos la versión de 1,460 casas y 79 atributos que distribuye OpenML. Los precios están en dólares y las áreas en pies cuadrados.
#
# Con 79 columnas es fácil perderse. Para esta libreta elegimos 20 que muestran todos los problemas que nos interesan.

# %%
casas = rdp.datos.cargar("casas_ames")
print(casas.shape)

numericas = ["GrLivArea", "LotArea", "LotFrontage", "TotalBsmtSF", "GarageArea",
             "OverallQual", "BedroomAbvGr", "FullBath"]
fechas = ["YearBuilt", "YearRemodAdd", "YrSold"]
nominales = ["Neighborhood", "BldgType", "CentralAir"]
ordinales = ["KitchenQual", "ExterQual", "BsmtQual", "FireplaceQu", "PoolQC"]
objetivo = "SalePrice"

datos = casas[numericas + fechas + nominales + ordinales + [objetivo]]
datos.head()

# %% [markdown]
# ### Primero se separa
#
# Como vimos en clase, la separación va **antes** de explorar: todo lo que decidamos mirando los datos (qué imputar, qué quitar, qué transformar) debe salir solo del conjunto de entrenamiento.

# %%
X = datos.drop(columns=objetivo)
y = datos[objetivo]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)
print(f"entrenamiento: {len(X_tr)} casas · prueba: {len(X_te)} casas")

# %% [markdown]
# ## 2. Faltantes: primero, preguntar por qué faltan

# %%
faltantes = X_tr.isna().mean().mul(100).round(1)
faltantes[faltantes > 0].sort_values(ascending=False)

# %% [markdown]
# Hay dos tipos muy distintos de faltante en esta tabla, y confundirlos es un error frecuente:
#
# - **`PoolQC`, `FireplaceQu`, `BsmtQual`**: la documentación de los datos dice que el faltante significa que la casa **no tiene** alberca, chimenea o sótano. No es un dato perdido: es información. Lo correcto es convertirlo en una categoría más, por ejemplo `"Ninguna"`.
# - **`LotFrontage`** (metros de frente del terreno): todas las casas tienen frente, así que aquí el dato **sí se perdió**. Lo imputamos con la mediana, y además agregamos una columna que diga si el valor fue imputado, por si el hecho de faltar dice algo.
#
# ¿La falta de chimenea importa para el precio? Veámoslo:

# %%
con_chimenea = X_tr["FireplaceQu"].notna()
y_tr.groupby(con_chimenea.map({True: "con chimenea", False: "sin chimenea"})).median()

# %% [markdown]
# Una diferencia de decenas de miles de dólares en la mediana. Si hubiéramos imputado el faltante con la categoría más frecuente, habríamos dicho que esas casas tienen chimenea y **borrado** esa información.

# %%
imputar_mediana = SimpleImputer(strategy="median", add_indicator=True)
imputar_mediana.fit(X_tr[["LotFrontage"]])
print("mediana aprendida en entrenamiento:", imputar_mediana.statistics_)
imputar_mediana.transform(X_te[["LotFrontage"]])[:5]

# %% [markdown]
# Observa la mecánica: `fit` **aprende** la mediana con el entrenamiento; `transform` la **aplica** a cualquier conjunto, incluido el de prueba. Todas las transformaciones de scikit-learn funcionan así.

# %% [markdown]
# ## 3. Categorías sin orden: *one-hot*
#
# `Neighborhood` (el barrio) es muy informativa, pero es texto y sus valores no tienen orden: no tiene sentido decir que un barrio es «mayor» que otro.

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
mediana_barrio = y_tr.groupby(X_tr["Neighborhood"]).median().sort_values() / 1000
ax.barh(mediana_barrio.index, mediana_barrio.values, color=rdp.estilo.AZUL)
ax.set(title="Precio mediano por barrio (entrenamiento)", xlabel="miles de dólares")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=9)
plt.show()

# %% [markdown]
# La codificación *one-hot* crea una columna binaria por categoría. Dos detalles prácticos:
#
# - `handle_unknown="ignore"`: si en prueba aparece un barrio que no estaba en entrenamiento, su fila queda en ceros en lugar de producir un error;
# - `min_frequency`: los barrios con muy pocas casas se agrupan en una categoría «poco frecuente», para no crear columnas casi vacías.

# %%
onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
onehot.fit(X_tr[["Neighborhood"]])
print(len(onehot.get_feature_names_out()), "columnas")
onehot.get_feature_names_out()[-5:]

# %% [markdown]
# ## 4. Categorías con orden: codificación ordinal
#
# Las columnas de calidad usan una escala: `Po` (mala), `Fa` (regular), `TA` (típica), `Gd` (buena), `Ex` (excelente). Aquí **sí** hay orden, y conviene conservarlo.

# %%
escala = ["Ninguna", "Po", "Fa", "TA", "Gd", "Ex"]
cocina = X_tr["KitchenQual"]
mediana_cocina = y_tr.groupby(cocina).median().reindex([e for e in escala if e in cocina.unique()]) / 1000

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(range(len(mediana_cocina)), mediana_cocina.values, marker="o")
ax.set(title="Precio mediano según la calidad de la cocina", ylabel="miles de dólares",
       xticks=range(len(mediana_cocina)), xticklabels=mediana_cocina.index)
plt.show()

# %% [markdown]
# El precio sube siempre con la calidad. Una codificación ordinal (0, 1, 2, …) captura esa relación con **una sola columna**; *one-hot* necesitaría cinco y el modelo tendría que descubrir el orden por su cuenta.
#
# Le damos el orden explícitamente. Si no, `OrdinalEncoder` usa el orden alfabético (`Ex` < `Fa` < `Gd` …), que no significa nada.

# %%
calidad = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="Ninguna"),
    OrdinalEncoder(categories=[escala] * len(ordinales)),
)
calidad.fit(X_tr[ordinales])
pd.DataFrame(calidad.transform(X_tr[ordinales].head()), columns=ordinales)

# %% [markdown]
# ## 5. Variables muy sesgadas: el logaritmo
#
# Los precios y las áreas suelen tener una cola larga a la derecha: muchas casas modestas y unas pocas muy caras. Para un modelo lineal eso es un problema, porque los errores en las casas caras dominan todo.

# %%
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
axs[0].hist(y_tr / 1000, bins=40, color=rdp.estilo.AZUL)
axs[0].set(title="Precio", xlabel="miles de dólares")
axs[1].hist(np.log10(y_tr), bins=40, color=rdp.estilo.AZUL)
axs[1].set(title="Logaritmo del precio", xlabel="log₁₀(dólares)")
plt.show()

# %% [markdown]
# Después del logaritmo la distribución es casi simétrica. Hay además una razón de fondo: en precios, lo natural es pensar en errores **relativos**. Equivocarse por 20 mil dólares es grave en una casa de 100 mil y poco importante en una de 500 mil. Un modelo que predice el logaritmo comete errores relativos parecidos en todo el rango.
#
# Para la variable objetivo, scikit-learn tiene `TransformedTargetRegressor`: entrena con `log(y)` y regresa las predicciones a dólares automáticamente. Para atributos como `LotArea` usamos `FunctionTransformer(np.log1p)`.

# %% [markdown]
# ## 6. Fechas: convertirlas en algo con sentido
#
# `YearBuilt = 1958` no es una cantidad que tenga sentido sumar o multiplicar. Lo que importa para el precio es la **edad** de la casa al venderse, y los años desde la última remodelación.

# %%
def edades(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "edad": t["YrSold"] - t["YearBuilt"],
        "desde_remodelacion": t["YrSold"] - t["YearRemodAdd"],
    })

edades(X_tr).describe().round(1)

# %% [markdown]
# Un detalle: el mínimo de `desde_remodelacion` es −1, una casa remodelada **después** de venderse. Es un buen ejemplo de por qué conviene mirar los resúmenes de cada columna que creamos.
#
# Esta transformación no aprende nada de los datos (es una resta), así que no hay riesgo de fuga. Aun así la metemos al `Pipeline`, para que se aplique igual a cualquier casa nueva.

# %% [markdown]
# ## 7. Escalas distintas: estandarizar
#
# Las áreas van de cientos a decenas de miles; el número de baños, de 0 a 3. Muchos modelos (los regularizados, los basados en distancias, las redes neuronales) son sensibles a esas diferencias. Estandarizar lleva cada columna a media 0 y desviación 1:
#
# $$
# z = \frac{x - \bar{x}_{\text{entrenamiento}}}{s_{\text{entrenamiento}}}
# $$
#
# Fíjate en el subíndice: la media y la desviación salen del **entrenamiento** y se aplican sin cambio a la prueba.

# %%
escalador = StandardScaler().fit(X_tr[["GrLivArea", "FullBath"]])
pd.DataFrame(escalador.transform(X_te[["GrLivArea", "FullBath"]]),
             columns=["GrLivArea", "FullBath"]).describe().round(2)

# %% [markdown]
# En el conjunto de prueba la media no es exactamente 0 ni la desviación exactamente 1, y **así debe ser**: si lo fueran, querría decir que usamos información de la prueba.

# %% [markdown]
# ## 8. Valores atípicos: entender antes de quitar

# %%
fig, ax = plt.subplots(figsize=(7, 4))
grandes = X_tr["GrLivArea"] > 4000
ax.scatter(X_tr["GrLivArea"], y_tr / 1000, s=14, alpha=0.6, color=rdp.estilo.AZUL, edgecolors="none")
ax.scatter(X_tr.loc[grandes, "GrLivArea"], y_tr[grandes] / 1000, s=60, color=rdp.estilo.ROJO, edgecolors="white")
ax.set(title="Precio contra área habitable", xlabel="área habitable (pies²)", ylabel="miles de dólares")
plt.show()
X_tr.loc[grandes, ["GrLivArea", "OverallQual", "Neighborhood"]].assign(precio=y_tr[grandes])

# %% [markdown]
# Las cuatro casas de más de 4,000 pies² no se comportan igual. Las dos del barrio Edwards son enormes y **baratas**: De Cock explica en la documentación que son ventas **parciales** (casas sin terminar) que no representan el mercado. Las dos de NoRidge son simplemente casas grandes y caras, con un precio razonable. Él mismo recomienda excluir todas las casas de más de 4,000 pies² al enseñar con estos datos, porque son muy pocas para aprender algo de ellas.
#
# Tres reglas para los atípicos:
#
# 1. Primero entender de dónde salen. Un atípico puede ser un error, un caso especial o la información más importante de la tabla.
# 2. Si se quitan, se quitan **solo del entrenamiento**, con una razón que no dependa de «empeoran mi métrica».
# 3. Documentar la decisión.

# %%
mantener = X_tr["GrLivArea"] <= 4000
X_tr, y_tr = X_tr[mantener], y_tr[mantener]
print(f"quitamos {(~mantener).sum()} casas del entrenamiento")

# %% [markdown]
# ## 9. Todo junto en un `Pipeline`
#
# Cada grupo de columnas tiene su propio tratamiento. `ColumnTransformer` aplica a cada grupo el suyo y junta los resultados en una sola matriz.

# %%
sesgadas = ["GrLivArea", "LotArea"]
otras_numericas = [c for c in numericas if c not in sesgadas]

preprocesamiento = ColumnTransformer([
    ("sesgadas", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), sesgadas),
    ("numericas", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), otras_numericas),
    ("edades", make_pipeline(FunctionTransformer(edades), StandardScaler()), fechas),
    ("nominales", OneHotEncoder(handle_unknown="ignore", min_frequency=10), nominales),
    ("ordinales", make_pipeline(calidad, StandardScaler()), ordinales),
])

modelo = TransformedTargetRegressor(
    regressor=make_pipeline(preprocesamiento, RidgeCV(alphas=np.logspace(-2, 3, 20))),
    func=np.log, inverse_func=np.exp,
)
modelo

# %% [markdown]
# Para medir cuánto ayudó el preprocesamiento, lo comparamos con la versión «ingenua» que muchos escriben primero: tomar solo las columnas numéricas, rellenar huecos con la media y entrenar sobre el precio en dólares.

# %%
ingenuo = make_pipeline(
    ColumnTransformer([("num", SimpleImputer(strategy="mean"), numericas + fechas)]),
    RidgeCV(alphas=np.logspace(-2, 3, 20)),
)

cv = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
for nombre, m in [("ingenuo", ingenuo), ("con preprocesamiento", modelo)]:
    error = -cross_val_score(m, X_tr, y_tr, cv=cv, scoring="neg_median_absolute_error")
    print(f"{nombre:<22} error mediano = {error.mean():>8,.0f} dólares  (± {error.std():,.0f})")

# %% [markdown]
# Con el mismo modelo, preparar bien los datos reduce el error de forma importante. Y todo vive dentro de un solo objeto: al hacer `modelo.fit(X_tr, y_tr)`, cada mediana, cada vocabulario y cada media se aprende **solo** con el entrenamiento, también dentro de cada partición de la validación cruzada.
#
# No tocamos el conjunto de prueba: lo reservamos para la evaluación final. Eso lo haremos en la libreta 3.

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u01", ids=["u01-p14", "u01-p15", "u01-p16", "u01-p07"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. Agrega al modelo la columna `GarageYrBlt` (año de construcción de la cochera). Tiene faltantes. ¿Qué significan? ¿Cómo los tratarías?
# 2. Cambia `ExterQual` y `KitchenQual` a codificación *one-hot* en lugar de ordinal. ¿Cambia el error? ¿Cuántas columnas tiene ahora la matriz?
# 3. Quita el logaritmo de la variable objetivo (usa directamente `RidgeCV` sin `TransformedTargetRegressor`). ¿Qué pasa con el error mediano? ¿Y con el error en las casas más caras?
# 4. Vuelve a incluir las casas de más de 4,000 pies² en el entrenamiento. ¿Qué tanto cambia el error? ¿Por qué crees que una sola casa puede mover tanto un modelo lineal?
