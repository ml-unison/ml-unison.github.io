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
# # Un primer *pipeline* sin trampas: el Titanic
#
# **Unidad 1 · Ejemplo 3**
#
# En la libreta anterior preparamos datos con muchos problemas. Aquí juntamos todo en un flujo de trabajo completo y honesto:
#
# 1. plantear el problema y decidir **qué información es legítimo usar**,
# 2. separar los datos **antes** de cualquier otra cosa,
# 3. comparar contra una **línea base**,
# 4. construir un `Pipeline` de preprocesamiento y modelo,
# 5. detectar una **fuga de información** real,
# 6. y evaluar **una sola vez** en el conjunto de prueba.
#
# El modelo será una regresión logística, que por ahora usamos como una caja negra que clasifica; la estudiaremos en la Unidad 4. Lo importante de esta libreta es el **flujo**, no el modelo.

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

import rdp

rdp.estilo.usar()
SEMILLA = 1942

# %% [markdown]
# ## 1. El problema y los datos
#
# El *RMS Titanic* se hundió en abril de 1912 en su viaje inaugural, de Southampton a Nueva York. De sus pasajeros sobrevivió menos de la mitad. La tabla que usamos tiene a los 1,309 pasajeros (no incluye a la tripulación). La compiló Thomas Cason a partir de la *Encyclopedia Titanica*, la distribuyó Frank Harrell (Universidad de Vanderbilt) y hoy está en OpenML.
#
# **Problema.** Con lo que se sabía de un pasajero **al momento de abordar**, predecir si sobrevivió. No es un problema práctico (nadie va a usar este modelo), pero es perfecto para aprender, porque los datos traen una trampa.
#
# **Métrica.** La **exactitud**: la proporción de pasajeros bien clasificados.
#
# Antes de tocar los datos, revisemos qué dice cada columna y **cuándo se conocía**:
#
# | Columna | Significado | ¿Se conoce al abordar? |
# |---|---|---|
# | `pclass` | clase del boleto (1, 2, 3) | sí |
# | `name`, `sex`, `age` | nombre, sexo, edad | sí |
# | `sibsp`, `parch` | hermanos o cónyuge, padres o hijos a bordo | sí |
# | `ticket`, `fare`, `cabin`, `embarked` | boleto, tarifa, camarote, puerto de embarque | sí |
# | `home.dest` | lugar de origen o destino | sí |
# | `boat` | bote salvavidas en el que subió | **no**: se registró después |
# | `body` | número de cuerpo recuperado | **no**: se registró después |
# | `survived` | sobrevivió (1) o no (0) | es lo que queremos predecir |

# %%
df = rdp.datos.cargar("titanic")
df.head()

# %%
df.info()

# %% [markdown]
# ## 2. Separar primero
#
# Antes de mirar distribuciones, imputar o escalar, apartamos un **conjunto de prueba** (20 %). La separación es **estratificada** para conservar la proporción de sobrevivientes en ambos conjuntos.
#
# A partir de aquí, el conjunto de prueba queda cerrado hasta la sección 7.

# %%
X = df.drop(columns="survived")
y = df["survived"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEMILLA
)
print(f"Entrenamiento: {X_tr.shape[0]} pasajeros   ·   Prueba: {X_te.shape[0]} pasajeros")
print(f"Proporción de sobrevivientes (entrenamiento): {y_tr.mean():.3f}")

# %% [markdown]
# ## 3. Exploración (solo con entrenamiento)

# %%
faltantes = X_tr.isna().mean().sort_values(ascending=False)
ax = faltantes[faltantes > 0].plot.barh(color=rdp.estilo.AZUL)
ax.set(xlabel="proporción de valores faltantes", title="Valores faltantes por columna")
ax.invert_yaxis()
plt.show()

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
pd.crosstab(X_tr["sex"], y_tr, normalize="index")[1].plot.bar(ax=axs[0], color=rdp.estilo.AZUL, rot=0)
axs[0].set(title="Supervivencia por sexo", ylabel="proporción", xlabel="")
pd.crosstab(X_tr["pclass"], y_tr, normalize="index")[1].plot.bar(ax=axs[1], color=rdp.estilo.DORADO, rot=0)
axs[1].set(title="Supervivencia por clase", xlabel="clase")
axs[2].hist([X_tr.loc[y_tr == 0, "age"].dropna(), X_tr.loc[y_tr == 1, "age"].dropna()],
            bins=20, label=["no sobrevivió", "sobrevivió"], color=[rdp.estilo.GRIS, rdp.estilo.AZUL])
axs[2].set(title="Edad", xlabel="años")
axs[2].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Sexo y clase parecen muy informativos; la edad, algo menos. Las columnas `cabin`, `boat`, `body` y `home.dest` tienen muchos faltantes. Antes de decidir qué hacer con ellas hay que preguntarse **qué significan**.

# %% [markdown]
# ## 4. Línea base
#
# Todo modelo debe compararse contra una **línea base** trivial. Un clasificador que siempre predice la clase más frecuente («no sobrevivió») ya acierta en más de la mitad de los casos. Cualquier modelo que no supere ese número no aprendió nada útil.
#
# Para estimar la exactitud sin tocar la prueba usamos **validación cruzada**: dividimos el entrenamiento en 5 partes, entrenamos con 4 y medimos en la quinta, y repetimos rotando. La veremos con detalle en la Unidad 3.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

def evaluar(modelo, X, y, nombre):
    """Exactitud estimada con validación cruzada de 5 particiones (solo con entrenamiento)."""
    acc = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
    print(f"{nombre:<34} exactitud = {acc.mean():.3f} ± {acc.std():.3f}")
    return acc.mean()

evaluar(DummyClassifier(strategy="most_frequent"), X_tr, y_tr, "Línea base (clase mayoritaria)");

# %% [markdown]
# ## 5. Un `Pipeline` honesto
#
# Elegimos atributos que se conocen **al abordar** y los preprocesamos según su tipo, como en la libreta anterior:
#
# | Tipo | Columnas | Transformación |
# |---|---|---|
# | numéricas | `age`, `fare`, `sibsp`, `parch` | imputación con la mediana → estandarización |
# | categóricas | `sex`, `embarked`, `pclass` | imputación con la moda → *one-hot* |
#
# Todo vive dentro de un `Pipeline`. En cada pliegue de la validación cruzada, la mediana, la media, la desviación y las categorías **se estiman solo con los datos de entrenamiento de ese pliegue**.

# %%
numericas = ["age", "fare", "sibsp", "parch"]
categoricas = ["sex", "embarked", "pclass"]

preprocesamiento = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numericas),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), categoricas),
])

modelo = Pipeline([
    ("prep", preprocesamiento),
    ("clf", LogisticRegression(max_iter=1000)),
])
modelo

# %%
evaluar(modelo, X_tr, y_tr, "Regresión logística");

# %% [markdown]
# ## 6. Una fuga de información
#
# ¿Y si agregamos la columna `boat` (número del bote salvavidas) como otro atributo categórico?

# %%
con_fuga = Pipeline([
    ("prep", ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numericas),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="ninguno"),
                              OneHotEncoder(handle_unknown="ignore")), categoricas + ["boat"]),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
])
evaluar(con_fuga, X_tr, y_tr, "Regresión logística + 'boat'");

# %% [markdown]
# ¡La exactitud sube a más del 95 %! Pero el modelo es **inútil**: `boat` solo se conoce **después** del naufragio, y haber subido a un bote casi equivale a haber sobrevivido. Lo mismo ocurre con `body` (número de cuerpo recuperado). La tabla de la sección 1 ya nos lo advertía.
#
# Esta es una **fuga de la variable objetivo** (*target leakage*): un atributo que codifica la respuesta y que no estará disponible cuando el modelo se use. Ninguna técnica de validación la detecta. La única defensa es **entender el problema y los datos**.

# %% [markdown]
# ## 7. Evaluación final (una sola vez)
#
# Con el modelo honesto ya elegido, lo entrenamos con todo el conjunto de entrenamiento y lo evaluamos **una vez** en el de prueba.

# %%
modelo.fit(X_tr, y_tr)
acc_cv = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
acc_te = modelo.score(X_te, y_te)
print(f"Exactitud estimada con validación cruzada : {acc_cv:.3f}")
print(f"Exactitud en prueba (n = {len(y_te)})            : {acc_te:.3f}")

# %% [markdown]
# La exactitud en prueba es algo menor que la estimada. Con solo 262 pasajeros de prueba, una diferencia de unos cuantos puntos es normal: el conjunto de prueba también es una muestra, y su exactitud también varía. En la Unidad 3 aprenderemos a cuantificar esa incertidumbre.
#
# Lo que **no** debemos hacer ahora es volver a ajustar el modelo para mejorar este número. En el momento en que usamos la prueba para decidir, deja de ser una estimación honesta.

# %% [markdown]
# ## 8. Autoevaluación

# %%
rdp.quiz.mostrar("u01", ids=["u01-p07", "u01-p08"])

# %% [markdown]
# ## 9. Para explorar
#
# 1. Agrega un atributo derivado del nombre: el **título** (`Mr`, `Mrs`, `Miss`, `Master`, …). Hazlo **dentro** del `Pipeline` con un `FunctionTransformer`. ¿Mejora la validación cruzada? ¿Es una fuga?
# 2. Repite la sección 6 con la columna `body` en lugar de `boat`. ¿Por qué la exactitud no sube tanto, si también es una fuga? Pista: revisa cuántos valores tiene.
# 3. Calcula la exactitud en **entrenamiento** del modelo final y compárala con la de prueba. Relaciónala con $E_{\text{in}}$ y $E_{\text{out}}$ de la presentación.
