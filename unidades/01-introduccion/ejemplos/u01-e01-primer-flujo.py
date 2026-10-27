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
# # Un primer flujo de trabajo completo con scikit-learn
#
# **Reconocimiento de Patrones · Unidad 1 · Ejemplo 1**
# Licenciatura en Ciencias de la Computación · Universidad de Sonora
#
# En esta libreta recorremos, de principio a fin, el ciclo de vida de un modelo de clasificación:
#
# 1. definir el problema y la métrica,
# 2. separar los datos **antes** de hacer cualquier otra cosa,
# 3. explorar solo el conjunto de entrenamiento,
# 4. construir un `Pipeline` de preprocesamiento y modelo,
# 5. compararlo contra una línea base con validación cruzada,
# 6. detectar una **fuga de información**,
# 7. y evaluar **una sola vez** en el conjunto de prueba.
#
# Los datos son los registros de pasajeros del *RMS Titanic* (1912), una base clásica que tiene variables numéricas, categóricas y faltantes, y una trampa interesante.

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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import rdp

rdp.estilo.usar()
SEMILLA = 1942

# %% [markdown]
# ## 1. El problema
#
# **Tarea.** Dadas las características de un pasajero conocidas *al momento de abordar*, predecir si sobrevivió al naufragio.
#
# **Métrica.** Usaremos la **exactitud** (*accuracy*) por simplicidad y, como las clases están desbalanceadas, también la **exactitud balanceada**, que promedia la exhaustividad (*recall*) de cada clase:
# $$
# \text{BA} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right).
# $$
#
# Observa la frase *al momento de abordar*: define qué información es legítimo usar. Volveremos a ella.

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
# Todo modelo debe compararse contra una **línea base** trivial. Un clasificador que siempre predice la clase mayoritaria ya obtiene una exactitud nada despreciable:

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

def evaluar(modelo, X, y, nombre):
    acc = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
    ba = cross_val_score(modelo, X, y, cv=cv, scoring="balanced_accuracy")
    print(f"{nombre:<38} exactitud = {acc.mean():.3f} ± {acc.std():.3f}   "
          f"balanceada = {ba.mean():.3f} ± {ba.std():.3f}")
    return acc.mean()

evaluar(DummyClassifier(strategy="most_frequent"), X_tr, y_tr, "Línea base (clase mayoritaria)");

# %% [markdown]
# ## 5. Un `Pipeline` honesto
#
# Elegimos atributos que se conocen **al abordar** y los preprocesamos según su tipo:
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
# ¡La exactitud sube a más del 95 %! Pero el modelo es **inútil**: `boat` solo se conoce **después** del naufragio. Tener un bote asignado casi equivale a haber sobrevivido. Lo mismo ocurre con `body` (número de cuerpo recuperado).
#
# Esta es una **fuga de la variable objetivo** (*target leakage*): un atributo que codifica la respuesta y que no estará disponible cuando el modelo se use. Ninguna técnica de validación la detecta. La única defensa es **entender el problema y los datos**.

# %% [markdown]
# ## 7. Evaluación final (una sola vez)
#
# Con el modelo honesto ya elegido, lo entrenamos con todo el conjunto de entrenamiento y lo evaluamos **una vez** en el de prueba.

# %%
modelo.fit(X_tr, y_tr)
y_pred = modelo.predict(X_te)
print(classification_report(y_te, y_pred, target_names=["no sobrevivió", "sobrevivió"], digits=3))

# %%
ConfusionMatrixDisplay.from_predictions(
    y_te, y_pred, display_labels=["no sobrevivió", "sobrevivió"], cmap="rdp_azul", colorbar=False
)
plt.title("Matriz de confusión (prueba)")
plt.grid(False)
plt.show()

# %% [markdown]
# ¿Qué tanto debería parecerse la exactitud en prueba a la estimada con validación cruzada? El conjunto de prueba es **finito**, así que su exactitud también es una variable aleatoria. Con la desigualdad de Hoeffding de la presentación, para una hipótesis **fija** (no elegimos el modelo mirando la prueba), con probabilidad al menos $1-\delta$:
# $$
# |E_{\text{in}} - E_{\text{out}}| \le \epsilon = \sqrt{\frac{\ln(2/\delta)}{2n}}.
# $$

# %%
acc_cv = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
acc_te = (y_pred == y_te).mean()
n, delta = len(y_te), 0.05
eps = np.sqrt(np.log(2 / delta) / (2 * n))
print(f"Exactitud estimada por validación cruzada : {acc_cv:.3f}")
print(f"Exactitud en prueba (n = {n})             : {acc_te:.3f}")
print(f"Intervalo de Hoeffding al 95 %             : [{acc_te - eps:.3f}, {acc_te + eps:.3f}]")

# %% [markdown]
# Con solo 262 ejemplos de prueba, el intervalo tiene un ancho de $\pm 0.08$. Ese es el **precio de evaluar con pocos datos**. La diferencia entre la validación cruzada y la prueba es compatible con la variabilidad del muestreo. En la Unidad 3 veremos estimaciones más finas, como intervalos binomiales y *bootstrap*.

# %% [markdown]
# ## 8. Autoevaluación

# %%
rdp.quiz.mostrar("u01", ids=["u01-p07", "u01-p08"])

# %% [markdown]
# ## 9. Para explorar
#
# 1. Agrega un atributo derivado del nombre: el **título** (`Mr`, `Mrs`, `Miss`, `Master`, …). ¿Mejora la validación cruzada? ¿Es una fuga?
# 2. Sustituye `LogisticRegression` por `DecisionTreeClassifier(max_depth=d)` y grafica la exactitud de validación contra `d`. ¿Qué observas para valores grandes de `d`?
# 3. Calcula la exactitud en **entrenamiento** del modelo final y compárala con la de prueba. Relaciónala con $E_{\text{in}}$ y $E_{\text{out}}$.
