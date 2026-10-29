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
# # Seguimiento de experimentos con MLflow
#
# **Unidad 8 · Ejemplo 1**
#
# En la Unidad 3 comparamos una regresión logística con un bosque aleatorio para estimar el riesgo de que un cliente de tarjeta de crédito no pague. Lo hicimos en una libreta: entrenamos, miramos los números y seguimos adelante. Seis meses después, alguien pregunta: «¿con qué hiperparámetros se entrenó el modelo que está en producción?, ¿con qué datos?, ¿por qué se eligió ése y no otro?». Si la respuesta está solo en la memoria de quien ejecutó la libreta (o en celdas que se ejecutaron en desorden y luego se borraron), no hay respuesta.
#
# En esta libreta vamos a:
#
# 1. ver con datos que el mismo código puede dar **números distintos**, y enumerar qué hay que guardar de cada **corrida** para poder repetirla;
# 2. registrar con **MLflow** una pequeña búsqueda de modelos (logística, bosque aleatorio y *boosting* con distintos hiperparámetros), evaluados con validación cruzada, y **compararlos** con una consulta;
# 3. guardar el mejor en el **registro de modelos**, con su **firma** de entrada y salida, como una **versión** con **alias** (`champion` y `challenger`), y cargarlo por su alias para predecir;
# 4. **reproducir** una corrida a partir de lo que quedó guardado;
# 5. ver por qué cargar un modelo con `pickle` equivale a ejecutar código ajeno, y qué hace `skops` al respecto;
# 6. escribir una **ficha del modelo** (*model card*) para el modelo elegido.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios) y MLflow. En tu máquina, si usas el entorno del repositorio, ya están instalados.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete" mlflow
    pass

# %%
import contextlib, hashlib, io, logging, pickle, platform, re, subprocess, tempfile, time, warnings
from pathlib import Path

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"                 # MLflow imprime sugerencias al importarse
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"  # y barras de progreso al descargar archivos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")  # nuestros datos no tienen faltantes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import lightgbm as lgb
import mlflow
from mlflow.models import infer_signature
import skops.io as sio
from IPython.display import Markdown

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss

import rdp

logging.getLogger("mlflow").setLevel(logging.ERROR)  # solo errores de MLflow (después de importarlo)
rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Datos.** Los mismos de la libreta de métricas de la Unidad 3: 30,000 clientes de tarjeta de crédito de un banco de Taiwán, de abril a septiembre de 2005, reunidos por I-Cheng Yeh y Che-hui Lien (*Expert Systems with Applications*, 2009; UCI, licencia CC BY 4.0). Por cliente hay límite de crédito, sexo, escolaridad, estado civil, edad y seis meses de estados de pago (`atraso_*`), facturas (`factura_*`) y pagos (`pago_*`). La respuesta `incumple` vale 1 si el cliente no pagó en octubre de 2005 (22 % de los clientes).
#
# **Problema.** El banco quiere un **puntaje de riesgo** para decidir a quién contactar o reducirle el límite antes de que caiga en mora. En la Unidad 3 la pregunta era *cómo medir* un clasificador. Ahora la pregunta es de ingeniería: el banco va a probar varias familias de modelos y varias configuraciones de cada una; necesita **saber qué se probó, con qué resultado, cuál se eligió y por qué**, poder **repetir** cualquiera de esos entrenamientos y **entregar** el elegido a otro sistema que lo use para predecir.
#
# Usamos exactamente la misma separación de la Unidad 3: 22,500 clientes para elegir el modelo (con validación cruzada) y 7,500 de prueba que no se tocan hasta el final.

# %%
credito = rdp.datos.cargar("credito_taiwan")
categoricas = ["sexo", "escolaridad", "estado_civil"]
X = credito.drop(columns="incumple")
y = credito["incumple"]
numericas = [c for c in X.columns if c not in categoricas]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
print(f"entrenamiento: {len(X_tr):,} clientes; prueba: {len(X_te):,}; prevalencia {y.mean():.1%}")

# %% [markdown]
# ## 2. El mismo código, números distintos
#
# Antes de hablar de herramientas, veamos el problema. Entrenemos el **mismo** bosque aleatorio (200 árboles, al menos 50 clientes por hoja) con diez semillas distintas, y evaluemos con el área bajo la curva ROC (AUC) en una misma partición de validación, separada del conjunto de entrenamiento. Luego hagamos lo contrario: dejemos fijo el modelo (una regresión logística, que no tiene nada aleatorio) y cambiemos la semilla con la que la validación cruzada reparte a los clientes en cinco pliegues.

# %%
def preprocesamiento():
    """One-hot para las categóricas y estandarización para las numéricas (lo necesita la logística)."""
    return ColumnTransformer([("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
                              ("numericas", StandardScaler(), numericas)])


X_a, X_v, y_a, y_v = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEMILLA)
auc_semilla_bosque = []
for s in range(10):
    b = RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=s).fit(X_a, y_a)
    auc_semilla_bosque.append(roc_auc_score(y_v, b.predict_proba(X_v)[:, 1]))

logistica_fija = make_pipeline(preprocesamiento(), LogisticRegression(max_iter=2000))
auc_semilla_cv = []
for s in range(10):
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
    auc_semilla_cv.append(cross_validate(logistica_fija, X_tr, y_tr, cv=cv_s, scoring="roc_auc")["test_score"].mean())

auc_semilla_bosque, auc_semilla_cv = np.array(auc_semilla_bosque), np.array(auc_semilla_cv)
for nombre, a in [("bosque, semilla del modelo", auc_semilla_bosque), ("logística, semilla de los pliegues", auc_semilla_cv)]:
    print(f"{nombre:36s} AUC de {a.min():.4f} a {a.max():.4f} (rango {a.max() - a.min():.4f})")

# %%
fig, ax = plt.subplots(figsize=(10, 3.8))
for k, (a, color, texto) in enumerate([(auc_semilla_bosque, ROJO, "bosque aleatorio:\ncambia la semilla del modelo"),
                                       (auc_semilla_cv, AZUL, "regresión logística:\ncambia la semilla de los pliegues")]):
    desvio = 1000 * (a - a.mean())   # en milésimas de AUC
    ax.scatter(desvio, np.full_like(a, k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(a)), s=80, color=color,
               alpha=0.85, zorder=3)
    ax.text(-1.15, k, texto, ha="right", va="center", color=TINTA, fontsize=11)
    print(f"{texto.splitlines()[0]:20s} media {a.mean():.4f}")
ax.axvline(0, color=TENUE, lw=1)
ax.set_yticks([])
ax.set_ylim(-0.6, 1.6)
ax.set_xlim(-2.6, 1.2)
ax.set_xlabel("diferencia con el promedio de las diez corridas (milésimas de AUC)")
ax.set_title("El mismo código, diez semillas, diez números")
ax.grid(axis="y", visible=False)
plt.show()

# %% [markdown]
# Las diferencias son pequeñas (del orden de milésimas), pero son del mismo tamaño que las que vamos a usar para elegir entre modelos. Si hoy reportamos «AUC de 0.7805» y mañana alguien obtiene 0.7790 con el «mismo» modelo, no hay forma de saber si cambió algo importante o solo la semilla. Fijada la semilla, en cambio, el resultado es idéntico bit por bit:

# %%
def auc_bosque(semilla):
    b = RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=semilla).fit(X_a, y_a)
    return roc_auc_score(y_v, b.predict_proba(X_v)[:, 1])

print("¿dos corridas con semilla 3 dan exactamente lo mismo?", auc_bosque(3) == auc_bosque(3))

# %% [markdown]
# ### Las fuentes de no reproducibilidad
#
# La semilla es solo una. Las que más se encuentran en la práctica:
#
# | Fuente | Ejemplo | Remedio |
# |---|---|---|
# | Azar | inicialización, muestras *bootstrap*, partición en pliegues, orden de los minilotes | fijar y **guardar** todas las semillas |
# | Versiones de bibliotecas | un valor por omisión que cambia entre versiones (en scikit-learn ha pasado, por ejemplo, con el *solver* de `LogisticRegression`) | registrar el entorno; fijar versiones (`uv.lock`, `requirements.txt`) |
# | Datos que cambian | la tabla se actualiza o alguien corrige un error en el CSV | guardar una **huella** (*hash*) de los datos, o una versión fechada |
# | Cómputo en paralelo y GPU | las sumas en punto flotante dependen del orden; en GPU y con varios hilos el orden puede variar | opciones deterministas (`deterministic=True` en LightGBM, `torch.use_deterministic_algorithms`) a cambio de velocidad |
# | La libreta misma | celdas ejecutadas fuera de orden, una variable que sobrevive a una celda borrada | antes de reportar, **reiniciar y ejecutar todo**; mejor aún, convertir el entrenamiento en un guion |
#
# ### Qué guardar de cada corrida
#
# Una **corrida** (*run*) es una ejecución del entrenamiento. Para poder repetirla y compararla con otras hay que guardar:
#
# 1. el **código**: idealmente el *commit* de git;
# 2. los **datos**: una huella que cambie si cambia un solo valor;
# 3. el **entorno**: versiones de Python y de las bibliotecas;
# 4. los **parámetros**: hiperparámetros, semillas, la partición de validación;
# 5. las **métricas**: con su desviación entre pliegues, no solo el promedio;
# 6. los **artefactos**: el modelo entrenado, gráficas, la configuración completa.
#
# Las tres primeras se pueden calcular una vez:

# %%
def huella_datos(df):
    """SHA-256 del contenido de la tabla (valores e índice): cambia si cambia un solo dato."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()[:16]


def commit_git():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "sin-git"   # en Colab la libreta no está dentro de un repositorio


ENTORNO = {
    "codigo.commit": commit_git(),
    "codigo.rama_curso": RAMA,
    "datos.nombre": "credito_taiwan",
    "datos.huella": huella_datos(credito),
    "entorno.python": platform.python_version(),
    "entorno.numpy": np.__version__,
    "entorno.pandas": pd.__version__,
    "entorno.sklearn": sklearn.__version__,
    "entorno.lightgbm": lgb.__version__,
    "entorno.mlflow": mlflow.__version__,
}
ENTORNO

# %% [markdown]
# La huella es la de **todos** los 30,000 clientes; la partición en entrenamiento y prueba queda determinada por ella y por la semilla 2011. Si alguien corrige un solo valor del CSV, la huella cambia:

# %%
copia = credito.copy()
copia.loc[0, "limite"] += 1
print(huella_datos(credito), "→", huella_datos(copia))

# %% [markdown]
# ## 3. Seguimiento de experimentos con MLflow
#
# Podríamos guardar todo eso en archivos JSON a mano. **MLflow** es una biblioteca de código abierto que lo organiza: un **experimento** agrupa **corridas**; cada corrida tiene **parámetros** (texto: lo que se decidió antes de entrenar), **métricas** (números, opcionalmente con un paso: lo que se midió), **etiquetas** (*tags*, texto libre) y **artefactos** (archivos: modelos, gráficas, tablas). Todo se guarda en un **servidor de seguimiento** (*tracking server*), que puede ser una base de datos local o un servicio compartido por todo un equipo.
#
# Aquí usamos una base de datos SQLite en un directorio temporal, para que la libreta no dependa de ningún servicio externo. En un equipo real se apuntaría a un servidor común con `mlflow.set_tracking_uri("http://…")` y todos verían las mismas corridas; en tu máquina, `mlflow ui` levanta una interfaz web para explorarlas.

# %%
DIR_MLFLOW = Path(tempfile.mkdtemp(prefix="mlflow-"))
mlflow.set_tracking_uri(f"sqlite:///{DIR_MLFLOW / 'seguimiento.db'}")
id_experimento = mlflow.create_experiment("riesgo-credito", artifact_location=(DIR_MLFLOW / "artefactos").as_uri())
mlflow.set_experiment(experiment_id=id_experimento)
print("las corridas se guardan en", DIR_MLFLOW)

# %% [markdown]
# ### Las configuraciones
#
# Probaremos siete configuraciones de tres familias. La logística necesita el preprocesamiento (*one-hot* y estandarización); los árboles reciben los códigos tal cual. El *boosting* es LightGBM, que veremos a fondo en la Unidad 5; aquí basta saber que construye árboles uno tras otro, cada uno corrigiendo los errores de los anteriores, con una tasa de aprendizaje que controla el tamaño de cada corrección. Le pedimos `deterministic=True` y un número fijo de hilos para que el resultado no dependa de la máquina.
#
# La clave es que **la configuración es un diccionario**: todo lo que hace falta para reconstruir el modelo está en él, y eso es lo que guardaremos.

# %%
CONFIGURACIONES = [
    {"nombre": "logística C=1",            "familia": "logistica", "C": 1.0},
    {"nombre": "logística C=0.01",         "familia": "logistica", "C": 0.01},
    {"nombre": "bosque hoja≥50",           "familia": "bosque",    "n_estimators": 200, "min_samples_leaf": 50},
    {"nombre": "bosque hoja≥5",            "familia": "bosque",    "n_estimators": 200, "min_samples_leaf": 5},
    {"nombre": "boosting η=0.05, 15 hojas", "familia": "boosting",  "learning_rate": 0.05, "num_leaves": 15, "n_estimators": 300},
    {"nombre": "boosting η=0.1, 31 hojas",  "familia": "boosting",  "learning_rate": 0.1,  "num_leaves": 31, "n_estimators": 100},
    {"nombre": "boosting η=0.3, 63 hojas",  "familia": "boosting",  "learning_rate": 0.3,  "num_leaves": 63, "n_estimators": 300},
]


def construir(config):
    """Reconstruye el modelo a partir de su configuración (sin nombre ni familia)."""
    p = {k: v for k, v in config.items() if k not in ("nombre", "familia")}
    if config["familia"] == "logistica":
        return make_pipeline(preprocesamiento(), LogisticRegression(max_iter=2000, **p))
    if config["familia"] == "bosque":
        return RandomForestClassifier(n_jobs=-1, random_state=SEMILLA, **p)
    if config["familia"] == "boosting":
        return lgb.LGBMClassifier(random_state=SEMILLA, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1, **p)
    raise ValueError(config["familia"])

# %% [markdown]
# ### Una función que entrena, evalúa y registra
#
# Cada configuración se evalúa con validación cruzada estratificada de cinco pliegues, siempre con la misma partición (semilla 2011), con dos métricas: el AUC (¿ordena bien?) y la *log-loss* o entropía cruzada (¿sus probabilidades son buenas?). La función registra en MLflow:
#
# - parámetros: la configuración y la partición de validación;
# - etiquetas: código, datos y entorno;
# - métricas: media y desviación estándar de cada métrica, y el AUC de cada pliegue como una serie con `step`;
# - un artefacto: la configuración completa en JSON (los parámetros de MLflow se guardan **como texto**; el JSON conserva los tipos).

# %%
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)


def correr(config):
    inicio = time.perf_counter()
    r = cross_validate(construir(config), X_tr, y_tr, cv=CV, scoring=["roc_auc", "neg_log_loss"])
    with mlflow.start_run(run_name=config["nombre"]) as corrida:
        mlflow.set_tags(ENTORNO)
        mlflow.log_params({k: v for k, v in config.items() if k != "nombre"})
        mlflow.log_params({"cv.pliegues": 5, "cv.semilla": SEMILLA, "semilla": SEMILLA})
        mlflow.log_metrics({
            "auc_media": r["test_roc_auc"].mean(), "auc_de": r["test_roc_auc"].std(),
            "logloss_media": -r["test_neg_log_loss"].mean(), "logloss_de": r["test_neg_log_loss"].std(),
            "segundos": time.perf_counter() - inicio,
        })
        for paso, auc in enumerate(r["test_roc_auc"]):
            mlflow.log_metric("auc_pliegue", auc, step=paso)
        mlflow.log_dict(config, "configuracion.json")
    return corrida.info.run_id


ids_corridas = {c["nombre"]: correr(c) for c in CONFIGURACIONES}

# %% [markdown]
# ### Comparar corridas
#
# Las corridas ya no viven en variables de la libreta sino en la base de datos. `mlflow.search_runs` las devuelve como un `DataFrame` de pandas, con filtros y orden escritos en un pequeño lenguaje parecido a SQL. Cualquier persona con acceso al servidor puede hacer la misma consulta.

# %%
corridas = mlflow.search_runs(experiment_ids=[id_experimento], order_by=["metrics.auc_media DESC"])
tabla = corridas[["tags.mlflow.runName", "params.familia", "metrics.auc_media", "metrics.auc_de",
                  "metrics.logloss_media", "metrics.segundos"]]
tabla.columns = ["corrida", "familia", "AUC (media)", "AUC (desv. est.)", "log-loss (media)", "segundos"]
tabla.round(4)

# %% [markdown]
# Los filtros permiten preguntas como «las corridas de *boosting* con AUC mayor que 0.775»:

# %%
mlflow.search_runs(experiment_ids=[id_experimento],
                   filter_string="params.familia = 'boosting' and metrics.auc_media > 0.775")[["tags.mlflow.runName", "metrics.auc_media"]]

# %%
orden = tabla.sort_values("AUC (media)")
color_familia = {"logistica": AZUL, "bosque": ROJO, "boosting": DORADO_OSC}
colores = orden["familia"].map(color_familia)
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, col, de, titulo in [(ejes[0], "AUC (media)", "AUC (desv. est.)", "AUC (más es mejor)"),
                            (ejes[1], "log-loss (media)", None, "log-loss (menos es mejor)")]:
    pos = np.arange(len(orden))
    if de:
        ax.errorbar(orden[col], pos, xerr=orden[de], fmt="none", ecolor=TENUE, elinewidth=1.5, capsize=4, zorder=2)
    ax.scatter(orden[col], pos, s=90, color=colores, zorder=3)
    ax.set_title(titulo)
    ax.grid(axis="y", visible=False)
ejes[0].set_yticks(np.arange(len(orden)), orden["corrida"])
ejes[0].set_xlabel("validación cruzada (media ± 1 desv. est. entre pliegues)")
ejes[1].set_xlabel("validación cruzada (media)")
fig.suptitle("Siete corridas, una consulta", x=0.01, ha="left", fontsize=14, color=rdp.estilo.MARINO, fontweight="semibold")
fig.tight_layout()
plt.show()

# %% [markdown]
# Tres observaciones:
#
# - **Las dos familias de árboles le ganan a la logística** por unos seis puntos de AUC. La logística trata los códigos de atraso (−2, −1, 0, 1, …, 9) como números en una escala; los árboles pueden cortar en «pagó a tiempo» contra «se atrasó», que es lo que importa.
# - **Los primeros tres lugares están dentro del ruido.** La diferencia entre el bosque con hojas de 50 y el *boosting* lento es de dos milésimas, menor que la desviación entre pliegues. Con estos datos no podemos decir que uno sea mejor; elegimos el primero, pero sabiendo que la elección es casi un empate.
# - **La *log-loss* ve cosas que el AUC no ve.** El *boosting* agresivo (η = 0.3, 63 hojas, 300 árboles) pierde poco AUC pero su *log-loss* se dispara: está sobreajustado y da probabilidades demasiado extremas, que se pagan muy caras cuando se equivoca. Si solo hubiéramos registrado el AUC, no lo sabríamos.
#
# ### *Autolog*
#
# Para muchas bibliotecas (scikit-learn, LightGBM, XGBoost, PyTorch…) MLflow puede registrar todo automáticamente: `mlflow.sklearn.autolog()` intercepta cada `fit` y guarda **todos** los hiperparámetros del estimador (también los que dejamos por omisión), métricas y, si se pide, el modelo. Es cómodo, pero conviene saber qué guarda:

# %%
mlflow.sklearn.autolog(log_models=False, log_datasets=False, silent=True)
with mlflow.start_run(run_name="autolog: logística") as corrida_auto:
    construir(CONFIGURACIONES[0]).fit(X_tr, y_tr)
mlflow.sklearn.autolog(disable=True)

registrada = mlflow.get_run(corrida_auto.info.run_id).data
print(f"{len(registrada.params)} parámetros registrados automáticamente, por ejemplo:")
print({k: registrada.params[k] for k in ["logisticregression__C", "logisticregression__solver", "logisticregression__max_iter"]})
print("métricas:", {k: round(v, 4) for k, v in registrada.metrics.items()})

# %% [markdown]
# Registró decenas de parámetros, incluido el *solver*, que no elegimos nosotros: si una versión futura de scikit-learn cambiara su valor por omisión, la corrida lo diría. Pero las métricas que guarda son **de entrenamiento** (`training_roc_auc`, `training_log_loss`…), calculadas con los mismos datos con que se ajustó el modelo: no sustituyen a la validación cruzada. *Autolog* sirve para no olvidar nada; la evaluación honesta la seguimos diseñando nosotros.

# %% [markdown]
# ## 4. El registro de modelos
#
# El seguimiento responde «¿qué se probó?». El **registro de modelos** (*model registry*) responde «¿qué modelo está en uso, y cuál lo va a reemplazar?». Un **modelo registrado** tiene un nombre (`riesgo-credito`) y una sucesión de **versiones** numeradas (1, 2, 3…), cada una ligada a la corrida que la produjo. Las versiones no se borran ni se modifican; lo que se mueve son los **alias**, etiquetas con nombre que apuntan a una versión:
#
# - `champion` (campeón): la versión que está en producción;
# - `challenger` (retador): una candidata que se está evaluando para reemplazarla.
#
# El sistema que usa el modelo nunca pide «la versión 2»: pide `models:/riesgo-credito@champion`. Promover un modelo, o regresar al anterior si algo sale mal, es mover un alias, sin tocar el código del sistema.
#
# ### La firma del modelo
#
# Al guardar un modelo conviene guardar también su **firma** (*signature*): qué columnas y de qué tipo espera, y qué devuelve. MLflow la infiere de un ejemplo, la guarda junto al modelo y la usa para **validar la entrada** cuando alguien le pide una predicción.
#
# Contemos una historia corta. Hace unos meses, el banco puso en producción la logística (versión 1, alias `champion`). Hoy, la búsqueda de arriba propone el bosque como retador (versión 2, alias `challenger`). Ambos se reentrenan con los 22,500 clientes de entrenamiento y se registran.
#
# Un detalle que explicaremos en la sección 6: MLflow guarda los modelos de scikit-learn en el formato seguro `skops`, que se niega a guardar (y a cargar) ciertos tipos de objetos a menos que declaremos que confiamos en ellos. Los árboles de scikit-learn son uno de esos tipos; como los entrenamos nosotros mismos, los declaramos confiables.

# %%
NOMBRE_MODELO = "riesgo-credito"
cliente = mlflow.MlflowClient()
# Tipos que skops pide revisar antes de cargar, por familia (sección 6). Solo se declaran porque los modelos son nuestros.
TIPOS_CONFIABLES = {"logistica": None, "bosque": ["sklearn.tree._tree.Tree"]}


def registrar(config, alias):
    """Entrena con todo el conjunto de entrenamiento, guarda el modelo con su firma y lo registra."""
    modelo = construir(config).fit(X_tr, y_tr)
    firma = infer_signature(X_tr.head(100), modelo.predict_proba(X_tr.head(100)))
    with mlflow.start_run(run_name=f"final: {config['nombre']}"), contextlib.redirect_stderr(io.StringIO()):
        mlflow.set_tags(ENTORNO)
        mlflow.log_params({k: v for k, v in config.items() if k != "nombre"})
        mlflow.log_dict(config, "configuracion.json")
        info = mlflow.sklearn.log_model(modelo, name="modelo", signature=firma, input_example=X_tr.head(3),
                                        pyfunc_predict_fn="predict_proba",   # que predict devuelva probabilidades
                                        registered_model_name=NOMBRE_MODELO,
                                        skops_trusted_types=TIPOS_CONFIABLES[config["familia"]])
    version = info.registered_model_version
    cliente.set_model_version_tag(NOMBRE_MODELO, version, "familia", config["familia"])
    cliente.set_registered_model_alias(NOMBRE_MODELO, alias, version)
    return modelo, version


mejor_config = next(c for c in CONFIGURACIONES if c["nombre"] == tabla.iloc[0]["corrida"])
modelo_v1, v1 = registrar(CONFIGURACIONES[0], "champion")
modelo_v2, v2 = registrar(mejor_config, "challenger")


def mostrar_registro():
    for v in cliente.search_model_versions(f"name = '{NOMBRE_MODELO}'", order_by=["version_number ASC"]):
        alias = cliente.get_model_version(NOMBRE_MODELO, v.version).aliases
        print(f"versión {v.version} ({v.tags.get('familia', '')}): corrida {v.run_id[:8]}…, alias {alias}")


mostrar_registro()

# %% [markdown]
# La firma quedó guardada con el modelo:

# %%
print(mlflow.models.get_model_info(f"models:/{NOMBRE_MODELO}/{v2}").signature)

# %% [markdown]
# ### Campeón contra retador
#
# Ahora sí, y por primera vez, usamos el conjunto de prueba: los 7,500 clientes que ninguna decisión anterior ha visto. Comparamos las dos versiones con las mismas métricas.

# %%
p_v1 = modelo_v1.predict_proba(X_te)[:, 1]
p_v2 = modelo_v2.predict_proba(X_te)[:, 1]
duelo = pd.DataFrame({
    f"v{v1} logística (champion)": {"AUC": roc_auc_score(y_te, p_v1), "log-loss": log_loss(y_te, p_v1)},
    f"v{v2} bosque (challenger)": {"AUC": roc_auc_score(y_te, p_v2), "log-loss": log_loss(y_te, p_v2)},
}).T
duelo.round(4)

# %% [markdown]
# El retador gana en las dos métricas, y por un margen mucho mayor que el ruido entre pliegues: se promueve. Promover es mover el alias `champion` a la versión 2 y quitar el de `challenger`. La versión 1 sigue en el registro: si mañana el bosque se portara mal, regresar a la logística es mover el alias de vuelta.

# %%
cliente.set_registered_model_alias(NOMBRE_MODELO, "champion", v2)
cliente.delete_registered_model_alias(NOMBRE_MODELO, "challenger")
mostrar_registro()

# %% [markdown]
# ### Cargar por alias y predecir
#
# Así es como lo usaría otro sistema (por ejemplo, el servicio web de la siguiente libreta): sin saber qué familia de modelo es, ni qué versión, ni dónde está guardado. `mlflow.pyfunc` es una interfaz común para modelos de cualquier biblioteca: todos tienen un método `predict`.

# %%
campeon = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@champion")
print("el alias champion apunta a la versión", cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion").version)
nuevos = X_te.head(5)
pd.DataFrame({"P(incumple) cargado por alias": campeon.predict(nuevos)[:, 1],
              "P(incumple) en memoria": modelo_v2.predict_proba(nuevos)[:, 1],
              "incumplió": y_te.head(5).to_numpy()}, index=nuevos.index).round(4)

# %% [markdown]
# La firma no es decorativa. Si el sistema que llama al modelo manda una tabla a la que le falta una columna, el modelo **se niega** a predecir en lugar de devolver basura en silencio:

# %%
try:
    campeon.predict(nuevos.drop(columns="limite"))
except mlflow.exceptions.MlflowException as e:
    print(type(e).__name__, "→", re.search(r"Model is missing inputs[^\n]*", str(e)).group(0))

# %% [markdown]
# ## 5. Reproducir una corrida
#
# Supongamos que pasaron seis meses y alguien quiere confirmar el número que justificó elegir el bosque. Todo lo necesario está en la corrida. Primero la buscamos y comprobamos que los datos y el entorno son los mismos:

# %%
id_original = ids_corridas[mejor_config["nombre"]]
original = mlflow.get_run(id_original)
print("parámetros (como texto):", {k: original.data.params[k] for k in ["familia", "min_samples_leaf", "n_estimators"]})
print("¿mismos datos?  ", original.data.tags["datos.huella"] == huella_datos(credito))
print("¿mismo sklearn? ", original.data.tags["entorno.sklearn"] == sklearn.__version__)

# %% [markdown]
# Los parámetros vuelven como **texto** (`'50'`, no `50`): MLflow los guarda así. Por eso guardamos también la configuración en JSON, que conserva los tipos. Con ella reconstruimos el modelo, repetimos la validación cruzada con la partición guardada y comparamos con la métrica registrada:

# %%
config_recuperada = mlflow.artifacts.load_dict(f"{original.info.artifact_uri}/configuracion.json")
cv_recuperada = StratifiedKFold(n_splits=int(original.data.params["cv.pliegues"]), shuffle=True,
                                random_state=int(original.data.params["cv.semilla"]))
auc_repetido = cross_validate(construir(config_recuperada), X_tr, y_tr, cv=cv_recuperada, scoring="roc_auc")["test_score"]
print("configuración recuperada:", config_recuperada)
print(f"AUC registrado: {original.data.metrics['auc_media']:.10f}")
print(f"AUC repetido:   {auc_repetido.mean():.10f}")
print("diferencia:", abs(original.data.metrics["auc_media"] - auc_repetido.mean()))

# %% [markdown]
# Coinciden en todos los decimales. Si la huella de los datos o las versiones no coincidieran, sabríamos **antes** de repetir el cálculo que no hay que esperar el mismo número, y sabríamos dónde buscar la diferencia.

# %% [markdown]
# ## 6. Un modelo guardado es código: `pickle` y `skops`
#
# La forma tradicional de guardar un modelo de scikit-learn es `pickle` (o `joblib`, que lo usa por dentro). El problema es que `pickle` no es un formato de datos: es un pequeño lenguaje de instrucciones para reconstruir objetos, y una de ellas es «llama a esta función con estos argumentos». Cualquier objeto puede decidir qué se ejecuta al cargarlo, con el método `__reduce__`. Aquí, inofensivamente, un `print`; un atacante pondría `os.system(...)`:

# %%
class ModeloSospechoso:
    def __reduce__(self):
        return (print, ("⚠ este código se ejecutó al CARGAR el archivo, antes de que pudiéramos revisarlo",))


bytes_modelo = pickle.dumps(ModeloSospechoso())
_ = pickle.loads(bytes_modelo)   # solo cargar: no llamamos a nada

# %% [markdown]
# La documentación de Python lo advierte explícitamente: nunca se deben cargar datos de `pickle` de una fuente no confiable. Eso incluye modelos descargados de internet o compartidos por un desconocido.
#
# **`skops`** es un formato pensado para modelos de scikit-learn que no ejecuta código arbitrario: guarda los objetos como datos y, al cargar, **audita** los tipos que aparecen y se niega a construir los que no estén en una lista de confianza. MLflow lo usa por omisión para los modelos de scikit-learn. Veamos qué tipos le parecen sospechosos en tres modelos: la logística, el bosque y un *boosting* de LightGBM.

# %%
caja_negra = construir(CONFIGURACIONES[4]).fit(X_tr, y_tr)
for nombre, m in [("logística", modelo_v1), ("bosque", modelo_v2), ("LightGBM", caja_negra)]:
    print(f"{nombre:10s} tipos que hay que revisar:", sio.get_untrusted_types(data=sio.dumps(m)))

# %% [markdown]
# - Para la **logística** la lista está vacía: sus objetos (el *pipeline*, el codificador, los coeficientes) son tipos que `skops` sabe reconstruir como simples datos.
# - Para el **bosque** pide revisar uno, la estructura interna de los árboles de scikit-learn. No es que ejecute código: guarda los índices de los nodos hijos, y scikit-learn los usa sin comprobar que estén en rango. Un archivo manipulado podría poner índices fuera de rango y, al predecir, leer memoria ajena o tumbar el proceso. Por eso, en la sección 4, tuvimos que declararlo confiable.
# - Para **LightGBM** pide revisar tres tipos que no son de scikit-learn; en MLflow se declaran con el mismo argumento `skops_trusted_types` de `log_model`, o se usa el formato propio de LightGBM (`mlflow.lightgbm`).
#
# Si intentamos cargar el bosque sin declarar nada, `skops` se niega:

# %%
try:
    sio.loads(sio.dumps(modelo_v2))
except Exception as e:
    print(type(e).__name__, "→", str(e).splitlines()[0][:150])

# %% [markdown]
# La regla práctica: **un modelo que se carga es código que se ejecuta**, o al menos datos en los que el código confía ciegamente. Se cargan solo modelos de fuentes confiables, se prefiere un formato que no ejecute nada, y la lista de tipos confiables se revisa uno por uno: pasarle a ciegas todo lo que reporta `get_untrusted_types` anula la protección.

# %% [markdown]
# ## 7. La ficha del modelo
#
# El registro guarda el modelo y sus números; no dice para qué sirve ni para qué **no** sirve. M. Mitchell y colaboradores propusieron en 2019 (*Model Cards for Model Reporting*, FAT\* 2019) acompañar cada modelo publicado con una **ficha** breve, con nueve secciones fijas: detalles del modelo, uso previsto, factores (los grupos y condiciones relevantes), métricas, datos de evaluación, datos de entrenamiento, análisis cuantitativos, consideraciones éticas, y advertencias y recomendaciones. Un punto central de la propuesta es reportar las métricas **desagregadas por grupo**, no solo el promedio, como hicimos en la Unidad 3.
#
# Calculemos lo que la ficha necesita, en el conjunto de prueba. Usamos el umbral de la Unidad 3, $t^\ast = 1/6$, que sale de suponer que dejar pasar a un incumplido cuesta cinco veces más que molestar a un cliente cumplido.

# %%
T_ESTRELLA = 1 / 6


def tasas(y, p, t=T_ESTRELLA):
    y, senal = np.asarray(y), np.asarray(p) >= t
    return {"clientes": len(y), "prevalencia": y.mean(), "AUC": roc_auc_score(y, p),
            "tasa de selección": senal.mean(), "TVP": senal[y == 1].mean(), "TFP": senal[y == 0].mean()}


grupos = {"todos": np.ones(len(X_te), bool),
          "hombres": (X_te["sexo"] == 1).to_numpy(), "mujeres": (X_te["sexo"] == 2).to_numpy(),
          "menores de 30": (X_te["edad"] < 30).to_numpy(), "30 a 45": X_te["edad"].between(30, 45).to_numpy(),
          "mayores de 45": (X_te["edad"] > 45).to_numpy()}
desagregado = pd.DataFrame({g: tasas(y_te[m], p_v2[m]) for g, m in grupos.items()}).T
desagregado["clientes"] = desagregado["clientes"].astype(int)
desagregado.round(3)

# %% [markdown]
# Con estos números escribimos la ficha. La generamos desde el código, para que las cifras siempre coincidan con el modelo, y la guardamos como artefacto de la corrida de la versión campeona, junto al modelo.

# %%
d = desagregado.round(3)
ficha = f"""
### Ficha del modelo `{NOMBRE_MODELO}`, versión {v2} (alias `champion`)

**Detalles.** Bosque aleatorio de scikit-learn {sklearn.__version__}: 200 árboles, al menos 50 clientes por hoja, semilla {SEMILLA}.
Entrenado el {time.strftime('%Y-%m-%d')} con los datos de huella `{ENTORNO['datos.huella']}`. Elegido entre 7 configuraciones por AUC con validación cruzada de 5 pliegues.

**Uso previsto.** Ordenar a los clientes de tarjeta de crédito por riesgo de no pagar el mes siguiente, para priorizar contactos preventivos.
Con el umbral $t^\\ast = 1/6$ (costo de un incumplido no detectado = 5 × costo de una alerta innecesaria).

**Usos fuera de alcance.** Negar crédito a solicitantes nuevos (el modelo usa el historial de pagos de clientes existentes); otros países o épocas.

**Datos.** 30,000 clientes de un banco de Taiwán, abril–septiembre de 2005, en plena crisis de deuda de tarjetas (Yeh y Lien, 2009).
Entrenamiento: {len(X_tr):,}; prueba: {len(X_te):,} (separación estratificada, semilla {SEMILLA}). Prevalencia: {100 * y.mean():.1f} %.

**Métricas en prueba** (umbral 1/6):

| grupo | clientes | prevalencia | AUC | tasa de selección | TVP | TFP |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {g} | {int(f.clientes):,} | {f.prevalencia:.3f} | {f.AUC:.3f} | {f['tasa de selección']:.3f} | {f.TVP:.3f} | {f.TFP:.3f} |"
                for g, f in d.iterrows()) + f"""

**Consideraciones éticas.** El modelo usa `sexo`, `edad` y `estado_civil`, variables cuyo uso en decisiones de crédito la regulación de muchos países limita o prohíbe.
Aun quitándolas, otras variables correlacionadas podrían reproducir las diferencias. Con el mismo umbral, la tasa de falsos positivos
no es igual entre grupos: un cliente cumplido es señalado con probabilidad {d.TFP['hombres']:.3f} si es hombre y {d.TFP['mujeres']:.3f} si es mujer,
y {d.TFP['mayores de 45']:.3f} si es mayor de 45 años.

**Advertencias.** Datos de una crisis de 2005: la relación entre historial y riesgo puede ser muy distinta hoy (cambio de distribución, siguiente libreta).
Las probabilidades del bosque no están calibradas (Unidad 3); si se usan como probabilidades y no solo para ordenar, hay que recalibrarlas.
Entre los dos mejores modelos la diferencia de AUC en validación cruzada fue menor que el ruido entre pliegues.
"""
with mlflow.start_run(run_id=cliente.get_model_version(NOMBRE_MODELO, v2).run_id):
    mlflow.log_text(ficha, "ficha_modelo.md")

Markdown(ficha)

# %% [markdown]
# La ficha cabe en una pantalla, y aun así contiene cosas que no estaban en ningún otro lado: para qué **no** debe usarse el modelo, en qué grupos se equivoca de manera distinta y por qué sus números podrían no valer hoy. Es la parte del registro que lee quien decide, no quien entrena.
#
# ## 8. Cierre
#
# Lo que hicimos en esta libreta cabe en una idea: **un modelo no es solo un archivo de pesos**. Es el resultado de un código, unos datos, un entorno y unas decisiones, y para confiar en él hay que poder decir cuáles fueron. MLflow no hace nada que no pudiéramos hacer con archivos JSON y disciplina; lo que aporta es un lugar común y una convención, para que la disciplina no dependa de la memoria de nadie.
#
# *Referencias:* D. Sculley et al., «Hidden Technical Debt in Machine Learning Systems», *NeurIPS* (2015). M. Mitchell et al., «Model Cards for Model Reporting», *Proc. FAT\** (2019). Documentación de MLflow: [mlflow.org/docs](https://mlflow.org/docs/latest/). I.-C. Yeh y C. Lien, *Expert Systems with Applications* 36 (2009).

# %% [markdown]
# ## 9. Autoevaluación

# %%
rdp.quiz.mostrar("u08", ids=["u08-p01", "u08-p02", "u08-p03", "u08-p04"])

# %% [markdown]
# ## 10. Ejercicios
#
# 1. **Búsqueda registrada.** Agrega al experimento una búsqueda en malla para el *boosting* (al menos 3 tasas de aprendizaje × 3 números de hojas) usando `correr`. Con `mlflow.search_runs`, haz una gráfica de AUC contra tasa de aprendizaje, una curva por número de hojas. ¿Cambia el ganador? ¿La diferencia supera la desviación entre pliegues?
# 2. **¿Cuándo una diferencia es real?** Las métricas por pliegue están registradas como la serie `auc_pliegue` (usa `cliente.get_metric_history`). Como las dos configuraciones se evaluaron en los **mismos** pliegues, compara el bosque y el *boosting* η = 0.05 con las diferencias pliegue a pliegue. Explica por qué la desviación estándar de la diferencia pareada puede ser mucho menor que la de cada modelo por separado (escribe $\operatorname{Var}(A - B)$ en términos de las varianzas y la covarianza) y decide si la diferencia es significativa. ¿Qué advertencia hay sobre la independencia de los pliegues?
# 3. **La huella de los datos.** Demuestra que `huella_datos` cambia si se reordenan las filas, y explica por qué eso puede ser un problema o una virtud. Escribe una variante que no dependa del orden de las filas (sugerencia: ordenar los *hashes* por fila antes de combinarlos). ¿Qué cambio en los datos no detectaría tu variante?
# 4. **El retador que no gana.** Registra el *boosting* η = 0.3 como versión 3 con alias `challenger` y evalúalo contra el campeón en prueba. Escribe una función `promover_si_mejora(nombre, metrica, margen)` que mueva el alias solo si el retador supera al campeón por más de `margen`, y registre la decisión como etiqueta en la versión. ¿Por qué no basta con «mejor en prueba»? (Piensa en cuántas veces se puede usar el mismo conjunto de prueba para decidir.)
# 5. **Una ficha honesta.** Completa la ficha con la calibración por grupo (curva de confiabilidad de hombres y mujeres) y con el desempeño del modelo si se quitan `sexo`, `edad` y `estado_civil`. ¿Cuánto AUC cuesta quitarlas? ¿Se igualan las tasas de falsos positivos entre grupos?
