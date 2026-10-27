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
# # Un servicio de predicción y la vigilancia del cambio
#
# **Unidad 8 · Ejemplo 2**
#
# Un modelo validado con cuidado es una promesa sobre datos que **se parecen** a los de entrenamiento. En producción esa promesa se pone a prueba todos los días: los usuarios cambian, el clima cambia, el negocio crece. En esta libreta seguimos la vida de un modelo de demanda de bicicletas durante su primer año en producción:
#
# 1. lo entrenamos con los datos de 2011 y estimamos su error con validación;
# 2. lo **servimos** como una API con FastAPI, con validación de la entrada, versión del modelo en la respuesta y una bitácora de lo que predice;
# 3. simulamos el tráfico de 2012, mes por mes, y estudiamos **qué puede cambiar**: la distribución de las entradas, la relación entre entradas y respuesta, o la de la respuesta;
# 4. construimos dos detectores, el **índice de estabilidad poblacional** (PSI) y la **prueba de Kolmogorov–Smirnov**, y vemos qué detectan, qué no y cuándo dan falsas alarmas;
# 5. medimos el error real cuando llegan las etiquetas, y comparamos **políticas de reentrenamiento** con y sin retraso en las etiquetas.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios) y se asegura de tener FastAPI y `httpx` (que usa su cliente de pruebas). En tu máquina, si usas el entorno del repositorio, ya están instalados.

# %%
import os, sys

RAMA = "rediseno-2027"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete" fastapi httpx
    pass

# %%
import datetime as dt
import hashlib, tempfile, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import ks_2samp

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

from fastapi import FastAPI
warnings.filterwarnings("ignore", message=".*httpx.*")   # aviso de Starlette sobre el cliente de pruebas, irrelevante aquí
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# %% [markdown]
# # Parte 1. Un modelo en producción
#
# ## 1. Los datos y el problema
#
# **Contexto.** Capital Bikeshare es el sistema de bicicletas compartidas de Washington, D. C.: la persona toma una bicicleta en una estación y la deja en otra, y paga por tiempo de uso. Para el operador es crucial anticipar la **demanda por hora**: de ella dependen cuántas bicicletas redistribuir entre estaciones (con camionetas, de madrugada), cuánto personal de mantenimiento programar y dónde poner estaciones nuevas.
#
# **Datos.** Las rentas por hora de 2011 y 2012 (17,379 horas), con el calendario y el clima de cada hora, reunidas por H. Fanaee-T y J. Gama (*Progress in Artificial Intelligence*, 2014) y publicadas en el repositorio de la UCI con licencia CC BY 4.0. Son los mismos datos de la Unidad 4. Usamos la hora del día, si es día laboral, el día de la semana, el clima (1 despejado, 2 nublado o neblina, 3 lluvia o nieve; la categoría 4, de solo 3 horas, se une a la 3), la temperatura (°C), la humedad (%) y el viento (km/h). La respuesta es `total`, el número de bicicletas rentadas en la hora.
#
# **Problema.** Es el 1 de enero de 2012. Tenemos un año de historia y queremos poner en producción un modelo que, dado el pronóstico de calendario y clima de una hora, prediga cuántas bicicletas se rentarán. Lo validaremos con 2011, lo serviremos como una API y lo **vigilaremos durante 2012**. La pregunta de fondo es la de cualquier sistema en producción: ¿el error que medimos en validación es el error que tendremos?

# %%
b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
b["clima"] = b.clima.clip(upper=3)
b["anio"] = b.fecha.dt.year
b["periodo"] = (b.anio - 2011) * 12 + b.mes        # 1 = enero de 2011, …, 24 = diciembre de 2012
ATRIBUTOS = ["hora", "laboral", "dia_semana", "clima", "temp", "humedad", "viento"]
b11, b12 = b[b.anio == 2011], b[b.anio == 2012]
print(f"2011: {len(b11):,} horas, {b11.total.sum():,} rentas · 2012: {len(b12):,} horas, {b12.total.sum():,} rentas")

# %% [markdown]
# Antes de entrenar nada, miremos las rentas por mes de los dos años. Es una gráfica que el día de la puesta en producción **no podríamos haber hecho**: la mitad derecha es el futuro.

# %%
mensual = b.groupby(["anio", "mes"]).total.mean().unstack(0)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(range(1, 13), mensual[2011], marker="o", color=AZUL, lw=2.5)
ax.plot(range(1, 13), mensual[2012], marker="o", color=ROJO, lw=2.5)
rdp.estilo.etiquetar(ax, 12.3, {"2011 (entrenamiento)": mensual[2011].iloc[-1], "2012 (producción)": mensual[2012].iloc[-1]}, fontsize=12)
ax.set_xticks(range(1, 13), MESES)
ax.set_xlim(0.6, 14.6)
ax.set_ylim(0, None)
ax.set_ylabel("bicicletas por hora (promedio del mes)")
ax.set_title("Las rentas de 2012 se parecen a las de 2011… multiplicadas")
fig.tight_layout()
plt.show()
print("cociente 2012/2011 por mes:", (mensual[2012] / mensual[2011]).round(2).tolist())

# %% [markdown]
# La forma del año es la misma (pocas rentas en invierno, muchas en verano y otoño), pero 2012 está muy por encima: el sistema **creció**. En 2012 se rentaron 65 % más bicicletas que en 2011; en enero, febrero y marzo, el doble o más. El clima de las dos temporadas fue parecido; lo que cambió fue el número de usuarios y de estaciones. Ningún atributo del modelo mide eso.

# %% [markdown]
# ## 2. El modelo y su error esperado
#
# Usamos *boosting* de gradiente con histogramas (Unidad 5) y **pérdida de Poisson**, porque la respuesta es un conteo (Unidad 4). Antes de entrenar con todo 2011, estimamos su error con una partición de 2011: los días 1–19 de cada mes para entrenar y del 20 en adelante para validar. Medimos la raíz del error cuadrático medio (RECM) y el **sesgo relativo** $\sum_n \hat y_n / \sum_n y_n - 1$, que dice en qué proporción el modelo sobreestima (positivo) o subestima (negativo) el total.

# %%
def nuevo_modelo():
    return HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.1, random_state=SEMILLA)


def sesgo(y, pred):
    return np.sum(pred) / np.sum(y) - 1


ent, val = b11[b11.fecha.dt.day <= 19], b11[b11.fecha.dt.day >= 20]
pred_val = nuevo_modelo().fit(ent[ATRIBUTOS], ent.total).predict(val[ATRIBUTOS])
RECM_VAL = root_mean_squared_error(val.total, pred_val)
print(f"validación (2011, días 20 en adelante): RECM = {RECM_VAL:.1f} bicicletas/hora, sesgo = {100 * sesgo(val.total, pred_val):+.1f} %")

# %% [markdown]
# Un error típico de unas 47 bicicletas por hora, para una respuesta que va de 1 a casi 1,000, y prácticamente sin sesgo: el modelo está listo. Lo entrenamos con **todo** 2011 y lo guardamos en disco, como haríamos para entregarlo a producción.
#
# Junto con el archivo guardamos una **versión**: un nombre legible y una huella (SHA-256) del archivo. Con la huella, cualquiera puede verificar que el modelo que responde en producción es exactamente el que se validó. (En la libreta anterior registramos los modelos en MLflow, que hace esto mismo por nosotros. Recuerda que `joblib` y `pickle` **ejecutan código** al cargar: solo se cargan archivos de fuentes confiables.)

# %%
modelo_2011 = nuevo_modelo().fit(b11[ATRIBUTOS], b11.total)

DIR = Path(tempfile.mkdtemp(prefix="bicis-"))
ruta = DIR / "modelo.joblib"
joblib.dump(modelo_2011, ruta)
huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
VERSION = f"bicis-2011-v1+{huella[:8]}"
print(f"{ruta.name}: {ruta.stat().st_size / 1024:.0f} KB · versión {VERSION.split('+')[0]} (la huella depende del entorno, por ejemplo de la versión de scikit-learn)")

# %% [markdown]
# # Parte 2. Servir el modelo
#
# ## 3. Formas de servir un modelo
#
# Hay tres maneras típicas de poner un modelo a trabajar:
#
# | forma | cuándo | ejemplo | lo que importa |
# |---|---|---|---|
# | **por lotes** (*batch*) | las predicciones se pueden calcular con anticipación | cada noche, la demanda de mañana hora por hora para planear la redistribución | rendimiento (predicciones por segundo), costo |
# | **en línea** (una API) | la predicción depende de algo que se sabe hasta el momento | la app pregunta «¿habrá bicicletas en esta estación en 30 minutos?» | **latencia** (milisegundos por respuesta), disponibilidad |
# | **en el dispositivo** | no hay red, o los datos no deben salir del teléfono | el teclado que predice la siguiente palabra | tamaño del modelo, consumo de energía |
#
# Para la redistribución nocturna bastaría el servicio por lotes. Construiremos una API en línea, porque es la forma más general: una API también puede recibir lotes.
#
# ## 4. Una API con FastAPI
#
# [FastAPI](https://fastapi.tiangolo.com) construye servicios web a partir de funciones de Python. Las piezas son:
#
# - un **esquema** de entrada con Pydantic: cada campo con su tipo y su rango válido. Si una petición no cumple el esquema, FastAPI responde con el código **422** y una explicación, sin llegar al modelo. Es la primera línea de defensa: un modelo de árboles predice *algo* con una temperatura de 300 °C; el esquema impide que lo haga;
# - un esquema de salida, que incluye la **versión** del modelo que respondió: cuando algo salga mal, lo primero que querremos saber es qué modelo estaba en producción;
# - los *endpoints*: funciones asociadas a una ruta (`/predecir`) y a un método HTTP (`POST`);
# - una **bitácora** de lo que el modelo recibe y predice. Sin ella no hay monitoreo posible.
#
# La fecha de la hora a predecir viaja en la petición aunque el modelo no la use: la bitácora la necesita para agrupar por mes. Envolvemos la construcción de la app en una función para poder cambiar de modelo sin cambiar el código del servicio.

# %%
class Hora(BaseModel):
    """Una hora para la que se quiere la demanda: calendario y pronóstico del clima."""
    fecha: dt.date
    hora: int = Field(ge=0, le=23)
    laboral: int = Field(ge=0, le=1, description="1 si no es fin de semana ni feriado")
    dia_semana: int = Field(ge=0, le=6, description="0 = domingo")
    clima: int = Field(ge=1, le=4, description="1 despejado, 2 nublado, 3 lluvia ligera, 4 lluvia intensa")
    temp: float = Field(ge=-25, le=50, description="°C")
    humedad: float = Field(ge=0, le=100, description="%")
    viento: float = Field(ge=0, le=100, description="km/h")


class Prediccion(BaseModel):
    rentas: float
    version: str


class Lote(BaseModel):
    horas: list[Hora]


class PrediccionLote(BaseModel):
    rentas: list[float]
    version: str


def crear_app(modelo, version: str, bitacora: list) -> FastAPI:
    app = FastAPI(title="Demanda de bicicletas")

    def predecir_df(horas: list[Hora]) -> np.ndarray:
        X = pd.DataFrame([h.model_dump() for h in horas])
        X["clima"] = X.clima.clip(upper=3)          # el mismo preprocesamiento que en el entrenamiento
        pred = modelo.predict(X[ATRIBUTOS])
        bitacora.append(X.assign(prediccion=pred, version=version))
        return pred

    @app.get("/salud")
    def salud():
        return {"estado": "ok", "version": version}

    @app.post("/predecir", response_model=Prediccion)
    def predecir(h: Hora):
        return Prediccion(rentas=float(predecir_df([h])[0]), version=version)

    @app.post("/predecir/lote", response_model=PrediccionLote)
    def predecir_lote(lote: Lote):
        return PrediccionLote(rentas=predecir_df(lote.horas).tolist(), version=version)

    return app


bitacora = []
cliente = TestClient(crear_app(joblib.load(ruta), VERSION, bitacora))
cliente.get("/salud").json()

# %% [markdown]
# `TestClient` hace peticiones HTTP a la app **sin levantar un servidor**: es la manera de escribir pruebas automáticas de un servicio. Una petición válida, el 1 de junio de 2012 a las 5 de la tarde, día laboral despejado:

# %%
peticion = {"fecha": "2012-06-01", "hora": 17, "laboral": 1, "dia_semana": 5,
            "clima": 1, "temp": 27.0, "humedad": 50, "viento": 15}
r = cliente.post("/predecir", json=peticion)
print(r.status_code, r.json())

# %% [markdown]
# Y una inválida: alguien envió la temperatura en Fahrenheit y la hora en formato de 12 horas con un error.

# %%
mala = peticion | {"temp": 81.0, "hora": 25}
r = cliente.post("/predecir", json=mala)
print(r.status_code)
for e in r.json()["detail"]:
    print(f"  {e['loc'][-1]}: {e['msg']}")

# %% [markdown]
# La app rechazó la petición antes de llegar al modelo y dijo exactamente qué estaba mal. Esto se puede convertir en pruebas automáticas que se ejecutan cada vez que cambia el código:

# %%
def test_servicio(cliente):
    assert cliente.post("/predecir", json=peticion).status_code == 200
    assert cliente.post("/predecir", json=peticion | {"hora": 24}).status_code == 422
    assert cliente.post("/predecir", json={k: v for k, v in peticion.items() if k != "temp"}).status_code == 422
    assert cliente.post("/predecir", json=peticion).json()["version"].startswith("bicis-")
    assert cliente.post("/predecir", json=peticion).json()["rentas"] >= 0       # Poisson: nunca negativas


test_servicio(cliente)
bitacora.clear()      # las pruebas no son tráfico real
print("pruebas aprobadas")

# %% [markdown]
# ### Latencia: una por una o por lotes
#
# Cada petición tiene un costo fijo (validar, convertir a tabla, llamar al modelo, serializar la respuesta) además del costo por fila. Midámoslo; las cifras exactas dependen de la máquina.

# %%
horas = [peticion | {"hora": h % 24} for h in range(500)]
t0 = time.perf_counter()
for h in horas[:100]:
    cliente.post("/predecir", json=h)
una = (time.perf_counter() - t0) / 100
t0 = time.perf_counter()
cliente.post("/predecir/lote", json={"horas": horas})
lote = (time.perf_counter() - t0) / len(horas)
bitacora.clear()
print(f"una por una: {1000 * una:.1f} ms por predicción · en lote de {len(horas)}: {1000 * lote:.3f} ms por predicción ({una / lote:.0f} veces menos)")

# %% [markdown]
# Una petición individual tarda milisegundos, y casi todo es costo fijo: en lote, cada predicción cuesta cientos de veces menos. Por eso los servicios en línea agrupan peticiones cuando pueden, y por eso, si las predicciones se pueden calcular con anticipación, el servicio por lotes es más barato.
#
# ### Fuera de la libreta: servidor y contenedor
#
# Para servirla de verdad, se guarda la app en un archivo `servicio.py` (que cargue el modelo con `joblib.load` al arrancar) y se levanta con un servidor ASGI:
#
# ```bash
# uvicorn servicio:app --host 0.0.0.0 --port 8000
# ```
#
# Para que corra igual en cualquier máquina se empaqueta en un **contenedor**: una imagen con el sistema operativo mínimo, Python, las bibliotecas en versiones fijas, el código y el modelo. Un `Dockerfile` corto:
#
# ```dockerfile
# FROM python:3.12-slim                         # sistema base con Python
# WORKDIR /app
# COPY requirements.txt .                       # versiones fijas: scikit-learn==1.9.0, fastapi==…
# RUN pip install --no-cache-dir -r requirements.txt
# COPY servicio.py modelo.joblib ./             # el código y el modelo validado (verifica su huella)
# EXPOSE 8000
# CMD ["uvicorn", "servicio:app", "--host", "0.0.0.0", "--port", "8000"]
# ```
#
# La versión de scikit-learn importa: un modelo guardado con una versión puede no cargarse, o cargarse con advertencias, en otra. El contenedor congela el entorno junto con el modelo.

# %% [markdown]
# ## 5. Un año de tráfico
#
# Simulamos 2012: cada mes, el sistema de planeación envía al servicio un lote con las horas del mes (en la vida real llegarían una por una o día por día, pero el resultado en la bitácora es el mismo). Las peticiones solo llevan calendario y clima: **las rentas reales no viajan con la petición**. Se conocerán después.

# %%
campos = ["hora", "laboral", "dia_semana", "clima", "temp", "humedad", "viento"]
for mes in range(1, 13):
    d = b12[b12.mes == mes]
    lote = d[campos].assign(fecha=d.fecha.dt.strftime("%Y-%m-%d")).to_dict(orient="records")
    r = cliente.post("/predecir/lote", json={"horas": lote})
    assert r.status_code == 200

registro = pd.concat(bitacora, ignore_index=True)
registro["mes"] = pd.to_datetime(registro.fecha).dt.month
print(f"{len(registro):,} predicciones registradas, versión {registro.version.unique()[0].split('+')[0]}")
registro.head(3)

# %% [markdown]
# # Parte 3. ¿Qué puede cambiar?
#
# ## 6. Tres maneras de que el mundo cambie
#
# Un modelo aprende de una distribución conjunta $P(\mathbf x, y)$. Hay dos formas de factorizarla, y cada factor puede cambiar por separado:
#
# $$P(\mathbf x, y) = P(\mathbf x)\,P(y \mid \mathbf x) = P(y)\,P(\mathbf x \mid y).$$
#
# - **Cambio de covariables** (*covariate shift*, una forma de *data drift*): cambia $P(\mathbf x)$ y $P(y\mid\mathbf x)$ se queda igual. Un marzo más caluroso que el del año pasado; una app que empieza a usarse en otra ciudad.
# - **Cambio de concepto** (*concept drift*): cambia $P(y \mid \mathbf x)$. Con el mismo clima y la misma hora se rentan más bicicletas porque hay más usuarios; con el mismo historial de pagos, un cliente es más riesgoso porque subieron las tasas de interés.
# - **Cambio de prevalencia** (*prior shift*): cambia $P(y)$ con $P(\mathbf x \mid y)$ fijo. Una epidemia hace que crezca la proporción de enfermos entre los que llegan a la clínica. En la Unidad 3 vimos que la **precisión** de un clasificador depende de la prevalencia aunque la TVP y la TFP no cambien: un cambio de prevalencia cambia la precisión sin que el modelo «empeore».
#
# **Por qué importa la diferencia.** Sea $e(\mathbf x) = \mathbb{E}\big[(y - h(\mathbf x))^2 \mid \mathbf x\big] = \big(f(\mathbf x) - h(\mathbf x)\big)^2 + \sigma^2(\mathbf x)$ el error esperado del modelo $h$ en el punto $\mathbf x$, con $f(\mathbf x) = \mathbb{E}[y\mid\mathbf x]$. Si las entradas pasan a seguir una distribución $Q$ en lugar de $P$,
#
# $$E_{\text{out}}(Q) = \mathbb{E}_{\mathbf x \sim Q}\big[e(\mathbf x)\big] = \mathbb{E}_{\mathbf x\sim P}\Big[\frac{q(\mathbf x)}{p(\mathbf x)}\,e(\mathbf x)\Big].$$
#
# Con **solo** cambio de covariables, $f$ y $\sigma^2$ no cambian: el error se **repondera** hacia las regiones que ahora son más frecuentes. Si el modelo era bueno en todas las regiones con datos de entrenamiento, el error cambia poco; el peligro está en las regiones donde $p(\mathbf x)$ era casi cero (extrapolación). Con cambio de **concepto**, en cambio, $f$ se mueve y aparece un término $(f_{\text{nueva}} - h)^2$ que ningún reponderado arregla: el modelo está equivocado incluso donde tenía muchos datos.
#
# ## 7. Las tres, en las bicicletas
#
# Comparemos cada mes de 2012 con **el mismo mes de 2011**: la temperatura (una entrada), la relación entre temperatura y rentas, y las rentas mismas.

# %%
def histo(a, bordes):
    return np.histogram(a, bordes)[0] / len(a)


fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
bt = np.arange(-2, 36, 2)
for d, c, et in [(b11[b11.mes == 3], AZUL, "marzo 2011"), (b12[b12.mes == 3], ROJO, "marzo 2012")]:
    ejes[0].stairs(histo(d.temp, bt), bt, color=c, lw=2.5)
ejes[0].text(4, 0.25, "marzo 2011", color=AZUL, fontsize=12)
ejes[0].text(22, 0.17, "marzo 2012", color=ROJO, fontsize=12)
ejes[0].set(title="$P(x)$: la temperatura de marzo", xlabel="temperatura (°C)", ylabel="proporción de horas")

bins_t = np.arange(-6, 42, 3)
for d, c, et in [(b11, AZUL, "2011"), (b12, ROJO, "2012")]:
    g = d.groupby(pd.cut(d.temp, bins_t), observed=True).total.agg(["mean", "size"])
    g = g[g["size"] >= 50]
    ejes[1].plot([i.mid for i in g.index], g["mean"], marker="o", color=c, lw=2.5)
ejes[1].text(27, 330, "2012", color=ROJO, fontsize=12, ha="right")
ejes[1].text(29, 160, "2011", color=AZUL, fontsize=12)
ejes[1].set(title="$P(y \\mid x)$: rentas según la temperatura", xlabel="temperatura (°C)", ylabel="bicicletas por hora (promedio)")

by = np.arange(0, 1000, 50)
for d, c, et in [(b11[b11.mes == 6], AZUL, "junio 2011"), (b12[b12.mes == 6], ROJO, "junio 2012")]:
    ejes[2].stairs(histo(d.total, by), by, color=c, lw=2.5)
ejes[2].text(270, 0.14, "junio 2011", color=AZUL, fontsize=12)
ejes[2].text(560, 0.065, "junio 2012", color=ROJO, fontsize=12)
ejes[2].set(title="$P(y)$: las rentas de junio", xlabel="bicicletas en la hora", ylabel="proporción de horas")
fig.tight_layout()
plt.show()

# %% [markdown]
# - **Izquierda.** Marzo de 2012 fue excepcionalmente caluroso en Washington: 18.4 °C de promedio contra 13.6 °C en marzo de 2011. Es un cambio en $P(\mathbf x)$, y uno benigno: el modelo vio temperaturas así en abril y mayo de 2011.
# - **Centro.** A la misma temperatura, en 2012 se rentaron muchas más bicicletas. Es un cambio en $P(y\mid\mathbf x)$: el que el modelo no puede ver, porque ningún atributo mide el tamaño del sistema.
# - **Derecha.** La distribución de las rentas se desplazó a la derecha. Es la consecuencia visible de lo anterior, pero solo la podemos dibujar cuando **ya tenemos las etiquetas**.

# %% [markdown]
# # Parte 4. Detectar el cambio sin etiquetas
#
# Lo que el servicio tiene a la mano, en tiempo real, es la bitácora: entradas y predicciones. Las rentas reales llegan después (en las bicicletas, una hora después; en el crédito, meses después; en un diagnóstico médico, quizá años). Así que el primer monitoreo compara la **distribución** de lo que llega con la de referencia.
#
# ## 8. El índice de estabilidad poblacional (PSI)
#
# Partimos el rango de un atributo en $K$ cubetas con los cuantiles de la **referencia** (aquí, deciles: $K = 10$, cada una con 10 % de la referencia). Sean $p_k$ la proporción de la referencia y $q_k$ la de los datos actuales en la cubeta $k$. El PSI es
#
# $$\text{PSI} = \sum_{k=1}^{K} (q_k - p_k)\ln\frac{q_k}{p_k}.$$
#
# Nació en la industria del crédito, pero tiene una lectura limpia en teoría de la información. Separando la suma,
#
# $$\text{PSI} = \sum_k q_k \ln\frac{q_k}{p_k} + \sum_k p_k\ln\frac{p_k}{q_k} = D(q\,\|\,p) + D(p\,\|\,q),$$
#
# la **divergencia de Kullback–Leibler simetrizada** (la divergencia de Jeffreys) entre las dos distribuciones discretizadas. Así que $\text{PSI} \ge 0$, con igualdad solo si $q = p$, y es simétrico. Si una cubeta queda vacía, $\ln(q_k/p_k)$ explota; en la práctica se reemplazan las proporciones nulas por un $\varepsilon$ pequeño.
#
# **¿Cuánto es mucho?** La regla práctica de la industria es: menos de 0.1, estable; entre 0.1 y 0.25, cambio moderado que hay que mirar; más de 0.25, cambio importante. Son umbrales de costumbre, no de ninguna teoría, y no toman en cuenta el tamaño de las muestras. Veamos por qué eso importa.
#
# **El piso de ruido.** Para cambios pequeños, con $q_k = p_k + \delta_k$, $\ln(q_k/p_k) \approx \delta_k/p_k$ y
#
# $$\text{PSI} \approx \sum_k \frac{\delta_k^2}{p_k},$$
#
# una distancia $\chi^2$. Si **no** hay ningún cambio, pero las proporciones se estiman con $N$ datos de referencia y $M$ actuales, cada $\hat q_k - \hat p_k$ tiene media 0 y varianza $p_k(1-p_k)(1/N + 1/M)$, así que
#
# $$\mathbb{E}[\text{PSI}] \approx \sum_k (1 - p_k)\Big(\frac1N + \frac1M\Big) = (K - 1)\Big(\frac1N + \frac1M\Big).$$
#
# Con $K = 10$ y un mes de datos ($M \approx 720$ horas) contra un mes de referencia ($N \approx 720$), el PSI de dos muestras **de la misma distribución** ya vale unos 0.025. Con un día de datos ($M = 24$) valdría 0.4: «cambio importante» por puro azar. Verifiquémoslo tomando dos muestras al azar de las temperaturas de 2011.

# %%
def psi(ref, act, k=10, eps=1e-4):
    """PSI con cubetas en los cuantiles de la referencia. Para atributos discretos, cada valor es una cubeta."""
    ref, act = np.asarray(ref, float), np.asarray(act, float)
    valores = np.unique(ref)
    if len(valores) <= k:                                   # atributo discreto (clima, laboral…)
        cortes = (valores[:-1] + valores[1:]) / 2
    else:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, k + 1)[1:-1]))
    p = np.bincount(np.searchsorted(cortes, ref, side="right"), minlength=len(cortes) + 1) / len(ref)
    q = np.bincount(np.searchsorted(cortes, act, side="right"), minlength=len(cortes) + 1) / len(act)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


rng = np.random.default_rng(SEMILLA)
temps = b11.temp.to_numpy() + rng.normal(0, 0.05, len(b11))   # rompemos empates (la temperatura viene redondeada)
tamanos = [24, 72, 168, 720, 2000, 8000]
filas = []
for m in tamanos:
    sim = [psi(rng.choice(temps, m), rng.choice(temps, m)) for _ in range(300)]
    filas.append({"horas en cada muestra": m, "PSI promedio (simulado)": np.mean(sim),
                  "(K − 1)·2/M (fórmula)": 9 * 2 / m, "% de veces > 0.1": 100 * np.mean(np.array(sim) > 0.1)})
piso = pd.DataFrame(filas).set_index("horas en cada muestra")
piso.round(4)

# %% [markdown]
# La fórmula acierta desde unas cien horas (con 24, muchas cubetas quedan vacías, el $\varepsilon$ entra en juego y el PSI se infla todavía más). La última columna es la advertencia: con una semana de datos (168 horas) contra otra semana, la regla «PSI > 0.1» da una falsa alarma más de la mitad de las veces; con un día, siempre. El umbral sensato depende de cuántos datos se comparan.

# %% [markdown]
# ### La trampa de la estacionalidad
#
# ¿Contra qué comparamos cada mes? La opción ingenua es usar todo 2011 como referencia. Hagámoslo con la temperatura **de los propios meses de 2011**, que son, por definición, datos de entrenamiento:

# %%
psi_ingenuo = [psi(b11.temp, b11[b11.mes == m].temp) for m in range(1, 13)]
psi_2012_ingenuo = [psi(b11.temp, registro[registro.mes == m].temp) for m in range(1, 13)]
psi_mismo_mes = [psi(b11[b11.mes == m].temp, registro[registro.mes == m].temp) for m in range(1, 13)]
pd.DataFrame({"2011 contra todo 2011": psi_ingenuo, "2012 contra todo 2011": psi_2012_ingenuo,
              "2012 contra el mismo mes de 2011": psi_mismo_mes}, index=MESES).round(2).T

# %% [markdown]
# Contra todo el año, **cada mes** de 2011 tiene un PSI de entre 1 y 5: «cambio importante» en los datos con los que se entrenó el modelo. No hay ningún problema: un mes de enero no se parece al año completo. Un monitor así suena todo el tiempo y en dos semanas nadie le hace caso. Cuando los datos tienen estaciones (horas, días de la semana, meses), la referencia debe respetarlas: comparamos cada mes con **el mismo mes del año anterior**. Así, la temperatura solo sale alta en los meses en que de verdad fue distinta del año anterior (marzo, enero y noviembre, sobre todo).
#
# ### El tablero de PSI
#
# Calculamos el PSI de cada atributo y de **las predicciones** del modelo, mes por mes, contra el mismo mes de 2011. Las predicciones de referencia son las del modelo sobre los datos de 2011.

# %%
b11 = b11.assign(prediccion=modelo_2011.predict(b11[ATRIBUTOS]))
vigilados = ["temp", "humedad", "viento", "clima", "prediccion"]
tablero = pd.DataFrame({v: [psi(b11[b11.mes == m][v], registro[registro.mes == m][v]) for m in range(1, 13)]
                        for v in vigilados}, index=MESES).T
tablero.index = ["temperatura", "humedad", "viento", "clima", "predicción"]


def mapa_psi(ax, tabla, titulo):
    im = ax.imshow(tabla.clip(upper=0.5), cmap=rdp.estilo.SECUENCIAL, vmin=0, vmax=0.5, aspect="auto")
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = tabla.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.25 else TINTA, fontweight="bold" if v > 0.25 else "normal")
    ax.set_xticks(range(12), MESES)
    ax.set_yticks(range(tabla.shape[0]), tabla.index)
    ax.grid(False)
    ax.set_title(titulo)
    return im


fig, ax = plt.subplots(figsize=(12, 3.9))
im = mapa_psi(ax, tablero, "PSI de cada mes de 2012 contra el mismo mes de 2011 (en negritas, > 0.25)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 0.1, 0.25, 0.5])
fig.tight_layout()
plt.show()

# %% [markdown]
# Hay alarmas en las entradas: la temperatura de enero, febrero, marzo, mayo y noviembre; la humedad de abril, septiembre y noviembre; el clima de septiembre. Es el tiempo atmosférico: un mes tiene pocas semanas, y dos meses de marzo pueden ser muy distintos. Pero la fila de abajo es la importante: **las predicciones casi no se movieron**. Solo en enero y marzo, los meses más calurosos que el año anterior, el PSI de las predicciones pasa de 0.1 (y no llega a 0.25): el modelo predice más rentas porque hace más calor, que es justo lo que debe hacer. Las demás alarmas de entrada no cambiaron lo que el modelo dice.
#
# Si solo miráramos este tablero, concluiríamos que todo va bien. Guardemos esa conclusión para la Parte 5.

# %% [markdown]
# ## 9. La prueba de Kolmogorov–Smirnov
#
# El PSI depende de cómo se elijan las cubetas. Una alternativa sin cubetas compara las **funciones de distribución empíricas** $F_1$ y $F_2$ de las dos muestras:
#
# $$D = \sup_x \big|F_1(x) - F_2(x)\big|.$$
#
# $D$ va de 0 a 1 y se lee directo: «hay un valor $x$ tal que la proporción de datos por debajo de él difiere en $D$ entre las dos muestras». Su ventaja teórica es que, si las dos muestras vienen de la misma distribución continua, la distribución de $D$ **no depende de cuál sea**: para $n$ y $m$ datos, $\sqrt{nm/(n+m)}\,D$ converge a la distribución de Kolmogorov, y se rechaza la igualdad al 5 % cuando
#
# $$D > 1.358\,\sqrt{\frac{n+m}{nm}}.$$
#
# El supremo solo puede alcanzarse en los datos, así que basta evaluar $F_1 - F_2$ en la muestra combinada y ordenada:

# %%
def ks(a, c):
    a, c = np.sort(a), np.sort(c)
    x = np.concatenate([a, c])
    F1 = np.searchsorted(a, x, side="right") / len(a)
    F2 = np.searchsorted(c, x, side="right") / len(c)
    return np.max(np.abs(F1 - F2))


a, c = b11[b11.mes == 3].temp, registro[registro.mes == 3].temp
print(f"temperatura de marzo: D = {ks(a, c):.4f} (nuestra) · {ks_2samp(a, c).statistic:.4f} (scipy) · p = {ks_2samp(a, c).pvalue:.1e}")
print(f"umbral al 5 % con n = {len(a)} y m = {len(c)}: {1.358 * np.sqrt((len(a) + len(c)) / (len(a) * len(c))):.3f}")

# %% [markdown]
# ### Con muchos datos, todo es «significativo»
#
# La prueba KS responde a la pregunta «¿son **exactamente** iguales las dos distribuciones?», y con suficientes datos la respuesta casi siempre es que no. Lo que nos importa es otra pregunta: ¿el cambio es lo bastante **grande** para afectar al modelo? Simulémoslo: tomamos temperaturas de 2011 y a una de las dos muestras le sumamos 0.5 °C, un cambio que ningún modelo de demanda de bicicletas notaría.

# %%
filas = []
for n in [100, 300, 1000, 3000, 10_000, 30_000, 100_000]:
    res = [ks_2samp(rng.choice(temps, n), rng.choice(temps, n) + 0.5) for _ in range(20)]
    filas.append({"n en cada muestra": n, "D (mediana)": np.median([r.statistic for r in res]),
                  "valor p (mediana)": np.median([r.pvalue for r in res]), "umbral de D al 5 %": 1.358 * np.sqrt(2 / n)})
ks_n = pd.DataFrame(filas).set_index("n en cada muestra")

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
ejes[0].plot(ks_n.index, ks_n["D (mediana)"], marker="o", color=AZUL, lw=2.5)
ejes[0].plot(ks_n.index, ks_n["umbral de D al 5 %"], ls="--", color=ROJO, lw=2)
ejes[0].text(1.5e4, 0.05, "D observado", color=AZUL, fontsize=12)
ejes[0].text(1.2e4, 0.0, "umbral de rechazo", color=ROJO, fontsize=12)
ejes[0].set(xscale="log", xlabel="datos en cada muestra", ylabel="D", title="El tamaño del efecto no cambia…")
ejes[1].plot(ks_n.index, ks_n["valor p (mediana)"], marker="o", color=AZUL, lw=2.5)
ejes[1].axhline(0.05, ls="--", color=ROJO, lw=2)
ejes[1].text(3e4, 1e-6, "p = 0.05", color=ROJO, fontsize=12)
ejes[1].set(xscale="log", yscale="log", ylim=(1e-80, 1e3), yticks=[1e-80, 1e-60, 1e-40, 1e-20, 1], xlabel="datos en cada muestra", ylabel="valor p", title="…pero la significancia, sí")
fig.tight_layout()
plt.show()
ks_n.round(4)

# %% [markdown]
# $D$ se estabiliza en unos 0.04, que es el tamaño real del cambio (con pocos datos sale algo mayor por puro ruido de muestreo, pero queda debajo del umbral). El valor p, en cambio, pasa de 0.37 con 100 datos a alrededor de $10^{-22}$ con 30,000 y $10^{-72}$ con 100,000: el cambio es el mismo, lo único que creció es la muestra. Un servicio con mucho tráfico tendrá valores p minúsculos todos los días. Por eso en monitoreo se reporta y se pone umbral al **tamaño del efecto** ($D$, PSI), no al valor p. Hay un segundo problema: la prueba supone observaciones independientes, y las horas consecutivas no lo son (el clima de las 3 y de las 4 de la tarde es casi el mismo); los valores p de la prueba son todavía más optimistas de lo que deberían.
#
# ### Un reporte por atributo
#
# Juntamos todo en un reporte mensual: PSI y $D$ por atributo, y una alerta cuando el PSI pasa de 0.25.

# %%
def reporte(ref, act, variables, umbral=0.25):
    filas = [{"atributo": v, "PSI": psi(ref[v], act[v]), "D (KS)": ks(ref[v], act[v])} for v in variables]
    rep = pd.DataFrame(filas).set_index("atributo")
    rep["alerta"] = np.where(rep.PSI > umbral, "revisar", "")
    return rep.round(3)


reporte(b11[b11.mes == 3], registro[registro.mes == 3], vigilados)

# %% [markdown]
# # Parte 5. Cuando llegan las etiquetas
#
# ## 10. El error real, mes por mes
#
# Al cerrar cada mes, el operador sabe cuántas bicicletas se rentaron cada hora. Juntamos esas etiquetas con la bitácora y medimos el error que el monitoreo de distribuciones no podía ver.

# %%
registro["total"] = b12.total.to_numpy()     # la bitácora está en el mismo orden que el tráfico enviado
error_fijo = registro.groupby("mes").apply(
    lambda d: pd.Series({"RECM": root_mean_squared_error(d.total, d.prediccion), "sesgo %": 100 * sesgo(d.total, d.prediccion)}),
    include_groups=False)
error_fijo.index = MESES
tablero.loc["rentas observadas"] = [psi(b11[b11.mes == m].total, registro[registro.mes == m].total) for m in range(1, 13)]
print(f"2012 completo: RECM = {root_mean_squared_error(registro.total, registro.prediccion):.1f}, "
      f"sesgo = {100 * sesgo(registro.total, registro.prediccion):+.1f} %   (validación: RECM = {RECM_VAL:.1f})")
error_fijo.round(1).T

# %% [markdown]
# El modelo que en validación tenía un error de 47 bicicletas por hora y sesgo nulo tuvo en 2012 un error de 125 (entre 66 y 164 según el mes), y **subestimó la demanda todos los meses**, entre 28 % y 45 %. Volvamos al tablero, ahora con una fila más: la de las rentas observadas.

# %%
fig, ax = plt.subplots(figsize=(12, 4.4))
im = mapa_psi(ax, tablero, "PSI contra el mismo mes de 2011, con las etiquetas")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 0.1, 0.25, 0.5])
fig.tight_layout()
plt.show()

# %% [markdown]
# La fila de las rentas está encendida casi todo el año, mientras que la de las predicciones sigue tranquila. Esa combinación es la firma de un **cambio de concepto**: las entradas se parecen a las de antes, el modelo dice lo mismo que antes, y la respuesta es otra. Ningún monitor de entradas lo habría detectado; hacía falta la etiqueta.
#
# ¿Cuánto del error es «solo» el nivel? Si multiplicamos todas las predicciones de 2012 por una sola constante, la que hace que su total coincida con el observado, el error baja mucho:

# %%
factor = registro.total.sum() / registro.prediccion.sum()
recm_fijo = root_mean_squared_error(registro.total, registro.prediccion)
recm_escalado = root_mean_squared_error(registro.total, factor * registro.prediccion)
print(f"factor = {factor:.2f} · RECM de 2012: {recm_fijo:.1f} → {recm_escalado:.1f} al reescalar")

# %% [markdown]
# Con un solo número, el error de 2012 baja de 125 a 65: la mayor parte del daño es un cambio de **nivel**, el crecimiento del sistema. La forma de la demanda (horas pico, efecto de la lluvia y la temperatura) sigue siendo, en buena medida, la que el modelo aprendió. Claro que el factor 1.55 lo calculamos con las etiquetas de todo el año, que el 1 de enero no teníamos.

# %% [markdown]
# ## 11. Reentrenar
#
# La respuesta natural es reentrenar con datos recientes. Comparamos cuatro políticas; en todas, el modelo de cada mes de 2012 se entrena al empezar el mes con los datos etiquetados disponibles:
#
# - **sin reentrenar**: el modelo de 2011 todo el año;
# - **ventana de 12 meses**: los últimos 12 meses con etiqueta;
# - **ventana de 3 meses**: solo los últimos 3 meses;
# - **ventana de 3 meses, etiquetas con 3 meses de retraso**: como la anterior, pero las rentas de un mes se conocen tres meses después (en enero de 2012 solo hay etiquetas hasta septiembre de 2011).

# %%
def reentrenado(periodo, ventana, retraso):
    ultimo = periodo - 1 - retraso                 # último mes con etiquetas disponibles
    datos = b[(b.periodo <= ultimo) & (b.periodo > ultimo - ventana)]
    return nuevo_modelo().fit(datos[ATRIBUTOS], datos.total)


politicas = {"ventana de 12 meses": (12, 0), "ventana de 3 meses": (3, 0), "3 meses, etiquetas 3 meses tarde": (3, 3)}
recm = {"sin reentrenar": error_fijo["RECM"].to_numpy()}
for nombre, (ventana, retraso) in politicas.items():
    errores = []
    for mes in range(1, 13):
        d = b12[b12.mes == mes]
        errores.append(root_mean_squared_error(d.total, reentrenado(12 + mes, ventana, retraso).predict(d[ATRIBUTOS])))
    recm[nombre] = np.array(errores)
recm = pd.DataFrame(recm, index=MESES)

fig, ax = plt.subplots(figsize=(11, 4.8))
for (nombre, serie), color, estilo in zip(recm.items(), [GRIS, DORADO_OSC, AZUL, ROJO], ["-", "-", "-", "--"]):
    ax.plot(range(1, 13), serie, marker="o", color=color, lw=2.5, ls=estilo)
ax.axhline(RECM_VAL, color=TENUE, lw=1.2, ls=":")
ax.text(8.6, RECM_VAL - 9, f"error de validación ({RECM_VAL:.0f})", color=TENUE, fontsize=11)
rdp.estilo.etiquetar(ax, 12.3, {k: v.iloc[-1] for k, v in recm.items()}, fontsize=11)
ax.set_xticks(range(1, 13), MESES)
ax.set_xlim(0.6, 15.8)
ax.set_ylim(0, 175)
ax.set_ylabel("RECM del mes (bicicletas por hora)")
ax.set_title("Error de cada mes de 2012 según la política de reentrenamiento")
fig.tight_layout()
plt.show()
recm.mean().round(1).rename("promedio de las RECM mensuales").to_frame().T

# %% [markdown]
# - La **ventana de 12 meses** ayuda (el promedio de las RECM mensuales baja de 122 a 89), pero cada modelo mezcla meses viejos, del nivel de 2011, con nuevos, y sigue subestimando.
# - La **ventana de 3 meses** es la mejor (71) y a partir de abril se acerca al error de validación: los datos recientes traen el nivel nuevo. Marzo es su peor mes: entrenó con diciembre, enero y febrero, y el marzo caluroso le pide temperaturas que esa ventana casi no tiene (aquí sí importa el cambio de covariables: es una extrapolación).
# - Con **etiquetas tres meses tarde**, la ventana corta se vuelve una mala idea (103): en junio se predice con un modelo entrenado con enero, febrero y marzo, de otra estación, y en diciembre con uno de junio a agosto. En mayo, junio y diciembre es peor que no reentrenar.
#
# No hay una política buena para siempre: una ventana larga es estable pero lenta para adaptarse; una corta se adapta pero olvida las estaciones y es frágil cuando las etiquetas llegan tarde. Entre las salidas intermedias están ponderar los datos por antigüedad, reentrenar con todo pero agregar un atributo de tendencia que el modelo pueda usar (cuidado: los árboles no extrapolan), o corregir el nivel con las etiquetas más recientes mientras llegan las demás. Lo que sí es universal es el procedimiento: **medir el error real en cuanto haya etiquetas** y decidir con esos números, no con los de validación.

# %% [markdown]
# ## 12. Cierre: lo que hay que vigilar
#
# Un modelo en producción necesita, como mínimo:
#
# 1. **una versión** en cada respuesta y una bitácora de entradas y predicciones;
# 2. **validación de entradas** en el servicio (rangos, tipos, faltantes), con el mismo preprocesamiento que en el entrenamiento;
# 3. un **tablero de distribuciones** (PSI o KS por atributo y de las predicciones) contra una referencia que respete las estaciones, con umbrales sobre el tamaño del efecto y conscientes del tamaño de muestra;
# 4. el **error real** en cuanto lleguen las etiquetas, porque el cambio de concepto no se ve en las entradas;
# 5. una **política de reentrenamiento** probada contra el pasado, como hicimos aquí, que tome en cuenta el retraso de las etiquetas, y una ficha del modelo que se actualice con cada versión.
#
# Es la lección de Zillow Offers. Durante 2021 Zillow compró miles de casas a precios basados en su modelo, para revenderlas meses después; la «etiqueta» de cada compra, el precio de reventa, llegaba hasta que la casa se vendía. Cuando el mercado se enfrió, la empresa tuvo que reducir en 304 millones de dólares el valor de su inventario en el tercer trimestre y anunció el cierre del negocio y el recorte de cerca de 25 % de su personal (Zillow Group, informe del tercer trimestre de 2021, 2 de noviembre de 2021). Su director lo resumió así: la imprevisibilidad de los precios de las casas «excede por mucho lo que anticipamos».
#
# *Referencias:* H. Fanaee-T y J. Gama, «Event labeling combining ensemble detectors and background knowledge», *Progress in Artificial Intelligence* 2 (2014). J. Quiñonero-Candela, M. Sugiyama, A. Schwaighofer y N. D. Lawrence (eds.), *Dataset Shift in Machine Learning*, MIT Press (2009), cap. 1. J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy y A. Bouchachia, «A survey on concept drift adaptation», *ACM Computing Surveys* 46 (2014). C. Huyen, *Designing Machine Learning Systems*, O'Reilly (2022), caps. 7 a 9. S. Rabanser, S. Günnemann y Z. Lipton, «Failing loudly: an empirical study of methods for detecting dataset shift», *NeurIPS* (2019).

# %% [markdown]
# ## 13. Autoevaluación

# %%
rdp.quiz.mostrar("u08", ids=["u08-p05", "u08-p06", "u08-p07", "u08-p08"])

# %% [markdown]
# ## 14. Ejercicios
#
# 1. **PSI y $\chi^2$.** Demuestra que, para $q_k = p_k + \delta_k$ con $\sum_k \delta_k = 0$, $\text{PSI} = \sum_k \delta_k^2/p_k + O(\delta^3)$. Sugerencia: desarrolla $\ln(1 + \delta_k/p_k)$ a segundo orden y observa que el término de segundo orden de $(q_k - p_k)\ln(q_k/p_k)$ es exactamente $\delta_k^2/p_k$. ¿Qué pasa con el término de tercer orden en la suma de las dos divergencias KL?
# 2. **Umbral según el tamaño de muestra.** Con la aproximación $\text{PSI} \approx \chi^2$, $\frac{NM}{N+M}\,\text{PSI}$ se distribuye aproximadamente como $\chi^2_{K-1}$ cuando no hay cambio. Úsalo para proponer un umbral de PSI que dé 1 % de falsas alarmas al comparar una semana de datos contra un mes de referencia, y verifícalo por simulación con las temperaturas de 2011.
# 3. **Cambio de prevalencia.** Con la libreta de métricas de la Unidad 3 (crédito), simula que la proporción de incumplidos pasa de 22 % a 35 % muestreando con reemplazo cada clase. ¿Qué pasa con la TVP, la TFP y la precisión del mismo modelo, con el mismo umbral? ¿Lo detectaría un PSI sobre las predicciones?
# 4. **Una política mejor.** Diseña una política de reentrenamiento que sea buena con y sin retraso en las etiquetas: por ejemplo, ventana de 12 meses con pesos que decrecen con la antigüedad (`sample_weight`), o el modelo de 12 meses multiplicado por el cociente entre rentas observadas y predichas del último mes etiquetado. Compárala con las cuatro de la sección 11.
# 5. **Pruebas del servicio.** Agrega al esquema `Hora` una validación que rechace peticiones incoherentes, como `laboral = 1` con `dia_semana = 0` (domingo), usando un `model_validator` de Pydantic, y escribe la prueba correspondiente. Verifica con los datos que la regla no rechaza ninguna hora real (cuidado con los feriados entre semana, que tienen `laboral = 0`).
