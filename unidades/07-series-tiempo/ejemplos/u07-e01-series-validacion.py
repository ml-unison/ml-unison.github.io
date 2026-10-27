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
# # El clima de Hermosillo: tendencia, estacionalidad y cómo validar un pronóstico
#
# **Unidad 7 · Ejemplo 1**
#
# Hasta ahora supusimos que los datos eran independientes e idénticamente distribuidos: el orden de los renglones no importaba y podíamos barajarlos para formar pliegos de validación cruzada. Con una **serie de tiempo** el orden es justamente la información: la temperatura de hoy se parece a la de ayer, julio se parece a julio, y lo que queremos es decir algo sobre **días que todavía no ocurren**. En esta libreta vamos a:
#
# 1. separar la temperatura diaria de Hermosillo en sus componentes: **ciclo anual**, variación de cada día (**anomalías**) y **tendencia**, con la **climatología** y la descomposición **STL**;
# 2. medir cuánto recuerda la serie su pasado con la **autocorrelación**, y definir **ruido blanco**, **estacionariedad** y **caminata aleatoria**;
# 3. preguntarnos con cuidado si Hermosillo se está calentando, y ver que la autocorrelación cambia la respuesta;
# 4. construir los **pronósticos de referencia** (ingenuo, estacional ingenuo, climatología, deriva) y las métricas para compararlos (MAE, RECM, MAPE, MASE);
# 5. ver con números por qué la **validación cruzada aleatoria** engaña con series de tiempo, y validar hacia adelante con **origen móvil**.

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
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import rdp

rdp.estilo.usar()
SEMILLA = 2027
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# %% [markdown]
# ## 1. Los datos y el problema
#
# **La pregunta.** ¿Qué temperatura hará mañana en Hermosillo? Antes de seguir leyendo, piensa qué responderías sin ningún modelo. Hay dos respuestas sencillas: «la de hoy» y «la normal para esta fecha». Veremos que las dos son sorprendentemente difíciles de vencer, y que para saber si un modelo las vence hay que medir con cuidado.
#
# **Los datos.** La red mundial de estaciones climatológicas diarias (*Global Historical Climatology Network – Daily*, GHCN-D; Menne et al., *Journal of Atmospheric and Oceanic Technology*, 2012) reúne las observaciones de decenas de miles de estaciones y las publica la NOAA de los Estados Unidos, en el dominio público. Tomamos la estación del aeropuerto internacional de Hermosillo (MXM00076160), de 1992 a 2024. Por día tenemos la temperatura media `tmed` (°C), casi completa; `tmax`, `tmin` y `lluvia` tienen muchos faltantes y no las usaremos.
#
# **El problema.** Con la temperatura media diaria hasta hoy, pronosticar la de los próximos días (horizonte $h = 1, 2, \dots$), y **medir honestamente** qué tan bien lo hacemos. Usaremos 1992–2020 para entrenar y guardaremos 2021–2024 como el futuro que no se toca hasta el final.
#
# **Notación.** La serie es $y_1, \dots, y_T$; el pronóstico de $y_{T+h}$ hecho con la información hasta $T$ se escribe $\hat y_{T+h\mid T}$, y $h$ es el **horizonte**.

# %%
datos = rdp.datos.cargar("temperatura_hermosillo", parse_dates=["fecha"])
print(f"{len(datos):,} días registrados, de {datos.fecha.min():%d/%m/%Y} a {datos.fecha.max():%d/%m/%Y}")

# El archivo omite los días sin dato: reindexamos a frecuencia diaria para verlos.
y_crudo = datos.set_index("fecha")["tmed"].asfreq("D")
faltan = y_crudo.isna()
print(f"{len(y_crudo):,} días en el periodo; faltan {faltan.sum()}")
print("faltantes por año:", faltan.groupby(faltan.index.year).sum().loc[lambda s: s > 0].to_dict())

# %% [markdown]
# Lo primero con una serie de tiempo es **reindexar a su frecuencia**: el archivo trae 11,972 renglones, pero el periodo tiene 12,054 días. Si no lo hiciéramos, «el renglón anterior» no siempre sería «el día anterior», y todos los rezagos que calculemos estarían mal sin avisar. Casi todos los faltantes están en 1994 (59 días, en varios tramos). Como son pocos (0.7 %), los rellenamos por **interpolación lineal** entre el día anterior y el siguiente con dato. Es una decisión: suaviza un poco esos días, pero no cambia nada de lo que vamos a medir.

# %%
y = y_crudo.interpolate()
ENTRENA, PRUEBA = slice("1992", "2020"), slice("2021", "2024")

fig, ejes = plt.subplots(2, 1, figsize=(12, 7.2))
ejes[0].plot(y.index, y, lw=0.4, color=AZUL)
ejes[0].set_title("Temperatura media diaria en el aeropuerto de Hermosillo, 1992–2024")
ejes[0].set_ylabel("°C")
ejes[0].axvspan(pd.Timestamp("2021-01-01"), y.index[-1], color=rdp.estilo.PALETA["azul-tenue"], alpha=0.5, lw=0)
ejes[0].text(pd.Timestamp("2021-03-01"), 7.5, "prueba", color=TENUE, fontsize=11)
anio = y["2023"]
ejes[1].plot(anio.index, anio, lw=1.2, color=AZUL)
ejes[1].set_title("Un año de cerca: 2023")
ejes[1].set_ylabel("°C")
fig.tight_layout()
plt.show()

# %% [markdown]
# Se ven tres cosas a distintas escalas de tiempo:
#
# - un **ciclo anual** enorme: de unos 16 °C en enero a unos 34 °C en julio;
# - una **variación de día a día**, de un par de grados, montada sobre el ciclo: frentes fríos en invierno, días nublados o con lluvia en el verano del monzón;
# - quizá cambios **de un año a otro** o de largo plazo (2014 y 2020 se ven más calientes), que habrá que medir con cuidado.
#
# Ésas son las componentes clásicas de una serie: **tendencia** (cambios lentos del nivel), **estacionalidad** (un patrón que se repite con periodo fijo $m$; aquí $m = 365.25$ días), **ciclo** (oscilaciones sin periodo fijo, como las de El Niño) y **ruido** (lo que queda).

# %% [markdown]
# ## 2. El ciclo anual: climatología y anomalías
#
# ### La climatología
#
# La «temperatura normal para esta fecha» es la **climatología**: el promedio de la temperatura de cada día del año en los años de entrenamiento. Calculada tal cual, con solo 29 valores por día, sale ruidosa. Como el ciclo anual es suave, conviene ajustarlo con una **regresión lineal sobre armónicos** (Unidad 4): con $\omega = 2\pi/365.25$ y $d$ el día del año,
#
# $$c(d) = \beta_0 + \sum_{k=1}^{K}\big[a_k\cos(k\,\omega d) + b_k \sin(k\,\omega d)\big].$$
#
# Con $K = 3$ son 7 parámetros. Los estimamos **solo con 1992–2020**, para poder usar la climatología como pronóstico de 2021–2024 sin hacer trampa.

# %%
def armonicos(fechas, K=3):
    """Matriz de diseño [1, cos(kωd), sen(kωd)] para k = 1..K, con d el día del año."""
    t = 2 * np.pi * (fechas.day_of_year - 1) / 365.25
    columnas = [np.ones(len(fechas))]
    for k in range(1, K + 1):
        columnas += [np.cos(k * t), np.sin(k * t)]
    return np.column_stack(columnas)


y_tr = y[ENTRENA]
beta, *_ = np.linalg.lstsq(armonicos(y_tr.index), y_tr.to_numpy(), rcond=None)
clim = pd.Series(armonicos(y.index) @ beta, index=y.index)          # climatología suave
clim_cruda = y_tr.groupby(y_tr.index.day_of_year).mean()             # promedio de cada día del año

print(f"climatología: mínimo {clim.min():.1f} °C (día {clim.idxmin().day_of_year}), "
      f"máximo {clim.max():.1f} °C (día {clim.idxmax().day_of_year}); amplitud {clim.max() - clim.min():.1f} °C")

# %%
d_tr = y_tr.index.day_of_year
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.scatter(d_tr, y_tr, s=2, color=GRIS, alpha=0.15, edgecolors="none")
ax.plot(clim_cruda.index, clim_cruda, color=DORADO_OSC, lw=1.2)
dias = pd.date_range("2001-01-01", "2001-12-31")
ax.plot(dias.day_of_year, armonicos(dias) @ beta, color=AZUL, lw=3)
ax.set_xlabel("día del año")
ax.set_ylabel("°C")
ax.set_title("Climatología de Hermosillo, 1992–2020")
ax.set_xlim(1, 380)
rdp.estilo.etiquetar(ax, 368, {"promedio\nde cada día": clim_cruda.iloc[-1] - 3.5, "armónicos\n(K = 3)": float((armonicos(dias[-1:]) @ beta)[0]) + 3.5})
plt.show()

# %% [markdown]
# Cada punto gris es un día de 1992–2020. El promedio crudo de cada día del año (dorado) se mueve unos décimos de grado de un día al siguiente por puro azar; los armónicos (azul) capturan la forma: un mínimo de 16.4 °C a principios de enero y un máximo de 34.0 °C a principios de julio, 17.6 °C de amplitud. La curva no es una senoide: sube rápido en primavera, se aplana en el verano del monzón y baja en otoño, y eso lo dan el segundo y el tercer armónico.
#
# ### Las anomalías
#
# Restando la climatología queda la **anomalía**, $a_t = y_t - c(d_t)$: cuánto más caliente o más fresco que lo normal estuvo cada día. Es la parte «interesante» para pronosticar a corto plazo, porque el ciclo anual ya lo conocemos.

# %%
anom = y - clim
print(f"desviación estándar: de la serie {y[ENTRENA].std():.2f} °C, de las anomalías {anom[ENTRENA].std():.2f} °C")
print(f"la climatología explica el {1 - anom[ENTRENA].var() / y[ENTRENA].var():.1%} de la varianza diaria")

# %% [markdown]
# La climatología sola explica el 82 % de la varianza de la temperatura diaria. Lo que queda, las anomalías, tiene una desviación estándar de 2.8 °C.
#
# ### Descomposición STL
#
# Una forma más general de separar componentes es **STL** (*Seasonal-Trend decomposition using Loess*; R. B. Cleveland, W. S. Cleveland, J. E. McRae e I. Terpenning, *Journal of Official Statistics*, 1990). Escribe
#
# $$y_t = T_t + S_t + R_t$$
#
# (tendencia + estacionalidad + residuo) y estima las componentes alternando suavizamientos locales (*loess*): suaviza cada «subserie» de días equivalentes de distintos años para obtener $S_t$, suaviza lo que queda para obtener $T_t$, y repite. A diferencia de los armónicos, deja que la forma del ciclo anual cambie lentamente de un año a otro. Con 33 años de datos diarios y periodo 365 tarda un par de segundos.

# %%
stl = STL(y, period=365).fit()
fig, ejes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
for ax, serie, nombre, color in zip(ejes, [y, stl.trend, stl.seasonal, stl.resid],
                                    ["serie", "tendencia", "estacionalidad", "residuo"], [AZUL, ROJO, DORADO_OSC, GRIS]):
    ax.plot(serie.index, serie, lw=0.5 if nombre in ("serie", "residuo") else 1.6, color=color)
    ax.set_ylabel("°C")
    ax.set_title(nombre, fontsize=12)
fig.suptitle("Descomposición STL de la temperatura de Hermosillo", x=0.07, ha="left", color=rdp.estilo.MARINO, fontweight="semibold")
fig.tight_layout()
plt.show()
print(f"rango de la tendencia STL: {stl.trend.min():.1f} a {stl.trend.max():.1f} °C; "
      f"desviación estándar del residuo: {stl.resid.std():.2f} °C")

# %% [markdown]
# La tendencia de STL se mueve entre 24.3 y 28.0 °C: baja a mediados de la década de 2000, sube en 2014 y en 2020. ¿Es eso «calentamiento»? Antes de responder necesitamos una herramienta: medir cuánto se parece la serie a sí misma en el tiempo.

# %% [markdown]
# ## 3. Autocorrelación
#
# ### Definición
#
# La **autocovarianza muestral** con rezago $k$ y la **autocorrelación muestral** (ACF) son
#
# $$\hat\gamma(k) = \frac{1}{T}\sum_{t=k+1}^{T}(y_t - \bar y)(y_{t-k} - \bar y), \qquad \hat\rho(k) = \frac{\hat\gamma(k)}{\hat\gamma(0)} .$$
#
# Es la correlación entre la serie y la misma serie recorrida $k$ pasos. Dos detalles: se divide entre $T$ y no entre $T - k$ (así la matriz de autocovarianzas es semidefinida positiva, y los rezagos grandes, con pocos pares, se encogen hacia cero), y se usa la media global $\bar y$. Programarla son tres líneas; comparemos con `statsmodels`:

# %%
def acf_muestral(x, rezagos):
    x = np.asarray(x, dtype=float) - np.mean(x)
    g0 = x @ x / len(x)
    return np.array([x[k:] @ x[:len(x) - k] / len(x) / g0 for k in range(rezagos + 1)])


a_tr = anom[ENTRENA].to_numpy()
print("a mano:     ", acf_muestral(a_tr, 5).round(4))
print("statsmodels:", acf(a_tr, nlags=5).round(4))

# %% [markdown]
# ### La serie cruda contra las anomalías

# %%
T = len(a_tr)
banda = 1.96 / np.sqrt(T)
acf_y = acf(y[ENTRENA], nlags=730)
acf_a = acf(a_tr, nlags=730)

fig, ejes = plt.subplots(1, 2, figsize=(13, 4.4))
ejes[0].plot(np.arange(731), acf_y, color=AZUL)
ejes[0].set_title("Temperatura: la ACF oscila con el año")
ejes[1].vlines(np.arange(31), 0, acf_a[:31], color=ROJO, lw=2.5)
ejes[1].plot(np.arange(31), acf_a[:31], "o", color=ROJO, ms=5)
ejes[1].set_title("Anomalías: la memoria dura unos días")
for ax in ejes:
    ax.axhspan(-banda, banda, color=rdp.estilo.PALETA["azul-tenue"], alpha=0.6, lw=0)
    ax.axhline(0, color=TENUE, lw=0.8)
    ax.set_xlabel("rezago $k$ (días)")
ejes[0].set_ylabel(r"$\hat\rho(k)$")
ejes[0].set_xticks([0, 182, 365, 547, 730])
fig.tight_layout()
plt.show()
print(f"banda ±1.96/√T = ±{banda:.3f} con T = {T:,}")
print("ACF de la temperatura en k = 1, 91, 182, 365:", acf_y[[1, 91, 182, 365]].round(2))
print("ACF de las anomalías en k = 1, 2, 3, 7, 14, 30:", acf_a[[1, 2, 3, 7, 14, 30]].round(2))

# %% [markdown]
# A la izquierda, la ACF de la temperatura cruda está dominada por el ciclo anual: vale 0.96 a un día, 0 a tres meses (primavera contra verano), −0.79 a medio año (enero contra julio) y vuelve a 0.79 a un año. Eso no es «memoria» del clima: es que julio siempre es caliente. **La estacionalidad aparece en la ACF como una oscilación con el periodo $m$**, y tapa todo lo demás.
#
# A la derecha, sin el ciclo anual, queda la memoria de verdad: la anomalía de hoy tiene correlación 0.79 con la de ayer, 0.58 con la de anteayer, 0.21 con la de hace una semana. Un frente frío dura unos días. Pero la ACF no llega a cero: a 30 días todavía vale 0.12, muy por encima de la banda. Hay meses enteros más calientes o más fríos que lo normal, y años también (lo veremos en la sección 4).
#
# ### Ruido blanco y la banda $\pm 1.96/\sqrt{T}$
#
# Una serie es **ruido blanco** si sus valores tienen media cero, varianza constante $\sigma^2$ y no están correlacionados entre sí: $\operatorname{Cov}(\varepsilon_t, \varepsilon_s) = 0$ para $t \neq s$. Es el caso «sin memoria», el que no se puede pronosticar mejor que con su media. Si $y$ es ruido blanco (con cuarto momento finito), para cada $k \geq 1$ fijo se cumple que $\hat\rho(k)$ es aproximadamente $\mathcal N(0, 1/T)$ cuando $T$ es grande. Por eso se dibuja la banda $\pm 1.96/\sqrt{T}$: un ruido blanco deja el 95 % de sus $\hat\rho(k)$ adentro. Con $T = 10,593$ la banda es de $\pm 0.019$, casi invisible.
#
# Ojo: es una banda **para cada rezago por separado**. Con 40 rezagos, es normal que dos se salgan aunque la serie sea ruido blanco. Para probar muchos rezagos a la vez está la prueba de Ljung–Box, que veremos con los modelos ARIMA.

# %%
rng = np.random.default_rng(SEMILLA)
ruido = rng.normal(size=T)
r_ruido = acf(ruido, nlags=40)[1:]
print(f"ruido blanco simulado: {np.sum(np.abs(r_ruido) > banda)} de 40 rezagos fuera de la banda; "
      f"máximo |ρ̂| = {np.abs(r_ruido).max():.3f}")

# %% [markdown]
# ### Estacionariedad débil
#
# Toda la teoría de la ACF supone que «la correlación entre $y_t$ y $y_{t-k}$» depende de $k$ y no de $t$. Ésa es la definición:
#
# > Una serie $\{y_t\}$ es **débilmente estacionaria** (o estacionaria de segundo orden) si $\mathbb{E}[y_t] = \mu$ no depende de $t$, $\operatorname{Var}(y_t) < \infty$, y $\operatorname{Cov}(y_t, y_{t-k}) = \gamma(k)$ depende solo del rezago $k$.
#
# La temperatura cruda **no** es estacionaria: su media cambia con la fecha (16 °C en enero, 34 °C en julio). Por eso su ACF muestral no estima nada que tenga sentido: mezcla en una sola cifra pares de días de estaciones distintas. Las anomalías se parecen mucho más a una serie estacionaria; aunque su varianza también cambia algo con la estación (los frentes fríos del invierno dan anomalías más grandes que los días del verano), es la serie sobre la que tiene sentido hablar de memoria. Quitar la estacionalidad y la tendencia para quedarse con algo estacionario es el primer paso de casi cualquier modelo de series de tiempo.
#
# ### Un ejemplo no estacionario: la caminata aleatoria
#
# Sea $\varepsilon_t$ ruido blanco con varianza $\sigma^2$ y $y_t = y_{t-1} + \varepsilon_t$ con $y_0 = 0$. Entonces $y_t = \varepsilon_1 + \dots + \varepsilon_t$, así que
#
# $$\mathbb{E}[y_t] = 0, \qquad \operatorname{Var}(y_t) = \sum_{s=1}^{t}\operatorname{Var}(\varepsilon_s) = t\,\sigma^2 ,$$
#
# porque los $\varepsilon_s$ no están correlacionados. La media es constante, pero la varianza crece con $t$: **no es estacionaria**. Además, $\operatorname{Cov}(y_t, y_{t-k}) = (t-k)\sigma^2$ depende de $t$, y la correlación $\sqrt{(t-k)/t}$ tiende a 1: la caminata «recuerda» todo. Su ACF muestral decae lentísimo, casi en línea recta, y eso es una señal típica de no estacionariedad. La **diferencia** $y_t - y_{t-1} = \varepsilon_t$ sí es estacionaria: diferenciar es la otra forma clásica de quitar tendencias.

# %%
n_caminos, largo = 2000, 500
caminos = rng.normal(size=(n_caminos, largo)).cumsum(axis=1)
var_emp = caminos.var(axis=0)

fig, ejes = plt.subplots(1, 3, figsize=(14, 4.2))
tt = np.arange(1, largo + 1)
for c in caminos[:25]:
    ejes[0].plot(tt, c, lw=0.8, color=AZUL, alpha=0.5)
ejes[0].plot(tt, 1.96 * np.sqrt(tt), color=ROJO, lw=2)
ejes[0].plot(tt, -1.96 * np.sqrt(tt), color=ROJO, lw=2)
ejes[0].set_title("25 caminatas y ±1.96√t")
ejes[0].set_xlabel("$t$")
ejes[1].plot(tt, var_emp, color=AZUL, label="varianza de 2,000 caminatas")
ejes[1].plot(tt, tt, color=ROJO, ls="--", label=r"$t\sigma^2$")
ejes[1].set_title("La varianza crece como $t\\sigma^2$")
ejes[1].set_xlabel("$t$")
ejes[1].legend(loc="upper left")
ejes[2].plot(np.arange(101), acf(caminos[0], nlags=100), color=AZUL)
ejes[2].plot(np.arange(101), acf(np.diff(caminos[0]), nlags=100), color=ROJO)
ejes[2].axhspan(-1.96 / np.sqrt(largo), 1.96 / np.sqrt(largo), color=rdp.estilo.PALETA["azul-tenue"], alpha=0.6, lw=0)
ejes[2].set_title("ACF de una caminata y de su diferencia")
ejes[2].set_xlabel("rezago $k$")
ejes[2].text(30, 0.7, "caminata", color=AZUL, fontsize=12)
ejes[2].text(30, 0.14, "diferencia", color=ROJO, fontsize=12)
fig.tight_layout()
plt.show()
print(f"varianza empírica en t = 100, 250, 500: {var_emp[[99, 249, 499]].round(1)}  (teórica: 100, 250, 500)")

# %% [markdown]
# ## 4. ¿Se está calentando Hermosillo?
#
# Es una pregunta natural con estos datos, y una buena ocasión para ver por qué la autocorrelación importa. Promediemos las anomalías de cada año (respecto a la climatología de 1992–2020) y ajustemos una recta:
#
# $$\bar a_{\text{año}} = \alpha + \beta\cdot\text{(décadas desde 1992)} + e .$$

# %%
anual = anom.groupby(anom.index.year).mean()
decadas_anual = (anual.index.to_numpy() + 0.5 - 1992) / 10
ols_anual = sm.OLS(anual.to_numpy(), sm.add_constant(decadas_anual)).fit()
b, ee = ols_anual.params[1], ols_anual.bse[1]
lo, hi = ols_anual.conf_int()[1]
print(f"pendiente con promedios anuales: {b:+.2f} °C por década, error estándar {ee:.2f}; IC 95 %: [{lo:+.2f}, {hi:+.2f}]")
print(f"anomalía anual: mínima {anual.min():+.2f} °C ({anual.idxmin()}), máxima {anual.max():+.2f} °C ({anual.idxmax()}); "
      f"desviación estándar {anual.std():.2f} °C")

# %%
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.bar(anual.index, anual, color=np.where(anual > 0, ROJO, AZUL), width=0.75)
malla = np.linspace(anual.index.min() - 0.5, anual.index.max() + 0.5, 50)
Xm = sm.add_constant((malla + 0.5 - 1992) / 10)
pr = ols_anual.get_prediction(Xm).summary_frame(alpha=0.05)
ax.fill_between(malla, pr.mean_ci_lower, pr.mean_ci_upper, color=GRIS, alpha=0.2, lw=0)
ax.plot(malla, pr["mean"], color=TINTA, lw=2)
ax.axhline(0, color=TENUE, lw=0.8)
ax.set_ylabel("°C respecto a 1992–2020")
ax.set_title(f"Anomalía media anual y su recta: {b:+.2f} ± {1.96 * ee:.2f} °C por década".replace("-", "−"))
plt.show()

# %% [markdown]
# La pendiente es de −0.05 °C por década, con un intervalo de 95 % que va de −0.36 a +0.26: **con esta estación y estos 33 años no podemos distinguir la tendencia de cero.** Los años varían mucho (de −1.21 °C en 2006 a +2.08 °C en 2014) y además vienen en rachas: a mediados de la década de 2000 hubo cinco años seguidos alrededor de −1 °C.
#
# ### Lo que pasa si ignoramos la autocorrelación
#
# Supongamos que, en vez de promediar por año, ajustamos la recta a las 12,054 anomalías diarias con mínimos cuadrados ordinarios. La pendiente es casi la misma, pero el error estándar que reporta OLS supone residuos independientes, y aquí están fuertemente correlacionados. Un remedio estándar es el error estándar **robusto a heterocedasticidad y autocorrelación** (HAC, de Newey y West, 1987), que estima la varianza del estimador sumando las covarianzas de los residuos hasta cierto rezago (aquí, un año):

# %%
decadas_diario = ((y.index.year + (y.index.day_of_year - 1) / 365.25) - 1992) / 10
X_diario = sm.add_constant(decadas_diario.to_numpy())
ols_diario = sm.OLS(anom.to_numpy(), X_diario).fit()
hac_diario = sm.OLS(anom.to_numpy(), X_diario).fit(cov_type="HAC", cov_kwds={"maxlags": 365})
tabla_tend = pd.DataFrame({
    "pendiente (°C/década)": [ols_diario.params[1], hac_diario.params[1], b],
    "error estándar": [ols_diario.bse[1], hac_diario.bse[1], ee],
}, index=["diaria, OLS ingenuo", "diaria, HAC (Newey–West)", "promedios anuales"])
tabla_tend["t"] = tabla_tend.iloc[:, 0] / tabla_tend.iloc[:, 1]
tabla_tend.round(3)

# %% [markdown]
# El error estándar ingenuo (0.027) es **5.5 veces menor** que el correcto (0.149). Con él tendríamos $t \approx -1.9$, casi «significativo»: estaríamos a punto de anunciar que Hermosillo se enfría. La razón es que OLS cree tener 12,054 observaciones independientes, cuando por la memoria de la serie la información equivale a muchísimas menos. Los promedios anuales (33 datos, casi independientes) y el HAC coinciden: no hay tendencia detectable.
#
# **La regla:** con datos autocorrelacionados, las fórmulas de varianza que suponen independencia son optimistas. Lo mismo nos va a pasar con la validación cruzada en la sección 7.
#
# **Con cautela.** Este resultado no contradice el calentamiento documentado en el noroeste de México: dice que una sola estación con 33 años de datos y una variación interanual de ±1 °C no alcanza para medir una tendencia de unas décimas de grado por década. Además, una estación de aeropuerto puede cambiar de instrumentos, de ubicación o de entorno (pistas, urbanización), y esos cambios producen saltos que no son clima; la racha fría de 2004–2008 podría ser, en parte, un caso así. Los estudios de tendencias climáticas usan muchas estaciones y series **homogeneizadas**, corregidas por esos cambios.

# %% [markdown]
# ## 5. Pronósticos de referencia
#
# Antes de cualquier modelo hay que saber cuánto se equivoca lo más simple. Con información hasta el día $T$:
#
# | método | pronóstico $\hat y_{T+h\mid T}$ | idea |
# |---|---|---|
# | ingenuo | $y_T$ | mañana igual que hoy |
# | estacional ingenuo | $y_{T+h-m}$ (aquí $m = 365$) | igual que la misma fecha del año pasado |
# | climatología | $c(d_{T+h})$ | lo normal para esa fecha |
# | deriva | $y_T + h\,\dfrac{y_T - y_1}{T - 1}$ | seguir la recta entre el primer y el último dato |
#
# La **deriva** tiene sentido para series con tendencia, como la población o el CO₂; con la temperatura, la pendiente entre el primer y el último día es minúscula y el pronóstico es prácticamente el ingenuo, así que no la usaremos. Agreguemos en cambio una combinación que sugieren las dos gráficas de la ACF: empezar en la anomalía de hoy y **dejar que se desvanezca** hacia la climatología al ritmo de la autocorrelación,
#
# $$\hat y_{T+h\mid T} = c(d_{T+h}) + \hat\phi^{\,h}\,a_T, \qquad \hat\phi = \hat\rho_a(1) .$$
#
# Es un modelo autorregresivo de orden 1 sobre las anomalías, el AR(1), que estudiaremos a fondo en la siguiente presentación.
#
# ### Origen móvil
#
# Para evaluar en 2021–2024 usamos un **origen móvil**: cada día $T$ del periodo de prueba es un origen desde el que se pronostica $T + h$, usando solo datos hasta $T$. Lo que se estima (la climatología y $\hat\phi$) lo re-ajustamos cada 1 de enero con todos los años anteriores (**ventana creciente**). Así cada pronóstico se hace exactamente con la información que se tendría en ese momento.

# %%
def ajustar_referencia(hasta):
    """Climatología (armónicos) y φ̂ = ρ̂(1) de las anomalías, con los datos anteriores a `hasta`."""
    previo = y[: pd.Timestamp(hasta) - pd.Timedelta(days=1)]
    coef, *_ = np.linalg.lstsq(armonicos(previo.index), previo.to_numpy(), rcond=None)
    a = previo.to_numpy() - armonicos(previo.index) @ coef
    return coef, acf(a, nlags=1)[1]


def pronosticos(h):
    """Pronósticos de cada día de 2021–2024 con origen h días antes, re-ajustando cada 1 de enero."""
    filas = []
    for anio_ in range(2021, 2025):
        coef, phi = ajustar_referencia(f"{anio_}-01-01")
        dias_ = y[str(anio_)].index
        c = armonicos(dias_) @ coef
        origen = dias_ - pd.Timedelta(days=h)
        y_origen = y.reindex(origen).to_numpy()
        a_origen = y_origen - armonicos(origen) @ coef
        filas.append(pd.DataFrame({
            "real": y[str(anio_)].to_numpy(),
            "ingenuo": y_origen,
            "estacional ingenuo": y.reindex(dias_ - pd.Timedelta(days=365)).to_numpy(),
            "climatología": c,
            "AR(1) de anomalías": c + phi**h * a_origen,
        }, index=dias_))
    return pd.concat(filas)


coef_2021, phi_2021 = ajustar_referencia("2021-01-01")
print(f"φ̂ con 1992–2020: {phi_2021:.3f}")
p1 = pronosticos(1)
p1.head()

# %% [markdown]
# Veamos cómo se ven: desde un origen fijo, el 15 de enero de 2023, pronosticamos los siguientes 30 días.

# %%
origen = pd.Timestamp("2023-01-15")
coef, phi = ajustar_referencia("2023-01-01")
futuro = pd.date_range(origen + pd.Timedelta(days=1), periods=30)
hs = np.arange(1, 31)
a_T = y[origen] - (armonicos(pd.DatetimeIndex([origen])) @ coef)[0]
c_fut = armonicos(futuro) @ coef
caminos_ref = {
    "ingenuo": np.full(30, y[origen]),
    "estacional ingenuo": y.reindex(futuro - pd.Timedelta(days=365)).to_numpy(),
    "climatología": c_fut,
    "AR(1)": c_fut + phi**hs * a_T,
}
pasado = y[origen - pd.Timedelta(days=45): origen]
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.plot(pasado.index, pasado, color=TINTA, lw=1.8)
ax.plot(futuro, y[futuro], color=TINTA, lw=1.2, ls=":", marker="o", ms=3)
for (nombre, v), color in zip(caminos_ref.items(), [ROJO, GRIS, AZUL, DORADO_OSC]):
    ax.plot(futuro, v, color=color, lw=2.2, label=nombre)
ax.axvline(origen, color=TENUE, lw=1, ls="--")
ax.text(origen, 23.8, " origen $T$", color=TENUE, fontsize=11, va="top")
ax.set_ylabel("°C")
ax.set_title("Pronósticos de referencia desde el 15 de enero de 2023 (puntos: lo que pasó)")
ax.legend(loc="lower left", ncols=2)
plt.show()

# %% [markdown]
# El ingenuo se queda en la temperatura del origen; la climatología ignora el origen y sigue lo normal de enero; el AR(1) arranca en el ingenuo y en unos días se funde con la climatología; el estacional ingenuo repite el año anterior, con todo su ruido.
#
# ## 6. Métricas de error
#
# Con errores $e_t = y_t - \hat y_t$ en un conjunto de $n$ pronósticos:
#
# - **MAE** (error absoluto medio) $= \frac1n\sum|e_t|$, en °C; lo minimiza la mediana.
# - **RECM** (raíz del error cuadrático medio) $= \sqrt{\frac1n\sum e_t^2}$, en °C; lo minimiza la media y castiga más los errores grandes.
# - **MAPE** (error porcentual absoluto medio) $= \frac{100}{n}\sum|e_t / y_t|$. Parece cómodo porque no tiene unidades, pero explota cuando $y_t$ se acerca a cero y **depende del origen de la escala**: no es lo mismo en °C que en kelvin, aunque el pronóstico sea idéntico.
# - **MASE** (error absoluto medio escalado; R. J. Hyndman y A. B. Koehler, *International Journal of Forecasting*, 2006) $= \text{MAE} / q$, donde $q$ es el MAE **dentro de la muestra de entrenamiento** de un pronóstico ingenuo, aquí el estacional ingenuo: $q = \frac{1}{T-m}\sum_{t=m+1}^{T}|y_t - y_{t-m}|$. MASE $< 1$ quiere decir «mejor que el ingenuo estacional dentro de la muestra». No tiene unidades, no depende del origen de la escala y no explota cerca de cero.

# %%
def metricas(real, pred, escala):
    e = real - pred
    return {"MAE": np.mean(np.abs(e)), "RECM": np.sqrt(np.mean(e**2)),
            "MAPE °C (%)": 100 * np.mean(np.abs(e / real)),
            "MAPE K (%)": 100 * np.mean(np.abs(e / (real + 273.15))),
            "MASE": np.mean(np.abs(e)) / escala}


v_tr = y[ENTRENA].to_numpy()
q = np.mean(np.abs(v_tr[365:] - v_tr[:-365]))
print(f"escala de MASE: MAE del estacional ingenuo en 1992–2020 = {q:.2f} °C")
metodos = ["ingenuo", "estacional ingenuo", "climatología", "AR(1) de anomalías"]
tabla_h1 = pd.DataFrame({m: metricas(p1["real"], p1[m], q) for m in metodos}).T
tabla_h1.round(3)

# %% [markdown]
# Para mañana ($h = 1$), en los 1,461 días de 2021–2024:
#
# - el **ingenuo** se equivoca en promedio 1.28 °C, mucho mejor que la **climatología** (2.17 °C): el tiempo de hoy dice más de mañana que lo normal para la fecha;
# - el **estacional ingenuo** es el peor (3.20 °C): un solo año pasado trae todo el ruido de ese año. Su MASE es cercano a 1 por construcción;
# - el **AR(1)** combina lo mejor de los dos y gana en todas las métricas, aunque por poco a un día.
#
# Las columnas de MAPE muestran su defecto: los mismos pronósticos del ingenuo tienen MAPE de 5.5 % en °C y de 0.43 % en kelvin. Ninguno de los dos números dice nada por sí solo; el MAE y el MASE no cambian.
#
# ### El error contra el horizonte

# %%
horizontes = np.arange(1, 31)
mae_h = {}
for h in horizontes:
    p_h = pronosticos(h)
    mae_h[h] = p_h[metodos].rsub(p_h["real"], axis=0).abs().mean()
mae_h = pd.DataFrame(mae_h).T
fig, ax = plt.subplots(figsize=(11, 4.6))
for m, color in zip(metodos, [ROJO, GRIS, AZUL, DORADO_OSC]):
    ax.plot(horizontes, mae_h[m], color=color, marker="o", ms=4, label=m)
ax.set_xlabel("horizonte $h$ (días)")
ax.set_ylabel("MAE en 2021–2024 (°C)")
ax.set_title("¿Qué tan lejos sirve cada referencia?")
ax.set_xlim(0, 31)
ax.set_ylim(0, 4.2)
ax.legend(loc="lower right", ncols=2)
plt.show()
mae_h.loc[[1, 2, 3, 4, 7, 14, 30]].round(2)

# %% [markdown]
# El ingenuo es excelente a un día y empeora rápido: en $h = 3$ empata con la climatología (2.18 contra 2.17 °C) y de ahí en adelante es peor que ella, que no depende del horizonte. El AR(1) es el mejor en todos los horizontes, y a partir de una o dos semanas es indistinguible de la climatología: más allá de eso, **la mejor información sobre la temperatura de un día es la fecha**. Ése es el límite de lo que se puede pronosticar con la serie misma; para ir más lejos hacen falta otras variables (los modelos meteorológicos usan el estado de toda la atmósfera).
#
# Moraleja: cualquier modelo para la temperatura diaria tiene que reportarse **por horizonte** y compararse contra el ingenuo y la climatología. Un MAE de 2 °C es excelente a diez días y malo a uno.

# %% [markdown]
# ## 7. Validación: por qué no se baraja el tiempo
#
# ### El problema
#
# En la Unidad 3 estimamos el error fuera de muestra con validación cruzada de $K$ pliegos (*K-fold*): se barajan los renglones, se parten en $K$ grupos y cada grupo se predice con un modelo entrenado en los demás. El supuesto es que los renglones son independientes. Con una serie, si barajamos, el día $t$ queda en el pliego de prueba y los días $t-1$ y $t+1$ en el de entrenamiento. Y sabemos que la anomalía de $t$ tiene correlación 0.79 con la de sus vecinos: **el modelo puede «ver» la respuesta de prueba a través de sus vecinos**. Es una fuga del futuro (Unidad 1).
#
# ### La demostración
#
# Planteemos un problema realista: pronosticar la temperatura **a una semana** ($h = 7$). Para el día $t$ construimos atributos con la información disponible el día $t - 7$:
#
# - las temperaturas de $t-7$ y $t-8$ (rezagos),
# - la media de los 7 y de los 30 días que terminan en $t-7$ (estadísticas móviles),
# - el día del año (como $\cos$ y $\sin$) y el año, para que el modelo pueda capturar el ciclo y una posible tendencia.
#
# Ningún atributo usa información posterior a $t - 7$: la matriz no tiene fuga. Compararemos una regresión lineal y un bosque aleatorio (Unidad 5), un modelo flexible que puede aproximar funciones muy irregulares. Estimaremos su error con dos esquemas de validación sobre 1992–2020 y luego mediremos el error real en 2021–2024.

# %%
H = 7
X_rez = pd.DataFrame({
    "rezago_7": y.shift(H),
    "rezago_8": y.shift(H + 1),
    "media_7": y.shift(H).rolling(7).mean(),
    "media_30": y.shift(H).rolling(30).mean(),
    "cos_dia": np.cos(2 * np.pi * (y.index.day_of_year - 1) / 365.25),
    "sen_dia": np.sin(2 * np.pi * (y.index.day_of_year - 1) / 365.25),
    "anio": y.index.year,
}, index=y.index)
tabla_rez = X_rez.assign(y=y).dropna()
X_tr, y_tr_ = tabla_rez.loc[ENTRENA].drop(columns="y"), tabla_rez.loc[ENTRENA, "y"]
X_te, y_te_ = tabla_rez.loc[PRUEBA].drop(columns="y"), tabla_rez.loc[PRUEBA, "y"]
print(f"entrenamiento: {len(X_tr):,} días; prueba: {len(X_te):,} días")
X_tr.tail(3)

# %%
modelos = {
    "regresión lineal": LinearRegression(),
    "bosque aleatorio": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.5,
                                              n_jobs=4, random_state=SEMILLA),
}
esquemas = {
    "KFold aleatorio": KFold(5, shuffle=True, random_state=SEMILLA),
    "origen móvil (5 años)": TimeSeriesSplit(n_splits=5, test_size=365),
}
resultados = {}
for nombre, modelo in modelos.items():
    fila = {}
    for esquema, cv in esquemas.items():
        fila[esquema] = -cross_val_score(modelo, X_tr, y_tr_, cv=cv, scoring="neg_mean_absolute_error").mean()
    fila["futuro real (2021–2024)"] = np.mean(np.abs(modelo.fit(X_tr, y_tr_).predict(X_te) - y_te_))
    resultados[nombre] = fila
resultados = pd.DataFrame(resultados).T
resultados.round(3)

# %%
fig, ax = plt.subplots(figsize=(11, 4.4))
anchura = 0.36
cols_res = list(resultados.columns)
for i, (nombre, color) in enumerate(zip(resultados.index, [AZUL, ROJO])):
    xs = np.arange(len(cols_res)) + (i - 0.5) * anchura
    ax.bar(xs, resultados.loc[nombre], width=anchura, color=color, label=nombre)
    for xx, v in zip(xs, resultados.loc[nombre]):
        ax.text(xx, v + 0.03, f"{v:.2f}", ha="center", fontsize=11, color=TINTA)
ax.axhline(mae_h.loc[H, "climatología"], color=TENUE, ls="--", lw=1.2)
ax.text(2.55, mae_h.loc[H, "climatología"] + 0.04, "climatología", color=TENUE, fontsize=11, ha="right")
ax.set_xticks(range(len(cols_res)), cols_res)
ax.set_ylim(0, 2.8)
ax.set_ylabel("MAE a 7 días (°C)")
ax.set_title("La validación cruzada aleatoria premia al modelo equivocado")
ax.legend(loc="upper left", ncols=2)
plt.show()

# %% [markdown]
# Lee la tabla por columnas:
#
# - Con **KFold aleatorio**, el bosque parece mucho mejor que la regresión lineal (1.75 contra 2.23 °C) y que la climatología (2.17 °C). Elegiríamos el bosque y prometeríamos un error de 1.75 °C a una semana.
# - Con **origen móvil** (entrenar con el pasado, validar con el año siguiente, cinco veces), el orden se invierte: la lineal es mejor (2.28 contra 2.37 °C).
# - En el **futuro real**, 2021–2024, la lineal tiene 2.05 °C y el bosque 2.19 °C: el bosque ni siquiera le gana a la climatología. El KFold prometió 1.75.
#
# **¿Por qué solo el bosque se beneficia de la fuga?** Porque con el día del año y el año el bosque puede ubicar a cada día casi exactamente en el tiempo, y sus hojas terminan promediando a los vecinos en el calendario, cuyas temperaturas en $t\pm 1$, $t\pm 2$ conocen la anomalía del día que se quiere predecir. La regresión lineal no tiene esa flexibilidad: una sola recta en cada atributo no puede memorizar vecindarios, y por eso su estimación casi no cambia entre los dos esquemas. **La fuga es peor justo con los modelos más flexibles**, que son los que más necesitamos evaluar bien. (El error del origen móvil, 2.28 y 2.37, es algo mayor que el del futuro real porque 2016–2020 incluyó años más difíciles de pronosticar, como 2020; eso es variación de verdad, no fuga.)
#
# ### Origen móvil: ventana creciente y deslizante
#
# `TimeSeriesSplit` implementa la **validación con origen móvil** (también llamada *evaluación en un origen que rueda* o *backtesting*): el pliego $k$ entrena con todo lo anterior a un corte y prueba con el bloque siguiente; el corte avanza. Hay dos variantes:
#
# - **ventana creciente** (la de arriba): cada vez se entrena con todo el pasado;
# - **ventana deslizante** (`max_train_size`): se entrena solo con los últimos $w$ días, para olvidar un pasado que quizá ya no se parece al presente.
#
# Y dos parámetros que hay que decidir según el uso: el **horizonte** (cuántos pasos adelante se pronostica; si se pronostica a 7 días, el bloque de prueba debe empezar 7 días después del último dato usado, lo que en scikit-learn se controla con `gap`) y el **salto** entre orígenes.

# %%
def dibujar_pliegos(ax, pliegos, n, titulo):
    for k, (i_tr, i_te) in enumerate(pliegos):
        ax.scatter(i_tr, np.full(len(i_tr), k), marker="|", s=60, color=AZUL, linewidths=0.6)
        ax.scatter(i_te, np.full(len(i_te), k), marker="|", s=60, color=ROJO, linewidths=0.6)
    ax.set_yticks(range(len(pliegos)), [f"pliego {k + 1}" for k in range(len(pliegos))])
    ax.set_xlim(0, n)
    ax.invert_yaxis()
    ax.set_title(titulo, fontsize=12.5)
    ax.grid(False)


n_demo = 120
idx_demo = np.arange(n_demo)
fig, ejes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
dibujar_pliegos(ejes[0], list(KFold(5, shuffle=True, random_state=SEMILLA).split(idx_demo)), n_demo,
                "KFold aleatorio: el futuro se mezcla con el pasado")
dibujar_pliegos(ejes[1], list(TimeSeriesSplit(5, test_size=15).split(idx_demo)), n_demo,
                "Origen móvil, ventana creciente")
dibujar_pliegos(ejes[2], list(TimeSeriesSplit(5, test_size=15, max_train_size=40, gap=3).split(idx_demo)), n_demo,
                "Origen móvil, ventana deslizante de 40 días y hueco de 3 (horizonte)")
ejes[2].set_xlabel("tiempo (días) · azul: entrenamiento · rojo: prueba")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Un caso real: Google Flu Trends
#
# En 2008 Google lanzó *Google Flu Trends* (GFT), que estimaba cada semana la proporción de consultas médicas por enfermedad tipo influenza en los Estados Unidos a partir de las búsquedas en Google, antes de que los Centros para el Control y la Prevención de Enfermedades (CDC) publicaran sus cifras, que llegan con una o dos semanas de retraso (J. Ginsberg et al., *Nature* 457, 2009). D. Lazer, R. Kennedy, G. King y A. Vespignani analizaron qué pasó después («The parable of Google Flu: traps in big data analysis», *Science* 343, 2014):
#
# - en febrero de 2013, GFT pronosticaba **más del doble** de las consultas por enfermedad tipo influenza que reportaron los CDC;
# - del 21 de agosto de 2011 al 1 de septiembre de 2013, GFT **sobreestimó en 100 de 108 semanas**, y sus errores estaban autocorrelacionados y tenían estacionalidad: el error de la semana pasada predecía el de esta;
# - la versión original se construyó buscando, entre 50 millones de términos de búsqueda, los que mejor se ajustaban a 1,152 datos; los autores la describen como «en parte detector de influenza, en parte detector de invierno», y no detectó la pandemia de influenza A H1N1 de 2009, que no ocurrió en invierno;
# - un modelo sencillo con los datos de los CDC rezagados y estacionalidad de 52 semanas tuvo menos error que GFT fuera de la muestra (MAE de 0.311 contra 0.486).
#
# Las lecciones son las de esta libreta: un patrón que se ajusta muy bien al pasado no garantiza nada sobre el futuro si las relaciones cambian (los hábitos de búsqueda y el propio buscador cambiaron); hay que **comparar contra referencias sencillas**; y hay que **validar hacia adelante**, una y otra vez, porque las series cambian.

# %% [markdown]
# ## 9. Resumen
#
# - Una serie de tiempo se reindexa a su frecuencia antes de todo. Sus componentes (tendencia, estacionalidad, ciclo, ruido) se separan con una climatología o con STL; lo que queda, las anomalías, es lo que se pronostica a corto plazo.
# - La ACF mide la memoria. La estacionalidad aparece como una oscilación de periodo $m$; la no estacionariedad, como una ACF que decae muy lento. La banda $\pm 1.96/\sqrt{T}$ es para ruido blanco, rezago por rezago.
# - La autocorrelación hace que las fórmulas que suponen independencia sean optimistas: el error estándar de una tendencia y la validación cruzada aleatoria.
# - Todo pronóstico se compara contra el ingenuo, el estacional ingenuo y la climatología, **por horizonte**, con MAE, RECM o MASE (no con MAPE si la escala tiene un cero arbitrario o hay valores cerca de cero).
# - Se valida con origen móvil: entrenar solo con el pasado, probar con el bloque siguiente y avanzar.
#
# *Referencias:* R. J. Hyndman y G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed., OTexts (2021), capítulos 2, 3 y 5 (en línea en otexts.com/fpp3). R. B. Cleveland et al., «STL: a seasonal-trend decomposition procedure based on loess», *Journal of Official Statistics* 6 (1990). R. J. Hyndman y A. B. Koehler, «Another look at measures of forecast accuracy», *International Journal of Forecasting* 22 (2006). C. Bergmeir y J. M. Benítez, «On the use of cross-validation for time series predictor evaluation», *Information Sciences* 191 (2012). D. Lazer et al., «The parable of Google Flu», *Science* 343 (2014).

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u07", ids=["u07-p01", "u07-p02", "u07-p03", "u07-p04"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. **Varianza de la caminata con deriva.** Sea $y_t = \delta + y_{t-1} + \varepsilon_t$ con $y_0 = 0$ y $\varepsilon_t$ ruido blanco de varianza $\sigma^2$. Calcula $\mathbb{E}[y_t]$, $\operatorname{Var}(y_t)$ y $\operatorname{Cov}(y_t, y_{t-k})$. Muestra que el pronóstico óptimo (en error cuadrático) de $y_{T+h}$ dado $y_1,\dots,y_T$ es $y_T + h\delta$ y que la varianza de su error es $h\sigma^2$. ¿Qué pronóstico de referencia de la sección 5 estima esto?
# 2. **El AR(1) de las anomalías.** Supón que $a_t = \phi a_{t-1} + \varepsilon_t$ con $|\phi| < 1$ y es estacionaria. Demuestra que $\rho(k) = \phi^k$ y que el pronóstico óptimo a $h$ pasos es $\phi^h a_T$, con error cuadrático medio $\sigma^2(1 - \phi^{2h})/(1 - \phi^2)$. Compara $\hat\rho_a(k)$ de la sección 3 con $\hat\phi^k$: ¿por qué la ACF empírica de las anomalías decae **más lento** que $0.79^k$ para $k$ grande?
# 3. **MAPE en °F.** Repite la tabla de métricas para $h = 1$ con la temperatura en grados Fahrenheit. ¿Qué métricas cambian y cuáles no? Demuestra que MASE es invariante ante cualquier transformación afín $y \mapsto \alpha y + \beta$ con $\alpha > 0$ aplicada a datos y pronósticos.
# 4. **La fuga y el atributo «año».** Repite la demostración de la sección 7 sin el atributo `anio`, y luego sin `cos_dia`, `sen_dia` ni `anio`. ¿Cómo cambia la distancia entre el KFold aleatorio y el futuro real para el bosque? Prueba también un `KFold` sin barajar (`shuffle=False`), que forma bloques contiguos: ¿se elimina la fuga? ¿Qué problema queda en los bordes de cada bloque?
# 5. **Ventana deslizante.** Cambia `ajustar_referencia` para que la climatología y $\hat\phi$ se estimen solo con los últimos 10 años antes de cada 1 de enero (ventana deslizante) en lugar de todo el pasado. ¿Mejora o empeora el MAE en 2021–2024 para $h = 1$ y $h = 14$? ¿Qué esperarías si de verdad hubiera una tendencia de calentamiento de 0.3 °C por década?
