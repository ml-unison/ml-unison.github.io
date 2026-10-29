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
# # Pronosticar la demanda de bicicletas hora por hora
#
# **Unidad 7 · Ejemplo 3**
#
# Los modelos ARIMA de la libreta anterior describen **una** serie con unos cuantos parámetros. Funcionan muy bien cuando la serie es suave y la estructura es simple, pero les cuesta incorporar muchas variables (el calendario, los feriados, el clima) y aprender patrones no lineales, como que la hora pico de un martes no se parece a la de un sábado. La alternativa que domina hoy muchas aplicaciones es **convertir el pronóstico en un problema de aprendizaje supervisado** y usar las herramientas que ya conocemos: regresión lineal, *boosting* y redes neuronales.
#
# En esta libreta vamos a:
#
# 1. preparar una serie por hora con huecos y decidir qué hacer con ellos;
# 2. reformular «pronosticar las próximas 24 horas» como regresión, con **atributos rezagados**, estadísticas móviles y calendario, **sin fuga del futuro**;
# 3. comparar pronósticos de referencia, regresión lineal y LightGBM, eligiendo hiperparámetros con **validación de origen móvil**;
# 4. ver qué pasa al usar el **clima** como variable exógena, y por qué ese resultado engaña;
# 5. comparar las estrategias **recursiva**, **directa** y **multisalida** (esta última con una red pequeña en PyTorch) midiendo el error contra el horizonte;
# 6. construir **intervalos de predicción** con la pérdida cuantil y medir su cobertura;
# 7. mirar de frente el día en que todos los modelos fallan.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado. LightGBM y PyTorch ya vienen en Colab.

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
import lightgbm as lgb
import torch
from torch import nn
from sklearn.linear_model import LinearRegression

import rdp

rdp.estilo.usar()
SEMILLA = 2012
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE
AZUL_TENUE = rdp.estilo.PALETA["azul-tenue"]
MESES = {7: "julio", 8: "agosto", 9: "septiembre"}
MES_CORTO = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def dia_corto(d):
    """Etiqueta «lun 12» para las gráficas (sin depender del idioma del sistema)."""
    return f"{['lun', 'mar', 'mié', 'jue', 'vie', 'sáb', 'dom'][d.day_of_week]} {d.day}"


# La verificación automática del repositorio usa menos épocas para la red (sección 6).
# Las cifras del texto son las de la ejecución normal.
RAPIDO = os.environ.get("RDP_RAPIDO") == "1"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
# LightGBM y PyTorch traen cada uno su biblioteca de hilos (OpenMP); en algunas máquinas, usarlas
# en el mismo proceso con varios hilos hace que PyTorch se bloquee. La red es pequeña: un hilo basta.
torch.set_num_threads(1)

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Contexto.** Capital Bikeshare es el sistema de bicicletas compartidas de Washington, D. C.: se toma una bicicleta en una estación y se deja en otra. Para que siempre haya bicicletas donde se necesitan (y lugares libres donde se devuelven), el operador mueve bicicletas en camionetas entre estaciones, programa el mantenimiento y decide cuánto personal asignar. Todo eso se planea con un pronóstico de la demanda de las próximas horas.
#
# **Datos.** Son las rentas por hora de 2011 y 2012 que ya usamos en las unidades 4 y 5 (H. Fanaee-T y J. Gama, *Progress in Artificial Intelligence*, 2014; UCI, CC BY 4.0). Por hora tenemos `total` (rentas), el calendario (`dia_semana`, `feriado`, `laboral`) y el clima medido en esa hora (`temp`, `humedad`, `clima`).
#
# **Problema.** Cada hora, con la información disponible **hasta esa hora**, pronosticar las rentas de cada una de las **24 horas siguientes**. Entrenaremos con enero de 2011 a septiembre de 2012 y evaluaremos en octubre a diciembre de 2012, el último trimestre, que ningún modelo ve al entrenar.
#
# Hasta ahora tratamos cada hora como una observación independiente. Ahora el orden importa: la hora siguiente se parece a la actual, y el martes se parece al martes pasado. Primero, una revisión que en una serie de tiempo es obligatoria: **¿están todas las horas?**

# %%
b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
b.index = b.fecha + pd.to_timedelta(b.hora, unit="h")
horas = pd.date_range("2011-01-01 00:00", "2012-12-31 23:00", freq="h")
faltan = horas.difference(b.index)
print(f"{len(b):,} horas registradas de {len(horas):,} posibles: faltan {len(faltan)}")
print(f"menor número de rentas en una hora registrada: {b.total.min()}")

# %%
fig, ejes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [3, 2]})
cuenta = pd.Series(faltan.hour).value_counts().reindex(range(24), fill_value=0)
ejes[0].bar(cuenta.index, cuenta.values, color=AZUL, width=0.8)
ejes[0].set_xlabel("hora del día")
ejes[0].set_ylabel("horas faltantes")
ejes[0].set_title("Las horas faltantes son casi todas de madrugada")
ejes[0].set_xticks(range(0, 24, 3))
por_dia = pd.Series(1, index=faltan).groupby(faltan.normalize()).sum()
grandes = por_dia[por_dia >= 6]
ejes[1].barh([f"{d.day} {MES_CORTO[d.month - 1]} {d.year}" for d in grandes.index], grandes.values, color=ROJO)
ejes[1].invert_yaxis()
ejes[1].set_xlabel("horas faltantes ese día")
ejes[1].set_title("Días con 6 o más horas faltantes")
fig.tight_layout()
plt.show()

# %% [markdown]
# Dos observaciones deciden qué hacer con las 165 horas faltantes:
#
# - La tabla **no tiene ninguna hora con cero rentas**: el mínimo es 1. Y las horas que faltan se concentran entre las 2 y las 5 de la mañana. Lo más razonable es que la tabla se construyó contando rentas, así que una hora sin rentas simplemente no aparece: el faltante **es** un cero.
# - Unos cuantos días pierden muchas horas. Varios coinciden con eventos conocidos: una nevada fuerte (26 y 27 de enero de 2011), el huracán Irene (27 y 28 de agosto de 2011) y el huracán Sandy (29 y 30 de octubre de 2012), cuando el gobierno federal cerró sus oficinas y el metro de Washington suspendió el servicio. En esas horas tampoco se rentaron bicicletas.
#
# **Decisión:** reindexamos a las 17,544 horas, rellenamos las rentas faltantes con **0** y guardamos una columna `faltante` por si después queremos tratarlas aparte. Las variables de calendario se toman del día; el clima, que en esas horas no se registró, se interpola (solo lo usaremos en la sección 5). Nota que Sandy cae en el periodo de **prueba**: volveremos a eso al final.

# %%
s = b.reindex(horas)
s["faltante"] = s.total.isna()
s["total"] = s.total.fillna(0)
s["hora"] = horas.hour
calendario = b.groupby("fecha")[["dia_semana", "feriado", "laboral"]].first()
for c in calendario:
    s[c] = calendario[c].reindex(horas.normalize()).to_numpy()
s[["temp", "humedad", "clima"]] = s[["temp", "humedad", "clima"]].interpolate().ffill()

y = s.total.to_numpy(dtype=float)
hora, dsem = s.hora.to_numpy(), s.dia_semana.to_numpy().astype(int)
laboral, feriado = s.laboral.to_numpy().astype(int), s.feriado.to_numpy().astype(int)
temp, clima = s.temp.to_numpy(), s.clima.to_numpy()
INICIO_PRUEBA = horas.get_loc(pd.Timestamp("2012-10-01 00:00"))
print(f"rentas medias por hora en prueba (oct–dic 2012): {y[INICIO_PRUEBA:].mean():.0f}")

# %%
fig, ejes = plt.subplots(2, 1, figsize=(13, 7.5))
diario = s.total.resample("D").sum()
ejes[0].axvspan(horas[INICIO_PRUEBA], horas[-1], color=AZUL_TENUE, alpha=0.6, lw=0)
ejes[0].plot(diario.index, diario.values, color=AZUL, lw=1)
ejes[0].plot(diario.rolling(28, center=True).mean(), color=TINTA, lw=2)
ejes[0].text(horas[INICIO_PRUEBA] + pd.Timedelta(days=5), 8400, "prueba", color=TENUE, fontsize=12)
ejes[0].annotate("Sandy", (pd.Timestamp("2012-10-29"), 300), (pd.Timestamp("2012-08-01"), 1200),
                 color=ROJO, fontsize=12, arrowprops=dict(arrowstyle="->", color=ROJO))
ejes[0].set_ylabel("rentas por día")
ejes[0].set_title("Dos años de rentas: el sistema crece y la demanda sigue a las estaciones del año")
dos_semanas = s.loc["2012-09-10":"2012-09-23", "total"]
ejes[1].plot(dos_semanas.index, dos_semanas.values, color=AZUL, lw=1.6)
for d in pd.date_range("2012-09-10", "2012-09-23"):
    ejes[1].text(d + pd.Timedelta(hours=12), 1060, dia_corto(d), ha="center", fontsize=10, color=TENUE)
ejes[1].set_ylim(0, 1120)
ejes[1].set_ylabel("rentas por hora")
ejes[1].set_title("Dos semanas de septiembre de 2012: dos picos los días laborales, una loma el fin de semana")
fig.tight_layout()
plt.show()

# %% [markdown]
# La serie tiene tres escalas de tiempo: un **ciclo diario** muy marcado (con dos picos, entrada y salida del trabajo, en días laborales), un **ciclo semanal** (los fines de semana son otra cosa), y una **tendencia** con estacionalidad anual (el sistema creció mucho de 2011 a 2012, y en invierno se usa menos). Un buen modelo tiene que aprovechar las tres.

# %% [markdown]
# ## 2. El pronóstico como aprendizaje supervisado
#
# Llamemos **origen** a la hora $t$ en que se hace el pronóstico: conocemos $y_1, \dots, y_t$ y queremos $y_{t+h}$ para los **horizontes** $h = 1, \dots, 24$. La idea es construir, para cada origen del pasado, un renglón de atributos calculados **solo** con lo que se conocía en $t$, y usar como respuesta el valor que de verdad ocurrió en $t+h$:
#
# $$\mathbf{x}_t^{(h)} = \big(\underbrace{y_t, y_{t-1}, \dots, y_{t-23}}_{\text{últimas 24 horas}},\; \underbrace{y_{t+h-24},\; y_{t+h-168}}_{\text{misma hora ayer y hace una semana}},\; \underbrace{\bar y_{24}(t),\; \bar y_{168}(t)}_{\text{medias móviles}},\; \underbrace{\text{hora}, \text{día}, \text{laboral}, \text{feriado}}_{\text{calendario de } t+h}\big) \;\longmapsto\; y_{t+h}.$$
#
# Cada origen de la historia da un ejemplo de entrenamiento: con dos años de datos por hora tenemos unos 15,000. Tres detalles importan:
#
# - Los **rezagos** que dependen de $h$, como $y_{t+h-24}$, solo son legítimos si $t+h-24 \le t$, es decir, si $h \le 24$. Para pronosticar más lejos habría que usar rezagos más viejos.
# - El **calendario** de la hora objetivo sí se conoce de antemano (sabemos qué día es mañana y si es feriado). El **clima** de la hora objetivo, no: lo discutimos en la sección 5.
# - La partición en entrenamiento y prueba se hace **por tiempo**, y además la respuesta de los ejemplos de entrenamiento no puede caer en el periodo de prueba. Por eso el último origen de entrenamiento es el 30 de septiembre a las 23 h **menos 24 horas**.
#
# Guardamos para cada origen una **ventana** con las últimas 336 horas (dos semanas); los atributos se calculan a partir de ella. Escribirlo así tiene una ventaja que aprovecharemos en la sección 6: la misma función sirve cuando la ventana contiene pronósticos en lugar de datos.

# %%
VENTANA, H = 336, 24
HORIZONTES = np.arange(1, H + 1)
NOMBRES = (["y(t)"] + [f"y(t-{k})" for k in range(1, 24)] + ["y(t+h-24)", "y(t+h-168)", "media 24 h", "media 168 h",
           "hora", "día de la semana", "laboral", "feriado"])


def ventanas(origenes):
    """Matriz (n, 336): las últimas dos semanas de rentas hasta cada origen (la última columna es y_t)."""
    return y[origenes[:, None] + np.arange(-VENTANA + 1, 1)]


def atributos(V, origenes, h, con_clima=False):
    """Atributos para pronosticar y_{t+h} desde el origen t, usando solo la ventana V (sin fuga)."""
    t = origenes + h                                   # hora objetivo: solo se usa su calendario
    cols = [V[:, -1 - k] for k in range(24)]           # y_t, …, y_{t-23}
    cols += [V[:, h - 25], V[:, h - 169]]              # misma hora ayer y hace una semana (h ≤ 24)
    cols += [V[:, -24:].mean(1), V[:, -168:].mean(1)]
    cols += [hora[t], dsem[t], laboral[t], feriado[t]]
    if con_clima:                                      # ¡clima observado en la hora objetivo! (sección 5)
        cols += [temp[t], clima[t]]
    return np.column_stack(cols)


ULTIMO_ENTRENA = INICIO_PRUEBA - 1                     # 30 sep 2012, 23 h
o_tr = np.arange(VENTANA - 1, ULTIMO_ENTRENA - H + 1)  # orígenes cuyo horizonte 24 cae en entrenamiento
o_te = np.arange(ULTIMO_ENTRENA, len(y) - H)           # orígenes de prueba (información hasta oct–dic)
V_tr, V_te = ventanas(o_tr), ventanas(o_te)
Y_tr = y[o_tr[:, None] + HORIZONTES]                   # (n, 24): respuestas a 1, …, 24 horas
Y_te = y[o_te[:, None] + HORIZONTES]
assert o_tr[-1] + H <= ULTIMO_ENTRENA < o_te[0] + 1    # ninguna respuesta de entrenamiento cae en prueba
print(f"orígenes de entrenamiento: {len(o_tr):,} ({horas[o_tr[0]]:%Y-%m-%d %H} h a {horas[o_tr[-1]]:%Y-%m-%d %H} h)")
print(f"orígenes de prueba:        {len(o_te):,} ({horas[o_te[0]]:%Y-%m-%d %H} h a {horas[o_te[-1]]:%Y-%m-%d %H} h)")
pd.DataFrame(atributos(V_tr[:3], o_tr[:3], h=1), columns=NOMBRES).iloc[:, [0, 1, 2, 23, 24, 25, 26, 27, 28, 29, 30, 31]].round(1)

# %% [markdown]
# Evaluamos **todos** los orígenes del trimestre de prueba (2,185, uno por hora) con un modelo entrenado una sola vez: es un **origen móvil** sin reentrenar. Cada origen produce 24 pronósticos, así que podemos medir el error **por horizonte**. Usamos el error absoluto medio (MAE, en bicicletas por hora) y la raíz del error cuadrático medio (RECM), que castiga más los errores grandes.

# %%
def mae_por_h(P):
    return np.abs(P - Y_te).mean(axis=0)


def resumen(P):
    return {"MAE h=1": mae_por_h(P)[0], "MAE h=24": mae_por_h(P)[-1],
            "MAE (1–24)": np.abs(P - Y_te).mean(), "RECM (1–24)": np.sqrt(((P - Y_te) ** 2).mean())}


pronosticos = {}

# %% [markdown]
# ## 3. Pronósticos de referencia
#
# Antes de entrenar nada, dos pronósticos que no aprenden: el **estacional ingenuo diario**, $\hat y_{t+h} = y_{t+h-24}$ (lo que pasó ayer a esa hora), y el **estacional ingenuo semanal**, $\hat y_{t+h} = y_{t+h-168}$ (lo que pasó hace una semana a esa hora). Cualquier modelo que no les gane no sirve.

# %%
pronosticos["ingenuo: ayer"] = np.column_stack([V_te[:, h - 25] for h in HORIZONTES])
pronosticos["ingenuo: semana pasada"] = np.column_stack([V_te[:, h - 169] for h in HORIZONTES])
pd.DataFrame({k: resumen(v) for k, v in pronosticos.items()}).T.round(1)

# %% [markdown]
# Los dos se equivocan en promedio por unas 76 bicicletas por hora (en prueba se rentan 215 por hora en promedio), sin importar el horizonte (su pronóstico no usa nada de las últimas horas). El de «ayer» falla los lunes y los sábados, cuando el día anterior es de otro tipo; el de la semana pasada falla cuando cambia el clima o hay un feriado. Un modelo puede combinar las dos fuentes de información y además corregir con lo que pasó en las últimas horas.

# %% [markdown]
# ## 4. Modelos directos: regresión lineal y LightGBM
#
# La estrategia **directa** entrena un modelo distinto para cada horizonte: $\hat y_{t+h} = g_h(\mathbf{x}_t^{(h)})$, $h = 1, \dots, 24$. Son 24 regresiones independientes.
#
# ### Elegir hiperparámetros con origen móvil
#
# LightGBM (Unidad 5) tiene un hiperparámetro que hay que elegir con cuidado: el número de árboles. La validación cruzada con particiones aleatorias **no sirve** aquí: mezclaría horas del futuro en el entrenamiento y horas vecinas (casi iguales) entre entrenamiento y validación, y daría un error optimista (lo vimos en la libreta 1 de esta unidad). Usamos **origen móvil con ventana creciente**: entrenamos con todo lo anterior a julio de 2012 y validamos en julio; luego con todo lo anterior a agosto y validamos en agosto; luego septiembre. Siempre el pasado predice al futuro.
#
# Para no gastar tiempo, elegimos el número de árboles con el horizonte más difícil, $h = 24$, y lo usamos para todos.

# %%
PARAMETROS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=SEMILLA, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
MALLA_ARBOLES = np.arange(50, 801, 50)
curvas_cv = {}
h = 24
for mes in ["2012-07", "2012-08", "2012-09"]:
    inicio = horas.get_loc(pd.Timestamp(f"{mes}-01"))
    fin = horas.get_loc(pd.Timestamp(f"{mes}-01") + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23))
    entrena = o_tr + h < inicio                        # respuestas antes del mes de validación
    valida = (o_tr + h >= inicio) & (o_tr + h <= fin)  # respuestas dentro del mes
    m = lgb.LGBMRegressor(n_estimators=MALLA_ARBOLES[-1], **PARAMETROS)
    m.fit(atributos(V_tr[entrena], o_tr[entrena], h), Y_tr[entrena, h - 1])
    Xv = atributos(V_tr[valida], o_tr[valida], h)
    curvas_cv[mes] = [np.abs(m.predict(Xv, num_iteration=k) - Y_tr[valida, h - 1]).mean() for k in MALLA_ARBOLES]
curvas_cv = pd.DataFrame(curvas_cv, index=MALLA_ARBOLES)
ARBOLES = int(curvas_cv.mean(axis=1).idxmin())
print(f"número de árboles elegido: {ARBOLES} (MAE medio de validación {curvas_cv.mean(axis=1).min():.1f})")

# %%
fig, ax = plt.subplots(figsize=(10, 4.6))
for (mes, curva), color in zip(curvas_cv.items(), [AZUL, ROJO, DORADO_OSC]):
    ax.plot(curva.index, curva.values, color=color, lw=2)
ax.plot(curvas_cv.index, curvas_cv.mean(axis=1), color=TINTA, lw=2.6, ls="--")
ax.axvline(ARBOLES, color=TENUE, lw=1)
rdp.estilo.etiquetar(ax, 815, {**{f"valida {MESES[pd.Timestamp(k).month]}": v.iloc[-1] for k, v in curvas_cv.items()},
                               "promedio": curvas_cv.mean(axis=1).iloc[-1]})
ax.set_xlim(0, 940)
ax.set_xlabel("número de árboles")
ax.set_ylabel("MAE de validación (h = 24)")
ax.set_title("Origen móvil con ventana creciente: un mes de validación a la vez")
plt.show()

# %% [markdown]
# Cada mes tiene su propio nivel de error (septiembre resultó el más difícil y agosto el más fácil), pero las tres curvas coinciden en la forma: bajan rápido y luego se aplanan. La variación **entre** meses es mucho mayor que la diferencia entre 300 y 800 árboles; eso también es información: el error de un pronóstico depende de qué periodo toque.
#
# ### Los 24 modelos
#
# Ahora entrenamos con todo el periodo de entrenamiento: 24 LightGBM y, para comparar, 24 regresiones lineales. A la regresión lineal le damos la hora y el tipo de día de la hora objetivo en *one-hot* (las 48 combinaciones hora × laboral, como el modelo M4 de la Unidad 4), porque no puede aprender sola que el efecto de la hora no es lineal. Recortamos en 0 todos los pronósticos: no hay rentas negativas.

# %%
def lineal_atributos(X, origenes, h):
    """Rezagos y medias móviles + hora × laboral de la hora objetivo en one-hot."""
    t = origenes + h
    return np.column_stack([X[:, :28], np.eye(48)[2 * hora[t] + laboral[t]], feriado[t]])


modelos_directos, P_lin, P_dir = {}, [], []
for h in HORIZONTES:
    Xa, Xb = atributos(V_tr, o_tr, h), atributos(V_te, o_te, h)
    lin = LinearRegression().fit(lineal_atributos(Xa, o_tr, h), Y_tr[:, h - 1])
    P_lin.append(lin.predict(lineal_atributos(Xb, o_te, h)))
    modelos_directos[h] = lgb.LGBMRegressor(n_estimators=ARBOLES, **PARAMETROS).fit(Xa, Y_tr[:, h - 1])
    P_dir.append(modelos_directos[h].predict(Xb))
pronosticos["lineal directa"] = np.clip(np.column_stack(P_lin), 0, None)
pronosticos["LightGBM directa"] = np.clip(np.column_stack(P_dir), 0, None)
pd.DataFrame({k: resumen(v) for k, v in pronosticos.items()}).T.round(1)

# %% [markdown]
# Los dos modelos aprendidos le ganan con holgura a los ingenuos, y LightGBM a la regresión lineal. A una hora de distancia el error es la mitad que a 24 horas: lo que pasó en la última hora es la mejor pista de la siguiente. ¿Qué usa LightGBM? La **ganancia** total de las divisiones en cada atributo (Unidad 5) lo dice:

# %%
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
for ax, h in zip(ejes, [1, 24]):
    ganancia = pd.Series(modelos_directos[h].booster_.feature_importance("gain"), index=NOMBRES)
    ganancia = (ganancia / ganancia.sum()).sort_values().tail(8)
    ax.barh(ganancia.index, ganancia.values, color=AZUL)
    ax.set_title(f"horizonte h = {h}")
    ax.set_xlabel("fracción de la ganancia total")
fig.suptitle("Qué atributos usa LightGBM: a una hora, el presente; a un día, el calendario y la semana pasada", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# A una hora de distancia domina $y_t$, el valor actual, seguido de la misma hora de la semana pasada. (Para $h = 1$, `y(t-23)` y `y(t+h-24)` son el mismo número: la hora objetivo de ayer; el modelo usa uno u otro indistintamente.) A 24 horas, casi dos terceras partes de la ganancia vienen de **la misma hora hace una semana**, y después de ayer a esa hora y del valor actual: el modelo es una especie de estacional ingenuo «inteligente», que corrige lo de la semana pasada con lo que ve hoy. La hora y el día de la semana aportan poco por sí solos porque el rezago semanal ya los lleva implícitos: el lunes pasado a las 8 ya «sabe» que es lunes a las 8.

# %% [markdown]
# ## 5. Variables exógenas: el clima y una fuga tentadora
#
# La lluvia ahuyenta a los ciclistas; la Unidad 4 lo midió: multiplica las rentas por 0.61. ¿Por qué no agregar el clima? Probemos agregando la temperatura y el tipo de clima **observados en la hora objetivo** $t + h$.

# %%
fuga = {}
for h in [1, 6, 12, 24]:
    m = lgb.LGBMRegressor(n_estimators=ARBOLES, **PARAMETROS)
    m.fit(atributos(V_tr, o_tr, h, con_clima=True), Y_tr[:, h - 1])
    p = np.clip(m.predict(atributos(V_te, o_te, h, con_clima=True)), 0, None)
    fuga[h] = {"sin clima": mae_por_h(pronosticos["LightGBM directa"])[h - 1],
               "con el clima observado en t+h": np.abs(p - Y_te[:, h - 1]).mean()}
fuga = pd.DataFrame(fuga).T.rename_axis("h")
fuga["mejora (%)"] = 100 * (1 - fuga.iloc[:, 1] / fuga.iloc[:, 0])
fuga.round(1)

# %% [markdown]
# El error a 24 horas baja casi 15 %. Pero **ese modelo no se puede usar**: a las 8 de la noche de hoy no conocemos la temperatura ni la lluvia de mañana a las 8 de la noche. Es una **fuga** del futuro, de la misma familia que la duración de la llamada en el ejemplo del banco de la Unidad 5 (y del tema de fugas de la Unidad 1): un atributo que no existe en el momento de decidir. En la validación se ve espléndido; en producción, el atributo no está.
#
# Hay tres maneras honestas de usar el clima:
#
# 1. **Solo rezagos**: usar el clima observado hasta $t$ (ayuda poco a 24 horas).
# 2. **Pronósticos del clima**: usar el pronóstico meteorológico que se tenía en $t$ para la hora $t+h$. Es lo correcto, pero hay que entrenar también con los pronósticos **históricos** (no con el clima observado), porque el modelo debe aprender a confiar en ellos tanto como merecen: un pronóstico de lluvia a 24 horas se equivoca.
# 3. **Escenarios**: reportar el pronóstico condicionado («si llueve mañana, esperen 40 % menos»).
#
# La mejora de la tabla es, entonces, una **cota superior**: lo más que ganaríamos con un pronóstico del clima perfecto.

# %% [markdown]
# ## 6. Tres estrategias para pronosticar varios pasos
#
# Hasta ahora usamos la estrategia directa. Hay otras dos.
#
# - **Recursiva.** Se entrena **un solo** modelo a un paso, $\hat y_{t+1} = g(\mathbf{x}_t^{(1)})$, y se aplica 24 veces, metiendo cada pronóstico en la ventana como si fuera un dato: $\hat y_{t+2} = g(\dots, \hat y_{t+1}, y_t, \dots)$. Es lo que hace ARIMA. Su problema es que **los errores se acumulan**: el modelo se entrenó con rezagos verdaderos y en la prueba recibe rezagos pronosticados, que tienen error y son más suaves que los datos.
# - **Directa.** Un modelo por horizonte (sección 4). No acumula errores, pero son 24 modelos que no comparten nada, y sus pronósticos para horas consecutivas pueden ser incoherentes entre sí.
# - **Multisalida.** **Un** modelo que produce los 24 valores a la vez: $(\hat y_{t+1}, \dots, \hat y_{t+24}) = g(\mathbf{x}_t)$. Una red neuronal lo hace naturalmente: basta una capa de salida con 24 neuronas.
#
# ### Recursiva con LightGBM
#
# Gracias a que los atributos se calculan desde la ventana, la estrategia recursiva es un ciclo: pronosticar un paso, recorrer la ventana, pegar el pronóstico al final.

# %%
P_rec, V = [], V_te.copy()
for j in range(H):
    p = np.clip(modelos_directos[1].predict(atributos(V, o_te + j, 1)), 0, None)  # el modelo a un paso
    P_rec.append(p)
    V = np.column_stack([V[:, 1:], p])                 # el pronóstico entra a la ventana como si fuera dato
pronosticos["LightGBM recursiva"] = np.column_stack(P_rec)
print(f"MAE recursiva:  h=1 {mae_por_h(pronosticos['LightGBM recursiva'])[0]:.1f},  h=24 {mae_por_h(pronosticos['LightGBM recursiva'])[-1]:.1f}")
print(f"MAE directa:    h=1 {mae_por_h(pronosticos['LightGBM directa'])[0]:.1f},  h=24 {mae_por_h(pronosticos['LightGBM directa'])[-1]:.1f}")

# %% [markdown]
# En $h = 1$ son el mismo modelo. Después, la recursiva se va quedando atrás.
#
# ### Multisalida con una red neuronal
#
# Una red pequeña (un perceptrón multicapa, Unidad 6) recibe la última semana de rentas (168 valores, en escala $\log(1+y)$ para que los picos no dominen), la hora y el día de la semana del origen en *one-hot*, y los indicadores de día laboral de las 24 horas objetivo. Tiene dos capas ocultas de 256 neuronas y una salida de 24: una por horizonte. Se entrena minimizando el error cuadrático promedio sobre los 24 horizontes, con rentas divididas entre 100.

# %%
def entradas_red(V, origenes):
    semana = np.log1p(V[:, -168:]) / np.log1p(1000)
    return np.hstack([semana, np.eye(24)[hora[origenes]], np.eye(7)[dsem[origenes]],
                      laboral[origenes[:, None] + HORIZONTES]]).astype(np.float32)


ESCALA = 100.0
torch.manual_seed(SEMILLA)
red = nn.Sequential(nn.Linear(168 + 24 + 7 + 24, 256), nn.ReLU(),
                    nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, H)).to(DISPOSITIVO)
X_red = torch.tensor(entradas_red(V_tr, o_tr), device=DISPOSITIVO)
Y_red = torch.tensor(Y_tr / ESCALA, dtype=torch.float32, device=DISPOSITIVO)
optimizador = torch.optim.Adam(red.parameters(), lr=1e-3)
generador = torch.Generator().manual_seed(SEMILLA)
EPOCAS = 5 if RAPIDO else 60
historia_red = []
for epoca in range(EPOCAS):
    orden = torch.randperm(len(X_red), generator=generador).to(DISPOSITIVO)
    total = 0.0
    for i in range(0, len(orden), 128):
        lote = orden[i:i + 128]
        perdida = ((red(X_red[lote]) - Y_red[lote]) ** 2).mean()
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        total += perdida.item() * len(lote)
    historia_red.append(total / len(orden))
with torch.no_grad():
    salida = red(torch.tensor(entradas_red(V_te, o_te), device=DISPOSITIVO)).cpu().numpy()
pronosticos["red multisalida"] = np.clip(ESCALA * salida, 0, None)
print(f"pérdida de entrenamiento: época 1 {historia_red[0]:.3f}, época {EPOCAS} {historia_red[-1]:.3f}")
print(f"MAE red multisalida: h=1 {mae_por_h(pronosticos['red multisalida'])[0]:.1f},  h=24 {mae_por_h(pronosticos['red multisalida'])[-1]:.1f}")

# %% [markdown]
# ### El error contra el horizonte

# %%
orden_modelos = ["ingenuo: semana pasada", "lineal directa", "red multisalida", "LightGBM recursiva", "LightGBM directa"]
colores = {"ingenuo: semana pasada": GRIS, "lineal directa": DORADO_OSC, "red multisalida": TINTA,
           "LightGBM recursiva": ROJO, "LightGBM directa": AZUL}
fig, ax = plt.subplots(figsize=(11, 5.4))
for nombre in orden_modelos:
    ax.plot(HORIZONTES, mae_por_h(pronosticos[nombre]), color=colores[nombre], lw=2.4,
            ls="--" if nombre.startswith("ingenuo") else "-")
rdp.estilo.etiquetar(ax, 24.6, {k: mae_por_h(pronosticos[k])[-1] for k in orden_modelos}, fontsize=12)
ax.set_xlim(0.5, 31)
ax.set_ylim(0, 85)
ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
ax.set_xlabel("horizonte h (horas)")
ax.set_ylabel("MAE en prueba (bicicletas por hora)")
ax.set_title("Error de pronóstico contra horizonte, octubre a diciembre de 2012")
plt.show()
tabla = pd.DataFrame({k: resumen(pronosticos[k]) for k in ["ingenuo: ayer"] + orden_modelos}).T
tabla.round(1)

# %% [markdown]
# Lo que se ve:
#
# - **Todos** los modelos empeoran con el horizonte en las primeras horas y luego se estabilizan: después de unas 6 horas, las últimas observaciones ya no dicen mucho más que el calendario y la semana pasada.
# - La **recursiva** coincide con la directa en $h = 1$ y se separa conforme crece $h$: es la acumulación de errores. En $h = 24$ es la peor de los modelos aprendidos.
# - La **red multisalida**, un solo modelo sin ningún atributo diseñado a mano (solo la ventana y el calendario), queda cerca de LightGBM en el promedio de los 24 horizontes (MAE 52.4 contra 50.3, y una RECM ligeramente menor). Donde pierde claramente es a una hora: el dato decisivo, $y_t$, es para ella una entrada entre 223, mientras que a LightGBM se lo dimos como atributo. Con más datos o más series la diferencia suele cerrarse o invertirse; con 15,000 ejemplos, el *boosting* con buenos atributos es difícil de vencer.
#
# Esto es lo mismo que se encontró a gran escala en la competencia M5 (2020): los mejores métodos usaron LightGBM con atributos rezagados, y el ganador **promedió** modelos recursivos y no recursivos. Combinar pronósticos de modelos distintos casi siempre ayuda; prueba en los ejercicios a promediar la directa y la red.

# %% [markdown]
# ## 7. Intervalos de predicción con pérdida cuantil
#
# Un pronóstico puntual no basta para decidir cuántas bicicletas llevar a una estación: hace falta saber **cuánto podría equivocarse**. Un intervalo de predicción del 80 % es un par de cuantiles condicionales, el 10 % y el 90 %, de $y_{t+h}$ dados los atributos. ¿Cómo se aprende un cuantil con un modelo de regresión? Cambiando la pérdida.
#
# ### La pérdida cuantil
#
# Para $\tau \in (0, 1)$, la **pérdida cuantil** (*pinball loss*) de un residuo $u = y - q$ es
#
# $$\rho_\tau(u) = u\,\big(\tau - \mathbb{1}[u < 0]\big) = \begin{cases} \tau\,u & u \ge 0 \quad\text{(nos quedamos cortos)}\\ (\tau - 1)\,u = (1-\tau)\,|u| & u < 0 \quad\text{(nos pasamos)}.\end{cases}$$
#
# Es una «V» asimétrica: con $\tau = 0.9$, quedarse corto cuesta 0.9 por unidad y pasarse cuesta 0.1. Con $\tau = 0.5$ es la mitad del error absoluto.
#
# **Su minimizador es el cuantil $\tau$.** Sea $Y$ con distribución continua $F$ y $L(q) = \mathbb{E}[\rho_\tau(Y - q)]$. Separando las dos ramas,
#
# $$L(q) = \tau\int_{q}^{\infty}(y - q)\,dF(y) + (1 - \tau)\int_{-\infty}^{q}(q - y)\,dF(y).$$
#
# Derivando bajo la integral (los términos de frontera se anulan porque el integrando vale 0 en $y = q$):
#
# $$L'(q) = -\tau\,\big(1 - F(q)\big) + (1 - \tau)\,F(q) = F(q) - \tau .$$
#
# $L'$ es creciente (así que $L$ es convexa) y se anula exactamente cuando $F(q) = \tau$: **el mínimo está en el cuantil $\tau$** de $Y$. Con datos, la misma cuenta dice que la pendiente de la pérdida empírica es (fracción de datos debajo de $q$) $- \tau$, y se anula en el cuantil muestral. Lo verificamos con las rentas de los días laborales a las 8 de la mañana:

# %%
ocho = y[:INICIO_PRUEBA][(hora[:INICIO_PRUEBA] == 8) & (laboral[:INICIO_PRUEBA] == 1)]
malla_q = np.arange(0, 1001, 1.0)
fig, ax = plt.subplots(figsize=(10.5, 4.6))
for tau, color in zip([0.1, 0.5, 0.9], [ROJO, TINTA, AZUL]):
    u = ocho[None, :] - malla_q[:, None]
    perdida = (u * (tau - (u < 0))).mean(axis=1)
    q_opt = malla_q[perdida.argmin()]
    ax.plot(malla_q, perdida, color=color, lw=2.2)
    ax.plot(q_opt, perdida.min(), "o", color=color, ms=8)
    ax.vlines(q_opt, -8, perdida.min(), color=color, lw=1, ls=":")
    ax.text(q_opt, -12, f"τ = {tau}: mínimo en {q_opt:.0f}\n(cuantil muestral {np.quantile(ocho, tau):.0f})",
            ha="center", va="top", fontsize=11, color=TINTA)
ax.set_ylim(-85, 420)
ax.set_xlabel("pronóstico constante q (rentas a las 8 h, días laborales)")
ax.set_ylabel("pérdida cuantil media")
ax.set_title(f"La pérdida cuantil se minimiza en el cuantil ({len(ocho)} días laborales de entrenamiento)")
plt.show()

# %% [markdown]
# Cada mínimo cae en el cuantil muestral (salvo por la discretización de la malla). Ahora el mismo principio, pero **condicional**: LightGBM con `objective="quantile"` minimiza $\sum_n \rho_\tau\big(y_n - g(\mathbf{x}_n)\big)$ y aprende $g(\mathbf{x}) \approx$ cuantil $\tau$ de $y \mid \mathbf{x}$. Entrenamos los cuantiles 0.1 y 0.9 para algunos horizontes y medimos la **cobertura empírica** en prueba: la fracción de horas cuyo valor real cayó dentro del intervalo. Si el intervalo es honesto, debe ser cercana al 80 %.

# %%
TAUS = [0.1, 0.9]
H_CUANTIL = [1, 6, 12, 24]
cuantiles, filas = {}, []
for h in H_CUANTIL:
    Xa, Xb = atributos(V_tr, o_tr, h), atributos(V_te, o_te, h)
    for tau in TAUS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=ARBOLES, **PARAMETROS)
        cuantiles[h, tau] = np.clip(m.fit(Xa, Y_tr[:, h - 1]).predict(Xb), 0, None)
    bajo, alto, real = cuantiles[h, 0.1], cuantiles[h, 0.9], Y_te[:, h - 1]
    filas.append({"h": h, "cobertura (%)": 100 * np.mean((real >= bajo) & (real <= alto)),
                  "debajo del 10 % (%)": 100 * np.mean(real < bajo), "arriba del 90 % (%)": 100 * np.mean(real > alto),
                  "ancho medio": np.mean(alto - bajo), "cuantiles cruzados (%)": 100 * np.mean(bajo > alto)})
cobertura = pd.DataFrame(filas).set_index("h")
cobertura.round(1)

# %% [markdown]
# La cobertura queda **por debajo** del 80 % prometido, de 70 % a una hora a 64 % a un día, y la falla es asimétrica: hay más horas de las esperadas **debajo** del cuantil 0.1 (hasta 21 % en lugar de 10 %). El modelo **sobreestima** con más frecuencia de la que cree. No es un error de programación: es la diferencia entre entrenamiento y prueba. El último trimestre de 2012 trae la caída del otoño al invierno (la demanda diaria baja de unas 7,000 rentas a menos de 4,000), Sandy, el Día de Acción de Gracias y la Navidad; el entrenamiento tiene un solo otoño, el de 2011, con un sistema mucho más chico. Los cuantiles aprendidos describen la incertidumbre **del pasado**; si el futuro cambia, el intervalo se queda corto. Esto es la regla en series de tiempo, y por eso la cobertura se **mide** en datos posteriores al entrenamiento, nunca se supone.
#
# Unas cuantas horas tienen los cuantiles **cruzados** (el 10 % por arriba del 90 %): los dos modelos se entrenan por separado y nada los obliga a respetar el orden. En la práctica se ordenan los dos valores después de predecir.
#
# Veamos una semana de prueba, pronosticada **con un día de anticipación** ($h = 24$): cada punto es el pronóstico hecho 24 horas antes.

# %%
def serie_h(P, h, desde, hasta):
    """Pronósticos a h horas (columna h) indexados por la hora objetivo."""
    return pd.Series(P, index=horas[o_te + h]).loc[desde:hasta]


desde, hasta = "2012-11-12", "2012-11-18 23:00"
real = s.total.loc[desde:hasta]
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.fill_between(real.index, serie_h(cuantiles[24, 0.1], 24, desde, hasta), serie_h(cuantiles[24, 0.9], 24, desde, hasta),
                color=AZUL_TENUE, alpha=0.8, lw=0, step=None)
ax.plot(real.index, real.values, color=TINTA, lw=1.6)
ax.plot(real.index, serie_h(pronosticos["LightGBM directa"][:, 23], 24, desde, hasta), color=AZUL, lw=2)
for d in pd.date_range(desde, hasta):
    ax.text(d + pd.Timedelta(hours=12), 870, dia_corto(d), ha="center", fontsize=10, color=TENUE)
ax.text(pd.Timestamp("2012-11-12 12:00"), 825, "(feriado)", ha="center", fontsize=10, color=TENUE)
ax.set_ylim(0, 900)
ax.set_ylabel("rentas por hora")
ax.set_title("Pronóstico a 24 horas (azul), intervalo del 10 % al 90 % (banda) y lo observado (negro)")
plt.show()

# %% [markdown]
# El intervalo se ensancha en las horas pico, donde el error posible es mayor, y casi desaparece de madrugada: la incertidumbre **depende de la hora**, algo que un intervalo de la forma «pronóstico ± constante» no captaría.

# %% [markdown]
# ## 8. El día que todos los modelos fallaron
#
# El lunes 29 de octubre de 2012 el huracán Sandy llegó a la costa este de Estados Unidos. En Washington cerraron las oficinas federales y el metro, y durante dos días casi nadie rentó una bicicleta. Así se veían los pronósticos hechos el día anterior:

# %%
desde, hasta = "2012-10-26", "2012-11-01 23:00"
real = s.total.loc[desde:hasta]
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.fill_between(real.index, serie_h(cuantiles[24, 0.1], 24, desde, hasta), serie_h(cuantiles[24, 0.9], 24, desde, hasta),
                color=AZUL_TENUE, alpha=0.8, lw=0)
ax.plot(real.index, real.values, color=TINTA, lw=1.6)
ax.plot(real.index, serie_h(pronosticos["LightGBM directa"][:, 23], 24, desde, hasta), color=AZUL, lw=2)
ax.axvspan(pd.Timestamp("2012-10-29"), pd.Timestamp("2012-10-31"), color=rdp.estilo.PALETA["gris-claro"], lw=0, zorder=0)
ax.text(pd.Timestamp("2012-10-30"), 690, "Sandy", ha="center", fontsize=13, color=ROJO)
for d in pd.date_range(desde, hasta):
    ax.text(d + pd.Timedelta(hours=12), 870, dia_corto(d), ha="center", fontsize=10, color=TENUE)
ax.set_ylim(0, 900)
ax.set_ylabel("rentas por hora")
ax.set_title("Pronóstico a 24 horas alrededor del huracán Sandy")
plt.show()

sandy = (horas[o_te + 24] >= "2012-10-29") & (horas[o_te + 24] < "2012-10-31")
err24 = np.abs(pronosticos["LightGBM directa"][:, 23] - Y_te[:, 23])
print(f"MAE a 24 h en prueba: {err24.mean():.1f}; sin los dos días de Sandy: {err24[~sandy].mean():.1f}")
print(f"esos 48 de {len(err24):,} pronósticos (2 %) aportan el {100 * err24[sandy].sum() / err24.sum():.1f} % del error absoluto total")

# %% [markdown]
# El modelo pronosticó un lunes y un martes normales de otoño, con su intervalo «del 80 %», y la realidad fue casi cero. Nada en los rezagos ni en el calendario anunciaba el huracán; solo una variable exógena (el aviso meteorológico, el cierre oficial) lo habría permitido. Tres lecciones:
#
# - Un intervalo de predicción cuantifica la incertidumbre **que se parece a la del pasado**. Los eventos raros quedan fuera de él.
# - Antes de reportar un error, hay que ver **de dónde viene**: aquí, el 2 % de los pronósticos aporta una parte desproporcionada del error. Decidir si se excluyen de la evaluación (y decirlo) es parte del trabajo.
# - Un sistema de pronóstico en producción necesita **supervisión humana** y una manera de incorporar información que el modelo no tiene.
#
# ## 9. Cierre: ¿qué modelo para qué problema?
#
# | Situación | Familia que suele funcionar |
# |---|---|
# | Pocas series, largas y suaves, con estructura clara (CO₂, temperatura mensual) | ARIMA, suavizamiento exponencial: pocos parámetros, intervalos con teoría |
# | Muchas series relacionadas, calendario y variables exógenas (ventas, demanda, energía) | *Boosting* con atributos rezagados, un modelo **global** para todas las series |
# | Mucha historia, mucha estructura y muchos datos (miles de series, señales de alta frecuencia) | Redes neuronales (MLP, redes recurrentes, convolucionales, *transformers*) |
#
# Y en todos los casos: comparar contra un **ingenuo estacional**, validar **hacia adelante** con origen móvil, cuidar que ningún atributo venga del futuro y medir la **cobertura** de los intervalos en datos nuevos.
#
# *Referencias:* H. Fanaee-T y J. Gama, «Event labeling combining ensemble detectors and background knowledge», *Progress in Artificial Intelligence* 2 (2014). S. Makridakis, E. Spiliotis y V. Assimakopoulos, «M5 accuracy competition: Results, findings, and conclusions», *International Journal of Forecasting* 38 (2022). G. Bontempi, S. Ben Taieb e Y.-A. Le Borgne, «Machine learning strategies for time series forecasting», en *Business Intelligence*, LNBIP 138, Springer (2013). R. Koenker y G. Bassett, «Regression quantiles», *Econometrica* 46 (1978). R. J. Hyndman y G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed., OTexts (2021), capítulos 5 y 10–12.

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u07", ids=["u07-p09", "u07-p10", "u07-p11", "u07-p12"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. **Cuantiles de una distribución discreta.** Sea $y_1 \le \dots \le y_N$ una muestra ordenada. Demuestra que la pérdida cuantil empírica $L(q) = \frac1N\sum_n \rho_\tau(y_n - q)$ es convexa y lineal por tramos, que su pendiente en un punto $q$ que no es dato vale $\#\{n : y_n < q\}/N - \tau$, y concluye que el conjunto de minimizadores es un dato $y_{(k)}$ o un intervalo entre dos datos consecutivos. ¿Qué pasa con $\tau = 0.5$?
# 2. **Recursiva contra directa en un AR(1).** Sea $y_t = \phi y_{t-1} + \varepsilon_t$ y supón que estimamos $\hat\phi = \phi + \delta$. Muestra que el pronóstico recursivo a $h$ pasos, $\hat\phi^{\,h} y_t$, tiene un sesgo condicional aproximado de $h\,\phi^{h-1}\delta\, y_t$, mientras que la regresión directa de $y_{t+h}$ sobre $y_t$ estima $\phi^h$ sin acumular $\delta$. Simula con $\phi = 0.9$ para comprobarlo. ¿En qué caso gana la recursiva?
# 3. **Combinar pronósticos.** Promedia los pronósticos de la LightGBM directa y de la red multisalida, y mide el MAE por horizonte. ¿El promedio le gana a los dos? Explica por qué combinar dos pronósticos cuyos errores no están perfectamente correlacionados reduce el error cuadrático esperado (pista: la varianza de un promedio).
# 4. **Intervalos conformes.** Reserva septiembre de 2012 como calibración: entrena con lo anterior, calcula los residuos absolutos de la LightGBM directa a 24 horas en septiembre y usa su cuantil 0.8 como semiancho de un intervalo simétrico. Mide la cobertura en prueba y compárala con la de los cuantiles de LightGBM. ¿Cuál es más honesta? ¿Cuál se adapta a la hora del día?
# 5. **Reentrenar o no.** Reentrena la LightGBM directa a 24 horas al inicio de cada mes de prueba (octubre, noviembre, diciembre), con todo lo anterior. ¿Mejora el error? ¿Mejora la cobertura de los intervalos? Relaciónalo con el crecimiento del sistema.
