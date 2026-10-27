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
# # Métricas de clasificación: el riesgo de crédito
#
# **Unidad 3 · Ejemplo 4**
#
# Un clasificador no se evalúa con un solo número. «¿Qué tan bueno es?» se descompone en varias preguntas distintas: ¿ordena bien a los clientes del más al menos riesgoso? ¿Cuántos de los que señala son de verdad riesgosos? ¿Cuántos riesgosos se le escapan? ¿Sus probabilidades significan lo que dicen? ¿Se equivoca igual con todos los grupos? Cada pregunta tiene su métrica, y elegir la equivocada lleva a decisiones equivocadas.
#
# En esta libreta vamos a:
#
# 1. ver por qué la **exactitud** engaña cuando una clase es mucho más rara que la otra,
# 2. definir con precisión la **matriz de confusión** y las tasas que salen de ella,
# 3. estudiar la **curva ROC** y demostrar numéricamente qué significa su área (AUC), y la curva **precisión–exhaustividad**,
# 4. **elegir el umbral** de decisión a partir de los costos de cada error, derivando el umbral óptimo,
# 5. revisar si las probabilidades están **calibradas** y recalibrarlas,
# 6. medir las tasas de error **por grupo** y entender por qué no se pueden igualar todas a la vez.

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
from scipy.stats import mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             brier_score_loss, log_loss)

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Contexto.** A principios de la década de 2000, los bancos de Taiwán compitieron por el mercado de tarjetas de crédito y de efectivo relajando sus requisitos: dieron tarjetas a clientes con poca capacidad de pago y alentaron el consumo a crédito. Para 2005 muchos tarjetahabientes acumulaban deudas que no podían cubrir, y entre 2005 y 2006 estalló una crisis de deuda de tarjetas: la morosidad se disparó, los bancos tuvieron grandes pérdidas y el gobierno tuvo que intervenir. La confianza en el sistema tardó años en recuperarse.
#
# **Datos.** En ese contexto, I-Cheng Yeh y Che-hui Lien reunieron los registros de 30,000 clientes de tarjeta de crédito de un banco taiwanés, de abril a septiembre de 2005, para comparar métodos que estiman la probabilidad de incumplimiento (*Expert Systems with Applications*, 2009). La tabla está en el repositorio de aprendizaje automático de la UCI con licencia CC BY 4.0. Por cliente tenemos:
#
# - `limite`: límite de crédito, en nuevos dólares taiwaneses (NT\$; un NT\$ valía unos 30 centavos de peso de entonces);
# - `sexo` (1 hombre, 2 mujer), `escolaridad` (1 posgrado, 2 universidad, 3 preparatoria, 4 otra; 0, 5 y 6 no están documentados), `estado_civil` (1 casado, 2 soltero, 3 otro) y `edad`;
# - `atraso_1` a `atraso_6`: estado de pago de septiembre (`_1`) a abril (`_6`); −1 es «pagó a tiempo» y de 1 a 9 son meses de atraso (−2 y 0 aparecen en los datos pero la fuente no los documenta);
# - `factura_1` a `factura_6`: monto del estado de cuenta de cada mes;
# - `pago_1` a `pago_6`: lo que el cliente pagó cada mes;
# - `incumple`: 1 si el cliente **no pagó** en octubre de 2005, el mes siguiente.
#
# **Problema.** El banco quiere un **puntaje de riesgo**: un número por cliente que diga qué tan probable es que incumpla el mes próximo, para decidir a quién reducir el límite de crédito o contactar antes de que caiga en mora. Equivocarse cuesta, y no cuesta lo mismo en los dos sentidos: si el puntaje deja pasar a un cliente que incumple, el banco pierde buena parte de la deuda; si señala a un cliente que sí iba a pagar, el banco lo molesta, quizá lo pierde, y deja de cobrarle intereses. Las métricas que veremos sirven para decidir **qué modelo** usar y **con qué umbral**.

# %%
credito = rdp.datos.cargar("credito_taiwan")
print(credito.shape)
credito.head()

# %%
prevalencia = credito["incumple"].mean()
print(f"incumplen: {credito['incumple'].sum():,} de {len(credito):,} clientes ({prevalencia:.1%})")

# %% [markdown]
# Uno de cada cinco clientes incumple. Es una proporción enorme para un banco (consecuencia de la crisis), pero desde el punto de vista del clasificador las clases están **desbalanceadas**: la clase que nos interesa, los incumplidos, es la minoría. A la proporción de positivos la llamaremos **prevalencia**, $\pi = P(Y = 1)$.
#
# ### Separación en entrenamiento y prueba
#
# Separamos un 25 % para prueba. La separación es **estratificada**: el azar se aplica dentro de cada clase, para que ambos conjuntos tengan la misma prevalencia. Con clases desbalanceadas esto evita que, por mala suerte, la prueba quede con pocos positivos.

# %%
categoricas = ["sexo", "escolaridad", "estado_civil"]
X = credito.drop(columns="incumple")
y = credito["incumple"]
numericas = [c for c in X.columns if c not in categoricas]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEMILLA)
print(f"entrenamiento: {len(X_tr):,} clientes, prevalencia {y_tr.mean():.2%}")
print(f"prueba:        {len(X_te):,} clientes, prevalencia {y_te.mean():.2%}")

# %% [markdown]
# ### Dos modelos y una línea base
#
# - **Regresión logística.** Modela $P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ con $\sigma(z) = 1/(1 + e^{-z})$. Va en un `Pipeline` que estandariza las variables numéricas y codifica con *one-hot* las tres categóricas (los códigos de escolaridad o estado civil no tienen orden ni escala).
# - **Bosque aleatorio** (*random forest*). Promedia 300 árboles de decisión, cada uno entrenado con una muestra distinta de los datos; su puntaje es la fracción de árboles que votan por «incumple». Lo estudiaremos en la Unidad 5; aquí nos basta saber que capta interacciones y relaciones no lineales que la logística no ve. Los árboles no necesitan escalado ni *one-hot*: les damos los códigos tal cual.
# - **Línea base.** Un «modelo» que ignora a $\mathbf{x}$ y siempre dice la clase más frecuente: «paga».
#
# Los tres se entrenan solo con el conjunto de entrenamiento.

# %%
preprocesamiento = ColumnTransformer([
    ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ("numericas", StandardScaler(), numericas),
])
logistica = make_pipeline(preprocesamiento, LogisticRegression(max_iter=2000))
bosque = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEMILLA)
base = DummyClassifier(strategy="most_frequent")

for modelo in (logistica, bosque, base):
    modelo.fit(X_tr, y_tr)

# puntajes = probabilidad estimada de incumplir, en el conjunto de prueba
p_log = logistica.predict_proba(X_te)[:, 1]
p_bos = bosque.predict_proba(X_te)[:, 1]
y_te = y_te.to_numpy()

# %% [markdown]
# ### La exactitud engaña
#
# La **exactitud** (*accuracy*) es la fracción de aciertos. Para tener predicciones duras necesitamos un umbral; usemos por ahora el «natural», 0.5: se predice «incumple» si la probabilidad estimada pasa de 0.5.

# %%
exactitudes = {
    "siempre «paga»": accuracy_score(y_te, base.predict(X_te)),
    "regresión logística": accuracy_score(y_te, p_log >= 0.5),
    "bosque aleatorio": accuracy_score(y_te, p_bos >= 0.5),
}
pd.Series(exactitudes, name="exactitud").round(3).to_frame()

# %% [markdown]
# El modelo que **no sirve para nada**, porque nunca señala a nadie, acierta en 78 % de los casos: exactamente $1 - \pi$. Los dos modelos de verdad le ganan por apenas unos puntos. Con clases desbalanceadas la exactitud está dominada por la clase mayoritaria, y un 81 % puede ser tanto un buen modelo como uno mediocre. Necesitamos mirar por separado **cómo le va con cada clase**.

# %% [markdown]
# ## 2. La matriz de confusión
#
# Fijado un umbral $t$, cada cliente de prueba cae en una de cuatro casillas, según su clase real $y$ y la predicha $\hat y = \mathbb{1}[\,s(\mathbf{x}) \ge t\,]$, donde $s(\mathbf{x})$ es el puntaje:
#
# | | predice «paga» ($\hat y = 0$) | predice «incumple» ($\hat y = 1$) |
# |---|---|---|
# | **paga** ($y = 0$) | verdaderos negativos, $VN$ | falsos positivos, $FP$ |
# | **incumple** ($y = 1$) | falsos negativos, $FN$ | verdaderos positivos, $VP$ |
#
# «Positivo» es la clase que buscamos, aquí el incumplimiento; no tiene nada de bueno. Un falso positivo es un cliente cumplido al que señalamos; un falso negativo, un incumplido que se nos escapó.
#
# Antes de contar, veamos de dónde salen esas casillas: la distribución de puntajes del bosque para cada clase real.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 1, 41)
ax.hist(p_bos[y_te == 0], bins=bins, color=AZUL, alpha=0.6, density=True)
ax.hist(p_bos[y_te == 1], bins=bins, color=ROJO, alpha=0.6, density=True)
ax.axvline(0.5, color=rdp.estilo.TINTA, lw=1.5, ls="--")
ax.text(0.51, ax.get_ylim()[1] * 0.92, "umbral 0.5", color=rdp.estilo.TINTA)
ax.text(0.27, 4.2, "pagan", color=AZUL, fontsize=13)
ax.text(0.72, 1.4, "incumplen", color=ROJO, fontsize=13)
ax.set(title="Puntajes del bosque aleatorio en prueba, por clase real",
       xlabel="probabilidad estimada de incumplir", ylabel="densidad")
plt.show()

# %% [markdown]
# Las dos distribuciones se traslapan mucho: no hay umbral que separe perfectamente. Los cumplidos se concentran en puntajes bajos; los incumplidos se reparten en dos montones, uno con puntajes altos (los que ya venían atrasados, fáciles de detectar) y otro mezclado con los cumplidos. Todo lo que está a la derecha de la línea se predice positivo: el área roja a la derecha son los $VP$, la azul a la derecha los $FP$, la roja a la izquierda los $FN$.

# %%
def matriz(y, s, t):
    """Devuelve (VN, FP, FN, VP) al predecir positivo cuando s >= t."""
    return confusion_matrix(y, (s >= t).astype(int), labels=[0, 1]).ravel()

vn, fp, fn, vp = matriz(y_te, p_bos, 0.5)

fig, ax = plt.subplots(figsize=(6, 5))
M = np.array([[vn, fp], [fn, vp]])
ax.imshow(M, cmap=rdp.estilo.SECUENCIAL, vmin=0, vmax=M.max() * 1.6)
nombres = [["VN", "FP"], ["FN", "VP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{nombres[i][j]}\n{M[i, j]:,}", ha="center", va="center", fontsize=15,
                color=rdp.estilo.TINTA)
ax.set_xticks([0, 1], ["paga", "incumple"])
ax.set_yticks([0, 1], ["paga", "incumple"])
ax.set(xlabel="predicción", ylabel="clase real", title="Bosque aleatorio, umbral 0.5")
ax.grid(False)
plt.show()

# %% [markdown]
# ### Las tasas
#
# Con $P = VP + FN$ positivos reales y $N = VN + FP$ negativos reales:
#
# $$
# \begin{aligned}
# \text{exhaustividad} = TVP &= \frac{VP}{VP + FN} = P(\hat Y = 1 \mid Y = 1), \\[4pt]
# \text{especificidad} = TVN &= \frac{VN}{VN + FP} = P(\hat Y = 0 \mid Y = 0), \\[4pt]
# \text{tasa de falsos positivos } TFP &= \frac{FP}{VN + FP} = 1 - \text{especificidad}, \\[4pt]
# \text{precisión} = VPP &= \frac{VP}{VP + FP} = P(Y = 1 \mid \hat Y = 1), \\[4pt]
# F_1 &= \frac{2}{\dfrac{1}{\text{precisión}} + \dfrac{1}{\text{exhaustividad}}} = \frac{2\,VP}{2\,VP + FP + FN}.
# \end{aligned}
# $$
#
# Hay que fijarse en **qué se condiciona**. La exhaustividad (también llamada sensibilidad o tasa de verdaderos positivos, $TVP$) y la especificidad condicionan en la clase **real**: miden al modelo dentro de cada clase y no dependen de cuántos positivos haya. La precisión (o valor predictivo positivo, $VPP$) condiciona en la **predicción**: de los que señalamos, ¿cuántos eran incumplidos? Por el teorema de Bayes,
#
# $$
# VPP = \frac{TVP\,\pi}{TVP\,\pi + TFP\,(1 - \pi)},
# $$
#
# así que la precisión **sí** depende de la prevalencia $\pi$: el mismo modelo, aplicado a una población con menos incumplidos, tendría menor precisión. $F_1$ es la media armónica de precisión y exhaustividad; la media armónica castiga que alguna de las dos sea baja. Ojo: $F_1$ no usa los $VN$.

# %%
def tasas(y, s, t):
    vn, fp, fn, vp = matriz(y, s, t)
    return {
        "exactitud": (vp + vn) / len(y),
        "exhaustividad (TVP)": vp / (vp + fn),
        "especificidad": vn / (vn + fp),
        "TFP": fp / (vn + fp),
        "precisión (VPP)": vp / (vp + fp) if vp + fp else np.nan,
        "F1": 2 * vp / (2 * vp + fp + fn),
        "tasa de selección": (vp + fp) / len(y),
    }

pd.DataFrame({"logística": tasas(y_te, p_log, 0.5), "bosque": tasas(y_te, p_bos, 0.5)}).round(3)

# %% [markdown]
# Con umbral 0.5 los dos modelos son **conservadores**: señalan a pocos clientes (la tasa de selección es la fracción que se predice positiva) y, cuando señalan, suelen acertar (precisión cercana a 0.7), pero se les escapan entre dos tercios y cuatro quintos de los incumplidos (exhaustividad baja). Si el banco quiere detectar más incumplidos tiene que bajar el umbral, y entonces subirán los falsos positivos. Todas estas cifras son de **un** umbral; para comparar modelos sin comprometernos con uno, hay que ver todos los umbrales a la vez.

# %% [markdown]
# ## 3. La curva ROC y el AUC
#
# Al mover el umbral $t$ de 1 a 0, el modelo pasa de no señalar a nadie, $(TFP, TVP) = (0, 0)$, a señalar a todos, $(1, 1)$. La **curva ROC** (*receiver operating characteristic*, un nombre que viene del análisis de radares en la Segunda Guerra Mundial) es la trayectoria de $(TFP(t), TVP(t))$. Un clasificador que adivina al azar queda sobre la diagonal; uno perfecto pasa por la esquina $(0, 1)$.
#
# Como $TVP$ y $TFP$ condicionan en la clase real, la curva ROC **no depende de la prevalencia**. Y como solo importa el orden de los puntajes, tampoco cambia si les aplicamos cualquier transformación creciente.

# %%
fig, ax = plt.subplots(figsize=(7, 6.5))
auc = {}
for nombre, p, color in [("logística", p_log, AZUL), ("bosque", p_bos, ROJO)]:
    tfp, tvp, _ = roc_curve(y_te, p)
    auc[nombre] = roc_auc_score(y_te, p)
    ax.plot(tfp, tvp, color=color)
ax.plot([0, 1], [0, 1], color=GRIS, lw=1.5, ls="--")
ax.text(0.62, 0.55, "azar", color=GRIS, rotation=40)
ax.text(0.30, 0.52, f"logística\nAUC = {auc['logística']:.3f}", color=AZUL)
ax.text(0.07, 0.70, f"bosque\nAUC = {auc['bosque']:.3f}", color=ROJO)
for nombre, p, color in [("logística", p_log, AZUL), ("bosque", p_bos, ROJO)]:
    t = tasas(y_te, p, 0.5)
    ax.plot(t["TFP"], t["exhaustividad (TVP)"], "o", color=color, mec="white", ms=10)
ax.annotate("umbral 0.5", (0.055, 0.36), xytext=(0.22, 0.16), arrowprops=dict(arrowstyle="->", color=rdp.estilo.TENUE),
            color=rdp.estilo.TENUE)
ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal", title="Curvas ROC en prueba",
       xlabel="tasa de falsos positivos (1 − especificidad)", ylabel="exhaustividad (tasa de verdaderos positivos)")
plt.show()

# %% [markdown]
# El bosque domina a la logística en casi toda la curva: para cualquier tasa de falsos positivos que el banco tolere, detecta más incumplidos. El punto marcado es el umbral 0.5; está muy abajo a la izquierda, lo que confirma que 0.5 es un umbral muy exigente para este problema.
#
# ### Qué significa el AUC
#
# El **área bajo la curva ROC** (AUC) resume la curva en un número entre 0 y 1 (0.5 es el azar). Tiene una interpretación probabilística muy limpia:
#
# > **El AUC es la probabilidad de que un incumplido elegido al azar tenga un puntaje mayor que un cumplido elegido al azar** (contando los empates como medio acierto).
#
# *Esbozo de demostración.* Sean $S_1$ y $S_0$ los puntajes de un positivo y de un negativo independientes, con funciones de supervivencia $TVP(t) = P(S_1 \ge t)$ y $TFP(t) = P(S_0 \ge t)$. Recorrer la curva ROC es integrar $TVP$ contra $TFP$; al cambiar de variable al umbral $t$ (que va de $+\infty$ a $-\infty$ mientras $TFP$ va de 0 a 1), con $dTFP = -f_0(t)\,dt$ y $f_0$ la densidad de $S_0$:
#
# $$
# \text{AUC} = \int_0^1 TVP \, d\,TFP = \int_{-\infty}^{\infty} P(S_1 \ge t)\, f_0(t)\, dt = P(S_1 \ge S_0).
# $$
#
# La última igualdad es condicionar en $S_0 = t$. Con datos finitos, la versión empírica es comparar **todos los pares** (incumplido, cumplido):
#
# $$
# \widehat{\text{AUC}} = \frac{1}{n_1 n_0} \sum_{i:\,y_i = 1} \; \sum_{j:\,y_j = 0} \Big( \mathbb{1}[s_i > s_j] + \tfrac12 \mathbb{1}[s_i = s_j] \Big),
# $$
#
# que es exactamente el estadístico $U$ de Mann–Whitney (una prueba no paramétrica clásica para comparar dos muestras) dividido entre $n_1 n_0$. Verifiquémoslo con fuerza bruta: son unos nueve millones de pares, que NumPy compara en un instante.

# %%
s1 = p_bos[y_te == 1]
s0 = p_bos[y_te == 0]
print(f"pares: {len(s1):,} incumplidos × {len(s0):,} cumplidos = {len(s1) * len(s0):,}")

mayor = (s1[:, None] > s0[None, :]).mean()
empate = (s1[:, None] == s0[None, :]).mean()
auc_pares = mayor + empate / 2
U = mannwhitneyu(s1, s0).statistic

print(f"AUC por pares:          {auc_pares:.6f}  ({mayor:.4f} gana + {empate:.4f} empates / 2)")
print(f"U / (n1 n0) de scipy:   {U / (len(s1) * len(s0)):.6f}")
print(f"roc_auc_score:          {roc_auc_score(y_te, p_bos):.6f}")

# %% [markdown]
# Coinciden hasta el último decimal. La interpretación aclara qué mide el AUC y qué no: mide qué tan bien el puntaje **ordena** a los clientes. No dice nada de un umbral en particular, ni de si las probabilidades están bien calibradas (multiplicar todos los puntajes por 0.1 no cambia el AUC). Para el banco es justo lo que importa si va a revisar a los clientes del más riesgoso hacia abajo hasta donde le alcance el presupuesto.
#
# ### La curva precisión–exhaustividad
#
# La curva ROC tiene un punto ciego con clases desbalanceadas: una $TFP$ de 0.1 parece pequeña, pero con 5,841 cumplidos en prueba son 584 clientes señalados por error, más que muchos de los aciertos. La **curva precisión–exhaustividad** (PR) grafica, para cada umbral, la precisión contra la exhaustividad, y así pone en primer plano la pregunta «de los que señalo, ¿cuántos son?».
#
# Su línea base no es la diagonal: un clasificador al azar que señala a una fracción cualquiera de clientes tiene precisión igual a la prevalencia, $\pi \approx 0.22$, a cualquier exhaustividad. El área bajo la curva PR se resume con la **precisión promedio** (*average precision*, AP), $\text{AP} = \sum_k (R_k - R_{k-1})\,P_k$, a diferencia del AUC, depende de la prevalencia.

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ap = {}
for nombre, p, color in [("logística", p_log, AZUL), ("bosque", p_bos, ROJO)]:
    prec, exh, _ = precision_recall_curve(y_te, p)
    ap[nombre] = average_precision_score(y_te, p)
    ax.plot(exh, prec, color=color)
ax.axhline(y_te.mean(), color=GRIS, lw=1.5, ls="--")
ax.text(0.6, y_te.mean() - 0.06, f"azar: prevalencia = {y_te.mean():.2f}", color=GRIS)
ax.text(0.2, 0.45, f"logística, AP = {ap['logística']:.3f}", color=AZUL)
ax.text(0.52, 0.64, f"bosque, AP = {ap['bosque']:.3f}", color=ROJO)
ax.set(xlim=(0, 1), ylim=(0, 1.01), title="Curvas precisión–exhaustividad en prueba",
       xlabel="exhaustividad (fracción de incumplidos detectados)", ylabel="precisión (fracción de señalados que incumplen)")
plt.show()

# %% [markdown]
# La curva muestra el precio de detectar más: con el bosque, el primer 30 % de los incumplidos se encuentra con una precisión cercana a 0.7, pero para llegar a detectar el 60 % la precisión baja a alrededor de 0.4, es decir, por cada incumplido detectado señalamos a 1.5 clientes cumplidos. La logística está por debajo en casi todo el recorrido. Las dos curvas cuentan la misma historia que la ROC, pero en unidades que un gerente de crédito entiende mejor.

# %% [markdown]
# ## 4. Elegir el umbral por costos
#
# Ni el AUC ni la curva PR dicen qué umbral usar. Eso depende de cuánto cuesta cada error, y eso no lo sabe el modelo: lo sabe el banco. Propongamos costos razonables, medidos en una unidad común:
#
# - **Falso negativo**, $c_{FN}$: un incumplido al que no le hicimos nada. El banco pierde la parte de la deuda que no logra recuperar. La factura promedio en estos datos ronda los 50,000 NT\$; si se recupera la mitad, la pérdida es del orden de 25,000 NT\$.
# - **Falso positivo**, $c_{FP}$: un cliente cumplido al que le bajamos el límite o llamamos. Cuesta la llamada, el enojo y, sobre todo, el riesgo de que se vaya con otro banco con los intereses que habría pagado: digamos unos 5,000 NT\$.
# - Los aciertos cuestan 0 (es una simplificación: también una llamada a un incumplido cuesta, pero lo que importa es la diferencia entre decidir bien y mal).
#
# Es decir, $c_{FN} = 5$ y $c_{FP} = 1$ en unidades de 5,000 NT\$. Las cifras son ilustrativas; lo que importa es el razonamiento, y el banco pondría las suyas.
#
# ### El umbral óptimo
#
# Supongamos que el modelo da la probabilidad verdadera, $p = P(Y = 1 \mid \mathbf{x})$. Para un cliente con ese $p$, las dos decisiones tienen costo esperado
#
# $$
# \mathbb{E}[\text{costo} \mid \text{señalar}] = (1 - p)\, c_{FP}, \qquad
# \mathbb{E}[\text{costo} \mid \text{no señalar}] = p\, c_{FN}.
# $$
#
# Conviene señalar cuando el primero es menor: $(1 - p)\,c_{FP} < p\, c_{FN}$, es decir,
#
# $$
# p > t^\ast = \frac{c_{FP}}{c_{FP} + c_{FN}}.
# $$
#
# Esta es la **regla de decisión de Bayes** para costos asimétricos: como se toma la mejor decisión para cada $\mathbf{x}$ por separado, también minimiza el costo esperado total. Con costos iguales, $t^\ast = 1/2$ (el umbral «natural» supone que los dos errores cuestan lo mismo). Con nuestros costos, $t^\ast = 1/6 \approx 0.167$: basta una probabilidad de incumplir de 17 % para que valga la pena actuar, porque dejar pasar a un incumplido es cinco veces peor.
#
# Fíjate en la hipótesis: **la probabilidad debe ser verdadera**, o al menos estar calibrada. Si el modelo dice 0.17 cuando la frecuencia real es 0.30, el umbral $t^\ast$ deja de ser óptimo. Volveremos a esto en la sección 5.
#
# Veamos el costo promedio por cliente en prueba para todos los umbrales:

# %%
C_FP, C_FN = 1.0, 5.0
t_opt = C_FP / (C_FP + C_FN)

def costo(y, s, t):
    vn, fp, fn, vp = matriz(y, s, t)
    return (C_FP * fp + C_FN * fn) / len(y)

umbrales = np.linspace(0.01, 0.8, 159)
fig, ax = plt.subplots(figsize=(10, 5.5))
costos = {}
for nombre, p, color in [("logística", p_log, AZUL), ("bosque", p_bos, ROJO)]:
    c = np.array([costo(y_te, p, t) for t in umbrales])
    costos[nombre] = c
    ax.plot(umbrales, c, color=color)
ax.axhline(C_FN * y_te.mean(), color=GRIS, lw=1.5, ls=":")
ax.text(0.35, C_FN * y_te.mean() + 0.02, "no señalar a nadie", color=GRIS)
ax.axvline(t_opt, color=rdp.estilo.TINTA, lw=1.2, ls="--")
ax.text(t_opt + 0.01, 0.43, r"$t^\ast = 1/6$", color=rdp.estilo.TINTA, fontsize=13)
ax.axvline(0.5, color=rdp.estilo.TENUE, lw=1.2, ls="--")
ax.text(0.51, 0.43, "0.5", color=rdp.estilo.TENUE, fontsize=13)
rdp.estilo.etiquetar(ax, 0.81, {"logística": costos["logística"][-1], "bosque": costos["bosque"][-1]})
ax.set(xlim=(0, 0.9), ylim=(0.4, 1.2), title="Costo promedio por cliente según el umbral (prueba)",
       xlabel="umbral t", ylabel="costo promedio (unidades de 5,000 NT$)")
plt.show()

# %%
resumen = []
for nombre, p in [("logística", p_log), ("bosque", p_bos)]:
    i = np.argmin(costos[nombre])
    resumen.append({"modelo": nombre, "costo con t = 0.5": costo(y_te, p, 0.5),
                    "costo con t* = 1/6": costo(y_te, p, t_opt),
                    "mejor umbral en prueba": umbrales[i], "costo mínimo": costos[nombre][i]})
pd.DataFrame(resumen).set_index("modelo").round(3)

# %% [markdown]
# Para el bosque, pasar del umbral 0.5 al umbral por costos baja el costo esperado de forma notable, y $t^\ast$ queda muy cerca del mínimo de la curva. Para la logística, en cambio, el mínimo empírico está en otro lado: su curva de costo no tiene el mínimo donde la teoría dice. La teoría no falla; falla su hipótesis. Las probabilidades de la logística no están calibradas.
#
# (Una advertencia de método: elegir «el mejor umbral en prueba» y reportar su costo en prueba es hacer trampa, igual que elegir hiperparámetros con el conjunto de prueba. Lo mostramos solo como referencia. El umbral teórico $t^\ast$ no usa los datos de prueba para nada; un umbral empírico honesto se elegiría con validación cruzada en entrenamiento.)

# %% [markdown]
# ## 5. Calibración
#
# Un modelo está **calibrado** si, entre todos los clientes a los que les asigna probabilidad $p$, la fracción que incumple es $p$:
#
# $$
# P\big(Y = 1 \mid s(\mathbf{X}) = p\big) = p \quad \text{para todo } p.
# $$
#
# Para revisarlo, el **diagrama de confiabilidad** agrupa a los clientes por su puntaje (aquí en diez grupos del mismo tamaño), y grafica la frecuencia observada de incumplimiento contra el puntaje promedio de cada grupo. Un modelo calibrado queda sobre la diagonal.
#
# Dos métricas resumen la calidad de las probabilidades (no solo del orden):
#
# $$
# \text{Brier} = \frac{1}{n}\sum_{i=1}^n (s_i - y_i)^2, \qquad
# \text{log-loss} = -\frac{1}{n}\sum_{i=1}^n \big[y_i \ln s_i + (1 - y_i)\ln(1 - s_i)\big].
# $$
#
# El puntaje de Brier es el error cuadrático de la probabilidad; la log-loss es la entropía cruzada que minimiza la regresión logística. Ambas son **reglas de puntuación propias**: su valor esperado se minimiza cuando el modelo reporta la probabilidad verdadera, así que premian a la vez ordenar bien y estar calibrado. Como referencia, el modelo que a todos les asigna la prevalencia tiene Brier $\pi(1 - \pi) \approx 0.172$.

# %%
def diagrama(ax, y, p, color, nombre, marcador="o", ls="-"):
    frec, media = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(media, frec, marcador, ls=ls, color=color, mec="white", label=nombre)

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot([0, 1], [0, 1], color=GRIS, lw=1.5, ls="--")
diagrama(ax, y_te, p_log, AZUL, "logística")
diagrama(ax, y_te, p_bos, ROJO, "bosque")
ax.text(0.86, 0.60, "calibración\nperfecta", color=GRIS, ha="center")
ax.text(0.68, 0.71, "bosque", color=ROJO)
ax.text(0.53, 0.70, "logística", color=AZUL, ha="right")
ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title="Diagrama de confiabilidad (prueba, deciles)",
       xlabel="probabilidad predicha (promedio del grupo)", ylabel="frecuencia observada de incumplimiento")
plt.show()

def calidad(y, p):
    return {"Brier": brier_score_loss(y, p), "log-loss": log_loss(y, p), "AUC": roc_auc_score(y, p)}

pd.DataFrame({"logística": calidad(y_te, p_log), "bosque": calidad(y_te, p_bos)}).round(4)

# %% [markdown]
# La logística está mal calibrada, y no de una sola forma. En los puntajes bajos **subestima** el riesgo (dice 4 % cuando incumple casi 10 %); en la zona media, entre 0.15 y 0.25, lo **sobreestima** (dice alrededor de 20 % cuando incumple 13 %); y en el decil más alto vuelve a subestimarlo (dice 56 % cuando incumple 70 %). La razón es que la logística supone que el logaritmo de los momios es **lineal** en las variables; aquí la relación real es muy no lineal (un atraso de dos meses es mucho peor que dos veces un atraso de uno, y los códigos −2, −1 y 0 no están en ninguna escala), y el modelo compensa como puede. Ahora se entiende la sección 4: los clientes a los que la logística les da entre 1/6 y 0.22 en realidad incumplen menos de 1/6 de las veces, así que $t^\ast$ los señala sin que convenga, y el mejor umbral empírico se corre hacia arriba.
#
# El bosque está mucho más cerca de la diagonal, aunque no es perfecto.
#
# ### Recalibrar
#
# Recalibrar es aprender una función creciente $g$ tal que $g(s(\mathbf{x}))$ esté calibrada. Como $g$ es creciente, **no cambia el orden** de los puntajes (casi no cambia el AUC), pero sí corrige lo que significan. `CalibratedClassifierCV` lo hace con validación cruzada: entrena el modelo en 4/5 del conjunto de entrenamiento y ajusta $g$ con el quinto restante, cinco veces, y promedia. Hay dos opciones para $g$:
#
# - **sigmoide** (método de Platt): $g(s) = \sigma(a s + b)$, dos parámetros; robusta con pocos datos pero solo corrige distorsiones con forma de «S»;
# - **isotónica**: la función escalonada creciente que mejor ajusta los datos por mínimos cuadrados; más flexible, necesita más datos (aquí tenemos 22,500, suficientes).

# %%
log_iso = CalibratedClassifierCV(logistica, method="isotonic", cv=5).fit(X_tr, y_tr)
log_sig = CalibratedClassifierCV(logistica, method="sigmoid", cv=5).fit(X_tr, y_tr)
bos_iso = CalibratedClassifierCV(bosque, method="isotonic", cv=5).fit(X_tr, y_tr)

p_log_iso = log_iso.predict_proba(X_te)[:, 1]
p_log_sig = log_sig.predict_proba(X_te)[:, 1]
p_bos_iso = bos_iso.predict_proba(X_te)[:, 1]

pd.DataFrame({
    "logística": calidad(y_te, p_log),
    "logística + sigmoide": calidad(y_te, p_log_sig),
    "logística + isotónica": calidad(y_te, p_log_iso),
    "bosque": calidad(y_te, p_bos),
    "bosque + isotónica": calidad(y_te, p_bos_iso),
}).T.round(4)

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
for ax, (nombre, crudo, recal, color) in zip(axs, [("logística", p_log, p_log_iso, AZUL),
                                                     ("bosque", p_bos, p_bos_iso, ROJO)]):
    ax.plot([0, 1], [0, 1], color=GRIS, lw=1.5, ls="--")
    diagrama(ax, y_te, crudo, GRIS, "original", marcador="s", ls=":")
    diagrama(ax, y_te, recal, color, "recalibrado")
    ax.text(0.97, 0.05, "gris: original\ncolor: isotónica", ha="right", color=rdp.estilo.TENUE)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=f"{nombre}: antes y después",
           xlabel="probabilidad predicha", ylabel="frecuencia observada")
plt.show()

# %% [markdown]
# La sigmoide apenas ayuda a la logística: su distorsión no es una «S» simple, y una sigmoide aplicada a la salida de otra sigmoide no tiene mucha libertad. La isotónica sí la endereza, y baja su Brier y su log-loss; el AUC prácticamente no cambia, como se esperaba. El bosque, que ya estaba razonablemente calibrado, mejora poco.
#
# Ahora la prueba de fuego: ¿funciona el umbral teórico con las probabilidades recalibradas?

# %%
filas = []
for nombre, p in [("logística", p_log), ("logística + isotónica", p_log_iso),
                  ("bosque", p_bos), ("bosque + isotónica", p_bos_iso)]:
    c = np.array([costo(y_te, p, t) for t in umbrales])
    filas.append({"modelo": nombre, "costo con t* = 1/6": costo(y_te, p, t_opt),
                  "mejor umbral en prueba": umbrales[c.argmin()], "costo mínimo": c.min()})
pd.DataFrame(filas).set_index("modelo").round(3)

# %% [markdown]
# Con la logística recalibrada, el umbral teórico $t^\ast$ queda cerca del mejor umbral empírico y su costo baja. Esta es la razón práctica de la calibración: **si las probabilidades significan lo que dicen, el umbral se calcula con los costos y no hay que buscarlo a ciegas**. Y si mañana el banco cambia sus costos, basta recalcular $t^\ast$, sin reentrenar ni volver a buscar.
#
# En lo que sigue usamos el bosque recalibrado con $t^\ast = 1/6$.

# %% [markdown]
# ## 6. Equidad: las tasas por grupo
#
# El puntaje se aplica a personas, y es razonable preguntar si trata igual a distintos grupos. En muchos países la regulación limita el uso del sexo en las decisiones de crédito; nuestro modelo **usa** la variable `sexo` (un banco real probablemente no podría), pero aunque la quitáramos, otras variables correlacionadas con ella podrían reproducir las diferencias. Medir es el primer paso.
#
# Empecemos por lo más básico: ¿incumplen igual hombres y mujeres en estos datos?

# %%
sexo_te = X_te["sexo"].map({1: "hombres", 2: "mujeres"}).to_numpy()
grupos = ["hombres", "mujeres"]
pd.DataFrame({g: {"clientes en prueba": (sexo_te == g).sum(),
                  "prevalencia (incumple)": y_te[sexo_te == g].mean(),
                  "prevalencia en todos los datos": credito.loc[credito["sexo"] == (1 if g == "hombres" else 2), "incumple"].mean()}
              for g in grupos}).T.round(3)

# %% [markdown]
# Los hombres incumplen algo más (24 % contra 21 %). Esta diferencia de **prevalencias** está en los datos, no la inventó el modelo; puede venir de muchas causas (ingreso, edad, uso de la tarjeta) que la tabla no permite separar. Ahora veamos cómo trata el modelo a cada grupo con el umbral elegido.

# %%
s_final = p_bos_iso
tabla = {}
for g in grupos:
    m = sexo_te == g
    t = tasas(y_te[m], s_final[m], t_opt)
    tabla[g] = {"prevalencia": y_te[m].mean(),
                "puntaje promedio": s_final[m].mean(),
                "tasa de selección": t["tasa de selección"],
                "exhaustividad (TVP)": t["exhaustividad (TVP)"],
                "TFP": t["TFP"],
                "precisión (VPP)": t["precisión (VPP)"]}
equidad = pd.DataFrame(tabla).T
equidad.round(3)

# %%
metricas = ["tasa de selección", "exhaustividad (TVP)", "TFP", "precisión (VPP)"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metricas))
for k, (g, color) in enumerate(zip(grupos, [AZUL, DORADO_OSC])):
    barras = ax.bar(x + (k - 0.5) * 0.38, equidad.loc[g, metricas], width=0.36, color=color)
    ax.bar_label(barras, fmt="%.2f", padding=3, fontsize=11, color=rdp.estilo.TINTA)
ax.set_xticks(x, metricas)
ax.set(ylim=(0, 0.75), title="Bosque recalibrado con umbral 1/6: tasas por sexo (prueba)", ylabel="tasa")
ax.text(3.15, 0.66, "hombres", color=AZUL, fontsize=13)
ax.text(3.15, 0.60, "mujeres", color=DORADO_OSC, fontsize=13)
ax.grid(axis="x", visible=False)
plt.show()

# %% [markdown]
# Con el mismo umbral para todos, el modelo señala a una fracción mayor de hombres, y también tiene con ellos una tasa de falsos positivos más alta: un hombre que sí iba a pagar tiene más probabilidad de que le bajen el límite que una mujer que iba a pagar. ¿Es injusto? Depende de qué entendamos por justo, y aquí aparece un resultado matemático importante.
#
# ### Tres criterios que no se pueden cumplir a la vez
#
# Hay al menos tres nociones razonables de trato igual:
#
# 1. **Calibración por grupo**: un puntaje de 0.3 significa 30 % de riesgo tanto en hombres como en mujeres. Si no, el mismo número querría decir cosas distintas según el sexo.
# 2. **Igualdad de tasas de error** (*equalized odds*): misma $TVP$ y misma $TFP$ en los dos grupos. Un cumplido tiene la misma probabilidad de ser señalado por error, sea quien sea.
# 3. **Igual valor predictivo**: misma precisión; ser señalado significa lo mismo en los dos grupos.
#
# Chouldechova (2017) observó que, por la fórmula de Bayes de la sección 2 aplicada a cada grupo, estas cantidades están amarradas por una identidad:
#
# $$
# TFP = \frac{\pi}{1 - \pi} \cdot \frac{1 - VPP}{VPP} \cdot TVP.
# $$
#
# Si dos grupos tienen **prevalencias distintas**, el factor $\pi/(1-\pi)$ es distinto, y entonces no pueden tener a la vez la misma $VPP$, la misma $TVP$ y la misma $TFP$: al igualar dos, la tercera queda forzada a diferir. Kleinberg, Mullainathan y Raghavan (2016) probaron una versión más general para puntajes: salvo que el predictor sea perfecto o que las prevalencias sean iguales, **ningún puntaje puede estar calibrado en cada grupo y a la vez balanceado en sus errores** entre grupos. No es un defecto de este modelo ni de este algoritmo; es aritmética.
#
# Verifiquemos la identidad con nuestros números:

# %%
for g in grupos:
    pi, vpp, tvp = equidad.loc[g, "prevalencia"], equidad.loc[g, "precisión (VPP)"], equidad.loc[g, "exhaustividad (TVP)"]
    print(f"{g:8s}  TFP medida = {equidad.loc[g, 'TFP']:.4f}   "
          f"π/(1−π)·(1−VPP)/VPP·TVP = {pi / (1 - pi) * (1 - vpp) / vpp * tvp:.4f}")

# %% [markdown]
# Coinciden exactamente (es una identidad). Ahora revisemos la calibración por grupo del puntaje que usamos:

# %%
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot([0, 1], [0, 1], color=GRIS, lw=1.5, ls="--")
for g, color in zip(grupos, [AZUL, DORADO_OSC]):
    m = sexo_te == g
    diagrama(ax, y_te[m], s_final[m], color, g)
ax.text(0.28, 0.40, "hombres", color=AZUL, ha="right")
ax.text(0.32, 0.22, "mujeres", color=DORADO_OSC)
ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title="Calibración por sexo (bosque recalibrado)",
       xlabel="probabilidad predicha", ylabel="frecuencia observada")
plt.show()

# %% [markdown]
# El puntaje está razonablemente calibrado en los dos grupos (con el ruido esperable de deciles de 300 a 450 clientes), y precisamente por eso, al tener los hombres mayor prevalencia, reciben más puntajes altos y caen más veces sobre el umbral, incluidos los cumplidos: sus tasas de error no pueden ser iguales.
#
# ¿Y si forzamos igualdad de $TFP$ con un umbral distinto para cada grupo? Busquemos el umbral para hombres que les dé la misma $TFP$ que a las mujeres con $t^\ast$:

# %%
tfp_muj = equidad.loc["mujeres", "TFP"]
m_h = sexo_te == "hombres"
cumplidos_h = s_final[m_h & (y_te == 0)]
candidatos = np.unique(cumplidos_h)
tfp_h = np.array([(cumplidos_h >= t).mean() for t in candidatos])
t_h = candidatos[np.abs(tfp_h - tfp_muj).argmin()]  # el umbral cuya TFP en hombres más se acerca
ajuste = pd.DataFrame({
    "mujeres (t = 1/6)": tasas(y_te[~m_h], s_final[~m_h], t_opt),
    f"hombres (t = {t_h:.3f})": tasas(y_te[m_h], s_final[m_h], t_h),
}).loc[["TFP", "exhaustividad (TVP)", "precisión (VPP)", "tasa de selección"]]
ajuste.round(3)

# %% [markdown]
# Al subirles un poco el umbral a los hombres, las $TFP$ se igualan, pero ahora difieren la exhaustividad (0.70 contra 0.73) y, más todavía, la precisión (0.44 contra 0.38), justo como obliga la identidad: ahora a un hombre se le exige más riesgo que a una mujer para actuar, lo que viola la calibración como criterio de decisión (el mismo riesgo recibe un trato distinto), y además el banco paga más en incumplimientos no detectados. Ninguna opción es «neutral»: decidir qué criterio de equidad importa más es una decisión de política, del banco, del regulador y de la sociedad, no del algoritmo. Lo que sí le corresponde a quien construye el modelo es **medir y reportar** estas tasas por grupo, para que esa decisión se tome con información.
#
# *Referencias:* J. Kleinberg, S. Mullainathan y M. Raghavan, «Inherent trade-offs in the fair determination of risk scores», *ITCS* (2017; versión en arXiv de 2016). A. Chouldechova, «Fair prediction with disparate impact», *Big Data* 5(2), 2017.

# %% [markdown]
# ## 7. Cierre: qué métrica responde qué pregunta
#
# | Pregunta | Métrica | Depende del umbral | Depende de la prevalencia |
# |---|---|---|---|
# | ¿Acierta en general? | exactitud (engañosa con clases desbalanceadas) | sí | sí |
# | ¿Cuántos positivos detecta? | exhaustividad, $TVP$ | sí | no |
# | ¿Cuántos negativos molesta? | $TFP$ = 1 − especificidad | sí | no |
# | De los que señala, ¿cuántos son? | precisión, $VPP$ | sí | sí |
# | Un solo número que balancee precisión y exhaustividad | $F_1$ | sí | sí |
# | ¿Ordena bien, sin importar el umbral? | AUC (curva ROC) | no | no |
# | ¿Ordena bien en la clase rara? | precisión promedio (curva PR) | no | sí |
# | ¿Sus probabilidades significan lo que dicen? | diagrama de confiabilidad, Brier, log-loss | no | sí |
# | ¿Qué umbral usar? | costo esperado; $t^\ast = c_{FP}/(c_{FP}+c_{FN})$ si está calibrado | — | — |
# | ¿Trata igual a los grupos? | $TVP$, $TFP$, $VPP$, selección y calibración por grupo | sí | sí |
#
# Lo más importante: **no hay una métrica correcta en abstracto**. La métrica se elige a partir de la decisión que se va a tomar con el modelo y de lo que cuesta equivocarse.

# %% [markdown]
# ## 8. Autoevaluación

# %%
rdp.quiz.mostrar("u03", ids=["u03-p12", "u03-p13", "u03-p14"])

# %% [markdown]
# ## 9. Ejercicios
#
# 1. Repite la sección 4 con $c_{FN} = 2$ y con $c_{FN} = 10$ (dejando $c_{FP} = 1$). ¿Cómo cambia $t^\ast$? ¿Cuál es la tasa de selección en cada caso? Explica por qué, si el banco solo puede llamar a 500 clientes al mes, el umbral queda determinado por el presupuesto y no por los costos.
# 2. Demuestra que $F_1 = \dfrac{2\,VP}{2\,VP + FP + FN}$ a partir de su definición como media armónica. Construye un ejemplo en el que dos clasificadores tengan la misma $F_1$ pero matrices de confusión muy distintas.
# 3. Toma el conjunto de prueba y quita al azar la mitad de los incumplidos (la prevalencia baja a ~12 %). Recalcula AUC, precisión promedio y precisión con $t = 1/6$ para el bosque. ¿Cuáles cambian y cuáles no? Compara con la última columna de la tabla de la sección 7.
# 4. Demuestra que el AUC no cambia si a todos los puntajes se les aplica una función estrictamente creciente, y que el puntaje de Brier sí. Verifícalo numéricamente con $s \mapsto s^3$.
# 5. Calcula el AUC por pares para la logística con un ciclo de Python en lugar de NumPy, con una muestra de 500 incumplidos y 500 cumplidos. ¿Cuánto más tarda? Calcula además un intervalo de confianza *bootstrap* del 95 % para la diferencia de AUC entre el bosque y la logística.
# 6. Quita la variable `sexo` de los datos, reentrena el bosque y recalibra. ¿Cambian las tasas por sexo de la sección 6? ¿Por qué no desaparecen las diferencias?
