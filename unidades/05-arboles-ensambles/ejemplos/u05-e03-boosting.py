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
# # *Boosting* y una campaña de mercadeo bancario
#
# **Unidad 5 · Ejemplo 3**
#
# El *bagging* y los bosques aleatorios toman árboles profundos, con poco sesgo y mucha varianza, y los promedian para quitarles la varianza. El *boosting* hace casi lo contrario: toma modelos muy simples, con mucho sesgo y poca varianza, y los **suma uno tras otro**, cada uno encargado de corregir lo que los anteriores hicieron mal. Es la familia de modelos que gana casi todas las competencias con datos tabulares. En esta libreta vamos a:
#
# 1. programar **AdaBoost** desde cero con tocones (árboles de una sola pregunta) y ver cómo los pesos se concentran en los datos difíciles;
# 2. demostrar la cota de su **error de entrenamiento** y derivar el peso de cada tocón como el paso óptimo de un **modelo aditivo por etapas con pérdida exponencial**;
# 3. generalizar a ***gradient boosting***: descenso por gradiente en el espacio de funciones, programado en pocas líneas y verificado contra scikit-learn con las bicicletas de la Unidad 4;
# 4. predecir quién contrata un depósito a plazo en una **campaña de mercadeo telefónico** de un banco, caer a propósito en una **fuga de información** y salir de ella;
# 5. comparar logística, bosque, `HistGradientBoostingClassifier`, XGBoost, LightGBM y CatBoost, y estudiar la **tasa de aprendizaje**, el **número de árboles** y la **parada temprana**;
# 6. derivar el **peso óptimo de una hoja** y la **ganancia de una división** de XGBoost y comprobarlos contra la biblioteca;
# 7. ver por qué codificar una categoría con la respuesta crea una fuga, y cómo la evita CatBoost.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado. XGBoost y LightGBM vienen preinstalados en Colab; CatBoost se instala aquí.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    # %pip install -q catboost
    pass

# %%
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from threadpoolctl import threadpool_limits
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE
warnings.filterwarnings("ignore", message=".*eval_set.*")  # aviso de LightGBM 4.7 sobre el nombre de un argumento

# %% [markdown]
# # Parte 1. AdaBoost
#
# ## 1. Una pregunta teórica con respuesta práctica
#
# Entre 1988 y 1989, Michael Kearns y Leslie Valiant plantearon una pregunta dentro de la teoría del aprendizaje PAC (la de la Unidad 3): un **aprendiz débil** es un algoritmo que, para cualquier distribución de los datos, produce una hipótesis apenas mejor que lanzar una moneda, con error $\le \tfrac12 - \gamma$ para algún $\gamma > 0$ fijo. Un **aprendiz fuerte** alcanza error tan pequeño como se quiera. ¿Son equivalentes? ¿Se puede **impulsar** (*boost*) un aprendiz débil hasta convertirlo en uno fuerte?
#
# Robert Schapire demostró en 1990 que sí («The strength of weak learnability», *Machine Learning* 5), con una construcción recursiva poco práctica. Yoav Freund y Schapire propusieron en 1995 (publicado en 1997) un algoritmo sencillo, **AdaBoost** (*adaptive boosting*), por el que recibieron el premio Gödel en 2003. La idea:
#
# - entrenar un aprendiz débil con los datos **ponderados**;
# - **subir el peso** de los datos que clasificó mal y bajar el de los que clasificó bien;
# - entrenar el siguiente aprendiz con los nuevos pesos, que lo obligan a fijarse en los casos difíciles;
# - al final, que voten todos, cada uno con un peso que depende de qué tan bueno fue.
#
# ### Los datos y el problema
#
# **Datos.** Los pingüinos del archipiélago Palmer (Antártida) que conocimos en la Unidad 2: 344 pingüinos de tres especies (Adelia, barbijo y papúa; `Adelie`, `Chinstrap` y `Gentoo` en la tabla) medidos por Kristen Gorman entre 2007 y 2009 y publicados por Allison Horst, Alison Hill y Gorman en el paquete `palmerpenguins` (2020, licencia CC0).
#
# **Problema.** Distinguir a los pingüinos **barbijo** ($y = +1$) de los de las otras dos especies ($y = -1$) con dos medidas del pico: longitud y profundidad. Los barbijo tienen el pico **largo y profundo** a la vez: los Adelia lo tienen profundo pero corto, y los papúa largo pero delgado. Ninguna pregunta del tipo «¿$x_j > t$?» los separa por sí sola; hace falta combinar las dos.
#
# El aprendiz débil será un **tocón** (*decision stump*): un árbol de profundidad 1, que hace una sola pregunta. Un tocón aislado es un clasificador malo, con mucho sesgo; es exactamente lo que AdaBoost necesita.

# %%
ping = rdp.datos.cargar("pinguinos").dropna(subset=["bill_length_mm", "bill_depth_mm"])
Xp = ping[["bill_length_mm", "bill_depth_mm"]].to_numpy()
yp = np.where(ping["species"] == "Chinstrap", 1, -1)
print(f"{len(yp)} pingüinos, {np.sum(yp == 1)} barbijo ({np.mean(yp == 1):.1%})")

# %% [markdown]
# ## 2. El algoritmo
#
# Con etiquetas $y_n \in \{-1, +1\}$ y pesos iniciales $w_n^{(1)} = 1/N$, para $t = 1, \dots, T$:
#
# 1. ajustar un clasificador débil $h_t(\mathbf{x}) \in \{-1, +1\}$ a los datos con pesos $w^{(t)}$;
# 2. calcular su **error ponderado** $\varepsilon_t = \sum_n w_n^{(t)}\,\mathbb{1}[h_t(\mathbf{x}_n) \ne y_n]$;
# 3. darle el **voto** $\alpha_t = \tfrac12 \ln\dfrac{1 - \varepsilon_t}{\varepsilon_t}$ (positivo si $\varepsilon_t < \tfrac12$);
# 4. actualizar los pesos: $w_n^{(t+1)} = w_n^{(t)}\, e^{-\alpha_t y_n h_t(\mathbf{x}_n)} / Z_t$, donde $Z_t$ normaliza para que sumen 1.
#
# El clasificador final es $g(\mathbf{x}) = \operatorname{signo}\big(F_T(\mathbf{x})\big)$ con $F_T(\mathbf{x}) = \sum_{t=1}^T \alpha_t h_t(\mathbf{x})$.
#
# En el paso 4, $y_n h_t(\mathbf{x}_n) = +1$ si el tocón acertó y $-1$ si se equivocó: los aciertos se multiplican por $e^{-\alpha_t} < 1$ y los errores por $e^{\alpha_t} > 1$.
#
# ### El aprendiz débil: un tocón que minimiza el error ponderado
#
# Un tocón es $h(\mathbf{x}) = s$ si $x_j > u$ y $-s$ si no, con $s \in \{-1, +1\}$. Buscamos el atributo $j$, el umbral $u$ y el signo $s$ con menor error ponderado. Si ordenamos los datos por $x_j$ y cortamos después del $k$-ésimo, con $s = +1$ el tocón se equivoca con los positivos de la izquierda y los negativos de la derecha:
#
# $$\varepsilon(k) = \sum_{i \le k,\; y_i = +1} w_i \;+\; \sum_{i > k,\; y_i = -1} w_i ,$$
#
# y con $s = -1$ el error es $1 - \varepsilon(k)$. Con sumas acumuladas, todos los cortes de un atributo cuestan $O(N)$ después de ordenar.

# %%
class Tocon:
    """Árbol de una pregunta, h(x) = s si x_j > u y −s si no, que minimiza el error ponderado."""

    def fit(self, X, y, sample_weight):
        w = sample_weight
        self.error = np.inf
        for j in range(X.shape[1]):
            orden = np.argsort(X[:, j], kind="stable")
            xs, ys, ws = X[orden, j], y[orden], w[orden]
            pos_izq = np.r_[0, np.cumsum(ws * (ys == 1))]                 # peso de positivos a la izquierda del corte k
            neg_der = w[y == -1].sum() - np.r_[0, np.cumsum(ws * (ys == -1))]
            err = pos_izq + neg_der                                         # error con s = +1, para k = 0, …, N
            valido = np.r_[False, xs[1:] != xs[:-1], False]                 # solo cortes entre valores distintos
            for s, e in ((1, err), (-1, w.sum() - err)):
                k = np.flatnonzero(valido)[np.argmin(e[valido])]
                if e[k] < self.error:
                    self.error, self.j, self.u, self.s = e[k], j, (xs[k - 1] + xs[k]) / 2, s
        return self

    def predict(self, X):
        return self.s * np.where(X[:, self.j] > self.u, 1, -1)


def adaboost(X, y, T, aprendiz=Tocon):
    """AdaBoost discreto. y en {-1, +1}. Devuelve tocones, votos y la historia de pesos y errores."""
    N = len(y)
    w = np.full(N, 1 / N)
    F = np.zeros(N)
    tocones, alfas, pesos, eps = [], [], [w], []
    for t in range(T):
        h = aprendiz().fit(X, y, sample_weight=w)
        pred = h.predict(X)
        e = w[pred != y].sum()
        a = 0.5 * np.log((1 - e) / e)
        w = w * np.exp(-a * y * pred)
        w = w / w.sum()
        F += a * pred
        tocones.append(h); alfas.append(a); pesos.append(w); eps.append(e)
    return tocones, np.array(alfas), np.array(pesos), np.array(eps)


def F_ada(X, tocones, alfas):
    """Puntaje F_T(x) = suma de alfa_t h_t(x)."""
    return sum(a * h.predict(X) for h, a in zip(tocones, alfas))


T = 200
tocones, alfas, pesos, eps = adaboost(Xp, yp, T)
error_ent = np.mean(np.sign(F_ada(Xp, tocones, alfas)) != yp)
nombres_p = ["longitud", "profundidad"]
for t in range(3):
    h = tocones[t]
    print(f"tocón {t + 1}: {'barbijo' if h.s == 1 else 'otra'} si {nombres_p[h.j]} > {h.u:.2f} mm;"
          f"  ε = {eps[t]:.3f}, α = {alfas[t]:.3f}")
print(f"error de entrenamiento con {T} tocones: {error_ent:.3f}")

# %% [markdown]
# El primer tocón se equivoca con el 17 % de los pingüinos ponderados (todos pesan igual al principio), apenas mejor que decir siempre «otra especie», que se equivoca con el 20 %. Doscientos tocones juntos se equivocan con el 2.9 %.
#
# **Comparación con scikit-learn.** `AdaBoostClassifier` usa el algoritmo SAMME (Zhu et al., 2009), que para dos clases es AdaBoost con los votos multiplicados por 2: no cambia el signo de $F$ ni la actualización de pesos, que se normalizan. Su aprendiz débil por omisión es `DecisionTreeClassifier(max_depth=1)`, que elige la división por **Gini** ponderado y no por error ponderado. Para comparar el algoritmo y no el aprendiz, usamos nuestro `adaboost` con ese mismo árbol:

# %%
tocones_sk, alfas_sk, _, eps_sk = adaboost(Xp, yp, T, aprendiz=lambda: DecisionTreeClassifier(max_depth=1))
ada_sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=T, random_state=SEMILLA).fit(Xp, yp)
print("votos de scikit-learn / nuestros votos (primeros 5):", (ada_sk.estimator_weights_[:5] / alfas_sk[:5]).round(6))
print("¿mismas predicciones de entrenamiento?", np.array_equal(ada_sk.predict(Xp), np.sign(F_ada(Xp, tocones_sk, alfas_sk))))
print(f"primer árbol de Gini: ε = {eps_sk[0]:.3f}; predice {np.unique(tocones_sk[0].predict(Xp))} a ambos lados del corte")

# %% [markdown]
# El algoritmo coincide. Y aparece una curiosidad: el primer tocón de Gini tiene error 0.199, exactamente la proporción de barbijos, porque en **las dos** hojas la mayoría es «otra especie» y predice lo mismo a ambos lados. Es un clasificador constante, pero no inútil: reparte la mitad del peso entre los barbijos y la siguiente ronda ya hace una pregunta útil. Con nuestro tocón el primer paso es más informativo.

# %% [markdown]
# ### Dónde se concentran los pesos
#
# Dibujemos el clasificador $\operatorname{signo}(F_t)$ después de $t$ rondas, con el área de cada punto proporcional a su peso $w^{(t+1)}$, el que usará el siguiente tocón. La línea punteada es el tocón $h_t$ recién agregado.

# %%
g1, g2 = np.meshgrid(np.linspace(31, 61, 241), np.linspace(12.5, 22, 191))
malla = np.column_stack([g1.ravel(), g2.ravel()])
rondas = [1, 2, 3, 10, 50, 200]
fig, ejes = plt.subplots(2, 3, figsize=(14, 8.6), sharex=True, sharey=True)
for ax, t in zip(ejes.ravel(), rondas):
    zona = np.sign(F_ada(malla, tocones[:t], alfas[:t])).reshape(g1.shape)
    ax.contourf(g1, g2, zona, levels=[-2, 0, 2], colors=["#C0CEE3", "#F3C7C4"], alpha=0.8)
    ax.scatter(Xp[:, 0], Xp[:, 1], s=4000 * pesos[t], c=np.where(yp == 1, ROJO, AZUL),
               edgecolor="white", lw=0.5, alpha=0.85)
    h = tocones[t - 1]
    (ax.axvline if h.j == 0 else ax.axhline)(h.u, color=TINTA, ls="--", lw=1.4)
    err = np.mean(np.sign(F_ada(Xp, tocones[:t], alfas[:t])) != yp)
    ax.set_title(f"t = {t}: error de entrenamiento {err:.3f}", fontsize=12)
for ax in ejes[1]:
    ax.set_xlabel("longitud del pico (mm)")
for ax in ejes[:, 0]:
    ax.set_ylabel("profundidad del pico (mm)")
ejes[0, 0].text(33, 13.2, "otras especies", color=AZUL, fontsize=11, fontweight="bold")
ejes[0, 0].text(53, 21.2, "barbijo", color=ROJO, fontsize=11, fontweight="bold")
fig.suptitle("AdaBoost con tocones: el área de cada punto es su peso para la siguiente ronda", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# El primer tocón dice «barbijo si el pico mide más de 50.5 mm». Se equivoca con los barbijos de pico más corto y con los papúa de pico largo, y esos puntos crecen: en la ronda 2 cargan la mitad del peso total. El segundo tocón, obligado a atenderlos, pregunta por la profundidad; el tercero vuelve a la longitud con otro umbral (45.2 mm). Ronda tras ronda, los pesos se concentran en la **frontera** entre clases y en los pocos pingüinos que se enciman; los que están en el centro de su grupo casi desaparecen. La frontera final es una escalera de segmentos horizontales y verticales que encierra la esquina de los picos largos y profundos. Con 200 rondas también aparecen franjas en zonas sin datos (a la izquierda) y entrantes alrededor de pingüinos aislados: el modelo empieza a memorizar.
#
# Una observación sobre la forma: $F_T(\mathbf{x}) = \sum_t \alpha_t h_t(\mathbf{x})$ con tocones es una **suma de funciones de una sola variable**, $F(\mathbf{x}) = f_1(x_1) + f_2(x_2)$: un modelo aditivo. Para que el ensamble capture interacciones entre atributos, los árboles base deben tener profundidad mayor que 1.
#
# ## 3. Cota del error de entrenamiento
#
# **Proposición** (Freund y Schapire, 1997). Si $\varepsilon_t = \tfrac12 - \gamma_t$, el error de entrenamiento de $g = \operatorname{signo}(F_T)$ cumple
#
# $$\frac1N\sum_n \mathbb{1}[g(\mathbf{x}_n) \ne y_n] \;\le\; \frac1N\sum_n e^{-y_n F_T(\mathbf{x}_n)} \;=\; \prod_{t=1}^T 2\sqrt{\varepsilon_t(1-\varepsilon_t)} \;=\; \prod_{t=1}^T\sqrt{1 - 4\gamma_t^2} \;\le\; e^{-2\sum_t\gamma_t^2}.$$
#
# *Esbozo de la demostración.*
#
# - **Primera desigualdad.** Si $g(\mathbf{x}_n) \ne y_n$ entonces $y_nF_T(\mathbf{x}_n) \le 0$ y $e^{-y_nF_T(\mathbf{x}_n)} \ge 1$; si acierta, el término exponencial sigue siendo positivo. La pérdida exponencial es una cota superior de la pérdida 0-1.
# - **Igualdad con el producto.** Desenrollando la actualización de pesos, $w_n^{(T+1)} = \dfrac{1}{N}\,\dfrac{e^{-y_n\sum_t\alpha_t h_t(\mathbf{x}_n)}}{\prod_t Z_t}$. Como los pesos suman 1, $\frac1N\sum_n e^{-y_nF_T(\mathbf{x}_n)} = \prod_t Z_t$.
# - **Cada factor.** $Z_t = \sum_n w_n^{(t)}e^{-\alpha_t y_n h_t(\mathbf{x}_n)} = (1-\varepsilon_t)e^{-\alpha_t} + \varepsilon_t e^{\alpha_t}$. Derivando respecto de $\alpha_t$ e igualando a cero se obtiene justamente $\alpha_t = \tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$, y sustituyendo, $Z_t = 2\sqrt{\varepsilon_t(1-\varepsilon_t)}$. Es decir: **el voto de AdaBoost es el que minimiza la cota en cada ronda**.
# - **La última.** $1 - x \le e^{-x}$ con $x = 4\gamma_t^2$.
#
# Consecuencia: si cada aprendiz débil tiene una ventaja de al menos $\gamma$ sobre la moneda, el error de entrenamiento cae **exponencialmente** con el número de rondas, como $e^{-2\gamma^2 T}$. Ésa es la respuesta afirmativa a la pregunta de Kearns y Valiant (para el error de entrenamiento; la generalización requiere además controlar la complejidad, como en la Unidad 3).
#
# Verifiquemos la igualdad y veamos la cota en los pingüinos:

# %%
F_hist = np.cumsum(alfas[:, None] * np.array([h.predict(Xp) for h in tocones]), axis=0)  # F_t en cada dato
err_t = np.mean(np.sign(F_hist) != yp, axis=1)
exp_t = np.mean(np.exp(-yp * F_hist), axis=1)
cota_t = np.cumprod(2 * np.sqrt(eps * (1 - eps)))
print("máx |pérdida exponencial − producto|:", np.abs(exp_t - cota_t).max())
for t in [1, 10, 50, 200]:
    print(f"t = {t:>3}: error {err_t[t - 1]:.3f}   cota {cota_t[t - 1]:.3f}   ε_t = {eps[t - 1]:.3f}")

# %%
ts = np.arange(1, T + 1)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(ts, cota_t, color=DORADO_OSC)
ax.plot(ts, err_t, color=AZUL, drawstyle="steps-post")
ax.set_yscale("log")
ax.set_xlim(0, 225)
ax.set_ylim(0.02, 1)
ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1], ["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
ax.text(150, cota_t[149] * 1.25, "cota ∏ 2√(ε(1−ε))", color=DORADO_OSC, fontsize=12, fontweight="bold")
ax.text(150, err_t[149] * 0.72, "error de entrenamiento", color=AZUL, fontsize=12, fontweight="bold")
ax.set_xlabel("ronda t")
ax.set_ylabel("proporción (escala logarítmica)")
ax.set_title("El error de entrenamiento de AdaBoost queda bajo la pérdida exponencial")
plt.show()

# %% [markdown]
# La cota es floja (0.108 contra un error real de 0.029 con 200 tocones) pero baja de forma sostenida. Su caída se hace más lenta con las rondas: los errores ponderados $\varepsilon_t$ suben hacia 0.5, porque los pesos ya están concentrados en los pingüinos difíciles y ningún tocón los separa bien. Aquí no llega a cero porque hay pingüinos de especies distintas con picos casi idénticos: la hipótesis del aprendiz débil (una ventaja $\gamma$ fija para *cualquier* distribución de pesos) no se cumple con tocones en estos datos.
#
# ## 4. AdaBoost es un modelo aditivo con pérdida exponencial
#
# Durante años AdaBoost fue un algoritmo que funcionaba muy bien sin que estuviera claro **qué** optimizaba. Friedman, Hastie y Tibshirani (*Annals of Statistics*, 2000) dieron una lectura estadística: es un **modelado aditivo por etapas** (*forward stagewise additive modeling*). Se construye $F_t = F_{t-1} + \alpha_t h_t$ eligiendo en cada etapa el par $(\alpha_t, h_t)$ que más reduce una pérdida, **sin tocar** los términos anteriores. Con la pérdida exponencial $L(y, F) = e^{-yF}$:
#
# $$\sum_n e^{-y_n(F_{t-1}(\mathbf{x}_n) + \alpha h(\mathbf{x}_n))} = \sum_n \underbrace{e^{-y_nF_{t-1}(\mathbf{x}_n)}}_{\propto\, w_n^{(t)}}\, e^{-\alpha y_n h(\mathbf{x}_n)}.$$
#
# Los factores $e^{-y_nF_{t-1}(\mathbf{x}_n)}$ **son** los pesos de AdaBoost (salvo normalización): cada etapa resuelve un problema ponderado. Separando aciertos y errores, con $\sum_n w_n = 1$,
#
# $$\sum_n w_n e^{-\alpha y_n h(\mathbf{x}_n)} = e^{-\alpha}(1-\varepsilon) + e^{\alpha}\varepsilon = e^{-\alpha} + (e^{\alpha} - e^{-\alpha})\,\varepsilon .$$
#
# 1. Para cualquier $\alpha > 0$, la mejor $h$ es la que minimiza el **error ponderado** $\varepsilon$: lo que hace el paso 1 del algoritmo.
# 2. Con $h$ fija, derivando respecto de $\alpha$: $-e^{-\alpha}(1-\varepsilon) + e^{\alpha}\varepsilon = 0 \Rightarrow e^{2\alpha} = \frac{1-\varepsilon}{\varepsilon} \Rightarrow \alpha = \tfrac12\ln\frac{1-\varepsilon}{\varepsilon}$.
#
# Comprobémoslo numéricamente en la ronda 5: minimizamos la pérdida exponencial sobre $\alpha$ con un optimizador genérico.

# %%
t = 5
F_ant = F_hist[t - 2]                    # F_4 en los datos
h_t = tocones[t - 1].predict(Xp)
perdida = lambda a: np.mean(np.exp(-yp * (F_ant + a * h_t)))
optimo = minimize_scalar(perdida, bounds=(0, 3), method="bounded", options={"xatol": 1e-10})
print(f"α que minimiza la pérdida exponencial: {optimo.x:.6f}")
print(f"fórmula ½ ln((1 − ε)/ε):               {alfas[t - 1]:.6f}")

# %% [markdown]
# Esta lectura tiene dos consecuencias. La primera: el minimizador poblacional de $\mathbb{E}[e^{-yF(\mathbf{x})}\mid\mathbf{x}]$ es $F^*(\mathbf{x}) = \tfrac12\ln\frac{P(y=1\mid\mathbf{x})}{P(y=-1\mid\mathbf{x})}$, la **mitad del log-momio** (ejercicio 2): AdaBoost estima una regresión logística aditiva, y de $F$ se pueden sacar probabilidades. La segunda, más importante: si el truco es «sumar modelos por etapas para bajar una pérdida», ¿por qué no usar **cualquier** pérdida? La exponencial castiga muchísimo a los datos muy mal clasificados ($e^{-yF}$ crece exponencialmente), lo que hace a AdaBoost sensible a etiquetas erróneas. Ésa es la puerta al *gradient boosting*.

# %% [markdown]
# # Parte 2. *Gradient boosting*
#
# ## 5. Descenso por gradiente en el espacio de funciones
#
# Queremos minimizar $\sum_n L(y_n, F(\mathbf{x}_n))$ sobre funciones $F$. Pensemos en el vector de valores $\mathbf{F} = (F(\mathbf{x}_1), \dots, F(\mathbf{x}_N))$ como los parámetros. Un paso de descenso por gradiente sería
#
# $$F_m(\mathbf{x}_n) = F_{m-1}(\mathbf{x}_n) + \eta\, r_{nm}, \qquad r_{nm} = -\left.\frac{\partial L(y_n, F)}{\partial F}\right|_{F = F_{m-1}(\mathbf{x}_n)} .$$
#
# Los $r_{nm}$ se llaman **pseudo-residuos**. El problema: ese paso solo dice qué hacer en los $N$ puntos de entrenamiento, y necesitamos una función definida en todas partes. La solución de Friedman (*Annals of Statistics*, 2001): **ajustar un árbol de regresión** $h_m$ a los pares $(\mathbf{x}_n, r_{nm})$, que generaliza la dirección de descenso a cualquier $\mathbf{x}$, y avanzar en esa dirección:
#
# $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\, h_m(\mathbf{x}).$$
#
# | pérdida | $L(y, F)$ | pseudo-residuo $r = -\partial L/\partial F$ | qué es |
# |---|---|---|---|
# | cuadrática | $\tfrac12(y - F)^2$ | $y - F$ | el residuo de siempre |
# | absoluta | $\lvert y - F\rvert$ | $\operatorname{signo}(y - F)$ | solo la dirección: robusta a atípicos |
# | logística, $y \in \{0,1\}$, $F$ = log-momio | $-y\,F + \ln(1 + e^{F})$ | $y - p$, con $p = \sigma(F)$ | etiqueta menos probabilidad |
# | exponencial, $y \in \{-1,+1\}$ | $e^{-yF}$ | $y\,e^{-yF}$ | el peso de AdaBoost con signo |
#
# Con la pérdida cuadrática el algoritmo es muy intuitivo: **cada árbol se ajusta a lo que los anteriores no explicaron**. Con la logística, a la diferencia entre la etiqueta y la probabilidad actual: los datos bien clasificados con confianza aportan casi nada. Es el mismo «datos por residuo» de los modelos lineales generalizados de la Unidad 4.
#
# El parámetro $\eta \in (0, 1]$ es la **tasa de aprendizaje** (*shrinkage*): en lugar de dar el paso completo, damos uno pequeño y dejamos que los árboles siguientes sigan corrigiendo. Lo estudiaremos en la sección 10.
#
# ### Desde cero, con las bicicletas
#
# **Datos.** Las rentas por hora de Capital Bikeshare (Washington, D. C., 2011–2012; Fanaee-T y Gama, 2014; UCI) de la Unidad 4, con la misma partición: días 1–19 de cada mes para entrenar, del 20 en adelante para probar. **Problema:** predecir `total`, las bicicletas rentadas en una hora. Usamos los atributos crudos, como en el ejemplo de árboles de esta unidad, sin *one-hot* ni interacciones hechas a mano.
#
# El algoritmo para la pérdida cuadrática cabe en unas líneas: empezar con la constante que minimiza la pérdida (el promedio) y repetir «calcular residuos, ajustarles un árbol, avanzar un poco».

# %%
b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
b["anio"] = b.fecha.dt.year - 2011          # 0 = 2011, 1 = 2012
b["clima"] = b.clima.clip(upper=3)          # la categoría 4 tiene 3 horas: se une a la 3
entrena = b[b.fecha.dt.day <= 19]           # 10,886 horas
prueba = b[b.fecha.dt.day >= 20]            # 6,493 horas
atributos = ["hora", "laboral", "clima", "anio", "temp", "humedad", "viento"]
Xb, yb = entrena[atributos].to_numpy(float), entrena["total"].to_numpy(float)
Xb_te, yb_te = prueba[atributos].to_numpy(float), prueba["total"].to_numpy(float)


def gradient_boosting(X, y, M, eta, profundidad, semilla=0):
    """Gradient boosting para pérdida cuadrática con árboles de regresión."""
    azar = np.random.RandomState(semilla)  # el mismo generador que usa scikit-learn, para comparar
    f0 = y.mean()
    F = np.full(len(y), f0)
    arboles = []
    for m in range(M):
        r = y - F                                                   # pseudo-residuos
        h = DecisionTreeRegressor(max_depth=profundidad, random_state=azar).fit(X, r)
        F += eta * h.predict(X)
        arboles.append(h)
    return f0, arboles


def predecir_gb(X, f0, arboles, eta):
    return f0 + eta * sum(h.predict(X) for h in arboles)


M, ETA, PROF = 300, 0.1, 4
t0 = time.perf_counter()
f0, arboles = gradient_boosting(Xb, yb, M, ETA, PROF)
t_nuestro = time.perf_counter() - t0
pred_nuestro = predecir_gb(Xb_te, f0, arboles, ETA)

t0 = time.perf_counter()
gb_sk = GradientBoostingRegressor(n_estimators=M, learning_rate=ETA, max_depth=PROF, random_state=0).fit(Xb, yb)
t_sk = time.perf_counter() - t0
pred_sk = gb_sk.predict(Xb_te)

print(f"máxima diferencia entre predicciones: {np.abs(pred_nuestro - pred_sk).max():.1e} bicicletas")
print(f"RECM de prueba: nuestro {root_mean_squared_error(yb_te, pred_nuestro):.1f}, "
      f"scikit-learn {root_mean_squared_error(yb_te, pred_sk):.1f}")
print(f"tiempo: nuestro {t_nuestro:.1f} s, scikit-learn {t_sk:.1f} s")

# %% [markdown]
# Coinciden hasta el error de redondeo de punto flotante: `GradientBoostingRegressor` con pérdida cuadrática es exactamente este ciclo. (Para que coincidan también los desempates entre divisiones igual de buenas, le pasamos a cada árbol el mismo generador aleatorio que usa scikit-learn.)
#
# ¿Qué tan bueno es? Lo comparamos con referencias que usan los mismos siete atributos y la misma partición: predecir siempre el promedio, una regresión lineal y un solo árbol de regresión de profundidad 10. (En la Unidad 4, con la interacción hora × día laboral construida a mano, el mejor modelo lineal llegó a 86.4.)

# %%
hgb_b = HistGradientBoostingRegressor(loss="poisson", random_state=0).fit(Xb, yb)
referencias = {
    "predecir el promedio": np.full(len(yb_te), yb.mean()),
    "regresión lineal": LinearRegression().fit(Xb, yb).predict(Xb_te),
    "un árbol de profundidad 10": DecisionTreeRegressor(max_depth=10, random_state=0).fit(Xb, yb).predict(Xb_te),
    "gradient boosting, pérdida cuadrática": pred_sk,
    "HistGradientBoosting, pérdida de Poisson": hgb_b.predict(Xb_te),
}
tabla_bicis = pd.Series({k: root_mean_squared_error(yb_te, v) for k, v in referencias.items()}).to_frame("RECM de prueba")
tabla_bicis.round(1)

# %% [markdown]
# El *boosting* le gana con holgura a la regresión lineal y al mejor árbol solo, sin que nadie le diga qué interacciones buscar: los árboles de profundidad 4 pueden combinar hora, tipo de día y temperatura en una misma hoja. Y la idea de la Unidad 4 sigue valiendo: con la pérdida de Poisson (pseudo-residuos $y - e^{F}$, con $F$ en escala logarítmica) baja un poco más.
#
# ### Cómo se construye la suma
#
# Para ver el mecanismo, reduzcamos el problema a un solo atributo: las rentas de los días laborales contra la hora, con tocones y $\eta = 0.5$. Cada tocón parte las horas en dos y corrige el promedio de un lado y del otro.

# %%
lab = entrena[entrena.laboral == 1]
xh, yh = lab[["hora"]].to_numpy(float), lab["total"].to_numpy(float)
f0_h, tocones_h = gradient_boosting(xh, yh, 300, 0.5, 1)
horas = np.arange(24.0)[:, None]
promedio_hora = lab.groupby("hora").total.mean()

fig, ejes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
sacudida = np.random.default_rng(0).uniform(-0.3, 0.3, len(xh))
for ax, m in zip(ejes, [1, 3, 20, 300]):
    ax.scatter(xh[:, 0] + sacudida, yh, s=3, color=GRIS, alpha=0.18)
    ax.step(horas[:, 0], predecir_gb(horas, f0_h, tocones_h[:m], 0.5), where="mid", color=AZUL, lw=2.5)
    ax.set_title(f"{m} {'tocón' if m == 1 else 'tocones'}")
    ax.set_xlabel("hora (días laborales)")
    ax.set_xticks([0, 6, 12, 18, 23])
ejes[0].set_ylabel("bicicletas por hora")
ejes[-1].scatter(promedio_hora.index, promedio_hora, s=22, color=DORADO_OSC, zorder=4)
ejes[-1].text(9.5, 640, "promedio por hora", color=DORADO_OSC, fontsize=11, fontweight="bold",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
ejes[0].set_ylim(0, 1000)
fig.suptitle("Gradient boosting con tocones: la suma de escalones va tomando la forma del día", fontsize=14)
fig.tight_layout()
plt.show()
print(f"máx |F_300 − promedio por hora| = {np.abs(predecir_gb(horas, f0_h, tocones_h, 0.5) - promedio_hora.to_numpy()).max():.2f} bicicletas")

# %% [markdown]
# El primer tocón separa la madrugada del resto del día; los siguientes van marcando el inicio de la actividad y la noche; con unas decenas aparecen los dos picos de un día laboral, y con 300 la suma de escalones casi coincide con el promedio por hora (los puntos dorados), que es lo mejor que se puede hacer con la pérdida cuadrática y un solo atributo discreto. Fíjate en que cada tocón aislado es un clasificador malísimo: solo dice «antes o después de cierta hora».
#
# ## 6. Más allá de la pérdida cuadrática: el paso de cada hoja
#
# Un árbol de regresión ajustado a los pseudo-residuos tiene la **forma** correcta (la partición del espacio) pero, para pérdidas que no son la cuadrática, el promedio de los pseudo-residuos en una hoja no es el mejor valor. Friedman propuso re-optimizar el valor de cada hoja $R_j$:
#
# $$\gamma_j = \arg\min_\gamma \sum_{\mathbf{x}_n \in R_j} L\big(y_n, F_{m-1}(\mathbf{x}_n) + \gamma\big).$$
#
# Para la pérdida absoluta, $\gamma_j$ es la **mediana** de los residuos de la hoja. Para la logística no hay forma cerrada, y se usa un paso de Newton: $\gamma_j \approx \sum_{n \in R_j}(y_n - p_n) \big/ \sum_{n \in R_j} p_n(1-p_n)$, el gradiente entre la segunda derivada. Es lo que hace `GradientBoostingClassifier`, y es la semilla de XGBoost (sección 11).
#
# Ingredientes que completan el algoritmo moderno:
#
# - **Tasa de aprendizaje** $\eta$ y **número de árboles** $M$, que se compensan entre sí.
# - **Submuestreo** (*stochastic gradient boosting*, Friedman, 2002): cada árbol se ajusta con una fracción aleatoria de los datos (`subsample`), lo que agrega algo de *bagging*, reduce la varianza y acelera.
# - **Tamaño de los árboles**: profundidad 3 a 8, o un máximo de hojas. Árboles de profundidad $d$ capturan interacciones de hasta $d$ atributos.
# - **Parada temprana**: dejar de agregar árboles cuando el error en un conjunto de validación deja de bajar.

# %% [markdown]
# # Parte 3. Una campaña de mercadeo bancario
#
# ## 7. Los datos y el problema
#
# **Contexto.** Entre mayo de 2008 y noviembre de 2010, un banco portugués hizo campañas de **mercadeo telefónico** para vender depósitos a plazo: ejecutivos de un centro de llamadas marcaban a clientes y les ofrecían el producto. Llamar cuesta (tiempo de los ejecutivos y paciencia de los clientes), y la mayoría dice que no.
#
# **Datos.** Sérgio Moro, Paulo Cortez y Paulo Rita publicaron los registros de 45,211 llamadas (*Decision Support Systems* 62, 2014); la tabla está en el repositorio de la UCI (*Bank Marketing*, licencia CC BY 4.0). Las categorías conservan sus valores originales en inglés. Por llamada tenemos:
#
# - **el cliente**: `edad`, `trabajo`, `estado_civil`, `escolaridad`, si tiene un crédito en `incumplimiento`, su `saldo` promedio anual (euros), si tiene `hipoteca` o `prestamo` personal;
# - **la llamada**: tipo de `contacto` (celular o fijo), `dia` y `mes`, y su `duracion` en segundos;
# - **la historia**: cuántas veces se le ha llamado en esta campaña (`campana`), días desde el último contacto de una campaña anterior (`dias_previo`, −1 si nunca), contactos previos y el `resultado_previo`;
# - **la respuesta**: `suscribe` = 1 si contrató el depósito.
#
# **Problema.** Antes de marcar, estimar la probabilidad de que el cliente contrate, para ordenar la lista de llamadas y no gastar ejecutivos en quien casi seguro dirá que no. La métrica natural es la **AUC** (Unidad 3): cuánto mejor que el azar **ordena** el modelo a los clientes.

# %%
banco = rdp.datos.cargar("mercadeo_banco")
categoricas = ["trabajo", "estado_civil", "escolaridad", "incumplimiento", "hipoteca", "prestamo",
               "contacto", "mes", "resultado_previo"]
for c in categoricas:
    banco[c] = banco[c].astype("category")
print(f"{len(banco):,} llamadas; contratan {banco.suscribe.sum():,} ({banco.suscribe.mean():.1%})")
banco.head()

# %% [markdown]
# Separamos un 25 % para prueba, estratificado. Una advertencia: las filas están en **orden temporal**, y una partición al azar mezcla llamadas de 2010 en el entrenamiento con llamadas de 2008 en la prueba. Para un modelo que se usará en la siguiente campaña, una validación temporal (entrenar con el pasado, probar con el futuro) sería más honesta; la veremos en la Unidad 7.

# %%
y_banco = banco["suscribe"]
X_banco = banco.drop(columns="suscribe")
X_tr, X_te, y_tr, y_te = train_test_split(X_banco, y_banco, test_size=0.25, stratify=y_banco, random_state=SEMILLA)
print(f"entrenamiento: {len(X_tr):,} llamadas; prueba: {len(X_te):,}")

# %% [markdown]
# ## 8. Una fuga de información, a propósito
#
# Empecemos como empezaría cualquiera: todos los atributos, una regresión logística (Unidad 4) y un *boosting* con histogramas, `HistGradientBoostingClassifier`, que entiende las columnas categóricas de pandas directamente.

# %%
def logistica(columnas):
    numericas = [c for c in columnas if c not in categoricas]
    return make_pipeline(
        make_column_transformer((OneHotEncoder(handle_unknown="ignore"), categoricas), (StandardScaler(), numericas)),
        LogisticRegression(max_iter=2000))


todas = list(X_banco.columns)
fuga = {}
for nombre, modelo in [("logística", logistica(todas)),
                       ("HistGradientBoosting", HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEMILLA))]:
    modelo.fit(X_tr[todas], y_tr)
    fuga[nombre] = roc_auc_score(y_te, modelo.predict_proba(X_te[todas])[:, 1])
    print(f"{nombre:>22}: AUC de prueba {fuga[nombre]:.3f}")
print(f"{'solo la duración':>22}: AUC de prueba {roc_auc_score(y_te, X_te.duracion):.3f}")

# %% [markdown]
# Una AUC de 0.93 sería espectacular... y la duración de la llamada, **sola**, ya da 0.80. Es demasiado bueno. ¿Cuándo conocemos la duración de una llamada? **Cuando colgamos**, es decir, cuando ya sabemos si el cliente dijo que sí. Alguien que va a contratar habla más tiempo (hay que tomar sus datos); alguien que no, cuelga pronto. La documentación de UCI lo dice explícitamente: el atributo «afecta mucho la respuesta» y «debe descartarse si la intención es tener un modelo predictivo realista».

# %%
print("llamadas de duración 0:", (banco.duracion == 0).sum(), "→ contratan:", banco.suscribe[banco.duracion == 0].sum())
fig, ax = plt.subplots(figsize=(10, 4.2))
bins = np.arange(0, 1500, 30)
for v, color, nombre in [(0, AZUL, "no contrata"), (1, ROJO, "contrata")]:
    ax.hist(banco.duracion[banco.suscribe == v], bins=bins, density=True, color=color, alpha=0.55)
    ax.text(*((300, 0.0031) if v == 0 else (700, 0.0013)), nombre, color=color, fontsize=12, fontweight="bold")
ax.set_xlabel("duración de la llamada (segundos)")
ax.set_ylabel("densidad")
ax.set_title("La duración de la llamada se conoce después de la respuesta")
plt.show()

# %% [markdown]
# Es la **fuga de información** de la Unidad 1: un atributo que contiene la respuesta, o que solo existe después de ella. El modelo «funciona» en la prueba porque la prueba tiene el mismo defecto que el entrenamiento; fallaría el primer día de uso, porque antes de marcar la duración no existe. Lo descartamos de aquí en adelante. La pregunta que hay que hacerse ante cada columna es: **¿la tendré en el momento de decidir?**

# %%
cols = [c for c in todas if c != "duracion"]
Xtr, Xte = X_tr[cols], X_te[cols]

# %% [markdown]
# ## 9. Seis modelos sin trampa
#
# Comparamos, con los mismos datos de entrenamiento:
#
# - **logística** (*one-hot* y estandarización, Unidad 4);
# - **bosque aleatorio** de 500 árboles (ejemplo anterior), con las categorías codificadas como enteros;
# - **`HistGradientBoostingClassifier`** de scikit-learn con sus valores por omisión (tasa 0.1, hasta 100 árboles de 31 hojas, parada temprana automática con un 10 % de validación interna);
# - **XGBoost** (Chen y Guestrin, 2016), **LightGBM** (Ke et al., 2017) y **CatBoost** (Prokhorenkova et al., 2018), con tasa 0.05 y el número de árboles elegido por parada temprana, como explicamos abajo. CatBoost recibe las columnas categóricas tal cual y usa árboles de profundidad 6, su valor por omisión.
#
# Para los tres últimos separamos un 20 % del entrenamiento como **validación**: entrenamos con el 80 % restante agregando árboles hasta que la pérdida de validación lleva 100 rondas sin mejorar, nos quedamos con el número de árboles del mínimo y, con ese número, reentrenamos con todo el entrenamiento. El conjunto de prueba no se toca hasta el final.

# %%
X_a, X_v, y_a, y_v = train_test_split(Xtr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEMILLA)


def xgb_modelo(n, eta=0.05, **kw):
    return xgb.XGBClassifier(n_estimators=n, learning_rate=eta, max_depth=4, enable_categorical=True,
                             tree_method="hist", eval_metric="logloss", random_state=SEMILLA, n_jobs=4, **kw)


def lgb_modelo(n, eta=0.05):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=eta, num_leaves=15, random_state=SEMILLA,
                              n_jobs=4, verbose=-1)


paro_xgb = xgb_modelo(3000, early_stopping_rounds=100).fit(X_a, y_a, eval_set=[(X_v, y_v)], verbose=False)
paro_lgb = lgb_modelo(3000).fit(X_a, y_a, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(100, verbose=False)])
n_xgb, n_lgb = paro_xgb.best_iteration + 1, paro_lgb.best_iteration_


def cb_modelo(n, **kw):
    return CatBoostClassifier(iterations=n, learning_rate=0.05, depth=6, cat_features=categoricas,
                              random_seed=SEMILLA, thread_count=4, verbose=0, allow_writing_files=False, **kw)


paro_cb = cb_modelo(3000, early_stopping_rounds=100).fit(X_a, y_a, eval_set=(X_v, y_v))
n_cb = paro_cb.get_best_iteration() + 1
print(f"árboles elegidos por parada temprana: XGBoost {n_xgb}, LightGBM {n_lgb}, CatBoost {n_cb}")

# %%
modelos = {
    "logística": logistica(cols),
    "bosque aleatorio": make_pipeline(
        make_column_transformer((OrdinalEncoder(), categoricas), remainder="passthrough"),
        RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=4, random_state=SEMILLA)),
    "HistGradientBoosting": HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEMILLA),
    "XGBoost": xgb_modelo(n_xgb),
    "LightGBM": lgb_modelo(n_lgb),
    "CatBoost": cb_modelo(n_cb),
}
filas, prob = [], {}
for nombre, modelo in modelos.items():
    with threadpool_limits(4):          # todos con 4 hilos, para comparar tiempos en igualdad
        t0 = time.perf_counter()
        modelo.fit(Xtr, y_tr)
        segundos = time.perf_counter() - t0
    prob[nombre] = modelo.predict_proba(Xte)[:, 1]
    filas.append({"modelo": nombre, "AUC de prueba": roc_auc_score(y_te, prob[nombre]), "segundos": segundos})
comparacion = pd.DataFrame(filas).set_index("modelo")
comparacion.style.format({"AUC de prueba": "{:.3f}", "segundos": "{:.2f}"})

# %% [markdown]
# Sin la duración, el problema se ve como es: difícil. Los cuatro *boostings* quedan prácticamente empatados (0.804 a 0.810), apenas por encima del bosque (0.802) y claramente por encima de la logística (0.773). Las diferencias de milésimas entre ellos están dentro de lo que cambiaría con otra partición: no hay un ganador. Los tiempos dependen de la máquina; el orden de magnitud es lo que importa: unos cuantos segundos o menos para 34,000 llamadas (CatBoost, con más árboles y más profundos, es el más lento).
#
# ¿Cuánto valen tres centésimas de AUC? Traduzcámoslo a la decisión del banco: si solo hay ejecutivos para llamar a una fracción de la lista, ordenada por probabilidad, ¿qué proporción de todos los que contratarían está en esa fracción? (A esto se le llama una curva de **ganancia** o de captura.)

# %%
def captura(p, y, fraccion=0.10):
    k = int(round(fraccion * len(p)))
    return np.asarray(y)[np.argsort(-p)[:k]].sum() / np.sum(y)


pd.DataFrame({f"llamando al {q:.0%}": {nombre: captura(prob[nombre], y_te, q) for nombre in prob} for q in [0.1, 0.2, 0.3]}).style.format("{:.1%}")

# %% [markdown]
# Llamar a ciegas al 10 % de la lista alcanzaría al 10 % de los interesados; con cualquiera de los modelos, a más del 40 %. En la punta de la lista, la logística no se queda atrás: los clientes más obvios (los que ya contrataron en una campaña anterior) son fáciles para todos. La diferencia aparece después: llamando al 20 %, los *boostings* alcanzan al 63 % de los interesados y la logística al 56 %. En una campaña de decenas de miles de llamadas, siete puntos son cientos de clientes.
#
# ¿Qué atributos usa el modelo? En el ejemplo anterior vimos que la **importancia por permutación** en prueba es más confiable que la impureza media:

# %%
from sklearn.inspection import permutation_importance

imp = permutation_importance(modelos["LightGBM"], Xte, y_te, scoring="roc_auc", n_repeats=5, random_state=SEMILLA, n_jobs=4)
pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).head(6).round(3).to_frame("caída de AUC al permutar")

# %% [markdown]
# El **mes** encabeza la lista, seguido del tipo de **contacto**. No es que mayo sea un mal mes para ahorrar: el banco hizo campañas distintas en momentos distintos (en mayo, el mes con más llamadas, solo contrató el 7 %; en marzo, septiembre, octubre y diciembre, meses con pocas llamadas, cerca de la mitad), y la partición al azar deja que el modelo aprenda **qué campaña** era. Algo parecido pasa con el contacto: casi todas las llamadas con contacto `unknown` están al principio del archivo, en la primera campaña, y de ellas solo contrató el 4 %. Con una validación temporal, estos atributos valdrían mucho menos. Les siguen el resultado de la campaña anterior (quien ya contrató una vez tiende a repetir) y el día del mes. La Unidad 7 y el ejemplo siguiente (interpretabilidad) retoman estas preguntas.
#
# ## 10. Tasa de aprendizaje, número de árboles y parada temprana
#
# ¿Qué pasa si seguimos agregando árboles? Entrenemos XGBoost con tres tasas de aprendizaje y registremos la entropía cruzada (*log-loss*) en entrenamiento y en validación después de cada árbol.

# %%
curvas = {}
for eta in [0.3, 0.1, 0.03]:
    m = xgb_modelo(1500, eta=eta).fit(X_a, y_a, eval_set=[(X_a, y_a), (X_v, y_v)], verbose=False)
    ev = m.evals_result()
    curvas[eta] = {"entrena": np.array(ev["validation_0"]["logloss"]), "valida": np.array(ev["validation_1"]["logloss"])}
    v = curvas[eta]["valida"]
    print(f"η = {eta:<4}: mínimo de validación {v.min():.4f} con {v.argmin() + 1:>4} árboles; con 1,500 árboles: {v[-1]:.4f}")

# %%
fig, ax = plt.subplots(figsize=(10.5, 5))
for eta, color in zip(curvas, [ROJO, AZUL, DORADO_OSC]):
    v, e = curvas[eta]["valida"], curvas[eta]["entrena"]
    n = np.arange(1, len(v) + 1)
    ax.plot(n, e, color=color, lw=1.3, ls=":")
    ax.plot(n, v, color=color)
    ax.scatter([v.argmin() + 1], [v.min()], s=70, color=color, zorder=4, edgecolor="white")
    ax.text(1520 if v[-1] < 0.36 else 960, min(v[-1], 0.352), f"η = {eta}", color=color, va="center", fontsize=11, fontweight="bold")
ax.text(1100, 0.215, "entrenamiento (punteadas)", color=TENUE, fontsize=11)
ax.text(560, 0.312, "validación (continuas)", color=TENUE, fontsize=11)
ax.set_xlim(0, 1700)
ax.set_ylim(0.2, 0.36)
ax.set_xlabel("número de árboles")
ax.set_ylabel("entropía cruzada")
ax.set_title("Más árboles: el entrenamiento siempre mejora; la validación, no")
plt.show()

# %% [markdown]
# Tres lecciones:
#
# 1. **El *boosting* sí sobreajusta.** A diferencia de un bosque, donde agregar árboles nunca empeora el error esperado, aquí cada árbol persigue los residuos que quedan, y tarde o temprano persigue ruido: la curva de validación tiene un mínimo y luego sube, mientras que la de entrenamiento baja sin fin. El número de árboles es un hiperparámetro de complejidad, como la profundidad de un árbol o el grado de un polinomio.
# 2. **Tasa y número de árboles se compensan.** Con $\eta = 0.3$ el mínimo llega en pocas decenas de árboles y la subida es brusca; con $\eta = 0.03$ hacen falta cientos, pero el mínimo es igual o ligeramente mejor y la curva es plana cerca de él: no hay que atinarle con precisión. Una regla práctica: elegir $\eta$ tan pequeña como el tiempo lo permita y el número de árboles con parada temprana.
# 3. **La parada temprana es regularización.** Detenerse en el mínimo de validación limita cuánto se alejan las $F_m$ del punto de partida, de manera parecida a como la penalización $\lambda\lVert\mathbf{w}\rVert^2$ limita los pesos de un modelo lineal.

# %% [markdown]
# # Parte 4. XGBoost, LightGBM y CatBoost por dentro
#
# ## 11. La aproximación de segundo orden
#
# XGBoost (Chen y Guestrin, KDD 2016) minimiza en cada etapa una pérdida **regularizada**:
#
# $$\mathcal{L}^{(m)} = \sum_n L\big(y_n,\, F_{m-1}(\mathbf{x}_n) + h(\mathbf{x}_n)\big) + \Omega(h), \qquad \Omega(h) = \gamma\, J + \tfrac{\lambda}{2}\sum_{j=1}^J w_j^2,$$
#
# donde el árbol $h$ tiene $J$ hojas con valores $w_1, \dots, w_J$. Se aproxima la pérdida con su desarrollo de Taylor de **segundo** orden alrededor de $F_{m-1}$, con $g_n = \partial_F L(y_n, F_{m-1}(\mathbf{x}_n))$ y $h_n = \partial_F^2 L(y_n, F_{m-1}(\mathbf{x}_n))$:
#
# $$\mathcal{L}^{(m)} \approx \text{cte} + \sum_n\Big[g_n\, h(\mathbf{x}_n) + \tfrac12 h_n\, h(\mathbf{x}_n)^2\Big] + \gamma J + \tfrac{\lambda}{2}\sum_j w_j^2 .$$
#
# Como $h(\mathbf{x}_n) = w_j$ para todos los datos de la hoja $j$, agrupamos por hojas con $G_j = \sum_{n \in j} g_n$ y $H_j = \sum_{n \in j} h_n$:
#
# $$\mathcal{L}^{(m)} \approx \text{cte} + \sum_{j=1}^J\Big[G_j w_j + \tfrac12 (H_j + \lambda)\, w_j^2\Big] + \gamma J .$$
#
# Cada hoja es una parábola independiente en $w_j$. Su mínimo:
#
# $$w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad \mathcal{L}^{(m)}_{\min} = -\frac12\sum_{j=1}^J \frac{G_j^2}{H_j + \lambda} + \gamma J .$$
#
# Con $\lambda = 0$ y la pérdida logística, $w^* = \sum(y_n - p_n)/\sum p_n(1-p_n)$: el paso de Newton por hoja de la sección 6. Con la cuadrática, $g_n = F - y_n$ y $h_n = 1$, así que $w^* = \sum_{n\in j}(y_n - F)/(N_j + \lambda)$: el promedio de los residuos **encogido** hacia 0 por $\lambda$.
#
# **Ganancia de una división.** Si una hoja con $(G, H)$ se divide en izquierda $(G_I, H_I)$ y derecha $(G_D, H_D)$, la pérdida baja en
#
# $$\text{ganancia} = \frac12\left[\frac{G_I^2}{H_I + \lambda} + \frac{G_D^2}{H_D + \lambda} - \frac{(G_I + G_D)^2}{H_I + H_D + \lambda}\right] - \gamma .$$
#
# Es el criterio con el que XGBoost elige divisiones (en lugar de Gini o la suma de cuadrados), y $\gamma$ es el precio de cada hoja nueva: si ninguna división gana más que $\gamma$, la hoja no se divide (es una poda incorporada, como el $\alpha$ del costo–complejidad de los árboles).
#
# ### Verificación
#
# Un solo árbol de profundidad 1 sobre la edad, con pérdida logística, $\lambda = 1$, $\gamma = 0$, tasa 1 y punto de partida $F_0 = 0$ ($p = 0.5$ para todos). Calculamos $G$ y $H$ a mano con $g_n = p_n - y_n$ y $h_n = p_n(1-p_n)$ y comparamos con lo que reporta la biblioteca.

# %%
uno = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=1.0, gamma=0, min_child_weight=0,
                        base_score=0.5, tree_method="exact").fit(Xtr[["edad"]], y_tr)
arbol_xgb = uno.get_booster().trees_to_dataframe()
umbral = arbol_xgb.loc[0, "Split"]

p0 = 0.5
g = p0 - y_tr.to_numpy()
h = np.full(len(g), p0 * (1 - p0))
izq = Xtr["edad"].to_numpy() < umbral
lam = 1.0
G_I, H_I, G_D, H_D = g[izq].sum(), h[izq].sum(), g[~izq].sum(), h[~izq].sum()
w_I, w_D = -G_I / (H_I + lam), -G_D / (H_D + lam)
ganancia = 0.5 * (G_I**2 / (H_I + lam) + G_D**2 / (H_D + lam) - (G_I + G_D) ** 2 / (H_I + H_D + lam))

print(f"división elegida por XGBoost: edad < {umbral:g}")
print(f"G_I = {G_I:,.1f}, H_I = {H_I:,.2f}   G_D = {G_D:,.1f}, H_D = {H_D:,.2f}")
print(f"hojas  — a mano: w_I = {w_I:.6f}, w_D = {w_D:.6f};  XGBoost: {arbol_xgb.loc[1, 'Gain']:.6f}, {arbol_xgb.loc[2, 'Gain']:.6f}")
print(f"ganancia — a mano: {ganancia:.3f};  XGBoost reporta {arbol_xgb.loc[0, 'Gain']:.3f} (= 2 × ganancia, sin el ½)")

# %% [markdown]
# (En la tabla de XGBoost, la columna `Gain` de una hoja guarda su valor $w$.) Las dos hojas tienen valores negativos porque partimos de $p = 0.5$ y solo el 11.7 % contrata: ambas bajan el log-momio, pero la de mayores de 60 años lo baja mucho menos. Por eso en la práctica se parte de $F_0 = \ln\frac{\bar y}{1 - \bar y}$ (lo que hace XGBoost por omisión).
#
# ## 12. Histogramas, crecimiento por hojas y GOSS
#
# El paso caro de cualquier *boosting* es buscar la mejor división: para cada atributo, ordenar los datos y evaluar todos los umbrales, $O(dN\log N)$ por nivel, en cada uno de cientos de árboles. Las bibliotecas modernas lo abaratan así:
#
# - **Histogramas.** Antes de empezar, cada atributo continuo se discretiza en a lo más 255 intervalos (por cuantiles). Buscar una división es recorrer 255 sumas de $g$ y $h$ en lugar de $N$ umbrales, y el histograma de una hija se obtiene restando el de su hermana al del padre. Lo usan LightGBM, XGBoost (`tree_method="hist"`) y `HistGradientBoosting` de scikit-learn.
# - **Crecimiento por hojas** (*leaf-wise*, LightGBM). En lugar de completar cada nivel, se divide siempre la hoja con mayor ganancia, hasta un máximo de hojas (`num_leaves`). Con el mismo número de hojas baja más la pérdida, a cambio de árboles desbalanceados que pueden sobreajustar si no se limitan.
# - **GOSS** (*gradient-based one-side sampling*, LightGBM). Los datos con $|g_n|$ pequeño ya están bien ajustados y aportan poco a la ganancia: se conservan todos los de gradiente grande y una muestra de los de gradiente pequeño, reponderada para no sesgar las sumas.
# - **Categóricas nativas.** LightGBM y `HistGradientBoosting` dividen una variable categórica en dos grupos de categorías ordenándolas por $G/H$, sin *one-hot*.
#
# El efecto de los histogramas se ve directamente comparando el *gradient boosting* «exacto» de scikit-learn con el de histogramas, con los mismos 100 árboles:

# %%
X_num = make_column_transformer((OrdinalEncoder(), categoricas), remainder="passthrough").fit_transform(Xtr)
print(f"valores distintos del saldo: {Xtr.saldo.nunique():,}; con histogramas: a lo más 255 intervalos")
for nombre, modelo in [("exacto (GradientBoostingClassifier)", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=SEMILLA)),
                       ("histogramas (HistGradientBoosting)", HistGradientBoostingClassifier(max_iter=100, max_depth=4, early_stopping=False, random_state=SEMILLA))]:
    with threadpool_limits(4):
        t0 = time.perf_counter()
        modelo.fit(X_num, y_tr)
    print(f"{nombre:>36}: {time.perf_counter() - t0:5.2f} s")

# %% [markdown]
# Con 34,000 datos la diferencia ya se nota; como el costo de buscar divisiones deja de crecer con el número de valores distintos, con millones de datos es la diferencia entre poder entrenar o no.
#
# ## 13. CatBoost: categorías sin fuga
#
# Una columna categórica con muchos valores (un código postal, un producto, un cliente) no cabe bien en un *one-hot*: serían miles de columnas. Una alternativa popular es cambiar cada categoría por la **proporción de positivos** que tiene en el entrenamiento, suavizada hacia la proporción global $\bar p$ con un peso $a > 0$:
#
# $$\hat x^{(i)} = \frac{\sum_{j:\, c^{(j)} = c^{(i)}} y^{(j)} + a\,\bar p}{\#\{j : c^{(j)} = c^{(i)}\} + a},$$
#
# con $c^{(i)}$ la categoría del dato $i$. El problema: la suma incluye a $j = i$, así que **la codificación de cada dato contiene su propia respuesta**. En una categoría con un solo dato, $\hat x^{(i)} = (y^{(i)} + a\bar p)/(1 + a)$ es la respuesta, apenas disfrazada.
#
# Para verlo, agregamos a las llamadas una columna que no dice nada: un identificador **al azar** con unos 8,000 valores posibles.

# %%
azar = np.random.default_rng(0)
id_tr = pd.Series(azar.integers(0, 8000, len(Xtr)), index=Xtr.index).astype(str)
id_te = pd.Series(azar.integers(0, 8000, len(Xte)), index=Xte.index).astype(str)
previa, a = float(y_tr.mean()), 1.0


def codificacion_ingenua(c):
    """Proporción suavizada de positivos por categoría, con todos los datos de entrenamiento."""
    g = pd.DataFrame({"c": id_tr, "y": y_tr}).groupby("c").y.agg(["sum", "count"])
    return ((g["sum"].reindex(c).fillna(0) + a * previa) / (g["count"].reindex(c).fillna(0) + a)).to_numpy()


def codificacion_ordenada(c, semilla=0):
    """Lo mismo, pero cada dato solo ve a los que van antes que él en un orden al azar."""
    orden = np.random.default_rng(semilla).permutation(len(c))
    suma, cuenta, cod = {}, {}, np.empty(len(c))
    for k in orden:
        ck, yk = c.iloc[k], y_tr.iloc[k]
        cod[k] = (suma.get(ck, 0) + a * previa) / (cuenta.get(ck, 0) + a)
        suma[ck] = suma.get(ck, 0) + yk
        cuenta[ck] = cuenta.get(ck, 0) + 1
    return cod


print(f"identificadores distintos en entrenamiento: {id_tr.nunique():,}")
filas = []
for nombre, (c_tr, c_te) in {"sin el identificador": (None, None),
                             "codificación con todos los datos": (codificacion_ingenua(id_tr), codificacion_ingenua(id_te)),
                             "codificación ordenada": (codificacion_ordenada(id_tr), codificacion_ingenua(id_te))}.items():
    A, B = Xtr.copy(), Xte.copy()
    if c_tr is not None:
        A["id"], B["id"] = c_tr, c_te
    m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEMILLA).fit(A, y_tr)
    filas.append({"": nombre, "AUC de entrenamiento": roc_auc_score(y_tr, m.predict_proba(A)[:, 1]),
                  "AUC de prueba": roc_auc_score(y_te, m.predict_proba(B)[:, 1])})
pd.DataFrame(filas).set_index("").round(3)

# %% [markdown]
# Con la codificación ingenua, la columna sin información sube la AUC de entrenamiento a casi 0.96 y **derrumba** la de prueba a cerca de 0.54: el modelo aprendió a confiar en una columna que en entrenamiento lleva la respuesta escondida y en prueba no dice nada. Es una fuga de información creada por el preprocesamiento.
#
# **Estadísticas ordenadas** (la solución de CatBoost). Se barajan los datos de entrenamiento y cada uno se codifica solo con los que van **antes** que él ($j \prec i$):
#
# $$\hat x^{(i)} = \frac{\sum_{j \prec i,\; c^{(j)} = c^{(i)}} y^{(j)} + a\,\bar p}{\#\{j \prec i : c^{(j)} = c^{(i)}\} + a}.$$
#
# La respuesta del propio dato nunca entra; en prueba se usan todos los datos de entrenamiento. Con eso, la columna sin información deja de engañar al modelo (tercera fila). CatBoost hace esto internamente con varias permutaciones; si le damos el identificador como una categoría más:

# %%
A, B = Xtr.copy(), Xte.copy()
A["id"], B["id"] = id_tr, id_te
cb_id = CatBoostClassifier(iterations=n_cb, learning_rate=0.05, depth=6, cat_features=categoricas + ["id"],
                           random_seed=SEMILLA, thread_count=4, verbose=0, allow_writing_files=False).fit(A, y_tr)
print(f"CatBoost con el identificador como categoría: AUC de prueba {roc_auc_score(y_te, cb_id.predict_proba(B)[:, 1]):.3f}")

# %% [markdown]
# Otras dos ideas de CatBoost:
#
# - **Boosting ordenado** (`boosting_type="Ordered"`): la misma idea aplicada a los pseudo-residuos; el residuo de cada dato se calcula con un modelo que no lo vio. Ayuda sobre todo con pocos datos.
# - **Árboles simétricos** (*oblivious*): en cada nivel se hace la misma pregunta en todos los nodos. Un árbol de profundidad $d$ es una lista de $d$ preguntas y la hoja de un dato es el número binario formado por sus $d$ respuestas. Predecir es muy rápido y los árboles, más restringidos, sobreajustan menos.
#
# ## 14. Cierre
#
# - El *boosting* construye un modelo aditivo $F_M = \sum_m \eta\, h_m$ **por etapas**: cada término corrige a los anteriores. AdaBoost es el caso de la pérdida exponencial; *gradient boosting* lo extiende a cualquier pérdida diferenciable ajustando árboles a los pseudo-residuos.
# - El *bagging* reduce varianza promediando modelos complejos; el *boosting* reduce **sesgo** sumando modelos simples, y por eso sí sobreajusta si se agregan árboles sin control. Tasa de aprendizaje pequeña, árboles moderados y parada temprana son la receta.
# - XGBoost usa una aproximación de segundo orden con regularización: peso de hoja $-G/(H + \lambda)$ y una ganancia que penaliza cada hoja con $\gamma$. LightGBM y `HistGradientBoosting` hacen lo mismo con histogramas y crecimiento por hojas. CatBoost codifica las categorías con estadísticas ordenadas, que no dejan entrar la respuesta de cada dato, y usa árboles simétricos.
# - Ningún algoritmo protege de una **fuga de información**. La pregunta «¿tendré este dato al momento de decidir?» va antes que cualquier comparación de modelos.
#
# *Referencias:* Y. Freund y R. E. Schapire, «A decision-theoretic generalization of on-line learning and an application to boosting», *J. Computer and System Sciences* 55 (1997). J. Friedman, T. Hastie y R. Tibshirani, «Additive logistic regression: a statistical view of boosting», *Annals of Statistics* 28 (2000). J. H. Friedman, «Greedy function approximation: a gradient boosting machine», *Annals of Statistics* 29 (2001). T. Chen y C. Guestrin, «XGBoost: a scalable tree boosting system», *KDD* (2016). G. Ke et al., «LightGBM: a highly efficient gradient boosting decision tree», *NeurIPS* (2017). S. Moro, P. Cortez y P. Rita, «A data-driven approach to predict the success of bank telemarketing», *Decision Support Systems* 62 (2014). L. Prokhorenkova et al., «CatBoost: unbiased boosting with categorical features», *NeurIPS* (2018).

# %% [markdown]
# ## 15. Autoevaluación

# %%
rdp.quiz.mostrar("u05", ids=["u05-p09", "u05-p10", "u05-p11", "u05-p12"])

# %% [markdown]
# ## 16. Ejercicios
#
# 1. **El tocón recién usado queda en la moneda.** Demuestra que, con los pesos actualizados $w^{(t+1)}$, el error ponderado del tocón $h_t$ es exactamente $\tfrac12$. (Pista: calcula por separado el peso total de aciertos y de errores después de multiplicar por $e^{\mp\alpha_t}$ y usa la fórmula de $\alpha_t$.) ¿Qué implica esto sobre el tocón de la ronda $t+1$? Verifícalo con `pesos` y `tocones` de la sección 2.
# 2. **Qué estima AdaBoost.** Demuestra que la función que minimiza $\mathbb{E}\big[e^{-yF(\mathbf{x})} \mid \mathbf{x}\big]$, con $y \in \{-1, +1\}$, es $F^*(\mathbf{x}) = \tfrac12\ln\frac{P(y=1\mid\mathbf{x})}{P(y=-1\mid\mathbf{x})}$. Deduce una fórmula para convertir $F_T$ en probabilidades y compárala con `predict_proba` de `AdaBoostClassifier` en los pingüinos.
# 3. **Pseudo-residuos de Poisson.** Con $\log\mu = F$ y la log-verosimilitud de Poisson, deriva $g_n$ y $h_n$ y el peso de hoja de XGBoost. Modifica `gradient_boosting` para usar pseudo-residuos de Poisson (con el paso de Newton por hoja) y compara su RECM de prueba en las bicicletas con la de `HistGradientBoostingRegressor(loss="poisson")`.
# 4. **Submuestreo y λ.** Repite las curvas de la sección 10 con $\eta = 0.1$ y `subsample` ∈ {1, 0.8, 0.5}, y luego con `reg_lambda` ∈ {0, 1, 10, 100}. ¿Cómo cambian el mínimo de validación y la forma de la curva después del mínimo?
# 5. **Validación temporal.** Entrena LightGBM con las primeras 34,000 llamadas (en el orden del archivo) y prueba con las restantes. Compara la AUC con la de la partición al azar y la importancia por permutación de `mes` y `dia`. ¿Qué número le reportarías al banco?
