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
# # Máquinas de vectores de soporte y kernels
#
# **Unidad 4 · Ejemplo 4**
#
# Cuando dos clases se pueden separar con una recta, casi siempre hay **infinitas** rectas que lo hacen. Todas tienen $E_{\text{in}} = 0$, así que el error en la muestra no sirve para escoger entre ellas. Las máquinas de vectores de soporte (SVM, por *support vector machines*) responden con un criterio geométrico: quedarse con la recta que deja **la franja más ancha** entre las clases. De esa idea tan simple sale una cadena de resultados muy bonitos:
#
# 1. la separadora de **margen máximo** es la solución de un problema cuadrático convexo, que resolveremos a mano y compararemos con scikit-learn;
# 2. su **problema dual** muestra que la solución depende solo de unos cuantos puntos, los **vectores de soporte**, y da una cota del error de validación cruzada casi gratis;
# 3. el **margen suave** admite errores y resulta ser la minimización de la **pérdida bisagra** con regularización, igual que la regresión logística con otra pérdida;
# 4. el dual solo usa productos internos, y eso abre la puerta al **truco del *kernel***: fronteras no lineales con la misma maquinaria;
# 5. al final lo aplicamos a reconocer dígitos escritos a mano.
#
# Seguimos la notación de *Learning From Data*. En toda la libreta las etiquetas son $y \in \{-1, +1\}$, que es la convención natural para SVM: un punto está bien clasificado cuando $y\,(\mathbf{w}^\top\mathbf{x} + b) > 0$.

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
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sklearn.datasets import make_circles, make_classification, load_digits
from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict,
                                     StratifiedKFold, LeaveOneOut, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, sigmoid_kernel

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# %% [markdown]
# ## 1. La pregunta: muchas rectas separan, ¿cuál elegimos?
#
# ### Los datos y el problema
#
# **Datos.** Usamos otra vez los pingüinos del archipiélago Palmer (Antártida), que conocimos en la Unidad 2. Kristen Gorman los midió entre 2007 y 2009 como parte del programa de investigación ecológica de largo plazo de la estación Palmer; Allison Horst, Alison Hill y Gorman los publicaron en 2020 en el paquete `palmerpenguins`, con licencia CC0. Hay 344 pingüinos de tres especies (Adelia, barbijo y papúa; en la tabla, `Adelie`, `Chinstrap` y `Gentoo`) con las medidas del pico, la aleta y la masa corporal.
#
# **Problema.** Distinguir un pingüino **Adelia** de un **papúa** (*Gentoo*) con dos medidas: la profundidad (altura) del pico, `bill_depth_mm`, y la longitud de la aleta, `flipper_length_mm`. Los papúa son más grandes y tienen el pico más delgado, así que esperamos que una recta baste. Estandarizamos ambas variables (media 0, desviación 1), porque el margen es una **distancia** y una distancia solo tiene sentido si las dos coordenadas están en la misma escala. Codificamos papúa como $y = +1$ y Adelia como $y = -1$.

# %%
p = rdp.datos.cargar("pinguinos")
COLS_DURO = ["bill_depth_mm", "flipper_length_mm"]
pd_ = p.dropna(subset=COLS_DURO)
pd_ = pd_[pd_.species.isin(["Adelie", "Gentoo"])]
esc_duro = StandardScaler().fit(pd_[COLS_DURO])
X = esc_duro.transform(pd_[COLS_DURO])
y = np.where(pd_.species == "Gentoo", 1, -1)
N = len(y)
print(f"N = {N} pingüinos: {np.sum(y == 1)} papúa (+1), {np.sum(y == -1)} Adelia (−1)")
pd_.groupby("species")[COLS_DURO].agg(["mean", "std"]).round(1)

# %% [markdown]
# Para tener varias separadoras distintas usamos el **algoritmo del perceptrón** (PLA) de la Unidad 3: mientras haya un punto mal clasificado, se toma uno y se corrige $\mathbf{w} \leftarrow \mathbf{w} + y_n\mathbf{x}_n$. Como los datos son separables, el algoritmo termina con $E_{\text{in}} = 0$, pero **dónde** termina depende del orden en que visita los puntos. Lo corremos con tres semillas distintas.
#
# Para cada recta $\mathbf{w}^\top\mathbf{x} + b = 0$ medimos su **margen**: la distancia del punto más cercano a la recta (la fórmula de la distancia la deducimos en la sección 2).

# %%
def perceptron(X, y, semilla, max_iter=100_000):
    """PLA con orden aleatorio de visita. Regresa (w, b)."""
    rng = np.random.default_rng(semilla)
    Xa = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xa.shape[1])
    for _ in range(max_iter):
        malos = np.flatnonzero(np.sign(Xa @ w) != y)
        if malos.size == 0:
            break
        n = rng.choice(malos)
        w = w + y[n] * Xa[n]
    return w[1:], w[0]


def margen(w, b, X, y):
    """Distancia del punto más cercano a la recta (negativa si hay errores)."""
    return np.min(y * (X @ w + b)) / np.linalg.norm(w)


rectas_pla = [perceptron(X, y, s) for s in (1, 2, 3)]
svm_duro = SVC(kernel="linear", C=1e6).fit(X, y)
w_svm, b_svm = svm_duro.coef_[0], svm_duro.intercept_[0]

for i, (w, b) in enumerate(rectas_pla, 1):
    print(f"perceptrón {i}: E_in = {np.mean(np.sign(X @ w + b) != y):.0f}   margen = {margen(w, b, X, y):.3f}")
print(f"margen máximo:  E_in = {np.mean(svm_duro.predict(X) != y):.0f}   margen = {margen(w_svm, b_svm, X, y):.3f}")


# %%
def dibujar_puntos(ax, X, y, etiquetas=("Adelia (−1)", "papúa (+1)"), s=28):
    ax.scatter(*X[y == -1].T, color=ROJO, s=s, alpha=0.8, label=etiquetas[0])
    ax.scatter(*X[y == 1].T, color=AZUL, s=s, alpha=0.8, label=etiquetas[1])


def recta(ax, w, b, xlim, **kw):
    xs = np.linspace(*xlim, 50)
    ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)


LIM = (-2.6, 2.6)
fig, ax = plt.subplots(figsize=(8.5, 7))
dibujar_puntos(ax, X, y)
for (w, b), ls in zip(rectas_pla, ["--", "-.", ":"]):
    recta(ax, w, b, LIM, color=GRIS, lw=1.6, ls=ls)
recta(ax, w_svm, b_svm, LIM, color=TINTA, lw=2.6)
ax.set(xlim=LIM, ylim=LIM, aspect="equal",
       xlabel="profundidad del pico (estandarizada)", ylabel="longitud de la aleta (estandarizada)",
       title="Cuatro rectas con $E_{\\mathrm{in}} = 0$")
ax.text(-2.45, 2.2, "papúa", color=AZUL, fontsize=13, weight="semibold")
ax.text(1.2, -2.2, "Adelia", color=ROJO, fontsize=13, weight="semibold")
ax.annotate("margen máximo", (1.2, 1.55), xytext=(0.15, 2.35), color=TINTA, fontsize=12,
            arrowprops=dict(arrowstyle="->", color=TINTA, lw=1.2))
ax.text(-2.45, -2.45, "grises: tres corridas del perceptrón", color=GRIS, fontsize=11)
plt.show()

# %% [markdown]
# Las cuatro rectas clasifican perfecto los 274 pingüinos, pero no inspiran la misma confianza. Las del perceptrón pasan rozando algunos puntos: un pingüino nuevo un poco distinto a los de la muestra, del lado «correcto» pero cerca de la frontera, caería del lado equivocado. La recta negra deja la mayor distancia posible a ambas clases.
#
# **¿Por qué preferir un margen grande?** Tres argumentos, de menos a más formales:
#
# - **Tolerancia al ruido.** Si las mediciones tienen un error de tamaño menor que el margen, ninguna clasificación de la muestra cambia.
# - **Menos hipótesis efectivas.** Exigir margen $\rho$ restringe el conjunto de hipótesis: hay menos maneras de partir $N$ puntos con una franja ancha que con una línea sin grosor. En *Learning From Data* (sección 8.1) se muestra que la dimensión VC de los separadores con margen $\rho$ sobre datos en una bola de radio $R$ es a lo más $\lceil R^2/\rho^2 \rceil + 1$, **sin importar la dimensión** del espacio. Esto será crucial con los kernels, donde la dimensión puede ser infinita.
# - **Pocos puntos deciden.** La solución depende solo de los puntos en la orilla de la franja, y eso da una cota directa del error de validación cruzada (sección 3).

# %% [markdown]
# ## 2. Margen duro: el problema primal
#
# ### La distancia de un punto a un hiperplano
#
# Sea el hiperplano $\{\mathbf{x} : \mathbf{w}^\top\mathbf{x} + b = 0\}$. Para dos puntos $\mathbf{x}', \mathbf{x}''$ en él, $\mathbf{w}^\top(\mathbf{x}' - \mathbf{x}'') = 0$: el vector $\mathbf{w}$ es **normal** al hiperplano. La distancia de un punto $\mathbf{x}$ es la longitud de la proyección de $\mathbf{x} - \mathbf{x}'$ sobre la dirección unitaria $\mathbf{w}/\lVert\mathbf{w}\rVert$:
#
# $$
# \operatorname{dist}(\mathbf{x}) = \frac{\lvert\mathbf{w}^\top(\mathbf{x} - \mathbf{x}')\rvert}{\lVert\mathbf{w}\rVert} = \frac{\lvert\mathbf{w}^\top\mathbf{x} + b\rvert}{\lVert\mathbf{w}\rVert},
# $$
#
# usando $\mathbf{w}^\top\mathbf{x}' = -b$. Si el hiperplano separa los datos, $\lvert\mathbf{w}^\top\mathbf{x}_n + b\rvert = y_n(\mathbf{w}^\top\mathbf{x}_n + b)$, que es lo que usamos arriba.
#
# ### La normalización
#
# El par $(\mathbf{w}, b)$ y el par $(c\,\mathbf{w}, c\,b)$ con $c > 0$ describen **el mismo** hiperplano. Aprovechamos esa libertad para fijar la escala de modo que el punto más cercano cumpla
#
# $$
# \min_{n} \; y_n(\mathbf{w}^\top\mathbf{x}_n + b) = 1.
# $$
#
# Con esta normalización el margen vale $1/\lVert\mathbf{w}\rVert$, y el **ancho de la franja** (de orilla a orilla) es $2/\lVert\mathbf{w}\rVert$.
#
# ### El problema de optimización
#
# Maximizar $1/\lVert\mathbf{w}\rVert$ es lo mismo que minimizar $\tfrac12\mathbf{w}^\top\mathbf{w}$. La condición «el mínimo es 1» se puede relajar a «todos son al menos 1», porque en el óptimo alguna desigualdad debe ser igualdad (si todas fueran estrictas, podríamos encoger $\mathbf{w}$ y $b$ y bajar el objetivo). Queda el **problema primal** del margen duro:
#
# $$
# \boxed{\;\min_{\mathbf{w},\, b} \; \tfrac12\,\mathbf{w}^\top\mathbf{w} \quad \text{sujeto a} \quad y_n(\mathbf{w}^\top\mathbf{x}_n + b) \ge 1, \quad n = 1, \dots, N.\;}
# $$
#
# Es un **problema cuadrático** (*quadratic programming*, QP): objetivo cuadrático convexo con restricciones lineales. Tiene un único óptimo y hay algoritmos eficientes para encontrarlo. Lo resolvemos con un optimizador genérico de `scipy` (SLSQP, que acepta restricciones de desigualdad) sobre la variable $\mathbf{v} = (w_1, w_2, b)$, y comparamos con scikit-learn. `SVC` resuelve el margen suave de la sección 4; con un $C$ enorme ($10^6$) equivale al margen duro.

# %%
def primal_duro(X, y):
    d = X.shape[1]
    restriccion = {
        "type": "ineq",  # SLSQP pide restricciones de la forma g(v) >= 0
        "fun": lambda v: y * (X @ v[:d] + v[d]) - 1,
        "jac": lambda v: np.c_[y[:, None] * X, y],
    }
    res = minimize(lambda v: 0.5 * v[:d] @ v[:d], np.zeros(d + 1), jac=lambda v: np.r_[v[:d], 0.0],
                   constraints=[restriccion], method="SLSQP")
    return res.x[:d], res.x[d], res


w_qp, b_qp, res = primal_duro(X, y)
print(f"scipy (SLSQP):  w = {np.round(w_qp, 4)}, b = {b_qp:.4f}   ({res.message})")
print(f"scikit-learn:   w = {np.round(w_svm, 4)}, b = {b_svm:.4f}")
print(f"min_n y_n (w·x_n + b) = {np.min(y * (X @ w_qp + b_qp)):.4f}   (la normalización)")
print(f"ancho de la franja 2/‖w‖ = {2 / np.linalg.norm(w_svm):.3f} desviaciones estándar")
print(f"vectores de soporte: {len(svm_duro.support_)} de {N}  (Adelia: {svm_duro.n_support_[0]}, papúa: {svm_duro.n_support_[1]})")


# %% [markdown]
# Los dos coinciden hasta el tercer decimal. El ancho de la franja es de unas 0.68 desviaciones estándar, y **solo tres** pingüinos tocan sus orillas: esos son los **vectores de soporte**. Los demás 271 podrían moverse (sin cruzar la orilla) o desaparecer, y la solución sería exactamente la misma.

# %%
def dibujar_svm(ax, modelo, X, y, lim, titulo=None, etiquetas=("Adelia (−1)", "papúa (+1)"), fondo=True):
    """Regiones, frontera (s = 0), orillas (s = ±1) y vectores de soporte de un SVC ajustado."""
    g1, g2 = np.meshgrid(np.linspace(*lim[0], 300), np.linspace(*lim[1], 300))
    s = modelo.decision_function(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    if fondo:
        ax.contourf(g1, g2, s > 0, levels=[-0.5, 0.5, 1.5], cmap=rdp.estilo.CLASES2, alpha=0.6)
    ax.contour(g1, g2, s, levels=[-1, 0, 1], colors=[TINTA], linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
    dibujar_puntos(ax, X, y, etiquetas, s=22)
    vs = modelo.support_vectors_
    ax.scatter(*vs.T, s=110, facecolors="none", edgecolors=TINTA, linewidths=1.4)
    ax.set(xlim=lim[0], ylim=lim[1])
    if titulo:
        ax.set_title(titulo)


fig, ax = plt.subplots(figsize=(8.5, 7))
dibujar_svm(ax, svm_duro, X, y, (LIM, LIM))
ax.set(aspect="equal", xlabel="profundidad del pico (estandarizada)", ylabel="longitud de la aleta (estandarizada)",
       title=f"Margen duro: franja de ancho {2 / np.linalg.norm(w_svm):.2f} y {len(svm_duro.support_)} vectores de soporte")
# flecha que marca el ancho de la franja, perpendicular a la frontera
u = w_svm / np.linalg.norm(w_svm)
centro = -b_svm * w_svm / np.linalg.norm(w_svm) ** 2 + 1.2 * np.array([-u[1], u[0]])
ax.annotate("", centro + u / np.linalg.norm(w_svm), centro - u / np.linalg.norm(w_svm),
            arrowprops=dict(arrowstyle="<->", color=DORADO, lw=2))
ax.text(*(centro + u / np.linalg.norm(w_svm) + np.array([-0.05, 0.1])), "$2/\\|\\mathbf{w}\\|$", color=DORADO,
        fontsize=14, ha="right")
ax.text(-2.45, -2.45, "círculos: vectores de soporte", color=TINTA, fontsize=11)
ax.text(-2.45, 2.2, "papúa", color=AZUL, fontsize=13, weight="semibold")
ax.text(1.2, -2.2, "Adelia", color=ROJO, fontsize=13, weight="semibold")
plt.show()

# %% [markdown]
# La línea continua es la frontera $\mathbf{w}^\top\mathbf{x} + b = 0$ y las punteadas son las orillas $\mathbf{w}^\top\mathbf{x} + b = \pm 1$. Los tres vectores de soporte (dos papúa y un Adelia) están justo sobre las orillas, y ningún punto queda dentro de la franja.

# %% [markdown]
# ## 3. El problema dual
#
# ### Lagrangiano y condiciones KKT
#
# Para cada restricción introducimos un multiplicador de Lagrange $\alpha_n \ge 0$ y formamos el lagrangiano
#
# $$
# \mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \tfrac12\mathbf{w}^\top\mathbf{w} - \sum_{n=1}^{N} \alpha_n\big[y_n(\mathbf{w}^\top\mathbf{x}_n + b) - 1\big].
# $$
#
# Como el problema es convexo y las restricciones son lineales (y factibles), la teoría de dualidad garantiza que el óptimo cumple las **condiciones de Karush-Kuhn-Tucker** (KKT):
#
# 1. **Estacionariedad:** $\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{w} - \sum_n \alpha_n y_n\mathbf{x}_n = \mathbf{0}$ y $\partial\mathcal{L}/\partial b = -\sum_n \alpha_n y_n = 0$.
# 2. **Factibilidad:** $y_n(\mathbf{w}^\top\mathbf{x}_n + b) \ge 1$ y $\alpha_n \ge 0$.
# 3. **Holgura complementaria:** $\alpha_n\big[y_n(\mathbf{w}^\top\mathbf{x}_n + b) - 1\big] = 0$ para toda $n$.
#
# De la primera condición sale la forma de la solución:
#
# $$
# \boxed{\;\mathbf{w} = \sum_{n=1}^{N} \alpha_n y_n \mathbf{x}_n\;}
# $$
#
# y de la tercera, que **si un punto no está en la orilla** ($y_n(\mathbf{w}^\top\mathbf{x}_n + b) > 1$) **entonces $\alpha_n = 0$**. Los puntos con $\alpha_n > 0$ son, por definición, los vectores de soporte: $\mathbf{w}$ es una combinación lineal de ellos y de nadie más.
#
# ### El dual
#
# Sustituyendo $\mathbf{w} = \sum_n \alpha_n y_n\mathbf{x}_n$ y $\sum_n\alpha_n y_n = 0$ en $\mathcal{L}$, el término con $b$ desaparece y queda una función solo de $\boldsymbol{\alpha}$:
#
# $$
# \mathcal{L} = \tfrac12\sum_{n,m}\alpha_n\alpha_m y_n y_m\,\mathbf{x}_n^\top\mathbf{x}_m - \sum_{n,m}\alpha_n\alpha_m y_n y_m\,\mathbf{x}_n^\top\mathbf{x}_m + \sum_n\alpha_n
# = \sum_n\alpha_n - \tfrac12\sum_{n,m}\alpha_n\alpha_m y_n y_m\,\mathbf{x}_n^\top\mathbf{x}_m.
# $$
#
# El **problema dual** es maximizar esto sobre $\boldsymbol{\alpha}$:
#
# $$
# \boxed{\;\max_{\boldsymbol{\alpha}} \; \sum_{n=1}^N \alpha_n - \tfrac12\sum_{n=1}^N\sum_{m=1}^N \alpha_n\alpha_m\, y_n y_m\, \mathbf{x}_n^\top\mathbf{x}_m \quad\text{sujeto a}\quad \alpha_n \ge 0,\;\; \sum_{n=1}^N \alpha_n y_n = 0.\;}
# $$
#
# Es otro QP, ahora con $N$ variables en lugar de $d + 1$. Parecería un mal negocio, pero tiene dos virtudes: la solución es **rala** (casi todas las $\alpha_n$ valen cero) y **los datos solo aparecen a través de productos internos** $\mathbf{x}_n^\top\mathbf{x}_m$. La segunda es la que hace posible los kernels de la sección 5. El sesgo $b$ se recupera de cualquier vector de soporte $\mathbf{x}_s$: $b = y_s - \mathbf{w}^\top\mathbf{x}_s$.
#
# Resolvamos el dual con `scipy` y comparemos con lo que guarda scikit-learn: `support_` son los índices de los vectores de soporte y `dual_coef_` contiene los productos $\alpha_n y_n$ correspondientes.

# %%
def dual_duro(X, y):
    Q = (y[:, None] * X) @ (y[:, None] * X).T  # Q_nm = y_n y_m x_n·x_m
    res = minimize(lambda a: 0.5 * a @ Q @ a - a.sum(), np.zeros(len(y)), jac=lambda a: Q @ a - 1,
                   bounds=[(0, None)] * len(y), method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y.astype(float)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


t0 = time.perf_counter()
alfa = dual_duro(X, y)
print(f"dual resuelto en {time.perf_counter() - t0:.1f} s")
vs_dual = np.flatnonzero(alfa > 1e-6)
print(f"α_n > 0 en los índices {vs_dual}; scikit-learn: support_ = {np.sort(svm_duro.support_)}")
print(f"α (scipy)        = {np.round(alfa[vs_dual], 4)}")
orden = np.argsort(svm_duro.support_)
print(f"α_n y_n (sklearn) = {np.round(svm_duro.dual_coef_[0][orden], 4)}   (dual_coef_)")
print(f"Σ α_n y_n = {alfa @ y:.2e}")

w_dual = (alfa * y) @ X
s0 = vs_dual[0]
print(f"w = Σ α_n y_n x_n = {np.round(w_dual, 4)}   b = y_s − w·x_s = {y[s0] - w_dual @ X[s0]:.4f}")
print(f"con dual_coef_ @ support_vectors_: {np.round(svm_duro.dual_coef_[0] @ svm_duro.support_vectors_, 4)}")

# %% [markdown]
# De 274 multiplicadores, solo tres son distintos de cero, en los mismos índices que reporta scikit-learn; y reconstruir $\mathbf{w}$ como $\sum_n\alpha_n y_n\mathbf{x}_n$ da el mismo vector que el primal. (Los valores $\alpha_n y_n$ de `dual_coef_` tienen el signo de $y_n$, por eso aparecen negativos para los Adelia.)
#
# ### Una cota del error de validación cruzada «dejando uno fuera»
#
# La validación cruzada dejando uno fuera (*leave-one-out*, LOO) entrena $N$ veces, cada vez sin un punto, y cuenta en cuántas se equivoca con el punto que dejó fuera:
#
# $$
# E_{\text{cv}} = \frac1N\sum_{n=1}^N e_n, \qquad e_n = [\![\, g^{(-n)}(\mathbf{x}_n) \ne y_n \,]\!].
# $$
#
# **Afirmación.** Para el SVM de margen duro, $E_{\text{cv}} \le \dfrac{\#\text{VS}}{N}$.
#
# **Demostración.** Si $\mathbf{x}_n$ **no** es vector de soporte, $\alpha_n = 0$. Al quitarlo, la solución anterior sigue siendo factible para el problema sin él y sigue cumpliendo las condiciones KKT (el término de $\mathbf{x}_n$ no contribuía a nada), así que es **la** solución: $g^{(-n)} = g$. Como $g$ clasifica bien a $\mathbf{x}_n$, $e_n = 0$. Solo los vectores de soporte pueden aportar un error. $\blacksquare$
#
# Es una cota muy barata: sale de **un solo** entrenamiento, en lugar de $N$. Comparémosla con el LOO de verdad.

# %%
t0 = time.perf_counter()
e_loo = 1 - cross_val_score(SVC(kernel="linear", C=1e6), X, y, cv=LeaveOneOut()).mean()
print(f"E_cv (LOO, {N} entrenamientos, {time.perf_counter() - t0:.1f} s) = {e_loo:.4f}")
print(f"cota #VS/N = {len(svm_duro.support_)}/{N} = {len(svm_duro.support_) / N:.4f}")

# %% [markdown]
# La cota garantiza un error LOO de a lo más 1.1 %, y el real es 0: al quitar cualquiera de los tres vectores de soporte, la nueva franja (un poco más ancha) todavía clasifica bien al pingüino que se quitó. La cota es pesimista porque supone que **cada** vector de soporte que se quita se clasifica mal.
#
# Nota algo interesante: la cota no depende de la dimensión del espacio. Si un SVM en un espacio de dimensión enorme (o infinita, con un kernel) termina con pocos vectores de soporte, podemos confiar en que generaliza bien.

# %% [markdown]
# ## 4. Margen suave
#
# ### Cuando las clases se enciman
#
# Ahora un problema más difícil: distinguir **Adelia** ($y = -1$) de **barbijo** (*Chinstrap*, $y = +1$) con la longitud y la profundidad del pico, estandarizadas. Los barbijo tienen el pico más largo, pero hay pingüinos de ambas especies en la zona intermedia: ninguna recta separa las clases sin errores y el problema primal del margen duro **no tiene solución factible**.

# %%
COLS_SUAVE = ["bill_length_mm", "bill_depth_mm"]
ps = p.dropna(subset=COLS_SUAVE)
ps = ps[ps.species.isin(["Adelie", "Chinstrap"])]
esc_suave = StandardScaler().fit(ps[COLS_SUAVE])
X2 = esc_suave.transform(ps[COLS_SUAVE])
y2 = np.where(ps.species == "Chinstrap", 1, -1)
N2 = len(y2)
print(f"N = {N2} pingüinos: {np.sum(y2 == 1)} barbijo (+1), {np.sum(y2 == -1)} Adelia (−1)")
_, _, res_inf = primal_duro(X2, y2)
print(f"margen duro con estos datos: {res_inf.message}")

# %% [markdown]
# ### Las holguras
#
# La solución es permitir que cada punto **viole** el margen por una cantidad $\xi_n \ge 0$, llamada **holgura**, y cobrar por ello:
#
# $$
# \boxed{\;\min_{\mathbf{w},\, b,\, \boldsymbol{\xi}} \; \tfrac12\,\mathbf{w}^\top\mathbf{w} + C\sum_{n=1}^N \xi_n \quad\text{sujeto a}\quad y_n(\mathbf{w}^\top\mathbf{x}_n + b) \ge 1 - \xi_n, \quad \xi_n \ge 0.\;}
# $$
#
# - $\xi_n = 0$: el punto está fuera de la franja, del lado correcto.
# - $0 < \xi_n \le 1$: está dentro de la franja, pero todavía del lado correcto.
# - $\xi_n > 1$: está del lado equivocado de la frontera; es un error de clasificación. Por eso $\sum_n \xi_n$ es una cota superior del número de errores.
#
# El parámetro $C > 0$ decide el precio. Con $C$ grande, violar el margen sale caro y nos acercamos al margen duro (franja angosta, pocos errores en la muestra). Con $C$ pequeño, preferimos una franja ancha aunque muchos puntos queden dentro. El dual es el mismo de antes con un único cambio: $0 \le \alpha_n \le C$. Ahora son vectores de soporte todos los puntos sobre la orilla **o dentro de la franja** (o del lado equivocado), porque todos ellos tienen $\alpha_n > 0$.
#
# Probamos $C \in \{0.01,\ 0.1,\ 1,\ 100\}$. Para cada valor reportamos el número de vectores de soporte, el ancho de la franja, los errores en la muestra y la exactitud con validación cruzada de 5 pliegos estratificados.

# %%
VALORES_C = [0.01, 0.1, 1, 100]
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
modelos_c, filas = {}, []
for c in VALORES_C:
    m = SVC(kernel="linear", C=c).fit(X2, y2)
    modelos_c[c] = m
    filas.append({
        "C": c,
        "vectores de soporte": len(m.support_),
        "ancho 2/‖w‖": 2 / np.linalg.norm(m.coef_[0]),
        "errores en la muestra": int(np.sum(m.predict(X2) != y2)),
        "exactitud (VC 5)": cross_val_score(SVC(kernel="linear", C=c), X2, y2, cv=cv5).mean(),
    })
tabla_c = pd.DataFrame(filas).set_index("C")
tabla_c.round(3)

# %%
LIM2 = ((-2.6, 3.1), (-2.5, 2.9))
fig, axs = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
for ax, c in zip(axs.ravel(), VALORES_C):
    m = modelos_c[c]
    dibujar_svm(ax, m, X2, y2, LIM2, etiquetas=("Adelia (−1)", "barbijo (+1)"))
    errores = m.predict(X2) != y2
    ax.scatter(*X2[errores].T, marker="x", s=50, color=TINTA, linewidths=1.6)
    f = tabla_c.loc[c]
    ax.set_title(f"C = {c}: {f['vectores de soporte']:.0f} VS, ancho {f['ancho 2/‖w‖']:.2f}, "
                 f"{f['errores en la muestra']:.0f} errores", fontsize=12.5)
    ax.set_aspect("equal")
for ax in axs[1]:
    ax.set_xlabel("longitud del pico (estandarizada)")
for ax in axs[:, 0]:
    ax.set_ylabel("profundidad del pico (estandarizada)")
axs[0, 0].text(1.9, -2.2, "barbijo", color=AZUL, fontsize=13, weight="semibold")
axs[0, 0].text(-2.4, -2.2, "Adelia", color=ROJO, fontsize=13, weight="semibold")
plt.tight_layout()
fig.text(0.5, -0.01, "círculos: vectores de soporte · cruces: errores en la muestra · línea continua: frontera · punteadas: orillas",
         ha="center", color=TENUE, fontsize=11.5)
plt.show()

# %% [markdown]
# Al subir $C$ la franja se angosta (de 2.4 a 0.3 desviaciones estándar) y el número de vectores de soporte baja de 129 a 12: con $C = 0.01$ más de la mitad de los pingüinos están dentro de la franja y todos «votan» en la solución; con $C = 100$ solo los puntos difíciles. Los errores en la muestra bajan un poco (de 10 a 5), pero la exactitud por validación cruzada **no** mejora a partir de $C = 0.1$: cerca de 97 % para $C \ge 0.1$. Con $C = 0.01$ la franja es tan ancha que la frontera se inclina hacia la clase grande (Adelia, 151 contra 68) y pierde exactitud. $C$ es un hiperparámetro de regularización, y se elige con validación.
#
# La cota de dejar uno fuera también vale con margen suave (el argumento es el mismo: si $\alpha_n = 0$ el punto no influye). Veamos qué tan útil es aquí.

# %%
filas = []
for c in VALORES_C:
    loo = 1 - cross_val_score(SVC(kernel="linear", C=c), X2, y2, cv=LeaveOneOut()).mean()
    filas.append({"C": c, "E_cv (LOO)": loo, "cota #VS/N": len(modelos_c[c].support_) / N2})
pd.DataFrame(filas).set_index("C").round(3)

# %% [markdown]
# Con muchos vectores de soporte la cota se vuelve inútil (con $C = 0.01$ dice que el error es menor que 59 %); con $C = 100$ es razonable. La lección: un SVM con **pocos** vectores de soporte trae su propio certificado de generalización.
#
# ### El margen suave es una pérdida con regularización
#
# Fijemos $\mathbf{w}$ y $b$ y escribamos $s_n = \mathbf{w}^\top\mathbf{x}_n + b$ (la «calificación» del punto). La holgura más pequeña que cumple las dos restricciones $\xi_n \ge 1 - y_n s_n$ y $\xi_n \ge 0$ es
#
# $$
# \xi_n = \max(0,\; 1 - y_n s_n),
# $$
#
# y como el objetivo crece con cada $\xi_n$, en el óptimo las holguras toman justo ese valor. Sustituyendo, el problema con restricciones se convierte en uno **sin** restricciones:
#
# $$
# \min_{\mathbf{w}, b}\; C\sum_{n=1}^N \max(0, 1 - y_n s_n) + \tfrac12\lVert\mathbf{w}\rVert^2
# \quad\Longleftrightarrow\quad
# \min_{\mathbf{w}, b}\; \frac1N\sum_{n=1}^N \underbrace{\max(0, 1 - y_n s_n)}_{\text{pérdida bisagra}} + \lambda\lVert\mathbf{w}\rVert^2,
# \qquad \lambda = \frac{1}{2CN},
# $$
#
# (dividimos entre $CN$, que no cambia el minimizador). Es exactamente la forma «$E_{\text{in}}$ + regularización» de la Unidad 3: un error en la muestra medido con la **pérdida bisagra** (*hinge*) más una penalización *ridge*. $C$ grande es $\lambda$ pequeño. Comprobémoslo minimizando la forma sin restricciones con un método que no usa derivadas (la bisagra no es derivable en $y s = 1$), para $C = 1$:

# %%
def objetivo_bisagra(v, X, y, lam):
    w, b = v[:-1], v[-1]
    return np.mean(np.maximum(0, 1 - y * (X @ w + b))) + lam * w @ w


lam = 1 / (2 * 1 * N2)
opciones = {"xtol": 1e-10, "ftol": 1e-14, "maxiter": 100_000}
r = minimize(objetivo_bisagra, np.zeros(3), args=(X2, y2, lam), method="Powell", options=opciones)
# un refinamiento con Nelder-Mead a partir de ahí: la bisagra tiene «esquinas» y Powell se atora cerca del óptimo
r = minimize(objetivo_bisagra, r.x, args=(X2, y2, lam), method="Nelder-Mead",
             options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 100_000})
m1 = modelos_c[1]
print(f"bisagra + λ‖w‖² (scipy):  w = {np.round(r.x[:2], 4)}, b = {r.x[2]:.4f}")
print(f"SVC(C=1):                 w = {np.round(m1.coef_[0], 4)}, b = {m1.intercept_[0]:.4f}")

# %% [markdown]
# Coinciden. Esto pone al SVM en la misma familia que los modelos que ya vimos: **todos** minimizan una pérdida que depende de $y\,s$ más una regularización. La regresión logística usa $\ln(1 + e^{-ys})$; la regresión lineal usada como clasificador, $(y - s)^2 = (1 - ys)^2$ cuando $y = \pm 1$. Las graficamos junto con la pérdida 0-1, que es la que de verdad nos importa. Escalamos la logística a base 2, $\log_2(1 + e^{-ys})$, para que valga 1 en $ys = 0$ como las demás.

# %%
z = np.linspace(-2.5, 3, 500)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(z, (z <= 0).astype(float), color=TINTA, lw=2.6)
ax.plot(z, np.maximum(0, 1 - z), color=AZUL)
ax.plot(z, np.log2(1 + np.exp(-z)), color=ROJO)
ax.plot(z, (1 - z) ** 2, color=DORADO)
ax.axvline(0, color=TENUE, lw=0.8)
ax.set(xlim=(-2.5, 3), ylim=(-0.1, 4), xlabel="$y\\,s$ (positivo: bien clasificado)", ylabel="pérdida",
       title="Pérdidas sustitutas de la pérdida 0-1")
ax.text(-2.35, 1.12, "0-1", color=TINTA)
ax.text(-1.55, 2.75, "bisagra", color=AZUL)
ax.text(-2.4, 3.6, "logística ($\\log_2$)", color=ROJO)
ax.text(2.25, 2.0, "cuadrática", color=DORADO)
plt.show()

# %% [markdown]
# Las tres sustitutas son **convexas** (la 0-1 no lo es, y minimizarla directamente es un problema NP-difícil) y quedan **por encima** de la 0-1, así que minimizarlas acota el error de clasificación. Sus diferencias explican el comportamiento de cada modelo:
#
# - La **bisagra** vale exactamente cero cuando $ys \ge 1$: los puntos bien clasificados con holgura no aportan nada. De ahí sale que la solución dependa solo de los vectores de soporte.
# - La **logística** nunca llega a cero: todos los puntos influyen un poco. A cambio, da probabilidades.
# - La **cuadrática** castiga también a los puntos **demasiado** bien clasificados ($ys > 1$): un pingüino muy típico, lejos de la frontera, jala la recta hacia él. Por eso la regresión lineal es mala clasificadora.

# %% [markdown]
# ## 5. Kernels
#
# ### Transformaciones no lineales, otra vez
#
# En la Unidad 3 vimos que una transformación $\mathbf{z} = \phi(\mathbf{x})$ convierte un modelo lineal en $\mathbf{z}$ en uno no lineal en $\mathbf{x}$. El precio es que $\mathbf{z}$ puede tener dimensión enorme. El dual nos da una salida: los datos **solo** aparecen como productos internos $\mathbf{z}_n^\top\mathbf{z}_m$, y la predicción también,
#
# $$
# g(\mathbf{x}) = \operatorname{signo}\Big(\sum_{n \in \text{VS}} \alpha_n y_n\, \phi(\mathbf{x}_n)^\top\phi(\mathbf{x}) + b\Big).
# $$
#
# Si encontramos una función $K(\mathbf{x}, \mathbf{x}')$ que calcule $\phi(\mathbf{x})^\top\phi(\mathbf{x}')$ **sin construir** $\phi$, podemos trabajar en el espacio transformado al costo de trabajar en el original. A esa función se le llama **kernel**, y a sustituir los productos internos por $K$, el **truco del kernel**.
#
# ### Un ejemplo explícito: el kernel polinomial de grado 2
#
# Sea $K(\mathbf{x}, \mathbf{x}') = (1 + \mathbf{x}^\top\mathbf{x}')^2$ en $\mathbb{R}^2$. Desarrollando,
#
# $$
# (1 + x_1x_1' + x_2x_2')^2 = 1 + 2x_1x_1' + 2x_2x_2' + x_1^2x_1'^2 + x_2^2x_2'^2 + 2x_1x_2x_1'x_2',
# $$
#
# que es el producto interno $\phi(\mathbf{x})^\top\phi(\mathbf{x}')$ con
#
# $$
# \phi(\mathbf{x}) = \big(1,\; \sqrt2\,x_1,\; \sqrt2\,x_2,\; x_1^2,\; x_2^2,\; \sqrt2\,x_1x_2\big).
# $$
#
# Calcular $K$ cuesta un producto interno en $\mathbb{R}^2$ y un cuadrado; calcular $\phi$ y luego el producto, seis multiplicaciones más. Con $d$ variables y grado $Q$, $\phi$ tiene $\binom{d+Q}{Q}$ coordenadas (para $d = 64$ y $Q = 4$, más de 800 mil) y $K$ sigue costando $O(d)$. Verifiquemos la identidad con números al azar:

# %%
def phi2(X):
    x1, x2 = X[:, 0], X[:, 1]
    r2 = np.sqrt(2)
    return np.c_[np.ones(len(X)), r2 * x1, r2 * x2, x1**2, x2**2, r2 * x1 * x2]


rng = np.random.default_rng(SEMILLA)
A, B = rng.normal(size=(5, 2)), rng.normal(size=(4, 2))
K_directo = (1 + A @ B.T) ** 2
K_phi = phi2(A) @ phi2(B).T
print("K(x, x') con la fórmula:\n", K_directo.round(4))
print(f"máxima diferencia con φ(x)·φ(x'): {np.abs(K_directo - K_phi).max():.1e}")
print(f"y con polynomial_kernel de scikit-learn: {np.abs(K_directo - polynomial_kernel(A, B, degree=2, gamma=1, coef0=1)).max():.1e}")

# %% [markdown]
# ### Un problema que una recta no puede resolver
#
# Generamos dos anillos concéntricos con `make_circles`: 300 puntos, el anillo interior con radio 0.4 veces el exterior, y un poco de ruido. Ninguna recta en $(x_1, x_2)$ los separa. Pero si agregamos la coordenada $x_1^2 + x_2^2$ (el cuadrado de la distancia al origen), el anillo interior queda «abajo» y el exterior «arriba», y un **plano** en $(x_1, x_2, x_1^2 + x_2^2)$ los separa. Ese plano, visto en el espacio original, es un círculo.

# %%
Xc, yc = make_circles(n_samples=300, factor=0.4, noise=0.06, random_state=SEMILLA)
yc = np.where(yc == 1, 1, -1)  # 1 = anillo interior
Zc = np.c_[Xc, (Xc**2).sum(axis=1)]

lineal_2d = SVC(kernel="linear", C=1e6).fit(Xc, yc)
lineal_3d = SVC(kernel="linear", C=1e6).fit(Zc, yc)
print(f"recta en (x1, x2):                   exactitud en la muestra = {lineal_2d.score(Xc, yc):.3f}")
print(f"plano en (x1, x2, x1² + x2²):        exactitud en la muestra = {lineal_3d.score(Zc, yc):.3f}, "
      f"{len(lineal_3d.support_)} VS")

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
ax = axs[0]
dibujar_puntos(ax, Xc, yc, ("exterior (−1)", "interior (+1)"), s=22)
g1, g2 = np.meshgrid(np.linspace(-1.4, 1.4, 300), np.linspace(-1.4, 1.4, 300))
G = np.c_[g1.ravel(), g2.ravel()]
s = lineal_3d.decision_function(np.c_[G, (G**2).sum(axis=1)]).reshape(g1.shape)
ax.contour(g1, g2, s, levels=[-1, 0, 1], colors=[TINTA], linestyles=["--", "-", "--"], linewidths=[1.2, 2.2, 1.2])
ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
       title="Espacio original: la frontera es un círculo")
ax = axs[1]
dibujar_puntos(ax, np.c_[Xc[:, 0], Zc[:, 2]], yc, s=22)
w3, b3 = lineal_3d.coef_[0], lineal_3d.intercept_[0]
print(f"plano: {w3[0]:.2f}·x1 + {w3[1]:.2f}·x2 + {w3[2]:.2f}·(x1² + x2²) + {b3:.2f} = 0")
ax.axhline(-b3 / w3[2], color=TINTA, lw=2.2)
ax.set(xlabel="$x_1$", ylabel="$x_1^2 + x_2^2$", title="Con la nueva coordenada basta un plano")
ax.text(-1.15, 0.05, "interior", color=AZUL, fontsize=13, weight="semibold")
ax.text(0.12, 1.42, "exterior", color=ROJO, fontsize=13, weight="semibold")
ax.set_ylim(-0.05, 1.52)
plt.tight_layout()
plt.show()

# %% [markdown]
# En el panel derecho, la línea horizontal es el corte aproximado del plano (los coeficientes de $x_1$ y $x_2$ salen casi cero, así que el plano es prácticamente $x_1^2 + x_2^2 = \text{constante}$). El kernel polinomial de grado 2 contiene a $x_1^2$ y $x_2^2$ entre sus coordenadas, así que `SVC(kernel="poly", degree=2)` encontraría esta frontera **sin** que nosotros escribiéramos la nueva coordenada.
#
# ### El kernel gaussiano: un producto interno en dimensión infinita
#
# El kernel más usado es el **gaussiano** o RBF (*radial basis function*):
#
# $$
# K(\mathbf{x}, \mathbf{x}') = \exp\big(-\gamma\lVert\mathbf{x} - \mathbf{x}'\rVert^2\big), \qquad \gamma > 0.
# $$
#
# Es una **similitud**: vale 1 si los puntos coinciden y decae a 0 cuando se alejan; $\gamma$ fija qué tan rápido. ¿De qué $\phi$ es producto interno? Expandiendo $\lVert\mathbf{x} - \mathbf{x}'\rVert^2 = \lVert\mathbf{x}\rVert^2 + \lVert\mathbf{x}'\rVert^2 - 2\mathbf{x}^\top\mathbf{x}'$,
#
# $$
# K(\mathbf{x}, \mathbf{x}') = e^{-\gamma\lVert\mathbf{x}\rVert^2}\, e^{-\gamma\lVert\mathbf{x}'\rVert^2}\, e^{2\gamma\,\mathbf{x}^\top\mathbf{x}'}
# = e^{-\gamma\lVert\mathbf{x}\rVert^2}\, e^{-\gamma\lVert\mathbf{x}'\rVert^2} \sum_{k=0}^{\infty} \frac{(2\gamma)^k}{k!}\,(\mathbf{x}^\top\mathbf{x}')^k.
# $$
#
# Cada $(\mathbf{x}^\top\mathbf{x}')^k$ es un kernel polinomial: el producto interno de todos los monomios de grado $k$. La suma es entonces el producto interno de un vector que concatena **todos los monomios de todos los grados**, con pesos $\sqrt{(2\gamma)^k/k!}$, multiplicado por $e^{-\gamma\lVert\mathbf{x}\rVert^2}$. Un espacio de dimensión infinita, y aun así $K$ cuesta $O(d)$. Lo que impide que el SVM se sobreajuste sin remedio en ese espacio es el margen (recuerda que la cota VC con margen no depende de la dimensión), y en la práctica, elegir $C$ y $\gamma$ con validación.
#
# ### ¿Qué funciones son kernels? La condición de Mercer
#
# No cualquier función de similitud sirve. $K$ es un kernel válido (es decir, existe $\phi$ con $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top\phi(\mathbf{x}')$) si y solo si es simétrica y, para **cualquier** conjunto de puntos $\mathbf{x}_1, \dots, \mathbf{x}_N$, la **matriz de Gram** $\mathbf{K}_{nm} = K(\mathbf{x}_n, \mathbf{x}_m)$ es **semidefinida positiva**. La necesidad es fácil: si $\mathbf{K} = \boldsymbol{\Phi}\boldsymbol{\Phi}^\top$, entonces $\mathbf{v}^\top\mathbf{K}\mathbf{v} = \lVert\boldsymbol{\Phi}^\top\mathbf{v}\rVert^2 \ge 0$. La suficiencia es el teorema de Mercer. La condición importa porque garantiza que el dual sigue siendo un QP **convexo**.
#
# Revisemos los valores propios de la matriz de Gram en los 219 pingüinos de la sección 4 para tres funciones. La tercera, $\tanh(\mathbf{x}^\top\mathbf{x}' - 1)$, es el llamado kernel «sigmoide», que scikit-learn ofrece pero que **no** cumple la condición de Mercer en general.

# %%
grams = {
    "polinomial (1 + x·x')²": polynomial_kernel(X2, degree=2, gamma=1, coef0=1),
    "gaussiano, γ = 1": rbf_kernel(X2, gamma=1),
    "sigmoide tanh(x·x' − 1)": sigmoid_kernel(X2, gamma=1, coef0=-1),
}
for nombre, K in grams.items():
    ev = np.linalg.eigvalsh(K)
    print(f"{nombre:<26} valor propio mínimo = {ev.min():10.2e}   máximo = {ev.max():8.1f}   "
          f"negativos (< −1e−8): {np.sum(ev < -1e-8)}")

# %% [markdown]
# Los dos primeros tienen todos sus valores propios no negativos (los «negativos» del orden de $10^{-14}$ son error de redondeo). El sigmoide tiene valores propios claramente negativos: no corresponde a ningún producto interno, y con él el dual puede dejar de ser convexo.
#
# ### El papel de $\gamma$
#
# Con el kernel gaussiano, la función de decisión es una suma de «campanas» centradas en los vectores de soporte, $\sum_n \alpha_n y_n e^{-\gamma\lVert\mathbf{x}_n - \mathbf{x}\rVert^2} + b$. Con $\gamma$ pequeño las campanas son anchas y la frontera es suave (casi lineal); con $\gamma$ grande son estrechas, cada punto tiene su propia isla y la frontera se vuelve arbitrariamente complicada. Lo vemos con los pingüinos Adelia y barbijo, con $C = 1$.

# %%
GAMMAS = [0.1, 5, 100]
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
for ax, gam in zip(axs, GAMMAS):
    m = SVC(kernel="rbf", C=1, gamma=gam).fit(X2, y2)
    exac_cv = cross_val_score(SVC(kernel="rbf", C=1, gamma=gam), X2, y2, cv=cv5).mean()
    dibujar_svm(ax, m, X2, y2, LIM2, etiquetas=("Adelia (−1)", "barbijo (+1)"))
    ax.set_title(f"γ = {gam}: {len(m.support_)} VS\nexactitud: muestra {m.score(X2, y2):.3f}, VC {exac_cv:.3f}", fontsize=12.5)
    ax.set(aspect="equal", xlabel="longitud del pico (est.)")
axs[0].set_ylabel("profundidad del pico (est.)")
axs[0].text(-2.4, -2.2, "Adelia", color=ROJO, fontsize=13, weight="semibold")
axs[0].text(1.9, -2.2, "barbijo", color=AZUL, fontsize=13, weight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# - Con $\gamma = 0.1$ la frontera es casi una recta: el modelo se parece al SVM lineal de la sección 4.
# - Con $\gamma = 5$ la frontera se curva para acomodar a los pingüinos de la zona de traslape: gana en la muestra, pero no en validación cruzada. Aquí la recta ya bastaba; la flexibilidad extra no compra nada.
# - Con $\gamma = 100$ cada campana cubre apenas los alrededores de su punto: el modelo casi memoriza la muestra y fuera de las islas de barbijo todo es Adelia. Casi todos los puntos son vectores de soporte y la validación cruzada se desploma. Es **sobreajuste**, y la cota $\#\text{VS}/N$ lo anuncia.

# %% [markdown]
# ## 6. Práctica: reconocimiento de dígitos escritos a mano
#
# ### Los datos y el problema
#
# **Datos.** La colección `digits` de scikit-learn es una versión reducida de la base de dígitos manuscritos que E. Alpaydin y C. Kaynak (Universidad de Boğaziçi, Estambul) depositaron en el repositorio de la UCI en 1998. Cuarenta y tres personas llenaron formularios con dígitos; cada dígito escaneado se dividió en una cuadrícula de $8 \times 8$ celdas y se contó cuántos píxeles negros había en cada celda, un número de 0 a 16. Son 1,797 imágenes, unas 180 por dígito.
#
# **Problema.** Reconocer el dígito (0 a 9) a partir de las 64 intensidades. Es un problema de 10 clases; `SVC` lo resuelve entrenando un SVM binario para cada **par** de clases (45 en total) y decidiendo por votación (*uno contra uno*). Dividimos las intensidades entre 16 para dejarlas en $[0, 1]$ y separamos 25 % para prueba.

# %%
Xd, yd = load_digits(return_X_y=True)
Xd = Xd / 16
Xd_ent, Xd_pru, yd_ent, yd_pru = train_test_split(Xd, yd, test_size=0.25, stratify=yd, random_state=SEMILLA)
print(f"entrenamiento: {len(yd_ent)} imágenes · prueba: {len(yd_pru)} imágenes · {Xd.shape[1]} variables")

fig, axs = plt.subplots(2, 10, figsize=(13, 3))
for ax, i in zip(axs.ravel(), [np.flatnonzero(yd_ent == k)[j] for j in range(2) for k in range(10)]):
    ax.imshow(Xd_ent[i].reshape(8, 8), cmap="gray_r")
    ax.axis("off")
fig.suptitle("Ejemplos del conjunto de entrenamiento (8 × 8 celdas)", color=rdp.estilo.MARINO, x=0.02, ha="left", weight="semibold")
plt.show()

# %% [markdown]
# ### Búsqueda en malla de $C$ y $\gamma$
#
# Con el kernel gaussiano hay dos hiperparámetros, y hay que elegirlos **juntos**: un $\gamma$ grande (modelo flexible) pide un $C$ más pequeño (más regularización) y viceversa. Hacemos una búsqueda en malla con validación cruzada de 5 pliegos sobre el conjunto de entrenamiento, con $C$ y $\gamma$ en escala aproximadamente logarítmica (dos valores por década).

# %%
MALLA_C = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000])
MALLA_G = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
t0 = time.perf_counter()
busqueda = GridSearchCV(SVC(kernel="rbf"), {"C": MALLA_C, "gamma": MALLA_G}, cv=cv5).fit(Xd_ent, yd_ent)
print(f"{len(MALLA_C) * len(MALLA_G)} combinaciones × 5 pliegos en {time.perf_counter() - t0:.0f} s")
mejor = busqueda.best_estimator_
print(f"mejor: C = {busqueda.best_params_['C']:g}, γ = {busqueda.best_params_['gamma']:g}, "
      f"exactitud VC = {busqueda.best_score_:.4f}")
print(f"vectores de soporte: {mejor.n_support_.sum()} de {len(yd_ent)} ({mejor.n_support_.sum() / len(yd_ent):.0%})")
print(f"exactitud en prueba: {mejor.score(Xd_pru, yd_pru):.4f}")

# %%
exactitud = busqueda.cv_results_["mean_test_score"].reshape(len(MALLA_C), len(MALLA_G))
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(exactitud, cmap=rdp.estilo.SECUENCIAL, vmin=0, vmax=1, origin="lower")
for i in range(len(MALLA_C)):
    for j in range(len(MALLA_G)):
        v = exactitud[i, j]
        ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1.0", ha="center", va="center", fontsize=10,
                color="white" if v > 0.6 else TINTA)
i_m, j_m = np.unravel_index(np.argmax(exactitud), exactitud.shape)
ax.add_patch(plt.Rectangle((j_m - 0.5, i_m - 0.5), 1, 1, fill=False, edgecolor=DORADO, lw=3))
ax.set_xticks(range(len(MALLA_G)), [f"{g:g}" for g in MALLA_G], rotation=45)
ax.set_yticks(range(len(MALLA_C)), [f"{c:g}" for c in MALLA_C])
ax.set(xlabel="$\\gamma$", ylabel="$C$", title="Exactitud por validación cruzada (5 pliegos)")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="exactitud")
plt.show()

# %% [markdown]
# El mapa tiene la forma típica:
#
# - **Columnas de la derecha** ($\gamma \ge 3$): campanas tan estrechas que cada imagen solo se parece a sí misma; el modelo memoriza y, fuera de la muestra, se equivoca mucho. Con $\gamma = 10$ acierta 15 % (el azar puro daría 10 %), sin importar $C$: ninguna cantidad de regularización compensa un kernel que no generaliza.
# - **Esquina inferior izquierda** ($C$ y $\gamma$ pequeños): demasiada regularización con un kernel casi lineal; el modelo es muy rígido. La frontera de esa zona es una diagonal: si $\gamma$ baja, hace falta un $C$ más grande para compensar.
# - Una **región amplia** de buenos valores, con $\gamma$ entre 0.03 y 0.3 y $C \ge 3$, todos con exactitud de 0.99 al redondear. El recuadro dorado marca el máximo, pero hay muchas combinaciones casi igual de buenas, lo que es una buena noticia: el resultado no depende de afinar con precisión.
#
# ### Comparación con la regresión logística
#
# ¿Cuánto aportan la no linealidad y el kernel? Comparamos con la regresión logística multinomial (lineal en los 64 píxeles), con su $C$ también elegido por validación cruzada, y con el SVM **lineal**.

# %%
logistica = GridSearchCV(LogisticRegression(max_iter=5000), {"C": np.logspace(-2, 3, 6)}, cv=cv5).fit(Xd_ent, yd_ent)
svm_lineal = GridSearchCV(SVC(kernel="linear"), {"C": np.logspace(-2, 2, 5)}, cv=cv5).fit(Xd_ent, yd_ent)
comparacion = pd.DataFrame({
    "modelo": ["regresión logística", "SVM lineal", "SVM gaussiano"],
    "hiperparámetros": [f"C = {logistica.best_params_['C']:g}", f"C = {svm_lineal.best_params_['C']:g}",
                        f"C = {busqueda.best_params_['C']:g}, γ = {busqueda.best_params_['gamma']:g}"],
    "exactitud VC": [logistica.best_score_, svm_lineal.best_score_, busqueda.best_score_],
    "exactitud prueba": [logistica.score(Xd_pru, yd_pru), svm_lineal.score(Xd_pru, yd_pru), mejor.score(Xd_pru, yd_pru)],
}).set_index("modelo")
comparacion.round(4)

# %%
errores_log = int(np.sum(logistica.predict(Xd_pru) != yd_pru))
errores_svm = int(np.sum(mejor.predict(Xd_pru) != yd_pru))
print(f"errores en las {len(yd_pru)} imágenes de prueba: logística {errores_log}, SVM gaussiano {errores_svm}")

# %% [markdown]
# Los modelos lineales ya son buenos (los dígitos de $8 \times 8$ son un problema relativamente fácil), pero el kernel gaussiano reduce los errores de prueba a una fracción de los de la logística. Ojo: con 450 imágenes de prueba, cada error vale 0.2 puntos porcentuales, así que diferencias de uno o dos errores no significan nada; la diferencia entre la logística y el SVM gaussiano sí es clara, y la validación cruzada la confirma.
#
# ### El costo computacional
#
# El SVM con kernel tiene un talón de Aquiles: el dual tiene $N$ variables y la matriz de Gram tiene $N^2$ entradas. Los algoritmos especializados (como SMO, que usa `libsvm`, la biblioteca detrás de `SVC`) no forman la matriz completa, pero su tiempo crece entre $O(N^2)$ y $O(N^3)$ según los datos y $C$. Con decenas de miles de puntos ya es lento, y con millones, impráctico. Además, predecir cuesta $O(\#\text{VS} \cdot d)$ por punto, porque hay que comparar con cada vector de soporte.
#
# Midamos cómo crece el tiempo de entrenamiento con $N$. Con los dígitos el entrenamiento tarda unos milisegundos y los costos fijos esconden el crecimiento, así que usamos datos sintéticos más difíciles: `make_classification` con 20 variables y 10 % de etiquetas volteadas al azar, de modo que muchos puntos queden dentro del margen.

# %%
tamanos = [1000, 2000, 4000, 8000]
filas = []
for n in tamanos:
    Xs, ys = make_classification(n_samples=n, n_features=20, n_informative=10, flip_y=0.1, random_state=SEMILLA)
    t0 = time.perf_counter()
    m = SVC(kernel="rbf").fit(Xs, ys)
    filas.append({"N": n, "segundos": time.perf_counter() - t0, "vectores de soporte": len(m.support_)})
tiempos = pd.DataFrame(filas).set_index("N")
pendiente = np.polyfit(np.log(tiempos.index), np.log(tiempos["segundos"]), 1)[0]
print(tiempos.round(3))
print(f"pendiente en escala log-log: {pendiente:.2f}  (tiempo ∝ N^{pendiente:.1f})")

# %% [markdown]
# Cada vez que $N$ se duplica, el tiempo se multiplica por bastante más de 2. (La pendiente exacta varía de una máquina a otra y depende de cuántos vectores de soporte haya.) Para conjuntos grandes hay dos alternativas comunes:
#
# - **SVM lineal** (`LinearSVC`, `SGDClassifier(loss="hinge")`): resuelve el primal directamente, sin kernel, en tiempo casi lineal en $N$. Con muchas variables (texto, por ejemplo) suele bastar.
# - **Características aleatorias de Fourier** (Rahimi y Recht, NIPS 2007; `RBFSampler` en scikit-learn): construyen un $\phi$ **explícito** de dimensión finita $D$ (unos cientos o miles) tal que $\phi(\mathbf{x})^\top\phi(\mathbf{x}') \approx e^{-\gamma\lVert\mathbf{x} - \mathbf{x}'\rVert^2}$, y luego entrenan un modelo lineal sobre $\phi(\mathbf{x})$. Es el truco del kernel al revés: renunciamos a la dimensión infinita a cambio de un costo lineal en $N$.

# %% [markdown]
# ## 7. Cierre
#
# - Entre todas las separadoras con $E_{\text{in}} = 0$, el SVM elige la de **margen máximo**. Con la normalización $\min_n y_n(\mathbf{w}^\top\mathbf{x}_n + b) = 1$, eso es minimizar $\tfrac12\lVert\mathbf{w}\rVert^2$ con restricciones lineales: un problema cuadrático convexo con solución única.
# - El **dual** muestra que $\mathbf{w} = \sum_n\alpha_n y_n\mathbf{x}_n$ con $\alpha_n > 0$ solo en los **vectores de soporte**. De ahí sale la cota $E_{\text{cv}} \le \#\text{VS}/N$, que no depende de la dimensión.
# - El **margen suave** admite violaciones pagando $C\sum_n\xi_n$; equivale a minimizar la **pérdida bisagra** más $\lambda\lVert\mathbf{w}\rVert^2$, con $\lambda = 1/(2CN)$. $C$ es un parámetro de regularización.
# - El dual solo usa **productos internos**: sustituirlos por un **kernel** $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top\phi(\mathbf{x}')$ da fronteras no lineales sin construir $\phi$, incluso en dimensión infinita (kernel gaussiano). $K$ es válido si sus matrices de Gram son semidefinidas positivas (Mercer).
# - Con el kernel gaussiano, $C$ y $\gamma$ se eligen juntos por validación cruzada. El costo crece entre $N^2$ y $N^3$; para datos masivos se usan SVM lineales o características aleatorias.

# %% [markdown]
# ## 8. Autoevaluación

# %%
rdp.quiz.mostrar("u04", ids=["u04-p13", "u04-p14"])

# %% [markdown]
# ## 9. Ejercicios
#
# 1. **Margen duro a mano.** Tres puntos en $\mathbb{R}^2$: $\mathbf{x}_1 = (0, 0)$ con $y_1 = -1$, $\mathbf{x}_2 = (2, 2)$ y $\mathbf{x}_3 = (2, 0)$ con $y = +1$. Encuentra a mano la separadora de margen máximo resolviendo el dual (pista: por simetría, adivina cuáles son los vectores de soporte y usa $\sum_n\alpha_ny_n = 0$). Verifica con `SVC(kernel="linear", C=1e6)` y con la función `primal_duro` de esta libreta.
# 2. **Demostración.** Demuestra que en el óptimo del margen duro $\lVert\mathbf{w}\rVert^2 = \sum_n \alpha_n$, y concluye que el ancho de la franja es $2/\sqrt{\sum_n\alpha_n}$. (Pista: multiplica la condición de holgura complementaria por $\alpha_n$ y suma sobre $n$.) Comprueba la identidad con las $\alpha_n$ de la sección 3.
# 3. **Los vectores de soporte del margen suave.** Escribe las condiciones KKT del margen suave (con multiplicadores $\alpha_n$ para $y_n s_n \ge 1 - \xi_n$ y $\mu_n$ para $\xi_n \ge 0$) y demuestra que $0 \le \alpha_n \le C$, que $\alpha_n = C$ para los puntos dentro de la franja, y que $0 < \alpha_n < C$ implica que el punto está exactamente en la orilla. Para $C = 1$, cuenta en `dual_coef_` cuántos vectores de soporte tienen $|\alpha_n y_n| = C$ y compáralo con el número de puntos con $y_n s_n < 1$.
# 4. **Construir kernels.** Demuestra, usando la condición de Mercer, que si $K_1$ y $K_2$ son kernels válidos entonces $K_1 + K_2$ y $cK_1$ (con $c > 0$) también lo son. Para el producto $K_1K_2$ el resultado también es cierto (teorema de Schur); verifícalo numéricamente con las matrices de Gram de la sección 5.
# 5. **Características aleatorias.** Usa `RBFSampler(gamma=γ, n_components=D, random_state=0)` con el mejor $\gamma$ de la sección 6 y un `LinearSVC` encima, para $D \in \{50, 200, 1000\}$. Compara exactitud de prueba y tiempo de entrenamiento con el SVC gaussiano. ¿Con qué $D$ se acerca?
