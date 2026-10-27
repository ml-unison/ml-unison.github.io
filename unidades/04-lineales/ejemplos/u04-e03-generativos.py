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
# # Modelos generativos: análisis discriminante y Bayes ingenuo
#
# **Unidad 4 · Ejemplo 3**
#
# Hay dos maneras de construir un clasificador probabilístico.
#
# - **Discriminativa.** Se modela directamente lo que queremos, $p(y \mid \mathbf{x})$. Es lo que hace la regresión logística: no le importa cómo se distribuyen los $\mathbf{x}$, solo dónde está la frontera entre clases.
# - **Generativa.** Se modela cómo se **generan** los datos de cada clase, $p(\mathbf{x} \mid y)$, y qué tan frecuente es cada clase, $p(y)$. Luego se invierte con el teorema de Bayes:
#
# $$
# p(y = k \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid y = k)\, p(y = k)}{\sum_{l} p(\mathbf{x} \mid y = l)\, p(y = l)}.
# $$
#
# Un modelo generativo es más ambicioso: describe la población entera de cada clase, y con él podríamos incluso **simular** pingüinos o mensajes nuevos. Ese esfuerzo extra se paga con supuestos más fuertes, pero se cobra con estimadores muy simples (promedios y conteos) y con buen desempeño cuando hay pocos datos.
#
# En esta libreta vamos a:
#
# 1. construir desde cero el **análisis discriminante lineal** (LDA) con los pingüinos, y ver por qué su frontera es lineal;
# 2. relajar un supuesto para obtener el **análisis discriminante cuadrático** (QDA) y comparar los tres modelos;
# 3. usar LDA como **reducción de dimensión** y compararlo con PCA;
# 4. clasificar mensajes de texto en spam y normales con **Bayes ingenuo**, y descubrir que también es un clasificador lineal;
# 5. comparar generativo contra discriminativo con **curvas de aprendizaje**.

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
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Ellipse
from scipy.linalg import eigh
from scipy.special import logsumexp
from scipy.stats import chi2

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS

# %% [markdown]
# # Parte 1. Análisis discriminante con los pingüinos
#
# ## 1. Los datos y el problema
#
# Son los pingüinos del archipiélago Palmer, en la Antártida, que ya usamos en la Unidad 2 para agrupar: 344 aves de tres especies (Adelie, Chinstrap y Gentoo) medidas por Kristen Gorman entre 2007 y 2009, con el largo y la profundidad del pico, el largo de la aleta y la masa corporal.
#
# **Problema.** En la Unidad 2 escondimos la especie. Ahora la usamos: queremos un clasificador que, a partir de las **dos medidas del pico**, diga de qué especie es un pingüino. Con dos variables podemos dibujar todo: las nubes de cada clase, el modelo que el método generativo se hace de ellas y las fronteras de decisión que resultan.

# %%
pinguinos = rdp.datos.cargar("pinguinos")
pico = ["bill_length_mm", "bill_depth_mm"]
p = pinguinos.dropna(subset=pico).reset_index(drop=True)

X = p[pico].to_numpy()
y = p["species"].to_numpy()
ESPECIES = ["Adelie", "Chinstrap", "Gentoo"]
COLOR = {"Adelie": AZUL, "Chinstrap": ROJO, "Gentoo": DORADO_OSC}
N, d = X.shape
K = len(ESPECIES)
print(f"N = {N} pingüinos, d = {d} medidas, K = {K} especies")
p["species"].value_counts()

# %% [markdown]
# ## 2. LDA desde cero
#
# ### El modelo
#
# El análisis discriminante lineal supone que, dentro de cada clase, los datos son **normales multivariados** con su propia media $\boldsymbol{\mu}_k$ pero con **la misma matriz de covarianza** $\boldsymbol{\Sigma}$ para todas las clases:
#
# $$
# p(\mathbf{x} \mid y = k) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\Big(-\tfrac12 (\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)\Big), \qquad p(y = k) = \pi_k.
# $$
#
# En palabras: las tres nubes tienen la misma forma y la misma inclinación, y solo cambia dónde están. Viendo la figura de la Unidad 2 (tres nubes alargadas e inclinadas en la misma dirección) el supuesto no parece descabellado.
#
# ### Estimación: promedios y conteos
#
# Los estimadores de máxima verosimilitud son los obvios. Si $N_k$ es el número de ejemplos de la clase $k$,
#
# $$
# \hat\pi_k = \frac{N_k}{N}, \qquad
# \hat{\boldsymbol{\mu}}_k = \frac{1}{N_k} \sum_{i:\, y_i = k} \mathbf{x}_i, \qquad
# \hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_{k=1}^{K} \sum_{i:\, y_i = k} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)^\top.
# $$
#
# $\hat{\boldsymbol{\Sigma}}$ es la **covarianza agrupada** (*pooled*): cada punto se centra en la media de **su** clase y se promedian todos los productos; equivale a promediar las covarianzas de cada clase, ponderadas por su tamaño. Muchos textos dividen entre $N - K$ en lugar de $N$ para hacerla insesgada (se pierde un grado de libertad por cada media estimada); scikit-learn usa $N$, el estimador de máxima verosimilitud, y nosotros también. Con $N = 342$ la diferencia es menor al 1 %. No hay optimización iterativa: el «entrenamiento» es una pasada por los datos.

# %%
pi_k = np.array([(y == k).mean() for k in ESPECIES])
mu_k = np.array([X[y == k].mean(axis=0) for k in ESPECIES])
centrados = X - mu_k[[ESPECIES.index(e) for e in y]]   # cada punto menos la media de su clase
Sigma = centrados.T @ centrados / N

print("previas π_k:", dict(zip(ESPECIES, pi_k.round(3))))
print("medias μ_k (mm):\n", pd.DataFrame(mu_k, index=ESPECIES, columns=pico).round(2))
print("covarianza agrupada Σ (mm²):\n", Sigma.round(3))

# %% [markdown]
# ### ¿Por qué la frontera es lineal?
#
# La regla de Bayes asigna $\mathbf{x}$ a la clase con mayor $p(y = k \mid \mathbf{x})$. El denominador de Bayes es el mismo para todas las clases, así que basta comparar $\log\big(p(\mathbf{x} \mid y = k)\,\pi_k\big)$:
#
# $$
# \log p(\mathbf{x} \mid y = k) + \log \pi_k = -\tfrac12 \mathbf{x}^\top \boldsymbol{\Sigma}^{-1}\mathbf{x} + \mathbf{x}^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k - \tfrac12 \boldsymbol{\mu}_k^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k + \log\pi_k \;\underbrace{- \tfrac12\log|\boldsymbol{\Sigma}| - \tfrac d2 \log 2\pi}_{\text{igual para toda } k}.
# $$
#
# El término cuadrático $-\tfrac12 \mathbf{x}^\top \boldsymbol{\Sigma}^{-1}\mathbf{x}$ **no depende de $k$**, porque todas las clases comparten $\boldsymbol{\Sigma}$. Al comparar clases se cancela, junto con las constantes, y queda la **función discriminante**
#
# $$
# \delta_k(\mathbf{x}) = \mathbf{x}^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k - \tfrac12 \boldsymbol{\mu}_k^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k + \log\pi_k
# = \mathbf{w}_k^\top \mathbf{x} + w_{k0},
# $$
#
# con $\mathbf{w}_k = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k$. Es **lineal** en $\mathbf{x}$. La frontera entre las clases $k$ y $l$ es el conjunto donde $\delta_k(\mathbf{x}) = \delta_l(\mathbf{x})$:
#
# $$
# (\mathbf{w}_k - \mathbf{w}_l)^\top \mathbf{x} + (w_{k0} - w_{l0}) = 0,
# $$
#
# un hiperplano (en el plano, una recta) con vector normal $\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_k - \boldsymbol{\mu}_l)$. Fíjate en el detalle: la recta **no** es perpendicular al segmento que une las medias (eso pasaría si $\boldsymbol{\Sigma}$ fuera la identidad); $\boldsymbol{\Sigma}^{-1}$ la inclina para tomar en cuenta la forma de las nubes. Y las previas la empujan hacia la clase menos frecuente: con $\pi_k > \pi_l$, la región de $k$ crece.
#
# Además, como $p(y = k \mid \mathbf{x}) \propto e^{\delta_k(\mathbf{x})}$, las probabilidades posteriores son un ***softmax*** de funciones lineales: exactamente la forma de la regresión logística multinomial. Volveremos a esta coincidencia en la Parte 2.

# %%
Sigma_inv = np.linalg.inv(Sigma)
W = mu_k @ Sigma_inv                                                  # renglón k: w_k = Σ⁻¹ μ_k
w0 = -0.5 * np.einsum("kj,kj->k", W, mu_k) + np.log(pi_k)             # w_k0


def delta_lda(Xn):
    """Funciones discriminantes δ_k(x), una columna por clase."""
    return Xn @ W.T + w0


def posterior(delta):
    """p(y = k | x) = softmax de las δ_k."""
    return np.exp(delta - logsumexp(delta, axis=1, keepdims=True))


pred_mano = np.array(ESPECIES)[delta_lda(X).argmax(axis=1)]

lda = LinearDiscriminantAnalysis().fit(X, y)
print("¿mismas predicciones que scikit-learn?", (pred_mano == lda.predict(X)).all())
print("máxima diferencia en p(y | x):         ", np.abs(posterior(delta_lda(X)) - lda.predict_proba(X)).max())
print(f"exactitud en entrenamiento: {(pred_mano == y).mean():.3f}")

# %% [markdown]
# Coinciden en los 342 pingüinos, y las probabilidades también, hasta el error de redondeo. Ahora la figura. Cada elipse es la región que contiene el 95 % de la normal estimada de su clase (para una normal en $d = 2$, $(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$ sigue una $\chi^2$ con 2 grados de libertad, así que la elipse es el contorno donde esa forma cuadrática vale $\chi^2_{2;\,0.95} \approx 5.99$). En LDA las tres elipses son copias trasladadas de la misma.

# %%
CLARO = ListedColormap([tuple(0.72 + 0.28 * np.array(to_rgb(COLOR[e]))) for e in ESPECIES])
x1 = np.linspace(31, 61, 600)
x2 = np.linspace(12.5, 22.5, 400)
M1, M2 = np.meshgrid(x1, x2)
REJILLA = np.c_[M1.ravel(), M2.ravel()]
POS_ETIQUETA = {"Adelie": (34.5, 21.9), "Chinstrap": (55.5, 21.4), "Gentoo": (57.5, 13.3)}


def elipse(ax, mu, S, color, nivel=0.95):
    """Elipse que contiene la fracción `nivel` de una normal bivariada N(mu, S)."""
    val, vec = np.linalg.eigh(S)
    r = np.sqrt(chi2.ppf(nivel, df=2) * val)
    angulo = np.degrees(np.arctan2(vec[1, 1], vec[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * r[1], 2 * r[0], angle=angulo, fill=False, color=color, lw=2.2))


def regiones(ax, clases, titulo, covarianzas, referencia=None):
    """Regiones de decisión, puntos por especie, medias y elipses del 95 %.

    Si se da `referencia` (otras predicciones en la rejilla), sus fronteras se dibujan punteadas.
    """
    indice = np.searchsorted(ESPECIES, clases).reshape(M1.shape)
    ax.contourf(M1, M2, indice, levels=[-0.5, 0.5, 1.5, 2.5], cmap=CLARO)
    ax.contour(M1, M2, indice, levels=[0.5, 1.5], colors=rdp.estilo.TENUE, linewidths=1.2)
    if referencia is not None:
        ref = np.searchsorted(ESPECIES, referencia).reshape(M1.shape)
        ax.contour(M1, M2, ref, levels=[0.5, 1.5], colors=rdp.estilo.TINTA, linewidths=1.2, linestyles="--")
    for k, e in enumerate(ESPECIES):
        m = y == e
        ax.scatter(X[m, 0], X[m, 1], s=18, color=COLOR[e], alpha=0.8, edgecolors="white", linewidths=0.4)
        ax.plot(*mu_k[k], "X", color=rdp.estilo.TINTA, ms=10, mec="white")
        elipse(ax, mu_k[k], covarianzas[k], COLOR[e])
        ax.text(*POS_ETIQUETA[e], e, color=COLOR[e], fontsize=13, fontweight="semibold", ha="center")
    ax.set(xlim=(x1[0], x1[-1]), ylim=(x2[0], x2[-1]), title=titulo,
           xlabel="largo del pico (mm)", ylabel="profundidad del pico (mm)")
    ax.grid(False)


pred_rejilla_lda = np.array(ESPECIES)[delta_lda(REJILLA).argmax(axis=1)]
fig, ax = plt.subplots(figsize=(9, 6.5))
regiones(ax, pred_rejilla_lda, "LDA: misma elipse para las tres especies, fronteras rectas", [Sigma] * K)
plt.show()

# %% [markdown]
# Las fronteras son segmentos de recta, como predijo la derivación, y pasan por donde se tocan las elipses. La elipse común es un compromiso: le queda un poco ancha a Adelie y un poco angosta a Chinstrap, cuyas nubes tienen inclinaciones algo distintas. ¿Y si dejamos que cada especie tenga su propia elipse?

# %% [markdown]
# ## 3. QDA: una covarianza por clase
#
# El análisis discriminante cuadrático cambia un solo supuesto: cada clase tiene su propia $\boldsymbol{\Sigma}_k$, estimada solo con sus datos, $\hat{\boldsymbol{\Sigma}}_k = \frac{1}{N_k - 1}\sum_{i:\,y_i=k}(\mathbf{x}_i - \hat{\boldsymbol\mu}_k)(\mathbf{x}_i - \hat{\boldsymbol\mu}_k)^\top$. Repitiendo la cuenta, ahora $-\tfrac12\mathbf{x}^\top\boldsymbol{\Sigma}_k^{-1}\mathbf{x}$ **sí depende de $k$** y ya no se cancela, ni tampoco $\log|\boldsymbol\Sigma_k|$:
#
# $$
# \delta_k(\mathbf{x}) = -\tfrac12 \log|\boldsymbol{\Sigma}_k| - \tfrac12 (\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) + \log\pi_k.
# $$
#
# Las fronteras $\delta_k = \delta_l$ son ahora **cuádricas**: en el plano, curvas cónicas (elipses, parábolas o hipérbolas).

# %%
Sigma_k = np.array([np.cov(X[y == e], rowvar=False) for e in ESPECIES])   # np.cov divide entre N_k − 1


def delta_qda(Xn):
    cols = []
    for k in range(K):
        D = Xn - mu_k[k]
        maha = np.einsum("ij,jk,ik->i", D, np.linalg.inv(Sigma_k[k]), D)
        cols.append(-0.5 * np.linalg.slogdet(Sigma_k[k])[1] - 0.5 * maha + np.log(pi_k[k]))
    return np.column_stack(cols)


qda = QuadraticDiscriminantAnalysis().fit(X, y)
print("¿mismas predicciones que scikit-learn?",
      (np.array(ESPECIES)[delta_qda(X).argmax(axis=1)] == qda.predict(X)).all())

fig, ax = plt.subplots(figsize=(9, 6.5))
regiones(ax, np.array(ESPECIES)[delta_qda(REJILLA).argmax(axis=1)],
         "QDA: una elipse por especie, fronteras curvas (punteadas: LDA)", Sigma_k,
         referencia=pred_rejilla_lda)
plt.show()

# %% [markdown]
# Cada elipse sigue ahora la forma de su nube: la de Adelie es más redonda, las de Chinstrap y Gentoo más alargadas e inclinadas. Las fronteras se curvan, aunque en esta ventana la curvatura es suave: donde están los pingüinos, QDA y LDA (líneas punteadas) cortan casi en los mismos lugares; las diferencias se notan cerca del punto donde se juntan las tres regiones y en la frontera entre Adelie y Gentoo, donde nunca hay pingüinos. Lejos de los datos, en cambio, las cónicas pueden hacer cosas raras, como que la región de una clase «envuelva» a otra (lo verás si amplías la rejilla).
#
# ### ¿Cuántos parámetros?
#
# Que QDA sea más flexible tiene un precio en parámetros por estimar. Con $d$ variables y $K$ clases:
#
# | | medias | covarianzas | previas | total |
# |---|---|---|---|---|
# | LDA | $Kd$ | $\frac{d(d+1)}{2}$ | $K - 1$ | $Kd + \frac{d(d+1)}{2} + K - 1$ |
# | QDA | $Kd$ | $K\,\frac{d(d+1)}{2}$ | $K - 1$ | $Kd + K\frac{d(d+1)}{2} + K - 1$ |
# | logística multinomial | — | — | — | $(K - 1)(d + 1)$ |
#
# La logística solo modela la frontera; por eso necesita mucho menos. LDA estima más cosas, pero su frontera también queda descrita por $(K-1)(d+1)$ números (las diferencias $\mathbf{w}_k - \mathbf{w}_K$ y $w_{k0} - w_{K0}$): el resto de los parámetros describe cómo son los datos, no dónde cortar. La diferencia importante está en QDA, que crece como $K d^2/2$:

# %%
def parametros(d, K):
    return {"LDA": K * d + d * (d + 1) // 2 + K - 1,
            "QDA": K * d + K * d * (d + 1) // 2 + K - 1,
            "logística": (K - 1) * (d + 1)}

pd.DataFrame({f"d = {d_}, K = {K_}": parametros(d_, K_)
              for d_, K_ in [(2, 3), (4, 3), (50, 3), (784, 10)]})

# %% [markdown]
# Con dos medidas del pico, QDA tiene 17 parámetros para 342 pingüinos: ningún problema. Con imágenes de dígitos de $28 \times 28 = 784$ píxeles y 10 clases, QDA necesitaría más de tres millones, y cada $\hat{\boldsymbol\Sigma}_k$ tendría más parámetros que ejemplos de su clase: sería singular. Es el dilema sesgo–varianza de la Unidad 3 en estado puro: LDA tiene más sesgo (supone covarianzas iguales), QDA más varianza.
#
# ### ¿Cuál clasifica mejor?
#
# Medimos la exactitud con validación cruzada de 5 pliegues estratificados. Para la logística estandarizamos antes (con regularización por omisión, $C = 1$, el escalamiento importa).

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
modelos = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "logística": make_pipeline(StandardScaler(), LogisticRegression()),
}
exactitud_cv = {}
for nombre, modelo in modelos.items():
    s = cross_val_score(modelo, X, y, cv=cv)
    exactitud_cv[nombre] = {"exactitud media": s.mean(), "desv. estándar": s.std(),
                            "parámetros": parametros(d, K)[nombre]}
pd.DataFrame(exactitud_cv).T.round(3)

# %% [markdown]
# Los tres están entre 95 y 96.5 %: LDA queda arriba (0.965), luego QDA (0.959) y la logística (0.950). Las diferencias equivalen a uno o dos pingüinos por pliegue, menos que la desviación estándar entre pliegues, así que no hay un ganador claro. Lo interesante es que el modelo generativo **más restringido** no pierde: si sus supuestos son aproximadamente ciertos (nubes más o menos normales con formas parecidas), usarlos es aprovechar información que la logística tiene que aprender de los datos.

# %% [markdown]
# ## 4. LDA como reducción de dimensión
#
# LDA tiene una segunda vida, que se debe a R. A. Fisher (1936, con los famosos lirios): encontrar las direcciones que mejor **separan** las clases.
#
# Para una dirección $\mathbf{w}$, proyectamos cada punto a $z = \mathbf{w}^\top\mathbf{x}$. Queremos que las medias proyectadas de las clases queden lejos entre sí, pero **medido en unidades de la dispersión dentro de cada clase**. Con las matrices de dispersión entre clases y dentro de las clases,
#
# $$
# \mathbf{S}_B = \sum_k N_k (\boldsymbol\mu_k - \boldsymbol\mu)(\boldsymbol\mu_k - \boldsymbol\mu)^\top, \qquad
# \mathbf{S}_W = \sum_k \sum_{i:\,y_i = k} (\mathbf{x}_i - \boldsymbol\mu_k)(\mathbf{x}_i - \boldsymbol\mu_k)^\top,
# $$
#
# Fisher propuso maximizar el **cociente de Rayleigh**
#
# $$
# J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}.
# $$
#
# Derivando e igualando a cero se llega a $\mathbf{S}_B\mathbf{w} = J\,\mathbf{S}_W\mathbf{w}$: un **problema de valores propios generalizado**. La mejor dirección es el vector propio del mayor valor propio, la segunda la del siguiente, etc. Como $\mathbf{S}_B$ es suma de $K$ matrices de rango 1 con una restricción (las desviaciones ponderadas suman cero), tiene rango a lo más $K - 1$: con tres especies hay **a lo más dos** direcciones útiles. Y no es una idea distinta del clasificador: con normales de covarianza común, clasificar con $\delta_k$ equivale a asignar cada punto a la media más cercana en estas coordenadas.
#
# Compárese con **PCA**, que maximiza $\mathbf{w}^\top\mathbf{S}_T\mathbf{w}$ con $\|\mathbf{w}\| = 1$, donde $\mathbf{S}_T = \mathbf{S}_B + \mathbf{S}_W$ es la dispersión total: busca hacia dónde se **extienden** los datos, sin mirar las etiquetas. Usemos ahora las cuatro medidas, estandarizadas.

# %%
medidas = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
p4 = pinguinos.dropna(subset=medidas).reset_index(drop=True)
Z = StandardScaler().fit_transform(p4[medidas])
y4 = p4["species"].to_numpy()

lda4 = LinearDiscriminantAnalysis(n_components=2).fit(Z, y4)
pca4 = PCA(n_components=2).fit(Z)

# Las direcciones de Fisher, a mano: valores propios generalizados de (S_B, S_W)
media = Z.mean(axis=0)
S_W = sum((Z[y4 == e] - Z[y4 == e].mean(0)).T @ (Z[y4 == e] - Z[y4 == e].mean(0)) for e in ESPECIES)
S_B = sum((y4 == e).sum() * np.outer(Z[y4 == e].mean(0) - media, Z[y4 == e].mean(0) - media) for e in ESPECIES)
valores, vectores = eigh(S_B, S_W)
print("valores propios generalizados (J de cada dirección):", valores[::-1].round(3))


def J(w):
    return (w @ S_B @ w) / (w @ S_W @ w)


print(f"J(1.ª dirección de Fisher) = {J(vectores[:, -1]):.2f}   J(1.er componente de PCA) = {J(pca4.components_[0]):.2f}")
print("fracción de la separación entre clases que capta cada eje de LDA:", lda4.explained_variance_ratio_.round(3))
print("fracción de la varianza total que capta cada eje de PCA:        ", pca4.explained_variance_ratio_.round(3))

# %% [markdown]
# Hay dos valores propios distintos de cero (los otros dos son cero, como predice el rango de $\mathbf{S}_B$). En la dirección de Fisher, la dispersión entre medias es 15 veces la dispersión dentro de las clases; en la primera dirección de PCA, que no buscaba eso, apenas 7. Veámoslo.

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, (titulo, P, ejes, pos_etiqueta) in zip(axs, [
        ("PCA: máxima varianza total (sin etiquetas)", pca4.transform(Z), ("componente principal 1", "componente principal 2"),
         {"Adelie": (-2.4, 1.5), "Chinstrap": (-1.3, 2.4), "Gentoo": (2.9, -1.2)}),
        ("LDA: máxima separación entre especies", lda4.transform(Z), ("discriminante 1", "discriminante 2"),
         {"Adelie": (-6.0, 3.4), "Chinstrap": (0.6, -4.8), "Gentoo": (5.3, 3.6)})]):
    for e in ESPECIES:
        m = y4 == e
        ax.scatter(P[m, 0], P[m, 1], s=18, color=COLOR[e], alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.text(*pos_etiqueta[e], e, color=COLOR[e], fontsize=13, fontweight="semibold", ha="center")
    ax.set(title=titulo, xlabel=ejes[0], ylabel=ejes[1])
plt.show()

# %% [markdown]
# PCA encuentra que lo que más varía entre pingüinos es el **tamaño** (aleta y masa), y eso separa muy bien a Gentoo, que es mucho más grande, pero deja a Adelie y Chinstrap encimados: miden lo mismo de aleta y de masa. LDA, que sí conoce las etiquetas, encuentra una dirección (el discriminante 2) que usa sobre todo el largo del pico, justo lo que distingue a Chinstrap de Adelie, y separa las tres especies casi sin traslape.
#
# La moraleja: la dirección de mayor varianza **no tiene por qué** ser la que separa las clases. PCA es no supervisado (Unidad 2) y no tiene manera de saber qué variación nos interesa. Por otro lado, LDA solo da $K - 1$ dimensiones y se apoya en el supuesto de covarianza común; con muchas variables y pocos datos, $\mathbf{S}_W$ es casi singular y las direcciones de Fisher se sobreajustan.

# %% [markdown]
# # Parte 2. Bayes ingenuo y el spam en SMS
#
# ## 5. Los datos y el problema
#
# **Datos.** En 2011, Tiago Almeida, José María Gómez Hidalgo y Akebo Yamakami reunieron una colección pública de 5,574 mensajes SMS en inglés, etiquetados como **spam** (publicidad no solicitada, concursos falsos, servicios de cobro) o **normales** (*ham*, en la jerga del área). Los spam salieron en su mayoría de **Grumbletext**, un foro británico donde la gente transcribía y denunciaba los SMS basura que recibía; los normales, en su mayoría del **NUS SMS Corpus**, una colección de mensajes que donaron voluntariamente estudiantes de la Universidad Nacional de Singapur. La tabla está en el repositorio de la UCI con licencia CC BY 4.0.
#
# **Problema.** Decidir, a partir del texto, si un mensaje es spam, para mandarlo a una carpeta aparte. Equivocarse no cuesta igual en los dos sentidos: un spam que se cuela molesta; un mensaje normal que se esconde («ya llegué, ábreme») puede costar mucho más. Por eso, además de la exactitud, nos importará la **precisión** de la clase spam.
#
# Usamos la misma partición que en el resto de la unidad: 25 % para prueba, estratificada.

# %%
sms = rdp.datos.cargar("sms_spam")
print(f"{len(sms):,} mensajes; spam: {sms['spam'].mean():.1%}")
pd.set_option("display.max_colwidth", 110)
sms.groupby("spam").sample(3, random_state=SEMILLA)

# %%
entrena, prueba = train_test_split(sms, test_size=0.25, stratify=sms.spam, random_state=2011)
print(f"entrenamiento: {len(entrena):,} mensajes · prueba: {len(prueba):,} mensajes")

# %% [markdown]
# Los ejemplos ya muestran el contraste: el spam habla de premios, números de teléfono, códigos y precios por mensaje («150p»); los normales son abreviados, coloquiales, con faltas de ortografía y mucho *singlish* (el inglés coloquial de Singapur: «lor», «lah»). Esa última diferencia viene de **dónde** se recolectó cada clase, no del spam en sí; tenlo presente cuando veamos qué palabras usa el modelo.

# %% [markdown]
# ## 6. Bolsa de palabras
#
# Un clasificador necesita vectores, no textos. La representación más simple es la **bolsa de palabras** (*bag of words*): se arma un vocabulario con todas las palabras del conjunto de entrenamiento y cada mensaje se convierte en el vector $\mathbf{x} \in \mathbb{N}^{|V|}$ de **conteos**: $x_j$ es cuántas veces aparece la palabra $j$. Se pierde el orden («perro muerde a hombre» y «hombre muerde a perro» dan el mismo vector), pero para detectar spam el orden importa poco.
#
# `CountVectorizer` hace todo: pasa a minúsculas, separa en palabras de dos o más letras o dígitos, y cuenta.

# %%
vectorizador = CountVectorizer().fit(entrena.texto)
X_tr = vectorizador.transform(entrena.texto)
X_te = vectorizador.transform(prueba.texto)
y_tr, y_te = entrena.spam.to_numpy(), prueba.spam.to_numpy()
vocab = vectorizador.get_feature_names_out()

print(f"vocabulario: {len(vocab):,} palabras")
print(f"matriz de entrenamiento: {X_tr.shape[0]:,} × {X_tr.shape[1]:,}, "
      f"con {X_tr.nnz / np.prod(X_tr.shape):.2%} de entradas distintas de cero")

ejemplo = "WINNER!! You have won a prize. Call now to claim your prize"
x_ej = vectorizador.transform([ejemplo])
print("\n", ejemplo)
print({vocab[j]: int(c) for j, c in zip(x_ej.indices, x_ej.data)})

# %% [markdown]
# 7,515 palabras distintas, y cada mensaje usa unas pocas: la matriz es casi toda ceros y se guarda como **matriz dispersa**. En el ejemplo, «prize» cuenta 2 y los signos de puntuación desaparecen.

# %% [markdown]
# ## 7. El modelo multinomial
#
# ### Cómo se genera un mensaje
#
# El modelo generativo es una caricatura del proceso de escribir: para generar un mensaje de la clase $k$, primero se lanza una moneda cargada para elegir la clase (con probabilidad $\pi_k$), y luego se sacan palabras **una por una, de manera independiente**, de un dado de $|V|$ caras con probabilidades $\boldsymbol\theta_k = (\theta_{k1}, \dots, \theta_{k|V|})$, propio de cada clase. El dado del spam tiene la cara «prize» cargada; el de los normales, la cara «lor». La probabilidad de unos conteos $\mathbf{x}$ es entonces la multinomial:
#
# $$
# p(\mathbf{x} \mid y = k) = \frac{\big(\sum_j x_j\big)!}{\prod_j x_j!} \prod_{j=1}^{|V|} \theta_{kj}^{x_j}.
# $$
#
# El coeficiente del frente no depende de $k$, así que al aplicar Bayes se cancela.
#
# ### Por qué es «ingenuo»
#
# El supuesto de que cada palabra se saca independientemente de las demás, **dada la clase**, es la **independencia condicional**, y es evidentemente falso: después de «call» es mucho más probable «now» que «lor»; «free» y «prize» aparecen juntas. Bayes ingenuo (*naive Bayes*) lo asume de todos modos, porque sin él tendríamos que estimar probabilidades de combinaciones de palabras, que son astronómicamente muchas. Lo sorprendente es lo bien que funciona a pesar del supuesto: para **clasificar** basta con que la clase correcta quede con mayor probabilidad, aunque las probabilidades mismas estén muy mal (suelen ser exageradas, demasiado cerca de 0 o de 1, porque cuenta como evidencia independiente lo que en realidad es la misma evidencia repetida).
#
# ### Estimación y el problema de los ceros
#
# El estimador de máxima verosimilitud de $\theta_{kj}$ es la fracción de las palabras de la clase $k$ que son la palabra $j$: $\hat\theta_{kj} = N_{kj} / N_k$, con $N_{kj}$ el total de veces que aparece la palabra $j$ en los mensajes de la clase $k$ y $N_k = \sum_j N_{kj}$. Veamos qué problema tiene.

# %%
N_kj = np.vstack([np.asarray(X_tr[y_tr == k].sum(axis=0)).ravel() for k in (0, 1)])  # renglón 0: normal, 1: spam
nunca_en_spam = (N_kj[1] == 0)
print(f"palabras del vocabulario que nunca aparecen en spam:   {nunca_en_spam.sum():,} de {len(vocab):,}")
print(f"palabras que nunca aparecen en mensajes normales:      {(N_kj[0] == 0).sum():,}")
j = list(vocab).index("lor")
print(f"«lor»: {N_kj[0, j]} veces en normales, {N_kj[1, j]} en spam  →  θ̂(lor | spam) = 0 sin suavizar")

# %% [markdown]
# Dos tercios del vocabulario nunca aparecen en spam, simplemente porque hay pocos spam en entrenamiento (560 mensajes). Con $\hat\theta_{\text{spam},\,\text{lor}} = 0$, **cualquier** mensaje que contenga «lor» tendría $p(\mathbf{x} \mid \text{spam}) = 0$, sin importar que además diga «WINNER, claim your prize, call 0800…». Una sola palabra vetaría todas las demás. Peor: si un mensaje tiene una palabra que nunca salió en spam y otra que nunca salió en normales, las dos clases tendrían probabilidad cero.
#
# El **suavizado de Laplace** (o aditivo) suma $\alpha > 0$ a cada conteo, como si hubiéramos visto cada palabra $\alpha$ veces más en cada clase:
#
# $$
# \hat\theta_{kj} = \frac{N_{kj} + \alpha}{N_k + \alpha |V|}.
# $$
#
# Con $\alpha = 1$ es la regla de sucesión de Laplace. Bayesianamente, es la media posterior de $\boldsymbol\theta_k$ con una previa de Dirichlet; intuitivamente, es encoger las probabilidades estimadas hacia la uniforme, más cuanto menos datos tiene la clase. Es regularización.
#
# ### Todo en logaritmos
#
# Un mensaje con 30 palabras tiene un producto de 30 probabilidades del orden de $10^{-4}$: unos $10^{-120}$, cerca de lo que un `float64` puede representar, y con textos más largos se vuelve exactamente cero (*underflow*). Por eso todo se calcula en logaritmos, donde los productos son sumas:
#
# $$
# \log\big(\pi_k\, p(\mathbf{x} \mid y = k)\big) = \log\pi_k + \sum_j x_j \log\theta_{kj} + \text{cte}.
# $$
#
# Para pasar a probabilidades se normaliza con *log-sum-exp*: $\log p(y = k \mid \mathbf{x}) = a_k - \log\sum_l e^{a_l}$, que se evalúa restando primero el máximo para no desbordar.

# %%
nb = MultinomialNB(alpha=1.0).fit(X_tr, y_tr)

# 1) los parámetros de scikit-learn son las fórmulas de arriba
alpha = 1.0
log_theta = np.log((N_kj + alpha) / (N_kj.sum(axis=1, keepdims=True) + alpha * len(vocab)))
log_pi = np.log(np.bincount(y_tr) / len(y_tr))
print("¿log θ̂ igual al de scikit-learn?", np.allclose(log_theta, nb.feature_log_prob_),
      "  ¿log π̂ igual?", np.allclose(log_pi, nb.class_log_prior_))

# 2) el log-posterior de un mensaje de prueba, a mano
i = 7
x = X_te[i].toarray().ravel()
print(f"\nmensaje: «{prueba.texto.iloc[i]}»  (spam real = {y_te[i]})")
print("palabras:", {vocab[j]: int(x[j]) for j in np.flatnonzero(x)})
a = log_pi + log_theta @ x                    # log π_k + Σ_j x_j log θ_kj, para k = normal, spam
log_post = a - logsumexp(a)
print(f"a_normal = {a[0]:.2f}   a_spam = {a[1]:.2f}")
print("log P(y | x) a mano:       ", log_post.round(6))
print("log P(y | x) scikit-learn: ", nb.predict_log_proba(X_te[i]).ravel().round(6))
print(f"P(spam | x) = {np.exp(log_post[1]):.6f}")
print(f"\nel producto directo de probabilidades sería  π_spam·Π θ^x = {np.exp(a[1]):.3g}")

# %% [markdown]
# Las cuentas a mano reproducen a scikit-learn. El mensaje (tonos de celular del equipo de Inglaterra, a £1.50) es spam sin discusión: $a_{\text{spam}}$ le gana a $a_{\text{normal}}$ por 51.5, así que la posterior de «normal» es $e^{-51.5} \approx 10^{-22}$. Fíjate en el tamaño de $e^{a_k}$ para un SMS de 28 palabras: ya es del orden de $10^{-85}$. Un `float64` no baja de $10^{-308}$; con un correo de 150 palabras el producto directo daría cero en las dos clases y la comparación sería imposible. En logaritmos no hay problema.

# %% [markdown]
# ## 8. Qué palabras delatan al spam
#
# Cada palabra contribuye a la decisión con $x_j$ veces
#
# $$
# w_j = \log\hat\theta_{\text{spam},\,j} - \log\hat\theta_{\text{normal},\,j} = \log\frac{\hat p(\text{palabra } j \mid \text{spam})}{\hat p(\text{palabra } j \mid \text{normal})}.
# $$
#
# Si $w_j > 0$, verla empuja hacia spam; si $w_j < 0$, hacia normal. Ordenemos el vocabulario por $w_j$.

# %%
w_nb = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
orden = np.argsort(w_nb)
normales, spam = orden[:12], orden[-12:]

fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, idx, color, titulo in [(axs[0], normales[::-1], AZUL, "Delatan un mensaje normal"),
                               (axs[1], spam, ROJO, "Delatan spam")]:
    ax.barh(np.arange(12), w_nb[idx], color=color)
    ax.set_yticks(np.arange(12), vocab[idx], fontsize=12)
    ax.set(title=titulo, xlabel=r"$\log \hat p(w \mid \mathrm{spam}) - \log \hat p(w \mid \mathrm{normal})$")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
axs[0].invert_xaxis()
axs[0].yaxis.tick_right()
plt.tight_layout()
plt.show()

# %% [markdown]
# Del lado del spam está todo el repertorio: «claim», «prize», «guaranteed», «awarded», «150p» y «150ppm» (el costo de los servicios de mensajes de cobro británicos: 150 peniques por mensaje o por minuto), «tone» (tonos para el celular), «18» (servicios «solo para mayores de 18»), «uk». Una palabra como «claim» es unas $e^{5.4} \approx 230$ veces más probable en spam que en un mensaje normal.
#
# Del lado normal hay pronombres («he», «she»), verbos de conversación («come», «said», «ask», «doing») y *singlish* («lor», «da»). Pero las dos primeras son «gt» y «lt», que no son palabras: vienen de `&lt;#&gt;`, una marca con la que en parte de los mensajes normales se sustituyeron números al recolectarlos. Es un **artefacto de la recolección**, como la etiqueta de un hospital en una radiografía: le sirve al modelo en esta tabla, pero no le serviría con mensajes reales, donde nadie escribe `&lt;#&gt;`. Lo mismo vale, en menor grado, para el *singlish*: un spam escrito en Singapur no se delataría por la falta de «lor». Mirar los pesos de un modelo lineal es una de las mejores maneras de encontrar estos atajos.

# %% [markdown]
# ## 9. ¿Qué tan bien clasifica?

# %%
def metricas(modelo):
    pred = modelo.predict(X_te)
    return {"exactitud": accuracy_score(y_te, pred),
            "precisión (spam)": precision_score(y_te, pred),
            "exhaustividad (spam)": recall_score(y_te, pred)}

logistica = LogisticRegression(C=10, max_iter=3000).fit(X_tr, y_tr)
tabla = pd.DataFrame({"Bayes ingenuo (α = 1)": metricas(nb), "logística (C = 10)": metricas(logistica)})
tabla.round(4)

# %%
M = confusion_matrix(y_te, nb.predict(X_te))
fig, ax = plt.subplots(figsize=(5.8, 5))
ax.imshow(M, cmap=rdp.estilo.SECUENCIAL, vmin=0, vmax=M.max() * 1.6)
for r in range(2):
    for c in range(2):
        ax.text(c, r, f"{M[r, c]:,}", ha="center", va="center", fontsize=16,
                color="white" if M[r, c] > M.max() / 2 else rdp.estilo.TINTA)
ax.set_xticks([0, 1], ["normal", "spam"])
ax.set_yticks([0, 1], ["normal", "spam"])
ax.set(xlabel="predicción", ylabel="clase real", title="Bayes ingenuo en prueba (1,394 mensajes)")
ax.grid(False)
plt.show()

# %% [markdown]
# Bayes ingenuo acierta en 98.4 % de los mensajes de prueba. De los 187 spam detecta 171 (exhaustividad 0.914) y se equivoca con 6 mensajes normales (precisión 0.966). La logística, con el mismo vocabulario, tiene exactitud algo menor (0.980) pero precisión casi perfecta (0.994): esconde un solo mensaje normal, a cambio de dejar pasar más spam. Cuál es mejor depende de cuánto cuesta cada error; con un umbral distinto de 0.5, cualquiera de los dos podría moverse a lo largo de esa disyuntiva (Unidad 3).
#
# ### El efecto de $\alpha$
#
# $\alpha$ es un hiperparámetro: lo elegimos con validación cruzada en el conjunto de entrenamiento. El vectorizador va dentro de la tubería, para que en cada pliegue el vocabulario se construya solo con los mensajes de entrenamiento de ese pliegue.

# %%
alfas = np.logspace(-2, 1, 13)
cv_sms = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
exact_alfa = np.array([cross_val_score(make_pipeline(CountVectorizer(), MultinomialNB(alpha=a)),
                                       entrena.texto, y_tr, cv=cv_sms).mean() for a in alfas])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(alfas, exact_alfa, "o-", color=AZUL, mec="white")
mejor = alfas[exact_alfa.argmax()]
ax.axvline(mejor, color=GRIS, ls="--", lw=1.3)
ax.text(mejor * 1.1, exact_alfa.min() + 0.002, f"mejor α ≈ {mejor:.2g}", color=GRIS)
ax.set(xscale="log", title="Bayes ingenuo: exactitud por validación cruzada según α",
       xlabel="α (suavizado de Laplace, escala logarítmica)", ylabel="exactitud (5 pliegues)")
plt.show()
pd.Series(exact_alfa, index=alfas.round(3), name="exactitud CV").round(4).to_frame().T

# %% [markdown]
# La curva es casi plana entre 0.01 y 0.5 (exactitud entre 0.985 y 0.987) y cae a partir de $\alpha \approx 1$: con $\alpha = 1$ es 0.984 y con $\alpha = 10$ baja a 0.953. Con $\alpha$ grande, el suavizado aplasta las diferencias reales entre clases, sobre todo en la clase spam, que tiene pocas palabras: con $\alpha = 10$ es como si hubiéramos visto 75,000 palabras ficticias repartidas por igual, varias veces más que todas las palabras reales del spam de entrenamiento. Es sesgo: la regularización es excesiva.
#
# Del lado de $\alpha$ chico no se ve el castigo que uno esperaría, y vale la pena entender por qué. Lo que destruía al clasificador era $\hat\theta = 0$ exactamente, que veta a una clase; con cualquier $\alpha > 0$ eso ya no pasa, y una palabra nunca vista en spam solo aporta un $w_j$ más negativo (alrededor de $\log \alpha$). Bajar $\alpha$ de 1 a 0.03 hace más confiado al modelo, pero aquí esa confianza casi siempre apunta en la dirección correcta. La diferencia entre $\alpha = 1$ (el valor por omisión, que usamos en el resto de la libreta) y el mejor $\alpha$ es de tres décimas de punto, del tamaño de unos cuantos mensajes por pliegue.

# %% [markdown]
# ## 10. Bayes ingenuo es un clasificador lineal
#
# Escribamos el logaritmo del cociente de las posteriores (los **log-momios** de spam):
#
# $$
# \log\frac{P(\text{spam} \mid \mathbf{x})}{P(\text{normal} \mid \mathbf{x})}
# = \underbrace{\log\frac{\hat\pi_{\text{spam}}}{\hat\pi_{\text{normal}}}}_{b}
# + \sum_{j=1}^{|V|} x_j \underbrace{\log\frac{\hat\theta_{\text{spam},\,j}}{\hat\theta_{\text{normal},\,j}}}_{w_j}
# = b + \mathbf{w}^\top\mathbf{x}.
# $$
#
# El denominador de Bayes y el coeficiente multinomial se cancelan en el cociente. Lo que queda es **lineal en los conteos**: exactamente la forma de la regresión logística, $\log\frac{P(y=1\mid\mathbf{x})}{P(y=0\mid\mathbf{x})} = w_0 + \mathbf{w}^\top\mathbf{x}$, con $P(\text{spam} \mid \mathbf{x}) = \sigma(b + \mathbf{w}^\top\mathbf{x})$. (Aquí usamos $y \in \{0, 1\}$, 1 = spam, como scikit-learn.) Verifiquémoslo:

# %%
b_nb = nb.class_log_prior_[1] - nb.class_log_prior_[0]
logmomios_lineal = b_nb + X_te @ w_nb
logmomios_nb = nb.predict_log_proba(X_te) @ np.array([-1, 1])
print("máxima diferencia:", np.abs(logmomios_lineal - logmomios_nb).max())

# %% [markdown]
# Lo mismo pasa con LDA: sus log-momios entre dos clases son lineales en $\mathbf{x}$. Entonces, ¿en qué se distinguen de la logística? **En cómo se estiman los pesos**:
#
# - **Bayes ingenuo** calcula cada $w_j$ **por separado**, con un cociente de frecuencias de la palabra $j$, sin mirar a las demás palabras. Es instantáneo y no necesita optimizar nada, pero si dos palabras siempre van juntas («free» y «entry»), cuenta la misma evidencia dos veces.
# - **La logística** elige todos los $w_j$ **a la vez**, maximizando la verosimilitud condicional $\prod_i P(y_i \mid \mathbf{x}_i)$. Si dos palabras son redundantes, el ajuste conjunto se lo reparte entre las dos. No supone nada sobre cómo se distribuyen los $\mathbf{x}$.
#
# Misma familia de fronteras, dos criterios distintos para escoger una. Comparemos los pesos de los dos modelos:

# %%
w_log = logistica.coef_.ravel()
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.scatter(w_nb, w_log, s=10, color=GRIS, alpha=0.35, edgecolors="none")
for palabra in ("lt", "he", "she", "lor", "ok", "prize", "150p", "18", "guaranteed", "free", "txt", "call"):
    j = list(vocab).index(palabra)
    ax.scatter(w_nb[j], w_log[j], s=28, color=ROJO if w_nb[j] > 0 else AZUL, edgecolors="white")
    ax.annotate(vocab[j], (w_nb[j], w_log[j]), xytext=(4, 3), textcoords="offset points", fontsize=11,
                color=rdp.estilo.TINTA)
ax.axhline(0, color=rdp.estilo.TENUE, lw=0.8)
ax.axvline(0, color=rdp.estilo.TENUE, lw=0.8)
corr = np.corrcoef(w_nb, w_log)[0, 1]
ax.set(title=f"Pesos por palabra: Bayes ingenuo contra logística (correlación {corr:.2f})",
       xlabel=r"Bayes ingenuo: $w_j = \log\hat\theta_{\mathrm{spam},j} - \log\hat\theta_{\mathrm{normal},j}$",
       ylabel=r"logística: $w_j$ ajustado")
plt.show()

# %% [markdown]
# Los signos coinciden casi siempre: las dos maneras están de acuerdo en qué palabras apuntan hacia dónde. Pero las magnitudes difieren mucho. Bayes ingenuo le da pesos grandes a palabras raras que solo aparecieron en una clase (la nube de puntos a la izquierda y a la derecha, en $\pm 2$ a $\pm 4$), aunque aparezcan en pocos mensajes; la logística, regularizada y ajustando todo junto, deja muchas de esas palabras cerca de cero y concentra el peso en palabras frecuentes y bien informativas como «txt» o «call».

# %% [markdown]
# ## 11. Generativo contra discriminativo: las curvas de aprendizaje
#
# Andrew Ng y Michael Jordan compararon justamente este par, Bayes ingenuo contra regresión logística, en un artículo muy citado (*On discriminative vs. generative classifiers*, NIPS 2001). Su resultado teórico, simplificando:
#
# - El modelo **discriminativo** tiene menor (o igual) error **asintótico**: con datos infinitos, la logística encuentra la mejor frontera lineal, mientras que Bayes ingenuo converge a la frontera que dictan sus supuestos, que puede no ser la mejor si los supuestos son falsos.
# - Pero el **generativo** se acerca a su error asintótico **mucho más rápido**: necesita del orden de $\log d$ ejemplos, contra del orden de $d$ para la logística, porque estima cada parámetro por separado con un promedio.
#
# La predicción es que con pocos datos gana Bayes ingenuo y, con suficientes, las curvas se **cruzan** y gana la logística. Pongámoslo a prueba: entrenamos ambos con submuestras aleatorias de $n$ mensajes de entrenamiento (reajustando el vocabulario a cada submuestra, como sería en la realidad), 20 repeticiones por tamaño, y medimos el error en el mismo conjunto de prueba. Las submuestras que por azar no traen ningún spam (pasa con $n = 20$) se descartan.

# %%
tamanos = [20, 50, 100, 200, 500, 1000, 2000, 4180]
rng = np.random.default_rng(0)
curvas = {"Bayes ingenuo": [], "logística": []}
for n in tamanos:
    err_nb, err_log = [], []
    for r in range(20 if n < len(entrena) else 1):   # con todo el entrenamiento no hay azar: una vez
        idx = rng.choice(len(entrena), n, replace=False) if n < len(entrena) else np.arange(len(entrena))
        sub = entrena.iloc[idx]
        if sub.spam.nunique() < 2:
            continue
        vec = CountVectorizer().fit(sub.texto)
        A, B = vec.transform(sub.texto), vec.transform(prueba.texto)
        err_nb.append(1 - accuracy_score(y_te, MultinomialNB().fit(A, sub.spam).predict(B)))
        err_log.append(1 - accuracy_score(y_te, LogisticRegression(C=10, max_iter=3000).fit(A, sub.spam).predict(B)))
    curvas["Bayes ingenuo"].append(np.mean(err_nb))
    curvas["logística"].append(np.mean(err_log))

pd.DataFrame(curvas, index=pd.Index(tamanos, name="n")).round(4)

# %%
fig, ax = plt.subplots(figsize=(10, 5.8))
for nombre, color in [("Bayes ingenuo", ROJO), ("logística", AZUL)]:
    ax.plot(tamanos, curvas[nombre], "o-", color=color, mec="white")
ax.axhline(sms.spam.mean(), color=GRIS, ls=":", lw=1.5)
ax.text(25, sms.spam.mean() + 0.004, "decir siempre «normal»", color=GRIS)
ax.set_xscale("log")
ax.set_xticks(tamanos, [f"{n:,}" for n in tamanos])
ax.minorticks_off()
ax.set(xlim=(15, 9500), ylim=(0, 0.16), title="Error de prueba según el tamaño de la muestra de entrenamiento",
       xlabel="n, mensajes de entrenamiento (escala logarítmica)", ylabel="error de prueba (promedio de 20 submuestras)")
rdp.estilo.etiquetar(ax, 4900, {"Bayes ingenuo": curvas["Bayes ingenuo"][-1], "logística": curvas["logística"][-1]},
                     separacion=0.012)
plt.show()

# %% [markdown]
# La primera parte de la predicción se cumple con claridad: con pocos datos Bayes ingenuo es mucho mejor. Con $n = 20$ mensajes su error es 0.109 contra 0.132 de la logística (que apenas le gana a decir siempre «normal», 0.134); con $n = 100$, 0.062 contra 0.100. A Bayes ingenuo le bastan unos pocos ejemplos de spam para saber que «prize» y «claim» son sospechosas; la logística tiene que ajustar miles de pesos a la vez con pocos datos.
#
# La segunda parte, el cruce, **no la vemos**: Bayes ingenuo gana en todo el rango, y con los 4,180 mensajes de entrenamiento sigue adelante (0.016 contra 0.020), aunque la distancia se achica. Hay que decirlo con honestidad: el teorema de Ng y Jordan habla de qué pasa **cuando $n$ crece lo suficiente**, y con estos datos no llegamos ahí; quizá con decenas de miles de mensajes la logística alcanzaría y rebasaría a Bayes ingenuo, o quizá no, si aquí el sesgo del supuesto ingenuo es pequeño para clasificar. Ellos mismos encontraron, en varios conjuntos de datos, curvas que no se cruzan. También importan detalles que el teorema idealiza: el vocabulario $d$ crece con $n$, y la logística que usamos está regularizada ($C = 10$). Lo que sí es robusto es la lección práctica: **con pocos datos, un buen modelo generativo es difícil de vencer**; con muchos, la ventaja se reduce y vale la pena probar el discriminativo.

# %% [markdown]
# ## 12. Cierre
#
# | | qué modela | frontera | cómo se estima | conviene cuando |
# |---|---|---|---|---|
# | LDA | $p(\mathbf{x}\mid y)$ normal, $\boldsymbol\Sigma$ común | lineal | medias, covarianza agrupada, proporciones | nubes con forma parecida; pocos datos |
# | QDA | $p(\mathbf{x}\mid y)$ normal, $\boldsymbol\Sigma_k$ propia | cuadrática | lo mismo, una covarianza por clase | formas distintas y muchos datos por clase ($\gg d^2$) |
# | Bayes ingenuo multinomial | conteos independientes dada la clase | lineal en los conteos | frecuencias con suavizado de Laplace | texto, $d$ enorme, pocos datos |
# | regresión logística | $p(y \mid \mathbf{x})$ directamente | lineal | máxima verosimilitud condicional (iterativo) | muchos datos; supuestos generativos dudosos |
#
# Las tres ideas que hay que llevarse:
#
# 1. Un modelo generativo clasifica con Bayes: $\arg\max_k\, p(\mathbf{x}\mid y = k)\,\pi_k$. Si los log-momios resultan lineales en $\mathbf{x}$ (LDA, Bayes ingenuo), el clasificador es lineal, de la misma familia que la logística.
# 2. La diferencia con la logística no está en la forma de la frontera sino en **cómo se escoge**: el generativo la deduce de un modelo de los datos (rápido, eficiente con pocos datos, sesgado si el modelo es falso); el discriminativo la optimiza directamente.
# 3. Los supuestos «ingenuos» o falsos pueden dar buenos clasificadores: para decidir basta con ordenar bien las clases, aunque las probabilidades estén mal calibradas.

# %% [markdown]
# ## 13. Autoevaluación

# %%
rdp.quiz.mostrar("u04", ids=["u04-p10", "u04-p11", "u04-p12"])

# %% [markdown]
# ## 14. Ejercicios
#
# 1. **Derivación.** Para dos clases con LDA, demuestra que $P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + w_0)$ y escribe $\mathbf{w}$ y $w_0$ en términos de $\boldsymbol\mu_0, \boldsymbol\mu_1, \boldsymbol\Sigma, \pi_0, \pi_1$. ¿Qué le pasa a la frontera si $\pi_1 \to 0$? Verifícalo numéricamente con Adelie contra Chinstrap.
# 2. **Previas.** `LinearDiscriminantAnalysis` acepta el argumento `priors`. Clasifica los pingüinos con previas iguales $(1/3, 1/3, 1/3)$ y dibuja las nuevas regiones encima de las originales. ¿Qué frontera se mueve más y hacia dónde? ¿Cuántos pingüinos cambian de predicción?
# 3. **QDA con muchas variables.** Agrega a las dos medidas del pico $m$ columnas de ruido normal estándar, con $m \in \{0, 5, 10, 20, 40\}$, y compara la exactitud por validación cruzada de LDA y QDA. Explica el resultado con el conteo de parámetros. ¿Qué hace el argumento `reg_param` de QDA?
# 4. **Bayes ingenuo de Bernoulli.** En lugar de contar, registra solo si cada palabra aparece o no (`CountVectorizer(binary=True)` y `BernoulliNB`). Escribe $p(\mathbf{x}\mid y)$ para este modelo y muestra que los log-momios también son lineales en $\mathbf{x}$, pero que ahora las palabras **ausentes** también aportan. Compara sus métricas de prueba con las del modelo multinomial.
# 5. **El artefacto.** Quita de los mensajes la marca `&lt;#&gt;` (y `&lt;`, `&gt;` en general) antes de vectorizar, reentrena Bayes ingenuo y compara las métricas de prueba y la lista de palabras más indicativas. ¿Cuánto dependía el modelo de ese atajo?
