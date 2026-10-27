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
# # Mezclas gaussianas y el algoritmo EM
#
# **Unidad 2 · Ejemplo 2**
#
# k-medias reparte los datos en grupos, pero tiene una idea muy rígida de lo que es un grupo: una nube redonda alrededor de un centro. Muchos grupos reales no son así; son alargados, inclinados o de tamaños muy distintos. En esta libreta vamos a:
#
# 1. ver con datos reales un caso en que k-medias se equivoca por esa razón,
# 2. construir un modelo más flexible, la **mezcla de gaussianas**, que describe cada grupo con una media y una matriz de covarianza,
# 3. programar desde cero el algoritmo que lo ajusta, **EM** (*expectation–maximization*), y compararlo con scikit-learn,
# 4. aprovechar lo que da un modelo probabilístico y no da k-medias: asignaciones con incertidumbre, un criterio para elegir el número de grupos y una densidad para detectar casos raros.

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
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

import rdp

rdp.estilo.usar()
SEMILLA = 2011

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Datos.** Entre 2007 y 2009, la ecóloga Kristen Gorman y el programa de investigación ecológica de largo plazo de la estación Palmer (*Palmer Station LTER*), en la Antártida, midieron a los pingüinos que anidaban en tres islas del archipiélago Palmer. Querían entender cómo se relacionan el tamaño y la forma del cuerpo de cada especie con su alimentación, y cómo los afecta el cambio en el hielo marino. Para cada pingüino registraron la especie, la isla, el sexo y cuatro medidas: el largo y la profundidad (el alto) del pico, el largo de la aleta y la masa corporal.
#
# Allison Horst, Alison Hill y Kristen Gorman publicaron la tabla en 2020 en el paquete de R `palmerpenguins`, con licencia CC0 (dominio público), y desde entonces se usa mucho en la enseñanza. Son 344 pingüinos de tres especies: **Adelia** (*Adelie*), **barbijo** (*Chinstrap*) y **papúa** (*Gentoo*). Usaremos los nombres en inglés, que son los de la tabla.

# %%
pinguinos = rdp.datos.cargar("pinguinos")
print(pinguinos.shape)
pinguinos.head()

# %%
pinguinos.isna().sum()

# %% [markdown]
# Dos pingüinos no tienen ninguna medida; los quitamos. A otros once les falta el sexo, pero eso no nos estorba porque no lo usaremos para agrupar.

# %%
medidas = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
datos = pinguinos.dropna(subset=medidas).reset_index(drop=True)
datos[medidas].describe().round(1)

# %% [markdown]
# **Problema.** Imaginemos que somos los primeros en llegar a la isla: tenemos las medidas de cientos de pingüinos, pero **no sabemos cuántas especies hay** ni a cuál pertenece cada uno. ¿Podemos descubrir grupos naturales solo con las medidas?
#
# Por eso la columna `species` la vamos a esconder: **solo la usaremos al final de cada paso, para evaluar**. En un problema real de agrupamiento no tendríamos esa respuesta (si la tuviéramos, sería un problema de clasificación). Aquí la tenemos y es una suerte, porque nos deja medir qué tan bien funciona cada método.
#
# Empecemos con dos medidas, el largo y la profundidad del pico, que se pueden dibujar en el plano. Antes de ver la respuesta: **¿cuántos grupos ves?**

# %%
largo, profundidad = datos["bill_length_mm"], datos["bill_depth_mm"]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.scatter(largo, profundidad, s=22, color=rdp.estilo.GRIS, alpha=0.7, edgecolors="none")
ax.set(title="344 pingüinos, sin saber de qué especie es cada uno",
       xlabel="largo del pico (mm)", ylabel="profundidad del pico (mm)")
plt.show()

# %% [markdown]
# Se adivinan al menos dos nubes: una abajo (picos poco profundos) y otra arriba. Con más cuidado, la nube de arriba parece tener dos partes, una a la izquierda y otra a la derecha. Ahora la respuesta:

# %%
ESPECIES = ["Adelie", "Chinstrap", "Gentoo"]
COLOR_ESPECIE = {"Adelie": rdp.estilo.AZUL, "Chinstrap": rdp.estilo.ROJO, "Gentoo": rdp.estilo.DORADO_OSC}


ETIQUETAS_PICO = {"Adelie": (37.5, 21.9), "Chinstrap": (52, 21.2), "Gentoo": (55, 13.8)}


def por_especie(ax, x, y, posiciones, s=22, alpha=0.75):
    """Dispersión coloreada por especie, con el nombre escrito en `posiciones[especie]`."""
    for esp in ESPECIES:
        m = datos["species"] == esp
        ax.scatter(x[m], y[m], s=s, color=COLOR_ESPECIE[esp], alpha=alpha, edgecolors="none")
        ax.text(*posiciones[esp], esp, ha="center", va="center",
                fontsize=12.5, fontweight="semibold", color=COLOR_ESPECIE[esp])


fig, ax = plt.subplots(figsize=(7.5, 5.5))
por_especie(ax, largo, profundidad, ETIQUETAS_PICO)
ax.set(title="Las mismas medidas, coloreadas por especie",
       xlabel="largo del pico (mm)", ylabel="profundidad del pico (mm)")
plt.show()

# %% [markdown]
# Hay tres especies, y cada una forma una nube **alargada e inclinada**: dentro de una especie, los pingüinos de pico más largo también lo tienen más profundo (los más grandes son más grandes en todo). Las nubes de Adelie y Chinstrap se tocan; un algoritmo tendrá que decidir dónde termina una y empieza la otra.

# %% [markdown]
# ## 2. k-medias se equivoca, y por qué
#
# Probemos k-medias con $k = 3$ sobre las dos medidas estandarizadas (como vimos en la libreta anterior, k-medias usa distancias, así que las escalas importan).

# %%
X = StandardScaler().fit_transform(datos[["bill_length_mm", "bill_depth_mm"]])
especie = datos["species"]

kmedias = KMeans(n_clusters=3, n_init=10, random_state=SEMILLA).fit(X)
ari_kmedias = adjusted_rand_score(especie, kmedias.labels_)
print(f"ARI de k-medias contra la especie: {ari_kmedias:.3f}")

# %% [markdown]
# El **índice de Rand ajustado** (ARI) compara dos particiones sin importar cómo se llamen los grupos: vale 1 si coinciden exactamente y alrededor de 0 si coinciden tanto como lo harían por azar. 0.83 no está mal, pero veamos dónde están los errores.
#
# Los números de los grupos son arbitrarios (k-medias no sabe nada de especies). **Solo para poder comparar las gráficas**, renumeramos cada grupo según la especie más frecuente en él, de modo que el grupo «de los Adelie» salga en azul, el «de los Chinstrap» en rojo y el «de los Gentoo» en dorado. Así, un punto cuyo color no coincide con el de la gráfica anterior es un error.

# %%
CICLO = rdp.estilo.CICLO


def alinear(etiquetas):
    """Renumera los grupos según la especie mayoritaria en cada uno (solo para comparar colores)."""
    mayoritaria = pd.crosstab(etiquetas, especie).idxmax(axis=1)
    orden = mayoritaria.map(ESPECIES.index)
    return orden.loc[etiquetas].to_numpy(), orden.to_numpy()


def dibujar_grupos(ax, X, etiquetas, titulo):
    for g in np.unique(etiquetas):
        m = etiquetas == g
        ax.scatter(X[m, 0], X[m, 1], s=22, color=CICLO[g], alpha=0.75, edgecolors="none")
    ax.set(title=titulo, xlabel="largo del pico (estandarizado)", ylabel="profundidad del pico (estandarizada)")


def tabla_cruzada(etiquetas):
    return pd.crosstab(especie, etiquetas, rownames=["especie"], colnames=["grupo"])


especie_num = especie.map(ESPECIES.index).to_numpy()
grupo_km, _ = alinear(kmedias.labels_)
errores_km = grupo_km != especie_num

fig, ax = plt.subplots(figsize=(7.5, 5.5))
dibujar_grupos(ax, X, grupo_km, f"Los tres grupos de k-medias ({errores_km.sum()} pingüinos mal agrupados)")
ax.scatter(X[errores_km, 0], X[errores_km, 1], s=90, facecolors="none", edgecolors=rdp.estilo.TINTA, linewidths=1.2)
ax.scatter(*kmedias.cluster_centers_.T, marker="X", s=220, color=rdp.estilo.TINTA, edgecolors="white", linewidths=1.5)
plt.show()
tabla_cruzada(grupo_km)

# %% [markdown]
# Los errores (encerrados en círculos) no están repartidos al azar: forman una franja en el centro, donde se juntan las tres nubes. Hay tres tipos:
#
# - los Gentoo de pico **más largo** (la punta superior derecha de su nube, alrededor de $y \approx 0$) quedaron con el grupo rojo de los Chinstrap;
# - varios Chinstrap de pico **más corto** quedaron con los Adelie o con los Gentoo;
# - unos cuantos Adelie de pico largo quedaron con los Chinstrap.
#
# La razón está en cómo decide k-medias: cada punto va al **centro más cercano**. Las fronteras entre grupos son entonces las mediatrices entre los centros (regiones de Voronoi), rectas que no saben nada de la forma de cada nube. Eso equivale a suponer que los grupos son **esféricos y de tamaño parecido**. Si una nube es alargada, sus puntos de los extremos quedan más cerca del centro de otra nube que del propio, y k-medias los manda al grupo equivocado.
#
# Necesitamos un modelo en el que cada grupo tenga su propia forma.

# %% [markdown]
# ## 3. La idea: los datos como una mezcla
#
# Pensemos cómo se **generaron** los datos. Para producir un pingüino, la naturaleza:
#
# 1. elige una especie $k$, con probabilidad $\pi_k$ (hay más Adelie que Chinstrap);
# 2. le da medidas al azar según la distribución de esa especie, que suponemos normal multivariada: $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.
#
# La media $\boldsymbol{\mu}_k$ dice dónde está la nube; la matriz de covarianza $\boldsymbol{\Sigma}_k$, qué forma tiene: qué tan ancha es en cada dirección y cuánto está inclinada. Si no sabemos de qué especie viene un punto, su densidad es el promedio ponderado de las $K$ normales:
#
# $$
# p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
# \qquad \pi_k \ge 0, \quad \sum_k \pi_k = 1 .
# $$
#
# Eso es una **mezcla de gaussianas** (*Gaussian mixture model*, GMM). Los parámetros son los pesos $\pi_k$, las medias y las covarianzas.
#
# Lo interesante es lo que el modelo nos dice de cada punto. Por la regla de Bayes, la probabilidad de que el punto $\mathbf{x}_i$ venga de la componente $k$ es
#
# $$
# r_{ik} = P(k \mid \mathbf{x}_i) = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j} \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} .
# $$
#
# A $r_{ik}$ se le llama la **responsabilidad** de la componente $k$ sobre el punto $i$. k-medias da asignaciones **duras** (el pingüino 17 es del grupo 2, punto); la mezcla da asignaciones **suaves** (el pingüino 17 es del grupo 2 con probabilidad 0.9 y del grupo 1 con probabilidad 0.1). Un pingüino en la frontera entre dos nubes queda marcado como dudoso, que es justamente lo que es.

# %% [markdown]
# ## 4. El algoritmo EM, programado desde cero
#
# Si conociéramos la especie de cada pingüino, estimar los parámetros sería fácil: la proporción, la media y la covarianza de cada especie. Si conociéramos los parámetros, calcular las responsabilidades también sería fácil (la fórmula de arriba). El problema es que no conocemos ninguna de las dos cosas.
#
# El algoritmo EM rompe el círculo alternando, igual que k-medias alterna entre asignar puntos y mover centros:
#
# **Paso E** (*expectation*): con los parámetros actuales, calcular las responsabilidades $r_{ik}$ con la regla de Bayes.
#
# **Paso M** (*maximization*): con las responsabilidades, reestimar los parámetros como promedios **ponderados**. Llamando $N_k = \sum_i r_{ik}$ (el número «efectivo» de puntos de la componente $k$):
#
# $$
# \pi_k = \frac{N_k}{n}, \qquad
# \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i} r_{ik}\, \mathbf{x}_i, \qquad
# \boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{i} r_{ik}\, (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top .
# $$
#
# Son las fórmulas de siempre para proporción, media y covarianza, solo que cada punto cuenta según qué tanto pertenece a la componente.
#
# Para saber si vamos bien medimos la **log-verosimilitud**, qué tan probables son los datos bajo el modelo:
#
# $$
# \ell = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) .
# $$
#
# Se puede demostrar que **cada iteración de EM nunca hace bajar $\ell$**. Así que el algoritmo sube hasta estabilizarse, y paramos cuando ya casi no cambia.

# %%
def em_mezcla(X, K, iteraciones=200, tol=1e-6, semilla=0):
    """Ajusta una mezcla de K gaussianas con EM. Regresa los parámetros, las
    responsabilidades, la log-verosimilitud por iteración y la historia de parámetros."""
    n, d = X.shape
    rng = np.random.default_rng(semilla)
    # Inicio: medias en K puntos al azar, covarianzas identidad, pesos iguales
    mu = X[rng.choice(n, K, replace=False)]
    sigma = np.array([np.eye(d)] * K)
    pi = np.full(K, 1 / K)
    historia, logver = [(pi, mu, sigma)], []

    for _ in range(iteraciones):
        # Paso E: densidad ponderada de cada punto bajo cada componente, y responsabilidades
        dens = np.column_stack([pi[k] * multivariate_normal(mu[k], sigma[k]).pdf(X) for k in range(K)])
        total = dens.sum(axis=1, keepdims=True)
        r = dens / total
        logver.append(np.log(total).sum())

        # Paso M: proporciones, medias y covarianzas ponderadas por las responsabilidades
        Nk = r.sum(axis=0)
        pi = Nk / n
        mu = (r.T @ X) / Nk[:, None]
        sigma = np.array([((r[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] for k in range(K)])
        sigma += 1e-6 * np.eye(d)  # evita covarianzas singulares (ver la sección 9)
        historia.append((pi, mu, sigma))

        if len(logver) > 1 and logver[-1] - logver[-2] < tol:
            break
    return pi, mu, sigma, r, np.array(logver), historia


pi, mu, sigma, resp, logver, historia = em_mezcla(X, K=3, semilla=SEMILLA)
print(f"EM se detuvo después de {len(logver)} iteraciones")
print(f"log-verosimilitud final: {logver[-1]:.2f}")

# Como con k-medias, renumeramos las componentes según la especie mayoritaria, solo para los colores
grupo_em, orden = alinear(resp.argmax(axis=1))
p = np.argsort(orden)
pi, mu, sigma, resp = pi[p], mu[p], sigma[p], resp[:, p]
historia = [(a[p], b[p], c[p]) for a, b, c in historia]

# %% [markdown]
# Primero comprobemos la promesa: la log-verosimilitud nunca baja.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(len(logver)), logver, marker="o", markersize=4)
ax.set(title="La log-verosimilitud sube en cada iteración de EM",
       xlabel="iteración", ylabel="log-verosimilitud")
plt.show()
print("¿alguna vez bajó?", bool(np.any(np.diff(logver) < -1e-9)))

# %% [markdown]
# Casi toda la mejora ocurre en las primeras diez iteraciones; después el algoritmo solo hace ajustes finos. Veamos qué pasa en el plano. Cada componente se dibuja con dos elipses: la de una y la de dos desviaciones estándar. Los ejes de la elipse son los vectores propios de $\boldsymbol{\Sigma}_k$, y su largo, la raíz de los valores propios.

# %%
def elipses(ax, mu, sigma, color, etiqueta=None):
    """Dibuja las elipses de 1 y 2 desviaciones estándar de una normal bivariada."""
    valores, vectores = np.linalg.eigh(sigma)
    angulo = np.degrees(np.arctan2(vectores[1, 1], vectores[0, 1]))
    for z, ancho_linea in [(1, 2.2), (2, 1.2)]:
        ancho, alto = 2 * z * np.sqrt(valores[::-1])
        ax.add_patch(Ellipse(mu, ancho, alto, angle=angulo, fill=False, color=color, linewidth=ancho_linea))
    ax.plot(*mu, marker="+", markersize=12, markeredgewidth=2.5, color=color)


momentos = [0, 2, 5, len(historia) - 1]
fig, axs = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
for ax, t in zip(axs.flat, momentos):
    p_t, mu_t, sigma_t = historia[t]
    ax.scatter(X[:, 0], X[:, 1], s=12, color=rdp.estilo.GRIS, alpha=0.5, edgecolors="none")
    for k in range(3):
        elipses(ax, mu_t[k], sigma_t[k], CICLO[k])
    ax.set_title("inicio" if t == 0 else f"iteración {t}" + (" (final)" if t == momentos[-1] else ""))
for ax in axs[1]:
    ax.set_xlabel("largo del pico (est.)")
for ax in axs[:, 0]:
    ax.set_ylabel("profundidad del pico (est.)")
ax.set(xlim=(-3, 3.3), ylim=(-2.9, 3.1))
fig.suptitle("EM en acción: las componentes se mueven y cambian de forma", x=0.06, ha="left",
             fontsize=14, fontweight="semibold", color=rdp.estilo.MARINO)
plt.show()

# %% [markdown]
# Al inicio, las tres componentes son círculos iguales centrados en pingüinos elegidos al azar. Desde las primeras iteraciones cada una se mueve hacia una nube y, a diferencia de k-medias, **se estira y se inclina** para seguir su forma. Al final, cada elipse cubre una especie.
#
# ¿Qué tanto coincide con las especies? Asignamos cada pingüino a la componente de mayor responsabilidad y calculamos el ARI.

# %%
ari_em = adjusted_rand_score(especie, grupo_em)
print(f"ARI de k-medias: {ari_kmedias:.3f}")
print(f"ARI de la mezcla (EM): {ari_em:.3f}")
tabla_cruzada(grupo_em)

# %% [markdown]
# La mezcla comete muchos menos errores que k-medias con exactamente los mismos datos. Solo cambió la idea de qué es un grupo.
#
# ### Comparación con scikit-learn
#
# `GaussianMixture` hace lo mismo, con más cuidado numérico (trabaja con logaritmos y descomposiciones de Cholesky) y más opciones. Comparemos las medias. El orden de las componentes es arbitrario, así que las ordenamos por su primera coordenada antes de compararlas.

# %%
gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=10, random_state=SEMILLA).fit(X)

orden_nuestro, orden_sk = np.argsort(mu[:, 0]), np.argsort(gmm.means_[:, 0])
comparacion = pd.DataFrame({
    "media largo (nuestra)": mu[orden_nuestro, 0], "media largo (sklearn)": gmm.means_[orden_sk, 0],
    "media prof. (nuestra)": mu[orden_nuestro, 1], "media prof. (sklearn)": gmm.means_[orden_sk, 1],
    "peso (nuestro)": pi[orden_nuestro], "peso (sklearn)": gmm.weights_[orden_sk],
}).round(3)
print(f"log-verosimilitud: nuestra {logver[-1]:.2f} · sklearn {gmm.score(X) * len(X):.2f}")
print(f"ARI de sklearn: {adjusted_rand_score(especie, gmm.predict(X)):.3f}")
comparacion

# %% [markdown]
# Las medias y los pesos coinciden en las dos primeras cifras, y las dos soluciones asignan igual a los pingüinos (mismo ARI). Las pequeñas diferencias vienen de cuándo se detiene cada uno: scikit-learn para cuando la log-verosimilitud **promedio** por punto mejora menos de $10^{-3}$ (su parámetro `tol`), un criterio más relajado que el nuestro, por eso nuestra log-verosimilitud es un poco mayor. Por la misma razón, `gmm.score(X)` regresa el promedio por punto, y lo multiplicamos por $n$ para comparar.

# %% [markdown]
# ## 5. Responsabilidades: cuándo el modelo duda
#
# Como cada pingüino tiene una probabilidad de pertenecer a cada grupo, podemos medir la **certeza** de su asignación con la mayor de sus responsabilidades, $\max_k r_{ik}$. Vale 1 si el modelo está seguro y $1/3$ si no tiene idea.

# %%
certeza = resp.max(axis=1)
ambiguos = certeza < 0.8
print(f"pingüinos con certeza menor a 0.8: {ambiguos.sum()} de {len(certeza)}")

fig, ax = plt.subplots(figsize=(8, 5.8))
for k in range(3):
    m = grupo_em == k
    ax.scatter(X[m, 0], X[m, 1], s=10 + 40 * certeza[m] ** 4, color=CICLO[k],
               alpha=np.clip(certeza[m] ** 2, 0.15, 0.85), edgecolors="none")
ax.scatter(X[ambiguos, 0], X[ambiguos, 1], s=110, facecolors="none", edgecolors=rdp.estilo.TINTA, linewidths=1.3)
ax.set(title="Grupos de la mezcla: tamaño y opacidad según la certeza",
       xlabel="largo del pico (estandarizado)", ylabel="profundidad del pico (estandarizada)")
ax.text(-2.1, -1.9, "○  certeza menor a 0.8", fontsize=11.5, color=rdp.estilo.TINTA)
plt.show()

# %% [markdown]
# Compara con la gráfica de k-medias: la mezcla sí reconoce la punta alargada de la nube de Gentoo, y los errores bajan a menos de la mitad. Los pingüinos dudosos (encerrados en círculos) están donde se tocan las nubes, sobre todo entre Adelie y Chinstrap. ¿Qué especie son en realidad?

# %%
errores_em = grupo_em != especie_num
print(f"pingüinos mal agrupados: {errores_em.sum()}, de los cuales {(errores_em & ambiguos).sum()} están entre los dudosos")
print(f"tasa de error entre los dudosos: {errores_em[ambiguos].mean():.0%} · entre los demás: {errores_em[~ambiguos].mean():.1%}")
datos.loc[ambiguos, ["species", "sex"] + medidas[:2]].assign(
    grupo=np.array(ESPECIES)[grupo_em[ambiguos]], certeza=certeza[ambiguos].round(2)).sort_values("certeza")

# %% [markdown]
# La columna `grupo` dice a qué especie corresponde la componente que ganó. Entre los dudosos, el modelo se equivoca muchísimo más que entre los demás. Casi todos los demás errores son Chinstrap de pico corto metidos de lleno en la nube de Adelie: con solo estas dos medidas son indistinguibles, y ningún método podría acertarlos. Que el modelo dude es valioso en la práctica: un modelo que **sabe cuándo no sabe** nos dice qué casos conviene revisar a mano. k-medias no puede hacerlo; para él, un punto en la frontera es tan miembro de su grupo como uno en el centro.

# %% [markdown]
# ## 6. k-medias es un caso particular
#
# Si obligamos a que todas las componentes tengan la misma covarianza esférica $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ y los mismos pesos, la responsabilidad queda
#
# $$
# r_{ik} = \frac{\exp\!\left(-\lVert \mathbf{x}_i - \boldsymbol{\mu}_k \rVert^2 / 2\sigma^2\right)}{\sum_j \exp\!\left(-\lVert \mathbf{x}_i - \boldsymbol{\mu}_j \rVert^2 / 2\sigma^2\right)} .
# $$
#
# Ahora hagamos $\sigma$ cada vez más pequeña. La exponencial del centro más cercano domina a todas las demás, así que $r_{ik} \to 1$ para el centro más cercano y $r_{ik} \to 0$ para los otros: el paso E se vuelve «asignar cada punto a su centro más cercano», y el paso M, «mover cada centro al promedio de sus puntos». Eso es exactamente **k-medias**.
#
# Es decir, k-medias es una mezcla de gaussianas con círculos iguales y con asignaciones duras. La mezcla general relaja las dos restricciones: cada grupo tiene su forma, y las asignaciones expresan incertidumbre.

# %% [markdown]
# ## 7. ¿Cuántos grupos? El criterio BIC
#
# Hasta ahora hicimos trampa: usamos $K = 3$ porque sabíamos que hay tres especies. Un explorador de verdad no lo sabría.
#
# No podemos elegir $K$ con la log-verosimilitud, porque siempre mejora al agregar componentes (con una componente por punto sería infinita). Hay que **castigar la complejidad**. El criterio de información bayesiano (BIC) hace eso:
#
# $$
# \text{BIC} = -2\,\ell + p \log n ,
# $$
#
# donde $p$ es el número de parámetros libres del modelo. **Menor es mejor.** El primer término premia ajustar bien; el segundo cobra cada parámetro.
#
# Ahora usamos las **cuatro medidas** (pico, aleta y masa), que es lo que haría el explorador. Además del número de componentes probamos cuatro tipos de covarianza, que son distintas formas de restringir las elipses:
#
# | `covariance_type` | Forma de cada componente | Parámetros de covarianza ($d = 4$) |
# |---|---|---|
# | `spherical` | esfera, de tamaño propio | 1 por componente |
# | `diag` | elipse alineada con los ejes | 4 por componente |
# | `tied` | elipse libre, **la misma** para todas | 10 en total |
# | `full` | elipse libre, propia | 10 por componente |

# %%
X4 = StandardScaler().fit_transform(datos[medidas])
tipos = ["full", "tied", "diag", "spherical"]
ks = range(1, 8)

bic = pd.DataFrame({
    tipo: [GaussianMixture(k, covariance_type=tipo, n_init=5, random_state=SEMILLA).fit(X4).bic(X4) for k in ks]
    for tipo in tipos
}, index=ks)
bic.index.name = "componentes"
bic.round(0)

# %%
mejor_tipo, mejor_k = bic.stack().idxmin()[::-1]
print(f"mejor modelo por BIC: covariance_type='{mejor_tipo}', k = {mejor_k}")

fig, ax = plt.subplots(figsize=(8, 4.8))
for tipo, color in zip(tipos, CICLO):
    ax.plot(ks, bic[tipo], marker="o", color=color)
ax.plot(mejor_k, bic.loc[mejor_k, mejor_tipo], marker="o", markersize=14, markerfacecolor="none",
        markeredgecolor=rdp.estilo.TINTA, markeredgewidth=1.5)
rdp.estilo.etiquetar(ax, 7.2, {t: bic.loc[7, t] for t in tipos})
ax.set(title="BIC según número de componentes y tipo de covarianza (menor es mejor)",
       xlabel="número de componentes", ylabel="BIC", xlim=(0.7, 8.2))
plt.show()

# %% [markdown]
# Tres observaciones:
#
# - Los modelos **esféricos y diagonales** necesitan muchas componentes para ajustar bien: como sus elipses no se pueden inclinar, aproximan cada nube inclinada con varias redondas, y el BIC sigue bajando hasta $k = 7$. Es el mismo problema de k-medias, ahora visto con un número.
# - Los modelos **full** y **tied** tienen su mínimo en 3 o 4 componentes, cerca del número de especies.
# - El mínimo global **no** es 3: es `tied` con $k = 4$, aunque muy cerca de `tied` con $k = 3$. ¿Qué es ese cuarto grupo? Comparemos con la especie y el sexo.

# %%
mejor = GaussianMixture(mejor_k, covariance_type=mejor_tipo, n_init=5, random_state=SEMILLA).fit(X4)
grupo_bic = mejor.predict(X4)
print(f"ARI del mejor modelo por BIC contra la especie: {adjusted_rand_score(especie, grupo_bic):.3f}")
pd.crosstab([datos["species"], datos["sex"].fillna("sin dato")], grupo_bic,
            rownames=["especie", "sexo"], colnames=["grupo"])

# %% [markdown]
# Gentoo y Chinstrap quedan cada una en su propio grupo, casi sin errores. Pero los Adelie se reparten entre **dos** grupos, y no al azar: en uno predominan las hembras y en el otro los machos, aunque la separación está lejos de ser limpia. Los machos tienden a ser más grandes y pesados, y con cuatro medidas el modelo encontró que «Adelie más pequeños» y «Adelie más grandes» se describen un poco mejor como dos nubes que como una. Por eso el ARI contra la especie baja, aunque el modelo no sea «peor».
#
# ¿Se equivocó el BIC? No exactamente. El BIC contesta la pregunta «¿cuántas gaussianas describen mejor estos datos?», que **no es la misma** que «¿cuántas especies hay?». Las especies no tienen por qué ser gaussianas, y dentro de cada una hay subgrupos reales (los sexos). Lo que es un grupo depende de las variables que medimos y de la escala de detalle que nos interesa. El BIC es una guía, no un oráculo: un buen análisis mira el mínimo, los modelos cercanos (aquí, $k = 3$ está a muy poca distancia) y el significado de cada grupo.
#
# Para la evaluación, el modelo `full` con $k = 3$ y las cuatro medidas:

# %%
gmm4 = GaussianMixture(3, covariance_type="full", n_init=5, random_state=SEMILLA).fit(X4)
km4 = KMeans(3, n_init=10, random_state=SEMILLA).fit(X4)
print(f"ARI con las 4 medidas · k-medias: {adjusted_rand_score(especie, km4.labels_):.3f}"
      f" · mezcla full: {adjusted_rand_score(especie, gmm4.predict(X4)):.3f}")

# %% [markdown]
# Con las cuatro medidas la mezcla casi reproduce las especies; k-medias, en cambio, empeora respecto a usar solo el pico.

# %% [markdown]
# ## 8. La densidad para detectar casos raros
#
# Una mezcla es un modelo de la **densidad** de los datos: para cualquier pingüino nos dice qué tan típico es. Los que caen en zonas de muy baja densidad son candidatos a atípicos. `score_samples` da la log-densidad de cada punto.

# %%
log_densidad = gmm4.score_samples(X4)
raros = np.argsort(log_densidad)[:6]
datos.iloc[raros][["species", "island", "sex"] + medidas].assign(log_densidad=log_densidad[raros].round(2))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
por_especie(axs[0], largo, profundidad, ETIQUETAS_PICO, s=16, alpha=0.5)
axs[0].set(title="Los 6 pingüinos menos típicos", xlabel="largo del pico (mm)", ylabel="profundidad del pico (mm)")
por_especie(axs[1], datos["flipper_length_mm"], datos["body_mass_g"],
            {"Adelie": (177, 4400), "Chinstrap": (207, 2850), "Gentoo": (215, 6250)}, s=16, alpha=0.5)
axs[1].set(xlabel="largo de la aleta (mm)", ylabel="masa corporal (g)")
for ax, (cx, cy) in zip(axs, [("bill_length_mm", "bill_depth_mm"), ("flipper_length_mm", "body_mass_g")]):
    ax.scatter(datos[cx].iloc[raros], datos[cy].iloc[raros], s=120, facecolors="none",
               edgecolors=rdp.estilo.TINTA, linewidths=1.5)
plt.show()

# %% [markdown]
# Ninguno parece un error de captura: las medidas son razonables para un pingüino. Son individuos que, **para su especie**, tienen una combinación poco común, por ejemplo un pico muy largo con una masa baja, o una aleta corta para su tamaño. Varios quedan en la orilla de su nube o entre dos especies. Como siempre con los atípicos, el modelo señala; la decisión de si es un error, un caso raro o algo interesante le corresponde a quien conoce los datos (aquí, a los biólogos que midieron).
#
# Fíjate además en que una densidad baja depende del modelo: el mismo pingüino podría ser normal con otra mezcla o con otras variables.

# %% [markdown]
# ## 9. Cierre: mezclas o k-medias
#
# **Cuándo preferir una mezcla de gaussianas:**
#
# - cuando los grupos son alargados, inclinados o de tamaños distintos;
# - cuando importa saber **qué tan segura** es cada asignación;
# - cuando queremos un modelo de la densidad (para detectar atípicos o generar datos nuevos);
# - cuando queremos un criterio con fundamento (BIC) para elegir el número de grupos.
#
# **Lo que cuesta:**
#
# - **Más parámetros.** Con `full`, cada componente en $d$ dimensiones tiene $d$ medias y $d(d+1)/2$ covarianzas. En 50 dimensiones son más de 1,300 por componente; hacen falta muchos datos, y por eso existen `diag`, `tied` y `spherical`.
# - **Mínimos locales.** EM solo garantiza que la log-verosimilitud no baja, no que llegue al máximo global. El resultado depende del inicio; por eso se corre varias veces (`n_init`), y scikit-learn inicializa por omisión con k-medias.
# - **Singularidades.** Si una componente se encoge sobre un solo punto, su covarianza tiende a cero y la densidad en ese punto, a infinito: la log-verosimilitud «mejora» sin límite con un modelo absurdo. La defensa es sumar una constante pequeña a la diagonal de cada covarianza; en scikit-learn es el parámetro `reg_covar` (por omisión $10^{-6}$), y en nuestra función es la línea `sigma += 1e-6 * np.eye(d)`.
# - **Supone normalidad.** Grupos con formas muy distintas de una elipse (anillos, medias lunas) siguen siendo difíciles. Para eso veremos métodos basados en densidad local, como DBSCAN.

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u02", ids=["u02-p07", "u02-p08", "u02-p09"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. Corre `em_mezcla` con cinco semillas distintas. ¿Llegan todas a la misma log-verosimilitud? Si alguna se queda en un valor menor, dibuja sus elipses: ¿qué mínimo local encontró?
# 2. Modifica `em_mezcla` para que todas las componentes compartan la misma covarianza (el caso `tied`). Pista: en el paso M, la covarianza común es el promedio de las $\boldsymbol{\Sigma}_k$ ponderado por $N_k$.
# 3. Quita la línea `sigma += 1e-6 * np.eye(d)` y ajusta una mezcla con $K = 10$ sobre las dos medidas del pico. ¿Qué pasa? ¿Por qué?
# 4. Repite la sección 7 usando solo los pingüinos Adelie y las cuatro medidas. ¿Cuántas componentes elige el BIC? ¿Coinciden con el sexo?
# 5. Genera 344 pingüinos «artificiales» con `gmm4.sample(344)` y grafícalos junto a los reales (regresa a las unidades originales con el `inverse_transform` del escalador). ¿Se parecen?
