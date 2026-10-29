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
# # Reducir dimensión: PCA, t-SNE y UMAP
#
# **Unidad 2 · Ejemplo 3**
#
# Una imagen de 8 × 8 píxeles es, para la computadora, una lista de 64 números: un punto en un espacio de 64 dimensiones. No podemos dibujar ese espacio, y casi todo lo que contiene es redundante (los píxeles vecinos se parecen, las orillas casi siempre están en blanco). En esta libreta buscamos **unas pocas coordenadas nuevas que conserven lo importante** de los datos originales.
#
# Eso sirve para:
#
# - **visualizar**: ver en dos dimensiones cómo se agrupan los datos;
# - **comprimir**: guardar 10 números por imagen en lugar de 64;
# - **quitar ruido**: lo que se descarta suele ser, en buena parte, ruido;
# - **preprocesar**: darle a otro modelo (k-medias, vecinos más cercanos, una regresión) menos columnas y menos correlacionadas.
#
# Veremos tres métodos que responden a la pregunta de formas muy distintas: **PCA** (lineal, el clásico), **t-SNE** y **UMAP** (no lineales, pensados para visualizar).

# %% [markdown]
# ## 0. Preparación del entorno

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q umap-learn "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

import rdp

rdp.estilo.usar()
SEMILLA = 2011
warnings.filterwarnings("ignore", message="n_jobs value")  # aviso de umap al fijar la semilla

# %% [markdown]
# ## 1. El problema y los datos
#
# **Datos.** El conjunto *Optical Recognition of Handwritten Digits* del repositorio UCI, preparado por E. Alpaydin y C. Kaynak (Universidad Boğaziçi, Estambul, 1998). Se pidió a 43 personas que llenaran formularios con dígitos escritos a mano; los formularios se digitalizaron, cada dígito se recortó y se normalizó a una cuadrícula de 32 × 32 píxeles blancos y negros. Para reducir el tamaño y tolerar pequeñas deformaciones, la cuadrícula se dividió en bloques de 4 × 4 y en cada bloque se contaron los píxeles negros. El resultado es una imagen de **8 × 8 píxeles con 17 niveles de gris** (de 0 a 16).
#
# scikit-learn incluye la parte de prueba de ese conjunto: **1,797 dígitos**, más o menos 180 de cada uno del 0 al 9. Viene con la biblioteca, así que no hay nada que descargar.
#
# **Problema.** Cada dígito es un punto en $\mathbb{R}^{64}$. ¿Cómo encontrar unas pocas coordenadas que conserven lo importante, de modo que podamos verlo, guardarlo de forma compacta o usarlo con otros modelos?
#
# Usaremos la etiqueta (qué dígito es) **solo para colorear las gráficas** y juzgar si la reducción tiene sentido. Ningún método de esta libreta la ve.

# %%
digitos = load_digits()
X, y = digitos.data, digitos.target
print(X.shape, "· valores de", X.min(), "a", X.max())
print("dígitos por clase:", np.bincount(y))

# %%
fig, axs = plt.subplots(4, 10, figsize=(10, 4.4))
for ax, img, etiqueta in zip(axs.flat, digitos.images, y):
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=16)
    ax.set_title(etiqueta, fontsize=11, loc="center", pad=3)
    ax.axis("off")
fig.suptitle("Los primeros 40 dígitos del conjunto (8 × 8 píxeles)", x=0.02, ha="left",
             color=rdp.estilo.MARINO, fontweight="semibold")
plt.show()

# %% [markdown]
# Con 8 × 8 píxeles se reconocen casi todos, aunque algunos (un 1 con base, un 9 cerrado) ya son dudosos. Cada imagen se «desenrolla» en un renglón de 64 números:

# %%
print(digitos.images[0].astype(int))
print("→ renglón de X:", X[0].astype(int)[:16], "…")

# %% [markdown]
# ## 2. ¿Por qué molesta tener muchas dimensiones?
#
# Además de no poder dibujarlos, los espacios de muchas dimensiones se comportan de forma poco intuitiva. A esto se le llama **la maldición de la dimensión** (*curse of dimensionality*, R. Bellman, 1961).
#
# Un experimento sencillo: tomamos 500 puntos al azar, uniformes en el cubo $[0, 1]^d$, elegimos otro punto al azar como referencia y medimos la distancia a su vecino **más cercano** y a su vecino **más lejano**. Comparamos las dos con la razón
#
# $$
# \text{contraste} = \frac{d_{\max} - d_{\min}}{d_{\min}}.
# $$
#
# Si el contraste es grande, «cerca» y «lejos» significan cosas muy distintas. Si es casi 0, todos los puntos están más o menos a la misma distancia.

# %%
rng = np.random.default_rng(SEMILLA)

def contraste(d, n=500, consultas=50):
    """Promedio de (dmax - dmin) / dmin para `consultas` puntos de referencia."""
    razones = []
    for _ in range(consultas):
        puntos = rng.random((n, d))
        ref = rng.random(d)
        dist = np.linalg.norm(puntos - ref, axis=1)
        razones.append((dist.max() - dist.min()) / dist.min())
    return np.mean(razones)

dimensiones = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
contrastes = {d: contraste(d) for d in dimensiones}
for d in [2, 10, 100, 1000]:
    print(f"d = {d:>4}:  (dmax - dmin) / dmin = {contrastes[d]:7.2f}")

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(dimensiones, list(contrastes.values()), marker="o", color=rdp.estilo.AZUL)
for d in [2, 10, 100, 1000]:
    ax.annotate(f"{contrastes[d]:.2f}", (d, contrastes[d]), textcoords="offset points",
                xytext=(6, 8), fontsize=11, color=rdp.estilo.TINTA)
ax.set(xscale="log", yscale="log", xlabel="dimensión d (escala log)",
       ylabel="(dmax − dmin) / dmin  (escala log)",
       title="Al crecer la dimensión, el más cercano y el más lejano se parecen")
plt.show()

# %% [markdown]
# En el plano, el punto más lejano está decenas de veces más lejos que el más cercano. En 1,000 dimensiones, apenas un poco más lejos. La razón es que la distancia es una suma de $d$ términos independientes, y una suma de muchos términos se concentra alrededor de su media.
#
# **Moraleja.** k-medias, vecinos más cercanos y casi todo lo que usa distancias supone que «cerca» quiere decir «parecido». Con muchas dimensiones esa idea se debilita, y además se necesitan muchísimos datos para cubrir el espacio. Por suerte, los datos reales casi nunca llenan las 64 dimensiones: viven cerca de una superficie de dimensión mucho menor. Encontrarla es el trabajo de la reducción de dimensión.

# %% [markdown]
# ## 3. PCA desde cero
#
# El **análisis de componentes principales** (PCA, *principal component analysis*) busca nuevos ejes que son combinaciones lineales de los originales, ordenados de mayor a menor importancia. Hay dos maneras de decir qué es «importante», y dan la misma respuesta.
#
# Sea $\tilde{X}$ la matriz de datos **centrada** ($n$ renglones, $d$ columnas, a cada columna se le resta su media) y $S = \frac{1}{n-1}\tilde{X}^\top\tilde{X}$ su matriz de covarianzas.
#
# **1. Máxima varianza.** La primera componente es la dirección $w$ (con $\lVert w \rVert = 1$) en la que los datos proyectados varían más:
#
# $$
# w_1 = \arg\max_{\lVert w \rVert = 1} \; \frac{1}{n-1}\sum_{i=1}^{n} \left(w^\top \tilde{x}_i\right)^2 = \arg\max_{\lVert w \rVert = 1} \; w^\top S\, w .
# $$
#
# La segunda es la de máxima varianza entre las perpendiculares a la primera, y así sucesivamente. La solución son los **vectores propios** de $S$, y la varianza en cada dirección es su **valor propio**.
#
# **2. Mínimo error de reconstrucción.** Buscamos el subespacio de dimensión $k$ (con base ortonormal en las columnas de $W$) que deja a los puntos lo más cerca posible de su proyección:
#
# $$
# \min_{W^\top W = I_k} \; \sum_{i=1}^{n} \left\lVert \tilde{x}_i - W W^\top \tilde{x}_i \right\rVert^2 .
# $$
#
# **¿Por qué es lo mismo?** Por Pitágoras. Cada punto se descompone en su proyección sobre el subespacio más un residuo perpendicular, así que $\lVert \tilde{x}_i \rVert^2 = \lVert \text{proyección} \rVert^2 + \lVert \text{residuo} \rVert^2$. El lado izquierdo no depende de $W$: lo que ganamos en varianza proyectada lo perdemos en error, y viceversa. Maximizar una es minimizar la otra.
#
# **Cómo se calcula.** No hace falta formar $S$. La descomposición en valores singulares $\tilde{X} = U \Sigma V^\top$ da todo:
#
# $$
# S = \frac{1}{n-1}\tilde{X}^\top\tilde{X} = V \, \frac{\Sigma^2}{n-1} \, V^\top ,
# $$
#
# así que las **componentes** son las columnas de $V$ (los vectores singulares derechos), las **varianzas** son $s_j^2/(n-1)$ y las **coordenadas nuevas** de los datos son $\tilde{X} V_k = U_k \Sigma_k$.

# %%
media = X.mean(axis=0)
Xc = X - media
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

componentes = Vt                          # renglón j = componente j (vector en R^64)
varianzas = s**2 / (len(X) - 1)
Z = Xc @ Vt[:2].T                         # coordenadas en las dos primeras componentes

print("varianza total de los datos:", round(Xc.var(axis=0, ddof=1).sum(), 1))
print("suma de las varianzas de las componentes:", round(varianzas.sum(), 1))
print("primeras 5 varianzas:", varianzas[:5].round(1))

# %% [markdown]
# La varianza total se conserva: PCA solo la **reparte** entre ejes nuevos, cargando la mayor parte en los primeros. Comparemos con scikit-learn:

# %%
pca = PCA(random_state=SEMILLA).fit(X)

print("¿mismas varianzas?", np.allclose(varianzas, pca.explained_variance_))
# Un vector propio está definido salvo el signo: w y -w son la misma dirección.
productos = np.abs(np.sum(componentes[:20] * pca.components_[:20], axis=1))
print("|⟨ nuestra, sklearn ⟩| en las primeras 20 componentes:", productos.min().round(6), "a", productos.max().round(6))
print("signos que difieren en las primeras 10:",
      np.sign(np.sum(componentes[:10] * pca.components_[:10], axis=1)).astype(int))

# %% [markdown]
# Las componentes coinciden **salvo el signo**: si $w$ es una dirección de máxima varianza, $-w$ también lo es. Ningún signo es «el correcto», así que nunca hay que interpretar el signo de una componente por sí solo.

# %% [markdown]
# ## 4. ¿Cuántas componentes?
#
# La **proporción de varianza explicada** de la componente $j$ es $\lambda_j / \sum_l \lambda_l$. Su suma acumulada dice cuánta de la variación de los datos conservamos con las primeras $k$.

# %%
acumulada = np.cumsum(pca.explained_variance_ratio_)
niveles = [0.80, 0.90, 0.95]
k_nivel = {p: int(np.searchsorted(acumulada, p) + 1) for p in niveles}
for p, k in k_nivel.items():
    print(f"{p:.0%} de la varianza: {k:>2} componentes de 64")
print(f"PC1 sola: {pca.explained_variance_ratio_[0]:.1%} · PC1 + PC2: {acumulada[1]:.1%}")

# %%
fig, ax = plt.subplots(figsize=(8, 4.4))
k = np.arange(1, 65)
ax.bar(k, pca.explained_variance_ratio_, color=rdp.estilo.PALETA["azul-tenue"], width=0.8, label="por componente")
ax.plot(k, acumulada, color=rdp.estilo.AZUL, marker="o", markersize=3)
for p, kp in k_nivel.items():
    ax.plot([0, kp, kp], [p, p, 0], color=rdp.estilo.GRIS, lw=1, ls="--", zorder=0)
    ax.text(kp + 0.8, p - 0.03, f"{p * 100:.0f} % → {kp}", ha="left", va="top",
            fontsize=11, color=rdp.estilo.TINTA)
ax.text(40, 0.08, "varianza de cada componente", color=rdp.estilo.TENUE, fontsize=11)
ax.text(64, 1.02, "acumulada", ha="right", color=rdp.estilo.AZUL, fontsize=11)
ax.set(xlim=(0, 65), ylim=(0, 1.08), xlabel="número de componentes",
       ylabel="proporción de varianza", title="Varianza explicada por las componentes principales")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.show()

# %% [markdown]
# La curva sube rápido y luego se aplana. Con unas 20 componentes de 64 ya tenemos el 90 % de la variación; las últimas aportan casi nada (de hecho, algunos píxeles de la orilla valen 0 en todas las imágenes, así que tienen varianza nula).
#
# No hay un umbral universal. 90 % o 95 % son puntos de partida razonables para preprocesar; para visualizar, nos quedamos con 2 aunque expliquen poco.

# %% [markdown]
# ## 5. Ver las componentes y reconstruir
#
# Como cada componente es un vector de 64 números, podemos dibujarla como una imagen de 8 × 8. Usamos un mapa divergente centrado en 0: azul donde la componente suma tinta, rojo donde la quita.

# %%
fig, axs = plt.subplots(1, 9, figsize=(12, 2.4))
axs[0].imshow(pca.mean_.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
axs[0].set_title("promedio", fontsize=11, loc="center")
for j, ax in enumerate(axs[1:]):
    tope = np.abs(pca.components_[j]).max()  # escala simétrica: el blanco es 0
    ax.imshow(pca.components_[j].reshape(8, 8), cmap="rdp_div", vmin=-tope, vmax=tope)
    ax.set_title(f"PC{j + 1}", fontsize=11, loc="center")
for ax in axs:
    ax.axis("off")
plt.show()

# %% [markdown]
# Cada dígito se escribe como **el promedio más una combinación de estas imágenes**:
#
# $$
# x \approx \bar{x} + z_1 w_1 + z_2 w_2 + \dots + z_k w_k, \qquad z_j = w_j^\top (x - \bar{x}).
# $$
#
# La PC1, por ejemplo, es azul en la parte media e izquierda (donde tienen tinta el 4 y el 6) y roja arriba y a la derecha (donde la tienen el 3 y el 9): los 4 y los 6 obtienen coordenadas $z_1$ grandes y positivas, y los 3 y los 9, negativas. La PC2 es azul en un trazo vertical central y roja a los lados: separa los 1 y los 7 (positivos) de los 0 (negativos, porque tienen tinta en los costados y hueco en medio). Las componentes posteriores agregan detalles cada vez más finos. Veamos cómo se reconstruye un mismo dígito al ir agregando términos:

# %%
indice = 1000
x = X[indice]
z = pca.transform(x[None, :])[0]
cuantas = [2, 5, 10, 20, 40]

fig, axs = plt.subplots(1, len(cuantas) + 1, figsize=(10, 2.4))
for ax, kk in zip(axs, cuantas):
    reconstruida = pca.mean_ + z[:kk] @ pca.components_[:kk]
    ax.imshow(reconstruida.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
    ax.set_title(f"{kk} componentes", fontsize=11, loc="center")
axs[-1].imshow(x.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
axs[-1].set_title("original (64)", fontsize=11, loc="center")
for ax in axs:
    ax.axis("off")
plt.show()

# %% [markdown]
# Con 2 componentes la reconstrucción es una mancha genérica; con 10 ya se reconoce el dígito; con 20 es casi indistinguible del original. Medido sobre todo el conjunto, el error cuadrático medio por píxel baja así:

# %%
ks = np.arange(0, 65)
errores = []
for kk in ks:
    Zk = Xc @ Vt[:kk].T
    errores.append(np.mean((Xc - Zk @ Vt[:kk]) ** 2))
errores = np.array(errores)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(ks, errores, color=rdp.estilo.AZUL)
for kk in cuantas:
    ax.plot(kk, errores[kk], "o", color=rdp.estilo.ROJO)
    ax.annotate(f"{kk}: {errores[kk]:.2f}", (kk, errores[kk]), xytext=(6, 6),
                textcoords="offset points", fontsize=11, color=rdp.estilo.TINTA)
ax.set(xlabel="número de componentes k", ylabel="error cuadrático medio por píxel",
       title="Error de reconstrucción de los 1,797 dígitos")
plt.show()

# %% [markdown]
# Esta curva es la imagen en espejo de la varianza acumulada: el error con $k$ componentes es exactamente la suma de las varianzas **descartadas** (dividida entre 64 píxeles y con un factor $(n-1)/n$). Es la equivalencia de las dos formulaciones, ahora con números:

# %%
kk = 10
print(f"error con {kk} componentes:                {errores[kk]:.4f}")
print(f"varianza descartada · (n-1)/n / 64:      {varianzas[kk:].sum() * (len(X) - 1) / len(X) / 64:.4f}")

# %% [markdown]
# **Compresión y ruido.** Guardar 20 coordenadas por imagen en lugar de 64 (más el promedio y las 20 componentes, una sola vez) reduce el almacenamiento a menos de un tercio con poca pérdida. Y como el ruido de los píxeles se reparte en todas las direcciones mientras que la forma del dígito se concentra en las primeras, quedarse con las primeras componentes también funciona como un filtro de ruido (lo probarás en los ejercicios).

# %% [markdown]
# ## 6. La escala importa
#
# PCA busca direcciones de máxima **varianza**, y la varianza depende de las unidades. En los dígitos todas las columnas son píxeles en la misma escala (0 a 16), así que no hay problema. Pero ¿qué pasa si las columnas miden cosas distintas?
#
# Usamos los pingüinos del archipiélago Palmer (Antártida), registrados por K. Gorman en la estación Palmer del programa LTER y distribuidos por A. Horst, A. Hill y K. Gorman (2020). Tomamos las cuatro medidas: largo y profundidad del pico y largo de la aleta (en milímetros) y masa corporal (en gramos).

# %%
pinguinos = rdp.datos.cargar("pinguinos")
medidas = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
pinguinos = pinguinos.dropna(subset=medidas).reset_index(drop=True)
P = pinguinos[medidas].to_numpy()
print(len(pinguinos), "pingüinos")
pinguinos[medidas].describe().loc[["mean", "std"]].round(1)

# %%
pca_crudo = PCA().fit(P)
pca_estandar = PCA().fit(StandardScaler().fit_transform(P))

resumen = pd.DataFrame({
    "sin estandarizar": pca_crudo.components_[0],
    "estandarizadas": pca_estandar.components_[0],
}, index=medidas).round(3)
resumen.loc["varianza explicada por PC1"] = [pca_crudo.explained_variance_ratio_[0].round(4),
                                             pca_estandar.explained_variance_ratio_[0].round(4)]
resumen

# %% [markdown]
# Sin estandarizar, la masa tiene una desviación de unos 800 gramos, mientras que las demás medidas varían unos pocos milímetros. La PC1 es prácticamente «la masa» (peso casi 1 en esa columna) y se lleva más del 99 % de la varianza. No descubrimos nada: solo que los gramos son números grandes. Si hubiéramos medido la masa en kilogramos, el resultado sería otro.
#
# Estandarizadas (media 0 y desviación 1, como en la Unidad 1), las cuatro medidas pesan en la PC1, que ahora representa el **tamaño general** del pingüino. Veámoslo en el plano de las dos primeras componentes, coloreando por especie:

# %%
especies = pinguinos["species"].to_numpy()
colores = dict(zip(["Adelie", "Chinstrap", "Gentoo"], [rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC]))

fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, modelo, datos, titulo in [
    (axs[0], pca_crudo, P, "Sin estandarizar"),
    (axs[1], pca_estandar, StandardScaler().fit_transform(P), "Estandarizadas"),
]:
    Zp = modelo.transform(datos)
    for esp, color in colores.items():
        m = especies == esp
        ax.scatter(Zp[m, 0], Zp[m, 1], s=18, alpha=0.7, color=color, edgecolors="none")
        cx, cy = np.median(Zp[m, 0]), np.median(Zp[m, 1])
        ax.text(cx, cy, esp, color=color, fontsize=12, fontweight="bold", ha="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    r = modelo.explained_variance_ratio_
    ax.set(title=titulo, xlabel=f"PC1 ({r[0]:.2%} de la varianza)", ylabel=f"PC2 ({r[1]:.2%})")
plt.show()

# %% [markdown]
# A la izquierda, la PC1 es simplemente la masa: separa a los Gentoo (los más pesados) y deja a Adelie y Chinstrap encimados, porque pesan casi lo mismo. Fíjate en las escalas de los ejes: la PC1 va de −1,500 a 2,000 gramos y la PC2 de −25 a 15 milímetros; la PC2 explica apenas el 0.01 % de la varianza. A la derecha, las tres especies aparecen separadas: la PC2 recoge la forma del pico, que es justo lo que distingue a Chinstrap de Adelie.
#
# **Regla práctica.** Si las columnas tienen unidades distintas, estandariza antes de PCA. Si todas están en la misma unidad y su escala es significativa (píxeles, frecuencias de palabras), puedes no hacerlo.

# %% [markdown]
# ## 7. Los dígitos en dos dimensiones con PCA
#
# De regreso a los dígitos: proyectamos los 1,797 puntos sobre las dos primeras componentes y los coloreamos por el dígito que representan. Hay 10 clases, así que necesitamos 10 colores distinguibles (la paleta `tab10` de matplotlib); en lugar de una leyenda, escribimos cada número en la mediana de su grupo.

# %%
def dibujar_digitos(ax, Z, y, titulo, tam=7):
    """Diagrama de dispersión coloreado por dígito, con el número en la mediana de cada grupo."""
    paleta = plt.get_cmap("tab10")
    for d in range(10):
        m = y == d
        ax.scatter(Z[m, 0], Z[m, 1], s=tam, color=paleta(d), alpha=0.65, edgecolors="none")
    for d in range(10):
        cx, cy = np.median(Z[y == d], axis=0)
        ax.text(cx, cy, str(d), fontsize=15, fontweight="bold", color=paleta(d),
                ha="center", va="center", path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

Z_pca = PCA(n_components=2).fit_transform(X)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
dibujar_digitos(ax, Z_pca, y, f"PCA: dos primeras componentes ({acumulada[1]:.0%} de la varianza)")
plt.show()

# %% [markdown]
# Algunos dígitos ocupan su propia región (el 0, el 3, y el 4 y el 6, que solo se tocan en su frontera), pero en el centro el 1, el 7, el 8, el 5 y el 2 se enciman. No es de extrañar: dos ejes explican menos de un tercio de la varianza, y una proyección **lineal** solo puede «aplastar» el espacio desde un ángulo. Si los grupos están separados por una superficie curva, ninguna sombra plana los separa.

# %% [markdown]
# ## 8. t-SNE: conservar vecindarios
#
# **t-SNE** (*t-distributed stochastic neighbor embedding*, van der Maaten y Hinton, 2008) cambia el objetivo. No intenta conservar distancias ni varianzas, sino **quién es vecino de quién**.
#
# 1. En el espacio original, para cada punto $x_i$ define la probabilidad de que «elija» a $x_j$ como vecino, con una campana gaussiana centrada en $x_i$:
#    $$p_{j\mid i} = \frac{\exp\left(-\lVert x_i - x_j \rVert^2 / 2\sigma_i^2\right)}{\sum_{l \neq i}\exp\left(-\lVert x_i - x_l \rVert^2 / 2\sigma_i^2\right)} .$$
#    El ancho $\sigma_i$ se ajusta para que cada punto tenga aproximadamente el mismo número efectivo de vecinos: ese número es la **perplejidad**.
# 2. En el plano, coloca puntos $z_i$ y define probabilidades parecidas $q_{ij}$, pero con una distribución **t de Student** (de colas pesadas) en lugar de la gaussiana:
#    $$q_{ij} \propto \left(1 + \lVert z_i - z_j \rVert^2\right)^{-1} .$$
# 3. Mueve los $z_i$ (descenso por gradiente) para que las $q$ se parezcan lo más posible a las $p$, medido con la divergencia de Kullback-Leibler.
#
# ¿Por qué la $t$? En 64 dimensiones caben muchos puntos a distancia moderada de uno dado; en el plano no hay espacio para todos. Las colas pesadas permiten poner los puntos no vecinos **bastante más lejos** en el plano sin mucho costo, lo que deja espacio para que cada vecindario se vea como un grupo separado.

# %%
inicio = time.perf_counter()
Z_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEMILLA).fit_transform(X)
t_tsne = time.perf_counter() - inicio
print(f"t-SNE (perplejidad 30): {t_tsne:.1f} s")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
dibujar_digitos(ax, Z_tsne, y, "t-SNE (perplejidad 30)")
plt.show()

# %% [markdown]
# El cambio es enorme: los diez dígitos forman grupos casi limpios, sin haber visto nunca las etiquetas. Los pocos puntos fuera de lugar suelen ser dígitos ambiguos (un 1 que parece 7, un 9 que parece 3).
#
# ### El efecto de la perplejidad
#
# La perplejidad controla cuántos vecinos «mira» cada punto. Con valores pequeños, t-SNE se fija en detalles muy locales; con valores grandes, en estructura más global.

# %%
perplejidades = [5, 30, 100]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.9))
for ax, perp in zip(axs, perplejidades):
    Zp = Z_tsne if perp == 30 else TSNE(perplexity=perp, init="pca", random_state=SEMILLA).fit_transform(X)
    dibujar_digitos(ax, Zp, y, f"perplejidad {perp}", tam=4)
plt.show()

# %% [markdown]
# Con perplejidad 5 los grupos se ven deshilachados y se fragmentan en islas pequeñas (subestilos de escritura, o simplemente ruido); con 30 y con 100 son más compactos y las islas sueltas disminuyen. 30 es el valor por omisión y suele funcionar bien con unos cuantos miles de puntos.
#
# ### Cuidado al interpretar t-SNE
#
# Wattenberg, Viégas y Johnson (2016), en [«How to Use t-SNE Effectively»](https://distill.pub/2016/misread-tsne/) (*Distill*), muestran con muchos ejemplos qué **no** se puede leer en una gráfica de t-SNE:
#
# - **El tamaño de los grupos no significa nada.** t-SNE adapta $\sigma_i$ a la densidad local, así que un grupo muy disperso y uno muy compacto pueden verse del mismo tamaño.
# - **Las distancias entre grupos no significan nada** (o casi). Que el 9 quede junto al 3 en esta gráfica no quiere decir, por sí solo, que se parezcan más que el 9 y el 4.
# - **Cambia con la semilla y con la perplejidad.** Con inicialización aleatoria, cada corrida da un dibujo distinto. Aquí usamos `init="pca"` y una semilla fija para que el resultado sea más estable y reproducible, pero conviene probar varias.
# - **Puede inventar grupos.** Con perplejidad baja, incluso datos sin estructura producen «islas».
# - **No tiene `transform`.** t-SNE calcula coordenadas solo para los puntos que vio; para ubicar un punto nuevo hay que volver a correrlo todo. Por eso sirve para **visualizar**, no para preprocesar.

# %% [markdown]
# ## 9. UMAP
#
# **UMAP** (*uniform manifold approximation and projection*, McInnes, Healy y Melville, 2018) parte de la misma intuición. En una frase: **construye un grafo de vecinos más cercanos en el espacio original y busca un diseño en baja dimensión cuyo grafo se le parezca**. Sus dos parámetros principales:
#
# - `n_neighbors`: cuántos vecinos se usan para construir el grafo (el equivalente de la perplejidad: local contra global);
# - `min_dist`: qué tan apretados pueden quedar los puntos en el plano (valores pequeños dan grupos más compactos).
#
# La implementación de `umap-learn` usa numba, que compila el código la primera vez que se ejecuta. Para que la comparación de tiempos sea justa, hacemos antes una corrida pequeña de calentamiento.

# %%
umap.UMAP(n_neighbors=15, random_state=SEMILLA).fit(X[:200])  # calentamiento: compila con numba

inicio = time.perf_counter()
reductor = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEMILLA)
Z_umap = reductor.fit_transform(X)
t_umap = time.perf_counter() - inicio
print(f"t-SNE: {t_tsne:.1f} s · UMAP: {t_umap:.1f} s")

fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
dibujar_digitos(axs[0], Z_tsne, y, f"t-SNE ({t_tsne:.1f} s)", tam=5)
dibujar_digitos(axs[1], Z_umap, y, f"UMAP ({t_umap:.1f} s)", tam=5)
plt.show()

# %% [markdown]
# Los dos separan los dígitos de forma parecida. UMAP tiende a dar grupos más compactos y con más espacio blanco entre ellos, y se dice que conserva algo mejor la estructura global, aunque las advertencias sobre tamaños y distancias aplican igual. Con 1,797 puntos los tiempos son comparables; la ventaja de UMAP en velocidad se nota con decenas o cientos de miles de puntos (aunque la implementación de scikit-learn de t-SNE con el método de Barnes-Hut también escala razonablemente).
#
# ### UMAP sí puede ubicar puntos nuevos
#
# A diferencia de t-SNE, UMAP aprende una transformación: con `transform` coloca puntos que no vio en el ajuste. Lo probamos ajustando con 1,500 dígitos y proyectando los 297 restantes.

# %%
X_aj, X_nuevos, y_aj, y_nuevos = X[:1500], X[1500:], y[:1500], y[1500:]
reductor_aj = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEMILLA).fit(X_aj)
Z_aj = reductor_aj.embedding_
Z_nuevos = reductor_aj.transform(X_nuevos)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
dibujar_digitos(ax, Z_aj, y_aj, "UMAP: 1,500 dígitos de ajuste y 297 nuevos (×)", tam=5)
paleta = plt.get_cmap("tab10")
ax.scatter(Z_nuevos[:, 0], Z_nuevos[:, 1], marker="x", s=22, linewidths=1.2,
           c=[paleta(d) for d in y_nuevos])
plt.show()

# %% [markdown]
# Casi todos los dígitos nuevos caen en el grupo de su mismo número. Esto hace que UMAP pueda usarse dentro de un `Pipeline` como paso de preprocesamiento (por ejemplo, antes de un algoritmo de agrupamiento), aunque hay que hacerlo con cuidado: su resultado depende de los parámetros y de la semilla.

# %% [markdown]
# ## 10. Resumen: ¿cuál uso?
#
# | | PCA | t-SNE | UMAP |
# |---|---|---|---|
# | Tipo | lineal | no lineal | no lineal |
# | Conserva | varianza global (distancias grandes) | vecindarios locales | vecindarios locales y algo de estructura global |
# | Uso principal | preprocesar, comprimir, quitar ruido, primera visualización | solo visualizar | visualizar; con cuidado, preprocesar |
# | Puntos nuevos | sí (`transform`, una multiplicación de matrices) | no | sí (`transform`, aproximado) |
# | Ejes interpretables | sí (las componentes son combinaciones de las columnas) | no | no |
# | Velocidad | muy rápido, determinista | lento con muchos datos, depende de la semilla | rápido, depende de la semilla |
# | Parámetros principales | número de componentes | perplejidad | `n_neighbors`, `min_dist` |
#
# Una receta frecuente combina los dos mundos: primero **PCA** a unas 30–50 componentes (quita ruido y acelera) y después **t-SNE o UMAP** a 2 dimensiones para mirar. Y siempre, antes de PCA, pregúntate si las columnas están en la misma escala.

# %% [markdown]
# ## 11. Autoevaluación

# %%
rdp.quiz.mostrar("u02", ids=["u02-p10", "u02-p11", "u02-p12"])

# %% [markdown]
# ## 12. Ejercicios
#
# 1. **Quitar ruido.** Agrega ruido gaussiano con desviación 4 a los dígitos (`X + rng.normal(0, 4, X.shape)`), ajusta PCA con el 50 % de la varianza sobre los datos ruidosos y reconstruye. Dibuja algunos dígitos originales, ruidosos y reconstruidos. ¿Por qué funciona?
# 2. **PCA como preprocesamiento.** Con validación cruzada de 5 particiones, compara la exactitud de un clasificador de 5 vecinos más cercanos (`KNeighborsClassifier`) sobre los 64 píxeles contra un `Pipeline` de PCA con 10, 20 y 40 componentes seguido del mismo clasificador. ¿Cuántas componentes necesitas para no perder exactitud?
# 3. **La semilla de t-SNE.** Corre t-SNE con `init="random"` y tres semillas distintas. ¿Qué cambia entre las tres gráficas y qué se mantiene?
# 4. **Parámetros de UMAP.** Prueba `n_neighbors` en 5, 15 y 100, y `min_dist` en 0.0 y 0.8. Describe con tus palabras el efecto de cada uno.
# 5. **Pingüinos en kilogramos.** Repite la sección 6 cambiando la masa a kilogramos sin estandarizar. ¿Qué columna domina ahora la PC1? ¿Qué te dice esto sobre PCA sin estandarizar?
