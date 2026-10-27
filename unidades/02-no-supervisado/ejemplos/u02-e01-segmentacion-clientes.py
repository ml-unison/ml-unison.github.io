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
# # Segmentar clientes con k-medias, jerárquico y DBSCAN
#
# **Unidad 2 · Ejemplo 1**
#
# Hasta ahora siempre teníamos una columna que predecir. En esta libreta no: tenemos miles de clientes de una tienda y ninguna etiqueta que diga «este cliente es de tal tipo». Queremos que los datos mismos sugieran grupos. Eso es **agrupamiento** (*clustering*), la tarea más conocida del aprendizaje no supervisado.
#
# El recorrido:
#
# 1. de facturas a clientes: construir una tabla con tres números por cliente;
# 2. k-medias, primero escrito a mano en pocas líneas y luego con scikit-learn;
# 3. sus trampas: mínimos locales y la elección del número de grupos;
# 4. interpretar los grupos (y no creérnoslos demasiado);
# 5. dos alternativas con ideas distintas: el agrupamiento jerárquico y DBSCAN.

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

from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import rdp
from rdp.estilo import AZUL, ROJO, DORADO_OSC, GRIS, TENUE, TINTA

rdp.estilo.usar()
SEMILLA = 2011
COLORES = [GRIS, DORADO_OSC, ROJO, AZUL]  # un color por segmento (ver sección 6)

# %% [markdown]
# ## 1. El problema y los datos
#
# **Problema.** El área de mercadotecnia de una tienda en línea quiere entender **qué tipos de clientes tiene** para diseñar campañas distintas: no tiene sentido mandarle el mismo correo a quien compra cada semana que a quien compró una vez hace un año. Nadie ha clasificado a los clientes antes, así que no hay etiquetas.
#
# Conviene decirlo desde el principio: aquí **no hay una respuesta correcta** contra la cual medir. El objetivo es encontrar grupos que sean **útiles** (que sugieran acciones distintas) e **interpretables** (que se puedan describir en una frase). Eso cambia la forma de trabajar: más que optimizar una métrica, vamos a mirar, comparar y justificar decisiones.
#
# **Datos.** El conjunto *Online Retail* que Daqing Chen publicó en el UCI Machine Learning Repository (2015, licencia CC BY 4.0). Son todas las transacciones, de diciembre de 2010 a diciembre de 2011, de una tienda en línea del Reino Unido que vende artículos de regalo; muchos de sus clientes son **mayoristas** que revenden. El original tiene una fila por artículo vendido (más de medio millón); para el curso lo agregamos a una fila por **factura**:
#
# | Columna | Significado |
# |---|---|
# | `factura` | número de factura; si empieza con «C» es una **cancelación** |
# | `cliente` | identificador del cliente (puede faltar) |
# | `fecha` | fecha y hora |
# | `lineas`, `piezas` | número de artículos distintos y de piezas |
# | `importe` | total en libras esterlinas (negativo en las cancelaciones) |
# | `pais` | país del cliente |

# %%
facturas = rdp.datos.cargar("tienda_facturas", parse_dates=["fecha"])
print(facturas.shape)
facturas.head()

# %% [markdown]
# ## 2. De facturas a clientes
#
# Queremos agrupar **clientes**, no facturas, así que hay que resumir la historia de cada cliente en unos cuantos números. En mercadotecnia se usa desde hace décadas un resumen muy simple llamado **RFM**:
#
# - **Recencia**: cuántos días han pasado desde su última compra (menos es mejor);
# - **Frecuencia**: cuántas veces ha comprado (número de facturas);
# - **Monto**: cuánto ha gastado en total.
#
# Tres números fáciles de explicar a cualquiera, y que juntos dicen mucho: un cliente que compró hace poco, seguido y mucho es muy distinto de uno que compró una vez hace diez meses.
#
# ### Limpieza: dos decisiones
#
# **Facturas sin cliente.**

# %%
sin_cliente = facturas["cliente"].isna()
print(f"facturas sin cliente: {sin_cliente.mean():.1%}")
print(f"cancelaciones:        {facturas['factura'].str.startswith('C').mean():.1%}")

# %% [markdown]
# El 14 % de las facturas no tiene cliente. Probablemente son compras sin registrarse, y no hay forma de saber cuáles corresponden a la misma persona. Para segmentar clientes no sirven: las quitamos.
#
# **Cancelaciones.** Una cancelación es una factura con importe negativo que deshace (total o parcialmente) una compra anterior. Tenemos opciones: ignorarlas, quitarlas junto con la compra original (difícil de emparejar) o simplemente **sumarlas**. Elegimos lo último: el monto de un cliente será su **importe neto**, lo que de verdad pagó. La frecuencia, en cambio, cuenta solo las facturas de compra, porque una cancelación no es una visita más.
#
# Algunos clientes quedan con monto neto ≤ 0: sus cancelaciones superan sus compras, típicamente porque cancelaron algo comprado antes de diciembre de 2010. No sabemos su historia completa, así que también los quitamos.
#
# ### La tabla RFM
#
# La recencia se mide respecto a una **fecha de corte**: el día siguiente a la última factura de los datos, como si hiciéramos el análisis esa mañana.

# %%
compras = facturas[~sin_cliente].copy()
compras["cliente"] = compras["cliente"].astype(int)
compras["es_compra"] = ~compras["factura"].str.startswith("C")

corte = compras["fecha"].max().normalize() + pd.Timedelta(days=1)
print("fecha de corte:", corte.date())

rfm = compras.groupby("cliente").agg(
    ultima=("fecha", "max"),
    frecuencia=("es_compra", "sum"),
    monto=("importe", "sum"),
)
rfm["recencia"] = (corte - rfm["ultima"]).dt.days
rfm = rfm.loc[(rfm["monto"] > 0) & (rfm["frecuencia"] > 0), ["recencia", "frecuencia", "monto"]]
print(f"{len(rfm):,} clientes")
rfm.describe().round(1)

# %% [markdown]
# Mira la diferencia entre la mediana y el máximo: el cliente típico ha comprado 2 veces y gastado unas 660 libras, pero hay uno con 210 facturas y otro con casi 280 mil libras. Son los mayoristas grandes.

# %%
rfm.sort_values("monto", ascending=False).head(5)

# %% [markdown]
# ### Distribuciones muy sesgadas: el logaritmo
#
# Como vimos en la Unidad 1, cuando unas pocas observaciones son cientos de veces más grandes que el resto, el logaritmo ayuda. Aquí además tiene sentido de negocio: la diferencia entre gastar 100 y 1,000 libras se parece más a la diferencia entre 1,000 y 10,000 que a la diferencia entre 1,000 y 1,900.

# %%
fig, axs = plt.subplots(2, 3, figsize=(11, 5.6))
transformadas = {
    "recencia": (np.log1p, "log(1 + días)"),
    "frecuencia": (np.log, "log(facturas)"),
    "monto": (np.log, "log(libras)"),
}
unidades = {"recencia": "días", "frecuencia": "facturas", "monto": "libras"}
for j, (col, (f, etiqueta)) in enumerate(transformadas.items()):
    axs[0, j].hist(rfm[col], bins=40, color=AZUL)
    axs[0, j].set(title=col.capitalize(), xlabel=unidades[col])
    axs[1, j].hist(f(rfm[col]), bins=40, color=AZUL)
    axs[1, j].set(title=f"{col.capitalize()} en logaritmo", xlabel=etiqueta)
fig.tight_layout()
plt.show()

# %% [markdown]
# El monto pasa de una cola enorme a algo casi simétrico. La frecuencia sigue sesgada, porque es un conteo y un tercio de los clientes compró una sola vez (de ahí las barras separadas: 1, 2, 3… facturas), pero su cola ya no es de cientos de veces la mediana.
#
# En la recencia el logaritmo es más una decisión que una necesidad: dice que la diferencia entre 0 y 30 días sin comprar importa mucho más que la diferencia entre 300 y 330, lo cual es razonable para una tienda. Usamos `log(1 + días)` porque hay clientes que compraron el mismo día del corte (recencia 0).

# %% [markdown]
# ### Escalas distintas: estandarizar
#
# Todos los métodos de esta libreta se basan en **distancias** entre clientes. Si no ponemos las tres variables en la misma escala, la que tenga números más grandes decide todo. Veámoslo con los datos sin transformar: la desviación estándar del monto es mil veces la de la frecuencia.

# %%
print(rfm.std().round(1))

km_crudo = KMeans(n_clusters=4, n_init=10, random_state=SEMILLA).fit(rfm)
rfm.groupby(km_crudo.labels_).agg(clientes=("monto", "size"),
                                  monto_min=("monto", "min"), monto_max=("monto", "max"),
                                  recencia_min=("recencia", "min"), recencia_max=("recencia", "max"))

# %% [markdown]
# Con los datos crudos, los grupos son simplemente **rangos de monto**: un grupo con casi todos los clientes y unos cuantos grupos diminutos con los mayoristas enormes. La recencia va de 0 a más de 300 días en el grupo grande: el algoritmo ni la miró. No es lo que queremos.
#
# Así que trabajamos con logaritmos estandarizados (media 0, desviación 1). Aquí no hay conjunto de prueba, porque no estamos prediciendo nada: estandarizamos con todos los clientes.

# %%
logs = pd.DataFrame({col: f(rfm[col]) for col, (f, _) in transformadas.items()})
Z = StandardScaler().fit_transform(logs)
print(Z.shape, Z.mean(axis=0).round(2), Z.std(axis=0).round(2))

# %% [markdown]
# ## 3. k-medias paso a paso
#
# Para poder dibujar, empezamos con solo dos variables: frecuencia y monto (en logaritmo y estandarizadas).
#
# ### La idea
#
# Queremos repartir a los clientes en $k$ grupos y representar cada grupo con un punto, su **centroide** $\boldsymbol{\mu}_j$. Un buen agrupamiento es uno en que cada cliente queda cerca del centroide de su grupo. Si $c(i)$ es el grupo del cliente $i$, k-medias busca minimizar la **inercia**: la suma de distancias al cuadrado de cada cliente a su centroide,
#
# $$
# J = \sum_{i=1}^{n} \left\lVert \mathbf{x}_i - \boldsymbol{\mu}_{c(i)} \right\rVert^2 .
# $$
#
# Encontrar el mínimo exacto es un problema NP-difícil, pero hay un método muy simple que funciona bien en la práctica, el **algoritmo de Lloyd**. Alterna dos pasos, cada uno de los cuales nunca aumenta $J$:
#
# 1. **Asignar**: con los centroides fijos, cada cliente se va al centroide más cercano. (Es la mejor asignación posible para esos centroides.)
# 2. **Actualizar**: con las asignaciones fijas, cada centroide se mueve al promedio de sus clientes. (El promedio es el punto que minimiza la suma de distancias al cuadrado.)
#
# Se repite hasta que nada cambia. Como $J$ baja en cada paso y hay un número finito de asignaciones posibles, el algoritmo siempre termina.

# %%
def lloyd(X, k, semilla=0, max_iter=100):
    """k-medias de Lloyd. Regresa la historia de centroides, asignaciones e inercias."""
    rng = np.random.default_rng(semilla)
    mu = X[rng.choice(len(X), size=k, replace=False)]            # k clientes al azar
    historia = []
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)   # n × k distancias²
        c = d2.argmin(axis=1)                                      # 1. asignar
        historia.append((mu.copy(), c, d2[np.arange(len(X)), c].sum()))
        nuevo = np.array([X[c == j].mean(axis=0) for j in range(k)])  # 2. actualizar
        if np.allclose(nuevo, mu):                                 # ya nada cambia
            break
        mu = nuevo
    return historia

X2 = Z[:, 1:]   # log frecuencia y log monto, estandarizadas
historia = lloyd(X2, k=4, semilla=9)
print(f"{len(historia)} iteraciones")
print("inercia:", [round(J) for _, _, J in historia[:8]], "…", round(historia[-1][2]))

# %% [markdown]
# La frecuencia es un conteo, así que muchos clientes caen exactamente en el mismo punto horizontal. Para dibujar les agregamos un poco de ruido horizontal (solo en la gráfica, no en los cálculos).

# %%
rng = np.random.default_rng(SEMILLA)
X2_dibujo = X2 + np.column_stack([rng.normal(0, 0.04, len(X2)), np.zeros(len(X2))])

fig, axs = plt.subplots(1, 4, figsize=(13, 3.9), sharex=True, sharey=True)
for ax, t in zip(axs, [0, 1, 3, len(historia) - 1]):
    mu, c, J = historia[t]
    for j in range(4):
        ax.scatter(*X2_dibujo[c == j].T, s=5, alpha=0.35, color=COLORES[j], edgecolors="none")
    # trayectoria del centroide hasta esta iteración
    trayectoria = np.array([h[0] for h in historia[: t + 1]])
    for j in range(4):
        ax.plot(trayectoria[:, j, 0], trayectoria[:, j, 1], color=TINTA, lw=1, alpha=0.7)
    ax.scatter(mu[:, 0], mu[:, 1], s=150, marker="X", color=COLORES, edgecolors=TINTA, linewidths=1.2, zorder=3)
    ax.set(title=f"Iteración {t}" + (" (final)" if t == len(historia) - 1 else ""), xlabel="log frecuencia (est.)")
    ax.text(0.97, 0.04, f"J = {J:,.0f}", transform=ax.transAxes, ha="right", color=TENUE)
axs[0].set_ylabel("log monto (est.)")
fig.tight_layout()
plt.show()

# %% [markdown]
# En la iteración 0 los centroides son cuatro clientes elegidos al azar y la partición es mala. Las primeras iteraciones mueven mucho los centroides (las ✕, con su trayectoria en negro); después solo se ajustan poco a poco hasta que nada cambia. La inercia lo confirma: cae mucho al principio y luego casi nada.

# %%
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot([J for _, _, J in historia], marker="o", color=AZUL)
ax.set(title="La inercia nunca aumenta", xlabel="iteración", ylabel="inercia J")
plt.show()

# %% [markdown]
# ### Con scikit-learn
#
# `KMeans` hace lo mismo, con una inicialización más cuidadosa (la veremos en un momento) y repitiendo varias veces. La inercia final queda en `inertia_` y los centroides en `cluster_centers_`.

# %%
km2 = KMeans(n_clusters=4, n_init=10, random_state=SEMILLA).fit(X2)
print(f"nuestra versión: J = {historia[-1][2]:,.1f}  ({len(historia)} iteraciones)")
print(f"scikit-learn:    J = {km2.inertia_:,.1f}")

# %% [markdown]
# Las dos llegan a la misma inercia: nuestras quince líneas de numpy son, en esencia, k-medias.

# %% [markdown]
# ## 4. Mínimos locales
#
# Lloyd garantiza que $J$ baja, pero no que llegue al **mínimo global**: se detiene en el primer lugar donde ningún paso mejora. Dónde se detiene depende de los centroides iniciales. Con $k = 4$ estos datos son benignos, pero con $k = 6$ ya se nota. Corramos k-medias veinte veces, cada una con una sola inicialización al azar (`init="random"`, `n_init=1`):

# %%
semillas = range(20)
J_azar = [KMeans(6, init="random", n_init=1, random_state=s).fit(X2).inertia_ for s in semillas]
J_mas = [KMeans(6, init="k-means++", n_init=1, random_state=s).fit(X2).inertia_ for s in semillas]

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.scatter(J_azar, np.full(20, 1) + rng.uniform(-0.12, 0.12, 20), s=45, color=ROJO)
ax.scatter(J_mas, np.full(20, 0) + rng.uniform(-0.12, 0.12, 20), s=45, color=AZUL)
ax.set(title="Inercia final de 20 corridas con k = 6", xlabel="inercia J (menos es mejor)",
       yticks=[0, 1], yticklabels=["k-medias++", "al azar"], ylim=(-0.6, 1.6))
ax.grid(axis="y", visible=False)
plt.show()
print(f"al azar:     mín {min(J_azar):,.0f} · máx {max(J_azar):,.0f}")
print(f"k-medias++:  mín {min(J_mas):,.0f} · máx {max(J_mas):,.0f}")

# %% [markdown]
# Las corridas terminan en soluciones distintas: la mayoría cerca de la mejor, pero algunas claramente peores (≈ 10 % más de inercia). Son **mínimos locales**: particiones en las que ningún paso de Lloyd mejora, aunque exista otra mejor. Dos remedios, que scikit-learn usa por omisión:
#
# - **k-medias++**: en lugar de elegir los centroides iniciales totalmente al azar, elige el primero al azar y cada siguiente con probabilidad proporcional al cuadrado de su distancia al centroide más cercano ya elegido. Así los centroides iniciales quedan repartidos. Aquí mejora un poco las cosas (menos corridas malas), pero no las elimina.
# - **Varias inicializaciones** (`n_init`): correr el algoritmo varias veces y quedarse con la de menor inercia. Es barato y es lo que de verdad nos protege.
#
# Moraleja: si alguna vez ves `n_init=1`, desconfía.

# %% [markdown]
# ## 5. ¿Cuántos grupos?
#
# Ahora usamos las tres variables RFM. k-medias necesita que le digamos $k$, y los datos no traen escrita la respuesta. Hay dos herramientas comunes.
#
# **La curva de inercia («el codo»).** La inercia siempre baja al aumentar $k$ (con $k = n$ es cero), así que no sirve minimizarla. Se busca el punto donde agregar un grupo deja de ayudar mucho.
#
# **La silueta.** Para cada cliente $i$ se calcula $a(i)$, la distancia promedio a los de su propio grupo, y $b(i)$, la distancia promedio a los del grupo vecino más cercano:
#
# $$
# s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}} \in [-1, 1].
# $$
#
# Cerca de 1, el cliente está bien metido en su grupo; cerca de 0, está en la frontera; negativo, quizá está en el grupo equivocado. La **silueta media** resume todo el agrupamiento.

# %%
ks = range(2, 9)
inercias, siluetas = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Z)
    inercias.append(km.inertia_)
    siluetas.append(silhouette_score(Z, km.labels_))

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].plot(ks, inercias, marker="o", color=AZUL)
axs[0].set(title="Inercia", xlabel="número de grupos k")
axs[1].plot(ks, siluetas, marker="o", color=AZUL)
axs[1].set(title="Silueta media (más es mejor)", xlabel="número de grupos k")
axs[1].scatter([4], [siluetas[2]], s=160, facecolors="none", edgecolors=ROJO, linewidths=2, zorder=3)
fig.tight_layout()
plt.show()
pd.DataFrame({"k": ks, "inercia": np.round(inercias), "silueta": np.round(siluetas, 3)}).set_index("k").T

# %% [markdown]
# Ninguna de las dos gráficas da una respuesta tajante:
#
# - la curva de inercia baja suavemente, sin un codo claro;
# - la silueta es mayor con $k = 2$, y luego casi igual para $k = 3$ y $k = 4$.
#
# Siluetas alrededor de 0.3–0.4 dicen algo importante: **los clientes no forman grupos bien separados**, sino una nube continua. k-medias siempre devuelve $k$ grupos, haya o no grupos «de verdad»; en este caso está **partiendo** la nube en regiones, no descubriendo islas.
#
# ¿Entonces $k = 2$? Dos grupos (más o menos «activos» y «no activos») son poco útiles para mercadotecnia: se sabían sin ningún algoritmo. Elegimos **$k = 4$**, con una silueta prácticamente igual a la de $k = 3$ y que, como veremos, produce grupos fáciles de describir y con acciones distintas. Es una decisión razonable, no la verdad: la métrica orienta, pero el criterio final es la utilidad.

# %%
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=SEMILLA).fit(Z)

# Renumeramos los grupos por monto mediano, para que el orden (y los colores) no dependa del azar
orden = rfm.groupby(km.labels_)["monto"].median().sort_values().index
renumerar = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
rfm["grupo"] = pd.Series(km.labels_, index=rfm.index).map(renumerar)
centroides = km.cluster_centers_[orden]

# %% [markdown]
# ## 6. Interpretar los segmentos
#
# Los centroides están en logaritmos estandarizados, difíciles de leer. Es mejor resumir cada grupo en las unidades originales, con **medianas** (las medias se las llevan los mayoristas enormes).

# %%
resumen = rfm.groupby("grupo").agg(
    clientes=("monto", "size"),
    recencia=("recencia", "median"),
    frecuencia=("frecuencia", "median"),
    monto=("monto", "median"),
)
resumen["% clientes"] = (100 * resumen["clientes"] / len(rfm)).round(1)
resumen["% del monto"] = (100 * rfm.groupby("grupo")["monto"].sum() / rfm["monto"].sum()).round(1)
resumen.round(0)

# %% [markdown]
# Con esta tabla podemos ponerles nombres **tentativos**:

# %%
nombres = {
    0: "dormidos",             # última compra hace ~6 meses, una sola vez, poco dinero
    1: "nuevos u ocasionales", # compraron hace poco, pero pocas veces y poco
    2: "regulares",            # varias compras, gasto medio, algo de tiempo sin volver
    3: "leales de alto valor", # compran seguido, gastan mucho y volvieron hace días
}
resumen.index = resumen.index.map(nombres)
resumen.round(0)

# %%
# Ruido pequeño en los conteos (frecuencia y recencia), solo para dibujar
Zd = Z + np.column_stack([rng.normal(0, 0.03, len(Z)), rng.normal(0, 0.04, len(Z)), np.zeros(len(Z))])
grupo = rfm["grupo"].to_numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (x, etiqueta_x) in zip(axs, [(1, "log frecuencia (estandarizada)"), (0, "log recencia (estandarizada)")]):
    for g in range(K):
        m = grupo == g
        ax.scatter(Zd[m, x], Zd[m, 2], s=6, alpha=0.35, color=COLORES[g], edgecolors="none")
    for g in range(K):
        ax.scatter(centroides[g, x], centroides[g, 2], s=220, marker="X", color=COLORES[g],
                   edgecolors=TINTA, linewidths=1.3, zorder=3)
    ax.set(xlabel=etiqueta_x)
axs[0].set(ylabel="log monto (estandarizado)", title="Frecuencia contra monto")
axs[1].set(title="Recencia contra monto")
# nombres junto a los centroides, en el panel de recencia, donde los cuatro se distinguen
desplazamientos = {0: (-20, -32), 1: (-30, -32), 2: (12, 12), 3: (12, 12)}
for g in range(K):
    axs[1].annotate(nombres[g], centroides[g, [0, 2]], xytext=desplazamientos[g], textcoords="offset points",
                    fontsize=12, color=TINTA, weight="semibold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
fig.suptitle("Cuatro segmentos de clientes (k-medias sobre R, F y M)", x=0.01, ha="left",
             color=rdp.estilo.MARINO, weight="semibold", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# Cada panel muestra dos de las tres variables. A la izquierda, «dormidos» y «nuevos u ocasionales» se enciman: compraron poco y pocas veces. A la derecha se ve qué los separa: la **recencia** (a la derecha están los que llevan más tiempo sin comprar).
#
# Algunas lecturas útiles para mercadotecnia:
#
# - Los **leales de alto valor** son menos de una quinta parte de los clientes, pero concentran la mayor parte del dinero. Perder a uno de ellos cuesta mucho: campañas de retención y trato preferente.
# - Los **nuevos u ocasionales** compraron hace poco: es el momento de invitarlos a volver.
# - A los **dormidos** quizá convenga una campaña de reactivación barata, o ninguna.
#
# **Cuidado con sobreinterpretar.** Los nombres los pusimos nosotros a partir de medianas; dentro de cada grupo hay mucha variación, y vimos que las fronteras entre grupos son arbitrarias (la silueta es baja). Un cliente en la frontera entre «regulares» y «leales» no es esencialmente distinto de su vecino del otro lado. Además, con otra semilla, otro $k$ u otra transformación los grupos cambiarían. Los segmentos son un resumen conveniente de una nube continua, no tipos naturales de clientes.

# %% [markdown]
# ## 7. Agrupamiento jerárquico
#
# k-medias pide $k$ por adelantado. El **agrupamiento jerárquico aglomerativo** construye en cambio **todas** las particiones a la vez:
#
# 1. empieza con cada cliente en su propio grupo;
# 2. en cada paso une los dos grupos más cercanos;
# 3. termina cuando todo es un solo grupo.
#
# El resultado es un árbol, el **dendrograma**, que se puede cortar a la altura que queramos. Falta decidir qué significa «los dos grupos más cercanos». Esa regla se llama **enlace** (*linkage*):
#
# | Enlace | Distancia entre los grupos $A$ y $B$ | Tiende a producir |
# |---|---|---|
# | sencillo (*single*) | la del par más cercano, un punto de $A$ y uno de $B$ | cadenas largas; un grupo gigante y muchos sueltos |
# | completo (*complete*) | la del par más lejano | grupos compactos de tamaño parecido |
# | promedio (*average*) | el promedio de todas las distancias entre pares | un punto intermedio |
# | Ward | cuánto aumenta la inercia $J$ al unirlos | grupos compactos, parecidos a los de k-medias |
#
# El enlace de Ward es el que conecta con lo anterior: en cada paso hace la unión que menos empeora la misma función objetivo de k-medias.
#
# El dendrograma de 4,000 clientes sería ilegible, así que lo dibujamos para una muestra de 150.

# %%
muestra = np.random.default_rng(SEMILLA).choice(len(Z), size=150, replace=False)
set_link_color_palette([AZUL, ROJO, DORADO_OSC, GRIS])
ARRIBA = rdp.estilo.PALETA["gris"]  # color de las uniones por encima del corte

fig, axs = plt.subplots(2, 2, figsize=(12, 7))
for ax, metodo, titulo in zip(axs.flat, ["single", "complete", "average", "ward"],
                              ["Enlace sencillo", "Enlace completo", "Enlace promedio", "Ward"]):
    arbol = linkage(Z[muestra], method=metodo)
    corte4 = (arbol[-4, 2] + arbol[-3, 2]) / 2   # altura que deja 4 grupos
    dendrogram(arbol, ax=ax, no_labels=True, color_threshold=corte4, above_threshold_color=ARRIBA)
    ax.axhline(corte4, color=TINTA, ls="--", lw=1)
    ax.set(title=titulo, ylabel="distancia de unión")
    ax.grid(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# La línea punteada corta cada árbol en 4 grupos. Con enlace sencillo se ve el «encadenamiento»: los clientes se van pegando uno a uno a un grupo enorme, y los grupos que quedan son unos cuantos puntos aislados. Ward produce ramas equilibradas y una separación más clara entre las uniones de abajo y las de arriba.
#
# Ahora con todos los clientes. `AgglomerativeClustering` hace el mismo proceso y corta en el número de grupos que pidamos. Para comparar con k-medias usamos el **índice de Rand ajustado** (ARI): mide qué tanto coinciden dos particiones, contando los pares de clientes que ambas ponen juntos o ambas ponen separados, y corrigiendo por el azar. Vale 1 si las particiones son idénticas (aunque los números de grupo sean otros) y alrededor de 0 si coinciden lo que coincidirían al azar.

# %%
filas = []
for metodo in ["single", "complete", "average", "ward"]:
    etiquetas = AgglomerativeClustering(n_clusters=K, linkage=metodo).fit_predict(Z)
    tamanos = sorted(np.bincount(etiquetas), reverse=True)
    filas.append({"enlace": metodo, "tamaños de los grupos": tamanos,
                  "ARI contra k-medias": round(adjusted_rand_score(rfm["grupo"], etiquetas), 2)})
    if metodo == "ward":
        ward = etiquetas
pd.DataFrame(filas).set_index("enlace")

# %% [markdown]
# El enlace sencillo es inútil aquí: un grupo con casi todos los clientes y tres con uno o dos. Ward da grupos de tamaños razonables, y aun así coincide con k-medias **solo a medias**. ¿Dónde difieren?

# %%
pd.crosstab(rfm["grupo"].map(nombres), ward, rownames=["k-medias"], colnames=["Ward"])

# %% [markdown]
# Los dos métodos coinciden en lo grueso: Ward nunca pone en un mismo grupo a «dormidos» y «leales». Pero cortan de forma distinta: Ward parte a los dormidos y a los leales en dos pedazos cada uno y reparte a los regulares entre tres grupos. Es lo esperable cuando los datos son una nube continua: hay muchas formas casi igual de buenas de partirla, y cada algoritmo elige una. Si dos métodos razonables no coinciden, eso nos dice que las fronteras de los grupos **no son una propiedad de los datos**, sino de nuestras decisiones.

# %% [markdown]
# ## 8. DBSCAN: grupos por densidad
#
# DBSCAN parte de otra idea: un grupo es una **región densa** de puntos, separada de otras por regiones vacías. Tiene dos parámetros:
#
# - `eps` ($\varepsilon$): el radio de la vecindad de cada punto;
# - `min_samples`: cuántos puntos debe haber en esa vecindad para que el punto sea **central**.
#
# Los puntos centrales que están a menos de $\varepsilon$ unos de otros forman un mismo grupo, junto con los puntos de su vecindad. Lo que no queda en ningún grupo es **ruido**. No hay que decir cuántos grupos; DBSCAN los encuentra (o no) según la densidad.
#
# Para elegir $\varepsilon$ se usa una gráfica sencilla: para cada cliente, la distancia a su vecino número `min_samples`, ordenadas de menor a mayor. Los puntos dentro de regiones densas tienen esa distancia chica; los aislados, grande. Se busca el «codo» donde la curva se dispara. (`NearestNeighbors` cuenta al propio cliente como su primer vecino, igual que DBSCAN lo cuenta en `min_samples`; por eso usamos los mismos 10.)

# %%
MIN_PUNTOS = 10
distancias, _ = NearestNeighbors(n_neighbors=MIN_PUNTOS).fit(Z).kneighbors(Z)
d_k = np.sort(distancias[:, -1])
EPS = 0.5

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(d_k, color=AZUL)
ax.axhline(EPS, color=ROJO, ls="--", lw=1.2)
ax.text(50, EPS + 0.05, f"ε = {EPS}", color=ROJO)
ax.set(title=f"Distancia de cada cliente a su vecino número {MIN_PUNTOS}",
       xlabel="clientes, ordenados", ylabel="distancia")
plt.show()

# %% [markdown]
# La curva crece despacio para la gran mayoría y se dispara en los últimos cien o doscientos clientes. Tomamos $\varepsilon = 0.5$, cerca de donde empieza a subir.

# %%
db = DBSCAN(eps=EPS, min_samples=MIN_PUNTOS).fit(Z)
rfm["dbscan"] = db.labels_
print(f"grupos encontrados: {db.labels_.max() + 1} · clientes marcados como ruido: {(db.labels_ == -1).sum()}")
rfm.groupby("dbscan").agg(clientes=("monto", "size"), recencia=("recencia", "median"),
                          frecuencia=("frecuencia", "median"), monto=("monto", "median")).round(0)

# %%
fig, ax = plt.subplots(figsize=(8, 5.2))
estilos = {0: (AZUL, "grupo 0"), 1: (GRIS, "grupo 1"), -1: (ROJO, "ruido")}
for g, (color, nombre) in estilos.items():
    m = db.labels_ == g
    ax.scatter(Zd[m, 1], Zd[m, 2], s=7 if g >= 0 else 28, alpha=0.4 if g >= 0 else 0.9,
               color=color, edgecolors="none", label=f"{nombre} ({m.sum():,})")
ax.legend(loc="lower right", markerscale=2)
ax.set(title=f"DBSCAN (ε = {EPS}, min_samples = {MIN_PUNTOS})",
       xlabel="log frecuencia (estandarizada)", ylabel="log monto (estandarizado)")
plt.show()

# %% [markdown]
# DBSCAN cuenta una historia distinta. Encuentra solo **dos** grupos densos. ¿Qué los separa? Veamos la frecuencia:

# %%
pd.crosstab(rfm["dbscan"], rfm["frecuencia"].clip(upper=3).map({1: "1", 2: "2", 3: "3 o más"}),
            rownames=["DBSCAN"], colnames=["facturas"])

# %% [markdown]
# El grupo 1 son exactamente los clientes que compraron **una sola vez**, y el grupo 0 todos los demás. No es un descubrimiento profundo: como la frecuencia es un conteo, entre 1 y 2 facturas hay un hueco en la escala logarítmica (≈ 0.77 desviaciones estándar) más ancho que $\varepsilon$, y DBSCAN lo interpreta como una frontera entre regiones densas. Es un buen recordatorio de que los algoritmos encuentran **cualquier** estructura que haya en la representación, incluidas las que le pusimos nosotros. (También explica en parte por qué la silueta prefería $k = 2$.)
#
# Además, DBSCAN marca como **ruido** a los clientes de las orillas de la nube. Entre ellos están los mayoristas más grandes (los de cientos de miles de libras), pero también algunos clientes raros en el otro extremo (montos de unas cuantas libras).

# %%
rfm[rfm["dbscan"] == -1].sort_values("monto", ascending=False).head(8)[["recencia", "frecuencia", "monto"]]

# %% [markdown]
# ¿Es bueno o malo? Depende de para qué:
#
# - Si queremos **detectar atípicos**, es justo lo que buscamos: una lista corta de clientes que no se parecen a nadie y merecen atención individual (y los mayoristas grandes la merecen: son una fracción importante de las ventas).
# - Si queremos **asignar cada cliente a una campaña**, es un problema: los clientes más valiosos de la tienda se quedaron sin grupo. Y los dos grupos que sí encontró («una compra» y «más de una») no necesitaban ningún algoritmo.
#
# DBSCAN brilla cuando hay grupos densos separados por huecos, con formas arbitrarias. En una nube continua como esta, su respuesta honesta es «aquí casi no hay grupos».

# %% [markdown]
# ## 9. Lo que aprendimos
#
# - **La representación importa más que el algoritmo.** Pasar de facturas a RFM, tomar logaritmos y estandarizar decidió qué significa «parecido». Sin escalar, el monto se comía todo.
# - **k-medias** es rápido, sencillo y da grupos compactos que se resumen con un centroide. Pide $k$, cae en mínimos locales (usa `n_init`) y supone grupos más o menos redondos y de tamaño parecido. Siempre parte los datos en $k$ pedazos, aunque no haya grupos.
# - **El jerárquico** no pide $k$ por adelantado y el dendrograma muestra la estructura a todas las escalas. Ward se parece a k-medias; el enlace sencillo encadena. Su costo crece al menos con $n^2$, así que con muchos datos se vuelve lento.
# - **DBSCAN** encuentra grupos de cualquier forma, decide cuántos hay y separa el ruido. Es sensible a $\varepsilon$ y sufre cuando las densidades varían mucho.
# - **No hay una respuesta correcta.** La silueta y el codo orientan; la decisión final depende de para qué se quieren los grupos. Comparar métodos (con el ARI, por ejemplo) ayuda a ver qué tan estables son las conclusiones.
#
# ### Una nota ética
#
# Aquí segmentamos para mandar correos distintos, algo de bajo riesgo. Pero las mismas técnicas se usan para decidir **precios, crédito o seguros**, y ahí pueden discriminar: si los grupos se correlacionan con el código postal, la edad o el origen, tratar distinto a cada grupo puede ser tratar distinto a personas por características que no deberían importar, aunque nunca hayamos usado esas variables directamente. Además, poner nombre a un grupo («dormidos», «de bajo valor») invita a pensar que describe a las personas que contiene, y el nombre se vuelve estereotipo. Los grupos son una herramienta para resumir datos, no una verdad sobre las personas.

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u02", ids=["u02-p01", "u02-p02", "u02-p03", "u02-p04", "u02-p05", "u02-p06"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. Repite el agrupamiento **sin** tomar logaritmos (pero estandarizando). ¿Qué pasa con los mayoristas grandes? ¿Y con el resto de los clientes?
# 2. Usa $k = 3$ y $k = 5$. Describe los grupos con la tabla de medianas. ¿Cuál de los tres valores de $k$ le propondrías a mercadotecnia y por qué?
# 3. Agrega una cuarta variable: el importe promedio por factura (monto / frecuencia). ¿Aparece algún grupo nuevo con interpretación clara, por ejemplo «pocas compras, pero enormes»?
# 4. En DBSCAN, prueba $\varepsilon = 0.3$ y $\varepsilon = 0.7$. ¿Cuántos grupos y cuánto ruido obtienes? ¿Hay algún valor que dé grupos útiles para campañas?
# 5. Compara la silueta media de k-medias y de Ward con $k = 4$. ¿Cuál «gana»? ¿Cambia eso cuál usarías?
