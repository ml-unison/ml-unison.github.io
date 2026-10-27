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
# # Árboles de decisión: de los pingüinos a las bicicletas
#
# **Unidad 5 · Ejemplo 1**
#
# En el juego de las «20 preguntas» se adivina un objeto haciendo preguntas de sí o no, y un buen jugador elige primero las preguntas que más descartan. Un **árbol de decisión** es exactamente eso: una sucesión de preguntas sencillas sobre los atributos («¿el pico mide más de 42 mm?»), donde cada respuesta decide la siguiente pregunta, hasta llegar a una predicción. Hasta ahora todos nuestros modelos eran lineales en algún espacio; los árboles no lo son, no necesitan escalar los datos, y un árbol pequeño se puede leer de una sola mirada. En esta libreta vamos a:
#
# 1. ver cómo un árbol parte el espacio de atributos en **rectángulos**, con los pingüinos de la Unidad 2;
# 2. definir la **impureza** de un nodo (Gini, entropía, error de clasificación) y entender por qué el error es una mala guía para crecer el árbol;
# 3. programar **desde cero** la búsqueda de la mejor división y la construcción recursiva de un árbol de clasificación, y compararlo con `DecisionTreeClassifier`;
# 4. usar **árboles de regresión** para predecir las rentas de bicicletas de la Unidad 4 y ver cómo el árbol encuentra solo la interacción hora × día laboral que allá construimos a mano;
# 5. controlar el **sobreajuste** con la **poda por costo–complejidad** y validación cruzada;
# 6. medir la **inestabilidad** de los árboles, que motiva los ensambles de la siguiente libreta.

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
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score, cross_validate
from sklearn.metrics import root_mean_squared_error, accuracy_score

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE
COLORES = [AZUL, ROJO, DORADO_OSC]                          # Adelie, Chinstrap, Gentoo
REGIONES = ListedColormap(["#C0CEE3", "#F3C7C4", "#EFDDAA"])  # las mismas tintas de las presentaciones

# %% [markdown]
# # Parte 1. Árboles de clasificación: los pingüinos
#
# ## 1. Los datos y el problema
#
# **Datos.** Son los pingüinos del archipiélago Palmer, en la Antártida, que usamos en las unidades 2 y 4: 344 aves de tres especies (Adelie, Chinstrap y Gentoo) medidas por Kristen Gorman y el programa de investigación ecológica de largo plazo de la estación Palmer, entre 2007 y 2009 (paquete `palmerpenguins` de A. Horst, A. Hill y K. Gorman, 2020; licencia CC0). Dos aves no tienen medidas y las descartamos.
#
# **Problema.** Identificar la especie de un pingüino a partir de las dos medidas del pico, el **largo** y la **profundidad** (en milímetros). Usamos solo dos atributos para poder dibujar todo lo que hace el árbol; al final agregaremos los demás.

# %%
pinguinos = rdp.datos.cargar("pinguinos").dropna(subset=["bill_length_mm", "bill_depth_mm"])
pico = ["bill_length_mm", "bill_depth_mm"]
ESPECIES = ["Adelie", "Chinstrap", "Gentoo"]
Xp = pinguinos[pico].to_numpy()
yp = pinguinos["species"].map({e: k for k, e in enumerate(ESPECIES)}).to_numpy()
print(f"{len(yp)} pingüinos:", dict(zip(ESPECIES, np.bincount(yp))))

# %% [markdown]
# ## 2. Un árbol es una sucesión de preguntas
#
# Empecemos con lo que hace scikit-learn y después lo programaremos nosotros. Un árbol de profundidad 2 hace a lo más dos preguntas por pingüino:

# %%
arbol2 = DecisionTreeClassifier(max_depth=2, random_state=SEMILLA).fit(Xp, yp)
print(export_text(arbol2, feature_names=["largo", "profundidad"], class_names=ESPECIES, show_weights=True))

# %% [markdown]
# Se lee de arriba abajo. La primera pregunta es si el pico mide 42.35 mm o menos de largo. Si sí (lado izquierdo), se pregunta por la profundidad: con más de 15.1 mm el pingüino es casi seguro Adelie (`weights` cuenta cuántos de cada especie llegan a cada hoja: 139 Adelie y un Chinstrap). Si el pico es largo, un pico poco profundo (16.35 mm o menos) es de Gentoo, sin excepción, y uno profundo es casi siempre de Chinstrap. Con dos preguntas el árbol ya acierta en el 94 % de los casos:

# %%
print(f"exactitud en entrenamiento con profundidad 2: {arbol2.score(Xp, yp):.3f}")

# %% [markdown]
# La misma información, dibujada con `plot_tree`. Cada **nodo interno** tiene una pregunta de la forma $x_j \le t$; cada **hoja** tiene una predicción: la especie más frecuente entre los pingüinos de entrenamiento que llegaron a ella.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
plot_tree(arbol2, feature_names=["largo del pico", "profundidad del pico"], class_names=ESPECIES,
          filled=False, impurity=True, proportion=False, rounded=True, fontsize=11, ax=ax)
ax.set_title("Árbol de profundidad 2 para las tres especies")
plt.show()

# %% [markdown]
# ### Las regiones son rectángulos
#
# Cada pregunta corta el plano con una recta **paralela a un eje**. Después de varias preguntas, cada hoja corresponde a un rectángulo (una caja, en más dimensiones), y en cada rectángulo la predicción es constante. Así se ve cómo crece la partición al permitir árboles más profundos:

# %%
g1, g2 = np.meshgrid(np.linspace(31, 61, 400), np.linspace(12.5, 22, 300))
malla = np.column_stack([g1.ravel(), g2.ravel()])


def dibujar_regiones(ax, modelo, titulo, puntos=True):
    """Regiones de decisión de un clasificador de pingüinos en el plano largo–profundidad del pico."""
    zona = modelo.predict(malla).reshape(g1.shape)
    ax.pcolormesh(g1, g2, zona, cmap=REGIONES, vmin=-0.5, vmax=2.5, shading="auto", rasterized=True)
    if puntos:
        for k in range(3):
            ax.scatter(Xp[yp == k, 0], Xp[yp == k, 1], s=16, color=COLORES[k], lw=0.4)
    ax.set_title(titulo, fontsize=12)
    ax.set_xlim(31, 61)
    ax.set_ylim(12.5, 22)
    ax.grid(False)


fig, ejes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
for ax, prof in zip(ejes.flat, [1, 2, 3, 4, 6, None]):
    m = DecisionTreeClassifier(max_depth=prof, random_state=SEMILLA).fit(Xp, yp)
    nombre = "sin límite" if prof is None else prof
    dibujar_regiones(ax, m, f"profundidad {nombre}: {m.get_n_leaves()} hojas, exactitud {m.score(Xp, yp):.3f}")
for ax in ejes[1]:
    ax.set_xlabel("largo del pico (mm)")
for ax in ejes[:, 0]:
    ax.set_ylabel("profundidad del pico (mm)")
for k, e in enumerate(ESPECIES):
    ejes[0, 0].text([34, 50, 51][k], [21.3, 21.3, 13.1][k], e, color=TINTA, fontweight="bold", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# Con una pregunta el árbol solo separa dos grupos (y nunca predice Chinstrap). Con dos o tres ya recupera las tres especies. A partir de ahí, cada nivel nuevo agrega rectángulos cada vez más pequeños para atrapar pingüinos individuales que quedaron del lado equivocado, y el árbol sin límite de profundidad acierta en **todos** los de entrenamiento, con rectángulos diminutos alrededor de un solo pingüino: el sobreajuste de la Unidad 3, ahora con forma geométrica. La validación cruzada lo confirma:

# %%
pliegues = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
filas = []
for prof in [1, 2, 3, 4, 5, 6, 8, None]:
    m = DecisionTreeClassifier(max_depth=prof, random_state=SEMILLA)
    cv = cross_validate(m, Xp, yp, cv=pliegues, return_train_score=True)
    filas.append({"profundidad": "sin límite" if prof is None else prof,
                  "exactitud (entrenamiento)": cv["train_score"].mean(),
                  "exactitud (validación cruzada)": cv["test_score"].mean()})
pd.DataFrame(filas).set_index("profundidad").round(3)

# %% [markdown]
# La exactitud de entrenamiento sube hasta 1; la de validación cruzada sube rápido hasta la profundidad 3 o 4 y después casi no cambia (de 0.939 a 0.947): los rectángulos nuevos ya no ayudan a clasificar pingüinos que el árbol no ha visto. ¿Cómo eligió el árbol sus preguntas? Para eso necesitamos medir qué tan «revuelto» está un nodo.

# %% [markdown]
# ## 3. Impureza: cuánto falta para que un nodo sea puro
#
# Sea un nodo con $n$ datos y $p_k$ la proporción de la clase $k$ entre ellos. Un nodo es **puro** si todos sus datos son de la misma clase. Tres medidas de **impureza** comunes, todas iguales a cero en un nodo puro y máximas cuando las clases están en iguales proporciones:
#
# $$\text{Gini: } \; G = \sum_k p_k(1 - p_k) = 1 - \sum_k p_k^2, \qquad
# \text{entropía: } \; H = -\sum_k p_k \log_2 p_k, \qquad
# \text{error: } \; E = 1 - \max_k p_k .$$
#
# **La ganancia de una división.** Si una pregunta parte el nodo en una hija izquierda con $n_I$ datos y una derecha con $n_D$, su **ganancia** (con la entropía se le llama **ganancia de información**) es lo que baja la impureza promedio:
#
# $$\Delta = i(\text{padre}) - \Big[\frac{n_I}{n}\, i(\text{izquierda}) + \frac{n_D}{n}\, i(\text{derecha})\Big].$$
#
# ### Dos lecturas del índice de Gini
#
# 1. **Probabilidad de error de un clasificador al azar.** Tomemos un dato al azar del nodo (su clase es $k$ con probabilidad $p_k$) y le asignamos una etiqueta también al azar, con las mismas proporciones del nodo. Se equivoca con probabilidad $\sum_k p_k(1 - p_k) = G$.
# 2. **Varianza de las indicadoras.** Si $Z_k = \mathbb{1}[y = k]$ para un dato al azar del nodo, $Z_k$ es una variable de Bernoulli con $\mathrm{Var}(Z_k) = p_k(1 - p_k)$; entonces $G = \sum_k \mathrm{Var}(Z_k)$. Esto anticipa el árbol de regresión: allá la impureza será la varianza de $y$ en el nodo.
#
# ### Por qué el error de clasificación es mala guía
#
# Un ejemplo: un nodo con 80 pingüinos Adelie y 20 Chinstrap. Comparemos dos divisiones posibles.

# %%
def gini(conteos):
    """Índice de Gini de uno o varios nodos (conteos por clase en el último eje)."""
    conteos = np.asarray(conteos, dtype=float)
    p = conteos / np.maximum(conteos.sum(-1, keepdims=True), 1)
    return 1 - (p**2).sum(-1)


def entropia(conteos):
    conteos = np.asarray(conteos, dtype=float)
    p = conteos / np.maximum(conteos.sum(-1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0).sum(-1)


def error_clasif(conteos):
    conteos = np.asarray(conteos, dtype=float)
    return 1 - conteos.max(-1) / np.maximum(conteos.sum(-1), 1)


def ganancia(impureza, padre, izq, der):
    n_i, n_d = np.sum(izq), np.sum(der)
    return impureza(padre) - (n_i * impureza(izq) + n_d * impureza(der)) / (n_i + n_d)


padre = [80, 20]
divisiones = {"A: (40, 20) | (40, 0)": ([40, 20], [40, 0]),
              "B: (70, 10) | (10, 10)": ([70, 10], [10, 10])}
pd.DataFrame({nombre: {f: ganancia(fn, padre, i, d) for f, fn in
                       [("error", error_clasif), ("Gini", gini), ("entropía", entropia)]}
              for nombre, (i, d) in divisiones.items()}).T.round(4)

# %% [markdown]
# Con el error de clasificación, **ninguna** de las dos divisiones gana nada: en ambas, las dos hijas siguen siendo de mayoría Adelie y se equivocan en los mismos 20 Chinstrap que el padre. Pero la división A es claramente útil: aísla 40 Adelie en un nodo puro, que ya no hay que tocar, y deja el problema concentrado en la otra hija; un nivel más abajo se podría terminar el trabajo. Gini y entropía sí lo reconocen.
#
# **La razón es geométrica.** Con dos clases, sea $p$ la proporción de la clase 2. Las proporciones del padre son el promedio ponderado de las de las hijas: $p = \frac{n_I}{n}p_I + \frac{n_D}{n}p_D$. Si la impureza $i(p)$ es una función **estrictamente cóncava** (Gini: $2p(1-p)$; entropía), la desigualdad de Jensen dice
#
# $$\frac{n_I}{n}\, i(p_I) + \frac{n_D}{n}\, i(p_D) \;\le\; i\Big(\frac{n_I}{n}p_I + \frac{n_D}{n}p_D\Big) = i(p),$$
#
# con igualdad solo si $p_I = p_D$. Es decir, **toda** división que cambie las proporciones tiene ganancia positiva. El error $\min(p, 1-p)$ es cóncavo pero **lineal a trozos**: si $p_I$ y $p_D$ caen del mismo lado de $1/2$, el promedio cae sobre el mismo segmento recto y la ganancia es exactamente cero. Lo vemos en la gráfica: el padre está sobre la curva, y la impureza promedio de las hijas es el punto de la **cuerda** que une a las dos hijas, justo debajo del padre.

# %%
p = np.linspace(0, 1, 401)
fig, ax = plt.subplots(figsize=(9.5, 5))
curvas = {"entropía / 2": entropia(np.column_stack([1 - p, p])) / 2, "Gini": gini(np.column_stack([1 - p, p])),
          "error": error_clasif(np.column_stack([1 - p, p]))}
for (nombre, v), color in zip(curvas.items(), [DORADO_OSC, AZUL, ROJO]):
    ax.plot(p, v, color=color)
rdp.estilo.etiquetar(ax, 1.01, {n: v[-40] for n, v in curvas.items()})
# división A: hijas con p = 20/60 y 0/40; padre p = 0.2
pI, pD, wI = 20 / 60, 0.0, 0.6
for fn, color in [(lambda q: 2 * q * (1 - q), AZUL), (lambda q: min(q, 1 - q), ROJO)]:
    ax.plot([pD, pI], [fn(pD), fn(pI)], color=color, ls="--", lw=1.4)
    ax.scatter([pD, pI], [fn(pD), fn(pI)], color=color, s=40, zorder=3)
    ax.scatter([0.2], [wI * fn(pI) + (1 - wI) * fn(pD)], color=color, s=70, marker="D", zorder=3)
    ax.scatter([0.2], [fn(0.2)], color=color, s=70, facecolor="white", lw=2, zorder=4)
ax.annotate("Gini: la cuerda queda abajo\n(ganancia 0.053)", (0.2, 0.267), (0.36, 0.17), fontsize=11, color=TINTA,
            arrowprops=dict(arrowstyle="->", color=TENUE))
ax.annotate("error: la cuerda pasa por el padre\n(ganancia 0)", (0.2, 0.2), (0.27, 0.035), fontsize=11, color=TINTA,
            arrowprops=dict(arrowstyle="->", color=TENUE))
ax.set_xlabel("proporción de la clase 2 en el nodo")
ax.set_ylabel("impureza")
ax.set_title("División A: padre (círculo), hijas (puntos) y promedio de las hijas (rombo)")
ax.set_xlim(0, 1.17)
plt.show()

# %% [markdown]
# (Dividimos la entropía entre 2 para dibujarla en la misma escala; en bits su máximo es 1.) En la práctica, Gini y entropía casi siempre eligen la misma división; scikit-learn usa Gini por omisión porque no requiere logaritmos.

# %% [markdown]
# ## 4. La mejor división, desde cero
#
# Para un atributo $x_j$ con $n$ datos en el nodo, solo hay $n - 1$ umbrales que vale la pena probar: los puntos medios entre valores consecutivos distintos. Probarlos uno por uno, recontando las clases cada vez, costaría $O(n^2)$. El truco es **ordenar una vez**: recorriendo los datos en orden, los conteos por clase del lado izquierdo son sumas acumuladas, y los del lado derecho, el total menos esas sumas. Así, evaluar todos los umbrales cuesta $O(n)$ después de ordenar, y el total es $O(n \log n)$ por atributo.

# %%
def mejor_division(x, y, K):
    """Mejor umbral para el atributo x según Gini. Regresa (impureza ponderada de las hijas, umbral)."""
    orden = np.argsort(x, kind="stable")
    xs, ys = x[orden], y[orden]
    uno = np.eye(K)[ys]                          # indicadoras de clase, n × K
    izq = np.cumsum(uno, axis=0)[:-1]            # conteos a la izquierda si cortamos después del i-ésimo
    der = uno.sum(axis=0) - izq
    n = len(y)
    n_izq = np.arange(1, n)
    imp = (n_izq * gini(izq) + (n - n_izq) * gini(der)) / n
    imp[xs[1:] == xs[:-1]] = np.inf              # no se puede cortar entre dos valores iguales
    i = np.argmin(imp)
    return imp[i], (xs[i] + xs[i + 1]) / 2


# la primera pregunta del árbol de scikit-learn, recalculada
for j, nombre in enumerate(["largo", "profundidad"]):
    imp, umbral = mejor_division(Xp[:, j], yp, 3)
    print(f"{nombre:>12}: mejor umbral {umbral:.2f} mm, Gini de las hijas {imp:.4f} (padre: {gini(np.bincount(yp)):.4f})")

# %% [markdown]
# Para el largo, el mejor corte es 42.35 mm, el mismo de scikit-learn, y baja la impureza de 0.636 a 0.325; el mejor corte por profundidad la deja más alta, así que el árbol pregunta primero por el largo. La curva completa muestra que el umbral óptimo no es obvio a simple vista:

# %%
def curva_division(x, y, K):
    orden = np.argsort(x, kind="stable")
    xs, ys = x[orden], y[orden]
    uno = np.eye(K)[ys]
    izq = np.cumsum(uno, axis=0)[:-1]
    der = uno.sum(axis=0) - izq
    n_izq = np.arange(1, len(y))
    imp = (n_izq * gini(izq) + (len(y) - n_izq) * gini(der)) / len(y)
    ok = xs[1:] > xs[:-1]
    return (xs[1:][ok] + xs[:-1][ok]) / 2, gini(np.bincount(y, minlength=K)) - imp[ok]


fig, ax = plt.subplots(figsize=(10.5, 4.6))
umbrales, gan = curva_division(Xp[:, 0], yp, 3)
ax.plot(umbrales, gan, color=AZUL)
ax.axvline(umbrales[np.argmax(gan)], color=TENUE, ls="--", lw=1.2)
ax.text(umbrales[np.argmax(gan)] + 0.4, gan.max(), f"máximo: {umbrales[np.argmax(gan)]:.2f} mm", color=TINTA, fontsize=11)
for k in range(3):
    xk = Xp[yp == k, 0]
    ax.scatter(xk, np.full_like(xk, -0.02 - 0.025 * k), s=12, color=COLORES[k], alpha=0.7, lw=0)
    ax.text(60.6, -0.02 - 0.025 * k, ESPECIES[k], color=TINTA, fontsize=10, va="center")
ax.set_xlabel("umbral sobre el largo del pico (mm)")
ax.set_ylabel("ganancia de Gini")
ax.set_title("Ganancia de cada umbral posible en la raíz (abajo, los pingüinos de cada especie)")
ax.set_yticks(np.arange(0, 0.31, 0.05))
ax.set_xlim(31, 64)
plt.show()

# %% [markdown]
# ### El algoritmo voraz (CART)
#
# El árbol se construye de arriba abajo con una regla **voraz** (*greedy*): en cada nodo se prueba cada atributo, se elige la división con mayor ganancia y se repite en cada hija, hasta que el nodo sea puro, llegue a la profundidad máxima o tenga muy pocos datos. Es el algoritmo CART (*Classification and Regression Trees*) de L. Breiman, J. Friedman, R. Olshen y C. Stone (1984).
#
# **Costo.** En cada nivel del árbol, los nodos se reparten los $N$ datos, así que buscar la mejor división en todos los nodos de un nivel cuesta $O(d\,N \log N)$ con $d$ atributos (se puede bajar a $O(dN)$ ordenando una sola vez al principio). Con profundidad $D$, el total es $O(D\, d\, N\log N)$.
#
# **¿Por qué voraz?** Porque buscar el árbol óptimo es intratable (salvo que P = NP): L. Hyafil y R. Rivest (*Information Processing Letters* 5, 1976) demostraron que encontrar el árbol binario de decisión con el menor número esperado de preguntas que clasifique correctamente un conjunto de datos es NP-completo. La regla voraz no garantiza el mejor árbol (una división mediocre ahora puede abrir paso a divisiones excelentes después), pero es rápida y funciona bien en la práctica.
#
# La construcción recursiva cabe en unas cuantas líneas. Cada nodo es un diccionario:

# %%
def crecer(X, y, K, max_prof, prof=0):
    """Árbol de clasificación CART con Gini, como diccionarios anidados."""
    conteos = np.bincount(y, minlength=K)
    nodo = {"conteos": conteos, "clase": int(np.argmax(conteos))}
    if prof == max_prof or gini(conteos) == 0:
        return nodo
    candidatas = [(*mejor_division(X[:, j], y, K), j) for j in range(X.shape[1])]
    imp, umbral, j = min(candidatas, key=lambda c: c[0])
    if not np.isfinite(imp):                      # todos los datos iguales: no hay corte posible
        return nodo
    izq = X[:, j] <= umbral
    nodo |= {"atributo": j, "umbral": umbral,
             "izq": crecer(X[izq], y[izq], K, max_prof, prof + 1),
             "der": crecer(X[~izq], y[~izq], K, max_prof, prof + 1)}
    return nodo


def predecir_uno(nodo, x):
    while "atributo" in nodo:
        nodo = nodo["izq"] if x[nodo["atributo"]] <= nodo["umbral"] else nodo["der"]
    return nodo["clase"]


def predecir(nodo, X):
    return np.array([predecir_uno(nodo, x) for x in X])


def hojas(nodo):
    return 1 if "atributo" not in nodo else hojas(nodo["izq"]) + hojas(nodo["der"])

# %% [markdown]
# Comparemos con `DecisionTreeClassifier` usando los cuatro atributos numéricos de los pingüinos (largo y profundidad del pico, largo de la aleta y masa) y varias profundidades. Entrenamos con el 75 % de los pingüinos y medimos, en el 25 % restante, la exactitud de cada árbol y qué fracción de sus predicciones coincide.

# %%
cuatro = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X4 = pinguinos[cuatro].to_numpy()
rng = np.random.default_rng(SEMILLA)
prueba_i = rng.permutation(len(yp))[:86]          # 25 % para prueba
entr_i = np.setdiff1d(np.arange(len(yp)), prueba_i)

filas = []
for prof in [1, 2, 3, 4, 6, None]:
    propio = crecer(X4[entr_i], yp[entr_i], 3, max_prof=-1 if prof is None else prof)
    sk = DecisionTreeClassifier(max_depth=prof, random_state=SEMILLA).fit(X4[entr_i], yp[entr_i])
    pp, ps = predecir(propio, X4[prueba_i]), sk.predict(X4[prueba_i])
    filas.append({"profundidad": "sin límite" if prof is None else prof, "hojas (nuestro)": hojas(propio),
                  "hojas (sklearn)": sk.get_n_leaves(), "exactitud de prueba (nuestro)": accuracy_score(yp[prueba_i], pp),
                  "exactitud de prueba (sklearn)": accuracy_score(yp[prueba_i], ps), "coinciden": (pp == ps).mean()})
pd.DataFrame(filas).set_index("profundidad").round(3)

# %% [markdown]
# (Con `max_prof=-1` la condición `prof == max_prof` nunca se cumple y el árbol crece hasta tener hojas puras.) Nuestro árbol de unas 30 líneas reproduce a scikit-learn: el mismo número de hojas en todas las profundidades y las mismas predicciones, salvo un pingüino de prueba con profundidad 4. Esas pequeñas diferencias vienen de **empates**: cuando dos atributos dan exactamente la misma ganancia, nosotros nos quedamos con el primero y scikit-learn recorre los atributos en un orden aleatorio (fijado por `random_state`). scikit-learn hace lo mismo que nosotros, pero en Cython, con una pila en lugar de recursión y con opciones adicionales (`min_samples_leaf`, pesos, faltantes).

# %% [markdown]
# # Parte 2. Árboles de regresión: las bicicletas
#
# ## 5. Los datos y el problema
#
# **Datos.** Las rentas por hora del sistema Capital Bikeshare de Washington, D. C., en 2011 y 2012 (H. Fanaee-T y J. Gama, *Progress in Artificial Intelligence*, 2014; UCI, CC BY 4.0), con la misma preparación y la misma partición de la Unidad 4: días 1 a 19 de cada mes para entrenar y del 20 en adelante para probar.
#
# **Problema.** Predecir `total`, las bicicletas rentadas en una hora, a partir de la hora, el tipo de día, el clima, el año y las variables meteorológicas. En la Unidad 4 la escalera de modelos lineales llegó a una RECM de prueba de 86.4 (M4), pero solo después de construir **a mano** la interacción hora × día laboral, y la regresión de Poisson bajó a 72.2. ¿Qué hace un árbol con los atributos crudos?

# %%
b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
b["anio"] = b.fecha.dt.year - 2011          # 0 = 2011, 1 = 2012
b["clima"] = b.clima.clip(upper=3)          # la categoría 4 tiene 3 horas: se une a la 3
entrena = b[b.fecha.dt.day <= 19]           # 10,886 horas
prueba = b[b.fecha.dt.day >= 20]            # 6,493 horas
atributos = ["hora", "laboral", "clima", "anio", "temp", "humedad", "viento"]
Xb, yb = entrena[atributos], entrena["total"]
Xb_te, yb_te = prueba[atributos], prueba["total"]
print(f"entrenamiento: {len(entrena):,} horas; prueba: {len(prueba):,} horas")

# %% [markdown]
# Ni *one-hot*, ni escalado, ni interacciones: a un árbol le da igual la escala de cada atributo (solo usa comparaciones $x_j \le t$), y `clima`, que es ordinal (1 despejado, 2 nublado, 3 lluvia), se puede dejar como número.
#
# ## 6. Hojas con promedios
#
# En regresión, cada hoja predice el **promedio** de las $y$ que llegan a ella, y la impureza de un nodo es el error cuadrático medio alrededor de ese promedio, es decir, la **varianza** de $y$ en el nodo. La mejor división es la que más reduce la suma de cuadrados:
#
# $$\min_{j,\,t}\;\Big[\sum_{x_{nj} \le t}(y_n - \bar y_I)^2 + \sum_{x_{nj} > t}(y_n - \bar y_D)^2\Big].$$
#
# Por qué el promedio: es la constante $c$ que minimiza $\sum_n (y_n - c)^2$ (derivando, $-2\sum_n(y_n - c) = 0$). Igual que en clasificación, con sumas acumuladas de $y$ y de $y^2$ todos los umbrales de un atributo se evalúan en $O(n)$ después de ordenar (es una de las preguntas del laboratorio). Veamos un árbol de profundidad 3:

# %%
arbol3 = DecisionTreeRegressor(max_depth=3, random_state=SEMILLA).fit(Xb, yb)
print(export_text(arbol3, feature_names=atributos, decimals=1))

# %% [markdown]
# Las primeras preguntas son sobre la **hora**: es el atributo que más explica. Y en la rama de las 6 de la mañana, el árbol pregunta si el día es **laboral**: 102.6 bicicletas en promedio contra 20.0 en fines de semana y feriados. Nadie le dijo que el efecto de la hora depende del tipo de día: lo encontró porque, dentro de ese nodo, partir por `laboral` reducía mucho la suma de cuadrados. **Una rama del árbol es una interacción**: el efecto de un atributo depende de las preguntas que se hicieron antes.
#
# ### Más profundidad, menos error... hasta cierto punto

# %%
filas = []
for prof in [2, 3, 4, 6, 8, 10, 12, 14, 16, 20, None]:
    m = DecisionTreeRegressor(max_depth=prof, random_state=SEMILLA).fit(Xb, yb)
    filas.append({"profundidad": 40 if prof is None else prof, "hojas": m.get_n_leaves(),
                  "RECM entrenamiento": root_mean_squared_error(yb, m.predict(Xb)),
                  "RECM prueba": root_mean_squared_error(yb_te, m.predict(Xb_te))})
prof_tabla = pd.DataFrame(filas)
mejor = prof_tabla.loc[prof_tabla["RECM prueba"].idxmin()]
print(f"árbol sin límite: profundidad {DecisionTreeRegressor(random_state=SEMILLA).fit(Xb, yb).get_depth()}")

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(prof_tabla.profundidad, prof_tabla["RECM entrenamiento"], marker="o", ms=5, color=GRIS)
ax.plot(prof_tabla.profundidad, prof_tabla["RECM prueba"], marker="o", ms=5, color=AZUL)
for valor, texto in [(86.4, "M4 lineal (U4): 86.4"), (72.2, "M4 Poisson (U4): 72.2")]:
    ax.axhline(valor, color=TENUE, ls=":", lw=1.2)
    ax.text(30, valor + 1.5, texto, color=TENUE, fontsize=10, ha="center", va="bottom")
rdp.estilo.etiquetar(ax, 41.5, {"entrenamiento": prof_tabla["RECM entrenamiento"].iloc[-1], "prueba": prof_tabla["RECM prueba"].iloc[-1]})
ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 20, 40], ["2", "4", "6", "8", "10", "12", "14", "16", "20", "sin\nlímite"])
ax.set_xlabel("profundidad máxima")
ax.set_ylabel("RECM (bicicletas por hora)")
ax.set_title("Árbol de regresión: error contra profundidad")
ax.set_xlim(1, 47)
plt.show()
prof_tabla.set_index("profundidad").round(1).T

# %% [markdown]
# La curva de la Unidad 3, otra vez. El error de entrenamiento baja sin parar hasta casi cero (el árbol sin límite tiene más de 10,000 hojas para 10,886 horas: casi una hoja por dato). El de prueba baja hasta la profundidad 10 (RECM 69.5, ya mejor que Poisson) y luego **sube**: los niveles adicionales memorizan el ruido. Con la profundidad máxima elegida a ojo sobre el conjunto de prueba estaríamos haciendo trampa; hay que elegir la complejidad con validación cruzada, y la profundidad no es la única ni la mejor forma de medirla.

# %% [markdown]
# ## 7. Poda por costo–complejidad
#
# Detener el crecimiento temprano tiene un problema: una división que por sí sola gana poco puede abrir paso a divisiones muy buenas (la regla voraz no ve el futuro). Breiman y coautores propusieron lo contrario: crecer un árbol grande $T_0$ y luego **podarlo**. Para un subárbol $T \subseteq T_0$ (obtenido colapsando nodos internos en hojas) se define el **costo–complejidad**
#
# $$R_\alpha(T) = R(T) + \alpha\,|T|,$$
#
# donde $R(T)$ es el error de entrenamiento del subárbol (en regresión, el error cuadrático medio; en clasificación, la impureza ponderada de las hojas), $|T|$ es su número de hojas y $\alpha \ge 0$ es el precio de cada hoja. Es otra vez **error en la muestra + penalización por complejidad**.
#
# **La sucesión anidada.** Para cada nodo interno $t$, colapsar su rama $T_t$ en una hoja aumenta el error en $R(t) - R(T_t)$ y ahorra $|T_t| - 1$ hojas; vale la pena cuando
#
# $$\alpha \;\ge\; g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}.$$
#
# La **poda del eslabón más débil** colapsa el nodo con el menor $g(t)$, recalcula y repite. Breiman et al. demostraron que así se obtiene una sucesión **anidada** de subárboles $T_0 \supset T_1 \supset \cdots \supset \{\text{raíz}\}$ y valores $0 = \alpha_0 < \alpha_1 < \cdots$ tales que $T_k$ minimiza $R_\alpha$ para todo $\alpha \in [\alpha_k, \alpha_{k+1})$. Basta entonces elegir **un número**, $\alpha$, con validación cruzada. `cost_complexity_pruning_path` calcula la sucesión:

# %%
ruta = DecisionTreeRegressor(random_state=SEMILLA).cost_complexity_pruning_path(Xb, yb)
print(f"{len(ruta.ccp_alphas):,} subárboles anidados; α va de 0 a {ruta.ccp_alphas[-2]:,.0f}")

# %% [markdown]
# ### Elegir $\alpha$ con validación cruzada (por días)
#
# Un detalle importante: las horas de un mismo día se parecen mucho entre sí. Si la validación cruzada repartiera horas sueltas al azar, el modelo vería las 8 y las 10 de la mañana de un día al predecir las 9, y la validación sería optimista (premiaría árboles que memorizan días). Por eso usamos `GroupKFold` con el **día** como grupo: cada pliegue deja fuera días completos.

# %%
alfas = np.geomspace(1, 3000, 36)
dias = entrena["fecha"].dt.date
cv_dias = GroupKFold(5)
filas = []
for a in alfas:
    m = DecisionTreeRegressor(ccp_alpha=a, random_state=SEMILLA)
    puntajes = -cross_val_score(m, Xb, yb, groups=dias, cv=cv_dias, scoring="neg_root_mean_squared_error")
    filas.append({"alfa": a, "RECM CV": puntajes.mean(), "ee": puntajes.std(ddof=1) / np.sqrt(len(puntajes)),
                  "hojas": DecisionTreeRegressor(ccp_alpha=a, random_state=SEMILLA).fit(Xb, yb).get_n_leaves()})
poda = pd.DataFrame(filas)
i_min = poda["RECM CV"].idxmin()
alfa_cv = poda.alfa[i_min]
podado = DecisionTreeRegressor(ccp_alpha=alfa_cv, random_state=SEMILLA).fit(Xb, yb)
recm_podado = root_mean_squared_error(yb_te, podado.predict(Xb_te))
print(f"α elegido por validación cruzada: {alfa_cv:.1f} → {podado.get_n_leaves()} hojas, profundidad {podado.get_depth()}")
print(f"RECM de validación cruzada: {poda['RECM CV'][i_min]:.1f}; RECM de prueba: {recm_podado:.1f}")

# %%
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.fill_between(poda.hojas, poda["RECM CV"] - poda.ee, poda["RECM CV"] + poda.ee, color=AZUL, alpha=0.15, lw=0)
ax.plot(poda.hojas, poda["RECM CV"], marker="o", ms=4, color=AZUL)
ax.scatter([podado.get_n_leaves()], [poda["RECM CV"][i_min]], s=110, facecolor="none", edgecolor=ROJO, lw=2, zorder=4)
ax.annotate(f"α = {alfa_cv:.0f}: {podado.get_n_leaves()} hojas", (podado.get_n_leaves(), poda["RECM CV"][i_min]),
            (podado.get_n_leaves() * 2.2, poda["RECM CV"][i_min] + 25), color=TINTA, fontsize=11,
            arrowprops=dict(arrowstyle="->", color=TENUE))
ax.set_xscale("log")
ax.set_xlabel("hojas del subárbol podado (escala log; α grande a la izquierda)")
ax.set_ylabel("RECM de validación cruzada")
ax.set_title("Poda por costo–complejidad: validación cruzada por días (banda: ± 1 error estándar)")
plt.show()

# %% [markdown]
# Con muy pocas hojas (α grande) el árbol es demasiado simple; con miles (α chico), sobreajusta. El mínimo está en un árbol de 150 hojas. Breiman et al. recomiendan además la **regla de un error estándar**: elegir el árbol más pequeño cuyo error de validación cruzada esté a menos de un error estándar del mínimo, porque las diferencias dentro de la banda no son confiables.
#
# Comparemos todo en el conjunto de prueba, que no hemos usado para elegir nada:

# %%
resumen = pd.DataFrame({
    "RECM de prueba": [165.0, 153.5, 107.4, 86.4, 72.2,
                       root_mean_squared_error(yb_te, arbol3.predict(Xb_te)),
                       prof_tabla.set_index("profundidad").loc[40, "RECM prueba"], recm_podado],
    "hojas": [None, None, None, None, None, arbol3.get_n_leaves(),
              prof_tabla.set_index("profundidad").loc[40, "hojas"], podado.get_n_leaves()]},
    index=["M1 lineal (U4)", "M2 lineal (U4)", "M3 lineal (U4)", "M4 lineal, interacción a mano (U4)", "M4 Poisson (U4)",
           "árbol de profundidad 3", "árbol sin límite", f"árbol podado (α = {alfa_cv:.0f})"])
resumen.round(1)

# %% [markdown]
# El árbol podado, con los siete atributos crudos y sin ninguna transformación, tiene un error de prueba más bajo que el mejor modelo lineal de la Unidad 4 y que la regresión de Poisson. El árbol sin podar es peor que el podado: tiene muy poco sesgo pero mucha varianza.
#
# ### La función que aprendió el árbol
#
# Un árbol de regresión es una **función escalonada**: constante en cada rectángulo. Fijemos condiciones típicas (2012, despejado, 25 °C, 50 % de humedad, viento de 10 km/h) y veamos la predicción hora por hora para los dos tipos de día, y, a las 5 de la tarde de un día laboral, en función de la temperatura:

# %%
def escenario(**cambios):
    base = {"hora": 17, "laboral": 1, "clima": 1, "anio": 1, "temp": 25.0, "humedad": 50, "viento": 10.0}
    n = max(np.size(v) for v in cambios.values())
    return pd.DataFrame({k: np.broadcast_to(cambios.get(k, v), n) for k, v in base.items()})[atributos]


horas = np.arange(24)
temps = np.linspace(0, 40, 401)
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax = ejes[0]
for lab, color, nombre in [(1, AZUL, "día laboral"), (0, ROJO, "fin de semana o feriado")]:
    ax.step(horas, podado.predict(escenario(hora=horas, laboral=lab)), where="mid", color=color)
    obs = entrena[(entrena.laboral == lab) & (entrena.anio == 1)].groupby("hora").total.mean()
    ax.plot(obs.index, obs.values, ls=":", color=color, lw=1.4)
ax.text(19.7, 620, "día laboral", color=AZUL, fontsize=11)
ax.text(11.2, 640, "fin de semana o feriado", color=ROJO, fontsize=11)
ax.text(0.2, 700, "escalones: árbol podado\npunteado: promedio observado\n(2012, todas las condiciones)", color=TENUE, fontsize=10)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("hora del día")
ax.set_ylabel("bicicletas por hora")
ax.set_title("Predicción por hora (2012, despejado, 25 °C)")
ax = ejes[1]
ax.step(temps, podado.predict(escenario(temp=temps)), where="post", color=AZUL)
ax.set_xlabel("temperatura (°C)")
ax.set_title("A las 17 h de un día laboral, según la temperatura")
fig.tight_layout()
plt.show()

# %% [markdown]
# A la izquierda, el árbol reprodujo los dos picos de los días laborales (entrada y salida del trabajo) y la loma del mediodía de los fines de semana: la interacción hora × día laboral que en la Unidad 4 tuvimos que crear con una variable `hora_lab` de 48 categorías. A la derecha, la respuesta a la temperatura es una escalera: el árbol no puede producir una curva suave, solo saltos en los umbrales que eligió, y fuera del rango de los datos de entrenamiento se queda constante (no extrapola). Las dos cosas son a la vez una limitación y una protección.

# %% [markdown]
# ## 8. Inestabilidad: árboles distintos con datos casi iguales
#
# Volvamos a los pingüinos. Entrenemos árboles sin límite de profundidad en **remuestras *bootstrap*** (sacar 342 pingüinos con reemplazo de los 342) y veamos cuánto cambian.

# %%
rng = np.random.default_rng(SEMILLA)
fig, ejes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
raices, predicciones = [], []
for r in range(200):
    i = rng.integers(0, len(yp), len(yp))
    m = DecisionTreeClassifier(random_state=SEMILLA).fit(Xp[i], yp[i])
    raices.append((pico[m.tree_.feature[0]], m.tree_.threshold[0]))
    predicciones.append(m.predict(malla))
    if r < 4:
        dibujar_regiones(ejes[r], m, f"remuestra {r + 1}: {m.get_n_leaves()} hojas", puntos=False)
        ejes[r].scatter(Xp[i, 0], Xp[i, 1], s=10, c=[COLORES[k] for k in yp[i]], lw=0)
        ejes[r].set_xlabel("largo del pico (mm)")
ejes[0].set_ylabel("profundidad del pico (mm)")
fig.tight_layout()
plt.show()

# %%
predicciones = np.array(predicciones)
moda = np.array([np.bincount(c, minlength=3).argmax() for c in predicciones.T])
desacuerdo = (predicciones != moda).mean(axis=0)
raices = pd.DataFrame(raices, columns=["atributo", "umbral"])
print(f"fracción del plano donde al menos 10 % de los árboles contradicen a la mayoría: {(desacuerdo >= 0.10).mean():.1%}")
print("primera pregunta de los 200 árboles:")
raices.groupby("atributo").umbral.describe()[["count", "min", "50%", "max"]].round(2)

# %% [markdown]
# Las fronteras gruesas se mantienen (el largo y la profundidad del pico sí separan a las especies), pero los detalles cambian de una remuestra a otra: rectángulos pequeños que aparecen y desaparecen, cortes que se mueven varios milímetros, y hasta la primera pregunta cambia de atributo en algunas remuestras. Como cada división depende de las anteriores, un cambio pequeño cerca de la raíz se propaga a todo el árbol.
#
# Ésa es la **varianza** del árbol: con poco sesgo (un árbol profundo puede aproximar cualquier frontera hecha de escalones) pero mucha varianza. La descomposición sesgo–varianza de la Unidad 3 sugiere el remedio: **promediar** muchos árboles para bajar la varianza sin subir el sesgo. Si los 200 árboles votan, la frontera se estabiliza; eso es *bagging*, y con un ingrediente más, un **bosque aleatorio**. Es el tema de la siguiente libreta.

# %% [markdown]
# ## 9. Detalles prácticos
#
# ### Transformaciones monótonas: no hay que escalar
#
# Un árbol solo compara $x_j \le t$. Si cambiamos $x_j$ por $\phi(x_j)$ con $\phi$ estrictamente creciente, el orden de los datos no cambia, así que se pueden hacer exactamente las mismas particiones de los datos de entrenamiento (con umbrales $\phi(t)$). Comprobémoslo cambiando la temperatura por $e^{\text{temp}/5}$ y la humedad por su raíz cuadrada:

# %%
def transformar(X):
    return X.assign(temp=np.exp(X.temp / 5), humedad=np.sqrt(X.humedad))


podado_t = DecisionTreeRegressor(ccp_alpha=alfa_cv, random_state=SEMILLA).fit(transformar(Xb), yb)
print(f"hojas: {podado.get_n_leaves()} y {podado_t.get_n_leaves()}")
print(f"máxima diferencia en entrenamiento: {np.abs(podado.predict(Xb) - podado_t.predict(transformar(Xb))).max():.2e}")
distintas = podado.predict(Xb_te) != podado_t.predict(transformar(Xb_te))
print(f"horas de prueba con predicción distinta: {distintas.sum()} de {len(Xb_te):,}")

# %% [markdown]
# Mismo árbol y exactamente las mismas predicciones en entrenamiento. En prueba difieren unas cuantas horas, y la razón es instructiva: el umbral se pone en el **punto medio** entre dos valores consecutivos de entrenamiento, y el punto medio en una escala no corresponde al punto medio en la otra. Un valor nuevo que cae en ese hueco (por ejemplo, 40.2 °C, la única temperatura de prueba que nunca apareció en entrenamiento) puede quedar de distinto lado; a eso se suma el redondeo, porque scikit-learn guarda los atributos en precisión sencilla. Las particiones de los datos de entrenamiento, que es lo que el árbol aprende, son idénticas.
#
# Por la misma razón, los árboles son **robustos a valores atípicos en los atributos**: un valor enorme solo cuenta como «el más grande». En la respuesta $y$, en cambio, un atípico sí mueve el promedio de su hoja.
#
# ### Atributos categóricos y faltantes
#
# - **Categóricos.** Con $q$ categorías sin orden hay $2^{q-1} - 1$ formas de partirlas en dos grupos. Para clasificación binaria y para regresión con pérdida cuadrática hay un atajo (Fisher, 1958; Breiman et al., 1984): ordenar las categorías por la proporción de positivos (o el promedio de $y$) y probar solo los $q - 1$ cortes en ese orden; la mejor partición está entre ellos. `DecisionTreeClassifier` de scikit-learn 1.9 no maneja categóricos directamente: hay que codificarlos (con números si tienen orden, con *one-hot* si no). LightGBM y `HistGradientBoosting` sí los manejan (lo veremos en la libreta 3). Ojo: los atributos con muchas categorías tienen muchas particiones posibles y el árbol tiende a preferirlos aunque no sirvan, un sesgo que volveremos a encontrar en las importancias de los bosques.
# - **Faltantes.** CART original usaba **divisiones sustitutas** (*surrogate splits*): si falta $x_j$, se usa otra pregunta que reparte a los datos de forma parecida. scikit-learn (desde la versión 1.3) aprende, en cada división, a qué lado mandar los datos con faltantes, el que dé mayor ganancia. Por ejemplo, los dos pingüinos sin medidas:

# %%
completos = rdp.datos.cargar("pinguinos")
con_faltantes = DecisionTreeClassifier(max_depth=3, random_state=SEMILLA).fit(
    completos[cuatro], completos["species"])
print(completos[completos[cuatro].isna().any(axis=1)][cuatro + ["species"]])
print("predicción para esos dos:", con_faltantes.predict(completos[completos[cuatro].isna().any(axis=1)][cuatro]))

# %% [markdown]
# El árbol se entrenó sin quejarse y los manda a alguna hoja; para un pingüino sin ninguna medida, cualquier predicción es una conjetura informada por cómo se repartieron los faltantes en entrenamiento.
#
# ### Árboles contra modelos lineales
#
# | | modelo lineal | árbol de decisión |
# |---|---|---|
# | forma de la función | hiperplano (en algún espacio) | escalones, constante en rectángulos |
# | interacciones | hay que construirlas | las descubre (una rama es una interacción) |
# | escala de los atributos | importa (regularización, gradiente) | no importa |
# | relaciones lineales suaves | las capta con pocos parámetros | necesita muchos escalones |
# | extrapolación | lineal (a veces absurda) | constante |
# | varianza | baja | alta |
# | interpretación | coeficientes | reglas (si el árbol es chico) |
#
# Una frontera diagonal, como $x_1 + x_2 > 0$, es trivial para un modelo lineal y costosísima para un árbol, que la aproxima con una escalera. Y al revés, una interacción como «a las 8 de la mañana importa si es día laboral» es natural para el árbol y había que inventarla para el modelo lineal.
#
# *Referencias:* L. Breiman, J. H. Friedman, R. A. Olshen y C. J. Stone, *Classification and Regression Trees*, Wadsworth (1984), caps. 1–3 y 8. L. Hyafil y R. L. Rivest, «Constructing optimal binary decision trees is NP-complete», *Information Processing Letters* 5 (1976). T. Hastie, R. Tibshirani y J. Friedman, *The Elements of Statistical Learning*, 2.ª ed., Springer (2009), sección 9.2.

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u05", ids=["u05-p01", "u05-p02", "u05-p03", "u05-p04"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. **Entropía y Gini cerca de un nodo puro.** Con dos clases, desarrolla $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ y $G(p) = 2p(1-p)$ alrededor de $p = 1/2$ hasta segundo orden y muestra que $H(p) \approx 1 - \frac{2}{\ln 2}(p - \frac12)^2$ y $G(p) = \frac12 - 2(p - \frac12)^2$. ¿Qué tan parecidas son $H/2$ y $G$? Encuentra un ejemplo con tres clases en el que Gini y entropía elijan divisiones distintas en el mismo nodo.
# 2. **Desigualdad de la ganancia.** Demuestra que, para cualquier división de un nodo con $K$ clases, la ganancia de Gini es no negativa, y que es cero si y solo si las dos hijas tienen las mismas proporciones de clases. Sugerencia: $G(\mathbf{p}) = 1 - \|\mathbf{p}\|^2$ y $\|\cdot\|^2$ es estrictamente convexa.
# 3. **La mejor división de regresión en $O(n\log n)$.** Escribe `mejor_division_regresion(x, y)` usando sumas acumuladas de $y$ y de $y^2$ (la suma de cuadrados de un grupo es $\sum y^2 - (\sum y)^2/n$). Verifica que, para la hora en las bicicletas de entrenamiento, encuentras la misma primera pregunta que `DecisionTreeRegressor(max_depth=1)`.
# 4. **Poda a mano.** Para `arbol3` (profundidad 3, ocho hojas) calcula $g(t)$ para cada uno de sus siete nodos internos a partir de `tree_.impurity` y `tree_.n_node_samples`, encuentra el eslabón más débil y verifica tu resultado con `cost_complexity_pruning_path`.
# 5. **Diagonales.** Genera 500 puntos uniformes en $[-1, 1]^2$ con clase $\mathbb{1}[x_1 + x_2 > 0]$. ¿Cuántas hojas necesita un árbol para llegar a 95 % de exactitud en prueba? Ahora rota los datos 45° (usa $x_1 + x_2$ y $x_1 - x_2$ como atributos) y repite. ¿Qué te dice esto sobre cuándo preferir un modelo lineal?
