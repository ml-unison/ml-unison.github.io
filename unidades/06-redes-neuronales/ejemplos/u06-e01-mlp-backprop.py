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
# # Una red neuronal desde cero, y luego con PyTorch
#
# **Unidad 6 · Ejemplo 1**
#
# Todos los modelos de la Unidad 4 eran lineales en algún espacio: en los atributos, en los log-momios o, con un kernel, en un espacio fijo que elegíamos nosotros. Una **red neuronal** da el paso siguiente: **aprende** la transformación de los datos al mismo tiempo que el clasificador lineal que va encima. En esta libreta vamos a:
#
# 1. ver con el XOR por qué una sola neurona no basta y qué hace la capa oculta con el espacio;
# 2. programar en NumPy un **perceptrón multicapa** de dos capas: la pasada hacia adelante, la **retropropagación** del gradiente en forma matricial y su **verificación** con diferencias finitas;
# 3. entrenarlo en un problema de dos dimensiones y ver por qué hace falta la **no linealidad**;
# 4. construir a mano, con neuronas ReLU, una aproximación de la temperatura de Hermosillo a lo largo del año, para entender qué dice (y qué no) el **teorema de aproximación universal**;
# 5. resolver los dígitos escritos a mano de la Unidad 4 con nuestra red y con **PyTorch**, cuya **diferenciación automática** da exactamente el gradiente que calculamos a mano;
# 6. entender por qué importa la **inicialización** de los pesos y qué es el **desvanecimiento del gradiente**.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado. PyTorch ya viene instalado en Colab.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F

from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import rdp

rdp.estilo.usar()
SEMILLA = 2011
torch.manual_seed(SEMILLA)
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE
REGIONES = ["#C0CEE3", "#F3C7C4"]   # tintas de azul y rojo para las regiones de decisión
print(f"PyTorch {torch.__version__} · dispositivo: {dispositivo}")

# %% [markdown]
# ## 1. Los datos y el problema
#
# Usaremos tres conjuntos de datos, cada uno para una pregunta distinta.
#
# - **Dos problemas de juguete en el plano**: el XOR (cuatro nubes de puntos en las esquinas de un cuadrado, con clases cruzadas) y las «lunas» de `make_moons`, dos medias lunas entrelazadas con ruido. Son sintéticos a propósito: en dos dimensiones podemos **dibujar** lo que hace la red.
# - **La temperatura media diaria de Hermosillo**, medida en el aeropuerto de 1992 a 2024 (red GHCN-Daily de la NOAA; Menne et al., 2012). Promediando los 33 años por día del año obtenemos la curva típica del calor sonorense, una función real de una variable que intentaremos aproximar con neuronas.
# - **Los dígitos escritos a mano** de la Unidad 4: 1,797 imágenes de $8 \times 8$ celdas con intensidades de 0 a 16 (Alpaydin y Kaynak, UCI, 1998). Usamos **la misma partición** que en la Unidad 4: 25 % para prueba, estratificada, con `random_state=2011`.
#
# **Problema.** Reconocer el dígito (0 a 9) a partir de sus 64 intensidades con un perceptrón multicapa, y compararlo con la regresión logística multinomial, que en la Unidad 4 acertó en 97 % de las imágenes de prueba. En el camino, entender cada pieza del entrenamiento lo bastante bien como para programarla sin bibliotecas.

# %%
Xd, yd = load_digits(return_X_y=True)
Xd = Xd / 16                                          # intensidades en [0, 1]
Xd_ent, Xd_pru, yd_ent, yd_pru = train_test_split(Xd, yd, test_size=0.25, stratify=yd, random_state=SEMILLA)
print(f"dígitos: {len(yd_ent):,} de entrenamiento y {len(yd_pru)} de prueba, {Xd.shape[1]} atributos")

logistica_digitos = LogisticRegression(C=10, max_iter=5000).fit(Xd_ent, yd_ent)
exac_logistica = logistica_digitos.score(Xd_pru, yd_pru)
print(f"regresión logística (C = 10): exactitud de prueba {exac_logistica:.4f} "
      f"({np.sum(logistica_digitos.predict(Xd_pru) != yd_pru)} errores)")

# %% [markdown]
# La regresión logística con $C = 10$ (el valor que eligió la validación cruzada en la Unidad 4) se equivoca en 13 de las 450 imágenes de prueba: exactitud de 0.971. Ésa es la cifra que hay que superar.
#
# ## 2. El XOR: lo que una recta no puede
#
# El perceptrón de la Unidad 3 y la regresión logística de la Unidad 4 trazan una frontera **lineal**. El ejemplo más pequeño que no se puede separar con una recta es el «o exclusivo»: $y = 1$ si exactamente una de las dos coordenadas es positiva. Generemos cuatro nubes de 50 puntos alrededor de las esquinas $(\pm 1, \pm 1)$.

# %%
def nubes_xor(n_por_nube, rng, desv=0.35):
    centros = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    X = np.concatenate([c + desv * rng.standard_normal((n_por_nube, 2)) for c in centros])
    y = np.repeat([0, 0, 1, 1], n_por_nube)          # clase 1: signos distintos
    return X, y


Xx, yx = nubes_xor(50, np.random.default_rng(SEMILLA))
log_xor = LogisticRegression().fit(Xx, yx)
print(f"regresión logística en el XOR: exactitud de entrenamiento {log_xor.score(Xx, yx):.3f}")

# %% [markdown]
# La logística no hace nada mejor que adivinar (0.5 sería el azar): cualquier recta deja de cada lado la mitad de cada clase. Esto es lo que M. Minsky y S. Papert usaron en *Perceptrons* (1969) como ejemplo de las limitaciones de una sola capa.
#
# **La solución con dos capas, a mano.** Para las cuatro esquinas exactas $\{0,1\}^2$ basta con dos neuronas ReLU, $\mathrm{ReLU}(z) = \max(0, z)$:
#
# $$h_1 = \mathrm{ReLU}(x_1 + x_2), \qquad h_2 = \mathrm{ReLU}(x_1 + x_2 - 1), \qquad \hat y = h_1 - 2 h_2 .$$
#
# Las esquinas $(0,0), (0,1), (1,0), (1,1)$ van a $(h_1, h_2) = (0,0), (1,0), (1,0), (2,1)$ y $\hat y = 0, 1, 1, 0$. La capa oculta **dobla el plano**: manda las dos esquinas de la clase 1 al mismo punto, y en el espacio $(h_1, h_2)$ las clases ya se separan con una recta.

# %%
esquinas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
h = np.maximum(0, np.column_stack([esquinas.sum(1), esquinas.sum(1) - 1]))
pd.DataFrame({"x1": esquinas[:, 0], "x2": esquinas[:, 1], "h1": h[:, 0], "h2": h[:, 1], "ŷ = h1 − 2 h2": h[:, 0] - 2 * h[:, 1]})

# %% [markdown]
# Lo interesante es que esa transformación no hay que diseñarla: se puede **aprender**. Más abajo entrenaremos una red con dos neuronas ocultas en las nubes del XOR y dibujaremos lo que aprendió. Primero, la red.
#
# ## 3. El perceptrón multicapa
#
# Un perceptrón multicapa (*multilayer perceptron*, MLP) de dos capas, para $K$ clases, calcula
#
# $$\mathbf z^{(1)} = W^{(1)}\mathbf x + \mathbf b^{(1)}, \quad
# \mathbf a^{(1)} = \sigma\big(\mathbf z^{(1)}\big), \quad
# \mathbf z^{(2)} = W^{(2)}\mathbf a^{(1)} + \mathbf b^{(2)}, \quad
# \hat{\mathbf y} = \operatorname{softmax}\big(\mathbf z^{(2)}\big),$$
#
# donde $W^{(1)}$ es de $H \times d$ ($H$ neuronas ocultas), $W^{(2)}$ es de $K \times H$, y $\sigma$ es una **función de activación** que se aplica entrada por entrada: tangente hiperbólica, sigmoide o ReLU. La última capa es exactamente la regresión *softmax* de la Unidad 4, solo que en lugar de recibir $\mathbf x$ recibe $\mathbf a^{(1)}$, unos atributos **aprendidos**. La pérdida es la entropía cruzada promedio, $E = -\frac1N\sum_n \log \hat y_{n, y_n}$.
#
# En el código, cada **renglón** de una matriz es un ejemplo (como en scikit-learn), así que $\mathbf z^{(1)} = W^{(1)}\mathbf x + \mathbf b^{(1)}$ para todos los ejemplos a la vez se escribe `Z1 = X @ W1.T + b1`. Guardamos $W$ con forma (salidas, entradas), igual que `nn.Linear` de PyTorch, para poder comparar después.

# %%
ACTIVACIONES = {   # función y su derivada, escrita en términos de z y de a = σ(z)
    "tanh": (np.tanh, lambda Z, A: 1 - A**2),
    "relu": (lambda Z: np.maximum(Z, 0), lambda Z, A: (Z > 0).astype(float)),
    "identidad": (lambda Z: Z, lambda Z, A: np.ones_like(Z)),
}


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)              # no cambia el resultado y evita desbordes
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def inicializar(d, H, K, rng, act="tanh"):
    """Pesos gaussianos con la varianza de He (ReLU) o de Xavier (tanh); sesgos en cero (ver sección 10)."""
    var1 = 2 / d if act == "relu" else 2 / (d + H)
    return {"W1": rng.normal(0, np.sqrt(var1), (H, d)), "b1": np.zeros(H),
            "W2": rng.normal(0, np.sqrt(2 / (H + K)), (K, H)), "b2": np.zeros(K)}


def adelante(p, X, act="tanh"):
    Z1 = X @ p["W1"].T + p["b1"]
    A1 = ACTIVACIONES[act][0](Z1)
    Z2 = A1 @ p["W2"].T + p["b2"]
    return softmax(Z2), (X, Z1, A1)


def entropia_cruzada(P, y):
    return -np.mean(np.log(P[np.arange(len(y)), y]))

# %% [markdown]
# ## 4. Retropropagación
#
# Necesitamos $\partial E/\partial W^{(1)}$, $\partial E/\partial \mathbf b^{(1)}$, $\partial E/\partial W^{(2)}$ y $\partial E/\partial \mathbf b^{(2)}$. La red es una **composición** de funciones simples, así que todo es la regla de la cadena; la retropropagación (*backpropagation*) es la forma de organizarla para no repetir cuentas. Trabajemos con un solo ejemplo $(\mathbf x, y)$, con $\mathbf y$ su codificación *one-hot*, y definamos el **error** de cada capa como la derivada de la pérdida respecto de su entrada lineal: $\boldsymbol\delta^{(\ell)} = \partial E / \partial \mathbf z^{(\ell)}$.
#
# **Capa de salida.** Con $E = -\sum_k y_k \log \hat y_k$ y $\log\hat y_k = z_k - \log\sum_j e^{z_j}$:
#
# $$\frac{\partial E}{\partial z_i} = -\sum_k y_k\Big(\mathbb 1[k = i] - \frac{e^{z_i}}{\sum_j e^{z_j}}\Big) = -y_i + \hat y_i \sum_k y_k = \hat y_i - y_i
# \qquad\Longrightarrow\qquad \boldsymbol\delta^{(2)} = \hat{\mathbf y} - \mathbf y .$$
#
# Es el mismo «predicción menos dato» de la regresión logística y de todos los GLM con liga canónica: *softmax* y entropía cruzada están hechas la una para la otra.
#
# **Pesos de una capa.** Como $z^{(2)}_i = \sum_j W^{(2)}_{ij} a^{(1)}_j + b^{(2)}_i$, el peso $W^{(2)}_{ij}$ solo afecta a $z^{(2)}_i$, y $\partial z^{(2)}_i/\partial W^{(2)}_{ij} = a^{(1)}_j$:
#
# $$\frac{\partial E}{\partial W^{(2)}_{ij}} = \delta^{(2)}_i\, a^{(1)}_j
# \qquad\Longrightarrow\qquad \frac{\partial E}{\partial W^{(2)}} = \boldsymbol\delta^{(2)}\,\mathbf a^{(1)\top}, \qquad \frac{\partial E}{\partial \mathbf b^{(2)}} = \boldsymbol\delta^{(2)} .$$
#
# El gradiente de una matriz de pesos es un producto exterior: «error que llega» por «activación que entra».
#
# **Hacia atrás.** La activación $a^{(1)}_j$ afecta a **todas** las $z^{(2)}_i$, así que se suman las contribuciones; luego se pasa por la activación:
#
# $$\frac{\partial E}{\partial a^{(1)}_j} = \sum_i W^{(2)}_{ij}\,\delta^{(2)}_i
# \qquad\Longrightarrow\qquad
# \boldsymbol\delta^{(1)} = \big(W^{(2)\top}\boldsymbol\delta^{(2)}\big) \odot \sigma'\big(\mathbf z^{(1)}\big),$$
#
# y otra vez $\partial E/\partial W^{(1)} = \boldsymbol\delta^{(1)}\mathbf x^\top$. Con más capas la receta es la misma: $\boldsymbol\delta^{(\ell)} = \big(W^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}\big) \odot \sigma'(\mathbf z^{(\ell)})$ y $\partial E/\partial W^{(\ell)} = \boldsymbol\delta^{(\ell)}\mathbf a^{(\ell-1)\top}$. El error viaja **hacia atrás** por las mismas matrices que usó la señal hacia adelante, transpuestas.
#
# **Con un lote de $N$ ejemplos** apilamos los $\boldsymbol\delta$ como renglones de una matriz $\Delta$, y los productos exteriores sumados se vuelven un producto de matrices: $\partial E/\partial W^{(2)} = \Delta^{(2)\top} A^{(1)}$. Como $E$ es un promedio, dividimos $\Delta^{(2)}$ entre $N$.

# %%
def atras(p, cache, P, y, act="tanh"):
    """Gradiente de la entropía cruzada promedio respecto de todos los parámetros."""
    X, Z1, A1 = cache
    N = len(y)
    D2 = P.copy()
    D2[np.arange(N), y] -= 1                          # Δ(2) = Ŷ − Y
    D2 /= N                                           # porque E es un promedio
    D1 = (D2 @ p["W2"]) * ACTIVACIONES[act][1](Z1, A1)   # Δ(1) = (Δ(2) W(2)) ⊙ σ'(Z(1))
    return {"W1": D1.T @ X, "b1": D1.sum(0), "W2": D2.T @ A1, "b2": D2.sum(0)}

# %% [markdown]
# Son cuatro renglones de cálculo. Nada nos garantiza todavía que estén bien: un signo o una transpuesta equivocados producen un «gradiente» que entrena mal, pero entrena, y el error puede pasar inadvertido mucho tiempo. Hay que verificarlo.
#
# ## 5. Verificación con diferencias finitas
#
# La derivada de $E$ respecto de un parámetro $\theta$ se puede aproximar sin saber nada de la red, evaluando la pérdida dos veces:
#
# $$\frac{\partial E}{\partial\theta} \approx \frac{E(\theta + h) - E(\theta - h)}{2h}\qquad\text{(diferencia centrada).}$$
#
# Por el desarrollo de Taylor, el error de truncamiento de la diferencia centrada es $O(h^2)$, contra $O(h)$ de la diferencia hacia adelante $\big(E(\theta + h) - E(\theta)\big)/h$. Comparamos con el **error relativo** $|g - \tilde g| / \max(|g|, |\tilde g|)$, que no depende de la escala del gradiente. Lo hacemos en aritmética de doble precisión con una red pequeña sobre unas lunas.

# %%
X_lunas, y_lunas = make_moons(400, noise=0.25, random_state=SEMILLA)
X_lunas_pru, y_lunas_pru = make_moons(2000, noise=0.25, random_state=SEMILLA + 1)   # datos nuevos para medir


def gradiente_numerico(p, X, y, act, h=1e-5):
    g = {}
    for nombre, M in p.items():
        g[nombre] = np.zeros_like(M)
        for i in np.ndindex(M.shape):
            original = M[i]
            M[i] = original + h; e_mas = entropia_cruzada(adelante(p, X, act)[0], y)
            M[i] = original - h; e_menos = entropia_cruzada(adelante(p, X, act)[0], y)
            M[i] = original
            g[nombre][i] = (e_mas - e_menos) / (2 * h)
    return g


p = inicializar(2, 8, 2, np.random.default_rng(SEMILLA), "tanh")
P, cache = adelante(p, X_lunas, "tanh")
g_atras = atras(p, cache, P, y_lunas, "tanh")
g_num = gradiente_numerico(p, X_lunas, y_lunas, "tanh")
for nombre in p:
    a, b = g_atras[nombre], g_num[nombre]
    rel = np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12))
    print(f"{nombre}: forma {str(a.shape):8}  error relativo máximo {rel:.1e}")

# %% [markdown]
# Errores relativos entre $10^{-11}$ y $10^{-8}$ en los 42 parámetros: la retropropagación está bien. (Como regla práctica, $10^{-7}$ o menos es correcto; $10^{-2}$ o más es un error en el código.)
#
# ### ¿Qué $h$ usar?
#
# No conviene un $h$ diminuto: al restar dos números casi iguales se pierden cifras. Con $\varepsilon \approx 10^{-16}$ la precisión de la máquina, el error de redondeo es del orden de $\varepsilon/h$, y compite con el de truncamiento. Veámoslo con un solo peso.

# %%
def error_derivada(h, centrada):
    M = p["W1"]; original = M[0, 0]
    M[0, 0] = original + h; e_mas = entropia_cruzada(adelante(p, X_lunas, "tanh")[0], y_lunas)
    M[0, 0] = original - h if centrada else original
    e_otro = entropia_cruzada(adelante(p, X_lunas, "tanh")[0], y_lunas)
    M[0, 0] = original
    aprox = (e_mas - e_otro) / (2 * h if centrada else h)
    return abs(aprox - g_atras["W1"][0, 0]) / abs(g_atras["W1"][0, 0])


hs = np.logspace(-1, -13, 49)
err_adelante = np.array([error_derivada(h, False) for h in hs])
err_centrada = np.array([error_derivada(h, True) for h in hs])

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(hs, err_adelante, marker="o", ms=4, color=ROJO)
ax.loglog(hs, err_centrada, marker="o", ms=4, color=AZUL)
ax.invert_xaxis()
ax.set_xlabel("paso h")
ax.set_ylabel("error relativo")
ax.set_title("Error de la derivada numérica de un peso")
ax.text(3e-2, err_adelante[2] * 2.5, "hacia adelante: O(h)", color=ROJO, fontsize=12)
ax.text(8e-2, 3e-9, "centrada: O(h²)", color=AZUL, fontsize=12)
ax.text(1e-10, 1e-9, "a la derecha domina\nel redondeo: ε / h", color=TENUE, fontsize=11, ha="center")
plt.show()
i_ad, i_ce = np.argmin(err_adelante), np.argmin(err_centrada)
print(f"mejor h hacia adelante: {hs[i_ad]:.0e} (error {err_adelante[i_ad]:.0e});  centrada: {hs[i_ce]:.0e} (error {err_centrada[i_ce]:.0e})")

# %% [markdown]
# La gráfica tiene forma de V. A la izquierda, al bajar $h$ diez veces, el error de la diferencia hacia adelante baja diez veces y el de la centrada, cien (pendientes 1 y 2). A la derecha manda el redondeo y el error vuelve a crecer como $1/h$. El mejor $h$ está cerca de $\sqrt\varepsilon \approx 10^{-8}$ hacia adelante y de $\varepsilon^{1/3} \approx 10^{-5}$ en la centrada, que además llega a un error unas mil veces menor.
#
# ### El costo: por qué no se entrena con diferencias finitas
#
# Las diferencias finitas necesitan **dos pasadas hacia adelante por parámetro**. La retropropagación obtiene **todos** los gradientes con una pasada hacia adelante y una hacia atrás, que cuesta lo mismo que unas dos pasadas hacia adelante (los mismos productos de matrices, transpuestos). Midámoslo con la red de los dígitos que usaremos en la sección 8, con 64 neuronas ocultas.

# %%
p_dig = inicializar(64, 64, 10, np.random.default_rng(SEMILLA), "relu")
n_param = sum(M.size for M in p_dig.values())

t0 = time.perf_counter()
for _ in range(20):
    P, cache = adelante(p_dig, Xd_ent, "relu"); g = atras(p_dig, cache, P, yd_ent, "relu")
t_atras = (time.perf_counter() - t0) / 20

t0 = time.perf_counter()
g_num_dig = gradiente_numerico(p_dig, Xd_ent, yd_ent, "relu")
t_num = time.perf_counter() - t0

dif = max(np.max(np.abs(g[k] - g_num_dig[k])) for k in g)
print(f"{n_param:,} parámetros → diferencias finitas: {2 * n_param:,} pasadas hacia adelante")
print(f"retropropagación: {1000 * t_atras:.1f} ms; diferencias finitas: {t_num:.1f} s "
      f"(unas {t_num / t_atras:,.0f} veces más); diferencia máxima entre ambos gradientes: {dif:.1e}")

# %% [markdown]
# Con 4,810 parámetros, las diferencias finitas hacen 9,620 pasadas por los datos y tardan miles de veces más que la retropropagación (la cifra exacta depende de la máquina). Una red moderna tiene millones o miles de millones de parámetros: sin la retropropagación no se podrían entrenar.
#
# Ésta es una propiedad general de la **diferenciación en modo inverso**: para una función $\mathbb{R}^P \to \mathbb{R}$ (muchos parámetros, una pérdida), el gradiente completo cuesta un múltiplo pequeño y constante de evaluar la función, sin importar $P$. El **modo directo**, que propaga derivadas hacia adelante junto con los valores, necesita una pasada por cada parámetro de entrada, como las diferencias finitas. El modo inverso para programas generales se describió por primera vez en la tesis de maestría de S. Linnainmaa (1970); su uso para entrenar redes se popularizó con D. Rumelhart, G. Hinton y R. Williams («Learning representations by back-propagating errors», *Nature* 323, 1986).
#
# ## 6. Entrenar la red en las lunas
#
# Con el gradiente en la mano, entrenar es descenso por gradiente. Usamos **minilotes**: en cada **época** barajamos los datos, los partimos en grupos de 20 y damos un paso con el gradiente de cada grupo (en la siguiente libreta estudiaremos por qué esto funciona mejor que usar todos los datos en cada paso).

# %%
def entrenar(p, X, y, act, eta, epocas, tam_lote, rng, guardar_en=()):
    """Descenso por gradiente con minilotes. Devuelve la pérdida de entrenamiento por época y copias de p."""
    historia, copias = [], {}
    if 0 in guardar_en:
        copias[0] = {k: M.copy() for k, M in p.items()}
    for epoca in range(1, epocas + 1):
        orden = rng.permutation(len(y))
        for i in range(0, len(y), tam_lote):
            lote = orden[i:i + tam_lote]
            P, cache = adelante(p, X[lote], act)
            g = atras(p, cache, P, y[lote], act)
            for k in p:
                p[k] -= eta * g[k]
        historia.append(entropia_cruzada(adelante(p, X, act)[0], y))
        if epoca in guardar_en:
            copias[epoca] = {k: M.copy() for k, M in p.items()}
    return historia, copias


def exactitud(p, X, y, act):
    return np.mean(adelante(p, X, act)[0].argmax(1) == y)


rng = np.random.default_rng(SEMILLA)
p_lunas = inicializar(2, 16, 2, rng, "tanh")
EPOCAS_FOTO = [0, 5, 20, 100, 300]
hist_lunas, fotos = entrenar(p_lunas, X_lunas, y_lunas, "tanh", eta=0.3, epocas=300, tam_lote=20, rng=rng, guardar_en=EPOCAS_FOTO)

log_lunas = LogisticRegression().fit(X_lunas, y_lunas)
print(f"red 2-16-2 (tanh): exactitud de entrenamiento {exactitud(p_lunas, X_lunas, y_lunas, 'tanh'):.3f}, "
      f"con datos nuevos {exactitud(p_lunas, X_lunas_pru, y_lunas_pru, 'tanh'):.3f}")
print(f"regresión logística: entrenamiento {log_lunas.score(X_lunas, y_lunas):.3f}, datos nuevos {log_lunas.score(X_lunas_pru, y_lunas_pru):.3f}")

# %%
g1, g2 = np.meshgrid(np.linspace(-2, 3, 250), np.linspace(-1.6, 2.1, 200))
malla = np.column_stack([g1.ravel(), g2.ravel()])


def dibujar_frontera(ax, prob, X, y, titulo):
    ax.contourf(g1, g2, prob.reshape(g1.shape), levels=[0, 0.5, 1], colors=REGIONES, alpha=0.8)
    ax.contour(g1, g2, prob.reshape(g1.shape), levels=[0.5], colors=TINTA, linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], s=12, c=np.where(y == 1, ROJO, AZUL), lw=0)
    ax.set_title(titulo, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


fig, ejes = plt.subplots(1, 5, figsize=(16, 3.6))
for ax, e in zip(ejes, EPOCAS_FOTO):
    dibujar_frontera(ax, adelante(fotos[e], malla, "tanh")[0][:, 1], X_lunas, y_lunas, f"época {e}")
fig.suptitle("La frontera de la red 2-16-2 durante el entrenamiento", x=0.01, ha="left", color=rdp.estilo.MARINO, weight="semibold")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, 301), hist_lunas, color=AZUL)
ax.axhline(log_loss_lunas := entropia_cruzada(log_lunas.predict_proba(X_lunas), y_lunas), color=GRIS, ls="--", lw=1.2)
ax.text(300, log_loss_lunas + 0.012, "regresión logística", color=GRIS, ha="right", fontsize=11)
ax.set_xscale("log")
ax.set_xlabel("época")
ax.set_ylabel("entropía cruzada (entrenamiento)")
ax.set_title("Curva de entrenamiento en las lunas")
plt.show()

# %% [markdown]
# Al principio (época 0, pesos al azar) la frontera es casi una recta: con pesos pequeños, $\tanh(z) \approx z$ y la red es casi lineal. Después de unas cuantas épocas ya se parece a la de la logística, y conforme avanza el entrenamiento se dobla para seguir la forma de las lunas. La red acierta en 94 % de los datos nuevos, contra 86 % de la regresión logística. Los picos de la curva de entrenamiento son el ruido de los minilotes con una tasa de aprendizaje grande; en la siguiente libreta veremos cómo controlarlo.
#
# ### Sin no linealidad no hay red
#
# ¿Y si quitamos la activación? Con $\sigma(z) = z$,
#
# $$W^{(2)}\big(W^{(1)}\mathbf x + \mathbf b^{(1)}\big) + \mathbf b^{(2)} = \underbrace{W^{(2)}W^{(1)}}_{W}\,\mathbf x + \underbrace{W^{(2)}\mathbf b^{(1)} + \mathbf b^{(2)}}_{\mathbf b}:$$
#
# la composición de funciones afines es afín, sin importar cuántas capas apilemos. Una red «profunda» sin activaciones no es más que una regresión *softmax* con una parametrización complicada. Comprobémoslo:

# %%
rng = np.random.default_rng(SEMILLA)
p_lineal = inicializar(2, 16, 2, rng, "identidad")
entrenar(p_lineal, X_lunas, y_lunas, "identidad", eta=0.3, epocas=300, tam_lote=20, rng=rng)
print(f"red 2-16-2 sin activación: exactitud de entrenamiento {exactitud(p_lineal, X_lunas, y_lunas, 'identidad'):.3f} "
      f"(logística: {log_lunas.score(X_lunas, y_lunas):.3f})")

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
dibujar_frontera(ejes[0], adelante(p_lineal, malla, "identidad")[0][:, 1], X_lunas, y_lunas, "16 neuronas ocultas, sin activación")
dibujar_frontera(ejes[1], adelante(p_lunas, malla, "tanh")[0][:, 1], X_lunas, y_lunas, "16 neuronas ocultas, tanh")
fig.tight_layout()
plt.show()

# %% [markdown]
# Mismas 16 neuronas, mismos datos: sin activación la frontera es una recta y la exactitud es prácticamente la de la logística (0.838 contra 0.843; la pequeña diferencia viene de que `LogisticRegression` está regularizada y de que nuestro descenso no llega exactamente al mínimo).
#
# ### Lo que aprende la capa oculta en el XOR
#
# Volvamos al XOR con una red de **dos** neuronas ocultas, para poder dibujar el espacio oculto $(a_1, a_2)$ en el plano.

# %%
rng = np.random.default_rng(SEMILLA)
p_xor = inicializar(2, 2, 2, rng, "tanh")
hist_xor, _ = entrenar(p_xor, Xx, yx, "tanh", eta=0.3, epocas=200, tam_lote=20, rng=rng)
_, (_, _, A_xor) = adelante(p_xor, Xx, "tanh")
print(f"red 2-2-2 en el XOR: exactitud {exactitud(p_xor, Xx, yx, 'tanh'):.3f}")
print("W1 =", p_xor["W1"].round(2).tolist(), " b1 =", p_xor["b1"].round(2).tolist())

fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
g1x, g2x = np.meshgrid(np.linspace(-2.2, 2.2, 200), np.linspace(-2.2, 2.2, 200))
prob = adelante(p_xor, np.column_stack([g1x.ravel(), g2x.ravel()]), "tanh")[0][:, 1].reshape(g1x.shape)
ejes[0].contourf(g1x, g2x, prob, levels=[0, 0.5, 1], colors=REGIONES, alpha=0.8)
ejes[0].scatter(*Xx.T, s=14, c=np.where(yx == 1, ROJO, AZUL), lw=0)
for j in range(2):   # rectas donde cada neurona oculta cambia de signo: z_j = 0
    (w1, w2), b0 = p_xor["W1"][j], p_xor["b1"][j]
    xs = np.array([-2.2, 2.2])
    ejes[0].plot(xs, -(b0 + w1 * xs) / w2, color=TENUE, ls="--", lw=1.2)
ejes[0].set(title="Espacio original: la frontera se dobla", xlabel="$x_1$", ylabel="$x_2$", xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
# en el espacio oculto la frontera es la recta donde z(2)_1 = z(2)_0
w = p_xor["W2"][1] - p_xor["W2"][0]; b = p_xor["b2"][1] - p_xor["b2"][0]
t = np.linspace(-1.05, 1.05, 2)
ejes[1].scatter(*A_xor.T, s=14, c=np.where(yx == 1, ROJO, AZUL), lw=0)
ejes[1].plot(t, -(b + w[0] * t) / w[1], color=TINTA, lw=1.5)
ejes[1].set(title="Espacio oculto: una recta basta", xlabel="$a_1 = \\tanh(z_1)$", ylabel="$a_2 = \\tanh(z_2)$",
            xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
for ax in ejes:
    ax.set_aspect("equal")
fig.tight_layout()
plt.show()

# %% [markdown]
# La red aprendió pesos fáciles de leer (las líneas punteadas son las rectas $z_1 = 0$ y $z_2 = 0$): aproximadamente $a_1 = \tanh\big(2(x_2 - x_1 + 1)\big)$ y $a_2 = \tanh\big(2(x_1 - x_2 + 1)\big)$. Cada neurona oculta es una **franja**: vale casi $+1$ de un lado de una recta paralela a la diagonal $x_1 = x_2$ y casi $-1$ del otro. Las dos nubes azules, que están sobre la diagonal, caen del lado positivo de ambas rectas y van a parar a la misma esquina $(1, 1)$ del espacio oculto; cada nube roja queda del lado negativo de una de las rectas y va a su propia esquina. La capa oculta **dobló** el plano para juntar lo que una recta no podía juntar. La última capa, una logística, hace el resto. Ésa es la idea central de las redes: **aprender una representación** en la que el problema se vuelva lineal.
#
# ## 7. ¿Qué tan expresiva es una capa oculta?
#
# El XOR y las lunas sugieren que una capa oculta puede dibujar fronteras bastante arbitrarias. El **teorema de aproximación universal** lo hace preciso: G. Cybenko (1989) demostró que, para la sigmoide, las funciones $\sum_{j=1}^H c_j\,\sigma(\mathbf w_j^\top\mathbf x + b_j)$ son densas en las funciones continuas sobre un compacto de $\mathbb{R}^d$: dada cualquier $f$ continua y cualquier $\varepsilon > 0$, existe una red de una capa oculta con $|g(\mathbf x) - f(\mathbf x)| < \varepsilon$ en todo el compacto. K. Hornik (1991) mostró que basta con que la activación sea acotada y no constante, y M. Leshno y colaboradores (1993) extendieron el resultado a toda activación que no sea un polinomio, lo que incluye a ReLU.
#
# En una dimensión hay una **construcción** sencilla con ReLU. Una sola neurona $\mathrm{ReLU}(x - t)$ es una rampa que empieza en $t$; una suma $g(x) = c_0 + \sum_i s_i\,\mathrm{ReLU}(x - t_i)$ es una función **lineal por pedazos** que cambia de pendiente en cada $t_i$ (en $s_i$). Tres rampas forman un «bulto» triangular,
#
# $$\mathrm{ReLU}(x - t_{i-1}) - 2\,\mathrm{ReLU}(x - t_i) + \mathrm{ReLU}(x - t_{i+1}),$$
#
# y sumando bultos de las alturas adecuadas obtenemos la **interpolación lineal** de $f$ en los nodos $t_0 < t_1 < \dots < t_H$. Para $f$ con segunda derivada acotada, el error de la interpolación lineal con nodos separados $\Delta$ es a lo más $\Delta^2\max|f''|/8$: con $H$ neuronas, el error baja como $1/H^2$.
#
# Probémoslo con una función de verdad: la temperatura media diaria de Hermosillo a lo largo del año.

# %%
temp = rdp.datos.cargar("temperatura_hermosillo", parse_dates=["fecha"])
temp = temp[~((temp.fecha.dt.month == 2) & (temp.fecha.dt.day == 29))]       # años de 365 días
temp["dia"] = temp.fecha.dt.day_of_year - ((temp.fecha.dt.is_leap_year) & (temp.fecha.dt.month > 2))
clima = temp.groupby("dia").tmed.mean()
dia, f_dia = clima.index.to_numpy(dtype=float), clima.to_numpy()
print(f"{temp.fecha.dt.year.min()}–{temp.fecha.dt.year.max()}: {temp.tmed.notna().sum():,} días con dato; "
      f"promedio por día del año entre {f_dia.min():.1f} y {f_dia.max():.1f} °C")


def red_relu_interpolante(x, y, H):
    """Red de una capa oculta con H neuronas ReLU que interpola linealmente a (x, y) en H + 1 nodos equiespaciados.

    Capa oculta: a_i = ReLU(x − t_i), con pesos 1 y sesgos −t_i (i = 0, …, H − 1).
    Capa de salida: g = c0 + Σ s_i a_i, donde s_i es el cambio de pendiente en t_i.
    """
    t = np.linspace(x.min(), x.max(), H + 1)
    v = np.interp(t, x, y)                             # alturas de la función en los nodos
    pendientes = np.diff(v) / np.diff(t)
    s = np.diff(np.concatenate([[0], pendientes]))     # cambio de pendiente en cada nodo
    return t[:-1], s, v[0]


def evaluar_relu(x, t, s, c0):
    return c0 + np.maximum(0, x[:, None] - t[None, :]) @ s


fig, ejes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
filas = []
for ax, H in zip(ejes, [3, 6, 12]):
    t, s, c0 = red_relu_interpolante(dia, f_dia, H)
    g = evaluar_relu(dia, t, s, c0)
    ax.scatter(dia, f_dia, s=6, color=GRIS, lw=0)
    ax.plot(dia, g, color=AZUL, lw=2.2)
    ax.scatter(np.append(t, dia.max()), evaluar_relu(np.append(t, dia.max()), t, s, c0), s=40, color=DORADO_OSC, zorder=3)
    ax.set_title(f"{H} neuronas ReLU · RECM {np.sqrt(np.mean((g - f_dia) ** 2)):.2f} °C", fontsize=12)
    ax.set_xlabel("día del año")
ejes[0].set_ylabel("temperatura media (°C)")
fig.suptitle("Temperatura media diaria en Hermosillo (1992–2024) y redes ReLU que la interpolan", x=0.01, ha="left",
             color=rdp.estilo.MARINO, weight="semibold")
fig.tight_layout()
plt.show()

for H in [1, 2, 3, 6, 12, 24, 48, 96]:
    t, s, c0 = red_relu_interpolante(dia, f_dia, H)
    filas.append({"neuronas": H, "RECM (°C)": np.sqrt(np.mean((evaluar_relu(dia, t, s, c0) - f_dia) ** 2))})
pd.DataFrame(filas).set_index("neuronas").T.round(2)

# %% [markdown]
# Con 3 neuronas la red ya distingue invierno, primavera y el verano de 33 a 35 °C; con 12 neuronas el error cuadrático medio es de alrededor de medio grado, comparable con el ruido que queda en el promedio de 33 años, y a partir de ahí agregar neuronas solo persigue ese ruido. Los pesos no se entrenaron: los **construimos**, con una neurona por nodo.
#
# **Lo que el teorema no dice.** (1) **Cuántas** neuronas hacen falta: en $d$ dimensiones, una construcción como ésta necesita una malla de nodos, del orden de $(1/\varepsilon)^{d/2}$ neuronas para un error $\varepsilon$; con las 64 dimensiones de los dígitos, es un número absurdo. (2) **Cómo encontrar** los pesos: el teorema dice que existen, no que el descenso por gradiente los encuentre. (3) Nada sobre **generalización**: una red que puede aproximar cualquier función también puede memorizar el ruido, y la Unidad 3 nos enseñó que la capacidad sin control es sobreajuste. Lo que sí dice es que una capa oculta suficientemente ancha no tiene un **sesgo** de aproximación irreducible, como sí lo tiene un modelo lineal.
#
# ## 8. Los dígitos con nuestra red
#
# Ahora el problema real. Una red 64-64-10 con ReLU, minilotes de 32, tasa de aprendizaje 0.1 y 40 épocas.

# %%
rng = np.random.default_rng(SEMILLA)
p_dig = inicializar(64, 64, 10, rng, "relu")
t0 = time.perf_counter()
hist_dig, _ = entrenar(p_dig, Xd_ent, yd_ent, "relu", eta=0.1, epocas=40, tam_lote=32, rng=rng)
exac_numpy = exactitud(p_dig, Xd_pru, yd_pru, "relu")
print(f"entrenamiento en {time.perf_counter() - t0:.1f} s; pérdida final {hist_dig[-1]:.4f}")
print(f"exactitud de entrenamiento {exactitud(p_dig, Xd_ent, yd_ent, 'relu'):.4f}; de prueba {exac_numpy:.4f} "
      f"({np.sum(adelante(p_dig, Xd_pru, 'relu')[0].argmax(1) != yd_pru)} errores)")

# %% [markdown]
# Nuestra red de unas cuantas líneas de NumPy llega a 0.978 de exactitud de prueba: 10 errores contra 13 de la logística. Es una mejora pequeña, y con 450 imágenes de prueba tres errores de diferencia no son una prueba contundente (cada error vale 0.2 puntos porcentuales). Los dígitos de $8 \times 8$ son un problema casi lineal; la ventaja de las redes aparece con imágenes más grandes y problemas más difíciles, como los que veremos en las siguientes libretas.
#
# ## 9. Lo mismo con PyTorch
#
# Escribir `atras` a mano para cada arquitectura no escala. Las bibliotecas de redes hacen **diferenciación automática** en modo inverso: cada operación sobre un tensor con `requires_grad=True` se anota en un **grafo de cómputo** que se construye mientras el código corre (un grafo **dinámico**: puede tener `if` y ciclos de Python), y `backward()` recorre ese grafo de atrás hacia adelante aplicando la regla de la cadena, exactamente como hicimos.
#
# ### El mismo gradiente
#
# Copiemos los pesos de nuestra red de las lunas (antes de entrenar) a tensores de doble precisión y comparemos.

# %%
p0 = inicializar(2, 8, 2, np.random.default_rng(SEMILLA), "tanh")      # los mismos pesos de la sección 5
P, cache = adelante(p0, X_lunas, "tanh")
g_mano = atras(p0, cache, P, y_lunas, "tanh")

pt = {k: torch.tensor(M, dtype=torch.float64, requires_grad=True) for k, M in p0.items()}
Xt, yt = torch.tensor(X_lunas, dtype=torch.float64), torch.tensor(y_lunas)
logits = torch.tanh(Xt @ pt["W1"].T + pt["b1"]) @ pt["W2"].T + pt["b2"]
perdida = F.cross_entropy(logits, yt)                # softmax + entropía cruzada, en una función estable
perdida.backward()

print(f"pérdida: a mano {entropia_cruzada(P, y_lunas):.15f}, PyTorch {perdida.item():.15f}")
for k in pt:
    print(f"{k}: diferencia máxima entre gradientes {np.max(np.abs(pt[k].grad.numpy() - g_mano[k])):.1e}")

# %% [markdown]
# Coinciden hasta el último decimal de la doble precisión. `F.cross_entropy` recibe los **logits** $\mathbf z^{(2)}$ (no las probabilidades) y combina *softmax* y logaritmo de forma numéricamente estable.
#
# ### Una red en PyTorch, con un ciclo de entrenamiento explícito
#
# Ahora la red de los dígitos en el estilo habitual de PyTorch: el modelo es un `nn.Module`, el optimizador se encarga del paso `p -= eta * g`, y el ciclo de entrenamiento se escribe a mano para que se vea cada paso. Las dos funciones `entrenar_epoca` y `evaluar` las reutilizaremos en toda la unidad.

# %%
class MLP(nn.Module):
    def __init__(self, d, H, K):
        super().__init__()
        self.oculta = nn.Linear(d, H)
        self.salida = nn.Linear(H, K)

    def forward(self, x):
        return self.salida(F.relu(self.oculta(x)))      # logits; la softmax va dentro de la pérdida


def entrenar_epoca(modelo, cargador, optimizador, perdida_fn):
    modelo.train()
    total, n = 0.0, 0
    for xb, yb in cargador:
        xb, yb = xb.to(dispositivo), yb.to(dispositivo)
        optimizador.zero_grad()              # los gradientes se acumulan: hay que borrarlos
        perdida = perdida_fn(modelo(xb), yb)
        perdida.backward()                   # retropropagación
        optimizador.step()                   # p ← p − η ∇p
        total += perdida.item() * len(yb); n += len(yb)
    return total / n


@torch.no_grad()
def evaluar(modelo, X, y):
    modelo.eval()
    pred = modelo(X.to(dispositivo)).argmax(1).cpu()
    return (pred == y).float().mean().item()


Xd_ent_t = torch.tensor(Xd_ent, dtype=torch.float32); yd_ent_t = torch.tensor(yd_ent)
Xd_pru_t = torch.tensor(Xd_pru, dtype=torch.float32); yd_pru_t = torch.tensor(yd_pru)
cargador = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xd_ent_t, yd_ent_t), batch_size=32, shuffle=True,
                                       generator=torch.Generator().manual_seed(SEMILLA))

torch.manual_seed(SEMILLA)
mlp = MLP(64, 64, 10).to(dispositivo)
optimizador = torch.optim.SGD(mlp.parameters(), lr=0.1)
print(mlp, f"\nparámetros: {sum(q.numel() for q in mlp.parameters()):,}")

t0 = time.perf_counter()
curva_torch = []
for epoca in range(40):
    curva_torch.append(entrenar_epoca(mlp, cargador, optimizador, nn.CrossEntropyLoss()))
exac_torch = evaluar(mlp, Xd_pru_t, yd_pru_t)
print(f"40 épocas en {time.perf_counter() - t0:.1f} s; exactitud de prueba {exac_torch:.4f}")

# %%
pd.DataFrame({"exactitud de prueba": [exac_logistica, exac_numpy, exac_torch],
              "errores (de 450)": [round((1 - e) * 450) for e in (exac_logistica, exac_numpy, exac_torch)]},
             index=["regresión logística (C = 10)", "MLP 64-64-10 en NumPy", "MLP 64-64-10 en PyTorch"]).round(4)

# %% [markdown]
# La red de PyTorch llega a 0.973 (12 errores), muy parecida a la nuestra; no es idéntica porque PyTorch inicializa los pesos con otra distribución (uniforme, con una varianza parecida a la de Xavier), baraja con otro generador y trabaja en precisión simple. Diferencias de uno o dos errores son ruido del entrenamiento: con otra semilla cambian.
#
# Veamos cuáles dígitos confunde.

# %%
with torch.no_grad():
    pred_t = mlp(Xd_pru_t.to(dispositivo)).argmax(1).cpu().numpy()
malos = np.flatnonzero(pred_t != yd_pru)
fig, ejes = plt.subplots(1, len(malos), figsize=(1.4 * len(malos), 2))
for ax, i in zip(np.atleast_1d(ejes), malos):
    ax.imshow(Xd_pru[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"{yd_pru[i]} → {pred_t[i]}", fontsize=12)
    ax.axis("off")
fig.suptitle("Errores de la red de PyTorch (dígito real → predicho)", x=0.01, ha="left", color=rdp.estilo.MARINO, weight="semibold")
fig.tight_layout()
plt.show()

# %% [markdown]
# Con $8 \times 8$ celdas, algunos de estos dígitos son ambiguos aun para una persona; otros son errores claros del modelo.
#
# ## 10. La inicialización importa
#
# ¿Por qué `inicializar` usa varianzas tan particulares? Pensemos en una capa con $n_{\text{in}}$ entradas, $z = \sum_{j=1}^{n_{\text{in}}} w_j a_j$, con pesos independientes de media cero, independientes de las activaciones. Entonces
#
# $$\operatorname{Var}(z) = \sum_j \operatorname{Var}(w_j a_j) = n_{\text{in}}\operatorname{Var}(w)\,\mathbb E[a^2] .$$
#
# Si $n_{\text{in}}\operatorname{Var}(w)$ es mayor que 1, la escala de las activaciones se **multiplica** en cada capa y explota exponencialmente con la profundidad; si es menor, se apaga. Para que no pase ninguna de las dos cosas:
#
# - con **tanh** (o sigmoide centrada), cerca de cero $a \approx z$ y $\mathbb E[a^2] \approx \operatorname{Var}(z_{\text{anterior}})$, así que queremos $n_{\text{in}}\operatorname{Var}(w) = 1$. Pidiendo lo mismo para los gradientes en la pasada hacia atrás (que usa $W^\top$, con $n_{\text{out}}$ en lugar de $n_{\text{in}}$), X. Glorot y Y. Bengio (2010) propusieron el compromiso $\operatorname{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$: la inicialización de **Xavier** o de Glorot;
# - con **ReLU**, la mitad de las $z$ se vuelven cero: si $z$ es simétrica, $\mathbb E[\mathrm{ReLU}(z)^2] = \tfrac12\operatorname{Var}(z)$. Para compensar, K. He y colaboradores (2015) propusieron $\operatorname{Var}(w) = 2/n_{\text{in}}$: la inicialización de **He**.
#
# Simulemos una red de 20 capas de 256 neuronas ReLU, alimentada con los dígitos de entrenamiento, y midamos la desviación estándar de las activaciones capa por capa con cuatro inicializaciones.

# %%
def activaciones_por_capa(X, capas, ancho, desv_w, act, rng):
    a, desv = X, []
    for _ in range(capas):
        W = rng.normal(0, desv_w(a.shape[1], ancho), (ancho, a.shape[1]))
        z = a @ W.T
        a = ACTIVACIONES[act][0](z)
        desv.append(a.std())
    return np.array(desv)


Xd_est = (Xd_ent - Xd_ent.mean(0)) / (Xd_ent.std(0) + 1e-8)     # entradas con varianza 1
INICIALIZACIONES = {
    "N(0, 0.01²)": lambda n_in, n_out: 0.01,
    "Xavier: 2/(n_in + n_out)": lambda n_in, n_out: np.sqrt(2 / (n_in + n_out)),
    "He: 2/n_in": lambda n_in, n_out: np.sqrt(2 / n_in),
    "N(0, 1)": lambda n_in, n_out: 1.0,
}
CAPAS = 20
desviaciones = {nombre: activaciones_por_capa(Xd_est, CAPAS, 256, f, "relu", np.random.default_rng(SEMILLA))
                for nombre, f in INICIALIZACIONES.items()}

fig, ax = plt.subplots(figsize=(10, 5.2))
for (nombre, d), color in zip(desviaciones.items(), [GRIS, DORADO_OSC, AZUL, ROJO]):
    ax.semilogy(range(1, CAPAS + 1), d, marker="o", ms=4, color=color)
    ax.text(CAPAS + 0.4, d[-1], nombre, color=TINTA, fontsize=11, va="center")
ax.set_xlim(0.5, CAPAS + 7)
ax.set_xlabel("capa")
ax.set_ylabel("desviación estándar de las activaciones")
ax.set_title("Red de 20 capas ReLU de 256 neuronas, con los dígitos como entrada")
plt.show()
pd.DataFrame({k: v[[0, 4, 9, 19]] for k, v in desviaciones.items()}, index=["capa 1", "capa 5", "capa 10", "capa 20"]).T

# %% [markdown]
# Con He la escala se mantiene constante a lo largo de las 20 capas. Con Xavier, pensada para tanh, cada capa ReLU multiplica la varianza por $\tfrac12$ y la desviación estándar decae como $2^{-\ell/2}$: en la capa 20 es unas 850 veces más chica que en la primera. Los pesos «chicos» $N(0, 0.01^2)$ dan $n_{\text{in}}\operatorname{Var}(w) = 256 \times 10^{-4} \approx 0.026$ y la señal desaparece en unas cuantas capas; los pesos $N(0,1)$ multiplican la varianza por $256/2 = 128$ en cada capa y las activaciones se disparan a $10^{20}$. Una red así no se puede entrenar: o no llega señal a la salida, o los números se desbordan.
#
# ### El desvanecimiento del gradiente
#
# La pasada hacia atrás multiplica en cada capa por $W^{(\ell)\top}$ y por $\sigma'(\mathbf z^{(\ell)})$. La derivada de la sigmoide, $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, vale a lo más $\tfrac14$ (en $z = 0$), así que aun con pesos bien escalados el gradiente se **encoge** al viajar hacia las primeras capas. Midámoslo con PyTorch: una red de 10 capas ocultas de 64 neuronas sobre un lote de dígitos, con sigmoide y con ReLU, comparando la norma del gradiente de la pérdida respecto de los pesos de cada capa.

# %%
def normas_gradiente(activacion, init, capas=10, ancho=64):
    torch.manual_seed(SEMILLA)
    lineales = [nn.Linear(64 if i == 0 else ancho, ancho) for i in range(capas)] + [nn.Linear(ancho, 10)]
    for capa in lineales:
        init(capa.weight); nn.init.zeros_(capa.bias)
    partes = []
    for capa in lineales[:-1]:
        partes += [capa, activacion()]
    red = nn.Sequential(*partes, lineales[-1])
    F.cross_entropy(red(Xd_ent_t[:256]), yd_ent_t[:256]).backward()
    return np.array([capa.weight.grad.norm().item() for capa in lineales[:-1]])


grad_sigmoide = normas_gradiente(nn.Sigmoid, nn.init.xavier_normal_)
grad_relu = normas_gradiente(nn.ReLU, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.semilogy(range(1, 11), grad_sigmoide, marker="o", color=ROJO)
ax.semilogy(range(1, 11), grad_relu, marker="o", color=AZUL)
ax.text(10.2, grad_sigmoide[-1] * 1.8, "sigmoide (Xavier)", color=TINTA, fontsize=11, va="center")
ax.text(10.2, grad_relu[-1] / 1.8, "ReLU (He)", color=TINTA, fontsize=11, va="center")
ax.set_xlim(0.5, 12.3)
ax.set_xticks(range(1, 11))
ax.set_xlabel("capa oculta (1 = la más cercana a la entrada)")
ax.set_ylabel("‖∂E/∂W‖ de la capa")
ax.set_title("Norma del gradiente por capa, en la inicialización")
plt.show()
print(f"sigmoide: la capa 10 recibe un gradiente {grad_sigmoide[-1] / grad_sigmoide[0]:,.0f} veces mayor que la capa 1")
print(f"ReLU:     cociente capa 10 / capa 1 = {grad_relu[-1] / grad_relu[0]:.1f}")

# %% [markdown]
# Con sigmoides, el gradiente de la primera capa es unas 455,000 veces menor que el de la última (cerca de un factor de 4 por capa, el inverso del máximo de $\sigma'$); con ReLU y la inicialización de He, todas las capas reciben gradientes de tamaño parecido. Con sigmoides, en el descenso por gradiente, las primeras capas prácticamente no aprenden. Éste fue uno de los obstáculos para entrenar redes profundas en los años noventa y dos mil. Las soluciones que veremos en el resto de la unidad atacan justo este producto de factores: activaciones ReLU (derivada 1 donde están activas), buena inicialización, normalización por lotes, conexiones residuales y, en las redes recurrentes, las compuertas de la LSTM.
#
# ## 11. Cierre
#
# Un perceptrón multicapa es una regresión *softmax* sobre atributos que la propia red aprende. Su entrenamiento descansa en tres ideas que acabamos de programar y verificar:
#
# - la **retropropagación** es la regla de la cadena organizada en forma matricial: $\boldsymbol\delta^{(L)} = \hat{\mathbf y} - \mathbf y$, $\boldsymbol\delta^{(\ell)} = (W^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}) \odot \sigma'(\mathbf z^{(\ell)})$, $\partial E/\partial W^{(\ell)} = \boldsymbol\delta^{(\ell)}\mathbf a^{(\ell-1)\top}$, y cuesta lo mismo que unas pocas pasadas hacia adelante;
# - todo gradiente programado a mano se **verifica** con diferencias centradas;
# - la **escala** de los pesos iniciales decide si la señal y el gradiente sobreviven a muchas capas.
#
# PyTorch automatiza la segunda pasada con diferenciación automática, y produce el mismo gradiente. En la siguiente libreta usaremos estas herramientas para entrenar redes más grandes en un problema más difícil, y nos ocuparemos de **cómo** descender: tasas de aprendizaje, momento, Adam y regularización.
#
# *Referencias:* F. Rosenblatt, «The perceptron: a probabilistic model for information storage and organization in the brain», *Psychological Review* 65 (1958). M. Minsky y S. Papert, *Perceptrons*, MIT Press (1969). D. E. Rumelhart, G. E. Hinton y R. J. Williams, «Learning representations by back-propagating errors», *Nature* 323 (1986). G. Cybenko, «Approximation by superpositions of a sigmoidal function», *Mathematics of Control, Signals and Systems* 2 (1989). K. Hornik, «Approximation capabilities of multilayer feedforward networks», *Neural Networks* 4 (1991). X. Glorot y Y. Bengio, «Understanding the difficulty of training deep feedforward neural networks», AISTATS (2010). K. He, X. Zhang, S. Ren y J. Sun, «Delving deep into rectifiers», ICCV (2015). I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press (2016), capítulo 6.

# %% [markdown]
# ## 12. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p01", "u06-p02", "u06-p03", "u06-p04"])

# %% [markdown]
# ## 13. Ejercicios
#
# 1. **Retropropagación con tres capas.** Agrega una segunda capa oculta a `adelante` y `atras` (red $d$-$H_1$-$H_2$-$K$) usando la recursión $\boldsymbol\delta^{(\ell)} = (W^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}) \odot \sigma'(\mathbf z^{(\ell)})$, y verifica tu gradiente con `gradiente_numerico`. ¿Cuál es el error relativo máximo?
# 2. **Sigmoide y entropía cruzada binaria.** Para una sola salida $\hat y = \sigma(z)$ con pérdida $E = -y\log\hat y - (1-y)\log(1-\hat y)$, demuestra que $\partial E/\partial z = \hat y - y$. ¿Qué pasaría con el gradiente si usáramos en cambio la pérdida cuadrática $(\hat y - y)^2/2$ y la neurona estuviera saturada ($\hat y \approx 0$ con $y = 1$)?
# 3. **Simetría.** Inicializa todos los pesos de la red de los dígitos en cero (o todos en el mismo valor) y entrénala. Demuestra que todas las neuronas ocultas reciben el mismo gradiente en cada paso y, por lo tanto, siguen siendo iguales. ¿Qué exactitud alcanza? ¿Por qué no pasa lo mismo con los sesgos de la capa de salida?
# 4. **Aproximación en 2D.** Construye a mano una función «meseta» de $\mathbb{R}^2$ que valga aproximadamente 1 dentro del cuadrado $[0,1]^2$ y 0 lejos de él, sumando neuronas ReLU de la forma $\mathrm{ReLU}(\pm x_i + b)$ y una ReLU final. ¿Cuántas neuronas usaste? Estima cuántas harían falta para aproximar una función suave en $[0,1]^d$ con una malla de $m$ nodos por eje.
# 5. **Anchura contra exactitud.** Entrena la red de los dígitos con $H = 4, 8, 16, 32, 64, 128$ neuronas ocultas (con PyTorch) y grafica la exactitud de entrenamiento y de prueba contra $H$. ¿A partir de qué anchura la red supera a la logística? ¿Ves sobreajuste? Relaciónalo con la discusión de sesgo y varianza de la Unidad 3.
