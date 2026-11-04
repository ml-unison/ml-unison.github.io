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
# # Una red neuronal con PyTorch: grafos de cómputo y diferenciación automática
#
# **Unidad 6 · Ejemplo 1**
#
# Todos los modelos de la Unidad 4 eran lineales en algún espacio: en los atributos, en los log-momios o, con un kernel, en un espacio fijo que elegíamos nosotros. Una **red neuronal** da el paso siguiente: **aprende** la transformación de los datos al mismo tiempo que el modelo lineal que va encima. En esta libreta vamos a:
#
# 1. ver con el XOR por qué una sola neurona no basta y qué hace la capa oculta con el espacio;
# 2. entender una red como **ingeniería de características aprendida** más un **modelo lineal generalizado** en la última capa, y por qué eso decide la función de pérdida;
# 3. construir un **grafo de cómputo** y calcular su gradiente con un motor de **diferenciación automática** de 40 líneas, y compararlo con `torch.autograd`;
# 4. medir por qué el modo inverso (la **retropropagación**) cambió la historia: cuesta lo mismo con 10 parámetros que con 10,000;
# 5. entrenar redes en PyTorch en dos dimensiones, aproximar la temperatura de Hermosillo con neuronas ReLU y resolver los dígitos escritos a mano;
# 6. entender qué son las inicializaciones de **Xavier** y de **He**, cómo se sortean los pesos y qué es el **desvanecimiento del gradiente**.

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
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F

from sklearn.datasets import make_moons, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

import rdp

rdp.estilo.usar()
SEMILLA = 2011
torch.manual_seed(SEMILLA)
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE
REGIONES = ["#C0CEE3", "#F3C7C4"]   # tintas de azul y rojo para las regiones de decisión
print(f"PyTorch {torch.__version__}")

# %% [markdown]
# Todas las redes de esta libreta son pequeñas y se entrenan en segundos en el procesador (CPU), así que no usamos GPU: así los resultados son reproducibles en cualquier máquina.
#
# ## 1. Los datos y el problema
#
# Usaremos tres conjuntos de datos, cada uno para una pregunta distinta.
#
# - **Dos problemas de juguete en el plano**: el XOR (cuatro nubes de puntos en las esquinas de un cuadrado, con clases cruzadas) y las «lunas» de `make_moons`, dos medias lunas entrelazadas con ruido. Son sintéticos a propósito: en dos dimensiones podemos **dibujar** lo que hace la red.
# - **La temperatura media diaria de Hermosillo**, medida en el aeropuerto de 1992 a 2024 (red GHCN-Daily de la NOAA; Menne et al., 2012). Promediando los 33 años por día del año obtenemos la curva típica del calor sonorense, una función real de una variable que intentaremos aproximar con neuronas.
# - **Los dígitos escritos a mano** de la Unidad 4: $M = 1{,}797$ imágenes de $8 \times 8$ celdas con intensidades de 0 a 16 (Alpaydin y Kaynak, UCI, 1998), es decir, $n = 64$ atributos. Usamos **la misma partición** que en la Unidad 4: 25 % para prueba, estratificada, con `random_state=2011`.
#
# **Problema.** Reconocer el dígito (0 a 9) a partir de sus 64 intensidades con un perceptrón multicapa, y compararlo con la regresión logística multinomial, que en la Unidad 4 acertó en 97 % de las imágenes de prueba. En el camino, entender qué hace PyTorch cuando calcula un gradiente.

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
# $$a_1 = \mathrm{ReLU}(x_1 + x_2), \qquad a_2 = \mathrm{ReLU}(x_1 + x_2 - 1), \qquad \hat y = a_1 - 2 a_2 .$$
#
# Las esquinas $(0,0), (0,1), (1,0), (1,1)$ van a $(a_1, a_2) = (0,0), (1,0), (1,0), (2,1)$ y $\hat y = 0, 1, 1, 0$. La capa oculta **dobla el plano**: manda las dos esquinas de la clase 1 al mismo punto, y en el espacio $(a_1, a_2)$ las clases ya se separan con una recta.

# %%
esquinas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
a = np.maximum(0, np.column_stack([esquinas.sum(1), esquinas.sum(1) - 1]))
pd.DataFrame({"x1": esquinas[:, 0], "x2": esquinas[:, 1], "a1": a[:, 0], "a2": a[:, 1], "ŷ = a1 − 2 a2": a[:, 0] - 2 * a[:, 1]})

# %% [markdown]
# Lo interesante es que esa transformación no hay que diseñarla: se puede **aprender**. Para eso necesitamos tres cosas: la red, una pérdida y una forma de calcular su gradiente.
#
# ## 3. Una red es ingeniería de características más un modelo lineal generalizado
#
# Un perceptrón multicapa (*multilayer perceptron*, MLP) con una capa oculta, para $K$ clases, calcula para un ejemplo $\mathbf x^{(i)} \in \mathbb R^n$
#
# $$\mathbf z^{[1]} = \mathbf W^{[1]}\mathbf x^{(i)} + \mathbf b^{[1]}, \quad
# \mathbf a^{[1]} = g\big(\mathbf z^{[1]}\big), \quad
# \mathbf z^{[2]} = \mathbf W^{[2]}\mathbf a^{[1]} + \mathbf b^{[2]}, \quad
# \hat{\mathbf y} = \operatorname{softmax}\big(\mathbf z^{[2]}\big),$$
#
# donde $\mathbf W^{[1]}$ es de $H \times n$ ($H$ neuronas ocultas), $\mathbf W^{[2]}$ es de $K \times H$, los $\mathbf b^{[l]}$ son los sesgos de la capa $l$ y $g$ es una **función de activación** que se aplica entrada por entrada: tangente hiperbólica, sigmoide o ReLU.
#
# Conviene leer la red en **dos partes**:
#
# - Las capas ocultas son una función $\boldsymbol\phi_\theta(\mathbf x) = \mathbf a^{[1]}$ que lleva los datos de un espacio a otro. Es exactamente lo que hacíamos a mano en las unidades anteriores cuando inventábamos atributos (cuadrados, productos, un kernel), solo que ahora los parámetros $\theta = (\mathbf W^{[1]}, \mathbf b^{[1]})$ de esa transformación **se aprenden**. Con más capas, la transformación es una cadena de espacios: $\mathbf x \to \mathbf a^{[1]} \to \mathbf a^{[2]} \to \cdots$.
# - La última capa es un **modelo lineal generalizado** sobre esos atributos aprendidos: aquí, la regresión *softmax* de la Unidad 4. Para regresión sería una lineal; para conteos, una de Poisson.
#
# Como la última capa es un GLM, la pérdida natural es la misma de siempre: menos el logaritmo de la verosimilitud de ese GLM. Para clasificación, la **entropía cruzada** promedio sobre los $M$ ejemplos,
#
# $$L(\theta) = \frac1M\sum_{i=1}^M \ell^{(i)}, \qquad \ell^{(i)} = -\ln \hat y^{(i)}_{y^{(i)}} ,$$
#
# con $\hat y^{(i)}_{y^{(i)}}$ la probabilidad que la red le da a la clase correcta del ejemplo $i$. En PyTorch, la red se escribe así (con `nn.Linear` cada capa guarda $\mathbf W$ con forma (salidas, entradas) y calcula $\mathbf X\mathbf W^\top + \mathbf b$ con un ejemplo por **renglón**):

# %%
def red_mlp(n, H, K, activacion=nn.Tanh, inicio="xavier", semilla=SEMILLA):
    """MLP n-H-K. Devuelve los logits z[2]: la softmax va dentro de la pérdida."""
    torch.manual_seed(semilla)
    red = nn.Sequential(nn.Linear(n, H), activacion(), nn.Linear(H, K))
    for capa in (red[0], red[2]):
        if inicio == "xavier":
            nn.init.xavier_normal_(capa.weight)
        elif inicio == "he":
            nn.init.kaiming_normal_(capa.weight, nonlinearity="relu")
        nn.init.zeros_(capa.bias)
    return red


red = red_mlp(2, 3, 2)
print(red)
print("parámetros:", {nombre: tuple(p.shape) for nombre, p in red.named_parameters()})

# %% [markdown]
# `red[0]` y `red[1]` son $\boldsymbol\phi_\theta$; `red[2]` es la regresión *softmax*. Las opciones `"xavier"` y `"he"` las explicamos en la sección 10. Antes, el ingrediente que hace posible todo lo demás: el gradiente de $L$ respecto de todos los parámetros.
#
# ## 4. Grafos de cómputo y diferenciación automática
#
# Una red, por grande que sea, es una **composición de operaciones simples**: sumas, productos, $\tanh$, $\exp$, $\ln$. Si anotamos cada operación como un nodo y cada resultado intermedio como una arista, obtenemos un **grafo de cómputo**. Tomemos la red más pequeña posible, una neurona logística con un atributo, en un ejemplo con $x = 1.5$ y $y = 1$, y pesos $w = -0.8$, $b = 0.4$:
#
# $$u = w\,x, \qquad z = u + b, \qquad \hat y = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \ell = -\ln \hat y .$$
#
# **Hacia adelante** se evalúa cada nodo en orden. **Hacia atrás** se calcula, para cada valor intermedio $v$, su **adjunto** $\bar v = \partial \ell / \partial v$, empezando con $\bar\ell = 1$ y usando solo la derivada **local** de cada nodo:
#
# $$\bar{\hat y} = -\frac{1}{\hat y}, \qquad \bar z = \bar{\hat y}\,\hat y(1 - \hat y), \qquad \bar u = \bar b = \bar z, \qquad \bar w = \bar u\, x .$$
#
# Si un valor alimenta a varios nodos, sus adjuntos **se suman** (es la regla de la cadena para varias variables). Eso es todo lo que hace falta para un motor de diferenciación automática en **modo inverso**. Aquí está uno completo para números escalares: cada `Valor` recuerda de qué valores salió y cómo repartir su adjunto entre ellos.

# %%
class Valor:
    """Un número que recuerda cómo se calculó, para poder derivar hacia atrás."""

    def __init__(self, dato, hijos=(), op=""):
        self.dato, self.grad = float(dato), 0.0
        self.hijos, self.op = hijos, op
        self._atras = lambda: None                     # cómo repartir self.grad entre los hijos

    def __add__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        sal = Valor(self.dato + otro.dato, (self, otro), "+")
        def _atras():                                  # + reparte el adjunto sin cambiarlo
            self.grad += sal.grad
            otro.grad += sal.grad
        sal._atras = _atras
        return sal

    def __mul__(self, otro):
        otro = otro if isinstance(otro, Valor) else Valor(otro)
        sal = Valor(self.dato * otro.dato, (self, otro), "×")
        def _atras():                                  # × intercambia: d(ab)/da = b
            self.grad += otro.dato * sal.grad
            otro.grad += self.dato * sal.grad
        sal._atras = _atras
        return sal

    __radd__, __rmul__ = __add__, __mul__

    def sigmoide(self):
        s = 1 / (1 + math.exp(-self.dato))
        sal = Valor(s, (self,), "σ")
        def _atras():
            self.grad += s * (1 - s) * sal.grad
        sal._atras = _atras
        return sal

    def tanh(self):
        t = math.tanh(self.dato)
        sal = Valor(t, (self,), "tanh")
        def _atras():
            self.grad += (1 - t**2) * sal.grad
        sal._atras = _atras
        return sal

    def menos_ln(self):
        sal = Valor(-math.log(self.dato), (self,), "−ln")
        def _atras():
            self.grad += -1 / self.dato * sal.grad
        sal._atras = _atras
        return sal

    def backward(self):
        """Ordena el grafo (orden topológico) y recorre los nodos del último al primero."""
        orden, vistos = [], set()
        def visitar(v):
            if v not in vistos:
                vistos.add(v)
                for h in v.hijos:
                    visitar(h)
                orden.append(v)
        visitar(self)
        self.grad = 1.0                                # el adjunto de la salida es 1
        for v in reversed(orden):
            v._atras()

    def __repr__(self):
        return f"Valor(dato={self.dato:.4f}, grad={self.grad:.4f})"


w, b, x = Valor(-0.8), Valor(0.4), 1.5
u = w * x
z = u + b
y_gorro = z.sigmoide()
perdida = y_gorro.menos_ln()
perdida.backward()
for nombre, v in [("w", w), ("b", b), ("u", u), ("z", z), ("ŷ", y_gorro), ("ℓ", perdida)]:
    print(f"{nombre:>2}: valor {v.dato:8.4f}   adjunto ∂ℓ/∂{nombre} = {v.grad:8.4f}")

# %% [markdown]
# El adjunto de $z$ es $-0.690 = \hat y - y$: «predicción menos dato», como en la regresión logística. Y el gradiente que necesitamos para entrenar está en los adjuntos de las hojas: $\partial\ell/\partial w = -1.035$ y $\partial\ell/\partial b = -0.690$.
#
# **Lo mismo con PyTorch.** Un tensor con `requires_grad=True` hace exactamente lo que hace nuestro `Valor`: cada operación con él crea un nodo del grafo, que queda guardado en el atributo `grad_fn` del resultado, y `backward()` recorre ese grafo hacia atrás.

# %%
w_t = torch.tensor(-0.8, requires_grad=True)
b_t = torch.tensor(0.4, requires_grad=True)
perdida_t = -torch.log(torch.sigmoid(w_t * 1.5 + b_t))
perdida_t.backward()
print(f"PyTorch: ℓ = {perdida_t.item():.4f}, ∂ℓ/∂w = {w_t.grad.item():.4f}, ∂ℓ/∂b = {b_t.grad.item():.4f}")


def mostrar_grafo(nodo, sangria=0):
    """Recorre el grafo que guardó PyTorch, desde la pérdida hacia las hojas."""
    if nodo is None:
        return
    print("   " * sangria + type(nodo).__name__)
    for hijo, _ in nodo.next_functions:
        mostrar_grafo(hijo, sangria + 1)


mostrar_grafo(perdida_t.grad_fn)

# %% [markdown]
# El grafo que imprime PyTorch es el nuestro, leído de la pérdida hacia atrás: `NegBackward0` (el signo menos), `LogBackward0`, `SigmoidBackward0`, `AddBackward0` y `MulBackward0`; las hojas `AccumulateGrad` son los lugares donde PyTorch **acumula** el gradiente de $w$ y de $b$ en su atributo `.grad`.
#
# Tres detalles de PyTorch que se siguen de este diseño, y que causan muchos errores:
#
# - **El grafo es dinámico**: se arma mientras el código de Python corre, así que puede tener `if`, ciclos y llamadas a funciones; cada evaluación construye un grafo nuevo.
# - **Los gradientes se acumulan** (el `+=` de nuestro motor): si llamamos `backward()` dos veces sin borrar, `.grad` tiene la suma. Por eso todo ciclo de entrenamiento empieza con `optimizador.zero_grad()`.
# - Lo que se calcula dentro de `with torch.no_grad():` no se anota en el grafo; se usa para actualizar los pesos y para evaluar.

# %%
x_t = torch.tensor(3.0, requires_grad=True)
for _ in range(2):
    (x_t * x_t + x_t).backward()                       # f(x) = x² + x, f'(3) = 7
    print(f"después de backward(): x.grad = {x_t.grad.item()}")
x_t.grad = None                                        # lo que hace zero_grad()

v = Valor(3.0)                                         # el mismo caso en nuestro motor: x se usa tres veces
f = v * v + v
f.backward()
print(f"nuestro motor: f'(3) = {v.grad}")

# %% [markdown]
# La segunda llamada da 14, no 7: el gradiente se sumó al anterior. Y en $f(x) = x\cdot x + x$, la variable $x$ alimenta tres veces al grafo; los tres adjuntos se suman y dan $2x + 1 = 7$.
#
# **No solo en PyTorch.** TensorFlow (`tf.GradientTape`) y JAX (`jax.grad`) hacen lo mismo con otra interfaz: TensorFlow «graba» las operaciones en una cinta, y JAX transforma una función de Python en la función que calcula su gradiente. Las tres bibliotecas pueden además **compilar** el grafo (`torch.compile`, `tf.function`, `jax.jit`) para que corra más rápido. Saber usar una de ellas a fondo es lo que permite escribir cualquier arquitectura nueva: uno escribe la pasada hacia adelante y el gradiente sale gratis. En el curso usamos PyTorch.
#
# ## 5. Por qué el modo inverso cambió la historia: el costo
#
# Sin diferenciación automática hay otra forma de obtener un gradiente, sin saber nada de la red: las **diferencias finitas**,
#
# $$\frac{\partial L}{\partial \theta_j} \approx \frac{L(\theta + h\,\mathbf e_j) - L(\theta - h\,\mathbf e_j)}{2h},$$
#
# con $\mathbf e_j$ el vector que vale 1 en la posición $j$ y 0 en las demás, y $h$ un paso pequeño. Son **dos evaluaciones de la pérdida por parámetro**. El modo inverso obtiene **todos** los gradientes con una pasada hacia adelante y una hacia atrás. Midámoslo con la red de los dígitos, 64-64-10.

# %%
Xd_ent_t = torch.tensor(Xd_ent, dtype=torch.float64); yd_ent_t = torch.tensor(yd_ent)
red_dig = red_mlp(64, 64, 10, nn.ReLU, "he").double()
params = list(red_dig.parameters())
n_param = sum(p.numel() for p in params)

t0 = time.perf_counter()
for _ in range(20):
    red_dig.zero_grad()
    F.cross_entropy(red_dig(Xd_ent_t), yd_ent_t).backward()
t_inverso = (time.perf_counter() - t0) / 20
g_inverso = [p.grad.clone() for p in params]


@torch.no_grad()
def gradiente_numerico(red, X, y, h=1e-6):
    gradientes = []
    for p in red.parameters():
        g, plano = torch.zeros_like(p), p.view(-1)
        for j in range(plano.numel()):
            original = plano[j].item()
            plano[j] = original + h; L_mas = F.cross_entropy(red(X), y).item()
            plano[j] = original - h; L_menos = F.cross_entropy(red(X), y).item()
            plano[j] = original
            g.view(-1)[j] = (L_mas - L_menos) / (2 * h)
        gradientes.append(g)
    return gradientes


t0 = time.perf_counter()
g_numerico = gradiente_numerico(red_dig, Xd_ent_t, yd_ent_t)
t_numerico = time.perf_counter() - t0

dif = max((a - b).abs().max().item() for a, b in zip(g_inverso, g_numerico))
print(f"{n_param:,} parámetros → diferencias finitas: {2 * n_param:,} evaluaciones de la pérdida")
print(f"modo inverso: {1000 * t_inverso:.1f} ms; diferencias finitas: {t_numerico:.1f} s "
      f"(unas {t_numerico / t_inverso:,.0f} veces más)")
print(f"diferencia máxima entre los dos gradientes: {dif:.1e}")

# %% [markdown]
# Con 4,810 parámetros, las diferencias finitas tardan miles de veces más (la cifra exacta depende de la máquina) y dan el mismo gradiente salvo errores de redondeo. Una red moderna tiene millones o miles de millones de parámetros: sin el modo inverso no se podría entrenar. Las diferencias finitas solo sirven para **verificar**, en redes diminutas, una operación nueva cuyo gradiente escribimos a mano; PyTorch trae la función `torch.autograd.gradcheck` para eso.
#
# **Modo directo contra modo inverso.** La diferenciación automática también se puede hacer **hacia adelante**: se propaga, junto con cada valor, su derivada respecto de **un** parámetro. Eso da una columna del jacobiano por pasada, así que para el gradiente de $L: \mathbb R^P \to \mathbb R$ hacen falta $P$ pasadas, igual que con las diferencias finitas. El modo inverso da un **renglón** del jacobiano por pasada, y la pérdida tiene un solo renglón. PyTorch tiene los dos modos (`torch.func.jacfwd` y `torch.func.jacrev`); comparémoslos en una pérdida logística con $P$ parámetros y $M = 500$ ejemplos.

# %%
from torch.func import jacfwd, jacrev


def tiempo(f, *args, repeticiones=5):
    f(*args)                                            # una vez para calentar
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        f(*args)
    return (time.perf_counter() - t0) / repeticiones


generador = torch.Generator().manual_seed(SEMILLA)
filas_costo = []
for P in [10, 30, 100, 300, 1000, 3000]:
    X_c = torch.randn(500, P, generator=generador)
    y_c = (torch.rand(500, generator=generador) < 0.5).float()
    perdida_logistica = lambda w: F.binary_cross_entropy_with_logits(X_c @ w, y_c)
    w0 = torch.zeros(P)
    filas_costo.append({"P": P, "modo directo (ms)": 1000 * tiempo(jacfwd(perdida_logistica), w0),
                        "modo inverso (ms)": 1000 * tiempo(jacrev(perdida_logistica), w0)})
costo = pd.DataFrame(filas_costo).set_index("P")

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.loglog(costo.index, costo["modo directo (ms)"], marker="o", color=ROJO)
ax.loglog(costo.index, costo["modo inverso (ms)"], marker="o", color=AZUL)
ax.text(costo.index[-1] * 1.15, costo["modo directo (ms)"].iloc[-1], "modo directo", color=TINTA, va="center")
ax.text(costo.index[-1] * 1.15, costo["modo inverso (ms)"].iloc[-1], "modo inverso", color=TINTA, va="center")
ax.set_xlim(7, 12000)
ax.set_xlabel("número de parámetros P")
ax.set_ylabel("tiempo del gradiente (ms)")
ax.set_title("El gradiente de una pérdida: modo directo contra inverso")
plt.show()
costo.round(2)

# %% [markdown]
# El costo del modo directo crece con $P$; el del modo inverso casi no cambia. Ése es el resultado general: para una función de muchos parámetros y **una** salida, el gradiente completo en modo inverso cuesta un múltiplo pequeño y constante de evaluar la función. El modo inverso para programas generales lo describió S. Linnainmaa en su tesis de maestría (1970), P. Werbos lo propuso para redes (1974) y D. Rumelhart, G. Hinton y R. Williams lo popularizaron con el nombre de **retropropagación** («Learning representations by back-propagating errors», *Nature* 323, 1986).
#
# Aplicado a una red, el modo inverso da una recursión de matrices que vale la pena reconocer: si $\boldsymbol\delta^{[l]} = \partial \ell / \partial \mathbf z^{[l]}$ es el adjunto de la entrada lineal de la capa $l$, entonces
#
# $$\boldsymbol\delta^{[l]} = \big(\mathbf W^{[l+1]\top}\boldsymbol\delta^{[l+1]}\big) \odot g'\big(\mathbf z^{[l]}\big), \qquad \frac{\partial \ell}{\partial \mathbf W^{[l]}} = \boldsymbol\delta^{[l]}\,\mathbf a^{[l-1]\top},$$
#
# con $\odot$ el producto entrada por entrada y $\mathbf a^{[0]} = \mathbf x^{(i)}$: el error viaja hacia atrás por las mismas matrices que la señal hacia adelante, transpuestas. Nadie la tiene que programar ya: es lo que hace `backward()` con los nodos `AddmmBackward0` y `TanhBackward0`.
#
# ## 6. Entrenar en el plano
#
# Con el gradiente resuelto, entrenar es **descenso por gradiente con minilotes**: en cada **época** barajamos los datos, los partimos en grupos de 20 y damos un paso $\theta \leftarrow \theta - \eta\,\nabla_\theta L_{\text{lote}}$ por grupo, con tasa de aprendizaje $\eta$. El ciclo de PyTorch es siempre el mismo: borrar gradientes, pasada hacia adelante, `backward()`, paso del optimizador.

# %%
def entrenar(red, X, y, eta, epocas, tam_lote, guardar_en=()):
    """SGD con minilotes. Devuelve la pérdida de entrenamiento por época y copias de los pesos."""
    X, y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y)
    cargador = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=tam_lote, shuffle=True,
                                           generator=torch.Generator().manual_seed(SEMILLA))
    optimizador = torch.optim.SGD(red.parameters(), lr=eta)
    historia, copias = [], {}
    if 0 in guardar_en:
        copias[0] = {k: v.clone() for k, v in red.state_dict().items()}
    for epoca in range(1, epocas + 1):
        for xb, yb in cargador:
            optimizador.zero_grad()                     # los gradientes se acumulan: hay que borrarlos
            F.cross_entropy(red(xb), yb).backward()     # pérdida y retropropagación
            optimizador.step()                          # θ ← θ − η ∇θ
        with torch.no_grad():
            historia.append(F.cross_entropy(red(X), y).item())
        if epoca in guardar_en:
            copias[epoca] = {k: v.clone() for k, v in red.state_dict().items()}
    return historia, copias


@torch.no_grad()
def probabilidad(red, X):
    return F.softmax(red(torch.as_tensor(X, dtype=torch.float32)), dim=1).numpy()


def exactitud(red, X, y):
    return float(np.mean(probabilidad(red, X).argmax(1) == y))


X_lunas, y_lunas = make_moons(400, noise=0.25, random_state=SEMILLA)
X_lunas_pru, y_lunas_pru = make_moons(2000, noise=0.25, random_state=SEMILLA + 1)   # datos nuevos para medir

red_lunas = red_mlp(2, 16, 2, nn.Tanh)
EPOCAS_FOTO = [0, 5, 20, 100, 300]
hist_lunas, fotos = entrenar(red_lunas, X_lunas, y_lunas, eta=0.3, epocas=300, tam_lote=20, guardar_en=EPOCAS_FOTO)

log_lunas = LogisticRegression().fit(X_lunas, y_lunas)
print(f"red 2-16-2 (tanh): exactitud de entrenamiento {exactitud(red_lunas, X_lunas, y_lunas):.3f}, "
      f"con datos nuevos {exactitud(red_lunas, X_lunas_pru, y_lunas_pru):.3f}")
print(f"regresión logística: entrenamiento {log_lunas.score(X_lunas, y_lunas):.3f}, "
      f"datos nuevos {log_lunas.score(X_lunas_pru, y_lunas_pru):.3f}")

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
copia = red_mlp(2, 16, 2, nn.Tanh)
for ax, e in zip(ejes, EPOCAS_FOTO):
    copia.load_state_dict(fotos[e])
    dibujar_frontera(ax, probabilidad(copia, malla)[:, 1], X_lunas, y_lunas, f"época {e}")
fig.suptitle("La frontera de la red 2-16-2 durante el entrenamiento", x=0.01, ha="left", color=rdp.estilo.MARINO, weight="semibold")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, 301), hist_lunas, color=AZUL)
perdida_log_lunas = -np.mean(np.log(log_lunas.predict_proba(X_lunas)[np.arange(len(y_lunas)), y_lunas]))
ax.axhline(perdida_log_lunas, color=GRIS, ls="--", lw=1.2)
ax.text(300, perdida_log_lunas + 0.012, "regresión logística", color=GRIS, ha="right", fontsize=11)
ax.set_xscale("log")
ax.set_xlabel("época")
ax.set_ylabel("entropía cruzada (entrenamiento)")
ax.set_title("Curva de entrenamiento en las lunas")
plt.show()

# %% [markdown]
# Al principio (época 0, pesos al azar) la frontera es casi una recta: con pesos pequeños, $\tanh(z) \approx z$ y la red es casi lineal. Conforme avanza el entrenamiento se dobla para seguir la forma de las lunas, y la red acierta en más datos nuevos que la regresión logística.
#
# **La última capa es una regresión logística.** Si congelamos la capa oculta y le damos sus 16 activaciones $\mathbf a^{[1]}$ a una `LogisticRegression` de scikit-learn, como si fueran atributos que alguien diseñó, obtenemos prácticamente la misma exactitud que la red completa. Toda la «inteligencia» del cambio está en la representación.

# %%
@torch.no_grad()
def capa_oculta(red, X):
    return red[1](red[0](torch.as_tensor(X, dtype=torch.float32))).numpy()     # a[1] = g(W[1] x + b[1])


A_lunas, A_lunas_pru = capa_oculta(red_lunas, X_lunas), capa_oculta(red_lunas, X_lunas_pru)
log_sobre_a = LogisticRegression(C=1e4, max_iter=5000).fit(A_lunas, y_lunas)
print(f"logística sobre x (2 atributos):       datos nuevos {log_lunas.score(X_lunas_pru, y_lunas_pru):.3f}")
print(f"logística sobre a[1] (16 atributos):   datos nuevos {log_sobre_a.score(A_lunas_pru, y_lunas_pru):.3f}")
print(f"red completa:                          datos nuevos {exactitud(red_lunas, X_lunas_pru, y_lunas_pru):.3f}")

# %% [markdown]
# ### Sin no linealidad no hay red
#
# ¿Y si quitamos la activación? Con $g(z) = z$,
#
# $$\mathbf W^{[2]}\big(\mathbf W^{[1]}\mathbf x + \mathbf b^{[1]}\big) + \mathbf b^{[2]} = \underbrace{\mathbf W^{[2]}\mathbf W^{[1]}}_{\mathbf W}\,\mathbf x + \underbrace{\mathbf W^{[2]}\mathbf b^{[1]} + \mathbf b^{[2]}}_{\mathbf b}:$$
#
# la composición de funciones afines es afín, sin importar cuántas capas apilemos. Una red «profunda» sin activaciones no es más que una regresión *softmax* con una parametrización complicada: la «ingeniería de características» no puede hacer nada que la última capa no hiciera ya. Comprobémoslo:

# %%
red_lineal = red_mlp(2, 16, 2, nn.Identity)
entrenar(red_lineal, X_lunas, y_lunas, eta=0.3, epocas=300, tam_lote=20)
print(f"red 2-16-2 sin activación: exactitud de entrenamiento {exactitud(red_lineal, X_lunas, y_lunas):.3f} "
      f"(logística: {log_lunas.score(X_lunas, y_lunas):.3f})")

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
dibujar_frontera(ejes[0], probabilidad(red_lineal, malla)[:, 1], X_lunas, y_lunas, "16 neuronas ocultas, sin activación")
dibujar_frontera(ejes[1], probabilidad(red_lunas, malla)[:, 1], X_lunas, y_lunas, "16 neuronas ocultas, tanh")
fig.tight_layout()
plt.show()

# %% [markdown]
# Mismas 16 neuronas, mismos datos: sin activación la frontera es una recta y la exactitud es prácticamente la de la logística (la pequeña diferencia viene de que `LogisticRegression` está regularizada y de que nuestro descenso no llega exactamente al mínimo).
#
# ### Lo que aprende la capa oculta en el XOR
#
# Volvamos al XOR con una red de **dos** neuronas ocultas, para poder dibujar el espacio oculto $(a^{[1]}_1, a^{[1]}_2)$ en el plano. Con tan pocas neuronas, el resultado depende de dónde empiece el descenso: entrenemos la misma red desde 20 inicializaciones al azar.

# %%
exactitud_por_semilla = {}
for s in range(SEMILLA, SEMILLA + 20):
    r = red_mlp(2, 2, 2, nn.Tanh, semilla=s)
    entrenar(r, Xx, yx, eta=0.3, epocas=200, tam_lote=20)
    exactitud_por_semilla[s] = exactitud(r, Xx, yx)
exitos = sum(e > 0.95 for e in exactitud_por_semilla.values())
print(f"{exitos} de 20 inicializaciones resuelven el XOR (exactitud > 0.95); "
      f"las demás se quedan en {sorted(e for e in exactitud_por_semilla.values() if e <= 0.95)}")

# %% [markdown]
# La red **puede** resolver el problema, pero el descenso por gradiente no siempre encuentra cómo: a veces se queda en un mínimo local donde una de las neuronas ocultas no sirve de nada. Es un ejemplo pequeño de algo que el teorema de aproximación universal (sección 7) no promete. Con más neuronas ocultas de las estrictamente necesarias, el problema prácticamente desaparece. Tomemos una de las inicializaciones que sí lo resuelven.

# %%
red_xor = red_mlp(2, 2, 2, nn.Tanh, semilla=2012)
entrenar(red_xor, Xx, yx, eta=0.3, epocas=200, tam_lote=20)
A_xor = capa_oculta(red_xor, Xx)
W1_xor, b1_xor = red_xor[0].weight.detach().numpy(), red_xor[0].bias.detach().numpy()
W2_xor, b2_xor = red_xor[2].weight.detach().numpy(), red_xor[2].bias.detach().numpy()
print(f"red 2-2-2 en el XOR: exactitud {exactitud(red_xor, Xx, yx):.3f}")
print("W[1] =", W1_xor.round(2).tolist(), " b[1] =", b1_xor.round(2).tolist())

fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
g1x, g2x = np.meshgrid(np.linspace(-2.2, 2.2, 200), np.linspace(-2.2, 2.2, 200))
prob = probabilidad(red_xor, np.column_stack([g1x.ravel(), g2x.ravel()]))[:, 1].reshape(g1x.shape)
ejes[0].contourf(g1x, g2x, prob, levels=[0, 0.5, 1], colors=REGIONES, alpha=0.8)
ejes[0].scatter(*Xx.T, s=14, c=np.where(yx == 1, ROJO, AZUL), lw=0)
for j in range(2):   # rectas donde cada neurona oculta cambia de signo: z[1]_j = 0
    (w1, w2), b0 = W1_xor[j], b1_xor[j]
    xs = np.array([-2.2, 2.2])
    ejes[0].plot(xs, -(b0 + w1 * xs) / w2, color=TENUE, ls="--", lw=1.2)
ejes[0].set(title="Espacio original: la frontera se dobla", xlabel="$x_1$", ylabel="$x_2$", xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
# en el espacio oculto la frontera es la recta donde z[2]_1 = z[2]_0
w_d = W2_xor[1] - W2_xor[0]; b_d = b2_xor[1] - b2_xor[0]
t = np.linspace(-1.05, 1.05, 2)
ejes[1].scatter(*A_xor.T, s=14, c=np.where(yx == 1, ROJO, AZUL), lw=0)
ejes[1].plot(t, -(b_d + w_d[0] * t) / w_d[1], color=TINTA, lw=1.5)
ejes[1].set(title="Espacio oculto: una recta basta", xlabel="$a^{[1]}_1$", ylabel="$a^{[1]}_2$",
            xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
for ax in ejes:
    ax.set_aspect("equal")
fig.tight_layout()
plt.show()

# %% [markdown]
# Cada neurona oculta vale casi $+1$ de un lado de una recta (las líneas punteadas, donde $z^{[1]}_j = 0$) y casi $-1$ del otro. Las dos rectas son perpendiculares a la diagonal: las neuronas solo miran $x_1 + x_2$, como en la solución a mano de la sección 2. Las nubes $(-1,-1)$ y $(1,1)$ quedan en lados opuestos de ambas rectas, y las rojas, entre ellas. La capa oculta **dobla** el plano para juntar lo que una recta no podía juntar: las nubes de una misma clase van a parar a la misma región del espacio oculto, y ahí la última capa, una logística, traza una recta. Ésa es la idea central de las redes: **aprender una representación** en la que el problema se vuelva lineal.
#
# ## 7. ¿Qué tan expresiva es una capa oculta?
#
# El XOR y las lunas sugieren que una capa oculta puede dibujar fronteras bastante arbitrarias. El **teorema de aproximación universal** lo hace preciso: G. Cybenko (1989) demostró que, para la sigmoide, las funciones $\sum_{r=1}^H c_r\,\sigma(\mathbf w_r^\top\mathbf x + b_r)$ son densas en las funciones continuas sobre un compacto de $\mathbb{R}^n$: dada cualquier $f$ continua y cualquier $\varepsilon > 0$, existe una red de una capa oculta con $|\hat f(\mathbf x) - f(\mathbf x)| < \varepsilon$ en todo el compacto. K. Hornik (1991) mostró que basta con que la activación sea acotada y no constante, y M. Leshno y colaboradores (1993) extendieron el resultado a toda activación que no sea un polinomio, lo que incluye a ReLU.
#
# En una dimensión hay una **construcción** sencilla con ReLU. Una sola neurona $\mathrm{ReLU}(x - \tau)$ es una rampa que empieza en $\tau$; una suma $\hat f(x) = c_0 + \sum_r s_r\,\mathrm{ReLU}(x - \tau_r)$ es una función **lineal por pedazos** que cambia de pendiente en cada nodo $\tau_r$ (en $s_r$). Tres rampas forman un «bulto» triangular,
#
# $$\mathrm{ReLU}(x - \tau_{r-1}) - 2\,\mathrm{ReLU}(x - \tau_r) + \mathrm{ReLU}(x - \tau_{r+1}),$$
#
# y sumando bultos de las alturas adecuadas obtenemos la **interpolación lineal** de $f$ en los nodos $\tau_0 < \tau_1 < \dots < \tau_H$. Para $f$ con segunda derivada acotada, el error de la interpolación lineal con nodos separados $\Delta$ es a lo más $\Delta^2\max|f''|/8$: con $H$ neuronas, el error baja como $1/H^2$.
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

    Capa oculta: a_r = ReLU(x − τ_r), con peso 1 y sesgo −τ_r (r = 0, …, H − 1).
    Capa de salida: f̂ = c0 + Σ s_r a_r, donde s_r es el cambio de pendiente en τ_r.
    """
    tau = np.linspace(x.min(), x.max(), H + 1)
    v = np.interp(tau, x, y)                           # alturas de la función en los nodos
    pendientes = np.diff(v) / np.diff(tau)
    s = np.diff(np.concatenate([[0], pendientes]))     # cambio de pendiente en cada nodo
    return tau[:-1], s, v[0]


def evaluar_relu(x, tau, s, c0):
    return c0 + np.maximum(0, x[:, None] - tau[None, :]) @ s


fig, ejes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
filas = []
for ax, H in zip(ejes, [3, 6, 12]):
    tau, s, c0 = red_relu_interpolante(dia, f_dia, H)
    f_gorro = evaluar_relu(dia, tau, s, c0)
    ax.scatter(dia, f_dia, s=6, color=GRIS, lw=0)
    ax.plot(dia, f_gorro, color=AZUL, lw=2.2)
    ax.scatter(np.append(tau, dia.max()), evaluar_relu(np.append(tau, dia.max()), tau, s, c0), s=40, color=DORADO_OSC, zorder=3)
    ax.set_title(f"{H} neuronas ReLU · RECM {np.sqrt(np.mean((f_gorro - f_dia) ** 2)):.2f} °C", fontsize=12)
    ax.set_xlabel("día del año")
ejes[0].set_ylabel("temperatura media (°C)")
fig.suptitle("Temperatura media diaria en Hermosillo (1992–2024) y redes ReLU que la interpolan", x=0.01, ha="left",
             color=rdp.estilo.MARINO, weight="semibold")
fig.tight_layout()
plt.show()

for H in [1, 2, 3, 6, 12, 24, 48, 96]:
    tau, s, c0 = red_relu_interpolante(dia, f_dia, H)
    filas.append({"neuronas": H, "RECM (°C)": np.sqrt(np.mean((evaluar_relu(dia, tau, s, c0) - f_dia) ** 2))})
pd.DataFrame(filas).set_index("neuronas").T.round(2)

# %% [markdown]
# Con 3 neuronas la red ya distingue invierno, primavera y el verano de 33 a 35 °C; con 12 neuronas el error cuadrático medio es de alrededor de medio grado, comparable con el ruido que queda en el promedio de 33 años, y a partir de ahí agregar neuronas solo persigue ese ruido. Los pesos no se entrenaron: los **construimos**, con una neurona por nodo.
#
# **Lo que el teorema no dice.** (1) **Cuántas** neuronas hacen falta: con $n$ atributos, una construcción como ésta necesita una malla de nodos, del orden de $(1/\varepsilon)^{n/2}$ neuronas para un error $\varepsilon$; con los 64 atributos de los dígitos, es un número absurdo. (2) **Cómo encontrar** los pesos: el teorema dice que existen, no que el descenso por gradiente los encuentre. (3) Nada sobre **generalización**: una red que puede aproximar cualquier función también puede memorizar el ruido, y la Unidad 3 nos enseñó que la capacidad sin control es sobreajuste. Lo que sí dice es que una capa oculta suficientemente ancha no tiene un **sesgo** de aproximación irreducible, como sí lo tiene un modelo lineal.
#
# ## 8. Los dígitos escritos a mano
#
# Ahora el problema real, en el estilo habitual de PyTorch: el modelo es un `nn.Module`, el optimizador se encarga del paso $\theta \leftarrow \theta - \eta\,\nabla_\theta L$, y el ciclo de entrenamiento se escribe a mano para que se vea cada paso. Red 64-64-10 con ReLU, minilotes de 32, $\eta = 0.1$ y 40 épocas. Las dos funciones `entrenar_epoca` y `evaluar` las reutilizaremos en toda la unidad.

# %%
class MLP(nn.Module):
    def __init__(self, n, H, K):
        super().__init__()
        self.oculta = nn.Linear(n, H)                   # φθ: los atributos aprendidos
        self.salida = nn.Linear(H, K)                   # el GLM: regresión softmax sobre a[1]

    def atributos(self, x):
        return F.relu(self.oculta(x))

    def forward(self, x):
        return self.salida(self.atributos(x))           # logits; la softmax va dentro de la pérdida


def entrenar_epoca(modelo, cargador, optimizador, perdida_fn):
    modelo.train()
    total, m = 0.0, 0
    for xb, yb in cargador:
        optimizador.zero_grad()              # los gradientes se acumulan: hay que borrarlos
        perdida = perdida_fn(modelo(xb), yb)
        perdida.backward()                   # retropropagación
        optimizador.step()                   # θ ← θ − η ∇θ
        total += perdida.item() * len(yb); m += len(yb)
    return total / m


@torch.no_grad()
def evaluar(modelo, X, y):
    modelo.eval()
    pred = modelo(X).argmax(1)
    return (pred == y).float().mean().item()


Xd_ent_t = torch.tensor(Xd_ent, dtype=torch.float32); yd_ent_t = torch.tensor(yd_ent)
Xd_pru_t = torch.tensor(Xd_pru, dtype=torch.float32); yd_pru_t = torch.tensor(yd_pru)
cargador = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xd_ent_t, yd_ent_t), batch_size=32, shuffle=True,
                                       generator=torch.Generator().manual_seed(SEMILLA))

torch.manual_seed(SEMILLA)
mlp = MLP(64, 64, 10)
optimizador = torch.optim.SGD(mlp.parameters(), lr=0.1)
print(mlp, f"\nparámetros: {sum(q.numel() for q in mlp.parameters()):,}")

t0 = time.perf_counter()
curva_torch = []
for epoca in range(40):
    curva_torch.append(entrenar_epoca(mlp, cargador, optimizador, nn.CrossEntropyLoss()))
exac_torch = evaluar(mlp, Xd_pru_t, yd_pru_t)
print(f"40 épocas en {time.perf_counter() - t0:.1f} s; exactitud de prueba {exac_torch:.4f}")

# %% [markdown]
# Otra vez, la última capa es una regresión *softmax*. Si tomamos los 64 atributos aprendidos $\mathbf a^{[1]}$ y ajustamos sobre ellos la logística de scikit-learn, sin regularización apreciable, llegamos al mismo lugar que la red.

# %%
with torch.no_grad():
    A_ent, A_pru = mlp.atributos(Xd_ent_t).numpy(), mlp.atributos(Xd_pru_t).numpy()
logistica_sobre_a = LogisticRegression(C=1e4, max_iter=10000).fit(A_ent, yd_ent)
exac_log_a = logistica_sobre_a.score(A_pru, yd_pru)

pd.DataFrame({"exactitud de prueba": [exac_logistica, exac_torch, exac_log_a],
              "errores (de 450)": [round((1 - e) * 450) for e in (exac_logistica, exac_torch, exac_log_a)]},
             index=["regresión logística sobre los píxeles (C = 10)", "MLP 64-64-10 en PyTorch",
                    "regresión logística sobre a[1] de la red"]).round(4)

# %% [markdown]
# Con 450 imágenes de prueba cada error vale 0.2 puntos porcentuales, así que diferencias de uno o dos errores son ruido del entrenamiento: con otra semilla cambian. Los dígitos de $8 \times 8$ son un problema casi lineal; la ventaja de las redes aparece con imágenes más grandes y problemas más difíciles, como los de las siguientes libretas.
#
# ¿Qué hizo la capa oculta con los dígitos? Proyectemos a dos dimensiones, con componentes principales, los píxeles y los atributos aprendidos.

# %%
pca_x = PCA(2).fit(Xd_pru)
pca_a = PCA(2).fit(A_pru)
fig, ejes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, Z, titulo in [(ejes[0], pca_x.transform(Xd_pru), "píxeles x (64 atributos)"),
                      (ejes[1], pca_a.transform(A_pru), "capa oculta a[1] (64 atributos aprendidos)")]:
    for k in range(10):
        c = Z[yd_pru == k]
        ax.scatter(*c.T, s=10, color=GRIS, alpha=0.35, lw=0)
        ax.text(*np.median(c, axis=0), str(k), fontsize=18, weight="bold", color=AZUL, ha="center", va="center")
    ax.set_title(f"PCA de {titulo}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
plt.show()
print(f"coeficiente de silueta de las 10 clases: píxeles {silhouette_score(Xd_pru, yd_pru):.2f}, "
      f"capa oculta {silhouette_score(A_pru, yd_pru):.2f}")

# %% [markdown]
# En el espacio aprendido, los dígitos de una misma clase quedan más juntos y más lejos de los demás: el 0, el 4 y el 6 se apartan y el 1 deja de estar encimado con el 7 y el 8. El coeficiente de silueta de la Unidad 2, calculado con las clases verdaderas como grupos, se duplica. La red estiró las direcciones que distinguen clases, que es justo lo que necesita la regresión *softmax* de la última capa. Veamos cuáles dígitos confunde todavía.

# %%
with torch.no_grad():
    pred_t = mlp(Xd_pru_t).argmax(1).numpy()
malos = np.flatnonzero(pred_t != yd_pru)
fig, ejes = plt.subplots(1, len(malos), figsize=(1.4 * len(malos), 2))
for ax, i in zip(np.atleast_1d(ejes), malos):
    ax.imshow(Xd_pru[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"{yd_pru[i]} → {pred_t[i]}", fontsize=12)
    ax.axis("off")
fig.suptitle("Errores de la red (dígito real → predicho)", x=0.01, ha="left", color=rdp.estilo.MARINO, weight="semibold")
fig.tight_layout()
plt.show()

# %% [markdown]
# Con $8 \times 8$ celdas, algunos de estos dígitos son ambiguos aun para una persona; otros son errores claros del modelo.
#
# ## 9. La última capa decide la pérdida
#
# Si la última capa es un GLM, la pérdida es menos el logaritmo de su verosimilitud, y la salida $\hat y = h(z)$ usa la inversa de su **liga canónica**:
#
# | tarea | salida $\hat y$ | distribución de $y$ | pérdida $\ell$ | en PyTorch |
# |---|---|---|---|---|
# | regresión | $z$ | normal | $\tfrac12(\hat y - y)^2$ | `nn.MSELoss` (sin el ½) |
# | dos clases | $\sigma(z)$ | Bernoulli | $-y\ln\hat y - (1-y)\ln(1-\hat y)$ | `nn.BCEWithLogitsLoss` |
# | $K$ clases | $\operatorname{softmax}(\mathbf z)$ | categórica | $-\ln \hat y_{y}$ | `nn.CrossEntropyLoss` |
# | conteos | $e^{z}$ | Poisson | $\hat y - y\ln \hat y$ | `nn.PoissonNLLLoss` |
#
# Con la liga canónica pasa algo notable: en los cuatro casos, el adjunto de la entrada lineal de la última capa es **predicción menos dato**, $\partial\ell/\partial z = \hat y - y$. No lo derivemos: pidámoselo a autograd.

# %%
gen = torch.Generator().manual_seed(SEMILLA)
z = torch.randn(6, generator=gen, requires_grad=True)         # entradas lineales de la última capa
M = len(z)
cabezas = {
    "identidad + ½ ECM": (lambda z, y: 0.5 * F.mse_loss(z, y), lambda z: z, torch.randn(6, generator=gen)),
    "sigmoide + BCE": (F.binary_cross_entropy_with_logits, torch.sigmoid, torch.tensor([0., 1, 1, 0, 1, 0])),
    "exp + Poisson": (lambda z, y: F.poisson_nll_loss(z, y, log_input=True), torch.exp, torch.tensor([0., 3, 1, 2, 0, 5])),
}
for nombre, (perdida_fn, salida, y) in cabezas.items():
    z.grad = None
    perdida_fn(z, y).backward()                                # L es el promedio de las ℓ: hay que multiplicar por M
    print(f"{nombre:18} máx |M ∂L/∂z − (ŷ − y)| = {(M * z.grad - (salida(z) - y)).abs().max().item():.1e}")

Z = torch.randn(6, 4, generator=gen, requires_grad=True)       # K = 4 clases
y = torch.tensor([0, 3, 1, 2, 2, 0])
F.cross_entropy(Z, y).backward()
print(f"{'softmax + CE':18} máx |M ∂L/∂z − (ŷ − y)| = {(M * Z.grad - (F.softmax(Z, 1) - F.one_hot(y, 4))).abs().max().item():.1e}")

# %% [markdown]
# Ésta es la razón de que las redes de clasificación y de regresión usen las mismas pérdidas que la regresión logística y la lineal: **hasta la última capa, una red es un GLM sobre atributos aprendidos**. La pérdida mide qué tan probable hace el modelo a los datos, y la red solo cambia el espacio en que trabaja el GLM.
#
# Eso cambia en las redes más sofisticadas. Hay modelos cuya pérdida ya no es la verosimilitud de un GLM: redes que aprenden a acercar en el espacio de representación pares de ejemplos parecidos (pérdidas **contrastivas**), dos redes que compiten entre sí (pérdidas **adversariales**), o redes que aprenden a quitar ruido de una imagen. La maquinaria es la misma: un grafo de cómputo que termina en un número y un `backward()`.
#
# ## 10. La inicialización: qué son Xavier y He
#
# **¿Por qué no empezar con todos los pesos en cero?** Porque todas las neuronas ocultas calcularían lo mismo, recibirían el mismo gradiente y seguirían siendo iguales para siempre: la capa oculta valdría lo mismo que una sola neurona. Con ReLU es peor aún: con pesos cero, $\mathbf z^{[1]} = \mathbf 0$, la derivada de ReLU ahí es 0 y la capa oculta no recibe gradiente. Probémoslo con todos los pesos en 0.1 (no cero) en los dígitos.

# %%
torch.manual_seed(SEMILLA)
simetrica = MLP(64, 64, 10)
with torch.no_grad():
    for p in simetrica.parameters():
        p.fill_(0.1)
opt_sim = torch.optim.SGD(simetrica.parameters(), lr=0.1)
for epoca in range(40):
    entrenar_epoca(simetrica, cargador, opt_sim, nn.CrossEntropyLoss())
W1_sim = simetrica.oculta.weight.detach()
print(f"renglones distintos en W[1] después de entrenar: {torch.unique(W1_sim.round(decimals=6), dim=0).shape[0]} de 64")
print(f"exactitud de prueba: {evaluar(simetrica, Xd_pru_t, yd_pru_t):.3f}")

# %% [markdown]
# Las 64 neuronas siguen siendo una sola. Los pesos iniciales se **sortean** al azar para romper la simetría. La pregunta es de qué tamaño.
#
# **El tamaño importa.** Una neurona suma $n_{\text{in}}$ entradas $a_j$ multiplicadas por pesos $w_j$, $z = \sum_{j=1}^{n_{\text{in}}} w_j a_j$. Si los pesos son independientes, de media cero, e independientes de las entradas,
#
# $$\operatorname{Var}(z) = \sum_{j=1}^{n_{\text{in}}} \operatorname{Var}(w_j a_j) = n_{\text{in}}\,\operatorname{Var}(w)\,\mathbb E\big[a^2\big] .$$
#
# Con $n_{\text{in}} = 256$ entradas de varianza 1: si $w \sim N(0, 1)$, $z$ tiene desviación estándar $\sqrt{256} = 16$; si $w \sim N(0, 0.01^2)$, $0.16$. Solo con $\operatorname{Var}(w) = 1/n_{\text{in}}$ la escala se conserva. Simulémoslo.

# %%
gen = torch.Generator().manual_seed(SEMILLA)
entradas = torch.randn(10000, 256, generator=gen)              # 10,000 ejemplos con 256 entradas de varianza 1
escalas = {"N(0, 1)": 1.0, "N(0, 0.01²)": 0.01, "N(0, 1/n_in)": 1 / math.sqrt(256)}
z_por_escala = {}
for nombre, desv in escalas.items():
    w_neurona = desv * torch.randn(256, generator=gen)
    z_por_escala[nombre] = (entradas @ w_neurona).numpy()
    print(f"w ~ {nombre:13}: desviación estándar de z = {z_por_escala[nombre].std():.2f}")

# %% [markdown]
# En una sola capa eso es un cambio de escala; en una red de 20 capas, el factor se **multiplica** 20 veces y la señal explota o se apaga exponencialmente. Las dos recetas estándar fijan la varianza para que el factor por capa sea 1:
#
# - **Xavier o Glorot** (X. Glorot y Y. Bengio, 2010), para $\tanh$ y la sigmoide. Cerca de cero, $\tanh(z) \approx z$, así que $\mathbb E[a^2] \approx \operatorname{Var}(z_{\text{anterior}})$ y hace falta $n_{\text{in}}\operatorname{Var}(w) = 1$. Pero el gradiente viaja hacia atrás por $\mathbf W^\top$, donde cuentan las $n_{\text{out}}$ salidas de la capa, y para él haría falta $n_{\text{out}}\operatorname{Var}(w) = 1$. El compromiso es el promedio:
# $$\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}.$$
# - **He o Kaiming** (K. He y colaboradores, 2015), para ReLU. ReLU anula la mitad de las $z$: si $z$ es simétrica, $\mathbb E[\mathrm{ReLU}(z)^2] = \tfrac12\operatorname{Var}(z)$. Para compensar esa mitad se duplica la varianza:
# $$\operatorname{Var}(w) = \frac{2}{n_{\text{in}}}.$$
#
# **Cómo se sortean.** Con la varianza decidida, los pesos se pueden sacar de una normal $N(0, \operatorname{Var}(w))$ o de una uniforme $U(-r, r)$, que tiene varianza $r^2/3$; para la misma varianza, $r = \sqrt{3\operatorname{Var}(w)}$. Para una capa de 64 entradas y 64 salidas:

# %%
n_in = n_out = 64
var_xavier, var_he = 2 / (n_in + n_out), 2 / n_in
pd.DataFrame({
    "Var(w)": [var_xavier, var_he],
    "desviación de la normal": [math.sqrt(var_xavier), math.sqrt(var_he)],
    "límite r de la uniforme": [math.sqrt(3 * var_xavier), math.sqrt(3 * var_he)],
    "en PyTorch": ["nn.init.xavier_normal_ / xavier_uniform_", "nn.init.kaiming_normal_ / kaiming_uniform_"],
}, index=["Xavier (tanh, sigmoide)", "He (ReLU)"]).round(4)

# %%
capa = nn.Linear(n_in, n_out)
print(f"nn.Linear por omisión:  desviación de los pesos {capa.weight.std().item():.4f} "
      f"(uniforme en ±1/√n_in = ±{1 / math.sqrt(n_in):.3f}, desviación teórica {1 / math.sqrt(3 * n_in):.4f})")
nn.init.xavier_uniform_(capa.weight)
print(f"xavier_uniform_:        desviación {capa.weight.std().item():.4f}, máximo |w| = {capa.weight.abs().max().item():.3f}")
nn.init.kaiming_normal_(capa.weight, nonlinearity="relu")
print(f"kaiming_normal_:        desviación {capa.weight.std().item():.4f}")

# %% [markdown]
# Por omisión, `nn.Linear` sortea en $U(-1/\sqrt{n_{\text{in}}}, 1/\sqrt{n_{\text{in}}})$, que da $\operatorname{Var}(w) = 1/(3 n_{\text{in}})$: una escala más chica que la de He. Para redes poco profundas no importa mucho; para redes profundas con ReLU conviene pedir He explícitamente. Los sesgos se inician en cero: la asimetría ya la ponen los pesos.
#
# Veamos qué pasa en una red de 20 capas de 256 neuronas ReLU, alimentada con los dígitos estandarizados, midiendo la desviación estándar de las activaciones capa por capa con cuatro inicializaciones.

# %%
def activaciones_por_capa(X, capas, ancho, desv_w, generador):
    a, desv = X, []
    for _ in range(capas):
        W = desv_w(a.shape[1], ancho) * torch.randn(ancho, a.shape[1], generator=generador, dtype=torch.float64)
        a = torch.relu(a @ W.T)
        desv.append(a.std().item())
    return np.array(desv)


Xd_est = torch.tensor((Xd_ent - Xd_ent.mean(0)) / (Xd_ent.std(0) + 1e-8))     # entradas con varianza 1
INICIALIZACIONES = {
    "N(0, 0.01²)": lambda n_in, n_out: 0.01,
    "Xavier: 2/(n_in + n_out)": lambda n_in, n_out: math.sqrt(2 / (n_in + n_out)),
    "He: 2/n_in": lambda n_in, n_out: math.sqrt(2 / n_in),
    "N(0, 1)": lambda n_in, n_out: 1.0,
}
CAPAS = 20
desviaciones = {nombre: activaciones_por_capa(Xd_est, CAPAS, 256, f, torch.Generator().manual_seed(SEMILLA))
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
# Con He la escala se mantiene a lo largo de las 20 capas. Con Xavier, pensada para tanh, cada capa ReLU multiplica la varianza por $\tfrac12$ y la desviación estándar decae como $2^{-l/2}$ en la capa $l$: en la capa 20 es cientos de veces más chica que en la primera. Los pesos «chicos» $N(0, 0.01^2)$ dan un factor $256 \times 10^{-4}/2 \approx 0.013$ por capa y la señal desaparece en unas cuantas capas; los pesos $N(0,1)$ multiplican la varianza por $256/2 = 128$ en cada capa y las activaciones se disparan a $10^{20}$. Una red así no se puede entrenar: o no llega señal a la salida, o los números se desbordan.
#
# ## 11. El desvanecimiento del gradiente
#
# La pasada hacia atrás multiplica en cada capa $l$ por $\mathbf W^{[l]\top}$ y por $g'(\mathbf z^{[l]})$. La derivada de la sigmoide, $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, vale a lo más $\tfrac14$ (en $z = 0$), así que aun con pesos bien escalados el gradiente se **encoge** al viajar hacia las primeras capas. Midámoslo: una red de 10 capas ocultas de 64 neuronas sobre un lote de dígitos, con sigmoide y con ReLU, comparando la norma del gradiente de la pérdida respecto de los pesos de cada capa.

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
ax.set_ylabel("‖∂L/∂W‖ de la capa")
ax.set_title("Norma del gradiente por capa, en la inicialización")
plt.show()
print(f"sigmoide: la capa 10 recibe un gradiente {grad_sigmoide[-1] / grad_sigmoide[0]:,.0f} veces mayor que la capa 1")
print(f"ReLU:     cociente capa 10 / capa 1 = {grad_relu[-1] / grad_relu[0]:.1f}")

# %% [markdown]
# Con sigmoides, el gradiente de la primera capa es unas 455,000 veces menor que el de la última (cerca de un factor de 4 por capa, el inverso del máximo de $\sigma'$); con ReLU y la inicialización de He, todas las capas reciben gradientes de tamaño parecido. Con sigmoides, en el descenso por gradiente, las primeras capas prácticamente no aprenden. Éste fue uno de los obstáculos para entrenar redes profundas en los años noventa y dos mil. Las soluciones que veremos en el resto de la unidad atacan justo este producto de factores: activaciones ReLU (derivada 1 donde están activas), buena inicialización, normalización por lotes, conexiones residuales y, en las redes recurrentes, las compuertas de la LSTM.
#
# ## 12. Cierre
#
# - Una red neuronal es **ingeniería de características aprendida** (las capas ocultas llevan los datos de un espacio a otro) más un **modelo lineal generalizado** en la última capa. Por eso usa las mismas pérdidas que la regresión lineal, la logística o la de Poisson, y por eso su error de salida es «predicción menos dato». Sin no linealidad, todo se colapsa a un modelo lineal.
# - Una red es un **grafo de cómputo**. La **diferenciación automática** en modo inverso recorre el grafo hacia atrás con la derivada local de cada nodo y obtiene todo el gradiente al costo de unas pocas pasadas hacia adelante, sin importar el número de parámetros. La retropropagación es ese modo inverso aplicado a una red, y fue lo que hizo posible entrenarlas.
# - PyTorch construye el grafo mientras el código corre y lo recorre con `backward()`; los gradientes se acumulan y hay que borrarlos. TensorFlow y JAX hacen lo mismo con otra interfaz. Dominar una de estas bibliotecas es lo que permite construir cualquier arquitectura.
# - Los pesos se sortean para romper la simetría, con una escala que conserve la señal: Xavier, $\operatorname{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$, para tanh; He, $\operatorname{Var}(w) = 2/n_{\text{in}}$, para ReLU.
#
# En la siguiente libreta usaremos estas herramientas para entrenar redes más grandes en un problema más difícil, y nos ocuparemos de **cómo** descender: tasas de aprendizaje, momento, Adam y regularización.
#
# *Referencias:* F. Rosenblatt, «The perceptron: a probabilistic model for information storage and organization in the brain», *Psychological Review* 65 (1958). M. Minsky y S. Papert, *Perceptrons*, MIT Press (1969). S. Linnainmaa, *The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors*, tesis de maestría, Universidad de Helsinki (1970). D. E. Rumelhart, G. E. Hinton y R. J. Williams, «Learning representations by back-propagating errors», *Nature* 323 (1986). G. Cybenko, «Approximation by superpositions of a sigmoidal function», *Mathematics of Control, Signals and Systems* 2 (1989). K. Hornik, «Approximation capabilities of multilayer feedforward networks», *Neural Networks* 4 (1991). X. Glorot y Y. Bengio, «Understanding the difficulty of training deep feedforward neural networks», AISTATS (2010). K. He, X. Zhang, S. Ren y J. Sun, «Delving deep into rectifiers», ICCV (2015). A. G. Baydin, B. A. Pearlmutter, A. A. Radul y J. M. Siskind, «Automatic differentiation in machine learning: a survey», *JMLR* 18 (2018). A. Paszke et al., «PyTorch: an imperative style, high-performance deep learning library», NeurIPS (2019). I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press (2016), capítulo 6.

# %% [markdown]
# ## 13. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p01", "u06-p02", "u06-p03", "u06-p04", "u06-p17", "u06-p18", "u06-p19"])

# %% [markdown]
# ## 14. Ejercicios
#
# 1. **Más operaciones para `Valor`.** Agrega a la clase `Valor` la resta, la división, `exp` y `relu`, cada una con su `_atras`. Usa tu motor para calcular el gradiente de una red 2-3-1 con tanh en un solo ejemplo, y compáralo con PyTorch.
# 2. **Una operación nueva en PyTorch.** Escribe una `torch.autograd.Function` para $\operatorname{softplus}(z) = \ln(1 + e^z)$ con su `forward` y su `backward`, y verifícala con `torch.autograd.gradcheck` en doble precisión. ¿Qué pasa con `gradcheck` si te equivocas a propósito en el signo de la derivada?
# 3. **Simetría.** Inicializa todos los pesos de la red de los dígitos en cero (no en 0.1) y entrénala con ReLU y luego con tanh. Explica en cada caso, con la regla de la cadena, qué gradiente recibe la capa oculta. ¿Por qué los sesgos de la capa de salida sí aprenden?
# 4. **Anchura contra exactitud.** Entrena la red de los dígitos con $H = 4, 8, 16, 32, 64, 128$ neuronas ocultas y grafica la exactitud de entrenamiento y de prueba contra $H$. ¿A partir de qué anchura la red supera a la logística? ¿Ves sobreajuste? Relaciónalo con la discusión de sesgo y varianza de la Unidad 3.
# 5. **Una cabeza de Poisson.** Con los datos de bicicletas de la Unidad 4, entrena un MLP con salida $e^{z}$ y `nn.PoissonNLLLoss(log_input=True)`, y compáralo con la regresión de Poisson. ¿Cuánto mejora la red, y en qué horas?
