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
# # Convoluciones con Fashion-MNIST y transferencia a imágenes de satélite
#
# **Unidad 6 · Ejemplo 3**
#
# Un perceptrón multicapa trata una imagen como una lista de números: si revolvemos los pixeles con la misma permutación en todas las imágenes, aprende exactamente igual. Pero una imagen no es una lista: los pixeles vecinos están relacionados, y un borde, un botón o una costura significan lo mismo en cualquier parte del cuadro. Las **redes convolucionales** incorporan esas dos ideas en la arquitectura. En esta libreta vamos a:
#
# 1. calcular cuántos parámetros necesitaría un perceptrón multicapa para ver una foto, y por qué es demasiado;
# 2. programar la **convolución** discreta, aplicar a una prenda filtros de bordes hechos a mano y verificar su **equivariancia a traslaciones**;
# 3. entender el **relleno**, el **paso**, los **canales**, el **campo receptivo** y el ***pooling***, y contar parámetros;
# 4. entrenar una red convolucional pequeña en Fashion-MNIST y compararla con un perceptrón multicapa que tiene once veces más parámetros; mirar sus filtros, sus mapas de activación y sus errores;
# 5. ver con una simulación por qué las **conexiones residuales** permiten entrenar redes muy profundas;
# 6. usar una ResNet-18 entrenada en ImageNet para clasificar **imágenes de satélite** con muy pocos ejemplos etiquetados (**transferencia**), y compararla con una red entrenada desde cero con los mismos datos.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado. PyTorch y torchvision ya vienen instalados en Colab. Si tienes una GPU (en Colab: *Entorno de ejecución → Cambiar tipo de entorno*), la libreta la usa; si no, corre en CPU en unos dos o tres minutos.

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

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# la verificación automática del repositorio usa menos datos y menos épocas
RAPIDO = os.environ.get("RDP_RAPIDO") == "1"
RAIZ_DATOS = os.path.expanduser("~/.cache/rdp-datos")
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEMILLA)
print("dispositivo:", dispositivo)

# %% [markdown]
# ## 1. Una pregunta para empezar: ¿cuántos parámetros?
#
# Una foto modesta de 224 × 224 pixeles en color tiene $224 \times 224 \times 3 = 150{,}528$ números. Si la conectamos a una capa oculta de 1,000 neuronas, como hicimos con los dígitos de 8 × 8, cada neurona necesita un peso por número de entrada:

# %%
entradas = 224 * 224 * 3
print(f"entradas: {entradas:,}")
print(f"pesos de la primera capa con 1,000 neuronas: {entradas * 1000 + 1000:,}")

# %% [markdown]
# Ciento cincuenta millones de parámetros **en la primera capa**, que además no sabe nada de imágenes: para ella, el pixel (0, 0) y el (0, 1) están tan relacionados como el (0, 0) y el (223, 223). Dos observaciones sobre las imágenes reducen esto drásticamente:
#
# - **Localidad.** Lo que hay en un pixel depende sobre todo de sus vecinos. Un detector de bordes o de texturas solo necesita mirar una ventana pequeña.
# - **Las cosas pueden estar en cualquier parte.** Un borde vertical es un borde vertical arriba o abajo del cuadro. El mismo detector, con **los mismos pesos**, sirve en todas las posiciones.
#
# Una capa que mira ventanas de 3 × 3 y comparte sus pesos en todas las posiciones tiene $3 \times 3 \times 3 + 1 = 28$ parámetros por detector, sin importar el tamaño de la imagen. Esa capa es una **convolución**.

# %% [markdown]
# ## 2. Los datos y el problema: Fashion-MNIST
#
# **Datos.** Fashion-MNIST (H. Xiao, K. Rasul y R. Vollgraf, 2017) contiene 70,000 imágenes en escala de grises de 28 × 28 pixeles de artículos del catálogo de Zalando, una tienda de ropa en línea europea: 60,000 para entrenar y 10,000 para probar, con 10 clases balanceadas. Sus autores lo publicaron como reemplazo directo de MNIST (los dígitos escritos a mano de 28 × 28), que se había vuelto demasiado fácil: con MNIST, casi cualquier método llega a 97 % de exactitud y no distingue entre ideas buenas y malas. Licencia MIT; `torchvision` lo descarga la primera vez (unos 30 MB) a `~/.cache/rdp-datos`.
#
# **Problema.** Dada la imagen de una prenda, decir de qué tipo es. Es la misma tarea de los dígitos, pero más difícil: una camisa y una camiseta, o un suéter y un abrigo, se parecen mucho a esta resolución.

# %%
CLASES = ["camiseta", "pantalón", "suéter", "vestido", "abrigo", "sandalia", "camisa", "tenis", "bolsa", "botín"]

fm_ent = torchvision.datasets.FashionMNIST(RAIZ_DATOS, train=True, download=True)
fm_pru = torchvision.datasets.FashionMNIST(RAIZ_DATOS, train=False, download=True)
X_ent = fm_ent.data.float().div(255).unsqueeze(1)    # (60000, 1, 28, 28), valores en [0, 1]
y_ent = fm_ent.targets
X_pru = fm_pru.data.float().div(255).unsqueeze(1)
y_pru = fm_pru.targets
print(X_ent.shape, X_pru.shape)
print("imágenes por clase (entrenamiento):", torch.bincount(y_ent).tolist())

# %%
fig, ejes = plt.subplots(4, 10, figsize=(12, 5.4))
for c in range(10):
    ids = torch.where(y_ent == c)[0][:4]
    for f in range(4):
        ax = ejes[f, c]
        ax.imshow(X_ent[ids[f], 0], cmap="gray_r")
        ax.axis("off")
    ejes[0, c].set_title(CLASES[c], fontsize=11, loc="center")
fig.suptitle("Fashion-MNIST: cuatro ejemplos de cada clase", x=0.01, ha="left", color=rdp.estilo.MARINO, fontweight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. La convolución
#
# ### Definición
#
# Sea $X$ una imagen (una matriz) y $K$ un **filtro** (o *kernel*) de tamaño $k \times k$. La operación que usan las redes convolucionales es, en rigor, la **correlación cruzada**:
#
# $$Y[i, j] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K[u, v]\; X[i + u,\, j + v] .$$
#
# Cada salida es el producto punto entre el filtro y la ventana de $k \times k$ de la imagen cuya esquina está en $(i, j)$; deslizando la ventana por toda la imagen se obtiene un **mapa de activación**. La convolución de los libros de procesamiento de señales voltea el filtro, $X[i - u, j - v]$; como en una red el filtro se aprende, voltearlo o no da lo mismo, y todas las bibliotecas implementan la correlación y la llaman convolución. Nosotros también.
#
# Sin relleno, una imagen de $n \times n$ produce un mapa de $(n - k + 1) \times (n - k + 1)$. Programémosla con dos ciclos, para ver que no hay magia, y comparémosla con la de PyTorch:

# %%
def correlacion2d(X, K):
    """Correlación cruzada 'válida' (sin relleno) de una imagen X con un filtro K."""
    k = K.shape[0]
    n1, n2 = X.shape[0] - k + 1, X.shape[1] - k + 1
    Y = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            Y[i, j] = np.sum(K * X[i:i + k, j:j + k])
    return Y


X_chica = np.array([[0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=float)
K_vertical = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype=float)
Y_chica = correlacion2d(X_chica, K_vertical)
print(Y_chica)

Y_torch = F.conv2d(torch.tensor(X_chica).view(1, 1, 5, 5), torch.tensor(K_vertical).view(1, 1, 3, 3))
print("¿igual a PyTorch?", np.allclose(Y_chica, Y_torch.numpy()[0, 0]))

# %% [markdown]
# La imagen de 5 × 5 tiene un bloque de unos arriba a la derecha. El filtro resta la columna izquierda de la ventana a la derecha: vale cero donde la imagen es constante y es grande donde la intensidad **sube de izquierda a derecha**. Por ejemplo, la salida $Y[0, 0]$ es $(1 + 1 + 1) - (0 + 0 + 0) = 3$: la ventana de las filas 0–2 y columnas 0–2 contiene el borde. En la última fila de $Y$ el borde ya solo ocupa una de las tres filas de la ventana, y la respuesta baja a 1.
#
# ### Filtros de bordes hechos a mano
#
# Apliquemos dos filtros de Sobel, uno para bordes verticales y otro para horizontales, a un botín del conjunto de prueba. Los filtros de Sobel suavizan en una dirección mientras derivan en la otra (pesos 1, 2, 1).

# %%
SOBEL_V = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_H = SOBEL_V.T

IDX_PRENDA = int(torch.where(y_pru == 9)[0][0])          # el primer botín del conjunto de prueba
prenda = X_pru[IDX_PRENDA, 0].numpy()
mapa_v = correlacion2d(prenda, SOBEL_V)
mapa_h = correlacion2d(prenda, SOBEL_H)

fig, ejes = plt.subplots(1, 3, figsize=(12, 4.3))
ejes[0].imshow(prenda, cmap="gray_r")
ejes[0].set_title(f"botín (prueba #{IDX_PRENDA})")
lim = np.abs(np.r_[mapa_v.ravel(), mapa_h.ravel()]).max()
for ax, mapa, nombre in zip(ejes[1:], (mapa_v, mapa_h), ("bordes verticales", "bordes horizontales")):
    ax.imshow(mapa, cmap=rdp.estilo.DIVERGENTE, vmin=-lim, vmax=lim)
    ax.set_title(nombre)
for ax in ejes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# En los mapas, azul es una respuesta positiva (la intensidad sube hacia la derecha o hacia abajo) y rojo una negativa. El filtro vertical marca los contornos laterales del botín; el horizontal, la suela y el borde superior. Nueve números bastan para detectar un tipo de borde en toda la imagen. La idea de una red convolucional es **no** diseñar estos filtros, sino **aprenderlos**.
#
# ### Equivariancia a traslaciones
#
# Si movemos la imagen, el mapa de activación se mueve igual. Sea $(T_{a,b}X)[i, j] = X[i - a, j - b]$ la traslación por $(a, b)$. Entonces
#
# $$\big(K \star T_{a,b}X\big)[i, j] = \sum_{u,v} K[u, v]\, X[i + u - a,\, j + v - b] = (K \star X)[i - a, j - b] = \big(T_{a,b}(K \star X)\big)[i, j] ,$$
#
# es decir, $K \star (T X) = T(K \star X)$: **convolucionar y trasladar conmutan**. A esto se le llama **equivariancia**. (En los bordes de la imagen hay que decidir qué valores entran a la ventana; si la imagen tiene fondo cero y no se sale del cuadro, la igualdad es exacta.) Verifiquémoslo:

# %%
def trasladar(X, a, b):
    """Mueve la imagen a filas hacia abajo y b columnas a la derecha, rellenando con ceros."""
    Y = np.zeros_like(X)
    Y[a:, b:] = X[:X.shape[0] - a, :X.shape[1] - b]
    return Y


a, b = 2, 3
lienzo = np.pad(prenda, 4)          # margen de ceros, para que la prenda no se salga del cuadro al moverla
izq = correlacion2d(trasladar(lienzo, a, b), SOBEL_V)       # trasladar y luego filtrar
der = trasladar(correlacion2d(lienzo, SOBEL_V), a, b)       # filtrar y luego trasladar
print("diferencia máxima:", np.abs(izq - der).max())

# %% [markdown]
# Una capa densa no tiene esta propiedad: sus pesos para el pixel (5, 5) no tienen nada que ver con los del pixel (7, 8), y lo que aprenda sobre un objeto en una posición no le sirve en otra.
#
# ## 4. Relleno, paso, canales y conteo de parámetros
#
# Las capas convolucionales de una red tienen cuatro ingredientes más.
#
# - **Relleno** (*padding*) $p$: se agregan $p$ filas y columnas de ceros alrededor de la imagen. Con $k = 3$ y $p = 1$, el mapa conserva el tamaño de la entrada.
# - **Paso** (*stride*) $s$: la ventana avanza de $s$ en $s$ pixeles. Con $s = 2$ el mapa tiene la mitad de filas y columnas.
# - **Canales.** Una imagen en color tiene 3 canales; la salida de una capa con 16 filtros tiene 16. Cada filtro abarca **todos** los canales de entrada: es un arreglo de $C_{\text{ent}} \times k \times k$ pesos, más un sesgo, y produce un canal de salida:
#
# $$Y_c[i, j] = b_c + \sum_{c'=1}^{C_{\text{ent}}} \sum_{u,v} K_{c, c'}[u, v]\; X_{c'}[s\,i + u,\, s\,j + v] .$$
#
# - **Tamaño de la salida:** $n_{\text{sal}} = \left\lfloor \dfrac{n + 2p - k}{s} \right\rfloor + 1$. **Parámetros:** $(C_{\text{ent}}\, k^2 + 1)\, C_{\text{sal}}$, sin importar $n$.
#
# El ***pooling*** (agrupamiento) reduce el tamaño de los mapas: el *max-pooling* de 2 × 2 con paso 2 se queda con el máximo de cada bloque de 2 × 2, en cada canal por separado. No tiene parámetros.
#
# La red que vamos a usar en Fashion-MNIST tiene dos bloques «convolución 3 × 3 + ReLU + *max-pooling*» y una capa lineal final:

# %%
class RedConv(nn.Module):
    def __init__(self, canales=(16, 32), n_clases=10):
        super().__init__()
        c1, c2 = canales
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.salida = nn.Linear(c2 * 7 * 7, n_clases)

    def atributos(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)     # 28×28 → 14×14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)     # 14×14 → 7×7
        return x

    def forward(self, x):
        return self.salida(self.atributos(x).flatten(1))


def contar(modelo):
    return sum(p.numel() for p in modelo.parameters())


red = RedConv()
x = torch.zeros(1, 1, 28, 28)
filas = []
for nombre, capa in [("conv1 (3×3, 1→16)", lambda x: red.conv1(x)), ("ReLU + max-pool", lambda x: F.max_pool2d(F.relu(x), 2)),
                     ("conv2 (3×3, 16→32)", lambda x: red.conv2(x)), ("ReLU + max-pool ", lambda x: F.max_pool2d(F.relu(x), 2)),
                     ("aplanar", lambda x: x.flatten(1)), ("lineal (1568→10)", lambda x: red.salida(x))]:
    x = capa(x)
    filas.append((nombre, tuple(x.shape[1:])))
params_capa = [contar(red.conv1), 0, contar(red.conv2), 0, 0, contar(red.salida)]
pd.DataFrame({"capa": [f for f, _ in filas], "forma de la salida": [s for _, s in filas], "parámetros": params_capa})

# %% [markdown]
# Verificamos con la fórmula: `conv1` tiene $(1 \cdot 9 + 1) \cdot 16 = 160$ parámetros y `conv2`, $(16 \cdot 9 + 1) \cdot 32 = 4{,}640$. La mayor parte de los 20,490 parámetros está en la capa lineal final, $(1{,}568 + 1) \cdot 10 = 15{,}690$: las convoluciones son baratas.
#
# ### El campo receptivo crece con la profundidad
#
# Una neurona de `conv1` ve una ventana de 3 × 3 de la imagen. Una de `conv2` ve 3 × 3 posiciones del mapa anterior, ya reducido a la mitad por el *pooling*: en la imagen original eso abarca más. Si $r_\ell$ es el lado del **campo receptivo** después de la capa $\ell$ y $j_\ell$ la distancia, en pixeles de la imagen, entre dos posiciones vecinas del mapa, una capa con ventana $k$ y paso $s$ cumple
#
# $$r_\ell = r_{\ell-1} + (k - 1)\, j_{\ell-1}, \qquad j_\ell = j_{\ell-1}\, s, \qquad r_0 = j_0 = 1 .$$
#
# Para nuestra red: `conv1` da $r = 3$; el *pooling* ($k = s = 2$) da $r = 4$, $j = 2$; `conv2` da $r = 4 + 2 \cdot 2 = 8$; el segundo *pooling*, $r = 8 + 1 \cdot 2 = 10$. Cada posición del mapa final «ve» un cuadro de 10 × 10 pixeles. Lo comprobamos con el gradiente: la salida de una posición solo puede depender de los pixeles en los que su gradiente no es cero.

# %%
def campo_receptivo(capas):
    r, j, res = 1, 1, []
    for k, s in capas:
        r, j = r + (k - 1) * j, j * s
        res.append(r)
    return res


print("teórico:", campo_receptivo([(3, 1), (2, 2), (3, 1), (2, 2)]))

torch.manual_seed(SEMILLA)
x = torch.randn(64, 1, 28, 28, requires_grad=True)
red.atributos(x)[:, :, 3, 3].sum().backward()      # posición (3, 3) del mapa final de 7×7
soporte = (x.grad.abs().sum(dim=(0, 1)) > 0)
filas_ok, cols_ok = torch.where(soporte.any(1))[0], torch.where(soporte.any(0))[0]
print(f"pixeles que influyen en la posición (3, 3): filas {filas_ok.min().item()}–{filas_ok.max().item()}, "
      f"columnas {cols_ok.min().item()}–{cols_ok.max().item()}  →  {len(filas_ok)} × {len(cols_ok)}")

# %% [markdown]
# Con más capas, el campo receptivo sigue creciendo: las primeras capas detectan rasgos locales (bordes, esquinas) y las profundas combinan esos rasgos en regiones cada vez mayores. Una red como ResNet-18 termina con campos receptivos más grandes que la imagen misma.
#
# ### *Pooling* e invariancia aproximada
#
# La convolución es **equivariante**: si la prenda se mueve un pixel, el mapa se mueve un pixel. Para clasificar queremos algo **invariante**: que la respuesta no cambie. El *max-pooling* da un poco de eso: si el máximo de un bloque de 2 × 2 se mueve dentro del mismo bloque, la salida no cambia. Midamos, en 1,000 imágenes de prueba, cuánto cambia la representación (en distancia relativa, $\lVert f(TX) - f(X)\rVert / \lVert f(X)\rVert$) cuando la imagen se mueve un pixel a la derecha, antes y después de cada *pooling*. Usamos la red con pesos aleatorios: la propiedad es de la arquitectura, no del entrenamiento.

# %%
def cambio_relativo(f, X, desp=1):
    Xt = torch.zeros_like(X)
    Xt[..., desp:] = X[..., :-desp]
    with torch.no_grad():
        a, b = f(X).flatten(1), f(Xt).flatten(1)
    return ((a - b).norm(dim=1) / a.norm(dim=1)).mean().item()


torch.manual_seed(SEMILLA)
red_azar = RedConv()
etapas = {
    "conv1": lambda x: F.relu(red_azar.conv1(x)),
    "conv1 + pool": lambda x: F.max_pool2d(F.relu(red_azar.conv1(x)), 2),
    "conv2": lambda x: F.relu(red_azar.conv2(F.max_pool2d(F.relu(red_azar.conv1(x)), 2))),
    "conv2 + pool": red_azar.atributos,
}
invariancia = {nombre: cambio_relativo(f, X_pru[:1000]) for nombre, f in etapas.items()}
pd.Series(invariancia, name="cambio relativo al mover 1 pixel").round(3)

# %% [markdown]
# Cada *pooling* reduce el cambio; después de dos, la representación es bastante más estable ante desplazamientos pequeños, aunque no del todo. La invariancia es **aproximada**: un desplazamiento que cruza la frontera de un bloque sí cambia el resultado, y el aumento de datos (mostrar a la red imágenes desplazadas) suele completar el trabajo.
#
# ## 5. Una red convolucional pequeña contra un perceptrón multicapa
#
# Entrenamos con el mismo procedimiento la red convolucional de la sección anterior (20,490 parámetros) y un perceptrón multicapa con dos capas ocultas de 256 y 128 neuronas (235,146 parámetros, once veces más). Los dos con Adam, tasa de aprendizaje 0.001, minilotes de 64 imágenes y 3 épocas, y con el mismo orden de los minilotes. Normalizamos los pixeles con la media y la desviación estándar del conjunto de entrenamiento.

# %%
MEDIA, DE = X_ent.mean().item(), X_ent.std().item()
print(f"media {MEDIA:.4f}, desviación estándar {DE:.4f}")
Z_ent, Z_pru = (X_ent - MEDIA) / DE, (X_pru - MEDIA) / DE

N_ENT = 10_000 if RAPIDO else len(Z_ent)
EPOCAS = 1 if RAPIDO else 3
cargador_pru = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Z_pru, y_pru), batch_size=1000)


def entrenar_epoca(modelo, cargador, optimizador):
    modelo.train()
    total, n = 0.0, 0
    for xb, yb in cargador:
        xb, yb = xb.to(dispositivo), yb.to(dispositivo)
        optimizador.zero_grad()
        perdida = F.cross_entropy(modelo(xb), yb)
        perdida.backward()
        optimizador.step()
        total, n = total + perdida.item() * len(yb), n + len(yb)
    return total / n


@torch.no_grad()
def predecir(modelo, cargador):
    modelo.eval()
    return torch.cat([modelo(xb.to(dispositivo)).argmax(1).cpu() for xb, _ in cargador])


def evaluar(modelo, cargador):
    y = torch.cat([yb for _, yb in cargador])
    return (predecir(modelo, cargador) == y).float().mean().item()


def ajustar(modelo, epocas):
    # cada modelo recibe su propio cargador con la misma semilla: los mismos minilotes en el mismo orden
    cargador_ent = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Z_ent[:N_ENT], y_ent[:N_ENT]),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEMILLA))
    modelo.to(dispositivo)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    hist, t0 = [], time.time()
    for e in range(epocas):
        perdida = entrenar_epoca(modelo, cargador_ent, opt)
        hist.append((e + 1, perdida, evaluar(modelo, cargador_pru)))
    return pd.DataFrame(hist, columns=["época", "pérdida de entrenamiento", "exactitud de prueba"]), time.time() - t0


torch.manual_seed(SEMILLA)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
torch.manual_seed(SEMILLA)
cnn = RedConv()

hist_mlp, t_mlp = ajustar(mlp, EPOCAS)
hist_cnn, t_cnn = ajustar(cnn, EPOCAS)
resumen = pd.DataFrame({
    "parámetros": [contar(mlp), contar(cnn)],
    "exactitud de prueba": [hist_mlp.iloc[-1, 2], hist_cnn.iloc[-1, 2]],
    "segundos": [round(t_mlp), round(t_cnn)],
}, index=["perceptrón multicapa", "red convolucional"])
resumen

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
for h, nombre, color in ((hist_mlp, "perceptrón multicapa", GRIS), (hist_cnn, "red convolucional", AZUL)):
    ax.plot(h["época"], 100 * h["exactitud de prueba"], "o-", color=color)
ax.set_xticks(range(1, EPOCAS + 1))
ax.set_xlim(0.8, EPOCAS + 0.9)
ax.set_xlabel("época")
ax.set_ylabel("exactitud de prueba (%)")
ax.set_title("Fashion-MNIST: la red convolucional aprende más con 11 veces menos parámetros")
rdp.estilo.etiquetar(ax, EPOCAS + 0.08, {f"perceptrón multicapa ({contar(mlp):,} parám.)": 100 * hist_mlp.iloc[-1, 2],
                                          f"red convolucional ({contar(cnn):,} parám.)": 100 * hist_cnn.iloc[-1, 2]})
plt.show()

# %% [markdown]
# Con 3 épocas la red convolucional supera al perceptrón multicapa por casi 3 puntos de exactitud (89.4 % contra 86.7 %), con once veces menos parámetros; es decir, comete una cuarta parte menos de errores. Ambos seguirían mejorando con más épocas (y regularización, como vimos en la libreta anterior); las redes convolucionales bien entrenadas pasan de 92 % en este conjunto. La ventaja no viene de tener más capacidad, sino de tener **la capacidad correcta**: la arquitectura ya «sabe» que los pixeles vecinos importan juntos y que un patrón vale lo mismo en cualquier posición. Es un **sesgo inductivo**, en el sentido de la Unidad 3: restringir el espacio de hipótesis de una forma que coincide con el problema reduce la varianza sin pagar en sesgo.
#
# ## 6. Qué aprendió la red
#
# ### Filtros y mapas de activación
#
# Los 16 filtros de la primera capa son matrices de 3 × 3. Dibujémoslos junto con los mapas de activación (después de la ReLU) que producen en el botín de antes.

# %%
filtros = cnn.conv1.weight.detach().cpu()[:, 0]                       # (16, 3, 3)
with torch.no_grad():
    mapas = F.relu(cnn.conv1(Z_pru[IDX_PRENDA:IDX_PRENDA + 1].to(dispositivo)))[0].cpu()   # (16, 28, 28)

fig, ejes = plt.subplots(4, 8, figsize=(12, 6.4))
lim = filtros.abs().max()
for i in range(16):
    f, c = 2 * (i // 8), i % 8
    ejes[f, c].imshow(filtros[i], cmap=rdp.estilo.DIVERGENTE, vmin=-lim, vmax=lim)
    ejes[f, c].set_title(f"filtro {i + 1}", fontsize=10, loc="center")
    ejes[f + 1, c].imshow(mapas[i], cmap="gray_r")
for ax in ejes.ravel():
    ax.axis("off")
fig.suptitle("Primera capa: filtros aprendidos (azul +, rojo −) y su mapa de activación en el botín",
             x=0.01, ha="left", color=rdp.estilo.MARINO, fontweight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Varios filtros son detectores de bordes en distintas orientaciones, parecidos a los de Sobel (una columna o fila positiva frente a una negativa); otros responden al fondo o a las zonas claras de la prenda. Nadie los diseñó: salieron de minimizar la entropía cruzada. Los mapas muestran **dónde** responde cada uno: en un contorno, en la suela, en el interior.
#
# ### Errores típicos
#
# La matriz de confusión de la red convolucional en el conjunto de prueba:

# %%
pred_cnn = predecir(cnn, cargador_pru).numpy()
mc = confusion_matrix(y_pru.numpy(), pred_cnn)

fig, ax = plt.subplots(figsize=(8.5, 7.5))
ax.imshow(mc, cmap=rdp.estilo.SECUENCIAL, vmax=300)
for i in range(10):
    for j in range(10):
        if mc[i, j] > 0:
            ax.text(j, i, mc[i, j], ha="center", va="center", fontsize=9,
                    color="white" if mc[i, j] > 200 else TINTA)
ax.set_xticks(range(10), CLASES, rotation=45, ha="right")
ax.set_yticks(range(10), CLASES)
ax.set_xlabel("predicción")
ax.set_ylabel("clase verdadera")
ax.grid(False)
ax.set_title("Matriz de confusión de la red convolucional (prueba)")
plt.show()

pares = [(mc[i, j], CLASES[i], CLASES[j]) for i in range(10) for j in range(10) if i != j]
pd.DataFrame(sorted(pares, reverse=True)[:6], columns=["casos", "verdadera", "predicha"])

# %%
camisas = np.where((y_pru.numpy() == 6) & (pred_cnn == 0))[0][:8]
print(f"exactitud por clase: camisa {mc[6, 6] / mc[6].sum():.2f}, camiseta {mc[0, 0] / mc[0].sum():.2f}, "
      f"pantalón {mc[1, 1] / mc[1].sum():.2f}")
fig, ejes = plt.subplots(1, 8, figsize=(12, 2.0))
for ax, i in zip(ejes, camisas):
    ax.imshow(X_pru[i, 0], cmap="gray_r")
    ax.axis("off")
fig.suptitle("Camisas que la red llamó «camiseta»", x=0.01, ha="left", color=rdp.estilo.MARINO, fontweight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Los errores no están repartidos al azar: casi todos están entre las prendas de la parte superior del cuerpo (camisa, camiseta, suéter, abrigo). La camisa es la clase más difícil: a 28 × 28 pixeles, en escala de grises, una camisa de manga corta sin botones visibles **es** una camiseta. Varias de las imágenes de arriba serían difíciles también para una persona. Un error que se concentra en pares de clases parecidas es una señal de que el límite está en la información de la imagen, no solo en el modelo.
#
# ## 7. Redes profundas y conexiones residuales
#
# **Un poco de historia.** Y. LeCun y sus colegas usaron redes convolucionales para leer dígitos de cheques y códigos postales desde finales de los años ochenta; su LeNet-5 (LeCun, Bottou, Bengio y Haffner, *Proceedings of the IEEE*, 1998) tenía dos capas convolucionales, cada una seguida de una reducción de tamaño, y unos 60,000 parámetros: es, en esencia, la red que acabamos de entrenar. En 2012, AlexNet (A. Krizhevsky, I. Sutskever y G. Hinton, NeurIPS 2012), con cinco capas convolucionales, 60 millones de parámetros, ReLU, *dropout* y dos GPU, ganó el concurso ImageNet (1.2 millones de fotos, 1,000 clases) con un error *top-5* de 15.3 %, contra 26.2 % del segundo lugar. Eso desató la carrera por redes más profundas: VGG (19 capas, 2014), GoogLeNet (22 capas, 2014)… y un obstáculo.
#
# **El problema de la degradación.** K. He, X. Zhang, S. Ren y J. Sun (CVPR 2016) observaron que, al agregar capas a una red convolucional «simple», el error **de entrenamiento** empeoraba: una red de 56 capas se ajustaba peor que una de 20 en CIFAR-10. No era sobreajuste (es error de entrenamiento) y en principio la red profunda puede imitar a la menos profunda (bastaría con que las capas extra fueran la identidad), pero el descenso por gradiente no encontraba esa solución.
#
# **La solución.** En lugar de pedirle a un bloque de capas que aprenda una transformación $H(\mathbf{x})$, se le pide que aprenda la **corrección** $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$, y se suma la entrada: $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + F_\ell(\mathbf{x}_\ell)$. Si lo mejor es no hacer nada, basta con $F = 0$, que es fácil de alcanzar. Y el gradiente tiene un camino directo:
#
# $$\frac{\partial \mathbf{x}_{\ell+1}}{\partial \mathbf{x}_\ell} = I + \frac{\partial F_\ell}{\partial \mathbf{x}_\ell}
# \qquad\Longrightarrow\qquad
# \frac{\partial E}{\partial \mathbf{x}_\ell} = \frac{\partial E}{\partial \mathbf{x}_L}\prod_{m=\ell}^{L-1}\Big(I + \frac{\partial F_m}{\partial \mathbf{x}_m}\Big) .$$
#
# Al desarrollar el producto aparece el término $\partial E/\partial \mathbf{x}_L$ tal cual: el error de la salida llega a cualquier capa sin atravesar ninguna multiplicación por matrices de pesos. En una red sin atajos, el producto es $\prod_m \partial F_m/\partial \mathbf{x}_m$, y si esas jacobianas encogen los vectores, el gradiente se desvanece exponencialmente con la profundidad.
#
# Veámoslo con una simulación: una red de 40 capas de ancho 64, con la inicialización por omisión de PyTorch, con y sin conexiones residuales. Medimos la norma del gradiente de los pesos de cada capa en un minilote de datos al azar, relativa a la de la última capa: cuánto del gradiente «sobrevive» al viajar hacia la entrada.

# %%
PROFUNDIDAD, ANCHO = 40, 64


class Profunda(nn.Module):
    def __init__(self, residual, inicializacion_he=False):
        super().__init__()
        self.residual = residual
        self.capas = nn.ModuleList(nn.Linear(ANCHO, ANCHO) for _ in range(PROFUNDIDAD))
        if inicializacion_he:
            for c in self.capas:
                nn.init.kaiming_normal_(c.weight, nonlinearity="relu"); nn.init.zeros_(c.bias)

    def forward(self, x):
        for c in self.capas:
            x = x + c(F.relu(x)) if self.residual else c(F.relu(x))
        return x


def normas_gradiente(modelo):
    torch.manual_seed(SEMILLA)
    x, y = torch.randn(256, ANCHO), torch.randn(256, ANCHO)
    modelo.zero_grad()
    F.mse_loss(modelo(x), y).backward()
    g = np.array([c.weight.grad.norm().item() for c in modelo.capas])
    return g / g[-1]          # relativa a la de la última capa


grad = {}
for nombre, residual, he in [("simple", False, False), ("simple, inicialización de He", False, True), ("residual", True, False)]:
    torch.manual_seed(SEMILLA)
    grad[nombre] = normas_gradiente(Profunda(residual, he))

fig, ax = plt.subplots(figsize=(9, 4.8))
capas = np.arange(1, PROFUNDIDAD + 1)
for (nombre, g), color in zip(grad.items(), (ROJO, DORADO_OSC, AZUL)):
    ax.semilogy(capas, g, "o-", ms=3.5, color=color)
ax.set_xlabel("capa (1 = la más cercana a la entrada)")
ax.set_ylabel("norma del gradiente (última capa = 1)")
ax.set_title(f"Red de {PROFUNDIDAD} capas al inicializarla: ¿llega el gradiente a las primeras capas?")
posiciones = {"simple": (12, 40), "simple, inicialización de He": (2, 6), "residual": (2, 0.12)}   # (capa, factor vertical)
for n, (c, fac) in posiciones.items():
    ax.text(c, grad[n][c - 1] * fac, n, fontsize=11.5, color=TINTA)
ax.set_xlim(0, PROFUNDIDAD + 1)
plt.show()
print({n: f"capa 1: {g[0]:.1e}" for n, g in grad.items()})

# %% [markdown]
# Con la inicialización por omisión de PyTorch (pesos uniformes con varianza $1/(3\,n_{\text{ent}})$), cada capa simple multiplica la varianza de la señal por cerca de $n_{\text{ent}} \cdot \frac{1}{3 n_{\text{ent}}} \cdot \frac12 = \frac16$, y el gradiente que llega a la primera capa es decenas de órdenes de magnitud menor que el de la última: esas capas no aprenden nada. La inicialización de He (Unidad 6, ejemplo 1) arregla la escala **al inicio**, pero no garantiza nada durante el entrenamiento, y es justo ahí donde He y sus colegas observaron la degradación. La red residual, con la misma inicialización mala, entrega gradientes del mismo orden a todas sus capas.
#
# Con conexiones residuales (y normalización por lotes) se entrenaron redes de 152 capas que ganaron ImageNet 2015 con 3.57 % de error *top-5*. ResNet-18, la que usaremos enseguida, es la versión pequeña: 18 capas con pesos y 11.7 millones de parámetros.
#
# ## 8. Transferencia: imágenes de satélite con pocas etiquetas
#
# ### Los datos y el problema
#
# **Datos.** EuroSAT (P. Helber, B. Bischke, A. Dengel y D. Borth, *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2019) tiene 27,000 imágenes de 64 × 64 pixeles tomadas por el satélite Sentinel-2 del programa Copernicus de la Unión Europea, sobre territorio europeo, etiquetadas en 10 clases de uso de suelo. Cada pixel cubre 10 m × 10 m, así que una imagen es un cuadro de 640 m de lado. Usamos la versión RGB (el satélite mide 13 bandas espectrales). Licencia MIT; `torchvision` la descarga la primera vez (unos 90 MB).
#
# **Problema.** Clasificar el uso de suelo de un cuadro de terreno. Etiquetar imágenes de satélite es caro (alguien tiene que saber qué hay en cada cuadro), así que la pregunta es: **¿qué tan bien podemos hacerlo con 10, 50 o 200 imágenes etiquetadas por clase?**
#
# **La idea de la transferencia.** Una ResNet-18 entrenada en ImageNet (fotos de perros, autos, frutas…) aprendió en sus capas intermedias detectores de bordes, texturas, patrones repetidos y formas que sirven para muchas otras imágenes. Quitamos su última capa (la que decide entre las 1,000 clases de ImageNet) y usamos los 512 números que entrega la penúltima como **atributos** de cada imagen de satélite; encima ajustamos una regresión logística, como en la Unidad 4. La red no ve ninguna imagen de satélite durante su entrenamiento.

# %%
CLASES_SAT = ["cultivo anual", "bosque", "vegetación herbácea", "carretera", "zona industrial",
              "pastizal", "cultivo permanente", "zona residencial", "río", "mar o lago"]

eurosat = torchvision.datasets.EuroSAT(RAIZ_DATOS, download=True)
y_sat = np.array(eurosat.targets)
print(f"{len(eurosat):,} imágenes;", dict(zip(eurosat.classes, np.bincount(y_sat))))

# %% [markdown]
# Para que la libreta corra en CPU, trabajamos con un subconjunto fijo: para cada clase, 200 imágenes al azar forman la reserva de entrenamiento y otras 100, distintas, la prueba (1,000 imágenes de prueba en total). Ordenamos la reserva de modo que sus primeras $10k$ imágenes tengan exactamente $k$ de cada clase; así, el conjunto con 10 por clase está contenido en el de 50, y este en el de 200.

# %%
POR_CLASE = (10, 50) if RAPIDO else (10, 50, 200)
N_PRUEBA_SAT = 50 if RAPIDO else 100
rng = np.random.default_rng(SEMILLA)
reserva, prueba = [], []
for c in range(10):
    ids = rng.permutation(np.where(y_sat == c)[0])
    reserva.append(ids[:max(POR_CLASE)])
    prueba.append(ids[max(POR_CLASE):max(POR_CLASE) + N_PRUEBA_SAT])
ids_ent = np.stack(reserva, axis=1).ravel()        # intercalado: clase 0, 1, …, 9, 0, 1, …
ids_pru = np.concatenate(prueba)


def cargar_sat(ids):
    return torch.stack([torchvision.transforms.functional.to_tensor(eurosat[int(i)][0]) for i in ids])


S_ent, S_pru = cargar_sat(ids_ent), cargar_sat(ids_pru)
ys_ent, ys_pru = y_sat[ids_ent], y_sat[ids_pru]
print(S_ent.shape, S_pru.shape)

fig, ejes = plt.subplots(3, 10, figsize=(13, 4.6))
for c in range(10):
    for f in range(3):
        ax = ejes[f, c]
        ax.imshow(S_ent[ys_ent == c][f].permute(1, 2, 0))
        ax.axis("off")
    ejes[0, c].set_title(CLASES_SAT[c].replace(" ", "\n", 1) if len(CLASES_SAT[c]) > 12 else CLASES_SAT[c], fontsize=10.5, loc="center")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Atributos de una red preentrenada
#
# Las imágenes se normalizan con la media y la desviación estándar de ImageNet por canal, que es lo que la red espera. No cambiamos su tamaño: ResNet-18 termina con un promedio global (*global average pooling*) y acepta imágenes de cualquier tamaño.

# %%
resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
print(f"ResNet-18: {contar(resnet):,} parámetros")
resnet.fc = nn.Identity()            # quitamos la capa de 1,000 clases: la salida es el vector de 512 atributos
resnet.eval().to(dispositivo)

MEDIA_IN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DE_IN = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
N_ent_sat, N_pru_sat = (S_ent - MEDIA_IN) / DE_IN, (S_pru - MEDIA_IN) / DE_IN


@torch.no_grad()
def atributos_resnet(X, lote=100):
    return torch.cat([resnet(X[i:i + lote].to(dispositivo)).cpu() for i in range(0, len(X), lote)]).numpy()


t0 = time.time()
A_ent, A_pru = atributos_resnet(N_ent_sat), atributos_resnet(N_pru_sat)
print(f"atributos: {A_ent.shape} y {A_pru.shape} en {time.time() - t0:.0f} s")

exact_transfer = {}
for k in POR_CLASE:
    logistica = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    logistica.fit(A_ent[:10 * k], ys_ent[:10 * k])
    exact_transfer[k] = logistica.score(A_pru, ys_pru)
pd.Series(exact_transfer, name="exactitud de prueba").rename_axis("imágenes por clase")

# %% [markdown]
# Con solo **10 imágenes por clase** (100 en total), una regresión logística sobre los atributos de ImageNet acierta en más de tres de cada cuatro imágenes de satélite, en un problema de 10 clases donde adivinar daría 10 %.
#
# ### La comparación: una red desde cero con los mismos datos
#
# ¿Es mérito de la red preentrenada o el problema es fácil? Entrenemos desde cero una red convolucional pequeña (tres bloques convolución + ReLU + *max-pooling*, un promedio global y una capa lineal: unos 24,000 parámetros) con las mismas imágenes. Para darle una oportunidad justa usamos **aumento de datos**: en cada paso volteamos al azar las imágenes horizontal y verticalmente (una foto de satélite volteada sigue siendo del mismo tipo de suelo). Le damos a cada tamaño de muestra el mismo presupuesto: 600 pasos de Adam con minilotes de 32.

# %%
def red_satelite():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 64 → 32
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 32 → 16
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 16 → 8
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


def entrenar_pasos(modelo, X, y, pasos, lr=1e-3, lote=32):
    """Adam con minilotes al azar y volteos aleatorios (aumento de datos)."""
    modelo.to(dispositivo).train()
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    g = torch.Generator().manual_seed(SEMILLA)
    y = torch.as_tensor(y)
    for _ in range(pasos):
        idx = torch.randint(len(X), (lote,), generator=g)
        xb = X[idx]
        if torch.rand(1, generator=g) < 0.5:
            xb = xb.flip(3)
        if torch.rand(1, generator=g) < 0.5:
            xb = xb.flip(2)
        opt.zero_grad()
        F.cross_entropy(modelo(xb.to(dispositivo)), y[idx].to(dispositivo)).backward()
        opt.step()
    modelo.eval()
    with torch.no_grad():
        pred = torch.cat([modelo(N_pru_sat[i:i + 200].to(dispositivo)).argmax(1).cpu() for i in range(0, len(N_pru_sat), 200)])
    return (pred.numpy() == ys_pru).mean()


PASOS = 150 if RAPIDO else 600
print(f"red desde cero: {contar(red_satelite()):,} parámetros")
exact_cero = {}
t0 = time.time()
for k in POR_CLASE:
    torch.manual_seed(SEMILLA)
    exact_cero[k] = entrenar_pasos(red_satelite(), N_ent_sat[:10 * k], ys_ent[:10 * k], PASOS)
print(f"{time.time() - t0:.0f} s")

comparacion = pd.DataFrame({"transferencia (ResNet-18 + logística)": exact_transfer, "red pequeña desde cero": exact_cero})
comparacion.index.name = "imágenes por clase"
comparacion

# %%
fig, ax = plt.subplots(figsize=(8, 4.6))
ks = list(POR_CLASE)
ax.plot(ks, [100 * exact_transfer[k] for k in ks], "o-", color=AZUL)
ax.plot(ks, [100 * exact_cero[k] for k in ks], "o-", color=ROJO)
ax.axhline(10, color=GRIS, ls=":", lw=1.5)
ax.text(ks[0], 12, "adivinar al azar", color=TENUE, fontsize=11)
ax.set_xscale("log")
ax.set_xticks(ks, [str(k) for k in ks])
ax.minorticks_off()
ax.set_xlim(ks[0] * 0.8, ks[-1] * 3.2)
ax.set_ylim(0, 100)
ax.set_xlabel("imágenes etiquetadas por clase (escala logarítmica)")
ax.set_ylabel("exactitud de prueba (%)")
ax.set_title("EuroSAT: transferencia contra entrenar desde cero")
rdp.estilo.etiquetar(ax, ks[-1] * 1.12, {"ResNet-18 de ImageNet + logística": 100 * exact_transfer[ks[-1]],
                                         "red pequeña desde cero": 100 * exact_cero[ks[-1]]})
plt.show()

# %% [markdown]
# La red preentrenada con 10 imágenes por clase supera a la red desde cero con 200. Entrenar desde cero no es inútil: con miles de imágenes por clase y más pasos, una red convolucional entrenada solo con EuroSAT llega a exactitudes muy altas (los autores del conjunto reportan 98.57 % ajustando redes profundas preentrenadas con el 80 % de los datos). La transferencia rinde más **cuando hay pocas etiquetas**, que es la situación común en la práctica: en imágenes médicas, agrícolas o industriales casi nunca hay millones de ejemplos etiquetados.
#
# ¿Por qué funcionan atributos aprendidos en fotos de perros y autos sobre imágenes de satélite? Porque las primeras capas aprenden cosas muy generales (bordes, colores, texturas) y las intermedias, combinaciones de ellas (rayas, cuadrículas, manchas) que también describen un campo cultivado, un bosque o una zona residencial. Veamos dónde se equivoca el modelo con más datos:

# %%
k = max(POR_CLASE)
logistica = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(A_ent[:10 * k], ys_ent[:10 * k])
mc_sat = confusion_matrix(ys_pru, logistica.predict(A_pru))
pares_sat = [(mc_sat[i, j], CLASES_SAT[i], CLASES_SAT[j]) for i in range(10) for j in range(10) if i != j]
print("exactitud por clase:", {CLASES_SAT[c]: round(mc_sat[c, c] / mc_sat[c].sum(), 2) for c in range(10)})
pd.DataFrame(sorted(pares_sat, reverse=True)[:5], columns=["casos", "verdadera", "predicha"])

# %% [markdown]
# Los errores vuelven a ser razonables: ríos y carreteras que se confunden entre sí (en los dos casos, una franja larga que cruza el cuadro) y tipos de vegetación y cultivo que se parecen a esta resolución.
#
# ### Ajuste fino (opcional, con GPU)
#
# La transferencia tiene un segundo nivel: en lugar de congelar la red y entrenar solo la logística de encima, se continúa el entrenamiento de **toda** la red con las imágenes nuevas y una tasa de aprendizaje pequeña, para que los atributos se adapten al problema. Esto se llama **ajuste fino** (*fine-tuning*). Es mucho más caro (hay que retropropagar por los 11 millones de parámetros), así que solo lo corremos si hay una GPU. En Colab, activa la GPU y compara su exactitud con la de la tabla anterior.

# %%
if dispositivo.type == "cuda":
    for k in POR_CLASE:
        torch.manual_seed(SEMILLA)
        ajustada = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        ajustada.fc = nn.Linear(512, 10)
        exactitud = entrenar_pasos(ajustada, N_ent_sat[:10 * k], ys_ent[:10 * k], pasos=300, lr=1e-4)
        print(f"ajuste fino con {k} imágenes por clase: exactitud {exactitud:.3f}")
else:
    print("Sin GPU: se omite el ajuste fino.")

# %% [markdown]
# ## 9. Lo que se hereda: el caso *Gender Shades*
#
# La transferencia es poderosa porque el modelo **hereda** lo que aprendió con otros datos. También hereda sus huecos. J. Buolamwini y T. Gebru («Gender Shades», *Conference on Fairness, Accountability and Transparency*, 2018) evaluaron los clasificadores comerciales de género a partir de fotos de rostros de Microsoft, IBM y Face++ con 1,270 fotos de parlamentarios de seis países (tres africanos y tres europeos), etiquetadas por género y por tono de piel con la escala de Fitzpatrick. Los tres sistemas tenían su mayor error en las **mujeres de piel oscura**, de hasta **34.7 %**, mientras que el error máximo en **hombres de piel clara** fue de **0.8 %**. Los autores también mostraron que dos conjuntos de rostros muy usados para entrenar y evaluar, IJB-A y Adience, tenían entre 79.6 % y 86.2 % de personas de piel clara.
#
# Un detector de atributos entrenado con datos que representan mal a un grupo produce atributos peores para ese grupo, y ningún clasificador que se ajuste encima puede recuperar la información que no está ahí. Al usar un modelo preentrenado conviene preguntarse, igual que con cualquier conjunto de datos (Unidad 1): **¿con qué datos se entrenó, a quién representan y a quién no?** Y evaluar por separado en cada grupo relevante, como hicimos en la Unidad 3.
#
# ## 10. Cierre
#
# - Una convolución es una capa lineal con dos restricciones: **conexiones locales** y **pesos compartidos**. El resultado es equivariante a traslaciones y tiene muchísimos menos parámetros que una capa densa.
# - El tamaño de la salida es $\lfloor (n + 2p - k)/s \rfloor + 1$; los parámetros, $(C_{\text{ent}}k^2 + 1)C_{\text{sal}}$; el campo receptivo crece con la profundidad y con el *pooling*.
# - Con once veces menos parámetros, una red convolucional pequeña le gana a un perceptrón multicapa en Fashion-MNIST: el sesgo inductivo correcto vale más que la capacidad.
# - Las conexiones residuales, $\mathbf{x} + F(\mathbf{x})$, dan al gradiente un camino directo y permiten entrenar redes de cientos de capas.
# - Con pocas etiquetas, los atributos de una red preentrenada en otro problema superan por mucho a entrenar desde cero; pero el modelo hereda los sesgos de sus datos de origen.
#
# *Referencias:* Y. LeCun, L. Bottou, Y. Bengio y P. Haffner, «Gradient-based learning applied to document recognition», *Proc. IEEE* 86 (1998). A. Krizhevsky, I. Sutskever y G. E. Hinton, «ImageNet classification with deep convolutional neural networks», NeurIPS (2012). K. He, X. Zhang, S. Ren y J. Sun, «Deep residual learning for image recognition», CVPR (2016). H. Xiao, K. Rasul y R. Vollgraf, «Fashion-MNIST», arXiv:1708.07747 (2017). P. Helber et al., «EuroSAT», *IEEE JSTARS* 12 (2019). J. Buolamwini y T. Gebru, «Gender Shades», PMLR 81 (2018). I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press (2016), capítulo 9.

# %% [markdown]
# ## 11. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p09", "u06-p10", "u06-p11", "u06-p12"])

# %% [markdown]
# ## 12. Ejercicios
#
# 1. **Contar a mano.** Una capa convolucional recibe un mapa de $32 \times 32 \times 64$ y aplica 128 filtros de $5 \times 5$ con relleno 2 y paso 2. ¿De qué tamaño es la salida y cuántos parámetros tiene? ¿Cuántos tendría una capa densa que produjera una salida del mismo tamaño? Comprueba con `nn.Conv2d` y `nn.Linear`.
# 2. **Equivariancia y *pooling*.** (a) Demuestra que la composición de dos convoluciones es equivariante a traslaciones y que también lo es una ReLU aplicada pixel por pixel. (b) Da un ejemplo de una imagen de una dimensión y un desplazamiento de un pixel para los que el *max-pooling* de tamaño 2 y paso 2 **no** es invariante. (c) ¿Para qué desplazamientos sí lo es siempre?
# 3. **Campo receptivo.** Usa la recurrencia de la sección 4 para calcular el campo receptivo de tres capas convolucionales de $3 \times 3$ seguidas, sin *pooling*, y compáralo con el de una sola capa de $7 \times 7$. Compara también el número de parámetros con $C$ canales de entrada y salida en todas. (Este fue el argumento de VGG para usar solo filtros de 3 × 3.)
# 4. **Gradiente en una red residual.** Para una red escalar $x_{\ell+1} = x_\ell + a\,x_\ell$ con $L$ bloques, calcula $\partial x_L/\partial x_0$ y compárala con la red simple $x_{\ell+1} = a\,x_\ell$ cuando $|a| < 1$. ¿Qué pasa en la red residual si $a$ es pequeño y $L$ grande? ¿Y si $a = -1$?
# 5. **Más etiquetas, otro modelo.** Agrega a la comparación de EuroSAT el caso de 1,000 imágenes por clase (la reserva tiene que crecer). ¿Alcanza la red desde cero a la transferencia? Prueba también más pasos. Si tienes GPU, compara con el ajuste fino.
