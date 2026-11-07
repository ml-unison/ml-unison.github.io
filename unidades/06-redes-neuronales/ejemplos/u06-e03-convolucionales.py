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
# # Redes convolucionales con Fashion-MNIST: de LeNet a EfficientNet
#
# **Unidad 6 · Ejemplo 3**
#
# Un perceptrón multicapa trata una imagen como una lista de números: si revolvemos los pixeles con la misma permutación en todas las imágenes, aprende exactamente igual. Pero una imagen no es una lista: los pixeles vecinos están relacionados, y un borde, un botón o una costura significan lo mismo en cualquier parte del cuadro. Las **redes convolucionales** incorporan esas dos ideas en la arquitectura. En esta libreta vamos a:
#
# 1. ver por qué las imágenes (y otros datos donde lo local importa) merecen una arquitectura propia, y cuántos parámetros necesitaría un perceptrón multicapa para ver una foto;
# 2. programar la **convolución** discreta, aplicar a una prenda filtros de bordes hechos a mano y verificar su **equivariancia a traslaciones**;
# 3. entender el **relleno**, el **paso**, los **canales**, el **campo receptivo** y el ***pooling***, y contar parámetros;
# 4. entrenar una red convolucional pequeña en Fashion-MNIST y compararla con un perceptrón multicapa que tiene once veces más parámetros; mirar sus filtros, sus mapas de activación y sus errores;
# 5. comprobar que la parte convolucional de la red es un **preprocesamiento**: convierte pixeles en atributos, y encima cualquier clasificador (una regresión logística, por ejemplo) hace el resto;
# 6. recorrer las arquitecturas que hicieron historia (LeNet, AlexNet, **VGG**, **ResNet**, **Inception**, **Xception** y **EfficientNet**), programando y contando las piezas que introdujo cada una.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado. PyTorch y torchvision ya vienen instalados en Colab. Si tienes una GPU (en Colab: *Entorno de ejecución → Cambiar tipo de entorno*), la libreta la usa; si no, corre en CPU en unos cinco minutos.

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
# ## 1. Por qué importan las imágenes (y no solo las imágenes)
#
# Buena parte de los datos que se generan hoy son imágenes: fotos de teléfonos, cámaras de vigilancia y de líneas de producción, estudios médicos, imágenes de satélite. Algunos ejemplos de problemas que se resolvieron con las redes de esta libreta:
#
# - **Medicina.** Detectar retinopatía diabética en fotografías del fondo del ojo (V. Gulshan et al., *JAMA*, 2016) o clasificar lesiones de la piel a partir de fotos (A. Esteva et al., *Nature*, 2017), con exactitudes comparables a las de especialistas en los conjuntos de prueba de esos estudios.
# - **Agricultura y medio ambiente.** Los satélites Sentinel-2 del programa europeo Copernicus fotografían toda la superficie terrestre cada pocos días y sus imágenes son libres; con ellas se clasifican cultivos, se mide la deforestación o se vigilan sequías. En el campo, se cuentan espigas de trigo en fotos para estimar el rendimiento (el conjunto *Global Wheat Head Detection*, E. David et al., *Plant Phenomics*, 2020).
# - **Industria y vida diaria.** Inspección visual de piezas defectuosas, lectura de placas y documentos, desbloqueo del teléfono con la cara.
#
# El momento en que las redes neuronales profundas se volvieron la herramienta estándar fue, precisamente, un concurso de imágenes: ImageNet 2012 (1.2 millones de fotos de entrenamiento en 1,000 clases), que ganó por mucho una red convolucional. Lo veremos en la sección 9.
#
# Pero lo que aprovechan estas redes no es que los datos sean fotos, sino que **lo local importa**: el valor en una posición se entiende mirando a sus vecinos, y el mismo patrón puede aparecer en cualquier lugar. Eso también pasa en:
#
# - el **audio**, convertido en espectrograma (una imagen de frecuencia contra tiempo), donde se reconocen sonidos y palabras (S. Hershey et al., ICASSP 2017);
# - las **series de tiempo**, con convoluciones de una dimensión a lo largo del tiempo (WaveNet, A. van den Oord et al., 2016; S. Bai, J. Z. Kolter y V. Koltun, 2018);
# - las **secuencias de ADN**, donde un filtro de una dimensión que recorre la cadena funciona como un detector de motivos, pequeñas subsecuencias con una función biológica (J. Zhou y O. G. Troyanskaya, *Nature Methods*, 2015).
#
# Todo lo que hagamos en esta libreta con imágenes de dos dimensiones se traslada a esos casos cambiando `nn.Conv2d` por `nn.Conv1d`.

# %% [markdown]
# ## 2. Una pregunta para empezar: ¿cuántos parámetros?
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
# ## 3. Los datos y el problema: Fashion-MNIST
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
# ## 4. La convolución
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
# ## 5. Relleno, paso, canales y conteo de parámetros
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
# Una neurona de `conv1` ve una ventana de 3 × 3 de la imagen. Una de `conv2` ve 3 × 3 posiciones del mapa anterior, ya reducido a la mitad por el *pooling*: en la imagen original eso abarca más. Si $r_l$ es el lado del **campo receptivo** después de la capa $l$ y $j_l$ la distancia, en pixeles de la imagen, entre dos posiciones vecinas del mapa, una capa con ventana $k$ y paso $s$ cumple
#
# $$r_l = r_{l-1} + (k - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s, \qquad r_0 = j_0 = 1 .$$
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
# ## 6. Una red convolucional pequeña contra un perceptrón multicapa
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
# Con 3 épocas la red convolucional supera al perceptrón multicapa por 2.5 puntos de exactitud (89.3 % contra 86.8 %), con once veces menos parámetros; es decir, comete una quinta parte menos de errores (10.7 % de las imágenes de prueba contra 13.2 %). Ambos seguirían mejorando con más épocas (y regularización, como vimos en la libreta anterior); las redes convolucionales bien entrenadas pasan de 92 % en este conjunto. La ventaja no viene de tener más capacidad, sino de tener **la capacidad correcta**: la arquitectura ya «sabe» que los pixeles vecinos importan juntos y que un patrón vale lo mismo en cualquier posición. Es un **sesgo inductivo**, en el sentido de la Unidad 3: restringir el espacio de hipótesis de una forma que coincide con el problema reduce la varianza sin pagar en sesgo.
#
# ## 7. Qué aprendió la red
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
# ## 8. La parte convolucional es un preprocesamiento
#
# Miremos otra vez la red que entrenamos. Tiene dos partes:
#
# - `atributos`: dos bloques «convolución + ReLU + *max-pooling*» que convierten los $28 \times 28 = 784$ pixeles de una prenda en 32 mapas de $7 \times 7$, es decir, $32 \cdot 7 \cdot 7 = 1{,}568$ números. Cada uno dice qué tanto aparece un patrón (el que detecta uno de los 32 filtros de `conv2`) en una región de 10 × 10 pixeles de la imagen.
# - `salida`: una capa lineal de 1,568 entradas a 10 salidas, seguida de la *softmax* que va dentro de la entropía cruzada. Eso es, exactamente, una **regresión logística multinomial** (Unidad 4) sobre esos 1,568 atributos.
#
# Visto así, una red convolucional es un **extractor de atributos** seguido de un clasificador de los que ya conocemos. La diferencia con la Unidad 4 es que los atributos no los diseñamos a mano (como los filtros de Sobel): se aprendieron junto con el clasificador, por descenso por gradiente.
#
# Si eso es cierto, deberíamos poder **congelar** la parte convolucional ya entrenada, pasarle todas las imágenes una vez, guardar sus 1,568 atributos como una tabla común y corriente $\mathbf{X} \in \mathbb{R}^{M \times 1568}$, y ajustar encima un clasificador cualquiera. Comparamos cinco combinaciones:
#
# 1. los **pixeles crudos** (784 atributos) con una regresión logística: el punto de partida;
# 2. la parte convolucional **con pesos al azar** (la `red_azar` de la sección 5, sin entrenar) con una regresión logística: ¿basta la arquitectura?;
# 3. la parte convolucional **entrenada**, congelada, con una regresión logística de `scikit-learn`;
# 4. la misma, con un perceptrón multicapa pequeño encima (una capa oculta de 128 neuronas);
# 5. la red convolucional completa, como la entrenamos en la sección 6.
#
# Para que la regresión logística corra en pocos segundos en una CPU, la ajustamos con un **submuestreo** de 10,000 imágenes de entrenamiento (las primeras), estandarizando cada atributo y con una regularización $L_2$ fuerte (`C=0.01`, es decir, $\lambda = 1/C = 100$ en la escala de `scikit-learn`): con 1,568 atributos y solo 10,000 imágenes, la regularización por omisión (`C=1`) sobreajusta. El perceptrón multicapa de la opción 4 sí usa todas. La exactitud siempre se mide en las 10,000 imágenes de prueba.

# %%
@torch.no_grad()
def extraer(f, Z, lote=2000):
    """Aplica la función f (la parte convolucional) por lotes y aplana: una fila de atributos por imagen."""
    return torch.cat([f(Z[i:i + lote].to(dispositivo)).flatten(1).cpu() for i in range(0, len(Z), lote)]).numpy()


cnn.eval()
red_azar.eval()
t0 = time.time()
A_ent, A_pru = extraer(cnn.atributos, Z_ent[:N_ENT]), extraer(cnn.atributos, Z_pru)
print(f"atributos de la parte convolucional entrenada: {A_ent.shape} y {A_pru.shape} en {time.time() - t0:.1f} s")

N_LOG = 3_000 if RAPIDO else 10_000          # submuestreo para la regresión logística


def logistica_sobre(X_e, X_p):
    """Regresión logística estandarizada, ajustada con las primeras N_LOG imágenes; exactitud de prueba y segundos."""
    t0 = time.time()
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000))
    modelo.fit(X_e[:N_LOG], y_ent[:N_LOG].numpy())
    return modelo.score(X_p, y_pru.numpy()), time.time() - t0


extractor = {}
extractor["pixeles crudos + logística"] = logistica_sobre(Z_ent[:N_LOG].flatten(1).numpy(), Z_pru.flatten(1).numpy())
extractor["conv. al azar + logística"] = logistica_sobre(extraer(red_azar.atributos, Z_ent[:N_LOG]), extraer(red_azar.atributos, Z_pru))
extractor["conv. entrenada + logística"] = logistica_sobre(A_ent, A_pru)
for nombre, (exact, seg) in extractor.items():
    print(f"{nombre:30s} exactitud {exact:.4f}   ({seg:.0f} s)")

# %% [markdown]
# Ahora un perceptrón multicapa pequeño sobre los mismos atributos congelados. Como la parte convolucional ya no cambia, entrenarlo es barato: la red de abajo no participa en el descenso por gradiente, solo la tabla de atributos.

# %%
torch.manual_seed(SEMILLA)
mlp_encima = nn.Sequential(nn.Linear(1568, 128), nn.ReLU(), nn.Linear(128, 10)).to(dispositivo)
opt = torch.optim.Adam(mlp_encima.parameters(), lr=1e-3)
T_ent, T_pru = torch.tensor(A_ent), torch.tensor(A_pru)
cargador_atr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(T_ent, y_ent[:N_ENT]), batch_size=64,
                                           shuffle=True, generator=torch.Generator().manual_seed(SEMILLA))
t0 = time.time()
for _ in range(EPOCAS):
    entrenar_epoca(mlp_encima, cargador_atr, opt)
exact_mlp_encima = evaluar(mlp_encima, torch.utils.data.DataLoader(torch.utils.data.TensorDataset(T_pru, y_pru), batch_size=1000))
extractor["conv. entrenada + perceptrón"] = (exact_mlp_encima, time.time() - t0)
extractor["red convolucional completa"] = (hist_cnn.iloc[-1, 2], t_cnn)

tabla_extractor = pd.DataFrame(extractor, index=["exactitud de prueba", "segundos"]).T
tabla_extractor["exactitud de prueba"] = tabla_extractor["exactitud de prueba"].round(4)
tabla_extractor["segundos"] = tabla_extractor["segundos"].round(1)
tabla_extractor

# %%
fig, ax = plt.subplots(figsize=(9, 4.2))
nombres = list(extractor)[::-1]
valores = [100 * extractor[n][0] for n in nombres]
colores = [AZUL if "entrenada" in n or "completa" in n else GRIS for n in nombres]
ax.barh(nombres, valores, color=colores)
for i, v in enumerate(valores):
    ax.text(v + 0.2, i, f"{v:.1f} %", va="center", fontsize=11, color=TINTA)
ax.set_xlim(75, 95)
ax.set_xlabel("exactitud de prueba (%)")
ax.set_title("Mismos clasificadores, distintos atributos")
ax.grid(axis="y", visible=False)
plt.show()

# %% [markdown]
# Las cifras cuentan la historia completa:
#
# - Con los pixeles crudos, la regresión logística llega a 83.6 %. Es lineal en los pixeles: no puede expresar «hay un borde vertical por aquí».
# - La parte convolucional **con pesos al azar** ya ayuda (87.2 %): la arquitectura (ventanas locales, ReLU, *pooling*) por sí sola produce atributos mejores que los pixeles. Pero el resto del camino lo da el entrenamiento.
# - Con la parte convolucional **entrenada** y congelada, la misma regresión logística, con solo 10,000 imágenes, llega a 89.3 %, la misma exactitud de la red completa (89.3 %). Un perceptrón pequeño encima, entrenado con las 60,000 imágenes, llega a 90.5 %, un poco más que la red original: su capa oculta le da al clasificador de encima algo de flexibilidad que la capa lineal no tenía.
#
# Es decir: **lo que la red aprendió está en sus atributos**. La parte convolucional transforma 784 pixeles, sobre los que un modelo lineal se queda corto, en 1,568 atributos locales sobre los que casi cualquier clasificador funciona bien. A partir de aquí, el «modelo de encima» puede ser una regresión logística, un perceptrón multicapa, una máquina de vectores de soporte o un bosque aleatorio.
#
# Esta forma de pensar tiene una consecuencia importante: una parte convolucional entrenada con muchas imágenes de un problema puede servir como preprocesamiento **para otro problema**, y encima se ajusta un clasificador nuevo con pocos datos. Esa idea, la **transferencia**, es el tema de la siguiente libreta. Y tiene una advertencia: en nuestro experimento los atributos se aprendieron con las etiquetas de las 60,000 imágenes de entrenamiento; la logística de encima usa solo 10,000, pero se beneficia de lo que la parte convolucional ya aprendió con las otras 50,000.
#
# ## 9. Arquitecturas: de LeNet a EfficientNet
#
# Nuestra red tiene dos capas convolucionales. Las redes que ganaron los concursos de visión de la última década tienen decenas o cientos, y cada una introdujo una idea que hoy es parte del vocabulario. Las recorremos en orden, programando y contando la pieza clave de cada una. Para contar sin descargar nada usamos `torchvision.models.<arquitectura>(weights=None)`, que construye la red con pesos al azar: el número de parámetros solo depende de la arquitectura.
#
# ### LeNet y AlexNet
#
# Y. LeCun y sus colegas usaron redes convolucionales para leer dígitos de cheques y códigos postales desde finales de los años ochenta; su LeNet-5 (LeCun, Bottou, Bengio y Haffner, *Proceedings of the IEEE*, 1998) tenía dos capas convolucionales, cada una seguida de una reducción de tamaño, y unos 60,000 parámetros: es, en esencia, la red que acabamos de entrenar. En 2012, AlexNet (A. Krizhevsky, I. Sutskever y G. Hinton, NeurIPS 2012), con cinco capas convolucionales, unos 60 millones de parámetros, ReLU, *dropout* y dos GPU, ganó el concurso ImageNet (1.2 millones de fotos de entrenamiento, 1,000 clases) con un error *top-5* de 15.3 %, contra 26.2 % del segundo lugar. Eso desató la carrera por redes más profundas.
#
# ### VGG: solo filtros de 3 × 3
#
# K. Simonyan y A. Zisserman (Oxford, *Visual Geometry Group*, de ahí el nombre; ICLR 2015) hicieron la pregunta más sencilla: ¿qué pasa si solo aumentamos la profundidad, con una regla fija? Su regla: **todas** las convoluciones son de 3 × 3 con relleno 1, y cada cierto número de ellas un *max-pooling* reduce el mapa a la mitad y se duplican los canales (64, 128, 256, 512).
#
# ¿Por qué 3 × 3? Porque dos convoluciones de 3 × 3 seguidas **ven lo mismo** que una de 5 × 5, y tres ven lo mismo que una de 7 × 7. Lo verificamos con la recurrencia del campo receptivo de la sección 5:

# %%
print("dos 3×3 seguidas:", campo_receptivo([(3, 1), (3, 1)]), "   tres 3×3:", campo_receptivo([(3, 1)] * 3))

C_VGG = 64
dos_3x3 = nn.Sequential(nn.Conv2d(C_VGG, C_VGG, 3, padding=1, bias=False), nn.ReLU(),
                        nn.Conv2d(C_VGG, C_VGG, 3, padding=1, bias=False))
una_5x5 = nn.Conv2d(C_VGG, C_VGG, 5, padding=2, bias=False)
vgg_cuenta = {"dos 3×3": contar(dos_3x3), "18 C²": 18 * C_VGG ** 2, "una 5×5": contar(una_5x5), "25 C²": 25 * C_VGG ** 2}
print(f"con C = {C_VGG} canales de entrada y de salida (sin sesgos):", vgg_cuenta)
print(f"la de 5×5 tiene {vgg_cuenta['una 5×5'] / vgg_cuenta['dos 3×3']:.2f} veces más parámetros")

# %% [markdown]
# Con $C$ canales de entrada y de salida, cada filtro de $k \times k$ tiene $C k^2$ pesos y hay $C$ filtros: $C^2 k^2$ pesos por capa. Dos capas de 3 × 3 suman $2 \cdot 9\, C^2 = 18\, C^2$; una de 5 × 5, $25\, C^2$. Con $C = 64$, son 73,728 contra 102,400: el mismo campo receptivo con 28 % menos parámetros y, además, una ReLU de más en medio, que hace a la función más expresiva.
#
# Ahora la red completa. VGG-16 tiene 13 capas convolucionales y 3 densas (16 capas con pesos):

# %%
def multiplicaciones(modelo, forma):
    """Multiplicaciones de las capas convolucionales y lineales al procesar una imagen de la forma dada."""
    total = 0

    def gancho(m, entrada, salida):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += salida[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        else:
            total += m.in_features * m.out_features

    ganchos = [m.register_forward_hook(gancho) for m in modelo.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    modelo.eval()
    with torch.no_grad():
        modelo(torch.zeros(1, *forma))
    for g in ganchos:
        g.remove()
    return total


vgg16 = torchvision.models.vgg16(weights=None)
vgg = {"parámetros": contar(vgg16), "en las capas densas": contar(vgg16.classifier),
       "primera densa (25,088 → 4,096)": contar(vgg16.classifier[0])}
vgg["fracción en las densas"] = vgg["en las capas densas"] / vgg["parámetros"]
mult_vgg_conv = multiplicaciones(vgg16.features, (3, 224, 224))
mult_vgg = multiplicaciones(vgg16, (3, 224, 224))
vgg["fracción de las multiplicaciones en las convoluciones"] = mult_vgg_conv / mult_vgg
print(vgg)
print(f"multiplicaciones por imagen de 224×224: {mult_vgg / 1e9:.1f} mil millones")

# %% [markdown]
# VGG-16 tiene 138.4 millones de parámetros, y el **89 %** de ellos está en las tres capas densas del final; solo la primera, que recibe los $512 \times 7 \times 7 = 25{,}088$ atributos de la parte convolucional, tiene 102.8 millones. Pero el trabajo de cómputo está al revés: el 99 % de las 15.5 mil millones de multiplicaciones por imagen ocurre en las convoluciones, que aplican sus pocos pesos en todas las posiciones.
#
# VGG no ganó ImageNet 2014 (quedó segunda en clasificación), pero se convirtió en la red **de uso general**: su diseño regular la hacía fácil de entender y de reutilizar, y sus pesos se publicaron. Su parte convolucional se usó como preprocesamiento para muchos otros problemas, exactamente en el sentido de la sección 8:
#
# - A. S. Razavian, H. Azizpour, J. Sullivan y S. Carlsson («CNN features off-the-shelf: an astounding baseline for recognition», CVPR Workshops 2014) mostraron que los atributos de una red entrenada en ImageNet, con una máquina de vectores de soporte lineal encima, competían con los mejores métodos diseñados a mano en problemas muy distintos (escenas, aves, flores, recuperación de imágenes).
# - L. A. Gatys, A. S. Ecker y M. Bethge («A neural algorithm of artistic style», 2015) usaron los mapas de activación de VGG para separar el «contenido» de una foto de su «estilo» y pintar una foto con el estilo de un cuadro.
# - S. Ren, K. He, R. Girshick y J. Sun (Faster R-CNN, NeurIPS 2015) construyeron su detector de objetos sobre la parte convolucional de VGG-16.
#
# ### ResNet: redes muy profundas con conexiones residuales
#
# **El problema de la degradación.** Después de VGG la pregunta obvia era: ¿y si apilamos más capas? K. He, X. Zhang, S. Ren y J. Sun (CVPR 2016) observaron que, al agregar capas a una red convolucional «simple», el error **de entrenamiento** empeoraba: una red de 56 capas se ajustaba peor que una de 20 en CIFAR-10. No era sobreajuste (es error de entrenamiento) y en principio la red profunda puede imitar a la menos profunda (bastaría con que las capas extra fueran la identidad), pero el descenso por gradiente no encontraba esa solución.
#
# **La solución.** En lugar de pedirle a un bloque de capas que aprenda una transformación $H(\mathbf{x})$, se le pide que aprenda la **corrección** $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$, y se suma la entrada: $\mathbf{x}_{l+1} = \mathbf{x}_l + F_l(\mathbf{x}_l)$. Si lo mejor es no hacer nada, basta con $F = 0$, que es fácil de alcanzar. Y el gradiente tiene un camino directo. Si $E$ es el error de la red y $L$ el número de bloques,
#
# $$\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} = I + \frac{\partial F_l}{\partial \mathbf{x}_l}
# \qquad\Longrightarrow\qquad
# \frac{\partial E}{\partial \mathbf{x}_l} = \frac{\partial E}{\partial \mathbf{x}_L}\prod_{m=l}^{L-1}\Big(I + \frac{\partial F_m}{\partial \mathbf{x}_m}\Big) .$$
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
# Con la inicialización por omisión de PyTorch (pesos uniformes con varianza $1/(3\,n_{\text{ent}})$), cada capa simple multiplica la varianza de la señal por cerca de $n_{\text{ent}} \cdot \frac{1}{3 n_{\text{ent}}} \cdot \frac12 = \frac16$, y el gradiente que llega a la primera capa es unos 14 órdenes de magnitud menor que el de la última: esas capas no aprenden nada. La inicialización de He (Unidad 6, ejemplo 1) arregla la escala **al inicio**, pero no garantiza nada durante el entrenamiento, y es justo ahí donde He y sus colegas observaron la degradación. La red residual, con la misma inicialización mala, entrega gradientes del mismo orden a todas sus capas.
#
# **Los bloques de una ResNet.** En una ResNet real, $F$ no es una capa lineal sino un par de convoluciones con **normalización por lotes** (*batch normalization*, que reescala cada canal con la media y la varianza del minilote). Hay dos versiones:
#
# - el **bloque básico** (ResNet-18 y ResNet-34): dos convoluciones de 3 × 3 con $C$ canales, y la suma con la entrada;
# - el **bloque *bottleneck*** (cuello de botella; ResNet-50, 101 y 152): una convolución de **1 × 1** que reduce los canales (de 256 a 64, por ejemplo), una de 3 × 3 que trabaja con pocos canales y otra de 1 × 1 que los vuelve a ampliar.
#
# Una convolución de 1 × 1 no mira a los vecinos: en cada pixel mezcla los $C$ canales con una capa lineal de $C$ a $C'$ números, la misma en todas las posiciones. Sirve para cambiar el número de canales barato. Programemos los dos bloques:

# %%
class BloqueBasico(nn.Module):
    """x + F(x), con F = conv 3×3 → BN → ReLU → conv 3×3 → BN."""

    def __init__(self, C):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C)
        self.conv2, self.bn2 = nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C)

    def forward(self, x):
        F_x = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(x + F_x)


class BloqueCuello(nn.Module):
    """x + F(x), con F = conv 1×1 (C → C_m) → conv 3×3 (C_m → C_m) → conv 1×1 (C_m → C), con BN y ReLU."""

    def __init__(self, C, C_m):
        super().__init__()
        self.reducir, self.bn1 = nn.Conv2d(C, C_m, 1, bias=False), nn.BatchNorm2d(C_m)
        self.conv, self.bn2 = nn.Conv2d(C_m, C_m, 3, padding=1, bias=False), nn.BatchNorm2d(C_m)
        self.ampliar, self.bn3 = nn.Conv2d(C_m, C, 1, bias=False), nn.BatchNorm2d(C)

    def forward(self, x):
        h = F.relu(self.bn1(self.reducir(x)))
        h = F.relu(self.bn2(self.conv(h)))
        return F.relu(x + self.bn3(self.ampliar(h)))


def pesos_conv(modelo):
    """Pesos de las capas convolucionales (sin contar la normalización por lotes)."""
    return sum(m.weight.numel() for m in modelo.modules() if isinstance(m, nn.Conv2d))


x = torch.randn(2, 256, 56, 56)
basico, cuello = BloqueBasico(256), BloqueCuello(256, 64)
print("formas:", tuple(basico(x).shape), tuple(cuello(x).shape))
bloques_resnet = pd.DataFrame({
    "pesos de las convoluciones": [pesos_conv(basico), pesos_conv(cuello)],
    "multiplicaciones (mapa de 56×56)": [multiplicaciones(basico, (256, 56, 56)), multiplicaciones(cuello, (256, 56, 56))],
}, index=["básico: dos 3×3 de 256 canales", "bottleneck: 1×1 (256→64), 3×3 (64), 1×1 (64→256)"])
bloques_resnet

# %% [markdown]
# Con 256 canales, dos convoluciones de 3 × 3 tienen $2 \cdot 9 \cdot 256^2 = 1{,}179{,}648$ pesos; el bloque *bottleneck* tiene $256 \cdot 64 + 9 \cdot 64^2 + 64 \cdot 256 = 69{,}632$, unas **17 veces menos**, con la misma entrada y la misma salida. Así se pudieron apilar 50, 101 y hasta 152 capas sin que el costo se disparara. Con conexiones residuales, normalización por lotes y bloques *bottleneck*, la ResNet-152 ganó ImageNet 2015 con 3.57 % de error *top-5* (un ensamble de varias redes).
#
# ### Inception: varias ventanas a la vez
#
# ¿De qué tamaño debe ser el filtro, 1 × 1, 3 × 3 o 5 × 5? C. Szegedy y sus colegas de Google (GoogLeNet, ganadora de ImageNet 2014, CVPR 2015) contestaron: **de todos**. Un **módulo Inception** aplica en paralelo, a la misma entrada, convoluciones de 1 × 1, 3 × 3 y 5 × 5 y un *max-pooling* de 3 × 3, y concatena sus salidas en la dimensión de los canales. La red aprende cuál rama le sirve en cada lugar.
#
# El problema es el costo: una convolución de 5 × 5 sobre 192 canales es cara. La solución es la misma convolución de 1 × 1 de los bloques *bottleneck* (de hecho, Inception la usó primero): **reducir** los canales antes de las ramas caras. Programamos el primer módulo de GoogLeNet («3a»), que recibe un mapa de 28 × 28 con 192 canales y entrega 64 + 128 + 32 + 32 = 256 canales; las ramas de 3 × 3 y de 5 × 5 se reducen antes a 96 y 16 canales.

# %%
class ModuloInception(nn.Module):
    def __init__(self, C_ent, c1, r3, c3, r5, c5, cp, reducir=True):
        super().__init__()
        conv = lambda a, b, k: nn.Conv2d(a, b, k, padding=k // 2, bias=False)
        self.rama1 = conv(C_ent, c1, 1)
        if reducir:
            self.rama3 = nn.Sequential(conv(C_ent, r3, 1), nn.ReLU(), conv(r3, c3, 3))
            self.rama5 = nn.Sequential(conv(C_ent, r5, 1), nn.ReLU(), conv(r5, c5, 5))
        else:
            self.rama3, self.rama5 = conv(C_ent, c3, 3), conv(C_ent, c5, 5)
        self.rama_pool = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), conv(C_ent, cp, 1))

    def forward(self, x):
        ramas = (self.rama1, self.rama3, self.rama5, self.rama_pool)
        return torch.cat([F.relu(r(x)) for r in ramas], dim=1)


inc_con = ModuloInception(192, 64, 96, 128, 16, 32, 32, reducir=True)
inc_sin = ModuloInception(192, 64, 96, 128, 16, 32, 32, reducir=False)
print("forma de la salida:", tuple(inc_con(torch.zeros(1, 192, 28, 28)).shape))
inception_cuenta = pd.DataFrame({
    "parámetros": [contar(inc_sin), contar(inc_con)],
    "multiplicaciones (28×28)": [multiplicaciones(inc_sin, (192, 28, 28)), multiplicaciones(inc_con, (192, 28, 28))],
}, index=["sin reducción 1×1", "con reducción 1×1 (GoogLeNet 3a)"])
inception_cuenta

# %% [markdown]
# Sin reducción, la rama de 3 × 3 tiene $192 \cdot 128 \cdot 9 = 221{,}184$ pesos y la de 5 × 5, $192 \cdot 32 \cdot 25 = 153{,}600$. Con la reducción, $192 \cdot 96 + 96 \cdot 128 \cdot 9 = 129{,}024$ y $192 \cdot 16 + 16 \cdot 32 \cdot 25 = 15{,}872$: la rama de 5 × 5 se vuelve casi diez veces más barata. El módulo completo pasa de 393,216 a 163,328 parámetros (2.4 veces menos), y de 308 a 128 millones de multiplicaciones por imagen. GoogLeNet completa, con nueve módulos así, tiene unos 6.6 millones de parámetros, veinte veces menos que VGG-16, y ganó ImageNet 2014, con VGG en segundo lugar.
#
# ### Xception: la convolución separable en profundidad
#
# Una convolución normal hace dos cosas a la vez: mira a los **vecinos** (la ventana de $k \times k$) y **mezcla los canales** (cada filtro abarca los $C$ canales de entrada). F. Chollet (Xception, «*Extreme Inception*», CVPR 2017) llevó la idea de Inception al extremo y las separó por completo:
#
# 1. **en profundidad** (*depthwise*): un filtro de $k \times k$ **por canal**, que solo mira a los vecinos dentro de su canal: $C k^2$ pesos;
# 2. **puntual** (*pointwise*): una convolución de 1 × 1 que mezcla los canales: $C C'$ pesos.
#
# En PyTorch, el paso 1 es una `nn.Conv2d` con `groups=C`: los canales se reparten en $C$ grupos de uno, y cada filtro ve solo su grupo.

# %%
def separable(C, C2, k):
    """Convolución separable en profundidad: k×k por canal (groups=C) seguida de 1×1 de C a C2 canales."""
    return nn.Sequential(nn.Conv2d(C, C, k, padding=k // 2, groups=C, bias=False),
                         nn.Conv2d(C, C2, 1, bias=False))


C_SEP, K_SEP = 256, 3
normal, sep = nn.Conv2d(C_SEP, C_SEP, K_SEP, padding=1, bias=False), separable(C_SEP, C_SEP, K_SEP)
x = torch.randn(1, C_SEP, 28, 28)
print("formas:", tuple(normal(x).shape), tuple(sep(x).shape))

filas_sep = []
for C2 in (64, 256):
    for k in (3, 5):
        p_normal, p_sep = contar(nn.Conv2d(C_SEP, C2, k, bias=False)), contar(separable(C_SEP, C2, k))
        filas_sep.append({"C": C_SEP, "C'": C2, "k": k, "normal: C C' k²": p_normal, "separable: C k² + C C'": p_sep,
                          "razón": p_sep / p_normal, "1/C' + 1/k²": 1 / C2 + 1 / k ** 2})
separable_cuenta = pd.DataFrame(filas_sep)
mult_sep = {"normal": multiplicaciones(normal, (C_SEP, 28, 28)), "separable": multiplicaciones(sep, (C_SEP, 28, 28))}
print(f"multiplicaciones con C = C' = {C_SEP}, k = 3, mapa de 28×28:", mult_sep,
      f"razón {mult_sep['separable'] / mult_sep['normal']:.3f}")
separable_cuenta.round(4)

# %% [markdown]
# La razón entre los dos conteos es
#
# $$\frac{C k^2 + C C'}{C C' k^2} = \frac{1}{C'} + \frac{1}{k^2} ,$$
#
# y con muchos canales de salida queda cerca de $1/k^2$: con filtros de 3 × 3, la separable usa unas **9 veces menos** parámetros (0.115 con $C = C' = 256$), y con 5 × 5, unas 24 veces menos. Las multiplicaciones bajan en la misma proporción, porque cada peso se aplica una vez por pixel del mapa: con $C = C' = 256$ en un mapa de 28 × 28, de 462 a 53 millones. El precio es que la separable es una función más restringida (no puede mezclar a la vez vecinos y canales), pero Chollet mostró que, apilando muchas, la red resultante (Xception, 22.9 millones de parámetros) era más exacta en ImageNet que Inception V3 con un número parecido de parámetros. La misma convolución es la base de MobileNet (A. G. Howard et al., 2017), pensada para teléfonos, y de EfficientNet.
#
# ### EfficientNet: crecer en las tres direcciones a la vez
#
# Para hacer más exacta una red hay tres perillas: más **profundidad** (más capas), más **ancho** (más canales por capa) o más **resolución** (imágenes de entrada más grandes). M. Tan y Q. V. Le (EfficientNet, ICML 2019) observaron que girar una sola se satura pronto, y propusieron el **escalamiento compuesto**: un solo coeficiente $\phi$ las mueve todas,
#
# $$\text{profundidad} \propto \alpha^\phi, \qquad \text{ancho} \propto \beta^\phi, \qquad \text{resolución} \propto \gamma^\phi, \qquad \text{con } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 .$$
#
# La restricción viene de contar multiplicaciones: el costo de una convolución crece en proporción a la profundidad, al **cuadrado** del ancho (canales de entrada por canales de salida) y al **cuadrado** de la resolución (filas por columnas). Así, cada unidad de $\phi$ duplica, aproximadamente, el costo. Con una búsqueda en una red pequeña (EfficientNet-B0, hecha de convoluciones separables en profundidad), los autores eligieron $\alpha = 1.2$, $\beta = 1.1$ y $\gamma = 1.15$:

# %%
ALFA, BETA, GAMMA = 1.2, 1.1, 1.15
print(f"α·β²·γ² = {ALFA * BETA ** 2 * GAMMA ** 2:.3f}")
escalamiento = pd.DataFrame({
    "profundidad (×)": [ALFA ** f for f in range(5)],
    "ancho (×)": [BETA ** f for f in range(5)],
    "resolución (pixeles)": [round(224 * GAMMA ** f) for f in range(5)],
    "costo (×)": [(ALFA * BETA ** 2 * GAMMA ** 2) ** f for f in range(5)],
}, index=pd.Index(range(5), name="φ"))
escalamiento.round(2)

# %% [markdown]
# Con $\phi = 3$, la red tiene 1.7 veces más capas, 1.33 veces más canales y ve imágenes de 341 pixeles en lugar de 224, por unas 7 veces el costo. (Los modelos B1 a B7 publicados redondean estos factores a resoluciones y números de capas concretos.)
#
# ### La historia en una tabla
#
# Juntemos todo. Contamos los parámetros de cada arquitectura con `torchvision` (sin descargar pesos) y anotamos la exactitud *top-1* en el conjunto de validación de ImageNet (la fracción de imágenes en las que la primera respuesta es la correcta) que documenta `torchvision` para sus pesos `IMAGENET1K_V1`. Esa cifra se lee de los metadatos de los pesos, sin descargarlos. Son entrenamientos de `torchvision`, no siempre idénticos a los de los artículos originales. Xception no está en `torchvision`; para ella usamos las cifras del artículo de Chollet (22.9 millones de parámetros y 79.0 % de exactitud *top-1*). Inception V3 (Szegedy et al., CVPR 2016) es la tercera versión de GoogLeNet; la contamos sin su clasificador auxiliar, que solo se usa al entrenar.

# %%
M = torchvision.models
redes = {   # nombre: (año, constructor sin pesos, pesos documentados)
    "AlexNet": (2012, lambda: M.alexnet(weights=None), M.AlexNet_Weights.IMAGENET1K_V1),
    "VGG-16": (2014, lambda: M.vgg16(weights=None), M.VGG16_Weights.IMAGENET1K_V1),
    "GoogLeNet": (2014, lambda: M.googlenet(weights=None, init_weights=False, aux_logits=False), M.GoogLeNet_Weights.IMAGENET1K_V1),
    "ResNet-18": (2015, lambda: M.resnet18(weights=None), M.ResNet18_Weights.IMAGENET1K_V1),
    "ResNet-50": (2015, lambda: M.resnet50(weights=None), M.ResNet50_Weights.IMAGENET1K_V1),
    "Inception V3": (2015, lambda: M.inception_v3(weights=None, init_weights=False, aux_logits=False), M.Inception_V3_Weights.IMAGENET1K_V1),
    "EfficientNet-B0": (2019, lambda: M.efficientnet_b0(weights=None), M.EfficientNet_B0_Weights.IMAGENET1K_V1),
}
filas_arq = []
for nombre, (anio, construir, pesos) in redes.items():
    filas_arq.append({"arquitectura": nombre, "año": anio, "parámetros (millones)": contar(construir()) / 1e6,
                      "exactitud top-1 (%)": pesos.meta["_metrics"]["ImageNet-1K"]["acc@1"], "fuente de la exactitud": "torchvision"})
filas_arq.append({"arquitectura": "Xception", "año": 2017, "parámetros (millones)": 22.9,
                  "exactitud top-1 (%)": 79.0, "fuente de la exactitud": "Chollet (2017)"})
arquitecturas = pd.DataFrame(filas_arq).sort_values("año", kind="stable").set_index("arquitectura")
arquitecturas.round(2)

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for nombre, f in arquitecturas.iterrows():
    color = DORADO_OSC if nombre == "EfficientNet-B0" else AZUL
    ax.scatter(f["parámetros (millones)"], f["exactitud top-1 (%)"], s=60, color=color, zorder=3)
    ax.annotate(f"{nombre} ({f['año']})", (f["parámetros (millones)"], f["exactitud top-1 (%)"]),
                xytext=(7, -4), textcoords="offset points", fontsize=11, color=TINTA)
ax.set_xscale("log")
ax.set_xlim(3, 400)
ax.set_ylim(54, 82)
ax.set_xlabel("millones de parámetros (escala logarítmica)")
ax.set_ylabel("exactitud top-1 en ImageNet (%)")
ax.set_title("Más exactitud no requirió más parámetros")
plt.show()

# %% [markdown]
# La gráfica resume la década: de AlexNet a VGG-16 la exactitud subió a costa de más del doble de parámetros; desde ahí, cada idea nueva (las conexiones residuales, la reducción con 1 × 1, las convoluciones separables, el escalamiento compuesto) dio **más exactitud con menos parámetros**. EfficientNet-B0, con 5.3 millones, es más exacta que VGG-16, que tiene 26 veces más. (Un detalle: en la tabla, GoogLeNet queda ligeramente por debajo de VGG-16, aunque ganó el concurso de 2014. Las cifras de `torchvision` son de una sola red evaluada con un solo recorte de cada imagen; en el concurso, ambos equipos usaron ensambles de varias redes y varios recortes.)
#
# ## 10. Cierre
#
# - Las imágenes, y en general los datos en los que lo local importa (audio, series, secuencias de ADN), tienen una estructura que una capa densa ignora.
# - Una convolución es una capa lineal con dos restricciones: **conexiones locales** y **pesos compartidos**. El resultado es equivariante a traslaciones y tiene muchísimos menos parámetros que una capa densa.
# - El tamaño de la salida es $\lfloor (n + 2p - k)/s \rfloor + 1$; los parámetros, $(C_{\text{ent}}k^2 + 1)C_{\text{sal}}$; el campo receptivo crece con la profundidad y con el *pooling*.
# - Con once veces menos parámetros, una red convolucional pequeña le gana a un perceptrón multicapa en Fashion-MNIST: el sesgo inductivo correcto vale más que la capacidad.
# - La parte convolucional es un **preprocesamiento aprendido**: con ella congelada, una regresión logística encima alcanza la exactitud de la red completa.
# - Las arquitecturas se volvieron más profundas y más eficientes: filtros de 3 × 3 apilados (VGG), conexiones residuales y bloques *bottleneck* (ResNet), ramas en paralelo con reducción 1 × 1 (Inception), convoluciones separables (Xception) y escalamiento compuesto (EfficientNet).
#
# *Referencias:* Y. LeCun, L. Bottou, Y. Bengio y P. Haffner, «Gradient-based learning applied to document recognition», *Proc. IEEE* 86 (1998). A. Krizhevsky, I. Sutskever y G. E. Hinton, «ImageNet classification with deep convolutional neural networks», NeurIPS (2012). K. Simonyan y A. Zisserman, «Very deep convolutional networks for large-scale image recognition», ICLR (2015). A. S. Razavian et al., «CNN features off-the-shelf: an astounding baseline for recognition», CVPR Workshops (2014). L. A. Gatys, A. S. Ecker y M. Bethge, «A neural algorithm of artistic style», arXiv:1508.06576 (2015). S. Ren, K. He, R. Girshick y J. Sun, «Faster R-CNN», NeurIPS (2015). K. He, X. Zhang, S. Ren y J. Sun, «Deep residual learning for image recognition», CVPR (2016). C. Szegedy et al., «Going deeper with convolutions», CVPR (2015), y «Rethinking the Inception architecture for computer vision», CVPR (2016). F. Chollet, «Xception: deep learning with depthwise separable convolutions», CVPR (2017). A. G. Howard et al., «MobileNets», arXiv:1704.04861 (2017). M. Tan y Q. V. Le, «EfficientNet: rethinking model scaling for convolutional neural networks», ICML (2019). H. Xiao, K. Rasul y R. Vollgraf, «Fashion-MNIST», arXiv:1708.07747 (2017). I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press (2016), capítulo 9. Exactitudes de `torchvision`: <https://pytorch.org/vision/stable/models.html>.

# %% [markdown]
# ## 11. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p09", "u06-p10", "u06-p11"])

# %% [markdown]
# ## 12. Ejercicios
#
# 1. **Contar a mano.** Una capa convolucional recibe un mapa de $32 \times 32 \times 64$ y aplica 128 filtros de $5 \times 5$ con relleno 2 y paso 2. ¿De qué tamaño es la salida y cuántos parámetros tiene? ¿Cuántos tendría una capa densa que produjera una salida del mismo tamaño? Comprueba con `nn.Conv2d` y `nn.Linear`.
# 2. **Equivariancia y *pooling*.** (a) Demuestra que la composición de dos convoluciones es equivariante a traslaciones y que también lo es una ReLU aplicada pixel por pixel. (b) Da un ejemplo de una imagen de una dimensión y un desplazamiento de un pixel para los que el *max-pooling* de tamaño 2 y paso 2 **no** es invariante. (c) ¿Para qué desplazamientos sí lo es siempre?
# 3. **Campo receptivo.** Usa la recurrencia de la sección 5 para calcular el campo receptivo de tres capas convolucionales de $3 \times 3$ seguidas, sin *pooling*, y compáralo con el de una sola capa de $7 \times 7$. Compara también el número de parámetros con $C$ canales de entrada y salida en todas.
# 4. **Otro clasificador encima.** En la sección 8, sustituye la regresión logística por un bosque aleatorio o una máquina de vectores de soporte sobre los mismos atributos congelados. ¿Cambia mucho la exactitud? ¿Y si usas los atributos de la **primera** capa convolucional (después de su *pooling*: $16 \times 14 \times 14$ números) en lugar de los de la segunda?
# 5. **Gradiente en una red residual.** Para una red escalar $x_{l+1} = x_l + a\,x_l$ con $L$ bloques, calcula $\partial x_L/\partial x_0$ y compárala con la red simple $x_{l+1} = a\,x_l$ cuando $|a| < 1$. ¿Qué pasa en la red residual si $a$ es pequeño y $L$ grande? ¿Y si $a = -1$?
# 6. **Una red separable.** Reemplaza `conv2` de `RedConv` por una convolución separable en profundidad (`separable(16, 32, 3)`), entrena con el mismo procedimiento y compara parámetros y exactitud con la red original.
