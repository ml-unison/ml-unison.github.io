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
# # Transferencia: lo que una red de ImageNet sabe de imágenes de satélite
#
# **Unidad 6 · Ejemplo 4**
#
# Tienes 200 fotos etiquetadas de tu problema (parcelas, radiografías, piezas defectuosas) y una red convolucional con 11 millones de parámetros. Entrenarla desde cero con tan pocos datos es una receta para el sobreajuste. Pero hay redes que ya se entrenaron con más de un millón de fotos de ImageNet, y lo que aprendieron sus capas no es exclusivo de perros y autos. En esta libreta vamos a:
#
# 1. mirar **qué aprende cada capa** de una VGG16 entrenada en ImageNet, con los dos métodos clásicos: los parches de imágenes que más activan a un canal y la imagen sintética que lo activa al máximo;
# 2. discutir por qué eso permite **reutilizar** la red en otros problemas;
# 3. ver **dónde se consiguen modelos preentrenados** y qué revisar antes de usarlos;
# 4. comparar en imágenes de satélite (EuroSAT), con 10, 50 y 200 imágenes etiquetadas por clase, **tres formas de transferir** una ResNet-18 de ImageNet contra entrenar una red desde cero;
# 5. recordar que un modelo preentrenado también **hereda los huecos** de sus datos.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado. PyTorch y torchvision ya vienen instalados en Colab. La libreta descarga unos 750 MB la primera vez (los pesos de VGG16 y ResNet-18, Imagenette y EuroSAT). Si tienes una GPU (en Colab: *Entorno de ejecución → Cambiar tipo de entorno*), la libreta la usa; si no, corre en CPU en unos 10 a 15 minutos (el ajuste fino de la sección 4 es lo más lento).

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
TINTA, TENUE, MARINO = rdp.estilo.TINTA, rdp.estilo.TENUE, rdp.estilo.MARINO

# la verificación automática del repositorio usa menos datos y menos pasos
RAPIDO = os.environ.get("RDP_RAPIDO") == "1"
RAIZ_DATOS = os.path.expanduser("~/.cache/rdp-datos")
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEMILLA)
print("dispositivo:", dispositivo)

# media y desviación estándar por canal de ImageNet: lo que esperan las redes de torchvision
MEDIA_IN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DE_IN = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def contar(modelo):
    return sum(p.numel() for p in modelo.parameters())

# %% [markdown]
# ## 1. Qué aprende cada capa
#
# Una red convolucional es una pila de capas; cada capa produce varios **mapas de activación**, uno por **canal** (uno por filtro). ¿Qué detecta un canal en particular? Las primeras capas de una red pequeña se pueden mirar directamente (sus filtros son de 3 × 3 pixeles), pero los filtros de una capa profunda combinan cientos de canales de la capa anterior y no se pueden «ver». M. Zeiler y R. Fergus («Visualizing and understanding convolutional networks», ECCV 2014) propusieron una idea sencilla: **mostrar los pedazos de imágenes reales que más activan a cada canal**. Con eso vieron, por primera vez con claridad, que las capas forman una jerarquía: bordes y colores, luego texturas, luego partes de objetos, luego objetos.
#
# Vamos a reproducir ese experimento con nuestras propias imágenes y una red real.
#
# ### Los datos: Imagenette
#
# **Datos.** Imagenette (J. Howard, fast.ai, 2019) es un subconjunto de ImageNet con 10 clases fáciles de distinguir: tenca (un pez), springer spaniel inglés (un perro), casetera, motosierra, iglesia, corno francés, camión de basura, bomba de gasolina, pelota de golf y paracaídas. Usamos la versión de 160 pixeles en el lado corto y su partición de validación (3,925 fotos). Licencia Apache 2.0; `torchvision` la descarga la primera vez (unos 95 MB). Como son fotos de ImageNet, son el tipo de imagen con el que se entrenó la red: justo lo que queremos para ver qué aprendió.

# %%
CLASES_INET = ["tenca", "springer spaniel", "casetera", "motosierra", "iglesia",
               "corno francés", "camión de basura", "bomba de gasolina", "pelota de golf", "paracaídas"]
LADO = 160
imagenette = torchvision.datasets.Imagenette(
    RAIZ_DATOS, split="val", size="160px", download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(LADO), torchvision.transforms.CenterCrop(LADO),
                                              torchvision.transforms.ToTensor()]))
print(f"{len(imagenette):,} fotos de validación")

N_IMAGENES = 60 if RAPIDO else 400
ids_inet = np.random.default_rng(SEMILLA).permutation(len(imagenette))[:N_IMAGENES]
fotos = torch.stack([imagenette[int(i)][0] for i in ids_inet])          # (N, 3, 160, 160), valores en [0, 1]
etiquetas_inet = np.array([imagenette[int(i)][1] for i in ids_inet])
print(fotos.shape, "· fotos por clase:", np.bincount(etiquetas_inet, minlength=10).tolist())

fig, ejes = plt.subplots(1, 10, figsize=(13, 1.9))
for c, ax in enumerate(ejes):
    ax.imshow(fotos[np.where(etiquetas_inet == c)[0][0]].permute(1, 2, 0))
    ax.set_title(CLASES_INET[c], fontsize=9.5, loc="center")
    ax.axis("off")
plt.show()

# %% [markdown]
# ### La red: VGG16
#
# VGG16 (K. Simonyan y A. Zisserman, «Very deep convolutional networks for large-scale image recognition», ICLR 2015) es una pila de 13 convoluciones de 3 × 3 con ReLU, agrupadas en cinco bloques separados por *max-pooling* de 2 × 2, más tres capas densas al final. Por su estructura tan regular fue la primera red que se usó masivamente **para otras cosas**: como extractor de atributos, en detección de objetos o en transferencia de estilo. Nombramos sus capas como en el artículo: `conv3_3` es la tercera convolución del tercer bloque. Solo necesitamos la parte convolucional (`vgg.features`).
#
# Cada pixel de un mapa de activación depende solo de una ventana de la imagen de entrada, su **campo receptivo**. Lo calculamos con la recurrencia de siempre: una convolución de $k \times k$ agrega $(k - 1)\cdot s$ pixeles al campo, donde el **salto** $s$ es la distancia en la imagen entre dos pixeles vecinos del mapa; cada *pooling* de paso 2 duplica el salto.

# %%
vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.eval().to(dispositivo)
for p in vgg.parameters():
    p.requires_grad_(False)
print(f"VGG16, parte convolucional: {contar(vgg):,} parámetros")

# índice en vgg.features de la ReLU que sigue a cada convolución que estudiamos
CAPAS = {"conv1_2": 3, "conv2_2": 8, "conv3_3": 15, "conv4_3": 22, "conv5_3": 29}

campo, salto = {}, {}
c_actual, s_actual = 1, 1
for i, m in enumerate(vgg):
    if isinstance(m, nn.Conv2d):
        c_actual += (m.kernel_size[0] - 1) * s_actual
    elif isinstance(m, nn.MaxPool2d):
        c_actual += (m.kernel_size - 1) * s_actual
        s_actual *= m.stride
    campo[i], salto[i] = c_actual, s_actual
pd.DataFrame({"canales": [vgg[i - 1].out_channels for i in CAPAS.values()],
              "campo receptivo (pixeles)": [campo[i] for i in CAPAS.values()],
              "salto": [salto[i] for i in CAPAS.values()]}, index=list(CAPAS))

# %% [markdown]
# Un canal de `conv1_2` mira una ventana de 5 × 5 pixeles; uno de `conv5_3`, en teoría, una de 196 × 196, más grande que nuestras fotos de 160. (En la práctica, el centro del campo pesa mucho más que la orilla: el **campo receptivo efectivo** es bastante menor; W. Luo et al., NeurIPS 2016.)
#
# ### Los parches que más activan a cada canal
#
# Pasamos las 400 fotos por la red y, para cada canal de cada capa, guardamos su activación máxima en cada foto y la posición donde ocurre. Para un canal, las fotos con mayor máximo son las que más lo «encienden»; recortamos de cada una la ventana del campo receptivo alrededor de esa posición. Para `conv4_3` y `conv5_3` limitamos la ventana a 112 pixeles: el campo receptivo completo sería casi toda la foto.

# %%
@torch.no_grad()
def maximos_por_canal(fotos, lote=25):
    """Para cada capa: activación máxima de cada canal en cada foto y la posición (fila, columna) del máximo."""
    maximos = {k: [] for k in CAPAS}
    posiciones = {k: [] for k in CAPAS}
    for i in range(0, len(fotos), lote):
        h = ((fotos[i:i + lote] - MEDIA_IN) / DE_IN).to(dispositivo)
        for j, capa in enumerate(vgg[:max(CAPAS.values()) + 1]):
            h = capa(h)
            for nombre, idx in CAPAS.items():
                if idx == j:
                    valor, donde = h.flatten(2).max(2)                  # (lote, canales)
                    ancho = h.shape[3]
                    maximos[nombre].append(valor.cpu())
                    posiciones[nombre].append(torch.stack([donde // ancho, donde % ancho], -1).cpu())
    return ({k: torch.cat(v) for k, v in maximos.items()}, {k: torch.cat(v) for k, v in posiciones.items()})


t0 = time.time()
maximos, posiciones = maximos_por_canal(fotos)
t_parches = time.time() - t0
print(f"{N_IMAGENES} fotos por la red en {t_parches:.0f} s;", {k: tuple(v.shape) for k, v in maximos.items()})

# %% [markdown]
# Elegimos cuatro canales por capa. Una red tiene cientos de canales por capa y no todos son igual de fáciles de interpretar: estos los escogimos mirando las figuras (es una lista fija, para que la libreta siempre muestre los mismos); cambia la lista y explora otros. Para cada canal mostramos los parches de las 9 fotos con mayor activación.

# %%
CANALES = {"conv1_2": [53, 14, 11, 4], "conv2_2": [38, 96, 119, 41], "conv3_3": [36, 223, 90, 139],
           "conv4_3": [456, 123, 330, 92], "conv5_3": [62, 184, 291, 103]}
TOP, LADO_MAX, PIX = 9, 112, 40       # parches por canal, ventana máxima, pixeles de cada parche en la figura


def parche(foto, nombre, fila, col):
    """Ventana del campo receptivo (a lo más LADO_MAX) centrada en la posición (fila, col) del mapa, dentro de la foto."""
    idx = CAPAS[nombre]
    lado = min(campo[idx], LADO_MAX)
    centro_f, centro_c = (fila + 0.5) * salto[idx], (col + 0.5) * salto[idx]
    f0 = int(np.clip(round(centro_f - lado / 2), 0, LADO - lado))
    c0 = int(np.clip(round(centro_c - lado / 2), 0, LADO - lado))
    return foto[:, f0:f0 + lado, c0:c0 + lado]


def mosaico_parches(nombre, canal):
    mejores = torch.argsort(maximos[nombre][:, canal], descending=True)[:TOP]
    ps = [parche(fotos[t], nombre, *posiciones[nombre][t, canal].tolist()) for t in mejores]
    modo = "nearest" if ps[0].shape[1] < PIX else "area"         # agrandar sin suavizar; reducir promediando
    ps = [F.interpolate(p[None], size=PIX, mode=modo)[0] for p in ps]
    return torchvision.utils.make_grid(torch.stack(ps), nrow=3, padding=2, pad_value=1.0)


mosaicos = {nombre: [mosaico_parches(nombre, c) for c in cs] for nombre, cs in CANALES.items()}

fig, ejes = plt.subplots(len(CANALES), 4, figsize=(10, 13))
for r, (nombre, ms) in enumerate(mosaicos.items()):
    for c, m in enumerate(ms):
        ejes[r, c].imshow(m.permute(1, 2, 0))
        ejes[r, c].set_xticks([]); ejes[r, c].set_yticks([])
        ejes[r, c].set_title(f"canal {CANALES[nombre][c]}", fontsize=10, loc="center")
    lado = min(campo[CAPAS[nombre]], LADO_MAX)
    ejes[r, 0].set_ylabel(f"{nombre}\n{lado} × {lado} px", fontsize=11, color=MARINO)
fig.suptitle("Los 9 parches que más activan a cada canal", x=0.01, ha="left", color=MARINO, fontweight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Lee la figura de arriba hacia abajo:
#
# - **`conv1_2`** (ventanas de 5 × 5): cada canal responde a algo elemental, un borde con cierta orientación o un contraste de colores. Los parches de un mismo canal no tienen nada que ver entre sí como objetos.
# - **`conv2_2`** (14 × 14): curvas, esquinas, puntos y franjas paralelas.
# - **`conv3_3`** (40 × 40): texturas y patrones que se repiten, como espirales y círculos concéntricos (el corno francés, las pelotas de golf, las llantas).
# - **`conv4_3`** (92 × 92): partes reconocibles, como las pantallas de las bombas de gasolina, las torres de las iglesias o las caras.
# - **`conv5_3`**: objetos completos, bastante independientes de la posición y la pose: cabezas de perro, pelotas de golf, paracaídas, fachadas de iglesias. No siempre son puros: el canal 291 mezcla paracaídas con personas.
#
# Nadie le dijo a la red que buscara bordes ni torres de iglesia: solo se le pidió clasificar fotos entre 1,000 clases. La jerarquía apareció sola, como consecuencia de apilar convoluciones (cada capa combina los detectores de la anterior en un campo receptivo más grande) y de entrenar todo junto.
#
# ### La imagen que más activa a cada canal
#
# Los parches dependen de las fotos que tengamos. La otra pregunta es: ¿**qué imagen** activaría más a un canal? Podemos buscarla con el mismo descenso por gradiente con el que entrenamos redes, pero al revés: dejamos **fijos los pesos** y movemos **los pixeles** para maximizar la activación promedio del canal (D. Erhan, Y. Bengio, A. Courville y P. Vincent, 2009). Si se optimizan los pixeles directamente, sale ruido de alta frecuencia que excita a la red pero que no nos dice nada. Dos trucos sencillos lo evitan (C. Olah, A. Mordvintsev y L. Schubert, «Feature visualization», *Distill*, 2017):
#
# - **parametrizar en frecuencia**: la imagen es la transformada inversa de Fourier de unos coeficientes que optimizamos, con las frecuencias altas atenuadas (divididas entre la frecuencia), y los colores pasan por una matriz que los correlaciona como en las fotos naturales;
# - **mover la imagen al azar** en cada paso (unos pixeles de traslación y un poco de cambio de escala): solo sobreviven los patrones que activan al canal en cualquier posición.
#
# Maximizamos la activación **antes** de la ReLU: después de ella, un canal apagado tiene gradiente cero y la optimización nunca arranca.

# %%
# matriz de correlación de colores de las fotos naturales (la de Olah et al.), normalizada
CORR_COLOR = torch.tensor([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]])
CORR_COLOR = CORR_COLOR / CORR_COLOR.norm(dim=0).max()


def imagen_maxima(nombre, canales, pasos, tam, lr=0.05):
    """Imágenes de tam × tam que maximizan la activación media de cada canal (una por canal, todas a la vez)."""
    g = torch.Generator().manual_seed(SEMILLA)
    n = len(canales)
    fy, fx = torch.fft.fftfreq(tam)[:, None], torch.fft.rfftfreq(tam)[None, :]
    atenuacion = 1.0 / torch.maximum(torch.sqrt(fx**2 + fy**2), torch.tensor(1.0 / tam))
    coef = (0.01 * torch.randn(n, 3, tam, tam // 2 + 1, 2, generator=g)).to(dispositivo).requires_grad_()
    atenuacion, corr = atenuacion.to(dispositivo), CORR_COLOR.to(dispositivo)
    media, de = MEDIA_IN.to(dispositivo), DE_IN.to(dispositivo)

    def a_imagen():
        x = torch.fft.irfft2(torch.view_as_complex(coef) * atenuacion, s=(tam, tam), norm="ortho")
        return torch.sigmoid(torch.einsum("ij,bjhw->bihw", corr, x))     # colores correlacionados, en [0, 1]

    red = vgg[:CAPAS[nombre]]                       # hasta la convolución, sin la ReLU
    opt = torch.optim.Adam([coef], lr=lr)
    for _ in range(pasos):
        x = F.pad(a_imagen(), (8, 8, 8, 8), mode="reflect")
        i, j = torch.randint(0, 17, (2,), generator=g).tolist()      # traslación al azar de hasta ±8 pixeles
        x = x[:, :, i:i + tam, j:j + tam]
        x = F.interpolate(x, scale_factor=float(torch.empty(1).uniform_(0.9, 1.1, generator=g)), mode="bilinear")
        activacion = red((x - media) / de)[torch.arange(n), canales].mean((1, 2))
        opt.zero_grad()
        (-activacion.sum()).backward()
        opt.step()
    return a_imagen().detach().cpu()


PASOS_MAX, TAM_MAX = (6, 48) if RAPIDO else (48, 96)
t0 = time.time()
imagenes_max = {nombre: imagen_maxima(nombre, cs, PASOS_MAX, TAM_MAX) for nombre, cs in CANALES.items()}
t_max = time.time() - t0
print(f"{sum(len(c) for c in CANALES.values())} imágenes de {TAM_MAX} × {TAM_MAX} con {PASOS_MAX} pasos en {t_max:.0f} s")

# %%
fig, ejes = plt.subplots(len(CANALES), 4, figsize=(10, 13))
for r, (nombre, ims) in enumerate(imagenes_max.items()):
    for c, im in enumerate(ims):
        ejes[r, c].imshow(im.permute(1, 2, 0))
        ejes[r, c].set_xticks([]); ejes[r, c].set_yticks([])
        ejes[r, c].set_title(f"canal {CANALES[nombre][c]}", fontsize=10, loc="center")
    ejes[r, 0].set_ylabel(nombre, fontsize=11, color=MARINO)
fig.suptitle("La imagen sintética que más activa a cada canal", x=0.01, ha="left", color=MARINO, fontweight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Las imágenes sintéticas cuentan la misma historia que los parches, sin depender de nuestras fotos. En `conv1_2` son rayas finas con una orientación y unos colores; en `conv2_2`, rayas más gruesas, curvas y puntos; en `conv3_3`, texturas repetidas (espirales, escamas); en `conv4_3`, estructuras con forma (agujas como torres, pantallas); en `conv5_3`, fragmentos de objetos: el canal 184, el de las pelotas de golf, dibuja una esfera con hoyuelos; el 62, el de los perros, pelaje con manchas y rizos. Con más pasos, imágenes más grandes y más trucos de regularización salen imágenes mucho más nítidas; el artículo de *Distill* las muestra para todos los canales de otra red, GoogLeNet.
#
# ## 2. Por qué esto sirve para otros problemas
#
# J. Yosinski, J. Clune, Y. Bengio y H. Lipson («How transferable are features in deep neural networks?», NeurIPS 2014) lo midieron directamente. Partieron ImageNet en dos mitades con clases distintas, entrenaron una red en cada mitad y copiaron las primeras $l$ capas de una red a una red nueva para la otra mitad. Encontraron que:
#
# - las **primeras capas son genéricas**: copiar congeladas las dos primeras capas entrenadas con la otra mitad funciona igual que entrenarlas con los datos propios;
# - las **últimas capas son específicas** de su tarea: a partir de la tercera, copiar más capas congeladas empeora el resultado, y la caída es mayor mientras más distintas son las tareas (al separar objetos hechos por el ser humano de seres naturales, la caída fue mayor que con dos mitades al azar);
# - si después de copiar las capas se **ajusta toda la red** con los datos nuevos, el resultado es incluso mejor que entrenar desde cero, aunque haya muchos datos.
#
# En la figura anterior se ve por qué: un detector de bordes o de texturas le sirve a casi cualquier problema de imágenes; un detector de cabezas de perro, a pocos. Esto sugiere la receta de la **transferencia**: tomar una red entrenada en un problema grande, quedarse con sus capas y reentrenar (solo o con más cuidado) lo que es específico.
#
# ## 3. Dónde están los modelos preentrenados
#
# No hay que entrenar nada para tener una red como VGG16: hay catálogos públicos con miles de modelos, con sus pesos listos para descargar.
#
# - **`torchvision.models`** (PyTorch): clasificación, detección, segmentación y video, con pesos de ImageNet, COCO y otros. Es lo que usamos en esta libreta.
# - **`timm`** (*PyTorch Image Models*, de R. Wightman) y el **Hub de Hugging Face**: cientos de arquitecturas de visión con varios juegos de pesos cada una, cargables con `timm.create_model("resnet18", pretrained=True)`; documentación en <https://huggingface.co/docs/timm> y <https://huggingface.co/docs/hub/models>.
# - **Keras Applications** (<https://keras.io/api/applications/>): las arquitecturas clásicas con pesos de ImageNet para Keras y TensorFlow.
# - **Kaggle Models** (<https://www.kaggle.com/models>): modelos de TensorFlow, PyTorch y JAX, muchos publicados por Google.
# - **Ultralytics** (<https://docs.ultralytics.com/models/>): los detectores YOLO, entrenados en COCO, que usaremos en la siguiente libreta. Ojo: su licencia es AGPL-3.0.
#
# En `torchvision`, cada arquitectura tiene uno o más juegos de pesos, y cada juego trae su ficha (`meta`) y el preprocesamiento con el que se entrenó (`transforms()`). Podemos listarlos sin descargar nada:

# %%
from torchvision.models import list_models, get_model_weights

clasificadores = list_models(module=torchvision.models)
print(f"{len(clasificadores)} arquitecturas de clasificación; por ejemplo:", clasificadores[:6])
print("juegos de pesos de ResNet-50:", [str(w) for w in get_model_weights("resnet50")])

filas = []
for nombre in ["vgg16", "resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0", "convnext_tiny", "vit_b_16"]:
    w = get_model_weights(nombre).DEFAULT
    filas.append({"modelo": nombre, "pesos": str(w).split(".")[-1],
                  "parámetros (M)": round(w.meta["num_params"] / 1e6, 1),
                  "exactitud top-1 en ImageNet (%)": w.meta["_metrics"]["ImageNet-1K"]["acc@1"],
                  "GFLOPs por imagen": w.meta["_ops"], "archivo (MB)": round(w.meta["_file_size"])})
catalogo = pd.DataFrame(filas).set_index("modelo")
catalogo

# %% [markdown]
# Los pesos que vienen por omisión (`DEFAULT`) son los mejores disponibles, que no siempre son los originales: los de ResNet-50 `IMAGENET1K_V2` se entrenaron en 2021 con una receta moderna (más épocas, aumento de datos agresivo) y suben de 76.1 % a 80.9 %. La ficha de un juego de pesos dice además las clases, la receta de entrenamiento y el tamaño mínimo de imagen, y `transforms()` da el preprocesamiento exacto:

# %%
pesos_r18 = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
print("ficha:", sorted(pesos_r18.meta))
print("receta:", pesos_r18.meta["recipe"])
print("primeras clases:", pesos_r18.meta["categories"][:5])
print(pesos_r18.transforms())

# %% [markdown]
# Antes de usar un modelo preentrenado conviene revisar:
#
# - **Con qué datos se entrenó.** Define qué sabe y qué no (fotos de objetos no son radiografías) y qué sesgos trae (sección 5).
# - **La licencia.** La del código y la de los pesos pueden ser distintas. La documentación de `torchvision` advierte que los pesos pueden tener términos derivados de los datos con que se entrenaron (los de ImageNet restringen el uso comercial de las imágenes), y que revisarlos es responsabilidad de quien los usa. Los modelos de Ultralytics son AGPL-3.0: si los usas en un servicio, tienes que publicar tu código o pagar una licencia comercial.
# - **El preprocesamiento esperado.** Tamaño de imagen, orden de canales (RGB o BGR), escala de los pixeles y normalización. Darle a la red las imágenes con otra normalización degrada el resultado sin ningún error visible.
# - **El tamaño y el costo.** Parámetros, MB del archivo y operaciones por imagen: un teléfono o una cámara no corren lo mismo que un servidor con GPU.

# %% [markdown]
# ## 4. Transferencia a imágenes de satélite con pocas etiquetas
#
# ### Los datos y el problema
#
# **Datos.** EuroSAT (P. Helber, B. Bischke, A. Dengel y D. Borth, *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2019) tiene 27,000 imágenes de 64 × 64 pixeles tomadas por el satélite Sentinel-2 del programa Copernicus de la Unión Europea, sobre territorio europeo, etiquetadas en 10 clases de uso de suelo. Cada pixel cubre 10 m × 10 m, así que una imagen es un cuadro de 640 m de lado. Usamos la versión RGB (el satélite mide 13 bandas espectrales). Licencia MIT; `torchvision` la descarga la primera vez (unos 90 MB).
#
# **Problema.** Clasificar el uso de suelo de un cuadro de terreno. Etiquetar imágenes de satélite es caro (alguien tiene que saber qué hay en cada cuadro), así que la pregunta es: **¿qué tan bien podemos hacerlo con 10, 50 o 200 imágenes etiquetadas por clase?** ImageNet no tiene imágenes de satélite: la red nunca vio nada parecido.

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
# Usaremos una ResNet-18 entrenada en ImageNet (11.7 millones de parámetros). Su última capa, `fc`, es una capa lineal que lleva los 512 atributos de la penúltima capa a las 1,000 clases de ImageNet. Hay tres maneras de aprovecharla, de menos a más invasiva:
#
# | | qué se entrena | qué queda fijo |
# |---|---|---|
# | (a) atributos congelados + regresión logística | una logística (fuera de la red) | toda la red |
# | (b) sustituir solo la última capa | la nueva `fc` (5,130 parámetros) | todo lo demás |
# | (c) ajuste fino | toda la red, con tasas pequeñas | nada |
#
# y la comparamos con (d), una red pequeña entrenada desde cero con las mismas imágenes.
#
# Normalizamos las imágenes con la media y la desviación estándar de ImageNet por canal, que es lo que la red espera. No cambiamos su tamaño: ResNet-18 termina con un promedio global (*global average pooling*) y acepta imágenes de cualquier tamaño. (Agrandarlas a 224 × 224, el tamaño de ImageNet, da mejores resultados, pero cuesta doce veces más cómputo.)

# %%
N_ent_sat, N_pru_sat = (S_ent - MEDIA_IN) / DE_IN, (S_pru - MEDIA_IN) / DE_IN
tiempos = {}

# %% [markdown]
# ### (a) Atributos congelados y regresión logística
#
# Quitamos la capa `fc` (la cambiamos por la identidad), pasamos cada imagen por la red una sola vez y usamos los 512 números de salida como **atributos**; encima ajustamos una regresión logística con regularización, como en la Unidad 4.

# %%
resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
print(f"ResNet-18: {contar(resnet):,} parámetros")
resnet.fc = nn.Identity()            # la salida es el vector de 512 atributos
resnet.eval().to(dispositivo)


@torch.no_grad()
def atributos_resnet(X, lote=100):
    return torch.cat([resnet(X[i:i + lote].to(dispositivo)).cpu() for i in range(0, len(X), lote)]).numpy()


t0 = time.time()
A_ent, A_pru = atributos_resnet(N_ent_sat), atributos_resnet(N_pru_sat)
exact_transfer = {}
for k in POR_CLASE:
    logistica = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    logistica.fit(A_ent[:10 * k], ys_ent[:10 * k])
    exact_transfer[k] = logistica.score(A_pru, ys_pru)
tiempos["(a) atributos + logística"] = time.time() - t0
pd.Series(exact_transfer, name="exactitud de prueba").rename_axis("imágenes por clase")

# %% [markdown]
# Con solo **10 imágenes por clase** (100 en total), una regresión logística sobre los atributos de ImageNet acierta en más de tres de cada cuatro imágenes de satélite, en un problema de 10 clases donde adivinar daría 10 %.
#
# ### (d) La comparación: una red desde cero con los mismos datos
#
# ¿Es mérito de la red preentrenada o el problema es fácil? Entrenemos desde cero una red convolucional pequeña (tres bloques convolución + ReLU + *max-pooling*, un promedio global y una capa lineal: unos 24,000 parámetros) con las mismas imágenes. Para darle una oportunidad justa usamos **aumento de datos**: en cada paso volteamos al azar las imágenes horizontal y verticalmente (una foto de satélite volteada sigue siendo del mismo tipo de suelo). Le damos a cada tamaño de muestra el mismo presupuesto: 600 pasos de Adam con minilotes de 32.
#
# La función `entrenar_pasos` nos servirá también para el ajuste fino (c): por eso recibe, además del modelo, el optimizador, que dice qué parámetros se mueven y con qué tasa.

# %%
def red_satelite():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 64 → 32
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 32 → 16
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 16 → 8
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


@torch.no_grad()
def predecir_sat(modelo, X, lote=200):
    modelo.eval()
    return torch.cat([modelo(X[i:i + lote].to(dispositivo)).argmax(1).cpu() for i in range(0, len(X), lote)]).numpy()


def entrenar_pasos(modelo, X, y, pasos, optimizador=None, lote=32):
    """Adam con minilotes al azar y volteos aleatorios (aumento de datos); devuelve las predicciones de prueba."""
    modelo.to(dispositivo).train()
    opt = optimizador or torch.optim.Adam(modelo.parameters(), lr=1e-3)
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
    return predecir_sat(modelo, N_pru_sat)


PASOS = 150 if RAPIDO else 600
print(f"red desde cero: {contar(red_satelite()):,} parámetros")
exact_cero = {}
t0 = time.time()
for k in POR_CLASE:
    torch.manual_seed(SEMILLA)
    exact_cero[k] = (entrenar_pasos(red_satelite(), N_ent_sat[:10 * k], ys_ent[:10 * k], PASOS) == ys_pru).mean()
tiempos["(d) red desde cero"] = time.time() - t0
pd.Series(exact_cero, name="exactitud de prueba").rename_axis("imágenes por clase")

# %% [markdown]
# ### (b) Sustituir solo la última capa
#
# La forma más común de transferir en PyTorch: se carga la red completa, se **congelan** todos sus parámetros (`requires_grad_(False)`: el gradiente no se calcula para ellos y el optimizador no los mueve) y se reemplaza la capa `fc` por una nueva con 10 salidas, que sí se entrena:
#
# ```python
# modelo = torchvision.models.resnet18(weights="IMAGENET1K_V1")
# modelo.requires_grad_(False)          # congelar todo
# modelo.fc = nn.Linear(512, 10)        # capa nueva: sus parámetros sí se entrenan
# optimizador = torch.optim.Adam(modelo.fc.parameters(), lr=1e-3)
# ```
#
# Una capa lineal con *softmax* sobre los 512 atributos, entrenada con entropía cruzada, **es una regresión logística multinomial**: (b) es esencialmente (a) entrenada con Adam dentro de la red en lugar de con `scikit-learn`. Las diferencias son de detalle: aquí no estandarizamos los atributos y la regularización es implícita (pocos pasos de Adam) en lugar de una penalización explícita. A cambio, todo queda en un solo modelo de PyTorch, que recibe imágenes y devuelve clases, listo para el paso siguiente.
#
# **Un atajo.** Como la red base no cambia, los 512 atributos de cada imagen son siempre los mismos: no hace falta pasar las imágenes por la red en cada paso. Los calculamos una vez (son `A_ent`, los mismos de la estrategia (a)) y entrenamos solo `modelo.fc` sobre ellos; el resultado es idéntico y cuesta milésimas de segundo por paso en lugar de una pasada por la red. Al final comprobamos que el modelo completo, aplicado a las imágenes, da lo mismo.
#
# **Una trampa.** La ResNet tiene **normalización por lotes**, que en modo de entrenamiento (`modelo.train()`) recalcula medias y varianzas con cada minilote y actualiza sus promedios guardados **aunque los pesos estén congelados**. Si entrenas la capa nueva pasando las imágenes por la red, deja la red en modo de evaluación (`modelo.eval()`); si no, la red base cambia sin que te des cuenta.

# %%
PASOS_FC = 300
T_ent, T_pru = torch.as_tensor(A_ent), torch.as_tensor(A_pru)      # atributos de la red congelada
exact_ultima = {}
t0 = time.time()
for k in POR_CLASE:
    torch.manual_seed(SEMILLA)
    modelo = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    modelo.requires_grad_(False)
    modelo.fc = nn.Linear(512, 10)
    opt = torch.optim.Adam(modelo.fc.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(SEMILLA)
    y_k = torch.as_tensor(ys_ent[:10 * k])
    for _ in range(PASOS_FC):
        idx = torch.randint(10 * k, (32,), generator=g)
        opt.zero_grad()
        F.cross_entropy(modelo.fc(T_ent[idx]), y_k[idx]).backward()
        opt.step()
    with torch.no_grad():
        exact_ultima[k] = (modelo.fc(T_pru).argmax(1).numpy() == ys_pru).mean()
tiempos["(b) solo la última capa"] = time.time() - t0
print(f"parámetros entrenables: {sum(p.numel() for p in modelo.parameters() if p.requires_grad):,} de {contar(modelo):,}")

# comprobación: el modelo completo, aplicado a las imágenes de prueba, predice lo mismo
modelo.eval().to(dispositivo)
with torch.no_grad():
    iguales = (predecir_sat(modelo, N_pru_sat[:200]) == modelo.fc(T_pru[:200].to(dispositivo)).argmax(1).cpu().numpy()).mean()
print(f"modelo completo y atajo coinciden en {100 * iguales:.0f} % de 200 imágenes")
pd.Series(exact_ultima, name="exactitud de prueba").rename_axis("imágenes por clase")

# %% [markdown]
# ### (c) Ajuste fino de toda la red
#
# El siguiente nivel es dejar que **todas** las capas se adapten al problema nuevo: se parte de los pesos de ImageNet y se sigue entrenando la red completa con las imágenes de satélite. Esto se llama **ajuste fino** (*fine-tuning*). Dos cuidados:
#
# - **Tasas de aprendizaje pequeñas.** Los pesos de ImageNet ya son buenos; con una tasa grande, los primeros pasos (cuando la capa nueva todavía da gradientes al azar) los destruirían. Usamos $\eta = 10^{-3}$ para la capa nueva y $10^{-4}$ para el resto.
# - **Tasas aún menores para las primeras capas.** Por lo que vimos en las secciones 1 y 2, las capas tempranas son genéricas y casi no necesitan cambiar; las tardías son las que más hay que adaptar. A la entrada de la red y a los dos primeros bloques les damos $\eta = 10^{-5}$. En PyTorch esto se hace con **grupos de parámetros**: una lista de diccionarios, cada uno con sus parámetros y su tasa.
#
# Es mucho más caro que (b): hay que retropropagar por los 11.7 millones de parámetros, tres veces el costo de solo evaluar la red. En CPU nos limitamos a 200 pasos por tamaño de muestra (con 200 imágenes por clase son apenas tres pasadas por los datos); con GPU conviene probar más pasos (1,000 o más) y agrandar las imágenes a 128 o 224 pixeles.

# %%
PASOS_FINO = 25 if RAPIDO else 200
exact_fino, pred_fino = {}, {}
t0 = time.time()
for k in POR_CLASE:
    torch.manual_seed(SEMILLA)
    modelo = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    modelo.fc = nn.Linear(512, 10)
    tempranas = [modelo.conv1, modelo.bn1, modelo.layer1, modelo.layer2]
    grupos = [
        {"params": [p for m in tempranas for p in m.parameters()], "lr": 1e-5},
        {"params": [*modelo.layer3.parameters(), *modelo.layer4.parameters()], "lr": 1e-4},
        {"params": modelo.fc.parameters(), "lr": 1e-3},
    ]
    pred_fino[k] = entrenar_pasos(modelo, N_ent_sat[:10 * k], ys_ent[:10 * k], PASOS_FINO, torch.optim.Adam(grupos))
    exact_fino[k] = (pred_fino[k] == ys_pru).mean()
tiempos["(c) ajuste fino"] = time.time() - t0
pd.Series(exact_fino, name="exactitud de prueba").rename_axis("imágenes por clase")

# %% [markdown]
# ### La comparación

# %%
ESTRATEGIAS = {"(a) atributos congelados + logística": exact_transfer, "(b) solo la última capa": exact_ultima,
               "(c) ajuste fino": exact_fino, "(d) red pequeña desde cero": exact_cero}
comparacion = pd.DataFrame(ESTRATEGIAS)
comparacion.index.name = "imágenes por clase"
print({k: f"{v:.0f} s" for k, v in tiempos.items()})
comparacion.round(3)

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ks = list(POR_CLASE)
colores = [GRIS, DORADO_OSC, AZUL, ROJO]
for (nombre, exact), color in zip(ESTRATEGIAS.items(), colores):
    ax.plot(ks, [100 * exact[k] for k in ks], "o-", color=color)
ax.axhline(10, color=GRIS, ls=":", lw=1.5)
ax.text(ks[0], 12, "adivinar al azar", color=TENUE, fontsize=11)
ax.set_xscale("log")
ax.set_xticks(ks, [str(k) for k in ks])
ax.minorticks_off()
ax.set_xlim(ks[0] * 0.8, ks[-1] * 4.5)
ax.set_ylim(0, 100)
ax.set_xlabel("imágenes etiquetadas por clase (escala logarítmica)")
ax.set_ylabel("exactitud de prueba (%)")
ax.set_title("EuroSAT: tres formas de transferir contra entrenar desde cero")
rdp.estilo.etiquetar(ax, ks[-1] * 1.12, {nombre: 100 * exact[ks[-1]] for nombre, exact in ESTRATEGIAS.items()})
plt.show()

# %% [markdown]
# Tres lecciones:
#
# - **Cualquier forma de transferir gana por mucho.** Con 10 imágenes por clase, las tres estrategias que parten de ImageNet aciertan entre 77.1 % y 77.9 %, contra 55.0 % de la red desde cero; con 200 por clase, entre 88.8 % y 91.4 %, contra 65.5 %. La red preentrenada con 10 imágenes por clase supera a la red desde cero con 200.
# - **(a) y (b) dan casi lo mismo** (77.1 % contra 77.9 % con 10 por clase, 88.8 % contra 88.9 % con 200), como debía ser: son el mismo modelo, una regresión logística sobre los mismos 512 atributos, ajustada de dos maneras.
# - **El ajuste fino rinde más mientras más datos hay.** Con 10 imágenes por clase no gana nada (77.4 %): con tan pocos ejemplos hay poco que adaptar y mucho riesgo de sobreajustar. Con 50 por clase sube a 87.8 % y con 200 a 91.4 %, 2.6 puntos sobre los atributos congelados: con 200 imágenes por clase, desaparece casi una de cada cuatro equivocaciones de (a). Es lo que encontraron Yosinski et al.: adaptar las capas tardías, que son las específicas de ImageNet, ayuda. Y en CPU es, por mucho, lo más caro.
#
# Entrenar desde cero no es inútil: con miles de imágenes por clase y más pasos, una red entrenada solo con EuroSAT llega a exactitudes muy altas (los autores del conjunto reportan 98.57 % ajustando redes profundas preentrenadas con el 80 % de los datos). La transferencia rinde más **cuando hay pocas etiquetas**, que es la situación común en la práctica: en imágenes médicas, agrícolas o industriales casi nunca hay millones de ejemplos etiquetados.
#
# ### Dónde se equivoca el mejor modelo

# %%
k = max(POR_CLASE)
mc_sat = confusion_matrix(ys_pru, pred_fino[k])
print("exactitud por clase:", {CLASES_SAT[c]: round(mc_sat[c, c] / mc_sat[c].sum(), 2) for c in range(10)})
pares_sat = [(mc_sat[i, j], CLASES_SAT[i], CLASES_SAT[j]) for i in range(10) for j in range(10) if i != j]
errores_comunes = pd.DataFrame(sorted(pares_sat, reverse=True)[:5], columns=["casos", "verdadera", "predicha"])
errores_comunes

# %%
fig, ax = plt.subplots(figsize=(6.8, 6))
ax.imshow(mc_sat, cmap="rdp_azul")
for i in range(10):
    for j in range(10):
        if mc_sat[i, j]:
            ax.text(j, i, mc_sat[i, j], ha="center", va="center", fontsize=9,
                    color="white" if mc_sat[i, j] > mc_sat.max() / 2 else TINTA)
ax.set_xticks(range(10), CLASES_SAT, rotation=60, ha="right", fontsize=9)
ax.set_yticks(range(10), CLASES_SAT, fontsize=9)
ax.set_xlabel("clase predicha")
ax.set_ylabel("clase verdadera")
ax.grid(False)
ax.set_title(f"Ajuste fino con {k} imágenes por clase: matriz de confusión", fontsize=12)
plt.show()

# %% [markdown]
# El modelo ajustado acierta más de 95 % en bosque, zona industrial, pastizal, zona residencial y mar o lago; sus clases más difíciles son cultivo permanente (83 %), carretera (85 %) y río (86 %). Los errores son razonables: carreteras y ríos que se confunden entre sí (8 y 6 casos; en los dos, una franja larga que cruza el cuadro) y tipos de vegetación y cultivo que se parecen a esta resolución (un huerto visto desde 640 m de altura y un matorral no son tan distintos).
#
# ## 5. Lo que se hereda: el caso *Gender Shades*
#
# La transferencia es poderosa porque el modelo **hereda** lo que aprendió con otros datos. También hereda sus huecos. J. Buolamwini y T. Gebru («Gender Shades», *Conference on Fairness, Accountability and Transparency*, 2018) evaluaron los clasificadores comerciales de género a partir de fotos de rostros de Microsoft, IBM y Face++ con 1,270 fotos de parlamentarios de seis países (tres africanos y tres europeos), etiquetadas por género y por tono de piel con la escala de Fitzpatrick. Los tres sistemas tenían su mayor error en las **mujeres de piel oscura**, de hasta **34.7 %**, mientras que el error máximo en **hombres de piel clara** fue de **0.8 %**. Los autores también mostraron que dos conjuntos de rostros muy usados para entrenar y evaluar, IJB-A y Adience, tenían entre 79.6 % y 86.2 % de personas de piel clara.
#
# Un detector de atributos entrenado con datos que representan mal a un grupo produce atributos peores para ese grupo, y ningún clasificador que se ajuste encima puede recuperar la información que no está ahí. Al usar un modelo preentrenado conviene preguntarse, igual que con cualquier conjunto de datos (Unidad 1): **¿con qué datos se entrenó, a quién representan y a quién no?** Y evaluar por separado en cada grupo relevante, como hicimos en la Unidad 3.
#
# ## 6. Cierre
#
# - Las capas de una red convolucional forman una jerarquía que nadie programó: bordes y colores, texturas, partes y objetos, con campos receptivos cada vez más grandes. Se ve con los parches que más activan a cada canal y con la imagen que lo maximiza.
# - Las primeras capas son genéricas y las últimas específicas de la tarea: por eso una red entrenada en ImageNet sirve como punto de partida para otros problemas de imágenes.
# - Hay catálogos de modelos preentrenados (`torchvision`, `timm` y Hugging Face, Keras, Kaggle, Ultralytics). Antes de usar uno: datos de origen, licencia, preprocesamiento y tamaño.
# - Tres formas de transferir, de menos a más invasiva: atributos congelados con un clasificador encima, sustituir solo la última capa (que es casi lo mismo) y ajuste fino de toda la red con tasas pequeñas, menores en las primeras capas.
# - Con pocas etiquetas, cualquiera de las tres supera por mucho a entrenar desde cero; pero el modelo hereda los sesgos de sus datos de origen.
#
# *Referencias:* M. D. Zeiler y R. Fergus, «Visualizing and understanding convolutional networks», ECCV (2014). D. Erhan, Y. Bengio, A. Courville y P. Vincent, «Visualizing higher-layer features of a deep network», reporte técnico 1341, Universidad de Montreal (2009). C. Olah, A. Mordvintsev y L. Schubert, «Feature visualization», *Distill* (2017), <https://distill.pub/2017/feature-visualization/>. W. Luo, Y. Li, R. Urtasun y R. Zemel, «Understanding the effective receptive field in deep convolutional neural networks», NeurIPS (2016). K. Simonyan y A. Zisserman, «Very deep convolutional networks for large-scale image recognition», ICLR (2015). K. He, X. Zhang, S. Ren y J. Sun, «Deep residual learning for image recognition», CVPR (2016). J. Yosinski, J. Clune, Y. Bengio y H. Lipson, «How transferable are features in deep neural networks?», NeurIPS (2014). J. Howard, Imagenette, <https://github.com/fastai/imagenette> (2019). P. Helber et al., «EuroSAT», *IEEE JSTARS* 12 (2019). J. Buolamwini y T. Gebru, «Gender Shades», PMLR 81 (2018). Documentación de los modelos de `torchvision`: <https://pytorch.org/vision/stable/models.html>.

# %% [markdown]
# ## 7. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p12"])

# %% [markdown]
# ## 8. Ejercicios
#
# 1. **Otros canales.** Cambia la lista `CANALES` por canales al azar de cada capa (usa `np.random.default_rng(...).choice`). ¿Se sigue viendo la progresión? ¿En qué capa es más difícil decir qué detecta un canal?
# 2. **Campo receptivo.** Verifica a mano, con la recurrencia de la sección 1, los campos receptivos de `conv2_2` y `conv3_3`. Luego mide el campo receptivo **efectivo** de un canal de `conv3_3`: calcula el gradiente de su activación en el pixel central del mapa con respecto a la imagen de entrada, promediado sobre varias fotos, y dibuja su magnitud. ¿Qué tan grande es la región donde el gradiente no es casi cero, comparada con los 40 pixeles teóricos?
# 3. **Sin trucos.** En `imagen_maxima`, optimiza los pixeles directamente (sin la transformada de Fourier) y sin las traslaciones al azar. ¿Qué cambia en las imágenes de `conv4_3` y `conv5_3`?
# 4. **Cuánto ajustar.** Repite el ajuste fino (c) con 50 imágenes por clase en tres variantes: todas las capas con $\eta = 10^{-4}$ (sin grupos), congelando `conv1`, `layer1` y `layer2` (`requires_grad_(False)`), y con $\eta = 10^{-2}$ para toda la red. ¿Cuál funciona mejor y por qué la última es mala idea?
# 5. **Más etiquetas.** Agrega a la comparación el caso de 1,000 imágenes por clase (la reserva tiene que crecer). ¿Alcanza la red desde cero a la transferencia? Si tienes GPU, agranda las imágenes a 224 × 224 con `F.interpolate` antes de la ResNet y compara.
# 6. **Otro modelo preentrenado.** Usa la tabla de la sección 3 para escoger otro modelo (por ejemplo `efficientnet_b0` o `convnext_tiny`), repite la estrategia (a) con sus atributos y compara exactitud y tiempo con ResNet-18. Revisa con `transforms()` qué preprocesamiento espera.
