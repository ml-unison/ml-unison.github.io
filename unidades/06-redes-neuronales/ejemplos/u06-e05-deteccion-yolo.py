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
# # Detección de objetos: de las ventanas deslizantes a YOLO
#
# **Unidad 6 · Ejemplo 5**
#
# Un clasificador de imágenes contesta *qué hay* en una foto: «un auto». En una calle de Hermosillo eso no basta. Un sistema de conteo de tráfico, un auto que se maneja solo o una cámara que vigila un cruce peatonal necesitan saber **cuántos** objetos hay, **de qué clase** es cada uno y **dónde** está. Eso es la **detección de objetos**. En esta libreta vamos a:
#
# 1. ver qué cambia al pasar de clasificar a detectar: cada objeto tiene una clase **y una caja**;
# 2. programar la **intersección sobre unión** (IoU), la medida con la que se comparan dos cajas, y entender cómo se evalúa un detector (mAP);
# 3. recorrer la historia: ventanas deslizantes con atributos hechos a mano, R-CNN y sus sucesores, y **YOLO**, que lo hace todo en una sola pasada;
# 4. programar la **supresión de no máximos** (NMS) y verla trabajar sobre las cajas crudas de un detector real;
# 5. usar YOLO26, la versión actual de Ultralytics, en **tres líneas** sobre fotos de Hermosillo y de la Universidad de Sonora;
# 6. compararlo con un detector de dos etapas (Faster R-CNN) en tiempo y en detecciones.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso y **Ultralytics**, la biblioteca que implementa YOLO (Colab no la trae). PyTorch y torchvision ya vienen instalados. Todo corre en CPU en unos minutos; la primera vez se descargan los modelos preentrenados (unos 5 MB de YOLO26n y unos 250 MB de los dos Faster R-CNN).

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    # %pip install -q ultralytics
    pass

# %%
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import torch
import torchvision
from torchvision.ops import box_iou, nms, batched_nms
from ultralytics import YOLO

import rdp

rdp.estilo.usar()
AZUL, ROJO, DORADO_OSC, GRIS, MARINO = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS, rdp.estilo.MARINO
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# la verificación automática del repositorio usa una sola foto y un solo detector de dos etapas
RAPIDO = os.environ.get("RDP_RAPIDO") == "1"
RAIZ_DATOS = os.path.expanduser("~/.cache/rdp-datos")
os.makedirs(RAIZ_DATOS, exist_ok=True)
torch.manual_seed(2016)
print("hilos de CPU que usa PyTorch:", torch.get_num_threads())

# %% [markdown]
# ## 1. Las fotos
#
# **Datos.** Tres fotos de Wikimedia Commons, todas con licencia CC BY-SA 3.0 (se pueden usar y modificar citando al autor y compartiendo con la misma licencia). Guardamos copias reducidas en la carpeta `datos/fotos` del repositorio del curso; `rdp.datos.archivo` las toma de ahí, o de GitHub si la libreta corre en Colab.
#
# - **Universidad de Sonora**: el Edificio Principal y el bulevar Luis Encinas desde la biblioteca universitaria, 2012. Autor: Metamario. <https://commons.wikimedia.org/wiki/File:LuisEncinasUnison.JPG> (reducida de 3260 × 2196 a 1280 × 862 pixeles).
# - **Calle Matamoros**, en el centro, con el cerro de la Campana al fondo, 2013. Autor: CarlosyFrancisca. <https://commons.wikimedia.org/wiki/File:Matamoros_final_centro.jpg> (tamaño original, 958 × 718).
# - **Bulevar Morelos**, un crucero en 2006. Autor: Sergio R. (Panoramio). <https://commons.wikimedia.org/wiki/File:PICT0922_-_panoramio.jpg> (reducida de 2048 × 1536 a 1280 × 960).
#
# **Problema.** Encontrar en cada foto todos los objetos de las 80 clases del conjunto COCO (personas, autos, autobuses, semáforos, bancas, perros…) con su caja y su clase.

# %%
FOTOS = {
    "unison": {"archivo": "fotos/unison_luis_encinas.jpg", "titulo": "Universidad de Sonora y bulevar Luis Encinas",
               "autor": "Metamario", "licencia": "CC BY-SA 3.0",
               "url": "https://commons.wikimedia.org/wiki/File:LuisEncinasUnison.JPG"},
    "matamoros": {"archivo": "fotos/calle_matamoros.jpg", "titulo": "Calle Matamoros, centro de Hermosillo",
                  "autor": "CarlosyFrancisca", "licencia": "CC BY-SA 3.0",
                  "url": "https://commons.wikimedia.org/wiki/File:Matamoros_final_centro.jpg"},
    "morelos": {"archivo": "fotos/boulevard_morelos.jpg", "titulo": "Bulevar Morelos, Hermosillo",
                "autor": "Sergio R.", "licencia": "CC BY-SA 3.0",
                "url": "https://commons.wikimedia.org/wiki/File:PICT0922_-_panoramio.jpg"},
}
if RAPIDO:
    FOTOS = {"matamoros": FOTOS["matamoros"]}

fotos = {nombre: Image.open(rdp.datos.archivo(f["archivo"])).convert("RGB") for nombre, f in FOTOS.items()}
for nombre, im in fotos.items():
    print(f"{nombre:10s} {im.width} × {im.height} pixeles")

# %%
fig, ejes = plt.subplots(1, len(fotos), figsize=(5.2 * len(fotos), 4), squeeze=False)
for ax, (nombre, im) in zip(ejes[0], fotos.items()):
    ax.imshow(im)
    ax.set_title(f"{FOTOS[nombre]['titulo']}\n© {FOTOS[nombre]['autor']}, {FOTOS[nombre]['licencia']}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. De clasificar a detectar
#
# Pregunta para el grupo: ¿cuántos autos hay en la foto de la calle Matamoros? Para contestar no basta con una etiqueta por foto. Un **detector** produce una **lista de longitud variable**: por cada objeto que encuentra, una **clase**, una **caja** que lo encierra y una **confianza** (un número entre 0 y 1 que el modelo usa para ordenar sus respuestas).
#
# La caja más común es un rectángulo con lados paralelos a los bordes de la imagen, que se escribe con sus esquinas superior izquierda e inferior derecha, $(x_1, y_1, x_2, y_2)$, en pixeles. El origen está en la esquina superior izquierda de la imagen y $y$ crece **hacia abajo**. Otros formatos guardan lo mismo de otra forma: el centro, el ancho y el alto $(x_c, y_c, w, h)$, que usa YOLO internamente, o las mismas cantidades divididas entre el ancho y el alto de la imagen, para que no dependan de su resolución. Pasar de uno a otro es aritmética:

# %%
def centro_a_esquinas(c):
    """(xc, yc, w, h) → (x1, y1, x2, y2). Funciona con un tensor de forma (..., 4)."""
    xc, yc, w, h = c.unbind(-1)
    return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=-1)


print(centro_a_esquinas(torch.tensor([50.0, 40.0, 20.0, 10.0])))

# %% [markdown]
# Detectar es, entonces, **clasificar y hacer regresión a la vez**, y además decidir **cuántos** objetos hay. Las dos dificultades nuevas son: (a) cómo comparar una caja predicha con la verdadera, y (b) qué hacer cuando el detector encuentra el mismo objeto varias veces. Las dos se resuelven con la misma medida.

# %% [markdown]
# ## 3. Intersección sobre unión (IoU)
#
# Una caja predicha casi nunca coincide al pixel con la verdadera. ¿Cuándo decimos que «le atinó»? La medida estándar es la **intersección sobre unión** (*intersection over union*, IoU, también llamada índice de Jaccard):
#
# $$\operatorname{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|},$$
#
# donde $|A|$ es el área de la caja $A$. Vale 1 si las cajas coinciden y 0 si no se tocan. Divide entre la unión, y no entre el área de una de las cajas, para castigar por igual a una caja demasiado chica y a una demasiado grande. La intersección de dos rectángulos alineados es otro rectángulo: su esquina superior izquierda es el **máximo** de las dos esquinas superiores izquierdas, y la inferior derecha, el **mínimo** de las inferiores derechas; si el ancho o el alto salen negativos, no se tocan.

# %%
def area(c):
    return (c[..., 2] - c[..., 0]).clamp(min=0) * (c[..., 3] - c[..., 1]).clamp(min=0)


def iou_matriz(A, B):
    """IoU de cada caja de A (N, 4) con cada caja de B (K, 4), en formato (x1, y1, x2, y2). Devuelve (N, K)."""
    sup_izq = torch.maximum(A[:, None, :2], B[None, :, :2])      # (N, K, 2)
    inf_der = torch.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = (inf_der - sup_izq).clamp(min=0).prod(dim=-1)        # ancho × alto, o 0 si no se tocan
    return inter / (area(A)[:, None] + area(B)[None, :] - inter)


caja_verdadera = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
ejemplos_iou = {
    "desplazada a la mitad": torch.tensor([[5.0, 0.0, 15.0, 10.0]]),
    "corrida un pixel en x y en y": torch.tensor([[1.0, 1.0, 11.0, 11.0]]),
    "chica, adentro": torch.tensor([[2.0, 2.0, 8.0, 8.0]]),
    "lejos": torch.tensor([[12.0, 0.0, 20.0, 10.0]]),
}
for nombre, caja in ejemplos_iou.items():
    print(f"{nombre:30s} IoU = {iou_matriz(caja_verdadera, caja).item():.3f}")

# verificación contra torchvision con cajas al azar
g = torch.Generator().manual_seed(0)
esq = torch.rand(200, 2, generator=g) * 100
A_az = torch.cat([esq, esq + 1 + torch.rand(200, 2, generator=g) * 50], dim=1)
esq = torch.rand(150, 2, generator=g) * 100
B_az = torch.cat([esq, esq + 1 + torch.rand(150, 2, generator=g) * 50], dim=1)
print("diferencia máxima con torchvision.ops.box_iou:", (iou_matriz(A_az, B_az) - box_iou(A_az, B_az)).abs().max().item())

# %%
fig, ejes = plt.subplots(1, len(ejemplos_iou), figsize=(13, 3.4))
for ax, (nombre, caja) in zip(ejes, ejemplos_iou.items()):
    for c, color, estilo in ((caja_verdadera[0], AZUL, "-"), (caja[0], ROJO, "--")):
        ax.add_patch(Rectangle((c[0], c[1]), c[2] - c[0], c[3] - c[1], fill=False, ec=color, lw=2.5, ls=estilo))
    ax.set_xlim(-2, 21)
    ax.set_ylim(13, -3)          # y crece hacia abajo, como en una imagen
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{nombre}\nIoU = {iou_matriz(caja_verdadera, caja).item():.2f}", fontsize=11)
fig.suptitle("Caja verdadera (azul, continua) y predicha (roja, punteada)", x=0.01, ha="left", color=MARINO, fontweight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Tres observaciones. Una caja del mismo tamaño corrida a la mitad tiene IoU de solo $50/150 = 1/3$. Correrla un pixel en cada dirección ya baja el IoU a $81/119 \approx 0.68$. Y una caja chica bien centrada, que cabe toda dentro de la verdadera, tiene IoU $36/100 = 0.36$: no basta con «estar adentro». La función vectorizada calcula todas las parejas a la vez con *broadcasting* y coincide con `torchvision.ops.box_iou` hasta el redondeo.
#
# ### Cómo se evalúa un detector: mAP
#
# Con el IoU se define qué es un acierto. Se fija un **umbral**, típicamente 0.5. Las detecciones de una clase se ordenan de mayor a menor confianza y se recorren en ese orden: una detección es un **verdadero positivo** si su IoU con alguna caja verdadera de la misma clase, aún no emparejada, es al menos el umbral; si no, es un **falso positivo** (una detección repetida del mismo objeto también lo es). Las cajas verdaderas que nadie encontró son **falsos negativos**. Al bajar poco a poco el umbral de confianza se obtiene una curva de **precisión** (qué fracción de las detecciones son correctas) contra **exhaustividad** (qué fracción de los objetos se encontraron). El área bajo esa curva es la **precisión promedio** (AP) de la clase, y su promedio sobre las clases es la **mAP** (*mean average precision*).
#
# - **mAP50** (o mAP@0.5) usa umbral de IoU 0.5; fue la métrica del concurso PASCAL VOC (20 clases).
# - **mAP50-95**, la métrica principal del conjunto **COCO** (T.-Y. Lin et al., 2014; 80 clases, unas 118 mil imágenes de entrenamiento), promedia la mAP con los diez umbrales 0.50, 0.55, …, 0.95. Premia las cajas **precisas**, no solo las que más o menos caen en su lugar. Cuando un artículo o una tabla dice «mAP» a secas sobre COCO, casi siempre es esta.

# %% [markdown]
# ## 4. Historia breve: de la ventana deslizante a YOLO
#
# ### Ventanas deslizantes con atributos hechos a mano (2001–2012)
#
# La idea más directa: si tengo un **clasificador** de «persona contra fondo» para recortes de tamaño fijo, lo aplico a **todas** las ventanas de la imagen, en todas las posiciones y en varias escalas, y me quedo con las ventanas que dicen «persona».
#
# - **Viola y Jones (2001)** detectaron rostros así con atributos muy baratos (diferencias de sumas de rectángulos claros y oscuros, calculadas en tiempo constante con una «imagen integral») y una **cascada** de clasificadores de *AdaBoost* que descarta en pocas operaciones la inmensa mayoría de las ventanas, que no son caras. Procesaban 15 imágenes de 384 × 288 por segundo en un Pentium III de 700 MHz: fue el detector de caras de las cámaras digitales durante una década.
# - **Dalal y Triggs (2005)** propusieron el **histograma de gradientes orientados** (HOG) para detectar peatones: la ventana de 64 × 128 pixeles se divide en celdas de 8 × 8, en cada celda se hace un histograma de 9 orientaciones del gradiente, se normaliza por bloques de 2 × 2 celdas y todo se concatena en un vector de 3,780 atributos que clasifica una máquina de soporte vectorial lineal.
#
# ¿Cuántas ventanas hay que clasificar? Contemos para una foto de 640 × 480 con la ventana de 64 × 128 de Dalal y Triggs, un paso de 8 pixeles y escalas que crecen 5 % cada vez (la imagen se reduce por 1.05 hasta que la ventana ya no cabe):

# %%
def contar_ventanas(ancho, alto, ventana=(64, 128), paso=8, factor=1.05):
    """Ventanas de tamaño fijo en una pirámide de imágenes que se reduce por `factor` en cada nivel."""
    filas, s = [], 1.0
    while ancho / s >= ventana[0] and alto / s >= ventana[1]:
        w, h = int(ancho / s), int(alto / s)
        n = ((w - ventana[0]) // paso + 1) * ((h - ventana[1]) // paso + 1)
        filas.append({"escala": round(s, 3), "ancho": w, "alto": h, "ventanas": n})
        s *= factor
    return pd.DataFrame(filas)


ventanas = contar_ventanas(640, 480)
total_ventanas = int(ventanas["ventanas"].sum())
print(f"niveles de la pirámide: {len(ventanas)};  ventanas en total: {total_ventanas:,}")
print(f"solo en la escala original: {ventanas['ventanas'].iloc[0]:,}")
print(f"multiplicaciones del SVM lineal sobre HOG (3,780 por ventana): {total_ventanas * 3780 / 1e6:,.0f} millones")
# si en cada ventana corriéramos una red como AlexNet (unos 720 millones de multiplicaciones-sumas por recorte)
print(f"con AlexNet en cada ventana: {total_ventanas * 720e6 / 1e12:,.0f} billones de multiplicaciones")
ventanas.head()

# %%
im_vent = fotos["matamoros"].resize((640, 480))
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.imshow(im_vent)
for _, fila in ventanas.iloc[[0, 8, 16]].iterrows():          # tres escalas
    s, w_v, h_v = fila["escala"], 64 * fila["escala"], 128 * fila["escala"]
    for x in np.arange(0, 640 - w_v + 1, 8 * s)[::9]:
        ax.add_patch(Rectangle((x, 300 - h_v / 2), w_v, h_v, fill=False, ec=DORADO_OSC, lw=1, alpha=0.8))
ax.set_title(f"Unas cuantas de las {total_ventanas:,} ventanas, en tres escalas", fontsize=11)
ax.axis("off")
plt.show()

# %% [markdown]
# Son 25,132 ventanas en 28 escalas (3,285 solo en la escala original) para **una** forma de ventana (la de un peatón de pie). Un auto visto de lado o un autobús necesitan otras proporciones, y cada forma multiplica la cuenta. Con atributos HOG y un clasificador lineal son unos 95 millones de multiplicaciones por imagen (sin contar el cálculo de los histogramas), algo manejable. Pero esos atributos hechos a mano se estancaron: hacia 2012, la mAP de los mejores sistemas en el concurso PASCAL VOC apenas mejoraba de un año a otro. Y si en cada ventana corriéramos una red convolucional como AlexNet, la que ganó ImageNet en 2012, serían unos 18 billones (millones de millones) de multiplicaciones por foto, y por una sola forma de ventana.
#
# ### R-CNN (2014): menos ventanas, mejores atributos
#
# R. Girshick, J. Donahue, T. Darrell y J. Malik («Rich feature hierarchies for accurate object detection and semantic segmentation», CVPR 2014) cambiaron las dos piezas:
#
# 1. En lugar de todas las ventanas, unas **2,000 propuestas de región** por imagen, producidas por la **búsqueda selectiva** (J. Uijlings et al., 2013), un algoritmo sin aprendizaje que agrupa pixeles parecidos en regiones de todos los tamaños y formas.
# 2. Cada propuesta se recorta, se deforma a 227 × 227 y se pasa por **una red convolucional entrenada en ImageNet** (AlexNet) para obtener 4,096 atributos; encima, una máquina de soporte vectorial lineal por clase. Los atributos de una red entrenada para **clasificar** ImageNet sirven para **detectar** en otro conjunto: es **transferencia**, y la red se ajustaba además con las imágenes de detección.
#
# En PASCAL VOC 2012 la mAP llegó a 53.3 %, más de 30 % por arriba (en términos relativos) del mejor resultado anterior. El precio: pasar 2,000 recortes por la red. El artículo de Fast R-CNN reporta **47 segundos por imagen en GPU** para R-CNN con la red VGG-16.
#
# ### Fast R-CNN y Faster R-CNN (2015): compartir el cómputo
#
# - **Fast R-CNN** (R. Girshick, ICCV 2015) pasa **la imagen completa una sola vez** por la red convolucional y recorta cada propuesta **del mapa de atributos** (*RoI pooling*), no de la imagen. Bajó el tiempo de prueba a 0.32 s por imagen (sin contar la búsqueda selectiva, que tardaba unos 2 s en CPU) con mejor mAP.
# - **Faster R-CNN** (S. Ren, K. He, R. Girshick y J. Sun, NeurIPS 2015) sustituye la búsqueda selectiva por una **red de propuestas de región** (RPN) que comparte el mapa de atributos: en cada posición del mapa se colocan 9 **anclas** (cajas de referencia de 3 tamaños y 3 proporciones) y la red predice, para cada ancla, si contiene un objeto y cómo corregir su caja. Unas 300 propuestas pasan a la segunda etapa, que clasifica y ajusta la caja. Corría a unas 5 imágenes por segundo en GPU con VGG-16. Por estas dos etapas se le llama detector **de dos etapas**.
#
# ### YOLO (2016): una sola mirada
#
# J. Redmon, S. Divvala, R. Girshick y A. Farhadi («You Only Look Once: Unified, real-time object detection», CVPR 2016) eliminaron las propuestas. La imagen se divide en una rejilla de $S \times S = 7 \times 7$ celdas. La celda que contiene **el centro** de un objeto es la responsable de detectarlo. Cada celda predice $B = 2$ cajas, cada una con 5 números ($x_c, y_c, w, h$ y una confianza), y $C = 20$ probabilidades de clase (las de PASCAL VOC). Toda la salida es un tensor de $7 \times 7 \times (2 \cdot 5 + 20) = 7 \times 7 \times 30$ que **una sola red convolucional** produce de **una sola pasada**, y se entrena de punta a punta con una pérdida cuadrática sobre todos esos números. Corría a **45 imágenes por segundo** en una GPU Titan X (155 la versión pequeña, *Fast YOLO*), con 63.4 % de mAP en VOC 2007: menos exacta que Faster R-CNN (73.2 %), sobre todo en objetos chicos y amontonados (una celda solo puede tener dos cajas), pero en tiempo real.

# %%
S = 7
im = fotos["matamoros"]
fig, ax = plt.subplots(figsize=(7.5, 5.6))
ax.imshow(im)
for k in range(1, S):
    ax.axvline(k * im.width / S, color="white", lw=1)
    ax.axhline(k * im.height / S, color="white", lw=1)
ax.set_title(f"La rejilla de {S} × {S} de YOLO: {S * S} celdas, cada una con 2 cajas y 20 clases", fontsize=11)
ax.axis("off")
plt.show()
print("números que predice YOLO (2016) por imagen:", S * S * (2 * 5 + 20))

# %% [markdown]
# ### De YOLO a YOLO26
#
# Desde entonces la familia creció muchísimo (y no todas las versiones son de los mismos autores):
#
# | Año | Versión | Qué cambió |
# |---|---|---|
# | 2017 | YOLOv2 (Redmon y Farhadi) | anclas, como Faster R-CNN; normalización por lotes |
# | 2018 | YOLOv3 (Redmon y Farhadi) | predicciones en tres escalas, para objetos chicos |
# | 2020 | YOLOv4 (Bochkovskiy et al.) | muchas mejoras de entrenamiento y aumento de datos |
# | junio de 2020 | YOLOv5 (Ultralytics) | implementación en PyTorch, fácil de usar y de ajustar |
# | enero de 2023 | YOLOv8 (Ultralytics) | sin anclas; detección, segmentación y pose en una sola biblioteca |
# | septiembre de 2024 | YOLO11 (Ultralytics) | menos parámetros con igual o mejor mAP |
# | enero de 2026 | **YOLO26** (Ultralytics) | de punta a punta, **sin NMS**; pensado para correr en CPU y dispositivos chicos |
#
# Según la documentación de Ultralytics (<https://docs.ultralytics.com/models/yolo26/>), YOLO26 quita la *distribution focal loss* del cabezal, se entrena con un optimizador nuevo (MuSGD, híbrido de SGD y Muon) y tiene un cabezal «uno a uno» que aprende a dar **una sola caja por objeto**, de modo que no necesita la supresión de no máximos de la sección siguiente. Su versión más chica, **YOLO26n** (*nano*), tiene 2.4 millones de parámetros y 40.9 de mAP50-95 en COCO, con 38.9 ms por imagen de 640 pixeles en CPU (con ONNX). La versión más grande, YOLO26x, llega a 57.5.
#
# **Licencia.** El código y los pesos de Ultralytics se distribuyen con licencia **AGPL-3.0**: puedes usarlos, estudiarlos y modificarlos libremente, pero si distribuyes un programa que los incorpore, o lo ofreces como servicio en red, tienes que publicar su código fuente con la misma licencia; para productos cerrados, Ultralytics vende una licencia empresarial.

# %% [markdown]
# ## 5. Supresión de no máximos (NMS)
#
# Un detector que mira muchas posiciones encuentra el mismo auto **muchas veces**: las ventanas vecinas, las anclas de tamaños parecidos o las celdas contiguas de la rejilla lo ven casi igual. Hay que quedarse con una caja por objeto. El algoritmo clásico es la **supresión de no máximos** (*non-maximum suppression*, NMS), y es voraz:
#
# 1. ordenar las cajas de mayor a menor confianza;
# 2. tomar la de mayor confianza y **conservarla**;
# 3. **eliminar** todas las restantes cuyo IoU con ella supere un umbral (por ejemplo 0.7);
# 4. repetir con las que quedan.
#
# Programémoslo y comparémoslo con `torchvision.ops.nms`:

# %%
def nms_propia(cajas, puntajes, umbral):
    """Índices de las cajas que sobreviven a la supresión de no máximos, en orden de confianza."""
    orden = torch.argsort(puntajes, descending=True)
    conservar = []
    while orden.numel() > 0:
        i = orden[0]
        conservar.append(int(i))
        resto = orden[1:]
        ious = iou_matriz(cajas[i][None], cajas[resto])[0]
        orden = resto[ious <= umbral]          # se eliminan las que se traslapan demasiado con la elegida
    return torch.tensor(conservar, dtype=torch.long)


def nms_por_clase(cajas, puntajes, clases, umbral):
    """NMS por separado en cada clase: un auto no suprime a la persona que está enfrente."""
    conservar = torch.cat([torch.where(clases == c)[0][nms_propia(cajas[clases == c], puntajes[clases == c], umbral)]
                           for c in clases.unique()])
    return conservar[torch.argsort(puntajes[conservar], descending=True)]


# verificación con cajas al azar y tres clases
p_az = torch.rand(200, generator=g)
c_az = torch.randint(0, 3, (200,), generator=g)
print("¿igual a torchvision.ops.nms?        ", torch.equal(nms_propia(A_az, p_az, 0.5), nms(A_az, p_az, 0.5)))
print("¿igual a torchvision.ops.batched_nms?", torch.equal(nms_por_clase(A_az, p_az, c_az, 0.5), batched_nms(A_az, p_az, c_az, 0.5)))

# %% [markdown]
# Ahora con un detector de verdad. Cargamos YOLO26n (la primera vez se descarga) y le pedimos sus cajas **crudas**: todas las que superan una confianza de 0.25, con el umbral de IoU de su NMS en 1.0, que en la práctica la apaga (ningún par de cajas distintas tiene IoU mayor que 1).

# %%
# Las «tres líneas» de la sección 6 son YOLO("yolo26n.pt"); aquí guardamos los pesos en la carpeta de datos del curso
modelo = YOLO(os.path.join(RAIZ_DATOS, "yolo26n.pt"))
NOMBRES = modelo.names                       # {0: 'person', 1: 'bicycle', 2: 'car', ...}, las 80 clases de COCO
# nombres en español de las clases de COCO que aparecen en nuestras fotos, y colores para dibujarlas
ES = {"person": "persona", "bicycle": "bicicleta", "car": "auto", "motorcycle": "motocicleta", "bus": "autobús",
      "truck": "camioneta", "traffic light": "semáforo", "stop sign": "señal de alto", "bench": "banca",
      "fire hydrant": "hidrante", "parking meter": "parquímetro", "dog": "perro", "bird": "pájaro",
      "umbrella": "paraguas", "handbag": "bolso", "backpack": "mochila", "potted plant": "maceta",
      "clock": "reloj", "chair": "silla", "train": "tren", "suitcase": "maleta", "cell phone": "celular"}
COLOR = {"car": AZUL, "person": ROJO, "truck": DORADO_OSC, "bus": MARINO}

FOTO_NMS = "matamoros"
CONF_CRUDAS, UMBRAL_NMS = 0.25, 0.7
r_crudas = modelo(fotos[FOTO_NMS], conf=CONF_CRUDAS, iou=1.0, verbose=False)[0]
cajas_crudas = r_crudas.boxes.xyxy
conf_crudas = r_crudas.boxes.conf
clases_crudas = r_crudas.boxes.cls.long()
print(f"cajas crudas con confianza ≥ {CONF_CRUDAS}: {len(cajas_crudas)}")
print("por clase:", dict(Counter(NOMBRES[int(c)] for c in clases_crudas)))

quedan = nms_por_clase(cajas_crudas, conf_crudas, clases_crudas, UMBRAL_NMS)
r_normal = modelo(fotos[FOTO_NMS], conf=CONF_CRUDAS, verbose=False)[0]     # con su NMS (umbral 0.7 por omisión)
print(f"después de nuestra NMS (IoU > {UMBRAL_NMS}): {len(quedan)} cajas")
print(f"YOLO con su propia NMS:                   {len(r_normal.boxes)} cajas")
print("¿las mismas cajas?", torch.allclose(cajas_crudas[quedan].sort(0).values, r_normal.boxes.xyxy.sort(0).values, atol=1e-3))

# ¿a qué caja conservada «pertenece» cada caja cruda? A la de su misma clase con la que tiene mayor IoU.
# (Toda caja eliminada tiene IoU > 0.7 con alguna conservada de su clase: fue la que la eliminó.)
iou_con_quedan = iou_matriz(cajas_crudas, cajas_crudas[quedan])
iou_con_quedan[clases_crudas[:, None] != clases_crudas[quedan][None, :]] = -1
grupo_crudas = iou_con_quedan.argmax(dim=1)
por_grupo = torch.bincount(grupo_crudas, minlength=len(quedan))
print("cajas crudas detrás de cada caja conservada:",
      [f"{ES.get(NOMBRES[int(clases_crudas[q])], NOMBRES[int(clases_crudas[q])])} {int(n)}" for q, n in zip(quedan, por_grupo)])

# %%
def dibujar(ax, im, cajas, clases, confs=None, ancho_linea=2, etiquetas=True):
    """Dibuja cajas (x1, y1, x2, y2) sobre una imagen; `clases` son nombres en inglés de COCO."""
    ax.imshow(im)
    for k, (c, nombre) in enumerate(zip(np.asarray(cajas), clases)):
        color = COLOR.get(nombre, GRIS)
        ax.add_patch(Rectangle((c[0], c[1]), c[2] - c[0], c[3] - c[1], fill=False, ec=color, lw=ancho_linea))
        if etiquetas:
            texto = ES.get(nombre, nombre) + ("" if confs is None else f" {float(confs[k]):.2f}")
            ax.text(c[0], c[1] - 3, texto, fontsize=8, color="white", va="bottom",
                    bbox=dict(boxstyle="square,pad=0.15", fc=color, ec="none"))
    ax.axis("off")


fig, ejes = plt.subplots(1, 2, figsize=(14, 5.4))
nombres_crudas = [NOMBRES[int(c)] for c in clases_crudas]
dibujar(ejes[0], fotos[FOTO_NMS], cajas_crudas, nombres_crudas, ancho_linea=1, etiquetas=False)
ejes[0].set_title(f"Antes: {len(cajas_crudas)} cajas con confianza ≥ {CONF_CRUDAS}", fontsize=11)
dibujar(ejes[1], fotos[FOTO_NMS], cajas_crudas[quedan], [nombres_crudas[i] for i in quedan], conf_crudas[quedan])
ejes[1].set_title(f"Después de la NMS: {len(quedan)} cajas", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# Las 82 cajas crudas se apilan sobre los mismos objetos, casi encimadas (por eso a la izquierda parecen pocas): el autobús aparece 7 veces, y hay autos que aparecen 10 o 12 veces. Después de la NMS quedan 11, una por objeto, y son exactamente las que da Ultralytics con su NMS. El umbral es un hiperparámetro con un costo en cada dirección: muy bajo, y dos personas que caminan juntas se vuelven una sola detección; muy alto, y sobreviven duplicados.
#
# YOLO26 tiene además un segundo cabezal, «uno a uno», entrenado para que cada objeto lo prediga **una sola** posición. Con `nms=False` se usa ese cabezal y la NMS desaparece del todo:

# %%
r_sin_nms = modelo(fotos[FOTO_NMS], conf=CONF_CRUDAS, nms=False, verbose=False)[0]
print(f"cabezal uno a uno, sin NMS: {len(r_sin_nms.boxes)} cajas")
print("por clase:", dict(Counter(NOMBRES[int(c)] for c in r_sin_nms.boxes.cls)))

# %% [markdown]
# Se obtiene prácticamente el mismo resultado sin el paso de NMS, que es secuencial y difícil de acelerar en dispositivos pequeños: esa es la razón de ser de YOLO26.

# %% [markdown]
# ## 6. Lo fácil que es usar YOLO
#
# Con Ultralytics, detectar objetos en una foto son tres líneas:
#
# ```python
# from ultralytics import YOLO
# modelo = YOLO("yolo26n.pt")          # baja los pesos entrenados en COCO la primera vez
# resultados = modelo("mi_foto.jpg")   # una ruta, una URL, una imagen de PIL o un arreglo
# ```
#
# Cada resultado trae las cajas (`boxes.xyxy`, en pixeles de la imagen original), las clases (`boxes.cls`, índices de `modelo.names`) y las confianzas (`boxes.conf`). Por dentro, la imagen se reescala para que su lado mayor mida 640 pixeles, pasa una vez por la red y sale un tensor con $4 + 80 = 84$ números (caja y puntaje de cada clase) para cada una de las 8,400 posiciones de tres rejillas de $80 \times 80$, $40 \times 40$ y $20 \times 20$ (con pasos de 8, 16 y 32 pixeles):

# %%
red = modelo.model.eval()
with torch.no_grad():
    salida = red(torch.zeros(1, 3, 640, 640))[0]
print("salida de la red para una imagen de 640 × 640:", tuple(salida.shape))
print("posiciones:", 80 * 80 + 40 * 40 + 20 * 20)
params_yolo = sum(p.numel() for p in red.parameters())
print(f"parámetros de YOLO26n: {params_yolo:,}")

# %% [markdown]
# Es la idea de YOLO de 2016 con rejillas más finas, varias escalas y sin anclas. Detectemos en las fotos, contemos objetos por clase y midamos el tiempo en CPU (la primera llamada es más lenta porque prepara el modelo, así que la descontamos, y de cinco repeticiones nos quedamos con la más rápida):

# %%
def cronometrar(f, repeticiones):
    """El menor tiempo de pared de f(), en segundos, de varias repeticiones.

    Se toma el menor, no el promedio, porque las demoras vienen de otros programas que comparten la CPU."""
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        f()
        tiempos.append(time.perf_counter() - t0)
    return min(tiempos)


REPETICIONES = 1 if RAPIDO else 5
modelo(fotos[FOTO_NMS], verbose=False)                         # calentamiento
det_yolo, tiempo_yolo = {}, {}
for nombre, im in fotos.items():
    tiempo_yolo[nombre] = cronometrar(lambda: modelo(im, verbose=False), REPETICIONES)
    r = modelo(im, verbose=False)[0]                          # confianza mínima 0.25, la de omisión
    det_yolo[nombre] = {"cajas": r.boxes.xyxy.numpy(), "clases": [NOMBRES[int(c)] for c in r.boxes.cls],
                        "conf": r.boxes.conf.numpy()}
    cuenta = Counter(ES.get(c, c) for c in det_yolo[nombre]["clases"])
    print(f"{nombre:10s} {tiempo_yolo[nombre] * 1000:5.0f} ms   {dict(cuenta.most_common())}")

# %%
fig, ejes = plt.subplots(len(fotos), 1, figsize=(10, 7.2 * len(fotos)), squeeze=False)
for ax, (nombre, im) in zip(ejes[:, 0], fotos.items()):
    d = det_yolo[nombre]
    dibujar(ax, im, d["cajas"], d["clases"], d["conf"])
    ax.set_title(f"YOLO26n: {len(d['clases'])} objetos en {tiempo_yolo[nombre] * 1000:.0f} ms (CPU)", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# En una o dos décimas de segundo por foto, en CPU y sin haber entrenado nada, el modelo encuentra los autos, el autobús, las personas y los semáforos. También se ven sus límites: en la foto de la Universidad deja ir casi todos los autos del fondo y a las personas del parque (la foto se reduce a 640 pixeles de ancho y esos objetos quedan de unos cuantos pixeles), y la clase `truck` de COCO mezcla camiones de carga con camionetas *pick-up*, que aquí traducimos como «camioneta».
#
# ## 7. Contra un detector de dos etapas
#
# torchvision trae varios Faster R-CNN entrenados en COCO. Usamos dos: uno ligero, con una MobileNetV3 como extractor de atributos, y otro grande, con una ResNet-50 (versión 2, de 2021), los dos con una pirámide de atributos (FPN). Medimos el tiempo por imagen en CPU y contamos detecciones con confianza mayor que 0.5 en las mismas fotos. Ojo: los Faster R-CNN de torchvision reescalan la imagen para que su lado **menor** mida 800 pixeles, más que los 640 del lado **mayor** de YOLO, lo que les ayuda con los objetos chicos y les cuesta tiempo.

# %%
from torchvision.models import detection

DETECTORES = {"Faster R-CNN MobileNetV3": (detection.fasterrcnn_mobilenet_v3_large_fpn,
                                           detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)}
if not RAPIDO:
    DETECTORES["Faster R-CNN ResNet-50 v2"] = (detection.fasterrcnn_resnet50_fpn_v2,
                                               detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

UMBRAL_CONF = 0.5
filas, det_frcnn = [], {}
filas += [{"modelo": "YOLO26n", "foto": n, "parámetros": params_yolo, "segundos": tiempo_yolo[n],
           "detecciones > 0.5": int((det_yolo[n]["conf"] > UMBRAL_CONF).sum())} for n in fotos]
for nombre_modelo, (constructor, pesos) in DETECTORES.items():
    frcnn = constructor(weights=pesos).eval()
    categorias = pesos.meta["categories"]                     # 91 índices de COCO, algunos 'N/A'
    n_params = sum(p.numel() for p in frcnn.parameters())
    with torch.no_grad():
        frcnn([torchvision.transforms.functional.to_tensor(fotos[FOTO_NMS])])        # calentamiento
        for n, im in fotos.items():
            x = torchvision.transforms.functional.to_tensor(im)
            seg = cronometrar(lambda: frcnn([x]), 1 if RAPIDO else 2)
            salida_f = frcnn([x])[0]
            buenas = salida_f["scores"] > UMBRAL_CONF
            det_frcnn[(nombre_modelo, n)] = {"cajas": salida_f["boxes"][buenas].numpy(),
                                             "clases": [categorias[int(c)] for c in salida_f["labels"][buenas]],
                                             "conf": salida_f["scores"][buenas].numpy()}
            filas.append({"modelo": nombre_modelo, "foto": n, "parámetros": n_params, "segundos": seg,
                          "detecciones > 0.5": int(buenas.sum())})

comparacion = pd.DataFrame(filas)
resumen = comparacion.groupby("modelo", sort=False).agg(parámetros=("parámetros", "first"),
                                                         segundos_por_imagen=("segundos", "mean"),
                                                         detecciones=("detecciones > 0.5", "sum"))
resumen["mAP50-95 en COCO"] = [40.9, 32.8, 46.7][:len(resumen)]       # documentación de Ultralytics y de torchvision
resumen["veces más lento que YOLO"] = resumen["segundos_por_imagen"] / resumen["segundos_por_imagen"].iloc[0]
print(comparacion.pivot(index="foto", columns="modelo", values="segundos").round(2), "\n")
print(comparacion.pivot(index="foto", columns="modelo", values="detecciones > 0.5"))
resumen.round(3)

# %%
nombre_grande = list(DETECTORES)[-1]
fig, ejes = plt.subplots(1, 2, figsize=(14, 5.4))
d = det_yolo[FOTO_NMS]
alta = d["conf"] > UMBRAL_CONF
dibujar(ejes[0], fotos[FOTO_NMS], d["cajas"][alta], [c for c, a in zip(d["clases"], alta) if a], d["conf"][alta])
ejes[0].set_title(f"YOLO26n: {resumen['segundos_por_imagen'].iloc[0]:.2f} s por imagen", fontsize=11)
d = det_frcnn[(nombre_grande, FOTO_NMS)]
dibujar(ejes[1], fotos[FOTO_NMS], d["cajas"], d["clases"], d["conf"])
ejes[1].set_title(f"{nombre_grande}: {resumen.loc[nombre_grande, 'segundos_por_imagen']:.2f} s por imagen", fontsize=11)
fig.suptitle(f"Detecciones con confianza mayor que {UMBRAL_CONF}", x=0.01, ha="left", color=MARINO, fontweight="semibold")
plt.tight_layout()
plt.show()

# %% [markdown]
# En esta computadora (CPU, 2 hilos), YOLO26n tarda una o dos décimas de segundo por foto; el Faster R-CNN con MobileNetV3, entre 2 y 3 segundos, y el de ResNet-50 v2, más de 10 segundos, con 17 veces más parámetros que YOLO26n. A cambio, el modelo grande encuentra muchas más cosas con confianza mayor que 0.5: 123 detecciones en las tres fotos contra 20 de YOLO26n, sobre todo autos lejanos y personas pequeñas (74 en la foto de la Universidad). No todas son aciertos: en la calle Matamoros llama «tren» al autobús desde el que se tomó la foto. Los tiempos cambian de una computadora a otra, y aun de una corrida a otra si hay otros programas usando la CPU; lo que no cambia es el orden: YOLO26n es decenas de veces más rápido.
#
# Las confianzas de dos modelos distintos no están en la misma escala (cada uno se calibra a su manera), así que contar cajas por arriba de 0.5 es una comparación gruesa. La comparación seria es la mAP en un conjunto etiquetado como COCO, que ya reportan los autores: 40.9 para YOLO26n, 32.8 para el Faster R-CNN con MobileNetV3 y 46.7 para el de ResNet-50 v2 (mAP50-95 en el conjunto de validación de COCO 2017, según la documentación de Ultralytics y de torchvision).

# %% [markdown]
# ## 8. ¿Y si mis objetos no están en COCO?
#
# YOLO26n sabe encontrar las 80 clases de COCO y nada más. Si queremos contar espigas de trigo en una parcela del Valle del Yaqui, señales de tránsito mexicanas o defectos en una pieza, tomamos este mismo modelo y lo **ajustamos**: se cambia la última capa del cabezal para que prediga nuestras clases y se continúa el entrenamiento con unos cientos o miles de fotos etiquetadas con cajas, aprovechando todo lo que la red ya aprendió en COCO (bordes, texturas, partes de objetos). Con Ultralytics es un archivo `data.yaml` que dice dónde están las imágenes y cuáles son las clases, y una llamada `modelo.train(data="data.yaml", epochs=...)`. Lo hacemos con espigas de trigo del conjunto *Global Wheat Head Detection* en la libreta del **ejemplo 6** (necesita una GPU, así que se corre en Colab).
#
# ## 9. Una nota sobre los usos
#
# Detectar personas en una calle es la pieza básica tanto de un sistema que avisa a un auto que alguien cruza como de uno de vigilancia masiva. Los detectores también heredan los sesgos de sus datos: B. Wilson, J. Hoffman y J. Morgenstern («Predictive inequity in object detection», 2019) encontraron que varios detectores de peatones entrenados en conjuntos públicos fallaban más con personas de piel oscura. Antes de poner un detector en la calle hay que preguntarse quién aparece en sus datos de entrenamiento, evaluarlo por separado en cada grupo relevante y pensar quién gana y quién pierde con su uso.
#
# ## 10. Cierre
#
# - Detectar = clasificar + hacer regresión de una caja + decidir cuántos objetos hay.
# - La IoU, $|A \cap B| / |A \cup B|$, compara cajas; con un umbral (0.5) define qué es un acierto, y la mAP resume la curva de precisión contra exhaustividad de cada clase (mAP50-95 promedia diez umbrales).
# - La ventana deslizante clasifica decenas de miles de recortes; R-CNN usó 2,000 propuestas y atributos de una red de ImageNet (transferencia); Fast y Faster R-CNN compartieron el cómputo; YOLO lo hizo en una sola pasada.
# - La supresión de no máximos elimina las detecciones repetidas; YOLO26 aprende a no producirlas.
# - Usar un detector moderno preentrenado son tres líneas y una o dos décimas de segundo por imagen en CPU; un detector de dos etapas grande encuentra más objetos chicos, pero tarda decenas de veces más.
#
# *Referencias:* P. Viola y M. Jones, «Rapid object detection using a boosted cascade of simple features», CVPR (2001). N. Dalal y B. Triggs, «Histograms of oriented gradients for human detection», CVPR (2005). R. Girshick, J. Donahue, T. Darrell y J. Malik, «Rich feature hierarchies for accurate object detection and semantic segmentation», CVPR (2014). R. Girshick, «Fast R-CNN», ICCV (2015). S. Ren, K. He, R. Girshick y J. Sun, «Faster R-CNN: Towards real-time object detection with region proposal networks», NeurIPS (2015). J. Redmon, S. Divvala, R. Girshick y A. Farhadi, «You Only Look Once: Unified, real-time object detection», CVPR (2016). T.-Y. Lin et al., «Microsoft COCO: Common objects in context», ECCV (2014). Documentación de Ultralytics YOLO26: <https://docs.ultralytics.com/models/yolo26/>. Detectores de torchvision: <https://pytorch.org/vision/stable/models.html#object-detection>.

# %% [markdown]
# ## 11. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p24", "u06-p25", "u06-p26"])

# %% [markdown]
# ## 12. Ejercicios
#
# 1. **IoU a mano.** Calcula la IoU de las cajas $(0, 0, 4, 4)$ y $(2, 2, 6, 6)$, y la de $(0, 0, 4, 4)$ y $(0, 0, 4, 2)$. Compruébalo con `iou_matriz`. Si una caja predicha tiene el mismo centro que la verdadera pero el doble de ancho y de alto, ¿cuánto vale su IoU? ¿Y si tiene la mitad de ancho y de alto?
# 2. **El umbral de la NMS.** Repite la sección 5 con umbrales de IoU de 0.3, 0.5, 0.7 y 0.9 y cuenta las cajas que sobreviven en cada caso. Busca en la foto un caso en que un umbral bajo elimine un objeto verdadero (dos autos que se tapan) y uno en que un umbral alto deje un duplicado.
# 3. **Tamaño de la imagen.** Llama a `modelo(foto, imgsz=1280)` en la foto de la Universidad. ¿Cuántos autos más encuentra? ¿Cuánto más tarda? Explica por qué, pensando en el tamaño en pixeles de los autos lejanos después de reducir la foto a 640.
# 4. **Ventanas deslizantes.** Usa `contar_ventanas` para una foto de 1920 × 1080 con la ventana de 64 × 128, y luego con tres formas de ventana (64 × 128, 128 × 64 y 96 × 96). ¿Cuántas ventanas hay? Compara con las 8,400 posiciones que evalúa YOLO26 en una sola pasada (para una imagen de 640 × 640).
# 5. **Tu propia foto.** Toma una foto de tu calle, de tu salón o del estacionamiento de la universidad y pásala por YOLO26n y por el Faster R-CNN grande. ¿Qué objetos encuentra cada uno y cuáles se le escapan? ¿Qué objetos importantes de tu foto no están entre las 80 clases de COCO?
