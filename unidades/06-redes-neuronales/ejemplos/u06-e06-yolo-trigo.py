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
# # Ajuste fino de YOLO: contar espigas de trigo
#
# **Unidad 6 · Ejemplo 6**
#
# Un detector como YOLO, entrenado con las 80 clases del conjunto COCO, encuentra personas, coches, perros y tazas. Pero no sabe qué es una **espiga de trigo**, y en el campo eso es justamente lo que interesa contar. En esta libreta tomamos ese detector ya entrenado y lo **ajustamos** a un problema nuevo con unas pocas miles de fotos etiquetadas. Vamos a:
#
# 1. bajar el conjunto *Global Wheat Head Detection 2020* y entender el **formato YOLO** de las etiquetas;
# 2. armar una partición entrenamiento/validación/prueba **sin fuga de información**, separando por sitio y por país;
# 3. ver que el YOLO preentrenado, tal cual, no sirve para este problema;
# 4. hacer el **ajuste fino** con Ultralytics y leer sus curvas y métricas (mAP, precisión, exhaustividad);
# 5. medir qué tan bien **cuenta** espigas en fotos de un país que no vio al entrenar;
# 6. guardar el modelo, recargarlo, usarlo en fotos nuevas y exportarlo.

# %% [markdown]
# > **Esta libreta se corre en Google Colab con GPU.** No la vemos en clase porque el ajuste fino tarda.
# >
# > 1. Ábrela en Colab con el botón de la página del curso.
# > 2. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador de hardware: GPU T4** → Guardar.
# > 3. **Entorno de ejecución → Ejecutar todas.**
# >
# > **Cuánto tarda y cuánto baja.** El conjunto de datos pesa **7 GB** (un archivo zip de 6.9 GB en Zenodo, más 3 MB de etiquetas); bajarlo y descomprimirlo en Colab toma de 10 a 20 minutos, según el día. El ajuste fino toma de 30 a 60 minutos en una T4. En total, calcula **una hora a hora y media** con la pestaña abierta: la cuota gratuita de Colab alcanza, pero si cierras la pestaña o te desconectas, el entorno se borra y hay que empezar de nuevo. Al final, la libreta escribe un archivo pequeño `resultados_yolo_trigo.json` con las cifras importantes: descárgalo antes de cerrar.
# >
# > Sin GPU (por ejemplo, en tu computadora) la libreta funciona, pero el ajuste fino tardaría muchas horas. Para revisar que todo el código corre, define la variable de entorno `RDP_RAPIDO=1`: la libreta fabrica unas cuantas imágenes sintéticas de «trigo» y entrena una sola época en unos minutos, sin descargar nada grande. Las cifras de ese modo no significan nada.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso y **Ultralytics**, la biblioteca que implementa YOLO (Colab no la trae). PyTorch ya viene instalado.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    # %pip install -q ultralytics
    pass

# %%
import json, shutil, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFilter

import torch
import ultralytics
from ultralytics import YOLO

import rdp

rdp.estilo.usar()
SEMILLA = 2020
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# la verificación automática del repositorio usa un conjunto sintético diminuto y una época
RAPIDO = os.environ.get("RDP_RAPIDO") == "1"
RAIZ_DATOS = os.path.expanduser("~/.cache/rdp-datos")
TRABAJO = Path(RAIZ_DATOS) / ("yolo-trigo-rapido" if RAPIDO else "yolo-trigo")  # pesos, corridas y gráficas
TRABAJO.mkdir(parents=True, exist_ok=True)
# si MLflow está instalado, Ultralytics registra ahí cada entrenamiento: que lo haga en la carpeta de trabajo
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{TRABAJO / 'mlflow.db'}")
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU = torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else "ninguna (CPU)"
T_INICIO = time.time()

print("Ultralytics", ultralytics.__version__, "· PyTorch", torch.__version__)
print("dispositivo:", dispositivo, "·", GPU)
if dispositivo.type == "cpu" and not RAPIDO:
    print("\n¡ATENCIÓN! No hay GPU. En Colab: Entorno de ejecución → Cambiar tipo de entorno → GPU T4.\n"
          "En CPU el ajuste fino completo tarda muchas horas.")

# %% [markdown]
# ## 1. El problema: contar espigas
#
# **Por qué nos importa en Sonora.** Sonora es el estado más importante de México en trigo: en 2022 se sembró aquí el 47.5 % de la superficie nacional de trigo grano (SIAP, *Panorama agroalimentario de Sonora 2023*, con el cierre agrícola de 2022). En el Valle del Yaqui, cerca de Ciudad Obregón, está el Campo Experimental Norman E. Borlaug del CIMMYT (Centro Internacional de Mejoramiento de Maíz y Trigo). Ahí, en los años cuarenta y cincuenta, Norman Borlaug, investigadores mexicanos y productores del valle desarrollaron las variedades de trigo de alto rendimiento y resistentes a enfermedades que hicieron a México autosuficiente en trigo y se sembraron después en la India y Pakistán: la llamada Revolución Verde, por la que Borlaug recibió el Premio Nobel de la Paz en 1970.
#
# **Por qué contar espigas.** El rendimiento de un campo de trigo (toneladas por hectárea) se descompone en factores: cuántas espigas hay por metro cuadrado, cuántos granos tiene cada espiga y cuánto pesa cada grano. La **densidad de espigas** es el primero, y en los programas de mejoramiento genético todavía se cuenta a mano, parcela por parcela (David et al., 2020). Con una cámara que mira hacia abajo desde una altura conocida, cada foto cubre un área conocida del suelo; si un detector encuentra las espigas, contarlas es contar cajas.
#
# **Por qué es difícil.** Las espigas se enciman, se mueven con el viento, cambian de color al madurar (verdes, amarillas, doradas), pueden tener barbas o no, y cada campo tiene otra luz, otro suelo, otra cámara y otras variedades. Un detector que funciona en Francia puede fallar en Japón: el **cambio de dominio** será un tema central de esta libreta.

# %% [markdown]
# ## 2. Los datos: Global Wheat Head Detection 2020
#
# **Datos.** El *Global Wheat Head Detection dataset* (GWHD; E. David, S. Madec y otros 15 autores, «Global Wheat Head Detection (GWHD) dataset: a large and diverse dataset of high resolution RGB labelled images to develop and benchmark wheat head detection methods», *Plant Phenomics*, 2020, [doi:10.34133/2020/3521852](https://doi.org/10.34133/2020/3521852)) reúne fotos de parcelas de trigo tomadas con la cámara hacia abajo, a entre 1.8 y 3 m del suelo, por nueve instituciones en 2016–2019. Las fotos originales (2,219) se reescalaron para que todas tuvieran una resolución parecida en el suelo (unos 0.4 mm por pixel) y se cortaron en **4,698 cuadros de 1024 × 1024 pixeles** con **188,445 espigas** marcadas con una caja cada una, unas 40 por imagen en promedio. Hay una sola clase: espiga.
#
# Las imágenes vienen agrupadas en **11 subconjuntos**, uno por sitio y campaña (los datos son de la Tabla 3 del artículo):
#
# | subconjunto | institución | país | imágenes | espigas |
# |---|---|---|---:|---:|
# | `arvalis_1`, `arvalis_2`, `arvalis_3` | Arvalis | Francia | 1,055 · 204 · 608 | 45,716 · 4,179 · 16,665 |
# | `inrae_1` | INRAE | Francia | 176 | 3,701 |
# | `usask_1` | Universidad de Saskatchewan | Canadá | 200 | 5,737 |
# | `rres_1` | Rothamsted Research | Reino Unido | 432 | 20,236 |
# | `ethz_1` | ETH Zúrich | Suiza | 747 | 51,489 |
# | `utokyo_1`, `utokyo_2` | NARO y Universidad de Tokio | Japón | 994 · 120 | 29,174 · 3,263 |
# | `nau_1` | Universidad Agrícola de Nanjing | China | 20 | 1,250 |
# | `uq_1` | Universidad de Queensland | Australia | 142 | 7,035 |
#
# Usamos la versión oficial del concurso de 2020, publicada en Zenodo (registro [4298502](https://zenodo.org/records/4298502)) con **licencia MIT**. En esa versión, las etiquetas de los cuatro subconjuntos de Japón, China y Australia (1,276 imágenes, el conjunto de prueba del concurso) **no son públicas**: hay que pedirlas a los autores. Ultralytics distribuye las etiquetas de los otros siete ya convertidas a su formato (archivo `GlobalWheat2020_labels.zip`, licencia AGPL-3.0 de Ultralytics), y es lo que usamos aquí.
#
# **Problema.** Dada una foto de 1024 × 1024 de una parcela, encontrar cada espiga con una caja; de ahí, contar cuántas hay.

# %%
DOMINIOS = ["arvalis_1", "arvalis_2", "arvalis_3", "ethz_1", "rres_1", "inrae_1", "usask_1",
            "utokyo_1", "utokyo_2", "nau_1", "uq_1"]
PAIS = {"arvalis_1": "Francia", "arvalis_2": "Francia", "arvalis_3": "Francia", "inrae_1": "Francia",
        "usask_1": "Canadá", "rres_1": "Reino Unido", "ethz_1": "Suiza", "utokyo_1": "Japón",
        "utokyo_2": "Japón", "nau_1": "China", "uq_1": "Australia"}
URL_IMAGENES = "https://zenodo.org/records/4298502/files/global-wheat-codalab-official.zip"  # 6.9 GB
URL_ETIQUETAS = "https://github.com/ultralytics/assets/releases/download/v0.0.0/GlobalWheat2020_labels.zip"  # 3 MB


def trigo_sintetico(raiz, por_dominio=3, tam=320, semilla=SEMILLA):
    """Conjunto diminuto con la MISMA estructura de carpetas que Global Wheat, para el modo rápido.

    Cada imagen es un fondo de tierra y hojas con elipses doradas («espigas») giradas al azar,
    y su archivo de etiquetas en formato YOLO. Solo sirve para comprobar que el código corre.
    """
    rng = np.random.default_rng(semilla)
    for d_i, dom in enumerate(DOMINIOS):
        (raiz / "images" / dom).mkdir(parents=True, exist_ok=True)
        (raiz / "labels" / dom).mkdir(parents=True, exist_ok=True)
        tono = rng.uniform(0.6, 1.1)  # cada «sitio» tiene otra luz
        for k in range(por_dominio if dom in DOMINIOS[:7] else 2):
            fondo = rng.normal(0, 1, (tam // 8, tam // 8, 3)) * 12 + np.array([95, 85, 55]) * tono
            img = Image.fromarray(np.clip(fondo, 0, 255).astype(np.uint8)).resize((tam, tam), Image.BICUBIC)
            dib = ImageDraw.Draw(img)
            for _ in range(rng.integers(8, 20)):  # hojas: líneas verdes que no son espigas
                x, y, a = rng.uniform(0, tam), rng.uniform(0, tam), rng.uniform(0, np.pi)
                L = rng.uniform(0.2, 0.5) * tam
                dib.line([(x, y), (x + L * np.cos(a), y + L * np.sin(a))], fill=(60, 110, 50), width=3)
            lineas = []
            for _ in range(rng.integers(3, 25)):
                cx, cy = rng.uniform(0.08, 0.92, 2) * tam
                L, W, a = rng.uniform(0.08, 0.16) * tam, rng.uniform(0.025, 0.04) * tam, rng.uniform(0, np.pi)
                t = np.linspace(0, 2 * np.pi, 24)
                px = cx + L / 2 * np.cos(t) * np.cos(a) - W / 2 * np.sin(t) * np.sin(a)
                py = cy + L / 2 * np.cos(t) * np.sin(a) + W / 2 * np.sin(t) * np.cos(a)
                color = tuple(int(c) for c in np.array([205, 170, 80]) * rng.uniform(0.8, 1.1))
                dib.polygon(list(zip(px, py)), fill=color, outline=(120, 95, 40))
                x0, x1 = np.clip([px.min(), px.max()], 0, tam)
                y0, y1 = np.clip([py.min(), py.max()], 0, tam)
                lineas.append(f"0 {(x0 + x1) / 2 / tam:.6f} {(y0 + y1) / 2 / tam:.6f} {(x1 - x0) / tam:.6f} {(y1 - y0) / tam:.6f}")
            nombre = f"sint_{d_i:02d}_{k:02d}"
            img.filter(ImageFilter.GaussianBlur(0.6)).save(raiz / "images" / dom / f"{nombre}.png")
            if dom in DOMINIOS[:7]:  # como en el conjunto real, los de prueba no traen etiquetas
                (raiz / "labels" / dom / f"{nombre}.txt").write_text("\n".join(lineas) + "\n")


def bajar_global_wheat(raiz):
    """Descarga Global Wheat 2020 (7 GB) y deja images/<dominio>/ y labels/<dominio>/."""
    from ultralytics.utils.downloads import download

    download([URL_IMAGENES, URL_ETIQUETAS], dir=raiz)  # descarga, descomprime y borra los zip
    origen = raiz / "global-wheat-codalab-official"
    (raiz / "images").mkdir(exist_ok=True)
    for dom in DOMINIOS:
        if (origen / dom).exists():
            (origen / dom).rename(raiz / "images" / dom)
    shutil.rmtree(origen, ignore_errors=True)  # quedan solo las anotaciones .json originales


t0 = time.time()
if RAPIDO:
    DIR_GW = TRABAJO / "trigo-sintetico"
    shutil.rmtree(DIR_GW, ignore_errors=True)
    trigo_sintetico(DIR_GW)
    print("MODO RÁPIDO: conjunto sintético en", DIR_GW)
else:
    DIR_GW = Path(RAIZ_DATOS) / "GlobalWheat2020"
    if not (DIR_GW / "images" / "uq_1").exists():
        bajar_global_wheat(DIR_GW)
T_DESCARGA = time.time() - t0
print(f"datos listos en {T_DESCARGA / 60:.1f} min")

# %% [markdown]
# ### El formato YOLO
#
# Por cada imagen `images/<subconjunto>/<nombre>.png` hay (si tiene espigas) un archivo de texto `labels/<subconjunto>/<nombre>.txt` con **un renglón por objeto**:
#
# $$\texttt{clase}\quad x_c \quad y_c \quad w \quad h$$
#
# donde `clase` es el número de la clase (aquí siempre 0, espiga), $(x_c, y_c)$ es el **centro** de la caja y $w, h$ su ancho y alto, todos **divididos entre el ancho o el alto de la imagen**, de modo que quedan entre 0 y 1. Así la misma etiqueta sirve si la imagen se reescala. Una imagen sin objetos simplemente no tiene archivo (o lo tiene vacío). Veamos uno:

# %%
def etiquetas(img):
    """Cajas de una imagen como arreglo (k, 4) con xc, yc, w, h normalizados (k = 0 si no hay archivo)."""
    txt = Path(str(img).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
    if not txt.exists() or txt.stat().st_size == 0:
        return np.zeros((0, 4))
    return np.loadtxt(txt, ndmin=2)[:, 1:5]


ejemplo = sorted((DIR_GW / "images" / "arvalis_1").glob("*.png"))[0]
txt = DIR_GW / "labels" / "arvalis_1" / (ejemplo.stem + ".txt")
print(txt.relative_to(DIR_GW))
print("".join(txt.read_text().splitlines(keepends=True)[:5]), "...")
print("tamaño de la imagen:", Image.open(ejemplo).size, "· espigas:", len(etiquetas(ejemplo)))

# %% [markdown]
# ### Cuántas espigas y de qué tamaño
#
# Juntamos en una tabla todas las imágenes de los siete subconjuntos con etiquetas: cuántas espigas tiene cada una y el tamaño de sus cajas en pixeles. Como «tamaño típico» de una caja usamos $\sqrt{w\,h}$, la raíz de su área, como en el artículo.

# %%
ETIQUETADOS = DOMINIOS[:7]
TAM_IMG = Image.open(ejemplo).size[0]  # 1024 en el conjunto real
filas, cajas = [], []
for dom in ETIQUETADOS:
    for img in sorted((DIR_GW / "images" / dom).glob("*.png")):
        e = etiquetas(img)
        filas.append({"imagen": img, "subconjunto": dom, "país": PAIS[dom], "espigas": len(e)})
        cajas.append(np.sqrt(e[:, 2] * e[:, 3]) * TAM_IMG)
tabla = pd.DataFrame(filas)
lado = np.concatenate(cajas)
print(tabla.groupby("subconjunto", sort=False)["espigas"].agg(imágenes="size", espigas="sum", promedio="mean").round(1))
print(f"\ntotal: {len(tabla):,} imágenes, {tabla.espigas.sum():,} espigas, "
      f"{(tabla.espigas == 0).sum()} imágenes sin espigas")
print(f"tamaño típico de una caja: mediana {np.median(lado):.0f} px "
      f"(percentiles 5 y 95: {np.percentile(lado, 5):.0f} y {np.percentile(lado, 95):.0f} px)")

# %%
fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
ejes[0].hist(tabla.espigas, bins=40, color=AZUL, edgecolor="white")
ejes[0].set(xlabel="espigas en la imagen", ylabel="imágenes", title="¿Cuántas espigas hay en una foto?")
ejes[1].hist(lado, bins=60, color=DORADO_OSC, edgecolor="white")
ejes[1].axvline(np.median(lado), color=TINTA, ls="--", lw=1)
ejes[1].set(xlabel=r"tamaño típico de la caja, $\sqrt{w\,h}$ (pixeles)", ylabel="cajas", title="¿Qué tan grandes son?")
plt.tight_layout()
plt.show()

# %% [markdown]
# Con las etiquetas de Ultralytics (que son las de la versión corregida de Zenodo y difieren un poco de la tabla del artículo), el conjunto real tiene 3,422 imágenes etiquetadas con 145,411 espigas: 42 por imagen en promedio, con mucha variación entre sitios (de 21 en `arvalis_2` a 66 en `ethz_1`, y hasta 112 en una sola foto). El tamaño típico de una caja tiene mediana de 71 pixeles, y el 90 % de las cajas mide entre 43 y 123. Es decir, **muchos objetos chicos y encimados**: lo más difícil para un detector. Si reducimos la imagen a 640 pixeles para entrenar más rápido, una espiga típica queda de unos $71 \times 640/1024 \approx 44$ pixeles; todavía se ve, pero las más chicas quedan de menos de 27.
#
# Ahora veamos algunas fotos de sitios distintos con sus cajas:

# %%
def dibujar(ax, img, reales=None, predichas=None, titulo=""):
    """Dibuja una imagen con cajas reales (azul, formato YOLO) y predichas (rojo, xyxy en pixeles)."""
    im = Image.open(img).convert("RGB")
    W, H = im.size
    ax.imshow(im)
    if reales is not None:
        for xc, yc, w, h in reales:
            ax.add_patch(Rectangle(((xc - w / 2) * W, (yc - h / 2) * H), w * W, h * H, fill=False, ec=AZUL, lw=1.2))
    if predichas is not None:
        for x0, y0, x1, y1 in predichas:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ROJO, lw=1.2, ls="--"))
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")


rng = np.random.default_rng(SEMILLA)
fig, ejes = plt.subplots(2, 4, figsize=(13, 6.8))
for ax, dom in zip(ejes.flat, ["arvalis_1", "arvalis_2", "arvalis_3", "inrae_1", "usask_1", "rres_1", "ethz_1", "uq_1"]):
    img = rng.choice(sorted((DIR_GW / "images" / dom).glob("*.png")))
    reales = etiquetas(img) if dom in ETIQUETADOS else None
    n = f"{len(reales)} espigas" if reales is not None else "sin etiquetas"
    dibujar(ax, img, reales, titulo=f"{dom} ({PAIS[dom]}) · {n}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Una partición sin fuga de información
#
# Ultralytics trae un archivo de configuración listo, `GlobalWheat2020.yaml`, con esta partición: entrenamiento con los siete subconjuntos etiquetados (3,422 imágenes), validación con `ethz_1` (748) y prueba con los cuatro de Japón, China y Australia. Tiene dos problemas:
#
# 1. **`ethz_1` está en entrenamiento y en validación.** El propio archivo lo advierte en un comentario. El modelo se evalúa con fotos que ya vio: las métricas de validación salen optimistas, y además el entrenamiento guarda como «mejor modelo» el que mejor memorizó esas fotos.
# 2. **La prueba no tiene etiquetas** en la versión pública, así que no se puede medir nada con ella.
#
# Hay una tercera trampa más sutil: cada foto original se cortó en 1 a 6 cuadros de 1024 × 1024. Si repartiéramos los **cuadros** al azar entre entrenamiento y validación, dos pedazos de la misma foto (mismo día, misma luz, mismas plantas) podrían quedar uno de cada lado. Eso también es fuga.
#
# La solución es partir por **subconjunto completo**: todas las fotos de un sitio y campaña quedan del mismo lado. Así:
#
# - **entrenamiento:** `arvalis_1`, `arvalis_3`, `inrae_1` (Francia), `usask_1` (Canadá) y `rres_1` (Reino Unido): 2,471 imágenes con 91,538 espigas;
# - **validación:** `arvalis_2` (Francia, otra campaña de Arvalis, 2019; 204 imágenes, 4,270 espigas): sirve para vigilar el entrenamiento y elegir el mejor modelo. Se parece a lo que el modelo ya vio, pero son otras fotos, otro año y otras parcelas;
# - **prueba:** `ethz_1` (Suiza; 747 imágenes, 49,603 espigas), un país, una cámara y una densidad de siembra que el modelo **nunca vio**. Esta es la medida honesta de qué tan bien generaliza a un campo nuevo, por ejemplo uno de Sonora.
#
# Y las fotos de Japón, China y Australia, sin etiquetas, nos sirven para mirar las predicciones a ojo.
#
# Para Ultralytics, una partición es un archivo YAML que dice dónde están las imágenes (las etiquetas se buscan cambiando `images` por `labels` en la ruta) y cómo se llaman las clases:

# %%
PARTICION = {"train": ["arvalis_1", "arvalis_3", "inrae_1", "usask_1", "rres_1"],
             "val": ["arvalis_2"], "test": ["ethz_1"]}
datos_yaml = TRABAJO / "trigo.yaml"
datos_yaml.write_text(
    f"path: {DIR_GW}\n"
    + "".join(f"{p}: [{', '.join('images/' + d for d in doms)}]\n" for p, doms in PARTICION.items())
    + "names:\n  0: espiga\n"
)
print(datos_yaml.read_text())
tabla["parte"] = tabla.subconjunto.map({d: p for p, doms in PARTICION.items() for d in doms})
print(tabla.groupby("parte", sort=False)["espigas"].agg(imágenes="size", espigas="sum"))

# %% [markdown]
# ## 3. Antes de ajustar: el YOLO de COCO no sabe qué es una espiga
#
# Bajamos YOLO26n («n» de *nano*, el más pequeño de la familia, unos 2.5 millones de parámetros), entrenado por Ultralytics con COCO, y le pedimos que detecte objetos en fotos de trigo. Sus 80 clases son cosas como persona, bicicleta, plátano o brócoli; ninguna es «espiga».

# %%
IMGSZ = 256 if RAPIDO else 640  # lado (en pixeles) al que se reescalan las imágenes
pesos_coco = TRABAJO / "yolo26n.pt"
modelo_coco = YOLO(str(pesos_coco))  # la primera vez lo descarga (unos 5 MB)
print(f"{len(modelo_coco.names)} clases; algunas: {list(modelo_coco.names.values())[:12]} ...")
print(f"parámetros: {sum(p.numel() for p in modelo_coco.model.parameters()):,}")

fotos_prueba = sorted((DIR_GW / "images" / "ethz_1").glob("*.png"))
fig, ejes = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, img in zip(ejes, fotos_prueba[:3]):
    r = modelo_coco.predict(img, imgsz=IMGSZ, conf=0.25, device=dispositivo, verbose=False)[0]
    clases = [r.names[int(c)] for c in r.boxes.cls]
    dibujar(ax, img, etiquetas(img), r.boxes.xyxy.cpu().numpy(),
            titulo=f"{len(etiquetas(img))} espigas · {len(clases)} detecciones")
    print(img.name, "→", pd.Series(clases, dtype=str).value_counts().to_dict() or "nada")
plt.tight_layout()
plt.show()

# %% [markdown]
# En azul, las espigas reales; en rojo punteado, lo que el modelo de COCO cree ver con confianza de al menos 0.25. Con las fotos reales, a 640 pixeles casi nunca detecta nada; si acaso, en alguna foto ve «brócoli» (`broccoli`), la clase de COCO que más se parece a un manojo de espigas. No es que el modelo sea malo: es que nunca le enseñaron qué es una espiga.
#
# Para tener una cifra, le damos ventaja: con `single_cls=True`, Ultralytics cuenta **cualquier** detección como si fuera de la clase «espiga», sin importar qué objeto de COCO creyó ver, y calcula el mAP (la métrica la explicamos en la sección 5) en validación y en prueba.

# %%
def evaluar(modelo, parte, **kw):
    """mAP50, mAP50-95, precisión y exhaustividad de un modelo en una parte de la partición."""
    m = modelo.val(data=str(datos_yaml), split=parte, imgsz=IMGSZ, batch=8, device=dispositivo,
                   project=str(TRABAJO), name=f"val-{parte}", exist_ok=True, plots=False, verbose=False, **kw)
    return {"mAP50": m.box.map50, "mAP50-95": m.box.map, "precisión": m.box.mp, "exhaustividad": m.box.mr}


antes = {parte: evaluar(modelo_coco, parte, single_cls=True) for parte in ["val", "test"]}
pd.DataFrame(antes).T.round(3)

# %% [markdown]
# ## 4. Ajuste fino
#
# ### Qué hace `train`
#
# Un detector YOLO tiene tres partes: un **esqueleto** (*backbone*, capas 0 a 10 de YOLO26) que convierte la imagen en mapas de atributos (bordes, texturas, partes de objetos); un **cuello** (*neck*) que combina esos mapas a tres escalas (para objetos chicos, medianos y grandes); y una **cabeza** (*head*) que, en cada posición de cada mapa, predice una caja y una probabilidad por clase.
#
# Cuando llamamos a `train` con nuestro YAML de una sola clase, Ultralytics construye la misma arquitectura pero con una cabeza de **1 clase** en lugar de 80, y copia del modelo de COCO todos los pesos que encajan: el esqueleto, el cuello y la parte de la cabeza que predice las cajas. Solo la última capa de clasificación (80 salidas) no cabe y empieza al azar. Después entrena **toda** la red con nuestras fotos. Eso es el **ajuste fino**: no empezamos de cero, sino de atributos visuales que ya sirven para reconocer objetos en general.
#
# ### Los parámetros que más importan
#
# - `epochs`: cuántas pasadas completas por las imágenes de entrenamiento. Con un modelo preentrenado bastan decenas; desde cero harían falta cientos.
# - `imgsz`: el lado al que se reescala cada imagen. Con **640**, cada imagen de 1024 se reduce y una espiga típica queda de unos 44 pixeles; con **1024** se conserva la resolución original y se detectan mejor las espigas chicas, pero cada época tarda unas $(1024/640)^2 \approx 2.6$ veces más y usa 2.6 veces más memoria. Elegimos 640 para caber en el tiempo; el ejercicio 2 propone probar 1024.
# - `batch`: cuántas imágenes por paso de gradiente; 16 cabe de sobra en los 15 GB de una T4 con 640.
# - `freeze`: congela las primeras capas (no se actualizan sus pesos). `freeze=10` congela casi todo el esqueleto y entrena solo lo demás: es más rápido y sirve cuando hay pocos datos, pero con miles de imágenes suele convenir ajustar todo (no congelamos nada).
# - **Aumento de datos.** En cada paso, Ultralytics transforma las imágenes al azar para que el modelo no memorice. El más característico es el **mosaico** (`mosaic=1.0`): pega cuatro imágenes de entrenamiento en una, recortadas y reescaladas, de modo que el modelo ve espigas en muchas escalas y contextos. Además voltea horizontalmente (`fliplr=0.5`), cambia tono, saturación y brillo (`hsv_h`, `hsv_s`, `hsv_v`) y escala y traslada (`scale`, `translate`). En las últimas `close_mosaic=10` épocas apaga el mosaico, para terminar entrenando con imágenes que se parecen a las reales. Como las fotos se toman desde arriba, voltear de arriba abajo también tiene sentido (`flipud=0.5`): una espiga vista desde arriba no tiene «arriba».
# - `patience`: si el mAP de validación no mejora en ese número de épocas, se detiene antes.
# - `cache="ram"`: guarda las imágenes ya decodificadas y reescaladas en memoria, para que leer los PNG de 2 MB no sea el cuello de botella (con 640 son unos 3 GB; Colab tiene 12).
#
# Con estos valores, en una T4 cada época sobre las 2,471 imágenes de entrenamiento debería tardar menos de un minuto, y las 30 épocas, de 20 a 40 minutos. Es una estimación: la celda imprime el tiempo real, y si Colab te asigna otra GPU o el disco va lento, puede variar.

# %%
EPOCAS = 1 if RAPIDO else 30
LOTE = 4 if RAPIDO else 16

modelo = YOLO(str(pesos_coco))
t0 = time.time()
modelo.train(
    data=str(datos_yaml),
    epochs=EPOCAS,
    imgsz=IMGSZ,
    batch=LOTE,
    device=dispositivo,
    workers=0 if RAPIDO else 2,          # Colab tiene 2 núcleos de CPU
    cache=False if RAPIDO else "ram",
    patience=10,
    flipud=0.5,                          # fotos cenitales: arriba y abajo da igual
    seed=SEMILLA,
    project=str(TRABAJO),
    name="ajuste-fino",
    exist_ok=True,
    plots=not RAPIDO,
    verbose=False,
)
T_ENTRENAMIENTO = time.time() - t0
CARPETA = Path(modelo.trainer.save_dir)
print(f"entrenamiento: {T_ENTRENAMIENTO / 60:.1f} min · resultados en {CARPETA}")

# %% [markdown]
# ## 5. ¿Qué tan bien quedó?
#
# ### Curvas de entrenamiento
#
# En cada época, Ultralytics escribe un renglón en `results.csv` con las pérdidas de entrenamiento y de validación y las métricas de validación. La pérdida de caja (`box_loss`) mide qué tan bien coinciden las cajas predichas con las reales; la de clase (`cls_loss`), qué tan bien distingue espiga de fondo.

# %%
historia = pd.read_csv(CARPETA / "results.csv")
historia.columns = historia.columns.str.strip()
print(list(historia.columns))
EPOCAS_HECHAS = int(historia.epoch.max())

fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, perdida in zip(ejes[:2], ["box_loss", "cls_loss"]):
    ax.plot(historia.epoch, historia[f"train/{perdida}"], "o-", color=AZUL, label="entrenamiento")
    ax.plot(historia.epoch, historia[f"val/{perdida}"], "o-", color=ROJO, label="validación")
    ax.set(xlabel="época", title=f"pérdida {perdida}")
    ax.legend()
for col, color in [("metrics/mAP50(B)", AZUL), ("metrics/mAP50-95(B)", DORADO_OSC)]:
    ejes[2].plot(historia.epoch, historia[col], "o-", color=color, label=col[8:-3])
ejes[2].set(xlabel="época", title="mAP en validación", ylim=(0, 1))
ejes[2].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Las métricas: precisión, exhaustividad y mAP
#
# Una caja predicha es un **acierto** si se encima lo suficiente con una caja real que no haya sido tomada por otra predicción. El encimamiento se mide con la **intersección sobre la unión** (IoU): el área común de las dos cajas entre el área que cubren juntas; vale 1 si coinciden exactamente y 0 si no se tocan.
#
# - **Precisión:** de las cajas que predijo el modelo, qué fracción son espigas de verdad.
# - **Exhaustividad** (*recall*): de las espigas reales, qué fracción encontró el modelo.
#
# Las dos dependen del umbral de confianza: con un umbral bajo, el modelo reporta más cajas (más exhaustividad, menos precisión). Recorriendo todos los umbrales se traza la curva precisión–exhaustividad, y el área bajo ella es la **precisión promedio** (AP). Con varias clases se promedia sobre ellas (*mean* AP, **mAP**); aquí hay una sola clase y mAP = AP.
#
# - **mAP50:** se cuenta como acierto con IoU ≥ 0.5.
# - **mAP50-95:** promedio del mAP con umbrales de IoU de 0.50, 0.55, …, 0.95; premia cajas más ajustadas. Es la métrica principal de COCO.
#
# Evaluamos el mejor modelo (el de mayor mAP en validación, que Ultralytics guarda como `best.pt`) en validación (Francia, otra campaña) y en prueba (Suiza, país nuevo), y lo comparamos con el modelo de COCO sin ajustar:

# %%
despues = {parte: evaluar(modelo, parte) for parte in ["val", "test"]}
NOMBRE_PARTE = {"val": "validación (Francia)", "test": "prueba (Suiza)"}
comparacion = pd.DataFrame(
    {f"{cuando} · {NOMBRE_PARTE[parte]}": d[parte]
     for cuando, d in [("antes", antes), ("después", despues)] for parte in ["val", "test"]}
).T.round(3)
comparacion

# %% [markdown]
# Dos lecturas. Primero, el salto de «antes» a «después»: con unos minutos de ajuste fino, el mismo detector pasa de no servir a encontrar espigas. Segundo, la diferencia entre validación y prueba: en un país, una cámara y una densidad de siembra nuevas (en `ethz_1` hay en promedio 66 espigas por imagen, muy encimadas, contra 37 en las imágenes de entrenamiento), el desempeño baja. Ese **cambio de dominio** es lo que hay que esperar al llevar el modelo a un campo de Sonora: antes de confiar en él, hay que etiquetar unas cuantas fotos locales y medir.

# %% [markdown]
# ### Predicciones dibujadas
#
# Cajas reales en azul, predichas en rojo punteado, en fotos de prueba (Suiza):

# %%
fig, ejes = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, img in zip(ejes, fotos_prueba[:3]):
    r = modelo.predict(img, imgsz=IMGSZ, conf=0.25, device=dispositivo, verbose=False)[0]
    dibujar(ax, img, etiquetas(img), r.boxes.xyxy.cpu().numpy(),
            titulo=f"reales: {len(etiquetas(img))} · predichas: {len(r.boxes)}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Contar espigas
#
# Para contar, basta un número: cuántas cajas tienen confianza mayor que cierto umbral $u$. ¿Qué umbral? Si lo elegimos mirando la prueba, volvemos a tener fuga. Lo elegimos en **validación**: el $u$ que hace mínimo el **error absoluto medio** del conteo,
#
# $$\text{EAM}(u) = \frac{1}{M}\sum_{i=1}^{M} \bigl|\, \hat c_u^{(i)} - c^{(i)} \bigr| ,$$
#
# donde $M$ es el número de imágenes, $c^{(i)}$ el número real de espigas en la imagen $i$ y $\hat c_u^{(i)}$ el número de cajas predichas con confianza de al menos $u$. Luego lo aplicamos, sin tocarlo, a la prueba.

# %%
def confianzas(modelo, parte):
    """Para cada imagen de una parte: número real de espigas y confianzas de las cajas predichas."""
    imgs = tabla[tabla.parte == parte].imagen.tolist()
    reales, confs = [], []
    for img in imgs:
        r = modelo.predict(img, imgsz=IMGSZ, conf=0.05, max_det=300, device=dispositivo, verbose=False)[0]
        reales.append(len(etiquetas(img)))
        confs.append(r.boxes.conf.cpu().numpy())
    return np.array(reales), confs


def conteo(confs, u):
    return np.array([(c >= u).sum() for c in confs])


UMBRALES = np.round(np.arange(0.05, 0.96, 0.025), 3)
reales_val, confs_val = confianzas(modelo, "val")
eam_val = [np.abs(conteo(confs_val, u) - reales_val).mean() for u in UMBRALES]
UMBRAL = float(UMBRALES[int(np.argmin(eam_val))])

reales_pru, confs_pru = confianzas(modelo, "test")
pred_pru = conteo(confs_pru, UMBRAL)
EAM_PRUEBA = float(np.abs(pred_pru - reales_pru).mean())
print(f"umbral elegido en validación: {UMBRAL} (EAM en validación {min(eam_val):.1f} espigas)")
print(f"prueba (Suiza): EAM = {EAM_PRUEBA:.1f} espigas por imagen, "
      f"con {reales_pru.mean():.1f} espigas reales en promedio "
      f"(error relativo {EAM_PRUEBA / max(reales_pru.mean(), 1e-9):.0%})")

fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
ejes[0].plot(UMBRALES, eam_val, color=AZUL)
ejes[0].axvline(UMBRAL, color=TINTA, ls="--", lw=1)
ejes[0].set(xlabel="umbral de confianza $u$", ylabel="EAM del conteo", title="Elegir el umbral (validación)")
tope = max(reales_pru.max(), pred_pru.max()) + 5
ejes[1].plot([0, tope], [0, tope], color=GRIS, lw=1)
ejes[1].scatter(reales_pru, pred_pru, s=14, color=ROJO, alpha=0.6)
ejes[1].set(xlabel="espigas reales", ylabel="espigas predichas", title=f"Prueba (Suiza): EAM = {EAM_PRUEBA:.1f}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Los puntos sobre la diagonal gris son conteos perfectos. Si la nube queda por debajo, el modelo **cuenta de menos**: típico cuando las espigas están muy encimadas y dos se confunden en una caja, o cuando la supresión de no máximos elimina una caja buena porque se encima mucho con otra.
#
# ### Fotos de otros continentes
#
# Las fotos de Japón, China y Australia no tienen etiquetas públicas, pero podemos mirar las predicciones. Fíjate en los cambios de color, de densidad y de iluminación:

# %%
fig, ejes = plt.subplots(1, 4, figsize=(14, 4))
for ax, dom in zip(ejes, ["utokyo_1", "utokyo_2", "nau_1", "uq_1"]):
    img = sorted((DIR_GW / "images" / dom).glob("*.png"))[0]
    r = modelo.predict(img, imgsz=IMGSZ, conf=UMBRAL, device=dispositivo, verbose=False)[0]
    dibujar(ax, img, None, r.boxes.xyxy.cpu().numpy(), titulo=f"{dom} ({PAIS[dom]}) · {len(r.boxes)} espigas")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Guardar, recargar y usar el modelo
#
# Ultralytics ya guardó dos archivos en `ajuste-fino/weights/`: `last.pt` (la última época) y `best.pt` (la de mayor mAP en validación). Cada uno trae la arquitectura, los pesos y los nombres de las clases, así que basta una línea para recargarlo, en esta u otra computadora:

# %%
mejor = CARPETA / "weights" / "best.pt"
print(f"{mejor} · {mejor.stat().st_size / 1e6:.1f} MB")
modelo_trigo = YOLO(str(mejor))
print("clases:", modelo_trigo.names)

foto_nueva = sorted((DIR_GW / "images" / "uq_1").glob("*.png"))[-1]  # o la ruta de una foto tuya
r = modelo_trigo.predict(foto_nueva, imgsz=IMGSZ, conf=UMBRAL, device=dispositivo, verbose=False)[0]
print(f"{foto_nueva.name}: {len(r.boxes)} espigas")
plt.figure(figsize=(6, 6))
plt.imshow(r.plot(labels=False, conf=False, line_width=2)[..., ::-1])  # plot() regresa BGR
plt.axis("off")
plt.show()

# %% [markdown]
# Para probar con una foto tuya en Colab, súbela con el ícono de carpeta del panel izquierdo (o con `from google.colab import files; files.upload()`) y cambia `foto_nueva` por su nombre. Recuerda que el modelo se entrenó con fotos cenitales, tomadas a uno o tres metros de altura: una foto de lado no se parece a nada que haya visto.
#
# **Exportar.** Para usar el modelo fuera de Python (en un teléfono, en un servidor sin PyTorch o en una tarjeta de cómputo en el campo), se exporta a un formato estándar. El más común es **ONNX** (*Open Neural Network Exchange*), que leen ONNX Runtime, OpenCV y muchos otros; hay también TensorRT, CoreML, TFLite, OpenVINO, etc. Es una línea: `modelo_trigo.export(format="onnx")`. En el modo rápido no exportamos, porque Ultralytics tendría que instalar el paquete `onnx`.

# %%
if not RAPIDO:
    archivo_onnx = modelo_trigo.export(format="onnx", imgsz=IMGSZ)
    print("exportado a", archivo_onnx)

# %% [markdown]
# ## 7. Guardar las cifras
#
# La última celda junta las cifras clave en un archivo pequeño, `resultados_yolo_trigo.json`. Si corriste la libreta completa en Colab, descárgalo (panel de archivos de la izquierda, o `files.download("resultados_yolo_trigo.json")`), junto con `best.pt` si quieres conservar el modelo.

# %%
def redondear(d):
    return {k: round(float(v), 4) for k, v in d.items()}


resultados = {
    "modo": "rapido (datos sinteticos, cifras sin significado)" if RAPIDO else "completo",
    "fecha": time.strftime("%Y-%m-%d"),
    "gpu": GPU,
    "ultralytics": ultralytics.__version__,
    "modelo": "yolo26n.pt",
    "imgsz": IMGSZ,
    "lote": LOTE,
    "epocas_pedidas": EPOCAS,
    "epocas_hechas": EPOCAS_HECHAS,
    "particion": {p: {"subconjuntos": doms, "imagenes": int((tabla.parte == p).sum()),
                      "espigas": int(tabla[tabla.parte == p].espigas.sum())} for p, doms in PARTICION.items()},
    "antes": {p: redondear(d) for p, d in antes.items()},
    "despues": {p: redondear(d) for p, d in despues.items()},
    "conteo_prueba": {"umbral": UMBRAL, "eam": round(EAM_PRUEBA, 2),
                      "espigas_reales_promedio": round(float(reales_pru.mean()), 2),
                      "espigas_predichas_promedio": round(float(pred_pru.mean()), 2)},
    "minutos": {"descarga": round(T_DESCARGA / 60, 1), "entrenamiento": round(T_ENTRENAMIENTO / 60, 1),
                "total": round((time.time() - T_INICIO) / 60, 1)},
}
salida = (TRABAJO if RAPIDO else Path.cwd()) / "resultados_yolo_trigo.json"
salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2))
print(salida)
print(json.dumps(resultados, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 8. Autoevaluación
#
# 1. ¿Por qué el YOLO entrenado con COCO no puede detectar espigas, aunque sea un buen detector?
# 2. En una línea de etiquetas `0 0.50 0.25 0.10 0.20` de una imagen de 1024 × 1024, ¿cuáles son las coordenadas en pixeles de las esquinas de la caja?
# 3. ¿Qué capas del modelo cambian al pasar de 80 clases a una? ¿Qué capas se entrenan durante el ajuste fino si no usamos `freeze`?
# 4. ¿Por qué validar con `ethz_1` cuando `ethz_1` también está en entrenamiento infla las métricas? ¿Por qué tampoco basta con repartir al azar los cuadros de 1024 × 1024?
# 5. ¿Qué diferencia hay entre mAP50 y mAP50-95? ¿Cuál es más exigente?
# 6. ¿Por qué el umbral de confianza para contar se eligió en validación y no en prueba?
#
# <details><summary>Respuestas</summary>
#
# 1. Porque su cabeza solo tiene salidas para las 80 clases de COCO; ninguna corresponde a una espiga. Sus atributos internos (bordes, texturas, formas) sí son útiles, y por eso el ajuste fino funciona con pocos datos.
# 2. El centro está en $(512, 256)$ y la caja mide $102.4 \times 204.8$ pixeles, así que las esquinas son $(460.8, 153.6)$ y $(563.2, 358.4)$.
# 3. Solo cambia la capa final de clasificación de la cabeza (de 80 salidas a 1), que empieza al azar. Sin `freeze` se entrenan todas las capas, partiendo de los pesos de COCO.
# 4. Porque el modelo ya vio esas fotos y puede haberlas memorizado: medimos memoria, no generalización, y además se elige como «mejor» el modelo que más memorizó. Repartir cuadros al azar deja pedazos de la misma foto original en entrenamiento y validación, que comparten luz, plantas y cámara.
# 5. mAP50 cuenta como acierto una caja con IoU ≥ 0.5; mAP50-95 promedia sobre umbrales de 0.5 a 0.95, así que exige cajas mucho más ajustadas. Es más exigente mAP50-95.
# 6. Porque el umbral es un hiperparámetro: si lo elegimos viendo la prueba, la prueba deja de ser una medida honesta del error en datos nuevos.
#
# </details>

# %% [markdown]
# ## 9. Ejercicios
#
# 1. **Desde cero.** Entrena con las mismas épocas e `imgsz` pero empezando de pesos al azar: `YOLO("yolo26n.yaml")` en lugar de `YOLO("yolo26n.pt")`. Compara las curvas de mAP en validación y el mAP en prueba. ¿Cuántas épocas más necesitaría el modelo desde cero para alcanzar al ajustado? (Tarda lo mismo que el ajuste fino.)
# 2. **Resolución.** Repite el ajuste fino con `imgsz=1024` (baja `batch` a 8 si se acaba la memoria de la GPU). ¿Mejora más el mAP50-95 o el mAP50? ¿Mejora el conteo? ¿Cuánto más tarda cada época?
# 3. **Congelar.** Entrena con `freeze=10` (esqueleto congelado). Compara el tiempo por época y el mAP en prueba. ¿En qué situación preferirías congelar?
# 4. **Otra partición.** Cambia la prueba a `rres_1` (Reino Unido) y pon `ethz_1` en entrenamiento. ¿Baja o sube el mAP en prueba? ¿Qué te dice eso sobre medir la generalización con un solo sitio de prueba?
# 5. **Densidad.** En `ethz_1`, cada pixel corresponde a 0.55 mm del suelo (Tabla 3 del artículo), así que una foto de 1024 × 1024 cubre unos $0.56 \times 0.56 \approx 0.32$ m². Convierte los conteos reales y predichos de la prueba a espigas por m² y calcula el error medio en esas unidades. ¿Qué error relativo tendría una estimación de rendimiento que usara esos conteos?
# 6. **Tu trigo.** Si tienes acceso a un campo de trigo (en el Valle del Yaqui o en la Costa de Hermosillo), toma 10 fotos desde arriba a 1.5–2 m de altura, etiqueta las espigas de dos o tres con una herramienta como [CVAT](https://www.cvat.ai) o [Label Studio](https://labelstud.io), y mide qué tan bien cuenta el modelo. Luego agrega tus fotos al entrenamiento y vuelve a medir.
