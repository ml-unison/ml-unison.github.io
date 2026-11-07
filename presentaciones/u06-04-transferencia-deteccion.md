---
theme: none
title: Transferencia y detección de objetos
unidad: Unidad 6
layout: portada
subtitulo: "Lo que una red entrenada con millones de fotos sabe de tu problema, y cómo encontrar objetos con ella"
fecha: 2026-2
canvasWidth: 1280
aspectRatio: 16/9
routerMode: hash
colorSchema: light
transition: fade
favicon: /marca/favicon.png
fonts:
  sans: Source Sans 3 Variable
  serif: Source Sans 3 Variable
  mono: JetBrains Mono Variable
  provider: none
highlighter: shiki
lineNumbers: false
browserExporter: true
drawings:
  persist: false
info: |
  Reconocimiento de Patrones · Unidad 6 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Transferencia y detección de objetos

---
layout: pregunta
---

# Tienes 200 imágenes etiquetadas de tu problema. ¿Entrenas desde cero una red con 11 millones de parámetros?

<v-click>

Casi seguro se sobreajusta. Pero alguien ya entrenó esa red con **1.2 millones** de fotos de ImageNet. ¿Sirve de algo lo que aprendió, si tu problema no tiene nada que ver con perros, autos o frutas?

</v-click>

<!--
Dejar que discutan. La respuesta corta es sí, y la razón está en lo que aprende cada capa.
-->

---
layout: seccion
numero: 1
---

# Qué aprende cada capa

Abrir la red y ver qué la activa.

---

# Las imágenes que más activan a cada canal

<JerarquiaCapas />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

VGG-16 entrenada en ImageNet. Para algunos canales de cinco capas, los recortes de fotos que **más lo activan** (el pedazo de imagen que esa neurona ve, su campo receptivo).

</div>
<div class="pequeno" v-click>

Abajo, colores y bordes; en medio, texturas y patrones; arriba, **partes de objetos**. La idea es de Zeiler y Fergus (2014); estos recortes los calculamos nosotros.

</div>
</div>

<!--
Cada fila es una capa (conv1_2 … conv5_3); cada grupo, un canal; dentro del grupo, las imágenes de Imagenette con mayor activación de ese canal, recortadas a su campo receptivo. Fíjense en cómo crece el recorte: es el campo receptivo, que crece con la profundidad.
-->

---

# La imagen que cada canal «quiere ver»

<JerarquiaCapas tipo="maxima" />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

Otra forma de preguntar: empezamos con ruido y hacemos **ascenso por gradiente sobre la imagen** (no sobre los pesos) para que un canal se active lo más posible.

</div>
<div class="pequeno" v-click>

Es la misma diferenciación automática de siempre, solo que derivamos respecto a la entrada. Otra vez: rayas y colores abajo, patrones cada vez más complejos arriba (Olah, Mordvintsev y Schubert, 2017).

</div>
</div>

---

# Genérico abajo, específico arriba

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Los bordes, las texturas y los colores **no son de ImageNet**: aparecen en radiografías, en fotos de satélite y en campos de trigo.

<v-click>

Yosinski, Clune, Bengio y Lipson (2014) lo midieron: copiaron las primeras $l$ capas de una red entrenada en un problema a otra red para un problema distinto. Las primeras capas se transfieren casi sin pérdida; las últimas, cada vez peor: se **especializan** en las clases originales.

</v-click>
</div>
<div v-click>

<div class="lema">

La parte convolucional de una red preentrenada es un **preprocesamiento** ya aprendido. Solo hay que poner encima el clasificador de *nuestro* problema.

</div>

</div>
</div>

---
layout: seccion
numero: 2
---

# Transferir

Una red de ImageNet aplicada a imágenes de satélite.

---

# Dónde están los modelos preentrenados

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

- **`torchvision.models`**: las redes clásicas (VGG, ResNet, EfficientNet…) con pesos de ImageNet.
- **`timm`** y el **Hub de Hugging Face**: cientos de miles de modelos, de visión y de todo.
- **Keras Applications** y **Kaggle Models**, para TensorFlow y JAX.
- **Ultralytics**: detectores YOLO listos para usar.

```python
from torchvision.models import resnet18, ResNet18_Weights
pesos = ResNet18_Weights.IMAGENET1K_V1
red = resnet18(weights=pesos)       # descarga 45 MB
prepro = pesos.transforms()        # su preprocesamiento
```

</div>
<div v-click>

<Caja tipo="cuidado" nombre="antes de usar uno">

- **¿Con qué datos se entrenó?** Lo que no vio, no lo sabe; y hereda sus sesgos.
- **Licencia**: algunas prohíben el uso comercial o te obligan a abrir tu código.
- **Preprocesamiento**: tamaño, escala y normalización iguales a los del entrenamiento.
- **Tamaño**: ¿cabe y corre a tiempo donde lo vas a usar?

</Caja>

</div>
</div>

---

# EuroSAT: uso de suelo desde el espacio

<div class="cols" style="grid-template-columns: 7fr 4fr; gap: 36px">
<div>

<MosaicoSatelite />

</div>
<div>

27,000 imágenes del satélite Sentinel-2 en 10 clases (Helber et al., 2019). Cada imagen es un cuadro de **640 m** de lado (64 × 64 pixeles de 10 m).

<v-click>

**Problema:** etiquetar imágenes de satélite es caro. ¿Qué tan bien clasificamos con **10, 50 o 200** imágenes etiquetadas por clase?

</v-click>

<v-click>

<div class="pequeno">

Ninguna de estas clases (río, autopista, cultivo…) está en ImageNet.

</div>

</v-click>
</div>
</div>

<!--
Prueba: 100 imágenes por clase, distintas de las de entrenamiento. Los subconjuntos están anidados: los 10 por clase están dentro de los 50, y estos dentro de los 200.
-->

---

# Tres formas de transferir

<TresTransferencias />

<div class="pequeno tenue" style="margin-top: -6px">

Gris: pesos de ImageNet congelados. Azul: se siguen entrenando. Dorado: capa nueva para nuestras 10 clases.

</div>

<!--
Las dos primeras son el mismo modelo (un clasificador lineal sobre 512 atributos fijos); cambia solo la herramienta: sklearn o PyTorch. La segunda permite aumento de datos, porque los atributos se recalculan con cada imagen transformada.
-->

---

# En código: la última capa o ajuste fino

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 32px">
<div>

**Sustituir solo la última capa**

```python
red = resnet18(weights="IMAGENET1K_V1")
red.requires_grad_(False)     # congelar todo
red.fc = nn.Linear(512, 10)   # nueva, entrenable
opt = torch.optim.Adam(red.fc.parameters(),
                       lr=1e-3)
```

<div class="pequeno">

Solo se entrenan $512 \cdot 10 + 10 = 5{,}130$ parámetros: rápido, y difícil de sobreajustar.

</div>
</div>
<div v-click>

**Ajuste fino** (*fine-tuning*)

```python
red = resnet18(weights="IMAGENET1K_V1")
red.fc = nn.Linear(512, 10)
cabeza = red.fc.parameters()
opt = torch.optim.Adam([
    {"params": cabeza, "lr": 1e-3},
    {"params": cuerpo, "lr": 1e-4}])
```

<div class="pequeno">

`cuerpo`: los demás parámetros. Tasa **pequeña** para no borrar lo aprendido en ImageNet: los 11 millones de pesos se mueven poco, hacia nuestro problema.

</div>
</div>
</div>

---

# Transferir le gana a empezar de cero

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<TransferenciaCurva />

</div>
<div>

Con **10 imágenes por clase**, las tres formas de transferir aciertan 77 %; la red desde cero, 55 %, y ni con 200 por clase (65.5 %) las alcanza.

<v-click>

Con más etiquetas, el **ajuste fino** se separa: 91.4 % contra 88.9 % con 200 por clase. Con 10 no gana nada: muy pocos datos para mover 11 millones de pesos.

</v-click>

<v-click>

<div class="pequeno">

Prueba: 1,000 imágenes. Desde cero: 3 bloques convolucionales, 24,234 parámetros y aumento de datos. La transferencia brilla **con pocas etiquetas**.

</div>

</v-click>
</div>
</div>

---

# ¿Cuál usar?

<div class="tabla-media">

| | pocas etiquetas | muchas etiquetas |
|---|---|---|
| **datos parecidos a los del preentrenamiento** | congelar y cambiar la última capa | ajuste fino |
| **datos muy distintos** | congelar, pero usar atributos de capas intermedias | ajuste fino, o entrenar desde cero si sobran datos |

</div>

<v-click>

<div class="pequeno" style="margin-top: 24px">

Siempre se empieza por lo barato: congelar y cambiar la última capa. Si no alcanza, ajuste fino con tasa pequeña. Y siempre se compara contra entrenar desde cero en un conjunto de validación.

</div>

</v-click>

<style>
.tabla-media table { font-size: 22px; width: 100%; }
.tabla-media td, .tabla-media th { padding: 12px 16px 12px 0; }
</style>

<!--
Las capas intermedias son menos específicas de las clases de ImageNet (lo que vimos con Yosinski et al.).
-->

---
layout: caso
caso: Caso · Equidad
cifra: 34.7 %
cifraNota: error máximo de los clasificadores comerciales de género en mujeres de piel oscura; en hombres de piel clara, a lo más 0.8 %.
fuente: J. Buolamwini y T. Gebru, «Gender Shades», Conference on Fairness, Accountability and Transparency, PMLR 81 (2018).
---

# El modelo hereda sus datos

Buolamwini y Gebru evaluaron los clasificadores de género de Microsoft, IBM y Face++ con fotos de 1,270 parlamentarios de seis países, tres africanos y tres europeos. Los tres fallaban más en mujeres de piel oscura.

<v-click>

Dos conjuntos de rostros muy usados (IJB-A y Adience) tenían entre **79.6 % y 86.2 %** de personas de piel clara.

</v-click>

<v-click>

La transferencia hereda lo que el modelo aprendió… **y lo que no**. Antes de usar una red preentrenada: ¿con qué datos se entrenó, a quién representan? Y medir el error por separado en cada grupo.

</v-click>

---
layout: seccion
numero: 3
---

# Detección de objetos

No solo qué hay, sino dónde.

---
layout: pregunta
---

# Tienes un clasificador excelente de «auto» contra «no auto». ¿Cómo encuentras todos los autos de una foto?

<v-click>

Recorrer la foto con una ventana, a varios tamaños, y clasificar cada recorte. Funciona… pero son **25,132** recortes para una sola foto de 640 × 480.

</v-click>

<!--
Ventana de 64 × 128 (la de HOG para personas), paso de 8 pixeles y 28 escalas que crecen 5 % cada una. Con una red como AlexNet en cada recorte serían unos 18 billones (18 × 10¹²) de multiplicaciones por foto Así trabajaban los detectores de rostros de Viola y Jones (2001) y de personas con HOG de Dalal y Triggs (2005), con atributos hechos a mano y clasificadores muy rápidos.
-->

---

# Detectar = clase + caja

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<DeteccionesFoto :foto="0" />

</div>
<div>

Un detector devuelve, por cada objeto, una **caja** $(x_1, y_1, x_2, y_2)$, una **clase** y una **confianza**.

<v-click>

Dos problemas nuevos:

1. ¿Cuándo una caja predicha «le atina» a la verdadera?
2. El detector suele marcar el mismo objeto **varias veces**.

</v-click>
</div>
</div>

---

# ¿Le atinó? Intersección sobre unión

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<IoUCajas />

</div>
<div>

$$
\text{IoU}(A, B) = \frac{\text{área}(A \cap B)}{\text{área}(A \cup B)}
$$

<div class="pequeno">

$A$: la caja verdadera; $B$: la predicha. Vale 1 si coinciden y 0 si no se tocan.

</div>

<v-click>

Una predicción cuenta como acierto si es de la clase correcta y su IoU con una caja verdadera es al menos **0.5**.

</v-click>

<v-click>

<div class="pequeno">

Con eso se calculan precisión y exhaustividad por clase; el área bajo su curva, promediada sobre las clases, es la **mAP**. COCO además promedia sobre umbrales de IoU de 0.5 a 0.95 (mAP50-95).

</div>

</v-click>
</div>
</div>

---

# Supresión de no máximos

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 36px">
<div>

<NMSAntesDespues />

</div>
<div>

El detector propone muchas cajas casi iguales para el mismo objeto.

<v-click>

<Caja tipo="algoritmo" nombre="supresión de no máximos">

1. Ordenar las cajas por confianza.
2. Quedarse con la mejor y **borrar** las que la traslapen con IoU > 0.7.
3. Repetir con las que quedan.

</Caja>

</v-click>

<v-click>

<div class="pequeno">

En esta foto, 82 cajas crudas quedan en 11; detrás del autobús había 7. Las cifras: confianza ≥ 0.25 y NMS por clase (un auto no borra a la persona de enfrente).

</div>

</v-click>
</div>
</div>

---

# R-CNN: propuestas más atributos transferidos

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Girshick, Donahue, Darrell y Malik (2014), en lugar de millones de ventanas:

1. un algoritmo clásico propone **unas 2,000 regiones** que podrían ser objetos;
2. cada región se recorta, se escala a 224 × 224 y pasa por una **red de ImageNet** con ajuste fino;
3. sobre esos atributos, una SVM por clase y un ajuste fino de la caja.

</div>
<div>

<v-click>

<div class="lema">

R-CNN **es transferencia**: no había suficientes cajas etiquetadas para entrenar una red desde cero.

</div>

</v-click>

<v-click>

<div class="pequeno" style="margin-top: 24px">

Subió la mAP en PASCAL VOC de 35 % a más de 50 %. Pero pasaba 2,000 recortes por la red: **47 segundos por imagen** con VGG-16, en una GPU.

</div>

</v-click>
</div>
</div>

<!--
Búsqueda selectiva (Uijlings et al., 2013) para las propuestas. La mAP en VOC 2010: 35.1 % del mejor sistema anterior (Uijlings et al., también con búsqueda selectiva) contra 53.7 % de R-CNN; con VGG-16 llegó a 66 % en VOC 2007. Los 47 s son de la tabla 4 de Fast R-CNN (Girshick, 2015).
-->

---

# Fast y Faster R-CNN: compartir el cómputo

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

**Fast R-CNN** (Girshick, 2015): pasar la imagen **una sola vez** por la parte convolucional y recortar cada región **del mapa de atributos**, no de la imagen. Una sola red, entrenada de una vez.

<v-click>

**Faster R-CNN** (Ren, He, Girshick y Sun, 2015): las propuestas también las hace una red, sobre el mismo mapa: en cada posición, unas cuantas **cajas ancla** de tamaños y proporciones fijas, y para cada una, «¿hay objeto?» y cómo corregirla.

</v-click>
</div>
<div>

<v-click>

<div class="pequeno">

Detectores de **dos etapas**: primero dónde podría haber algo, después qué es. De 47 s a unas **7 imágenes por segundo**, y con más exactitud.

</div>

</v-click>

<v-click>

<div class="pequeno">

La parte convolucional, otra vez, es una VGG-16 o una ResNet de ImageNet: el preprocesamiento compartido para todas las regiones.

</div>

</v-click>
</div>
</div>

---

# YOLO: mirar una sola vez

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 36px">
<div>

<RejillaYOLO />

</div>
<div>

Redmon, Divvala, Girshick y Farhadi (2016), *You Only Look Once*: detección como **una sola regresión**.

<v-click>

<div class="pequeno">

La imagen se divide en una rejilla de 7 × 7. La celda donde cae el **centro** de un objeto es la responsable de él: predice cajas (posición, tamaño, confianza) y las probabilidades de cada clase.

</div>

</v-click>

<v-click>

<div class="pequeno">

Una pasada por la red da todas las cajas a la vez. Después, NMS. **45 imágenes por segundo**: video en tiempo real.

</div>

</v-click>
</div>
</div>

<!--
YOLO original: S = 7, B = 2 cajas por celda, C = 20 clases de VOC: salida de 7 × 7 × 30. La parte convolucional se preentrenó en ImageNet (otra vez transferencia).
-->

---

# Velocidad contra exactitud

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<DetectoresVelocidad />

</div>
<div>

El primer YOLO era **menos exacto** que Faster R-CNN, sobre todo con objetos chicos y amontonados, pero 6 veces más rápido.

<v-click>

<div class="pequeno">

Las versiones siguientes cerraron la distancia: YOLOv2 y v3 (anclas y varias escalas), y desde 2020 las de Ultralytics. La vigente, **YOLO26** (enero de 2026), se entrena para dar una sola caja por objeto y ya no necesita NMS; la versión nano tiene 2.4 millones de parámetros.

</div>

</v-click>

<v-click>

<div class="pequeno tenue">

Ultralytics publica YOLO con licencia AGPL-3.0: si lo usas en un servicio, tienes que abrir tu código, o pagar una licencia comercial.

</div>

</v-click>
</div>
</div>

<!--
Cifras de los artículos originales en PASCAL VOC 2007 (ver comentario del componente). Las comparaciones modernas se hacen en COCO con mAP50-95.
-->

---

# Usar YOLO: tres líneas

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 32px">
<div>

```python
from ultralytics import YOLO
modelo = YOLO("yolo26n.pt")      # 80 clases de COCO
resultados = modelo("foto.jpg")
```

<v-click>

<div class="pequeno">

En la calle Matamoros: 7 autos, 4 personas y 1 autobús, en unas décimas de segundo en un procesador común. Un Faster R-CNN con ResNet-50 tarda más de 10 segundos en la misma foto y la misma computadora.

</div>

</v-click>

</div>
<div>

<DeteccionesFoto :foto="1" />

</div>
</div>

<!--
En clase se puede correr en vivo en Colab con la libreta del ejemplo 5. La «n» es la versión nano, la más chica.
-->

---

# Ajustar YOLO a tu problema: contar espigas de trigo

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 36px">
<div>

YOLO no conoce la clase «espiga». Pero su parte convolucional sí sabe de bordes, texturas y formas: **ajuste fino**.

```python
modelo = YOLO("yolo26n.pt")
modelo.train(data="GlobalWheat2020.yaml",
             epochs=30, imgsz=640)
```

<div class="pequeno">

El archivo `.yaml` dice dónde están las imágenes, sus cajas (un `.txt` por imagen) y los nombres de las clases. YOLO cambia la última capa para 1 clase y ajusta todo lo demás.

</div>
</div>
<div v-click>

<div class="pequeno">

**Global Wheat** (David et al., 2020): 3,422 fotos de campos de trigo de Europa y Canadá con **145,411 espigas** marcadas, unas 42 por foto. Contar espigas por metro cuadrado permite estimar el rendimiento de una parcela.

</div>

<div class="pequeno" style="margin-top: 14px">

Sonora siembra casi la mitad del trigo grano del país (47.5 % de la superficie, SIAP 2022), y en el Valle del Yaqui está el campo experimental del CIMMYT donde Borlaug creó los trigos de la Revolución Verde.

</div>
</div>
</div>

<!--
El ajuste fino tarda del orden de una hora en una GPU T4 de Colab, por eso no lo corremos en clase. La libreta del ejemplo 6 lo hace paso a paso.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 24px; line-height: 1.4">

1. Las capas de una red convolucional van de lo **genérico** (bordes, colores) a lo **específico** (partes, objetos).
2. Las primeras capas sirven para otros problemas: la parte convolucional es un **preprocesamiento ya aprendido**.
3. Transferir: congelar y **sustituir la última capa**; si no alcanza, **ajuste fino** con tasa pequeña. Con pocas etiquetas, le gana a entrenar desde cero.
4. Detección: IoU para calificar, NMS para limpiar. De la ventana deslizante a R-CNN (transferencia), Faster R-CNN (dos etapas) y YOLO (una pasada).
5. Un modelo preentrenado hereda sus datos: sus aciertos y sus sesgos.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libretas:** *Transferencia* (ejemplo 4), *Detección de objetos* (ejemplo 5) y, con GPU en Colab, *Ajuste fino de YOLO* (ejemplo 6), todas de la Unidad 6.

**Lectura:** Zhang et al., *Dive into Deep Learning*, capítulos 14.2 y 14.3–14.8; Zeiler y Fergus (ECCV 2014); Yosinski et al. (NeurIPS 2014); Redmon et al., «You Only Look Once» (CVPR 2016).

</div>
</div>
</div>
