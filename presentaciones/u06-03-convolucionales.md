---
theme: none
title: Redes convolucionales
unidad: Unidad 6
layout: portada
subtitulo: "Por qué las imágenes importan, la convolución como preprocesamiento aprendido y las arquitecturas de VGG a EfficientNet"
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

# Redes convolucionales

---
layout: pregunta
---

# ¿Cuántas fotos se toman en el mundo cada segundo?

<v-click>

Unas **60,000**. En 2025 se tomaron más de **2 billones** de fotos ($2 \times 10^{12}$), casi todas con un teléfono.

</v-click>

<v-click>

Nadie puede verlas. Si queremos sacar algo de ellas, lo tiene que hacer una computadora.

</v-click>

<!--
Fuente de la cifra: Rise Above Research, Worldwide Image Capture Forecast (reportado por PetaPixel, junio de 2025). 2 × 10¹² / (365 · 24 · 3600) ≈ 63,000 por segundo. Preguntar cuántas fotos tomaron ellos esta semana.
-->

---

# Imágenes que alguien necesita entender

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 28px 48px">
<div v-click>

### Medicina

Radiografías, tomografías, fotos de retina y de piel. De los más de 1,400 dispositivos médicos con inteligencia artificial autorizados por la FDA hasta 2025, **tres de cada cuatro** son de radiología.

</div>
<div v-click>

### Satélites

Sentinel-2 fotografía toda la superficie terrestre cada 5 días, con pixeles de 10 m. Uso de suelo, sequías, presas, incendios, crecimiento de las ciudades.

</div>
<div v-click>

### Agricultura

Contar espigas de trigo, detectar plagas y malezas, estimar el rendimiento de una parcela. Sonora es de los mayores productores de trigo del país.

</div>
<div v-click>

### Industria y vida diaria

Inspección de piezas en una línea de producción, lectura de placas, el desbloqueo del teléfono, el auto que frena solo.

</div>
</div>

<!--
Cifras de la FDA: lista pública «AI-Enabled Medical Devices» (FDA); ~1,450 autorizaciones acumuladas a fines de 2025, ~76 % del panel de radiología. Sentinel-2: dos satélites, revisita de 5 días en el ecuador (ESA). Preguntar qué otros ejemplos se les ocurren.
-->

---

# Lo que hace especial a una imagen: lo local importa

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Un pixel solo no dice nada. Lo que importa es **cómo se relaciona con sus vecinos**: un borde, una esquina, una textura.

<v-click>

Y un borde es un borde en cualquier parte del cuadro: lo que aprendemos en una posición debería servir en todas.

</v-click>
</div>
<div v-click>

La misma estructura aparece fuera de las imágenes:

- **audio**, convertido en espectrograma: una imagen de frecuencia contra tiempo;
- **series de tiempo**: un patrón de unas cuantas horas, en cualquier día;
- **ADN**: un motivo de 6 a 12 bases, en cualquier parte de la secuencia.

Las redes convolucionales sirven donde **la información local importa**.

</div>
</div>

---
layout: pregunta
---

# Una foto de 224 × 224 en color entra a un perceptrón multicapa con 1,000 neuronas ocultas. ¿Cuántos pesos tiene la primera capa?

<v-click>

$224 \times 224 \times 3 = 150{,}528$ entradas por neurona: **150 millones de pesos**, más del doble de todos los de AlexNet. Y esa capa no sabe que el pixel (0, 0) está junto al (0, 1).

</v-click>

<!--
Dejar que hagan la cuenta. La segunda frase prepara las dos ideas de la siguiente lámina: localidad y pesos compartidos.
-->

---

# Fashion-MNIST: prendas de 28 × 28

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<PrendasMosaico :tam="68" />

<div class="pequeno tenue" style="margin-top: 18px">

70,000 imágenes del catálogo de Zalando en 10 clases (Xiao, Rasul y Vollgraf, 2017): 60,000 para entrenar y 10,000 para probar. Nació como reemplazo de los dígitos de MNIST, que ya no distinguían entre métodos buenos y malos.

</div>
</div>
<div>

**Problema:** dada la imagen, decir qué prenda es.

<v-click>

Dos cosas que sabemos de cualquier imagen:

- **Localidad:** un pixel se relaciona sobre todo con sus vecinos.
- **Posición irrelevante:** una costura es una costura arriba o abajo.

</v-click>
</div>
</div>

<!--
Problema: dada la imagen, decir el tipo de prenda. Preguntar cuáles creen que se confunden (camisa y camiseta, suéter y abrigo): lo veremos al final de la segunda sección.
-->

---

# La convolución: un filtro que se desliza

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<ConvVentana />

</div>
<div>

$$
\mathbf Z[r, c] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} \vW[u, v]\; \mathbf A[r + u,\, c + v]
$$

<div class="pequeno">

$\mathbf A$: la imagen; $\vW$: el filtro de $k \times k$ pesos; $\mathbf Z$: el mapa de salida en el renglón $r$ y la columna $c$. Cada salida es el producto punto del filtro con una ventana de $k \times k$. **Los mismos $k^2$ pesos** en todas las posiciones.

</div>

<v-click>

<div class="pequeno">

Este filtro resta la columna izquierda de la derecha: vale 0 donde la imagen es constante y crece donde la intensidad **sube hacia la derecha**.

</div>

</v-click>

<div class="nota">

En rigor es correlación cruzada; la convolución voltea el filtro. Como el filtro se aprende, da lo mismo.

</div>
</div>
</div>

<!--
Usar los botones para recorrer las 9 posiciones; un clic en una celda de la salida salta a esa posición. Sin relleno, 5×5 con 3×3 da 3×3: d − k + 1.
-->

---

# Filtros de bordes hechos a mano

<FiltrosSobel />

<div class="cols" style="gap: 40px; margin-top: 22px">
<div>

Filtros de Sobel: derivan en una dirección y suavizan en la otra. Azul, respuesta positiva; rojo, negativa.

</div>
<div v-click>

**Nueve números** detectan un tipo de borde en toda la imagen. Una red convolucional no los diseña: los **aprende**.

</div>
</div>

<!--
El vertical marca los contornos laterales; el horizontal, la suela y el borde superior del botín. Es el primer botín del conjunto de prueba.
-->

---

# Trasladar y filtrar conmutan

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Sea $T$ la traslación de la imagen $\mathbf A$ por $\Delta_r$ renglones y $\Delta_c$ columnas, $(T \mathbf A)[r, c] = \mathbf A[r - \Delta_r,\, c - \Delta_c]$, y $\vW \star \mathbf A$ filtrar $\mathbf A$ con $\vW$. Entonces

$$
\begin{aligned}
(\vW \star T\mathbf A)[r, c] &= \sum_{u,v} \vW[u, v]\, \mathbf A[r + u - \Delta_r,\, c + v - \Delta_c] \\
&= (\vW \star \mathbf A)[r - \Delta_r,\, c - \Delta_c] \\
&= \big(T(\vW \star \mathbf A)\big)[r, c] .
\end{aligned}
$$

<v-click>

La convolución es **equivariante** a traslaciones: si la prenda se mueve, el mapa también. En el botín con margen de ceros, la diferencia máxima entre los dos lados es **exactamente 0**.

</v-click>

<v-click>

<div class="pequeno">

Una capa densa no tiene esta propiedad: lo que aprende sobre un objeto en una posición no le sirve en otra.

</div>

</v-click>
</div>
<div>

<Equivariancia />

</div>
</div>

---

# Relleno, paso, canales y parámetros

<div class="cols" style="grid-template-columns: 5fr 4fr; gap: 40px">
<div>

- **Ventana** $k$: el filtro es de $k \times k$; la entrada mide $d \times d$.
- **Relleno** $p$: ceros alrededor; con $k = 3$, $p = 1$ conserva el tamaño.
- **Paso** $s$: la ventana avanza de $s$ en $s$.
- **Canales**: cada filtro abarca los $C_{\text{ent}}$ canales de entrada y produce uno de salida.

$$
d_{\text{sal}} = \left\lfloor \frac{d + 2p - k}{s} \right\rfloor + 1 ,\qquad
\#\text{parám.} = (C_{\text{ent}}\,k^2 + 1)\,C_{\text{sal}}
$$

<div class="pequeno">

$d_{\text{sal}}$: lado de la salida; $C_{\text{sal}}$: número de filtros (canales de salida); el $+1$ es el sesgo de cada filtro. El número de parámetros **no depende del tamaño de la imagen**. El *max-pooling* de 2 × 2 se queda con el máximo de cada bloque y no tiene parámetros.

</div>
</div>
<div v-click class="tabla-chica">

Nuestra red para Fashion-MNIST:

| capa | salida | parám. |
|---|---|---|
| conv 1→16 | 16×28×28 | 160 |
| max-pool | 16×14×14 | 0 |
| conv 16→32 | 32×14×14 | 4,640 |
| max-pool | 32×7×7 | 0 |
| lineal | 10 | 15,690 |

<div class="pequeno tenue">

Filtros de 3 × 3 con relleno 1, ReLU tras cada convolución. Total: 20,490; casi todo está en la capa final.

</div>
</div>
</div>

<style>
.tabla-chica table { font-size: 19px; }
.tabla-chica td { padding: 8px 12px 8px 0; }
</style>

<!--
Verificar en voz alta uno: (16·9 + 1)·32 = 4,640.
-->

---

# El campo receptivo crece con la profundidad

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<CampoReceptivo />

</div>
<div>

Una capa con ventana $k$ y paso $s$:

$$
\begin{aligned}
r^{[l]} &= r^{[l-1]} + (k - 1)\, \Delta^{[l-1]} \\
\Delta^{[l]} &= \Delta^{[l-1]}\, s
\end{aligned}
$$

<div class="pequeno">

$r^{[l]}$: lado del cuadro de la imagen que ve una neurona de la capa $l$; $\Delta^{[l]}$: distancia en pixeles entre posiciones vecinas de su mapa. Empieza con $r^{[0]} = \Delta^{[0]} = 1$.

</div>

<v-click>

<div class="pequeno">

Comprobación: el gradiente de una posición del mapa final respecto a la imagen es distinto de cero exactamente en un cuadro de **10 × 10** (pixeles 9 a 18).

</div>

</v-click>

<v-click>

<div class="pequeno">

Las primeras capas ven bordes; las profundas combinan bordes en partes cada vez mayores.

</div>

</v-click>
</div>
</div>

<!--
Los botones muestran el campo después de cada etapa. Los cuadros corresponden a la posición (3, 3) del mapa final de 7×7.
-->

---

# *Pooling*: invariancia aproximada

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<InvarianciaPooling />

<div class="pequeno tenue">

Cambio relativo de la representación $f(\mathbf A)$ de una imagen $\mathbf A$ al moverla un pixel a la derecha ($T\mathbf A$), en 1,000 imágenes de prueba (red con pesos al azar).

</div>
</div>
<div>

Para clasificar no queremos equivariancia sino **invariancia**: que la respuesta no cambie si la prenda se mueve.

<v-click>

Si el máximo de un bloque de 2 × 2 se mueve dentro del bloque, el *max-pooling* no cambia. Tras dos *poolings*, el cambio baja de 0.30 a **0.14**.

</v-click>

<v-click>

<div class="pequeno">

Es una propiedad de la arquitectura, no del entrenamiento. Y es aproximada: el **aumento de datos** (entrenar con imágenes desplazadas) completa el trabajo.

</div>

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# Una red convolucional en Fashion-MNIST

Primero, atributos locales; después, un clasificador.

---

# Dos partes con dos trabajos distintos

<ExtractorClasificador />

<div class="cols" style="gap: 40px; margin-top: 10px">
<div class="pequeno">

**Preprocesamiento aprendido.** Las convoluciones y los *poolings* convierten 784 pixeles en 1,568 atributos: qué bordes, esquinas y texturas hay, y más o menos dónde.

</div>
<div class="pequeno" v-click>

**Clasificador.** Encima va un modelo como los que ya conocemos: un perceptrón multicapa o, como aquí, una sola capa lineal con softmax (un GLM). Se entrenan **juntos**.

</div>
</div>

<!--
Es la misma red de la tabla de parámetros: dos bloques conv+ReLU+pool y una capa lineal de 15,690 parámetros. Insistir: la parte convolucional no clasifica nada, solo cambia la representación.
-->

---

# ¿De verdad la parte convolucional es solo preprocesamiento?

<div class="cols" style="grid-template-columns: 5fr 4fr; gap: 40px">
<div>

Experimento: tomamos la red ya entrenada, **congelamos** sus convoluciones, sacamos los 1,568 atributos de cada imagen y ajustamos encima una **regresión logística** común.

<v-click>

<div class="tabla-chica">

| atributos | clasificador | exactitud |
|---|---|---|
| 784 pixeles | regresión logística | 83.6 % |
| convoluciones con pesos **al azar** | regresión logística | 87.2 % |
| convoluciones **entrenadas** y congeladas | regresión logística | **89.3 %** |
| red convolucional completa | | 89.3 % |

</div>

</v-click>
</div>
<div>

<v-click>

El mismo clasificador lineal pasa de 83.6 % a 89.3 % solo por cambiar la representación, y empata a la red completa. Hasta con pesos al azar, los atributos locales ayudan.

</v-click>

<v-click>

<Caja tipo="idea">

Las convoluciones hacen la **ingeniería de atributos** que antes se hacía a mano (como los filtros de Sobel), pero aprendida de los datos y para el problema.

</Caja>

</v-click>
</div>
</div>

<style>
.tabla-chica table { font-size: 20px; }
.tabla-chica td { padding: 8px 14px 8px 0; }
</style>

<!--
Si la parte convolucional es buena, cualquier clasificador encima funciona. Esta idea es la base de la transferencia: reutilizar la parte convolucional en otro problema.
-->

---

# Más exactitud con 11 veces menos parámetros

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

<CurvasCNNMLP />

</div>
<div>

Mismo entrenamiento para las dos: Adam (tasa 0.001), minilotes de 64 en el mismo orden, 3 épocas.

<v-click>

La red convolucional: **89.3 %** contra 86.8 %, una quinta parte menos de errores (10.7 % contra 13.2 %), con 20,490 parámetros contra 235,146.

</v-click>

<v-click>

<Caja tipo="idea">

No es más capacidad, es **la capacidad correcta**: restringir las hipótesis de una forma que coincide con el problema baja la varianza sin pagar en sesgo.

</Caja>

</v-click>
</div>
</div>

<!--
Las dos seguirían mejorando con más épocas; redes convolucionales bien entrenadas pasan de 92 % en este conjunto. El MLP tiene dos capas ocultas de 256 y 128.
-->

---

# Lo que aprendió la primera capa

<div class="cols" style="grid-template-columns: 3fr 1fr; gap: 32px">
<div>

<FiltrosAprendidos />

</div>
<div>

Arriba, cada filtro de 3 × 3 (azul +, rojo −); abajo, su mapa de activación en el botín.

<v-click>

<div class="pequeno">

Varios son detectores de bordes en distintas orientaciones, como los de Sobel; otros responden al fondo o al interior de la prenda.

</div>

</v-click>

<v-click>

<div class="pequeno">

**Nadie los diseñó**: salieron de minimizar la entropía cruzada.

</div>

</v-click>
</div>
</div>

---

# Dónde se equivoca

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<ConfusionPrendas />

</div>
<div>

Filas: clase verdadera; columnas: predicha. Los errores no están repartidos al azar: casi todos son entre prendas de la parte superior del cuerpo.

<v-click>

La camisa es la clase más difícil (**62 %** de aciertos): 166 camisas fueron «camiseta».

<CamisasCamiseta />

</v-click>

<v-click>

<div class="pequeno">

A 28 × 28 en grises, una camisa de manga corta sin botones visibles **es** una camiseta. El límite está en la información, no solo en el modelo.

</div>

</v-click>
</div>
</div>

---
layout: seccion
numero: 3
---

# Arquitecturas

De ocho capas a cientos, y de más parámetros a más eficiencia.

---

# De LeNet a AlexNet

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

**LeNet-5** (LeCun, Bottou, Bengio y Haffner, 1998): dos capas convolucionales y unos 60,000 parámetros; leía dígitos de cheques. La red que acabamos de entrenar es una versión reducida de la misma idea.

<v-click>

**AlexNet** (Krizhevsky, Sutskever y Hinton, 2012): 8 capas, 60 millones de parámetros, ReLU, *dropout* y dos GPU. Ganó ImageNet (1.2 millones de fotos, 1,000 clases) con 15.3 % de error *top-5*, contra **26.2 %** del segundo lugar.

</v-click>

<v-click>

<div class="pequeno">

Después, una carrera: ¿cómo hacer redes más profundas, que se puedan entrenar y que no cuesten tanto?

</div>

</v-click>
</div>
<div>

<ErrorImageNet />

<div class="pequeno tenue">

Error *top-5* en la prueba de ImageNet según los artículos originales. *Top-5*: la clase correcta no está entre las cinco más probables.

</div>
</div>
</div>

<!--
El segundo lugar de 2012 usaba atributos diseñados a mano (SIFT, vectores de Fisher) y clasificadores lineales: preprocesamiento a mano más un clasificador. AlexNet aprendió el preprocesamiento. Las cuatro cifras son de los artículos originales (GoogLeNet: Szegedy et al., 2015; ResNet: He et al., 2016) y las tres de redes corresponden a ensambles de varios modelos.
-->

---

# VGG: solo filtros de 3 × 3

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<VGGPila />

</div>
<div>

Simonyan y Zisserman (2014): una sola regla, **apilar convoluciones de 3 × 3**; 16 o 19 capas.

<v-click>

Dos capas de 3 × 3 ven un cuadro de 5 × 5, igual que una de 5 × 5. Con $C$ canales de entrada y de salida:

$$
\underbrace{2 \cdot 9\,C^2}_{\text{dos de } 3\times 3} = 18\,C^2
\quad\text{contra}\quad
\underbrace{25\,C^2}_{\text{una de } 5 \times 5}
$$

</v-click>

<v-click>

<div class="pequeno">

28 % menos parámetros **y** una ReLU más en medio: más capacidad de expresar funciones no lineales. Con $C = 64$: 73,728 contra 102,400.

</div>

</v-click>
</div>
</div>

---

# VGG: la red que todos reutilizaron

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

VGG-16 tiene **138 millones** de parámetros, y el 89 % están en las tres capas densas del final. Al revés, el 99 % de las 15,500 millones de multiplicaciones por imagen ocurren en las convoluciones.

<v-click>

Fue la primera red cuya **parte convolucional** se volvió una herramienta de uso general, separada del problema para el que se entrenó:

</v-click>

</div>
<div>

<v-clicks>

- **Atributos listos para usar**: los de una red de ImageNet más un clasificador lineal superaban a los mejores sistemas hechos a mano en decenas de problemas (Razavian et al., 2014).
- **Detección de objetos**: Faster R-CNN (Ren et al., 2015) usó la parte convolucional de VGG-16 para encontrar objetos.
- **Transferencia de estilo**: pintar una foto «como Van Gogh» comparando las activaciones de VGG (Gatys, Ecker y Bethge, 2015).

</v-clicks>

</div>
</div>

<!--
Es la idea de la sección anterior llevada al extremo: si la parte convolucional es un buen preprocesamiento, sirve para muchos problemas. De eso trata la transferencia.
-->

---
layout: pregunta
---

# Una red de 56 capas se equivoca más que una de 20… en los datos de entrenamiento. ¿Por qué?

<v-click>

No es sobreajuste: el error **de entrenamiento** es mayor. La red de 56 capas podría imitar a la de 20 (las 36 capas extra solo tendrían que copiar su entrada), pero el descenso por gradiente **no encuentra** esa solución.

</v-click>

<!--
He, Zhang, Ren y Sun (2016), figura 1, con CIFAR-10. Lo llamaron el problema de la degradación. Dejar que propongan explicaciones: gradientes que se desvanecen, mala inicialización, optimización difícil.
-->

---

# ResNet: aprender solo la corrección

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 32px">
<div>

<BloqueResidual />

</div>
<div>

He, Zhang, Ren y Sun (2016): cada bloque $l$ aprende una corrección $F^{[l]}$ que **se suma** a su entrada $\va^{[l]}$:

$$
\va^{[l+1]} = \va^{[l]} + F^{[l]}\big(\va^{[l]}\big)
$$

<v-click>

<div class="pequeno">

Si lo mejor es no hacer nada, basta con $F^{[l]} = 0$: pesos pequeños, fácil de encontrar. Copiar la entrada con dos convoluciones simples, en cambio, exige atinarle a la identidad.

</div>

</v-click>
</div>
</div>

<!--
F^{[l]} son dos convoluciones de 3×3 con normalización por lotes; la ReLU final va después de la suma. El atajo no tiene parámetros. Si cambia el número de canales, el atajo es una convolución 1×1.
-->

---

# El atajo también es un atajo para el gradiente

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

Derivando la regla del bloque y encadenando $D$ bloques:

$$
\frac{\partial \ell}{\partial \va^{[l]}} = \frac{\partial \ell}{\partial \va^{[D]}}\prod_{m=l}^{D-1}\Big(\mathbf I + \frac{\partial F^{[m]}}{\partial \va^{[m]}}\Big)
$$

<div class="pequeno">

$\ell$: la pérdida; $D$: el número de bloques; $\mathbf I$: la identidad. Al desarrollar el producto aparece $\partial \ell/\partial \va^{[D]}$ **tal cual**: el gradiente llega a cualquier capa sin pasar por ninguna matriz de pesos.

</div>

<v-click>

<div class="pequeno">

Sin atajo, el producto es solo de las $\partial F^{[m]}/\partial \va^{[m]}$: si son algo menores que 1, el gradiente se desvanece en pocas decenas de capas.

</div>

</v-click>
</div>
<div>

<v-click>

<GradienteResidual />

<div class="pequeno tenue">

40 capas de ancho 64, recién inicializadas. La inicialización de He arregla la escala al inicio; la conexión residual la protege con cualquier inicialización.

</div>

</v-click>
</div>
</div>

<!--
Con la inicialización por omisión de PyTorch, cada capa simple multiplica la varianza por cerca de 1/6: a la primera capa llega 10⁻¹⁴ del gradiente.
-->

---

# El cuello de botella: 1 × 1, 3 × 3, 1 × 1

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Las ResNet de 50 capas o más trabajan con 256 canales o más. Dos convoluciones de 3 × 3 de 256 a 256 canales son caras. El bloque **cuello de botella**:

1. una convolución **1 × 1** baja de 256 a 64 canales;
2. la de **3 × 3** trabaja con 64;
3. otra **1 × 1** regresa a 256.

<v-click>

<div class="pequeno">

Una convolución de 1 × 1 no mira a los vecinos: en cada pixel, mezcla los canales con una capa lineal. Es una forma barata de cambiar el número de canales.

</div>

</v-click>
</div>
<div v-click class="tabla-chica">

| bloque | parámetros |
|---|---|
| dos 3 × 3, 256 → 256 | 1,179,648 |
| 1 × 1, 3 × 3, 1 × 1 (256 → 64 → 256) | 69,632 |

<div class="pequeno" style="margin-top: 16px">

17 veces menos. Así, ResNet-152 ganó ImageNet 2015 con **3.57 %** de error *top-5*, con menos operaciones que VGG-16.

</div>
</div>
</div>

<style>
.tabla-chica table { font-size: 20px; }
.tabla-chica td { padding: 8px 14px 8px 0; }
</style>

<!--
Sin contar sesgos ni normalización por lotes. ResNet-152: 11.3 mil millones de multiplicaciones-sumas contra 15.3 de VGG-16 (He et al., 2016, tabla 1 y sección 4.1).
-->

---

# Inception: varios tamaños de ventana a la vez

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 32px">
<div>

<ModuloInception />

</div>
<div>

Szegedy et al. (2015), GoogLeNet: ¿ventana de 1, 3 o 5? **Todas**, en paralelo, y se apilan los resultados.

<v-click>

<div class="pequeno">

El truco está en las convoluciones **1 × 1 doradas**: reducen 192 canales a 96 o a 16 antes de las ventanas grandes. En este módulo: 393,216 parámetros sin reducir, **163,328** reduciendo (2.4 veces menos).

</div>

</v-click>

<v-click>

<div class="pequeno">

GoogLeNet: 22 capas y **unos 6.6 millones** de parámetros, contra 138 de VGG, con menos error en ImageNet (6.67 % *top-5*).

</div>

</v-click>
</div>
</div>

<!--
Módulo 3a de GoogLeNet (Szegedy et al., 2015, tabla 1): salidas 64, 128, 32 y 32 canales. El nombre viene de la película (el meme «we need to go deeper»), según el propio artículo.
-->

---

# Xception: separar espacio y canales

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 32px">
<div>

<ConvSeparable />

</div>
<div>

Una convolución normal mezcla **a la vez** vecinos y canales. Chollet (2017) llevó Inception al extremo: hacerlo **por separado**.

<v-click>

$$
\underbrace{C\,C'\,k^2}_{\text{normal}}
\quad\text{contra}\quad
\underbrace{C\,k^2}_{\text{por canal}} + \underbrace{C\,C'}_{1\times 1}
$$

<div class="pequeno">

$C$, $C'$: canales de entrada y de salida; $k$: lado de la ventana. La razón es $\tfrac{1}{C'} + \tfrac{1}{k^2}$: con $k = 3$ y muchos canales, **casi 9 veces menos** parámetros y operaciones. Con $C = C' = 256$: 589,824 contra 67,840.

</div>

</v-click>

<v-click>

<div class="pequeno">

Es la pieza básica de las redes para teléfonos (MobileNet) y de EfficientNet.

</div>

</v-click>
</div>
</div>

<!--
Xception: 22.9 millones de parámetros y 79.0 % de exactitud top-1 en ImageNet (Chollet, 2017), contra 78.2 % de Inception V3 con un número parecido de parámetros. En PyTorch: nn.Conv2d(C, C, 3, padding=1, groups=C) seguida de nn.Conv2d(C, C2, 1).
-->

---

# EfficientNet: crecer en las tres direcciones a la vez

<EscalamientoCompuesto />

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 6px">
<div class="pequeno">

Tan y Le (2019): ¿conviene una red más ancha (canales), más profunda (capas) o con imágenes de más resolución? Crecer en una sola dirección se satura pronto.

</div>
<div class="pequeno" v-click>

Escalamiento compuesto con un solo coeficiente $\phi$: profundidad $\times 1.2^{\phi}$, ancho $\times 1.1^{\phi}$, resolución $\times 1.15^{\phi}$; el costo se multiplica por $\approx 2^{\phi}$. De B0 (5.3 millones de parámetros) a B7.

</div>
</div>

<!--
Las constantes α = 1.2, β = 1.1, γ = 1.15 cumplen α·β²·γ² ≈ 2 (Tan y Le, 2019, ecuación 3): el ancho y la resolución cuentan al cuadrado en el costo. La red base B0 se encontró con búsqueda automática de arquitecturas y usa bloques con convoluciones separables en profundidad.
-->

---

# Exactitud contra tamaño

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<ArquitecturasImageNet />

</div>
<div>

Exactitud *top-1* en ImageNet contra millones de parámetros, para los pesos que distribuye `torchvision` (Xception: su artículo).

<v-click>

<div class="pequeno">

EfficientNet-B0 acierta 77.7 % con 5.3 millones de parámetros; VGG-16, 71.6 % con 138 millones: **26 veces menos** parámetros y 6 puntos más.

</div>

</v-click>

<v-click>

<div class="pequeno">

Todas siguen la misma receta: una parte convolucional que produce atributos y un clasificador pequeño al final. Todas se pueden descargar ya entrenadas.

</div>

</v-click>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 24px; line-height: 1.4">

1. Las imágenes son de los datos más abundantes y útiles; su información es **local** y puede estar en cualquier parte. Lo mismo pasa con audio, series y secuencias.
2. Convolución = capa lineal con **conexiones locales** y **pesos compartidos**: equivariante a traslaciones y con pocos parámetros. $d_{\text{sal}} = \lfloor (d + 2p - k)/s \rfloor + 1$; parámetros $(C_{\text{ent}}k^2 + 1)C_{\text{sal}}$.
3. Una red convolucional = **preprocesamiento aprendido** (atributos locales) + un clasificador común encima.
4. VGG: filtros de 3 × 3 y una parte convolucional reutilizable. ResNet: $\va + F(\va)$ deja pasar el gradiente.
5. Inception, Xception y EfficientNet: convoluciones 1 × 1, separables y escalamiento compuesto para gastar menos.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Redes convolucionales con Fashion-MNIST* (Unidad 6, ejemplo 3).

**Lectura:** Goodfellow, Bengio y Courville, *Deep Learning*, capítulo 9; Zhang et al., *Dive into Deep Learning*, capítulos 7 y 8; He et al., «Deep residual learning for image recognition» (CVPR 2016); Tan y Le, «EfficientNet» (ICML 2019).

</div>
</div>
</div>
