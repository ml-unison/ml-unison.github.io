---
theme: none
title: Redes convolucionales y transferencia
unidad: Unidad 6
layout: portada
subtitulo: "Localidad, pesos compartidos y lo que una red aprendida en fotos sabe de imágenes de satélite"
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

# Redes convolucionales y transferencia

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

Menos parámetros, mejor sesgo inductivo.

---

# Más exactitud con 11 veces menos parámetros

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

<CurvasCNNMLP />

</div>
<div>

Mismo entrenamiento para las dos: Adam (tasa 0.001), minilotes de 64 en el mismo orden, 3 épocas.

<v-click>

La red convolucional: **89.4 %** contra 86.7 %, una cuarta parte menos de errores, con 20,490 parámetros contra 235,146.

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

# De LeNet a ResNet

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

**LeNet-5** (LeCun, Bottou, Bengio y Haffner, 1998): dos capas convolucionales y unos 60,000 parámetros; leía dígitos de cheques. La red que acabamos de entrenar es una versión reducida de la misma idea.

<v-click>

**AlexNet** (Krizhevsky, Sutskever y Hinton, 2012): 8 capas, 60 millones de parámetros, ReLU, *dropout* y dos GPU. Ganó ImageNet (1.2 millones de fotos, 1,000 clases) con 15.3 % de error *top-5*, contra **26.2 %** del segundo lugar.

</v-click>

<v-click>

<div class="pequeno">

Después, la carrera por la profundidad, y un obstáculo: con más capas, el error **de entrenamiento** empeoraba.

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
El segundo lugar de 2012 usaba atributos diseñados a mano (SIFT, vectores de Fisher) y clasificadores lineales: lo que se hacía antes. Las cuatro cifras son de los artículos originales (GoogLeNet: Szegedy et al., 2015; ResNet: He et al., 2016) y las tres de redes corresponden a ensambles de varios modelos.
-->

---

# Conexiones residuales

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

He, Zhang, Ren y Sun (2016): una red simple de 56 capas se ajustaba **peor** que una de 20 en sus propios datos de entrenamiento. En vez de su salida, cada bloque $l$ aprende una corrección $F^{[l]}$ que se suma a su entrada $\va^{[l]}$:

$$
\va^{[l+1]} = \va^{[l]} + F^{[l]}\big(\va^{[l]}\big)
$$

<v-click>

$$
\frac{\partial \ell}{\partial \va^{[l]}} = \frac{\partial \ell}{\partial \va^{[D]}}\prod_{m=l}^{D-1}\Big(\mathbf I + \frac{\partial F^{[m]}}{\partial \va^{[m]}}\Big)
$$

<div class="pequeno">

$\ell$: la pérdida; $D$: el número de bloques; $\mathbf I$: la identidad. Al desarrollar el producto aparece $\partial \ell/\partial \va^{[D]}$ **tal cual**: el gradiente llega a cualquier capa sin pasar por ninguna matriz de pesos. Y si lo mejor es no hacer nada, basta $F^{[l]} = 0$.

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
Con la inicialización por omisión de PyTorch, cada capa simple multiplica la varianza por cerca de 1/6: a la primera capa llega 10⁻¹⁴ del gradiente. Con conexiones residuales y normalización por lotes, ResNet-152 ganó ImageNet 2015. ResNet-18, la que usaremos, tiene 11.7 millones de parámetros.
-->

---
layout: seccion
numero: 3
---

# Transferencia

Lo que una red entrenada con fotos sabe de imágenes de satélite.

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

Idea: usar los 512 atributos de la penúltima capa de una **ResNet-18 entrenada en ImageNet** (perros, autos, frutas…) y ajustar encima una regresión logística.

</div>

</v-click>
</div>
</div>

<!--
Prueba: 100 imágenes por clase, distintas de las de entrenamiento. Los subconjuntos están anidados: los 10 por clase están dentro de los 50, y estos dentro de los 200.
-->

---

# Transferir le gana a empezar de cero

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<TransferenciaCurva />

</div>
<div>

Con 10 por clase, la red preentrenada supera a la nueva con 200. La nueva: 3 bloques convolucionales, unos 24,000 parámetros, volteos al azar (**aumento de datos**), 600 pasos de Adam.

<v-click>

Las primeras capas aprenden bordes, colores y texturas; las intermedias, rayas y cuadrículas. Eso también describe un campo o una colonia.

</v-click>

<v-click>

<div class="pequeno">

Con miles de etiquetas por clase la distancia se acorta. La transferencia brilla **con pocas etiquetas**: lo común en medicina, agricultura o industria.

</div>

</v-click>
</div>
</div>

<!--
Los autores de EuroSAT reportan 98.57 % ajustando redes preentrenadas con el 80 % de los datos. Ajuste fino: seguir entrenando toda la red con tasa pequeña. En la libreta solo corre con GPU (Colab). Errores más comunes del modelo con 200 por clase: río contra carretera (una franja que cruza el cuadro) y vegetación herbácea contra cultivo permanente.
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

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Convolución = capa lineal con **conexiones locales** y **pesos compartidos**: equivariante a traslaciones y con pocos parámetros.
2. Con lado $d$, ventana $k$, relleno $p$ y paso $s$: $d_{\text{sal}} = \lfloor (d + 2p - k)/s \rfloor + 1$; parámetros $(C_{\text{ent}}k^2 + 1)C_{\text{sal}}$ con $C$ canales; el campo receptivo crece con la profundidad.
3. En Fashion-MNIST, 11 veces menos parámetros y más exactitud: el sesgo inductivo correcto.
4. Conexión residual $\va + F(\va)$: el gradiente tiene un camino directo; así se entrenan cientos de capas.
5. Con pocas etiquetas, transferir le gana a entrenar desde cero; pero se heredan los sesgos.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Convoluciones con Fashion-MNIST y transferencia a imágenes de satélite* (Unidad 6, ejemplo 3).

**Lectura:** Goodfellow, Bengio y Courville, *Deep Learning*, capítulo 9; Zhang et al., *Dive into Deep Learning*, capítulos 7, 8.6 y 14.2; He et al., «Deep residual learning for image recognition» (CVPR 2016).

</div>
</div>
</div>
