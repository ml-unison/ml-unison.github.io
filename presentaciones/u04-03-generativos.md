---
theme: none
title: Modelos generativos
unidad: Unidad 4
layout: portada
subtitulo: "Modelar cómo se generan los datos de cada clase: análisis discriminante y Bayes ingenuo"
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
  Reconocimiento de Patrones · Unidad 4 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Modelos generativos

---
layout: pregunta
---

# Para reconocer el spam, ¿hay que aprender la frontera entre spam y no spam, o aprender cómo se escribe cada uno?

<v-click>

Las dos cosas funcionan. La primera es el enfoque **discriminativo**, como la logística; la segunda, el **generativo**. Hoy, el segundo.

</v-click>

---

# Dos maneras de llegar a $P(y \mid \vx)$

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

**Discriminativo.** Modelar directamente $P(y \mid \vx)$, por ejemplo $\theta(\vw\T\vx)$, y ajustar con la verosimilitud condicional.

<v-click>

**Generativo.** Modelar cómo se generan los datos de cada clase, $p(\vx \mid y = k)$, y qué tan común es cada una, $\pi_k = P(y = k)$. Luego, Bayes:

$$
P(y = k \mid \vx) = \frac{\pi_k\, p(\vx \mid y = k)}{\sum_j \pi_j\, p(\vx \mid y = j)} .
$$

</v-click>
</div>
<div v-click>

<Caja tipo="proposicion">

Con pérdida 0-1, el clasificador que minimiza el error es el **de Bayes**:

$$
g^*(\vx) = \argmax_k \; \pi_k\, p(\vx \mid y = k) .
$$

</Caja>

<div class="pequeno" style="margin-top: 14px">

No lo conocemos: hay que estimar $\pi_k$ y $p(\vx \mid y = k)$. Cada supuesto sobre $p(\vx \mid y)$ da un clasificador distinto.

</div>
</div>
</div>

---
layout: seccion
numero: 1
---

# Análisis discriminante

Una gaussiana por clase.

---

# LDA: covarianza común, frontera lineal

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Supuesto: $\vx \mid y = k \sim \mathcal{N}(\boldsymbol\mu_k, \Sigma)$, con **la misma** $\Sigma$ para todas las clases. Los estimadores de máxima verosimilitud son las medias por clase, la covarianza agrupada y las proporciones.

<v-click>

$$
\ln \pi_k p(\vx \mid k) = -\tfrac12 (\vx - \boldsymbol\mu_k)\T \Sigma^{-1} (\vx - \boldsymbol\mu_k) + \ln \pi_k + c
$$

</v-click>
</div>
<div>

<v-click>

Al expandir, $-\tfrac12 \vx\T \Sigma^{-1} \vx$ es igual para todas las clases y **se cancela** al comparar. Queda

$$
\delta_k(\vx) = \vx\T \Sigma^{-1} \boldsymbol\mu_k - \tfrac12 \boldsymbol\mu_k\T \Sigma^{-1} \boldsymbol\mu_k + \ln \pi_k ,
$$

**lineal** en $\vx$.

</v-click>

<v-click>

La frontera entre $k$ y $l$ es el hiperplano con normal $\Sigma^{-1}(\boldsymbol\mu_k - \boldsymbol\mu_l)$. Con dos clases, $P(y = 1 \mid \vx)$ es **exactamente una logística**.

</v-click>
</div>
</div>

---

# Los pingüinos, con LDA

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<RegionesPinguinos tipo="lda" />

</div>
<div>

342 pingüinos de tres especies y dos medidas del pico (Unidad 2, ahora **con** etiquetas).

<v-clicks>

- Las tres elipses son **la misma**, trasladada: es la covarianza agrupada.
- Las fronteras son rectas.
- Exactitud por validación cruzada: **96.5 %**.

</v-clicks>
</div>
</div>

---

# QDA: una covarianza por clase

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 32px">
<div>

<RegionesPinguinos tipo="qda" />

</div>
<div>

Con $\Sigma_k$ distinta, el término $\vx\T \Sigma_k^{-1}\vx$ ya no se cancela: fronteras **cuadráticas**.

<v-click>

| | parámetros ($d=2$, $K=3$) | exactitud (CV) |
|---|---|---|
| LDA | $Kd + \frac{d(d+1)}{2} = 11$ | 96.5 % |
| QDA | $K\big(d + \frac{d(d+1)}{2}\big) = 17$ | 95.9 % |
| logística | $(K-1)(d+1) = 6$ | 95.0 % |

</v-click>

<v-click>

<div class="pequeno">

Con imágenes de 28 × 28 y 10 clases, QDA necesitaría **3 millones** de parámetros: más supuestos, menos varianza.

</div>

</v-click>
</div>
</div>

---

# LDA también reduce la dimensión

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

**Fisher (1936):** buscar la dirección $\vw$ que separe las medias de las clases en relación con la dispersión dentro de ellas,

$$
J(\vw) = \frac{\vw\T S_B\, \vw}{\vw\T S_W\, \vw} .
$$

</div>
<div>

<v-click>

Igualando el gradiente a cero: $S_B \vw = \lambda S_W \vw$, un problema de valores propios generalizado. Como $S_B$ tiene rango $\le K - 1$, hay a lo más $K - 1$ direcciones útiles.

</v-click>

<v-click>

Con las 4 medidas: $\lambda = 15.0$ y $2.3$. PCA (Unidad 2) busca varianza **sin** ver las etiquetas; LDA, separación **con** ellas.

</v-click>
</div>
</div>

---

# PCA contra LDA en los pingüinos

<FisherPCA />

<div class="pequeno" style="margin-top: 14px">

Las cuatro medidas estandarizadas, proyectadas a dos dimensiones. El primer eje de LDA concentra el 87 % de la separación entre especies: Gentoo queda aislado y Adelie y Chinstrap se separan en el segundo eje.

</div>

---
layout: seccion
numero: 2
---

# Bayes ingenuo

Cuando $\vx$ tiene miles de dimensiones.

---

# El problema: filtrar spam en SMS

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div class="grande">

- 5,574 mensajes SMS en inglés; **13.4 %** son spam.
- Vienen de un foro británico donde la gente denunciaba spam y de un corpus de mensajes de estudiantes de Singapur.
- **Problema:** decidir si un mensaje nuevo es spam.

</div>
<div>

Cada mensaje se representa como **bolsa de palabras**: un vector $\vx \in \mathbb{N}^{V}$ con cuántas veces aparece cada palabra del vocabulario. Aquí $V = 7{,}515$.

<p class="pequeno tenue" style="margin-top: 26px">T. A. Almeida, J. M. Gómez Hidalgo y A. Yamakami, <em>Proc. ACM DocEng</em> (2011). UCI, CC BY 4.0.</p>

</div>
</div>

---

# El supuesto ingenuo

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Una gaussiana en 7,515 dimensiones tendría unos 28 millones de parámetros. En cambio, suponemos que **dada la clase**, las palabras son independientes:

$$
p(\vx \mid y) \propto \prod_{j=1}^{V} \theta_{yj}^{\,x_j}, \qquad \sum_j \theta_{yj} = 1 .
$$

<v-click>

El estimador de máxima verosimilitud es la frecuencia: $\hat\theta_{yj} = N_{yj} / N_y$, con $N_{yj}$ las apariciones de la palabra $j$ en la clase $y$.

</v-click>
</div>
<div>

<v-click>

**El problema de los ceros.** «lor» aparece 121 veces en mensajes normales y **ninguna** en spam: $\hat\theta_{\text{spam}, \text{lor}} = 0$ y un solo «lor» veta para siempre la clase spam.

</v-click>

<v-click>

**Suavizado de Laplace:**

$$
\hat\theta_{yj} = \frac{N_{yj} + \alpha}{N_y + \alpha V} .
$$

</v-click>

<v-click>

<div class="pequeno">

**En logaritmos:** un mensaje de spam tiene $\prod \theta^{x}$ del orden de $10^{-85}$. Se suma $\ln \pi_y + \sum_j x_j \ln\hat\theta_{yj}$ y se normaliza con *logsumexp*.

</div>

</v-click>
</div>
</div>

---

# Qué aprendió

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<PalabrasSpam />

</div>
<div>

$\ln \hat\theta_{\text{spam},j} - \ln \hat\theta_{\text{normal},j}$ para las palabras más extremas.

<v-click>

| en 1,394 mensajes de prueba | Bayes ingenuo | logística |
|---|---|---|
| exactitud | **98.4 %** | 98.0 % |
| precisión | 96.6 % | 99.4 % |
| exhaustividad | **91.4 %** | 85.6 % |

</v-click>

<v-click>

<div class="nota">

«gt» y «lt» vienen de la marca `&lt;#&gt;` que sustituyó números en parte de los mensajes normales: el modelo aprende **cómo se recolectaron** los datos. Habría que limpiarla antes de usarlo.

</div>

</v-click>
</div>
</div>

---

# Bayes ingenuo es un clasificador lineal

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

$$
\ln \frac{P(\text{spam} \mid \vx)}{P(\text{normal} \mid \vx)} = \underbrace{\ln \frac{\pi_{\text{spam}}}{\pi_{\text{normal}}}}_{b} + \sum_j x_j \underbrace{\ln \frac{\hat\theta_{\text{spam},j}}{\hat\theta_{\text{normal},j}}}_{w_j}
$$

<v-click>

**La misma forma que la logística**: $\theta(b + \vw\T\vx)$.

</v-click>
</div>
<div>

<v-click>

La diferencia está en **cómo se estiman** los pesos:

- Bayes ingenuo cuenta, palabra por palabra, como si fueran independientes. Forma cerrada, una pasada por los datos.
- La logística ajusta todos los pesos **juntos** para maximizar $P(y \mid \vx)$: si dos palabras siempre aparecen juntas, no cuenta la evidencia dos veces.

</v-click>

<v-click>

<div class="pequeno tenue">

En los SMS, la correlación entre los dos vectores de pesos es solo 0.54.

</div>

</v-click>
</div>
</div>

---

# Generativo contra discriminativo

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<AprendizajeNBLR />

</div>
<div>

Promedio de 20 submuestras de entrenamiento por tamaño; los mismos 1,394 mensajes de prueba.

<v-click>

**Ng y Jordan (2001):** el generativo llega a **su** error asintótico con del orden de $\log V$ ejemplos; el discriminativo necesita del orden de $V$, pero su error asintótico es menor o igual.

</v-click>

<v-click>

<div class="nota">

Aquí Bayes ingenuo gana en todo el rango: con 4,180 mensajes todavía no se ve el cruce. Que la teoría prediga un cruce no garantiza verlo con los datos que hay.

</div>

</v-click>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Generativo: modelar $p(\vx \mid y)$ y $\pi_y$ y aplicar Bayes; discriminativo: modelar $P(y \mid \vx)$.
2. LDA (covarianza común) da fronteras lineales; QDA, cuadráticas con muchos más parámetros.
3. LDA de Fisher es una reducción de dimensión **supervisada**.
4. Bayes ingenuo: independencia condicional, suavizado de Laplace y logaritmos.
5. Más supuestos = menos datos necesarios, si los supuestos no están muy lejos de la verdad.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Modelos generativos: análisis discriminante y Bayes ingenuo* (Unidad 4, ejemplo 3).

**Laboratorio 4:** Bayes ingenuo multinomial desde cero.

**Lectura:** *The Elements of Statistical Learning*, secciones 4.3 y 6.6.3; A. Ng y M. Jordan, «On discriminative vs. generative classifiers», NIPS (2001).

</div>
</div>
</div>
