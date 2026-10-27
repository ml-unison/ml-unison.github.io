---
theme: none
title: Mezclas gaussianas y EM
unidad: Unidad 2
layout: portada
subtitulo: Grupos con forma, asignaciones con incertidumbre y un algoritmo que aparece en todas partes
fecha: 2027-1
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
  Reconocimiento de Patrones · Unidad 2 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Mezclas gaussianas y el algoritmo EM

---
layout: pregunta
---

# Un pingüino queda justo entre dos grupos. ¿A cuál pertenece?

<v-click>

k-medias lo obliga a elegir. Quizá la respuesta honesta es «60 % a uno, 40 % al otro».

</v-click>

---

# Donde k-medias se queda corto

<KMediasVsMezcla />

<div v-click class="nota">

Los grupos de pingüinos son **alargados e inclinados**: el pico largo va con el pico profundo dentro de cada especie. k-medias solo sabe trazar fronteras rectas a medio camino entre centros. Hoy: un modelo en el que cada grupo tiene su propia **forma**.

</div>

---
layout: seccion
numero: 1
---

# Un modelo de cómo se generaron los datos

---

# Una sola variable: el largo de la aleta

<Mezcla1D :componentes="$clicks >= 1" />

<div class="cols" style="gap: 40px">
<div class="pequeno">

Dos jorobas: el histograma no parece **una** campana.

</div>
<div v-click class="pequeno">

Pero sí la suma de **dos** campanas con pesos $\pi_1 + \pi_2 = 1$. (Las especies Adelie y Chinstrap tienen aletas parecidas y quedan juntas.)

</div>
</div>

---

# La historia que cuenta el modelo

<div class="cols" style="gap: 56px">
<div>

Para generar un pingüino:

<v-clicks>

1. La naturaleza elige una componente oculta $z \in \{1, \dots, K\}$ con probabilidad $\pi_k$.
2. Dada la componente, las medidas salen de una normal: $\;\vx \mid z = k \;\sim\; \mathcal{N}(\mu_k, \Sigma_k)$.

</v-clicks>

<div v-click>

Al observar solo $\vx$, sumamos sobre lo que no vemos:

$$
p(\vx) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\vx \mid \mu_k, \Sigma_k) .
$$

</div>
</div>
<div v-click>

<Caja tipo="idea">

Un **modelo generativo**: describe cómo podrían haberse producido los datos. Los parámetros son $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$.

Agrupar se vuelve un problema de **estimación**: encontrar el $\theta$ que hace más probables los datos observados.

</Caja>

</div>
</div>

---

# Asignaciones suaves: la regla de Bayes

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<Mezcla1D responsabilidad />

</div>
<div>

La **responsabilidad** de la componente $k$ sobre el punto $i$ es la probabilidad de que lo haya generado ella, dado lo que medimos:

$$
\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\vx_i \mid \mu_k, \Sigma_k)}{\sum_{j} \pi_j \, \mathcal{N}(\vx_i \mid \mu_j, \Sigma_j)}
$$

<p class="pequeno tenue">Lejos de la frontera es casi 0 o 1, como en k-medias. Cerca, expresa la duda.</p>

</div>
</div>

<!--
Mover el deslizador por la zona de 200–210 mm: la responsabilidad cambia suavemente. Lejos de la frontera es casi 0 o 1, como k-medias.
-->

---
layout: seccion
numero: 2
---

# El algoritmo EM

---

# El problema del huevo y la gallina

<div class="cols" style="gap: 48px">
<div>

### Si supiéramos las componentes $z_i$…

…estimar sería fácil: para cada grupo, su proporción, su media y su covarianza.

$$
\hat\mu_k = \frac{1}{n_k} \sum_{z_i = k} \vx_i
$$

</div>
<div v-click>

### Si supiéramos los parámetros $\theta$…

…las componentes también: las responsabilidades $\gamma_{ik}$ salen de la regla de Bayes.

</div>
</div>

<div v-click class="lema" style="margin-top: 40px; font-size: 28px">
No sabemos ninguna de las dos. Hagamos lo mismo que en k-medias: <em>alternar</em>.
</div>

---

# Esperanza y maximización

<div class="cols" style="gap: 48px">
<div>

<Caja tipo="algoritmo" nombre="EM para mezclas gaussianas">

**Paso E.** Con $\theta$ fijo, calcular las responsabilidades $\gamma_{ik}$.

**Paso M.** Con las $\gamma_{ik}$ fijas, reestimar como promedios **ponderados**, con $N_k = \sum_i \gamma_{ik}$:

$$
\pi_k = \frac{N_k}{n}, \qquad \mu_k = \frac{1}{N_k} \sum_i \gamma_{ik} \vx_i
$$

$$
\Sigma_k = \frac{1}{N_k} \sum_i \gamma_{ik} (\vx_i - \mu_k)(\vx_i - \mu_k)\T
$$

Repetir hasta que la log-verosimilitud deje de subir.

</Caja>

</div>
<div v-click>

### Comparado con k-medias

| | k-medias | EM |
|---|---|---|
| asignación | dura | suave, $\gamma_{ik}$ |
| centro | media | media ponderada |
| forma | esfera | $\Sigma_k$ |
| proporción | — | $\pi_k$ |

<p class="pequeno tenue">A. Dempster, N. Laird y D. Rubin, <em>J. Royal Statistical Society B</em> 39 (1977).</p>

</div>
</div>

---

# EM, paso a paso

<EMPasos />

<!--
Recorrer las iteraciones con los botones. En la 0, tres círculos iguales. Notar que la log-verosimilitud sube mucho al principio y luego se aplana; tarda más de 100 iteraciones en las últimas décimas.
-->

---

# ¿Por qué la verosimilitud nunca baja?

<div class="cols" style="grid-template-columns: 11fr 9fr; gap: 44px">
<div>

Para cualquier distribución $q_i$ sobre las componentes, por la desigualdad de Jensen ($\log$ es cóncava):

$$
\log p(\vx_i \mid \theta) \;\ge\; \sum_k q_{ik} \log \frac{p(\vx_i, z_i{=}k \mid \theta)}{q_{ik}} .
$$

Sumando sobre $i$, el lado derecho $\mathcal{L}(q, \theta)$ es una **cota inferior** de la log-verosimilitud $\ell(\theta)$.

</div>
<div>

<v-clicks>

- **E:** con $q_{ik} = \gamma_{ik}$ la cota es exacta, $\mathcal{L}(q, \theta^{t}) = \ell(\theta^{t})$.
- **M:** maximizar $\mathcal{L}(q, \theta)$ en $\theta$ da las fórmulas ponderadas.
- Entonces

</v-clicks>

<div v-click="3">

$$
\ell(\theta^{t+1}) \ge \mathcal{L}(q, \theta^{t+1}) \ge \mathcal{L}(q, \theta^{t}) = \ell(\theta^{t}).
$$

</div>
</div>
</div>

<div v-click="4" class="nota">

**Cuidado:** sube siempre, pero hacia un **máximo local**. Con otro inicio, los mismos pingüinos terminan en $\ell = -862$ (en vez de $-822$) y con ARI de 0.49. Por eso se corre varias veces (`n_init`) o se inicia con k-medias.

</div>

---

# k-medias es un caso límite de EM

<div class="cols cols-centro" style="flex: 1; gap: 56px">
<div class="grande">

Fijemos todas las covarianzas en $\Sigma_k = \sigma^2 I$ y los pesos iguales. Entonces

$$
\gamma_{ik} = \frac{\exp\!\big(-\norm{\vx_i - \mu_k}^2 / 2\sigma^2\big)}{\sum_j \exp\!\big(-\norm{\vx_i - \mu_j}^2 / 2\sigma^2\big)} .
$$

</div>
<div v-click class="grande">

Si $\sigma \to 0$, el término del centro **más cercano** domina: $\gamma_{ik} \to 1$ para ese centro y $0$ para los demás.

El paso E se vuelve «asignar» y el paso M, «mover». **Es k-medias.**

</div>
</div>

---

# La forma de los grupos: tipos de covarianza

<Covarianzas />

<div v-click class="nota">

Más libertad ajusta mejor, pero cuesta parámetros: en $d$ dimensiones, cada covarianza completa tiene $d(d+1)/2$. Con $d = 100$ son 5,050 por componente.

</div>

---
layout: seccion
numero: 3
---

# ¿Cuántas componentes?

---

# La verosimilitud siempre prefiere más

<div class="cols" style="gap: 48px">
<div>

Agregar una componente nunca baja la verosimilitud máxima. En el extremo, una componente pegada a cada punto con $\Sigma_k \to 0$ da $\ell \to \infty$.

<v-click>

El **criterio de información bayesiano** castiga la complejidad:

$$
\text{BIC} = -2\,\ell(\hat\theta) + p \log n ,
$$

con $p$ el número de parámetros libres y $n$ el de datos. Menor es mejor.

</v-click>
</div>
<div v-click>

<Caja tipo="cuidado" nombre="singularidades">

Una componente puede encogerse sobre un solo punto: su densidad allí tiende a infinito y EM «mejora» sin límite con un modelo absurdo. La defensa práctica es sumar una constante pequeña a la diagonal de cada $\Sigma_k$ (en scikit-learn, `reg_covar`).

</Caja>

</div>
</div>

---

# BIC con las cuatro medidas de los pingüinos

<div class="cols cols-6-4" style="gap: 40px">
<div>

<CurvaBIC />

</div>
<div>

El mejor modelo es de covarianza **compartida** y $k = 4$, no 3.

<v-click>

¿Un error? Dos componentes parten a los Adelie en dos, en buena parte **por sexo y tamaño**: una tiene sobre todo hembras y la otra sobre todo machos. Para el modelo es estructura real.

</v-click>

<v-click>

<div class="lema" style="font-size: 24px; margin-top: 20px">Lo que es un «grupo» depende de las variables y del nivel de detalle que nos interesa.</div>

</v-click>
</div>
</div>

---

# Un regalo: una densidad

<div class="cols" style="gap: 48px">
<div class="grande">

La mezcla no solo agrupa: da $p(\vx)$ para **cualquier** punto.

<v-clicks>

- **Detectar atípicos:** los puntos con $p(\vx)$ muy baja. El pingüino más raro es una Chinstrap con pico de 58 mm y aleta de 181 mm.
- **Generar** datos nuevos: muestrear $z$ y luego $\vx$.
- **Clasificar** con una mezcla por clase (Unidad 4).

</v-clicks>
</div>
<div v-click>

<Caja tipo="idea">

EM sirve para cualquier modelo con **variables ocultas**: modelos ocultos de Markov, temas en documentos, datos faltantes. La receta es siempre la misma: completar lo oculto con su esperanza y maximizar.

</Caja>

</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 26px; line-height: 1.4">

1. Una mezcla gaussiana es un **modelo generativo**: una componente oculta y luego una normal.
2. Las **responsabilidades** son asignaciones suaves.
3. **EM** alterna completar lo oculto (E) y reestimar (M): la verosimilitud no baja, pero hay máximos locales.
4. k-medias es el caso límite con $\Sigma_k = \sigma^2 I$, $\sigma \to 0$.
5. Elegir $k$ y la covarianza con **BIC**; luego interpretar.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Mezclas gaussianas y el algoritmo EM* (Unidad 2, ejemplo 2): EM en 30 líneas de NumPy.

**Lectura:** C. Bishop, *Pattern Recognition and Machine Learning* (2006), secciones 9.2 y 9.3.

</div>
</div>
</div>
