---
theme: none
title: ¿Es posible aprender?
unidad: Unidad 3
layout: portada
subtitulo: Hoeffding, la dimensión VC y lo que la teoría garantiza (y lo que no)
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
  Reconocimiento de Patrones · Unidad 3 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# ¿Es posible aprender?

---
layout: pregunta
---

# Un modelo acierta en 95 % de los datos que vio. ¿Qué nos garantiza eso sobre los datos que no ha visto?

<v-click>

Con certeza, nada. Con probabilidad, bastante, si somos cuidadosos. Esta clase es sobre ese «si».

</v-click>

---

# Con certeza no se puede

<div class="cols cols-6-4" style="gap: 48px">
<div>

Tres entradas binarias, $f : \{0,1\}^3 \to \{0, 1\}$. Conocemos $f$ en 5 de los 8 puntos:

<TablaMini :columnas="['x₁', 'x₂', 'x₃', 'f(x)']" :filas="[[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, '?'], [1, 1, 0, '?'], [1, 1, 1, '?']]" :tam="20" />

</div>
<div>

<v-click>

Hay $2^3 = 8$ funciones que coinciden con los 5 datos y difieren en los 3 restantes. Cualquier hipótesis $g$ que elijamos acierta en todas para unas y falla en todas para otras.

</v-click>

<v-click>

<Caja tipo="idea">

Los datos, solos, no dicen nada **fuera de la muestra**. Hace falta una hipótesis más: que los datos se generaron **al azar**, de la misma distribución que los datos futuros.

</Caja>

</v-click>
</div>
</div>

<!--
Es una versión del teorema de «no hay almuerzo gratis» (Wolpert, 1996): promediado sobre todas las f posibles, ningún algoritmo le gana a otro fuera de la muestra.
-->

---
layout: seccion
numero: 1
---

# La desigualdad de Hoeffding

Lo que una muestra dice de una población.

---

# Una urna

<div class="cols cols-centro" style="flex: 1; gap: 40px">
<div>

<Urna />

</div>
<div class="grande">

Una fracción $\mu$ desconocida de canicas es roja. Sacamos $M$ al azar, de manera independiente, y observamos la fracción $\nu$.

<v-click>

¿Qué dice $\nu$ sobre $\mu$?

$\nu$ **podría** ser muy distinta de $\mu$ (sacar puras azules por mala suerte), pero es **poco probable**.

</v-click>
</div>
</div>

---

# La desigualdad de Hoeffding

<div class="cols" style="gap: 48px">
<div>

<Caja tipo="teorema" nombre="Hoeffding, 1963">

Si $X_1, \dots, X_M$ son independientes, con valores en $[0, 1]$ y media $\mu$, y $\nu = \frac{1}{M}\sum_{i=1}^{M} X_i$ es su promedio, entonces para toda tolerancia $\varepsilon > 0$

$$
\Prob\big[\,|\nu - \mu| > \varepsilon\,\big] \;\le\; 2\,e^{-2\varepsilon^2 M} .
$$

</Caja>

</div>
<div>

<v-clicks>

- No depende de $\mu$ ni del tamaño de la urna: solo de $M$ y $\varepsilon$.
- Decrece **exponencialmente** con $M$.
- La afirmación «$\nu = \mu$» es **probablemente aproximadamente correcta** (PAC): aproximadamente, por $\varepsilon$; probablemente, por la cota.
- Pide que $\mu$ esté fija **antes** de sacar la muestra.

</v-clicks>

</div>
</div>

---

# ¿Qué tan buena es la cota?

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<HoeffdingExacto />

</div>
<div>

Con $\mu = 0.5$ y $\varepsilon = 0.1$, la probabilidad **exacta** (con la binomial) contra la cota.

<v-click>

La cota es holgada, pero tiene la **forma correcta**: la misma caída exponencial. Con una muestra de $M = 400$ la probabilidad de equivocarse por más de 0.1 es menor que una en 10 mil.

</v-click>
</div>
</div>

---

# Esbozo de la demostración

<div class="cols" style="gap: 48px">
<div>

Para un lado, con cualquier $s > 0$ y la desigualdad de Markov:

$$
\Prob[\nu - \mu \ge \varepsilon] = \Prob\big[e^{sM(\nu - \mu)} \ge e^{sM\varepsilon}\big] \le e^{-sM\varepsilon}\, \E\big[e^{sM(\nu-\mu)}\big] .
$$

<v-click>

Por independencia, la esperanza se factoriza:

$$
\E\big[e^{sM(\nu-\mu)}\big] = \prod_{i=1}^{M} \E\big[e^{s(X_i - \mu)}\big] .
$$

</v-click>
</div>
<div>

<v-click>

**Lema de Hoeffding:** si $X \in [0, 1]$, entonces $\E[e^{s(X - \E X)}] \le e^{s^2/8}$. Así

$$
\Prob[\nu - \mu \ge \varepsilon] \le e^{-sM\varepsilon + Ms^2/8} .
$$

</v-click>

<v-click>

Se elige el mejor $s$: minimizar el exponente da $s = 4\varepsilon$, y queda $e^{-2\varepsilon^2 M}$. El otro lado es igual; la unión de los dos da el factor 2.

</v-click>
</div>
</div>

---

# De la urna al aprendizaje

<div class="cols cols-centro" style="flex: 1; gap: 56px">
<div class="grande">

Fijemos **una** hipótesis $h$. Cada canica es un punto $\vx$ tomado de la distribución $P$ de los datos, y $f$ es la función desconocida que queremos aprender:

- **roja** si $h(\vx) \ne f(\vx)$;
- **azul** si $h(\vx) = f(\vx)$.

</div>
<div>

<v-click>

Entonces $\mu = \Eout(h)$, la probabilidad de que $h$ falle en un punto nuevo, y $\nu = \Ein(h)$, la fracción de errores en los $M$ datos. Hoeffding dice

$$
\Prob\big[\,|\Ein(h) - \Eout(h)| > \varepsilon\,\big] \le 2\,e^{-2\varepsilon^2 M} .
$$

</v-click>

<v-click>

<Caja tipo="cuidado">

Eso vale para una $h$ fijada **antes** de ver los datos. Pero aprender es justamente **elegir** $h$ después de verlos.

</Caja>

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# Elegir entre muchas hipótesis

---
layout: pregunta
---

# Mil personas lanzan una moneda justa 10 veces cada una. ¿Qué tan probable es que alguna saque puras águilas?

<v-click>

$1 - (1 - 2^{-10})^{1000} \approx 62\,\%$. Más probable que no. Y esa persona no tiene una moneda especial.

</v-click>

---

# Las 1,000 monedas

<Monedas />

<div class="cols" style="gap: 40px; margin-top: 10px">
<div class="pequeno">

Fracción de caras $\nu$ en 20,000 repeticiones del experimento. Las tres monedas son justas, $\mu = 0.5$.

</div>
<div class="pequeno" v-click>

Para la moneda **elegida por tener menos caras**, $\Prob[|\nu - \mu| > 0.35] = 1.0$, aunque Hoeffding diría $\le 0.17$. No hay contradicción: Hoeffding no aplica a una moneda elegida **después** de ver los resultados.

</div>
</div>

---

# La cota de la unión

<div class="cols" style="gap: 48px">
<div>

Con $\Hyp = \{h_1, \dots, h_{|\Hyp|}\}$ finito ($|\Hyp|$ hipótesis), el algoritmo elige $g \in \Hyp$. Si $g$ es «mala», alguna $h_r$ lo es:

$$
\big|\Ein(g) - \Eout(g)\big| > \varepsilon \;\Longrightarrow\; \bigcup_{r=1}^{|\Hyp|} \big\{\, |\Ein(h_r) - \Eout(h_r)| > \varepsilon \,\big\}
$$

<v-click>

$$
\Prob\big[|\Ein(g) - \Eout(g)| > \varepsilon\big] \;\le\; 2\,|\Hyp|\, e^{-2\varepsilon^2 M} .
$$

</v-click>
</div>
<div v-click>

### Buenas noticias

Con $|\Hyp|$ finito, aprender es posible: basta con un número de datos $M$ suficientemente grande.

### Malas noticias

El perceptrón, la regresión logística y casi todos los modelos del curso tienen $|\Hyp| = \infty$.

<div class="pequeno tenue" style="margin-top: 16px">

Pero la unión es muy pesimista: si dos rectas casi coinciden, sus eventos «malos» casi coinciden también, y aun así los sumamos como si fueran ajenos.

</div>
</div>
</div>

---
layout: seccion
numero: 3
---

# Dicotomías y dimensión VC

Contar hipótesis que se distinguen en los datos.

---

# Dicotomías

<Dicotomias />

<div class="cols" style="gap: 40px; margin-top: 18px">
<div class="pequeno">

Una **dicotomía** es lo que una hipótesis hace en los $M$ puntos de la muestra: un vector de etiquetas $\big(h(\vx^{(1)}), \dots, h(\vx^{(M)})\big)$. Infinitas rectas, pero en 3 puntos solo hay $2^3 = 8$ comportamientos distintos, y el perceptrón (clasificar según el lado de una recta) los produce todos.

</div>
<div class="pequeno" v-click>

La **función de crecimiento** cuenta el máximo de dicotomías posibles con $M$ puntos:

$$
m_{\Hyp}(M) = \max_{\vx^{(1)}, \dots, \vx^{(M)}} \big|\Hyp(\vx^{(1)}, \dots, \vx^{(M)})\big| \;\le\; 2^M ,
$$

donde $\Hyp(\vx^{(1)}, \dots, \vx^{(M)})$ es el conjunto de dicotomías que produce $\Hyp$ en esos puntos.

</div>
</div>

---

# Tres ejemplos, contados

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 44px">
<div>

<v-clicks>

- **Rayos positivos** en $\R$, $h(x) = \mathrm{sign}(x - a)$ con umbral $a$: los $M$ puntos dejan $M+1$ huecos donde poner $a$, así que $m_{\Hyp}(M) = M + 1$.
- **Intervalos positivos**: elegir 2 de los $M+1$ huecos, más el intervalo vacío: $\binom{M+1}{2} + 1$.
- **Perceptrón en $\R^2$**: $m_{\Hyp}(M) = M(M-1) + 2$. Rompe 3 puntos y no 4.
- **Conjuntos convexos**: con los puntos en un círculo, cualquier subconjunto es la intersección del círculo con un polígono convexo. $m_{\Hyp}(M) = 2^M$.

</v-clicks>

</div>
<div>

<Crecimiento />

</div>
</div>

---

# El punto de quiebre lo cambia todo

<div class="cols" style="gap: 48px">
<div>

Si ningún conjunto de $k$ puntos se puede **romper** (producir las $2^k$ dicotomías), $k$ es un **punto de quiebre**.

<Caja tipo="teorema" nombre="Sauer–Shelah">

Si $\Hyp$ tiene punto de quiebre $k$, entonces

$$
m_{\Hyp}(M) \;\le\; \sum_{q=0}^{k-1} \binom{M}{q} \;=\; O\big(M^{k-1}\big).
$$

</Caja>

</div>
<div v-click>

**Idea de la prueba.** Sea $B(M, k)$ el máximo de dicotomías en $M$ puntos sin romper ningún subconjunto de $k$. Separando el último punto se obtiene

$$
B(M, k) \le B(M-1, k) + B(M-1, k-1),
$$

que es la recurrencia del triángulo de Pascal. Por inducción, $B(M,k) \le \sum_{q<k}\binom{M}{q}$.

<div class="lema" style="font-size: 24px; margin-top: 18px">Un solo punto de quiebre convierte 2<sup>M</sup> en un <em>polinomio</em>.</div>

</div>
</div>

---

# La dimensión VC

<div class="cols" style="gap: 48px">
<div>

<Caja tipo="definicion">

La **dimensión de Vapnik–Chervonenkis** $d_{\text{VC}}(\Hyp)$ es el mayor $M$ tal que $m_{\Hyp}(M) = 2^M$. Si existe, $k = d_{\text{VC}} + 1$ es un punto de quiebre.

</Caja>

<div v-click="1" class="pequeno" style="margin-top: 16px">

Rayos: 1. Intervalos: 2. Perceptrón en $\R^2$: 3. Convexos: $\infty$.

</div>

<div v-click="3" class="pequeno tenue" style="margin-top: 24px">

Coincide con el número de parámetros, $n + 1$ ($n$ pesos y el sesgo). Por eso se dice que $d_{\text{VC}}$ mide los **grados de libertad efectivos**.

</div>
</div>
<div v-click="2">

**El perceptrón en $\R^n$**, $h(\vx) = \mathrm{sign}(\vw\T\vx + b)$ con pesos $\vw \in \R^n$ y sesgo $b$, **tiene $d_{\text{VC}} = n + 1$**.

- **$\ge n+1$:** los $n+1$ puntos $\mathbf{0}, \mathbf{e}_1, \dots, \mathbf{e}_n$ ($\mathbf{e}_j$, el $j$-ésimo vector canónico) se rompen: para cualesquiera etiquetas $y_0, y_1, \dots, y_n \in \{-1, +1\}$, basta tomar $b = y_0$ y $w_j = y_j - y_0$.
- **$\le n+1$:** por el teorema de Radon, $n + 2$ puntos de $\R^n$ siempre se parten en dos grupos cuyas envolventes convexas se cortan; ningún hiperplano los separa, así que esa dicotomía es imposible.

</div>
</div>

---

# La cota VC

<div class="cols" style="gap: 48px">
<div>

<Caja tipo="teorema" nombre="Vapnik y Chervonenkis, 1971">

Con probabilidad al menos $1 - \delta$ sobre la muestra de $M$ datos, para toda $g \in \Hyp$,

$$
\Eout(g) \;\le\; \Ein(g) + \underbrace{\sqrt{\frac{8}{M} \ln \frac{4\, m_{\Hyp}(2M)}{\delta}}}_{\Omega(M,\, \Hyp,\, \delta)} .
$$

</Caja>

</div>
<div>

<v-clicks>

- Es la cota de la unión con $|\Hyp|$ sustituido por $m_{\Hyp}(2M)$: un polinomio si $d_{\text{VC}} < \infty$.
- El $2M$ viene del truco de la prueba: comparar $\Ein$ con el error en una **segunda muestra fantasma**, que también es finita.
- Vale para **cualquier** algoritmo, **cualquier** distribución $P$ y **cualquier** $f$. Esa generalidad la hace muy holgada.

</v-clicks>

</div>
</div>

---

# ¿Cuántos datos necesitamos?

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<CotaVC />

</div>
<div>

Con $d_{\text{VC}} = 3$ y $\delta = 0.05$ (95 % de confianza), la cota garantiza una brecha $\Omega$ de 0.1 hasta con unos **30 mil** datos.

<v-click>

En la práctica bastan muchos menos. La regla empírica:

$$
M \;\gtrsim\; 10 \cdot d_{\text{VC}} .
$$

</v-click>

<v-click>

La libreta 1 compara la cota con la brecha real en simulaciones.

</v-click>
</div>
</div>

---

# El compromiso entre aproximar y generalizar

<div class="cols cols-6-4" style="gap: 40px">
<div>

<CompromisoVC />

<p class="tenue" style="font-size: 16px">Esquema, no datos.</p>

</div>
<div>

$\Hyp$ más grande:

- **$\Ein$ baja**: más posibilidades de acercarse a $f$.
- **$\Omega$ sube** (la holgura de la cota VC): más posibilidades de engañarse con los datos.

<v-click>

El mejor $\Hyp$ no es el más grande ni el más chico. Es la versión teórica de lo que pasa al subir el grado de un polinomio: se ajusta cada vez mejor a los datos y, pasado cierto punto, generaliza cada vez peor.

</v-click>
</div>
</div>

---

# Lo que la teoría no dice

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 36px">
<div>

### Que el error de entrenamiento será pequeño

La cota solo acerca $\Eout$ a $\Ein$. Que $\Ein$ sea pequeño depende de si $\Hyp$ contiene algo parecido a $f$ y de que el algoritmo lo encuentre.

</div>
<div v-click>

### Cuánto vale el error fuera de la muestra

Para eso usamos **datos de validación**, que el modelo no vio: dan una cota de Hoeffding para una sola hipótesis, mucho más ajustada.

</div>
<div v-click>

### Por qué funcionan las redes profundas

Una red grande puede memorizar etiquetas **aleatorias** de ImageNet: su $d_{\text{VC}}$ es enorme, y aun así generaliza con las etiquetas reales. Es una pregunta abierta.

<p class="tenue" style="font-size: 16px">C. Zhang et al., «Understanding deep learning requires rethinking generalization», ICLR (2017).</p>

</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Sin suponer que los datos son una muestra aleatoria, no se puede aprender.
2. **Hoeffding:** para una $h$ fija, $\Ein \approx \Eout$ con probabilidad que crece exponencialmente con $M$.
3. Elegir entre $|\Hyp|$ hipótesis cuesta un factor $|\Hyp|$ (las 1,000 monedas).
4. Lo que importa no es $|\Hyp|$ sino las **dicotomías**: un punto de quiebre las vuelve un polinomio.
5. **Cota VC:** $\Eout \le \Ein + \Omega$; más capacidad, más datos.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *¿Es posible aprender? Hoeffding y dimensión VC, con simulaciones* (Unidad 3, ejemplo 1).

**Lectura:** Abu-Mostafa, Magdon-Ismail y Lin, *Learning From Data* (2012), capítulos 1.3 y 2.1.

</div>
</div>
</div>
