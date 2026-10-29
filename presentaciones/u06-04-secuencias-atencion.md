---
theme: none
title: "Secuencias: redes recurrentes, atención y transformers"
unidad: Unidad 6
layout: portada
subtitulo: "Un modelo de lenguaje del Quijote, letra por letra"
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

# Secuencias: redes recurrentes, atención y *transformers*

---
layout: pregunta
---

# «En un lugar de la Man…» ¿Qué letra sigue? ¿Y después de «dijo »?

<v-click>

La primera, casi segura: una «c». La segunda puede ser casi cualquiera. Un **modelo de lenguaje** es eso: una distribución de probabilidad para lo que sigue, dado lo que ya se leyó.

</v-click>

<!--
Dejar que el grupo conteste en voz alta. La idea que hay que sacar: la incertidumbre depende del contexto, y a veces el contexto útil está lejos ("dijo" pide un sujeto; quién habla se decidió líneas atrás).
-->

---

# Predecir la siguiente letra

<div class="cols" style="grid-template-columns: minmax(0, 9fr) minmax(0, 11fr); gap: 40px">
<div>

**Datos:** el *Quijote* completo (Proyecto Gutenberg), **2,110,727 caracteres** de **92 símbolos**: letras, acentos, puntuación, rayas de diálogo, espacios y saltos de línea.

<v-click>

**Partición por posición, no al azar:** el primer 90 % (1,899,654 letras) para entrenar; el último 10 % (los capítulos LIX a LXXIV de la segunda parte) para validar.

</v-click>

<v-click>

<div class="pequeno tenue">

Partir letra por letra al azar dejaría a las vecinas de cada letra de validación en el entrenamiento: mediríamos memoria, no predicción.

</div>

</v-click>
</div>
<div>

<v-click>

Con $x_t$ la letra en la posición $t = 1, \dots, T$ (subíndice $t$, como en una serie de tiempo) y $x_{<t}$ las anteriores, por la regla de la cadena, **sin suponer nada**:

$$
\Prob(x_1, \dots, x_T) = \prod_{t=1}^{T} \Prob(x_t \mid x_{<t}) .
$$

</v-click>

<v-click>

Cada factor es un clasificador de 92 clases cuya «etiqueta» es la letra que sigue (aprendizaje **autosupervisado**); se entrena minimizando la **entropía cruzada** promedio ($\vth$: los parámetros):

$$
L(\vth) = -\frac1T \sum_{t=1}^{T} \ln \Prob_{\vth}(x_t \mid x_{<t}) .
$$

</v-click>
</div>
</div>

---

# Perplejidad: el número efectivo de opciones

<div class="cols" style="grid-template-columns: 9fr 11fr; gap: 48px">
<div>

$$
\text{perplejidad} = e^{L} = \Big(\prod_{t=1}^{T} \frac{1}{\Prob_{\vth}(x_t \mid x_{<t})}\Big)^{1/T}
$$

con $L$ la entropía cruzada promedio en las $T$ letras. Si el modelo duda **por igual entre $m$ opciones** (y la correcta está entre ellas), la perplejidad es exactamente $m$.

<v-clicks>

- No saber nada (uniforme): **92**.
- Solo las frecuencias de cada letra: **21.1**.
- Mirar la letra anterior (bigrama): **10.0**.

</v-clicks>

<div v-click class="pequeno tenue">

En bits por carácter: $\log_2(\text{perplejidad})$.

</div>
</div>
<div>

<SiguienteLetra />

</div>
</div>

<!--
Las gráficas son del bigrama de Laplace. Después de una "q" hay prácticamente una sola opción (perplejidad local 1.04); después de un espacio, una veintena (21.8). La perplejidad de un modelo es el promedio geométrico de esas dudas sobre todo el texto.
-->

---

# $n$-gramas: contar contextos

<div class="cols" style="grid-template-columns: 11fr 9fr; gap: 40px">
<div>

<div style="width: 90%; margin: 0 auto">
<NgramasCurva v-if="$clicks < 1" :suave="false" />
<NgramasCurva v-else />
</div>

<div v-click="2" class="nota">

Con 7 letras de contexto, el **18 %** de los 7-gramas de validación nunca aparece en 1.9 millones de letras. «dijo don Quijo» y «dijo don Quij» son filas **sin relación**: hace falta compartir información entre contextos parecidos.

</div>
</div>
<div>

Un $n$-grama usa como contexto $c$ las $n - 1$ letras anteriores, con el **suavizado de Laplace** de Bayes ingenuo:

$$
\hat \Prob(x \mid c) = \frac{\#(c, x) + \alpha}{\#(c) + \alpha V} ,
$$

con $\#(c, x)$ las veces que la letra $x$ sigue al contexto $c$, $\#(c)$ las veces que aparece $c$, $V = 92$ símbolos y $\alpha > 0$ el suavizado.

<div v-click="1">

Con $\alpha = 0.01$ el mejor es $n = 5$: **4.40**. Después, la validación empeora: la curva típica del sobreajuste.

</div>
</div>
</div>

<!--
Elegimos α mirando la validación; con rigor habría que reservar un tercer conjunto. El 5-grama con α = 0.01 guarda 137,848 conteos distintos: es un rival serio y lo usaremos como línea base.
-->

---
layout: seccion
numero: 1
---

# Redes recurrentes

Un estado que resume todo lo leído.

---

# Una red recurrente y su gradiente

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 48px">
<div>

$$
\mathbf{h}_t = \tanh\big(\vW\mathbf{h}_{t-1} + \mathbf U\mathbf{e}_{x_t} + \mathbf{b}_h\big)
$$

y se predice con $\operatorname{softmax}(\mathbf A\mathbf{h}_t + \mathbf{b}_y)$. $\mathbf h_t$ es el **estado** tras leer la letra $t$; $\mathbf{e}_{x_t}$, el **encaje** aprendido de la letra (un vector); $\vW, \mathbf U, \mathbf A$ son matrices de pesos y $\mathbf b_h, \mathbf b_y$ sesgos, los mismos en **todas** las posiciones.

</div>
<div v-click>

Retropropagación **en el tiempo**:

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{s=t+1}^{T} \operatorname{diag}\big(1 - \mathbf{h}_s^2\big)\, \vW
$$

($1 - \mathbf h_s^2$ es la derivada de $\tanh$). Como $0 < 1 - h^2 \le 1$, cada factor tiene norma $\le \norm{\vW}_2$, el mayor valor singular de $\vW$: la norma del producto queda acotada por $\norm{\vW}_2^{\,T-t}$, **exponencial** en la distancia $T - t$.

</div>
</div>

---

# El gradiente se desvanece o explota

<div class="cols" style="grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 40px">
<div>

<JacobianasRNN />

</div>
<div>

RNN de 256 unidades con $\vW = \rho\,\mathbf O$: $\mathbf O$ ortogonal y $\rho$ su escala, que fija $\norm{\vW}_2 = \rho$.

<v-click>

Con $\rho = 0.8$, $\norm{\partial \mathbf h_T / \partial \mathbf h_t}$ cae a $10^{-10}$ a 50 letras; con $\rho = 3$, sube a $10^{13}$ a 100.

</v-click>
</div>
</div>

<!--
Es el mismo desvanecimiento del gradiente de las redes profundas, agravado porque se multiplica la misma matriz una y otra vez. Con ρ = 1 también se desvanece: las derivadas de tanh son menores que 1. La franja buena (aquí cerca de ρ = 1.5) es angosta y depende de cuánto se saturan las tanh, o sea, de los datos.
-->

---

# Contra la explosión: recortar el gradiente

<div class="cols" style="grid-template-columns: 9fr 11fr; gap: 40px">
<div>

Si la norma del gradiente completo $\mathbf g$ (respecto a todos los parámetros) pasa de un umbral $\gamma$, se reescala:

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\Big(1, \frac{\gamma}{\norm{\mathbf{g}}}\Big) .
$$

La dirección se conserva; el paso no se dispara.

<v-click>

En nuestra LSTM, con $\gamma = 1$: mediana **0.13**, y el recorte actuó en **2 de 1,200 pasos**, los dos al principio.

</v-click>

<v-click>

<div class="nota">

Es un seguro barato. Contra el **desvanecimiento** no sirve reescalar: hay que cambiar la arquitectura.

</div>

</v-click>
</div>
<div>

<NormaRecorte />

</div>
</div>

---

# LSTM: una autopista para el gradiente

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

Una **celda** de memoria $\mathbf c_t$ junto al estado $\mathbf h_t$. Tres **compuertas** con valores en $(0, 1)$, función de $\mathbf{h}_{t-1}$ y $\mathbf{e}_{x_t}$: olvido $\mathbf{f}_t$, entrada $\mathbf{i}_t$, salida $\mathbf{o}_t$; y un candidato $\tilde{\mathbf{c}}_t$ ($\odot$: producto entrada por entrada).

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

<v-click>

Por el camino de la celda, $\dfrac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \operatorname{diag}(\mathbf{f}_t)$: **sin $\vW$ ni $\tanh$**. Si $\mathbf f_t \approx 1$, el gradiente pasa casi intacto.

</v-click>

<v-click>

<div class="pequeno tenue">

Hochreiter y Schmidhuber (1997). Es la misma idea que las conexiones residuales, donde cada bloque suma a su entrada $\va$ una corrección $F(\va)$: $\partial(\va + F(\va))/\partial\va = \mathbf I + \cdots$

</div>

</v-click>
</div>
<div v-click>

<GradienteDistancia :series="['lstm_ini', 'lstm']" />

<div class="pequeno tenue">

Influencia de cada letra en el gradiente de la última predicción. Entrenada: $4.5 \times 10^{-4}$ a 30 letras y $1.5 \times 10^{-5}$ a 120.

</div>
</div>
</div>

<!--
Sin entrenar, la compuerta de olvido (una logística) vale cerca de σ(0) = 0.5 y la LSTM se comporta como una RNN con ρ < 1: 10⁻¹² a 60 letras, y a 120 el número se vuelve cero en punto flotante. Entrenada, aprendió a abrir la autopista en algunas unidades. La LSTM (encaje de 64, estado de 256, 359,260 parámetros) se entrenó 1,200 pasos de 32 ventanas de 128 letras, con AdamW, calentamiento y coseno.
-->

---
layout: seccion
numero: 2
---

# Atención

Ir a buscar el dato en lugar de cargarlo.

---

# Una búsqueda suave en un diccionario

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Un diccionario compara la **consulta** con cada **clave** y devuelve el **valor** de la que coincide. La atención (Bahdanau, Cho y Bengio, 2015) lo hace en versión suave y diferenciable:

$$
\alpha_{ts} = \operatorname{softmax}_s\Big(\frac{\mathbf{q}_t\T \mathbf{k}_s}{\sqrt{d_k}}\Big),
\qquad
\text{salida}_t = \sum_{s \le t} \alpha_{ts}\, \mathbf{v}_s .
$$

<div class="pequeno">

$\mathbf q_t$: consulta de la posición $t$; $\mathbf k_s$, $\mathbf v_s$: clave y valor de la posición $s$, vectores de dimensión $d_k$; $\alpha_{ts}$: peso que $t$ le da a $s$ (suman 1).

</div>

<v-click>

**Autoatención:** $\mathbf{q}, \mathbf{k}, \mathbf{v}$ salen de la misma secuencia, con proyecciones aprendidas $\vW_Q, \vW_K, \vW_V$.

</v-click>
</div>
<div v-click>

<AtencionFila :consulta="35" />

<div class="pequeno" style="margin-top: 22px">

Una cabeza de nuestro *transformer*: la «a» de «entrambos» pone el 70 % de su peso en la «e» con que **empezó la palabra**.

</div>
</div>
</div>

<!--
La componente es interactiva: clic en cualquier letra para ver a dónde mira. El camino entre dos posiciones tiene longitud 1, sin importar la distancia: no hay producto de T − t jacobianas.
-->

---

# ¿Por qué dividir entre $\sqrt{d_k}$?

<div class="cols" style="grid-template-columns: 9fr 11fr; gap: 40px">
<div>

Si las $d_k$ componentes $q_j, k_j$ de $\mathbf{q}$ y $\mathbf{k}$ son independientes, con media 0 y varianza 1:

$$
\Var(q_j k_j) = \E[q_j^2]\,\E[k_j^2] = 1
\;\Rightarrow\;
\Var(\mathbf{q}\T\mathbf{k}) = d_k .
$$

<v-click>

Con $d_k = 64$ los puntajes andan en $\pm 8$: el softmax se vuelve casi *one-hot*, y su jacobiana $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}\T$, con $\mathbf p$ los pesos que produce, se acerca a **cero**: no fluye el gradiente.

</v-click>

<v-click>

Dividir entre $\sqrt{d_k}$ devuelve los puntajes a varianza 1 **para cualquier dimensión**. El mismo razonamiento que la inicialización de Xavier y He.

</v-click>
</div>
<div v-click>

<EscalaRaiz />

<div class="pequeno tenue">

2,000 consultas, 64 claves cada una, $\mathbf{q}, \mathbf{k} \sim \mathcal N(\mathbf 0, \mathbf I)$ (normal estándar). Con $d_k = 1024$ sin escalar, el 95 % del peso va a una sola clave y la jacobiana mide 0.065, contra 0.18 escalando.

</div>
</div>
</div>

---

# El bloque *transformer*

<div class="cols" style="grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 36px">
<div>

$$
\operatorname{Atención}(\mathbf Q, \mathbf K, \mathbf V) = \operatorname{softmax}\Big(\frac{\mathbf Q\mathbf K\T}{\sqrt{d_k}} + \mathbf C\Big) \mathbf V
$$

<div class="pequeno">

$\mathbf Q, \mathbf K, \mathbf V$: consultas, claves y valores de las $T$ posiciones, una por renglón.

</div>

<div style="font-size: 21px">

<v-clicks>

- **Máscara causal** $\mathbf C$: $C_{ts} = -\infty$ si $s > t$ (y 0 si no); nadie ve el futuro y las $T$ predicciones de la ventana salen en paralelo.
- **Varias cabezas**: $H$ atenciones con $d_k = d/H$, concatenadas; $d$ es la dimensión de los encajes.
- **Posición**: la atención no conoce el orden; se suma un encaje de posición (senos y cosenos en Vaswani et al., 2017; aprendido aquí).
- **Bloque**: con $\mathbf R$ la ventana representada (un renglón por posición), $\mathbf{R} \leftarrow \mathbf{R} + \operatorname{Atención}(\operatorname{NC}(\mathbf{R}))$, luego $\mathbf{R} \leftarrow \mathbf{R} + \operatorname{MLP}(\operatorname{NC}(\mathbf{R}))$, con normalización de capa NC.
- **Costo**: $O(T^2 d)$ la atención, $O(T d^2)$ el resto.

</v-clicks>

</div>
</div>
<div>

<div style="width: 380px; margin: 0 auto">
<MapaAtencion :bloque="1" :cabeza="2" :botones="false" />
</div>

<div class="pequeno tenue">

Pesos $\alpha_{ts}$ (renglón: consulta $t$; columna: letra atendida $s$) de una cabeza sobre las últimas 40 letras de una frase: triangular por la máscara.

</div>
</div>
</div>

<!--
Nuestro transformer: d = 128, 4 cabezas (d_k = 32), 2 bloques, contexto de 128 letras, 436,828 parámetros. La atención está escrita a mano en la libreta, sin los módulos de PyTorch que la traen hecha. Esta cabeza (bloque 1, cabeza 2) pone el 97 % del peso en la letra anterior: es casi un bigrama aprendido.
-->

---

# ¿Qué aprendió cada cabeza?

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div class="pequeno">

Peso promedio en 64 ventanas de validación (cabeza 1.2 = bloque 1, cabeza 2); $t-1$, $t-2$, $t-3$: una, dos y tres letras antes de la posición $t$:

| cabeza | $t-1$ | $t-2$ | $t-3$ | inicio de palabra | saltos de línea |
|---|---|---|---|---|---|
| 1.1 | 0.02 | 0.05 | **0.62** | 0.13 | 0.02 |
| 1.2 | **0.97** | 0.00 | 0.00 | 0.18 | 0.02 |
| 1.3 | 0.01 | **0.93** | 0.01 | 0.16 | 0.02 |
| 2.2 | 0.06 | 0.09 | 0.08 | **0.36** | 0.02 |
| 2.4 | 0.03 | 0.03 | 0.03 | 0.03 | **0.61** |

<v-click>

Nadie programó estas reglas: salieron de minimizar la entropía cruzada.

</v-click>
</div>
<div>

<div style="width: 440px; margin: 0 auto">
<MapaAtencion :bloque="2" :cabeza="2" />
</div>

</div>
</div>

<style>
:root .slidev-layout td { padding-top: 10px; padding-bottom: 10px; }
</style>

<!--
Bloque.cabeza. Los saltos de línea son el 1.9 % de los caracteres, y la cabeza 2.4 les pone el 61 % de su peso: mide cuánto lleva la línea actual. El texto de Gutenberg tiene saltos de línea cada 70 a 75 caracteres, y por eso el texto generado también los tiene. Las cabezas 1.4, 2.1 y 2.3 (omitidas en la tabla) reparten el peso sin un patrón tan claro. Los botones cambian de cabeza.
-->

---

# Resultados en validación

<div class="cols" style="grid-template-columns: 9fr 11fr; gap: 36px">
<div class="pequeno">

| modelo | parámetros | perplejidad |
|---|---|---|
| uniforme | 0 | 92 |
| unigrama | 92 | 21.13 |
| bigrama (Laplace) | 8,464 | 10.02 |
| 5-grama ($\alpha = 0.01$) | 137,848 | 4.40 |
| LSTM | 359,260 | 3.96 |
| *transformer* | 436,828 | **3.90** |

<div class="tenue">

Las mismas 211,008 letras de validación. En los $n$-gramas, «parámetros» son los conteos que hay que guardar. 1.96 bits por carácter.

</div>
</div>
<div>

<CurvasLenguaje />

<v-click>

<div class="pequeno">

La LSTM arranca mucho más rápido (6.1 contra 10.3 a los 100 pasos): la recurrencia ya «sabe» que lo reciente importa. Con minutos de CPU, la diferencia final no permite declarar ganador.

</div>

</v-click>
</div>
</div>

---

# El camino corto de la atención

<div class="cols" style="grid-template-columns: 11fr 9fr; gap: 40px">
<div>

<GradienteDistancia :series="['lstm', 'tf']" />

</div>
<div>

La misma medición de antes, ahora con las dos redes entrenadas.

<v-click>

En las dos, lo cercano influye más: para predecir la siguiente letra, es lo que más importa.

</v-click>

<v-click>

Pero a 120 letras, en el *transformer* la influencia es $2.8 \times 10^{-3}$: **casi 200 veces** la de la LSTM. Ninguna letra de la ventana está a más de un paso de atención.

</v-click>
</div>
</div>

---

# ¿Dónde duda el modelo?

Sorpresa $-\log_2 \Prob(x_t \mid x_{<t})$ del *transformer* en cada letra $x_t$ de una frase de validación, en bits:

<SorpresaLetras />

<v-clicks>

- Lo caro es **empezar** una palabra («q» de «que»: 6.2 bits) y las palabras raras: «Durmamos» cuesta 29 bits en total.
- Dentro de una palabra común, casi nada: la «u» y la «e» de «que» cuestan 0.0 y 0.07 bits.
- «—dijo Sancho»: la única duda es la «S» (2.1 bits), entre Sancho y don Quijote.

</v-clicks>

<!--
Frase del capítulo LIX de la segunda parte (validación). Promedio en la frase: 1.72 bits por letra, menos que los 1.96 del promedio de validación. Los espacios son baratos: saber que una palabra terminó es fácil. Pasar el ratón sobre una letra muestra sus bits.
-->

---

# Generar texto: la temperatura

<div class="cols" style="grid-template-columns: 6fr 14fr; gap: 36px">
<div>

Se muestrea una letra de $\Prob(x_t \mid x_{<t})$, se agrega al contexto y se repite. La **temperatura** $\tau$ divide los *logits* $z_k$, uno por cada símbolo $k$:

$$
p_k(\tau) = \frac{e^{z_k/\tau}}{\sum_{k'} e^{z_{k'}/\tau}} .
$$

<div class="pequeno">

$\tau \to 0$: siempre la más probable. $\tau = 1$: el modelo tal cual. $\tau > 1$: más plana, hacia la uniforme.

</div>
</div>
<div>

<TextoTemperatura />

</div>
</div>

<!--
Mismo inicio y misma semilla para las tres temperaturas. Con 0.5: frases hechas del libro, palabras reales, sin sentido. Con 0.8: «vuesa merced», un turno de diálogo con raya, palabras inventadas pronunciables. Con 1.2: las palabras se descomponen. Las líneas miden entre 60 y 80 caracteres, como en el archivo de Gutenberg: es la cabeza 2.4, la de los saltos de línea.
-->

---
layout: caso
caso: Caso · Ética
cifra: 175,000 <small>millones</small>
cifraNota: parámetros de GPT-3 (Brown et al., 2020). Nuestro transformer tiene 436,828, y la misma tarea de entrenamiento.
fuente: E. M. Bender, T. Gebru, A. McMillan-Major y S. Shmitchell, «On the Dangers of Stochastic Parrots», FAccT 2021.
---

# Loros estocásticos

Bender, Gebru y sus coautoras argumentan que un modelo entrenado solo con texto aprende a reproducir la **forma** del lenguaje, sin acceso al **significado**, y que su fluidez hace que le atribuyamos comprensión e intenciones.

<v-click>

También advierten del costo ambiental de entrenar modelos cada vez más grandes y de los sesgos de corpus tan enormes que nadie puede documentarlos.

</v-click>

<v-click>

Nuestro modelo escribe «vuesa merced» y abre diálogos con raya porque eso es frecuente en el *Quijote*, no porque sepa quién habla. Cuánto entendimiento puede surgir de esta misma tarea a gran escala sigue en discusión.

</v-click>

<!--
"Shmargaret Shmitchell" es el seudónimo con el que firmó Margaret Mitchell. El artículo se hizo famoso también por la salida de Timnit Gebru de Google en diciembre de 2020, en medio de una disputa sobre su publicación (Mitchell salió en febrero de 2021). Conviene no presentar el debate como cerrado: la pregunta de cuánta comprensión emerge de predecir el siguiente símbolo está abierta.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 23px; line-height: 1.4">

1. Un modelo de lenguaje factoriza $\Prob(x_1, \dots, x_T) = \prod_t \Prob(x_t \mid x_{<t})$; la **perplejidad** $e^L$, con $L$ la entropía cruzada promedio, es el número efectivo de opciones.
2. Los $n$-gramas no comparten información entre contextos: con contextos largos, memorizan.
3. En una RNN el gradiente es un **producto de jacobianas**: se desvanece o explota. La celda de la LSTM es una autopista; el recorte, un seguro.
4. Atención: $\operatorname{softmax}(\mathbf Q\mathbf K\T/\sqrt{d_k} + \mathbf C)\mathbf V$, con consultas, claves, valores y máscara causal; el $\sqrt{d_k}$ mantiene la varianza 1.
5. Camino corto entre posiciones y cálculo en paralelo, a cambio de un costo $O(T^2)$ en la longitud $T$ de la ventana.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Un modelo de lenguaje del Quijote, letra por letra* (Unidad 6, ejemplo 4).

**Lectura:** I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, capítulo 10; A. Vaswani et al., «Attention is all you need», NeurIPS 2017; S. J. D. Prince, *Understanding Deep Learning*, capítulo 12.

</div>
</div>
</div>
