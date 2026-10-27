---
theme: none
title: "Secuencias: redes recurrentes, atención y transformers"
unidad: Unidad 6
layout: portada
subtitulo: "Un modelo de lenguaje del Quijote, letra por letra"
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

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 56px">
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

Por la regla de la cadena, **sin suponer nada**:

$$
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t}) .
$$

</v-click>

<v-click>

Cada factor es un clasificador de 92 clases cuya entrada es el pasado. Se entrena con la **entropía cruzada**

$$
H = -\frac1N \sum_{t} \ln P_\theta(x_t \mid x_{<t}) .
$$

La «etiqueta» es la letra que de verdad sigue: aprendizaje **autosupervisado**.

</v-click>
</div>
</div>

---

# Perplejidad: el número efectivo de opciones

<div class="cols" style="grid-template-columns: 9fr 11fr; gap: 48px">
<div>

$$
\text{perplejidad} = e^{H} = \Big(\prod_{t=1}^{N} \frac{1}{P_\theta(x_t \mid x_{<t})}\Big)^{1/N}
$$

Si el modelo duda **por igual entre $k$ opciones** (y la correcta está entre ellas), la perplejidad es exactamente $k$.

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

<NgramasCurva v-if="$clicks < 1" :suave="false" />
<NgramasCurva v-else />

</div>
<div>

Contexto = las $n - 1$ letras anteriores, con el **suavizado de Laplace** de Bayes ingenuo (Unidad 4):

$$
\hat P(x \mid c) = \frac{N(c, x) + \alpha}{N(c) + \alpha V} .
$$

<v-click>

Con $\alpha = 0.01$ el mejor es $n = 5$: **4.40**. Después, la validación empeora: la curva de sobreajuste de la Unidad 3.

</v-click>

<v-click>

<div class="nota">

Con 7 letras de contexto, el **18 %** de los 7-gramas de validación nunca aparece en 1.9 millones de letras. «dijo don Quijo» y «dijo don Quij» son filas **sin relación**: hace falta compartir información entre contextos parecidos.

</div>

</v-click>
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

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

$$
\mathbf{h}_t = \tanh\big(W\mathbf{h}_{t-1} + U\mathbf{e}_{x_t} + \mathbf{b}\big)
$$

y se predice con $\operatorname{softmax}(A\mathbf{h}_t + \mathbf{c})$. Las mismas $W, U, A$ en **todas** las posiciones; $\mathbf{e}_{x_t}$ es el **encaje** aprendido de la letra.

<v-click>

Retropropagación **en el tiempo**:

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{s=t+1}^{T} \operatorname{diag}\big(1 - \mathbf{h}_s^2\big)\, W
$$

Como $0 < 1 - h^2 \le 1$, cada factor tiene norma $\le \sigma_{\max}(W)$: la norma del producto queda acotada por $\sigma_{\max}(W)^{T-t}$, **exponencial** en la distancia.

</v-click>
</div>
<div v-click>

<JacobianasRNN />

<div class="pequeno tenue">

RNN de 256 unidades con $W = \rho Q$, $Q$ ortogonal. Con $\rho = 0.8$, $10^{-10}$ a 50 letras; con $\rho = 3$, $10^{13}$ a 100.

</div>
</div>
</div>

<!--
Es el mismo fenómeno de las redes profundas de la clase 1, agravado porque se multiplica la misma matriz una y otra vez. Con ρ = 1 también se desvanece: las derivadas de tanh son menores que 1. La franja buena (aquí cerca de ρ = 1.5) es angosta y depende de cuánto se saturan las tanh, o sea, de los datos.
-->

---

# Contra la explosión: recortar el gradiente

<div class="cols" style="grid-template-columns: 9fr 11fr; gap: 40px">
<div>

Si la norma del gradiente completo pasa de un umbral $c$, se reescala:

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\Big(1, \frac{c}{\norm{\mathbf{g}}}\Big) .
$$

La dirección se conserva; el paso no se dispara.

<v-click>

En nuestra LSTM, con $c = 1$: mediana **0.13**, y el recorte actuó en **2 de 1,200 pasos**, los dos al principio.

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

Tres **compuertas** en $(0, 1)$, función de $\mathbf{h}_{t-1}$ y $\mathbf{e}_{x_t}$: olvido $\mathbf{f}_t$, entrada $\mathbf{i}_t$, salida $\mathbf{o}_t$; y un candidato $\tilde{\mathbf{c}}_t$.

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

<v-click>

Por el camino de la celda, $\dfrac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \operatorname{diag}(\mathbf{f}_t)$: **sin $W$ ni $\tanh$**. Si $f \approx 1$, el gradiente pasa casi intacto.

</v-click>

<v-click>

<div class="pequeno tenue">

Hochreiter y Schmidhuber (1997). Es la misma idea que las conexiones residuales: $\partial(\mathbf{x} + F(\mathbf{x}))/\partial\mathbf{x} = I + \cdots$

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
Sin entrenar, la compuerta de olvido vale cerca de σ(0) = 0.5 y la LSTM se comporta como una RNN con ρ < 1: 10⁻¹² a 60 letras, y a 120 el número se vuelve cero en punto flotante. Entrenada, aprendió a abrir la autopista en algunas unidades. La LSTM (encaje de 64, estado de 256, 359,260 parámetros) se entrenó 1,200 pasos de 32 ventanas de 128 letras, con AdamW, calentamiento y coseno.
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

<v-click>

**Autoatención:** $\mathbf{q}, \mathbf{k}, \mathbf{v}$ salen de la misma secuencia, con proyecciones aprendidas $W_Q, W_K, W_V$.

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

Si las componentes de $\mathbf{q}$ y $\mathbf{k}$ son independientes, con media 0 y varianza 1:

$$
\Var(q_i k_i) = \E[q_i^2]\,\E[k_i^2] = 1
\;\Rightarrow\;
\Var(\mathbf{q}\T\mathbf{k}) = d_k .
$$

<v-click>

Con $d_k = 64$ los puntajes andan en $\pm 8$: el softmax se vuelve casi *one-hot*, y su jacobiana $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}\T$ se acerca a **cero**: no fluye el gradiente.

</v-click>

<v-click>

Dividir entre $\sqrt{d_k}$ devuelve los puntajes a varianza 1 **para cualquier dimensión**. El mismo razonamiento que la inicialización de Xavier y He.

</v-click>
</div>
<div v-click>

<EscalaRaiz />

<div class="pequeno tenue">

2,000 consultas, 64 claves cada una, $\mathbf{q}, \mathbf{k} \sim N(0, I)$. Con $d_k = 1024$ sin escalar, el 95 % del peso va a una sola clave y la jacobiana mide 0.065, contra 0.18 escalando.

</div>
</div>
</div>

---

# El bloque *transformer*

<div class="cols" style="grid-template-columns: minmax(0, 11fr) minmax(0, 9fr); gap: 40px">
<div>

$$
\operatorname{Atención}(Q, K, V) = \operatorname{softmax}\Big(\frac{QK\T}{\sqrt{d_k}} + M\Big) V
$$

<div style="font-size: 22px">

<v-clicks>

- **Máscara causal**: $M_{ts} = -\infty$ si $s > t$ (y 0 si no); nadie ve el futuro; las $T$ predicciones de una ventana salen en una sola pasada, en paralelo.
- **Varias cabezas**: $H$ atenciones con $d_k = d/H$, concatenadas.
- **Posición**: la atención no conoce el orden; se suma un encaje de posición (senos y cosenos en Vaswani et al., 2017; aprendido aquí).
- **Bloque**: $\mathbf{x} \leftarrow \mathbf{x} + \operatorname{Atención}(\operatorname{NC}(\mathbf{x}))$, luego $\mathbf{x} \leftarrow \mathbf{x} + \operatorname{MLP}(\operatorname{NC}(\mathbf{x}))$, con normalización de capa NC.
- **Costo**: $O(T^2 d)$ la atención, $O(T d^2)$ el resto.

</v-clicks>

</div>
</div>
<div>

<div style="width: 400px; margin: 0 auto">
<MapaAtencion :bloque="1" :cabeza="2" :botones="false" />
</div>

<div class="pequeno tenue">

Pesos $\alpha_{ts}$ de una cabeza sobre las últimas 40 letras de una frase: triangular por la máscara.

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

Peso promedio en 64 ventanas de validación (cabeza 1.2 = bloque 1, cabeza 2):

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

Sorpresa $-\log_2 P(x_t \mid x_{<t})$ del *transformer* en cada letra de una frase de validación:

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

Se muestrea una letra de $P(x_t \mid x_{<t})$, se agrega al contexto y se repite. La **temperatura** $\tau$ divide los *logits*:

$$
p_i(\tau) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}} .
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
<div style="font-size: 25px; line-height: 1.4">

1. Un modelo de lenguaje factoriza $P(x_1, \dots, x_T) = \prod_t P(x_t \mid x_{<t})$; la **perplejidad** $e^H$ es el número efectivo de opciones.
2. Los $n$-gramas no comparten información entre contextos: con contextos largos, memorizan.
3. En una RNN el gradiente es un **producto de jacobianas**: se desvanece o explota. La celda de la LSTM es una autopista; el recorte, un seguro.
4. Atención: $\operatorname{softmax}(QK\T/\sqrt{d_k} + M)V$; el $\sqrt{d_k}$ mantiene la varianza 1.
5. Camino corto entre posiciones y cálculo en paralelo, a cambio de $O(T^2)$.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Un modelo de lenguaje del Quijote, letra por letra* (Unidad 6, ejemplo 4).

**Lectura:** I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, capítulo 10; A. Vaswani et al., «Attention is all you need», NeurIPS 2017; S. J. D. Prince, *Understanding Deep Learning*, capítulo 12.

</div>
</div>
</div>
