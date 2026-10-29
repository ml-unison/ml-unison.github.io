---
theme: none
title: Regresión lineal
unidad: Unidad 4
layout: portada
subtitulo: "Mínimos cuadrados: su geometría, cómo se calcula y qué significa «lineal»"
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

# Regresión lineal

---
layout: pregunta
---

# ¿Cuántas bicicletas se rentarán en Washington este martes a las 8 de la mañana?

<v-click>

Depende del clima, pero sobre todo de la **hora** y de si es **día laboral**. Y no de manera «lineal» en el sentido ingenuo.

</v-click>

---

# El problema de hoy: bicicletas compartidas

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div>

<PerfilHoras />

</div>
<div>

- **Capital Bikeshare**, Washington D. C.: 17,379 horas de 2011 y 2012 con las rentas, la fecha y el clima.
- **Problema:** predecir cuántas bicicletas se rentarán en una hora, para redistribuirlas entre estaciones.
- Entrenamos con los días 1 a 19 de cada mes y probamos con el resto.

<p class="pequeno tenue" style="margin-top: 22px">H. Fanaee-T y J. Gama, <em>Progress in Artificial Intelligence</em> 2 (2014). UCI, CC BY 4.0.</p>

</div>
</div>

<!--
Los dos picos de los días laborales son la gente que va y vuelve del trabajo. Los fines de semana hay una sola joroba al mediodía. Ya con esta figura se ve que la hora no puede entrar como un número.
-->

---
layout: seccion
numero: 1
---

# Mínimos cuadrados

El problema, su solución y su geometría.

---

# El modelo y el error

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

Con $M$ datos $(\vx^{(i)}, y^{(i)})$, $\vx^{(i)} \in \R^n$, pesos $\vw$ y sesgo $b$, el modelo y la pérdida cuadrática promedio son

$$
h(\vx) = \vw\T \vx + b, \qquad
L(\vw, b) = \frac1M \sum_{i=1}^M \big(\vw\T \vx^{(i)} + b - y^{(i)}\big)^2 .
$$

<div class="pequeno">

Con matrices: a $\vX \in \R^{M \times n}$ le agregamos una columna de unos, $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$, y juntamos $\tilde\vw = (b, \vw)$. Así $\tilde\vX\tilde\vw = b\mathbf{1} + \vX\vw$ y $L = \frac1M \norm{\tilde\vX\tilde\vw - \vy}^2$.

</div>

<v-click>

Con una sola variable, la temperatura: cada grado más trae unas **9 bicicletas** más por hora. Es la recta que minimiza la suma de cuadrados de las distancias verticales.

</v-click>
</div>
<div>

<TempRecta />

</div>
</div>

---

# Las ecuaciones normales

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Con $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$ (los datos y una columna de unos) y $\tilde\vw = (b, \vw)$, la pérdida $L$ es una cuadrática en $\tilde\vw$:

$$
L(\tilde\vw) = \tfrac1M \big(\tilde\vw\T \tilde\vX\T \tilde\vX \tilde\vw - 2\,\tilde\vw\T \tilde\vX\T \vy + \vy\T\vy\big)
$$

<v-click>

Su gradiente y su hessiana:

$$
\nabla L = \tfrac2M \big(\tilde\vX\T \tilde\vX \tilde\vw - \tilde\vX\T \vy\big), \qquad \nabla^2 L = \tfrac2M \tilde\vX\T \tilde\vX \succeq 0 .
$$

</v-click>
</div>
<div>

<v-click>

<Caja tipo="proposicion">

Si las columnas de $\tilde\vX$ son linealmente independientes, $\tilde\vX\T \tilde\vX$ es invertible y el único minimizador es

$$
(\hat b, \hat\vw) = (\tilde\vX\T \tilde\vX)^{-1} \tilde\vX\T \vy = \tilde\vX^{\dagger} \vy ,
$$

con $\tilde\vX^{\dagger}$ la pseudoinversa. La ecuación del sesgo da $\hat b = \bar y - \hat\vw\T \bar\vx$ (las medias): $b$ solo ajusta el nivel.

</Caja>

</v-click>

<v-click>

La hessiana es semidefinida positiva: $L$ es **convexa** y el punto crítico es mínimo global. Es de los pocos modelos de este curso con **solución en forma cerrada**.

</v-click>
</div>
</div>

---

# La geometría: una proyección

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<Proyeccion />

</div>
<div>

$\hat\vy = \tilde\vX\tilde\vw$, con $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$ y $\tilde\vw = (b, \vw)$, recorre el **espacio columna** de $\tilde\vX$; el más cercano a $\vy$ es su proyección ortogonal:

$$
\hat\vy = \underbrace{\tilde\vX (\tilde\vX\T \tilde\vX)^{-1} \tilde\vX\T}_{\mathbf{H}} \, \vy .
$$

<v-clicks>

- Las ecuaciones normales dicen $\tilde\vX\T(\vy - \hat\vy) = \mathbf{0}$: **el residuo es perpendicular** a cada columna. Por la columna de unos (el sesgo), los residuos suman cero.
- La **matriz sombrero** $\mathbf{H}$ es simétrica e idempotente ($\mathbf{H}^2 = \mathbf{H}$), con $\operatorname{tr} \mathbf{H} = n + 1$: los grados de libertad del modelo.

</v-clicks>
</div>
</div>

---

# La lectura probabilística

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Si $y = \vw\T \vx + b + \epsilon$, con ruido $\epsilon \sim \mathcal{N}(0, \sigma^2)$ independiente en cada dato, la log-verosimilitud $\ln \mathcal{L}$ de los $M$ datos es

$$
\ln \mathcal{L}(\tilde\vw) = -\frac M2 \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \norm{\vy - \tilde\vX\tilde\vw}^2 ,
$$

con $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$ y $\tilde\vw = (b, \vw)$.

<v-click>

Maximizarla es **minimizar la suma de cuadrados**: mínimos cuadrados es el estimador de máxima verosimilitud con ruido gaussiano. Y entonces

$$
\hat{\tilde\vw} \sim \mathcal{N}\big(\tilde\vw,\; \sigma^2 (\tilde\vX\T \tilde\vX)^{-1}\big) .
$$

</v-click>
</div>
<div v-click>

<Caja tipo="teorema">

**Gauss–Markov.** Si el vector de ruidos cumple $\E[\boldsymbol\epsilon] = \mathbf{0}$ y $\Var[\boldsymbol\epsilon] = \sigma^2 \mathbf{I}$, entre todos los estimadores **lineales e insesgados** de $(b, \vw)$, el de mínimos cuadrados tiene la menor varianza.

</Caja>

<div class="pequeno" style="margin-top: 14px">

No pide normalidad. Pero «insesgados» es una restricción fuerte: ridge, que además penaliza $\norm{\vw}^2$, es sesgado y puede tener menor error.

</div>
</div>
</div>

---
layout: seccion
numero: 2
---

# Cómo se calcula

Lo que dice el álgebra no siempre es lo que conviene programar.

---

# No inviertas $\tilde\vX\T \tilde\vX$

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

El **número de condición** de una matriz $\mathbf{A}$, $\kappa(\mathbf{A}) = \sigma_{\max} / \sigma_{\min}$ (el cociente de sus valores singulares extremos), mide cuánto se amplifican los errores de redondeo. Con $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$,

$$
\kappa(\tilde\vX\T \tilde\vX) = \kappa(\tilde\vX)^2 .
$$

<v-click>

Con $\tilde\vX = [\,\mathbf{1}, \text{temp}, \text{sensación}\,]$ (correlación 0.985): $\kappa(\tilde\vX) \approx 104$, pero $\kappa(\tilde\vX\T \tilde\vX) \approx 10{,}700$. En doble precisión se pierden unos 4 dígitos en lugar de 2.

</v-click>
</div>
<div v-click>

| método | costo | estabilidad |
|---|---|---|
| ecuaciones normales (Cholesky) | $O(Mn^2 + n^3)$ | depende de $\kappa(\tilde\vX)^2$ |
| QR de $\tilde\vX$ | $O(Mn^2)$ | depende de $\kappa(\tilde\vX)$ |
| SVD de $\tilde\vX$ | $O(Mn^2)$, mayor constante | tolera rango deficiente |
| descenso por gradiente | $O(Mn)$ por iteración | iteraciones $\propto \kappa$ |

<div class="pequeno tenue">

$M$ datos, $n$ atributos. `np.linalg.lstsq` y `LinearRegression` no forman $(\tilde\vX\T \tilde\vX)^{-1}$.

</div>
</div>
</div>

---

# Descenso por gradiente en una cuadrática

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

$$
\tilde\vw_{t+1} = \tilde\vw_t - \eta\, \nabla L(\tilde\vw_t)
$$

con $\tilde\vw = (b, \vw)$, $t$ la iteración y $\eta$ la tasa de aprendizaje.

<v-click>

Si la hessiana $\nabla^2 L = \mathbf{A}$ tiene valores propios $\lambda_1 \ge \dots \ge \lambda_{n+1} > 0$, el error en la dirección del vector propio $r$ se multiplica en cada paso por

$$
1 - \eta \lambda_r .
$$

</v-click>
</div>
<div>

<v-clicks>

- Para no diverger hace falta $|1 - \eta\lambda_r| < 1$, es decir $\eta < 2 / \lambda_1$.
- Con $\eta = 1/\lambda_1$, la dirección más plana se reduce por $1 - 1/\kappa$ en cada paso, con $\kappa = \lambda_1 / \lambda_{n+1}$: para reducir el error a una fracción $\varepsilon$ se necesitan del orden de $\kappa \ln(1/\varepsilon)$ iteraciones.
- **Escalar los atributos** no cambia el modelo, pero cambia $\kappa$.

</v-clicks>

</div>
</div>

---

# El mismo modelo, dos geometrías

<DescensoValle />

<div class="pequeno" style="margin-top: 6px">

Rentas contra temperatura, con paso $\eta = 0.9 \cdot 2/\lambda_{\max}$ ($\lambda_{\max}$: mayor valor propio de la hessiana). Sin centrar, $b$ y $w$ están acoplados (la temperatura media es 20 °C): los pasos rebotan entre las paredes del valle y luego avanzan a paso de tortuga. Estandarizada, con $\eta = 1/2$ bastaría un paso.

</div>

---
layout: seccion
numero: 3
---

# Lineal en los parámetros

No en los datos.

---

# Transformar antes de ajustar

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Con cualquier transformación fija $\Phi$ de los atributos, el modelo $h(\vx) = \vw\T \Phi(\vx) + b$ se ajusta con **los mismos** mínimos cuadrados, usando la matriz $\mathbf{Z}$ cuyo renglón $i$ es $\Phi(\vx^{(i)})$:

<v-clicks>

- polinomios: $\Phi(x) = (x, x^2, x^3, \dots)$;
- *one-hot*: una columna 0/1 por cada hora;
- **interacciones**: $\text{hora} \times \text{laboral}$, una columna por pareja.

</v-clicks>
</div>
<div v-click>

| modelo | atributos | RECM de prueba |
|---|---|---|
| M1 | temperatura | 165.0 |
| M2 | + hora **como número** | 153.5 |
| M3 | hora *one-hot*, clima, año, laboral | 107.4 |
| M4 | hora × laboral, clima, año | **86.4** |

<div class="pequeno tenue">

M4 tiene 47 columnas para las 48 combinaciones de hora y tipo de día: cada una tiene su propio nivel.

</div>
</div>
</div>

---

# Lo que ve cada modelo

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<PerfilModelos />

</div>
<div>

Días laborales de prueba, promedio por hora.

<v-clicks>

- **M2** solo puede decir «más tarde, más bicicletas»: una recta en la hora.
- **M3** aprende la forma de un día, pero la misma para laborales y fines de semana: promedia los dos picos con la joroba del mediodía.
- **M4** le da a cada tipo de día su propio perfil.

</v-clicks>
</div>
</div>

---

# Colinealidad: buenas predicciones, coeficientes inestables

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

$$
\Var[\hat w_j] = \frac{\sigma^2}{M\, \widehat{\Var}(x_j)} \cdot \frac{1}{1 - R_j^2}
$$

donde $\hat w_j$ es el peso estimado del atributo $x_j$, $\sigma^2$ la varianza del ruido, $M$ el número de datos, $\widehat{\Var}(x_j)$ la varianza del atributo en los datos y $R_j^2$ el $R^2$ de predecir $x_j$ con los demás atributos. El segundo factor es el **factor de inflación de la varianza**.

<v-click>

Temperatura y sensación térmica tienen correlación 0.985: el factor es $1/(1 - 0.985^2) \approx 33$. El error estándar del coeficiente de temp crece $\sqrt{33} \approx 5.8$ **veces**.

</v-click>
</div>
<div>

<Colinealidad />

<div class="pequeno tenue">

200 réplicas bootstrap: una variable compensa a la otra; la suma, y las predicciones, casi no cambian.

</div>
</div>
</div>

---
layout: seccion
numero: 4
---

# Diagnóstico

Lo que los residuos dicen del modelo.

---

# Los residuos de M4

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<Residuos />

</div>
<div>

<v-clicks>

- **Embudo:** la dispersión crece con el valor ajustado. El supuesto de ruido con varianza constante, $\Var[\epsilon] = \sigma^2$, no se cumple.
- **10 %** de las horas de prueba reciben una predicción **negativa**: el modelo cree que en la madrugada se devuelven bicicletas que nadie rentó.
- Las rentas son **conteos**: no negativos, y con varianza que crece con la media.

</v-clicks>

<div v-click class="nota">

Lo que hace falta: un modelo lineal para la **media de un conteo**, como la regresión de Poisson.

</div>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Mínimos cuadrados = **proyección** del vector de salidas $\vy$ sobre el espacio columna de los datos = máxima verosimilitud con ruido gaussiano.
2. Se resuelve con QR o SVD, no invirtiendo la matriz de las ecuaciones normales.
3. El descenso por gradiente tarda del orden del número de condición $\kappa$ en iteraciones: **escala los atributos**.
4. «Lineal» es en los parámetros: *one-hot* e interacciones cambian todo.
5. Con colinealidad, no interpretes coeficientes aislados.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Regresión lineal: cuántas bicicletas se rentarán* (Unidad 4, ejemplo 1).

**Lectura:** Abu-Mostafa, Magdon-Ismail y Lin, *Learning From Data*, sección 3.2; Hastie, Tibshirani y Friedman, *The Elements of Statistical Learning*, secciones 3.1–3.2.

</div>
</div>
</div>
