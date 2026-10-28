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

Con $\vx = (1, x_1, \dots, x_d)$ y la matriz de datos $X \in \R^{N \times (d+1)}$:

$$
h(\vx) = \vw\T \vx, \qquad
\Ein(\vw) = \frac1N \sum_{n=1}^N (\vw\T \vx_n - y_n)^2 = \frac1N \norm{X\vw - \mathbf{y}}^2 .
$$

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

$\Ein$ es una cuadrática en $\vw$:

$$
\Ein(\vw) = \tfrac1N \big(\vw\T X\T X \vw - 2\,\vw\T X\T \mathbf{y} + \mathbf{y}\T\mathbf{y}\big)
$$

<v-click>

Su gradiente y su hessiana:

$$
\nabla \Ein = \tfrac2N \big(X\T X \vw - X\T \mathbf{y}\big), \qquad \nabla^2 \Ein = \tfrac2N X\T X \succeq 0 .
$$

</v-click>
</div>
<div>

<v-click>

<Caja tipo="proposicion">

Si las columnas de $X$ son linealmente independientes, $X\T X$ es invertible y el único minimizador es

$$
\vw_{\text{lin}} = (X\T X)^{-1} X\T \mathbf{y} = X^{\dagger} \mathbf{y} .
$$

</Caja>

</v-click>

<v-click>

La hessiana es semidefinida positiva: $\Ein$ es **convexa** y el punto crítico es mínimo global. Es de los pocos modelos de este curso con **solución en forma cerrada**.

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

$\hat{\mathbf{y}} = X\vw$ recorre el **espacio columna** de $X$. El más cercano a $\mathbf{y}$ es su proyección ortogonal:

$$
\hat{\mathbf{y}} = \underbrace{X (X\T X)^{-1} X\T}_{H} \, \mathbf{y} .
$$

<v-clicks>

- Las ecuaciones normales dicen $X\T(\mathbf{y} - \hat{\mathbf{y}}) = 0$: **el residuo es perpendicular** a cada columna. Con intersección, los residuos suman cero.
- La **matriz sombrero** $H$ es simétrica e idempotente ($H^2 = H$), con $\operatorname{tr} H = d + 1$: los grados de libertad del modelo.
- Es la misma $H$ de la fórmula de dejar-uno-fuera de la Unidad 3.

</v-clicks>
</div>
</div>

---

# La lectura probabilística

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Si $y = \vw\T \vx + \epsilon$ con $\epsilon \sim \mathcal{N}(0, \sigma^2)$ independientes, la log-verosimilitud es

$$
\ell(\vw) = -\frac N2 \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \norm{\mathbf{y} - X\vw}^2 .
$$

<v-click>

Maximizarla es **minimizar la suma de cuadrados**: mínimos cuadrados es el estimador de máxima verosimilitud con ruido gaussiano. Y entonces

$$
\hat\vw \sim \mathcal{N}\big(\vw,\; \sigma^2 (X\T X)^{-1}\big) .
$$

</v-click>
</div>
<div v-click>

<Caja tipo="teorema">

**Gauss–Markov.** Si $\E[\boldsymbol\epsilon] = 0$ y $\Var[\boldsymbol\epsilon] = \sigma^2 I$, entre todos los estimadores **lineales e insesgados** de $\vw$, el de mínimos cuadrados tiene la menor varianza.

</Caja>

<div class="pequeno" style="margin-top: 14px">

No pide normalidad. Pero «insesgados» es una restricción fuerte: ridge (Unidad 3) es sesgado y puede tener menor error.

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

# No inviertas $X\T X$

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

El **número de condición** $\kappa(A) = \sigma_{\max} / \sigma_{\min}$ mide cuánto se amplifican los errores de redondeo. Y

$$
\kappa(X\T X) = \kappa(X)^2 .
$$

<v-click>

Con $X = [\,1, \text{temp}, \text{sensación}\,]$ (correlación 0.985): $\kappa(X) \approx 104$, pero $\kappa(X\T X) \approx 10{,}700$. En doble precisión se pierden unos 4 dígitos en lugar de 2.

</v-click>
</div>
<div v-click>

| método | costo | estabilidad |
|---|---|---|
| ecuaciones normales (Cholesky) | $O(Nd^2 + d^3)$ | depende de $\kappa(X)^2$ |
| QR de $X$ | $O(Nd^2)$ | depende de $\kappa(X)$ |
| SVD de $X$ | $O(Nd^2)$, mayor constante | tolera rango deficiente |
| descenso por gradiente | $O(Nd)$ por iteración | iteraciones $\propto \kappa$ |

<div class="pequeno tenue">

`np.linalg.lstsq` usa la SVD; `LinearRegression` de scikit-learn también resuelve por mínimos cuadrados sin formar $(X\T X)^{-1}$.

</div>
</div>
</div>

---

# Descenso por gradiente en una cuadrática

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

$$
\vw_{t+1} = \vw_t - \eta\, \nabla \Ein(\vw_t)
$$

<v-click>

Con $\nabla^2\Ein = A$ y valores propios $\lambda_1 \ge \dots \ge \lambda_p > 0$, el error en la dirección propia $i$ se multiplica en cada paso por

$$
1 - \eta \lambda_i .
$$

</v-click>
</div>
<div>

<v-clicks>

- Para no diverger hace falta $|1 - \eta\lambda_i| < 1$, es decir $\eta < 2 / \lambda_1$.
- Con $\eta = 1/\lambda_1$, la dirección más plana se reduce por $1 - 1/\kappa$ en cada paso: se necesitan del orden de $\kappa \ln(1/\varepsilon)$ iteraciones.
- **Escalar los atributos** no cambia el modelo, pero cambia $\kappa$.

</v-clicks>

</div>
</div>

---

# El mismo modelo, dos geometrías

<DescensoValle />

<div class="pequeno" style="margin-top: 6px">

Rentas contra temperatura, con $\eta = 0.9 \cdot 2/\lambda_{\max}$, hasta que $\Ein$ queda a una millonésima de su mínimo. Sin centrar, la intersección y la pendiente están acopladas (la temperatura media es 20 °C): los primeros pasos rebotan entre las paredes del valle y luego avanzan a paso de tortuga. Estandarizada, la hessiana es $2I$: con $\eta = 1/2$ bastaría un paso.

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

Con cualquier transformación fija $\Phi$, el modelo $h(\vx) = \vw\T \Phi(\vx)$ se ajusta con **los mismos** mínimos cuadrados, usando $Z = \Phi(X)$:

<v-clicks>

- polinomios: $\Phi(x) = (1, x, x^2, \dots)$, como en la Unidad 1;
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
\Var[\hat w_j] = \frac{\sigma^2}{N\, \widehat{\Var}(x_j)} \cdot \frac{1}{1 - R_j^2}
$$

donde $R_j^2$ es el $R^2$ de predecir $x_j$ con los demás atributos. El segundo factor es el **factor de inflación de la varianza**.

<v-click>

Temperatura y sensación térmica tienen correlación 0.985: el factor es $1/(1 - 0.985^2) \approx 33$. El error estándar del coeficiente de temp crece $\sqrt{33} \approx 5.8$ **veces**.

</v-click>
</div>
<div>

<Colinealidad />

<div class="pequeno tenue">

200 réplicas bootstrap. Juntas, una variable compensa a la otra (correlación negativa); la suma casi no cambia, y las predicciones tampoco.

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

- **Embudo:** la dispersión crece con el valor ajustado. El supuesto $\Var[\epsilon] = \sigma^2$ no se cumple.
- **10 %** de las horas de prueba reciben una predicción **negativa**: el modelo cree que en la madrugada se devuelven bicicletas que nadie rentó.
- Las rentas son **conteos**: no negativos, y con varianza que crece con la media.

</v-clicks>

<div v-click class="nota">

En la siguiente presentación: un modelo lineal para la **media de un conteo**, la regresión de Poisson.

</div>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Mínimos cuadrados = **proyección** de $\mathbf{y}$ sobre el espacio columna = máxima verosimilitud con ruido gaussiano.
2. Se resuelve con QR o SVD, no invirtiendo $X\T X$.
3. El descenso por gradiente tarda del orden de $\kappa$ iteraciones: **escala los atributos**.
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
