---
theme: none
title: Sesgo, varianza y regularización
unidad: Unidad 3
layout: portada
subtitulo: De dónde viene el error fuera de la muestra y cómo domar la complejidad sin quitar flexibilidad
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

# Sesgo, varianza y regularización

---
layout: pregunta
---

# Si pudiéramos repetir el experimento con mil conjuntos de entrenamiento distintos, ¿qué le pediríamos a un buen modelo?

<v-click>

Que en promedio le atine a $f$ (**poco sesgo**) y que no cambie mucho de un conjunto a otro (**poca varianza**).

</v-click>

---

# El escenario

<div class="cols" style="gap: 48px">
<div>

- Regresión con error cuadrático; $y = f(\vx) + \epsilon$, con $\E[\epsilon] = 0$ y $\Var[\epsilon] = \sigma^2$.
- El conjunto de entrenamiento $\Dat$ es **aleatorio**, así que la hipótesis aprendida $g^{(\Dat)}$ también lo es.
- La **hipótesis promedio**: $\;\bar g(\vx) = \E_{\Dat}\big[g^{(\Dat)}(\vx)\big]$.

</div>
<div v-click>

Queremos el error esperado fuera de la muestra, promediado sobre los conjuntos de entrenamiento:

$$
\E_{\Dat}\big[\Eout(g^{(\Dat)})\big] = \E_{\vx}\Big[\, \E_{\Dat, \epsilon}\big[(g^{(\Dat)}(\vx) - y)^2\big] \Big] .
$$

<div class="pequeno tenue">

Basta con analizar el término interior para un $\vx$ fijo.

</div>
</div>
</div>

---

# La descomposición, paso a paso

Con $\vx$ fijo, sumamos y restamos $\bar g(\vx)$ y $f(\vx)$:

$$
g^{(\Dat)} - y = \underbrace{(g^{(\Dat)} - \bar g)}_{A} + \underbrace{(\bar g - f)}_{B} - \underbrace{\epsilon}_{C}
$$

$$
\E\big[(A + B - C)^2\big] = \E[A^2] + B^2 + \E[C^2] + 2B\,\E[A] - 2\,\E[AC] - 2B\,\E[C] .
$$

<v-click>

Los productos cruzados se anulan: $\E_{\Dat}[A] = \bar g - \bar g = 0$ por definición de $\bar g$; $\E[C] = 0$; y $\E[AC] = \E[A]\,\E[C] = 0$, porque el ruido del punto nuevo es independiente de $\Dat$.

</v-click>

<div v-click>
<Caja tipo="teorema" nombre="sesgo y varianza">

$$
\E_{\Dat}\big[\Eout\big] \;=\; \underbrace{\E_{\vx}\big[(\bar g(\vx) - f(\vx))^2\big]}_{\text{sesgo}^2} \;+\; \underbrace{\E_{\vx}\big[\Var_{\Dat}\, g^{(\Dat)}(\vx)\big]}_{\text{varianza}} \;+\; \sigma^2
$$

</Caja>
</div>

<!--
El sesgo mide qué tan lejos queda H de f aun promediando; la varianza, cuánto nos mueve la muestra particular; sigma² no lo quita nadie.
-->

---

# Verlo: 25 conjuntos de entrenamiento, tres modelos

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 24px">
<AjustesMuchos :grado="1" />
<AjustesMuchos :grado="3" />
<AjustesMuchos :grado="9" />
</div>

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 24px; margin-top: 8px">
<div class="pequeno">

Las rectas casi no cambian, pero su promedio **no se parece** a $f$: mucho sesgo.

</div>
<div class="pequeno" v-click>

El grado 3 se mueve poco y su promedio casi coincide con $f$.

</div>
<div class="pequeno" v-click>El grado 9 le atina en promedio, pero cada ajuste se va por su lado: mucha varianza.</div>
</div>

<div class="tenue" style="font-size: 16px; margin-top: 10px">

$f(x) = \sin 2\pi x$, $N = 15$, ruido $\sigma = 0.25$, como en la Unidad 1. En gris, cada ajuste; en azul, $\bar g$; punteada, $f$.

</div>

---

# La U de siempre, ahora descompuesta

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<DescomposicionU />

</div>
<div>

Promediando 2,000 conjuntos de entrenamiento por grado (cada uno con un $x$ en cada quinceavo de $[0, 1]$).

<v-click>

El mejor es el **grado 3**: sesgo² ≈ 0.005, varianza ≈ 0.019 y, con el ruido, $\Eout \approx 0.086$.

</v-click>

<v-click>

El sesgo ya casi es cero desde el grado 5; lo que sigue solo agrega varianza, que con 15 puntos crece muy rápido a partir del grado 8.

</v-click>

<v-click>

<div class="pequeno tenue">

En la práctica no podemos calcular $\bar g$: tenemos un solo $\Dat$. La descomposición es para **pensar**; para **medir** usaremos validación.

</div>

</v-click>
</div>
</div>

---

# Curvas de aprendizaje

<CurvasAprendizaje />

<div class="cols" style="gap: 40px; margin-top: 4px">
<div class="pequeno">

$\Ein$ sube y $\Eout$ baja con $N$; las dos convergen a **sesgo² + $\sigma^2$** (la línea punteada es $\sigma^2 = 0.0625$). La recta se estanca pronto, lejos del ruido.

</div>
<div class="pequeno" v-click>

El grado 5 empieza con una brecha enorme (varianza), pero con suficientes datos llega más abajo que la recta.

</div>
</div>

---

# Diagnóstico con curvas de aprendizaje

<div class="cols" style="gap: 56px">
<div>

### Mucho sesgo

- $\Ein$ y $\Eout$ altos y **cercanos**.
- Más datos **no** ayudan.
- Hace falta un $\Hyp$ más rico: más atributos, un modelo más flexible, menos regularización.

</div>
<div v-click>

### Mucha varianza

- $\Ein$ bajo y $\Eout$ alto: **brecha grande**.
- Más datos **sí** ayudan.
- O reducir la flexibilidad: menos atributos, un modelo más simple o **regularizar**.

</div>
</div>

---
layout: seccion
numero: 1
---

# Regularización

Conservar la flexibilidad, pero cobrar por usarla.

---

# Restringir sin quitar

<div class="cols" style="gap: 48px">
<div>

Bajar del grado 12 al 3 es una restricción **dura**: $w_4 = \dots = w_{12} = 0$. Una restricción **suave**:

$$
\min_{\vw} \; \Ein(\vw) \quad \text{sujeto a} \quad \vw\T\vw \le C .
$$

</div>
<div v-click>

Si la restricción está activa, en la solución el gradiente de $\Ein$ es opuesto a la normal de la esfera: $\nabla \Ein(\vw^*) = -2\frac{\lambda}{N}\vw^*$ para algún $\lambda \ge 0$. Es la condición de óptimo de

$$
\min_{\vw} \; \Ein(\vw) + \frac{\lambda}{N}\, \vw\T\vw .
$$

<div class="pequeno tenue">

Cada $C$ corresponde a un $\lambda$: $C$ más chico, $\lambda$ más grande.

</div>
</div>
</div>

---

# Ridge: la solución exacta

<div class="cols" style="gap: 48px">
<div>

Con $\Ein(\vw) = \frac{1}{N}\norm{X\vw - \mathbf y}^2$, el gradiente del objetivo es

$$
\tfrac{2}{N}\big(X\T X\vw - X\T\mathbf y\big) + \tfrac{2\lambda}{N}\vw = 0
$$

$$
\Longrightarrow\quad \vw_{\text{ridge}} = \big(X\T X + \lambda I\big)^{-1} X\T \mathbf y .
$$

<div v-click class="pequeno">

$X\T X + \lambda I$ es invertible para todo $\lambda > 0$, aunque haya columnas colineales o más columnas que filas.

</div>
</div>
<div v-click>

Con la SVD $X = U D V\T$ las predicciones quedan

$$
\hat{\mathbf y} = \sum_{j} \mathbf u_j \, \frac{d_j^2}{d_j^2 + \lambda} \, \mathbf u_j\T \mathbf y .
$$

Ridge **encoge** cada dirección según su valor singular: las de poca varianza ($d_j$ chico) casi desaparecen. Los **grados de libertad efectivos** son $\;\mathrm{df}(\lambda) = \sum_j \frac{d_j^2}{d_j^2 + \lambda}$.

</div>
</div>

---

# El mismo polinomio de grado 12, con distintos $\lambda$

<RidgeCurvas />

<div v-click class="nota">

Con $\lambda = 0$, interpola. Con $\lambda = 10^{-2}$ se parece a $f$ con los mismos 12 grados. Con $\lambda = 1$ se aplana: demasiado sesgo. Un poco de regularización basta.

</div>

---

# $\lambda$ cambia sesgo por varianza

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<RegularizacionLam />

</div>
<div>

Polinomio de grado 12, $N = 15$, promediando 800 conjuntos de entrenamiento.

<v-click>

Al subir $\lambda$, la **varianza baja** varios órdenes de magnitud y el **sesgo sube** poco. El mejor $\lambda$ está entre $0.01$ y $0.03$: el grado 12 regularizado se acerca al grado 3.

</v-click>

<v-click>

<Caja tipo="cuidado">

La penalización $\vw\T\vw$ trata igual a todos los coeficientes: hay que **estandarizar** los atributos antes, como vimos en la Unidad 1.

</Caja>

</v-click>
</div>
</div>

---

# Lasso: cambiar el círculo por un rombo

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<GeometriaL1L2 />

</div>
<div>

$$
\min_{\vw} \; \Ein(\vw) + \frac{\lambda}{N} \sum_j |w_j|
$$

<v-click>

Las curvas de nivel suelen tocar al rombo en una **esquina**, donde algunos $w_j = 0$ exactamente. Lasso **selecciona** variables.

</v-click>

<v-click>

<div class="pequeno tenue">

No tiene forma cerrada ($|w|$ no es derivable en 0); se resuelve por descenso por coordenadas. R. Tibshirani, *J. Royal Statistical Society B* 58 (1996).

</div>

</v-click>
</div>
</div>

---

# Ridge y lasso en datos reales: diabetes

<div class="cols" style="gap: 32px">
<div><Caminos metodo="ridge" /></div>
<div><Caminos metodo="lasso" /></div>
</div>

<div class="cols" style="gap: 32px">
<div class="pequeno">

442 pacientes, 10 variables (edad, sexo, índice de masa corporal, presión y seis análisis de sangre) y la progresión de la enfermedad un año después. Efron et al., *Annals of Statistics* 32 (2004).

</div>
<div class="pequeno" v-click>

Ridge acerca todos los coeficientes a cero a la vez. Lasso los **apaga uno por uno**: con $\lambda$ grande solo quedan el índice de masa corporal y `s5`.

</div>
</div>

---

# Otra mirada: la regularización como creencia previa

<div class="cols" style="gap: 48px">
<div>

Con ruido gaussiano, maximizar la verosimilitud es minimizar el error cuadrático. Si además ponemos una **distribución previa** sobre $\vw$ y buscamos el máximo *a posteriori*:

$$
\max_{\vw}\; \log p(\mathbf y \mid X, \vw) + \log p(\vw) .
$$

</div>
<div>

<v-clicks>

- Previa **gaussiana** $w_j \sim \mathcal N(0, \tau^2)$: $\;\log p(\vw) = -\frac{1}{2\tau^2}\norm{\vw}_2^2 + c$. Es **ridge**, con $\lambda \propto \sigma^2/\tau^2$.
- Previa de **Laplace** $p(w_j) \propto e^{-|w_j|/b}$: es **lasso**.
- Regularizar es decir, antes de ver los datos, que **preferimos pesos pequeños**: la navaja de Occam en forma de ecuación.

</v-clicks>

</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Error esperado = **sesgo² + varianza + ruido**.
2. Más complejidad: menos sesgo, más varianza. Las **curvas de aprendizaje** dicen cuál domina.
3. **Ridge**: $\big(X\T X + \lambda I\big)^{-1} X\T\mathbf y$, encoge las direcciones débiles.
4. **Lasso**: pone coeficientes en cero, selecciona variables.
5. Estandarizar antes de regularizar; $\lambda$ se elige con **validación**.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Sesgo, varianza y regularización* (Unidad 3, ejemplo 2).

**Lectura:** Abu-Mostafa et al., *Learning From Data*, secciones 2.3 y 4.2; James et al., *ISLP*, sección 6.2.

</div>
</div>
</div>
