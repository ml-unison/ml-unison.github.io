---
theme: none
title: Máquinas de vectores de soporte
unidad: Unidad 4
layout: portada
subtitulo: "El margen máximo, su problema dual y el truco del kernel"
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

# Máquinas de vectores de soporte y kernels

---
layout: pregunta
---

# Si muchas rectas separan perfectamente los datos, ¿con cuál nos quedamos?

<v-click>

Con la que deja **más espacio** a los dos lados: si un pingüino nuevo cae un poco lejos de los demás, la recta con más margen tiene más probabilidad de clasificarlo bien.

</v-click>

---

# Todas tienen $\Ein = 0$

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<MargenDuro v-if="$clicks < 1" :margen="false" />
<MargenDuro v-else :otras="false" />

</div>
<div>

Adelie contra Gentoo, 274 pingüinos, dos medidas estandarizadas. Las punteadas son tres corridas del perceptrón.

<v-click>

La **máquina de vectores de soporte** busca la recta con la franja vacía más ancha. Solo la determinan los **3 puntos** encerrados: los vectores de soporte.

</v-click>

<v-click>

<div class="pequeno">

Distancia del punto más cercano a cada recta: 0.06, 0.18 y 0.08 en las del perceptrón; **0.34** en la de margen máximo.

</div>

</v-click>
</div>
</div>

---

# El problema del margen máximo

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

La distancia de $\vx_n$ al hiperplano $\vw\T\vx + b = 0$ es

$$
\frac{|\vw\T\vx_n + b|}{\norm{\vw}} .
$$

<v-click>

$(\vw, b)$ y $(c\vw, cb)$ son el mismo hiperplano. Fijamos la escala con $\min_n y_n(\vw\T\vx_n + b) = 1$; entonces el margen es $1/\norm{\vw}$ y maximizarlo es

$$
\min_{\vw, b}\; \tfrac12 \norm{\vw}^2 \quad \text{s.a.} \quad y_n(\vw\T\vx_n + b) \ge 1 .
$$

</v-click>
</div>
<div>

<v-click>

Un **problema cuadrático convexo** con $N$ restricciones lineales: tiene solución única y hay algoritmos eficientes.

</v-click>

<v-click>

<Caja tipo="idea">

**¿Por qué ayuda el margen?** Los hiperplanos «gordos», con margen $\rho$, no pueden producir todas las dicotomías: si los datos caben en una bola de radio $R$,

$$
d_{\text{VC}}(\rho) \le \big\lceil R^2/\rho^2 \big\rceil + 1 ,
$$

sin importar la dimensión $d$. La Unidad 3, otra vez.

</Caja>

</v-click>
</div>
</div>

---

# El problema dual

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Lagrangiano, con $\alpha_n \ge 0$:

$$
\mathcal{L} = \tfrac12\norm{\vw}^2 - \sum_n \alpha_n\big(y_n(\vw\T\vx_n + b) - 1\big)
$$

<v-click>

$\partial_\vw \mathcal{L} = 0$ y $\partial_b \mathcal{L} = 0$ dan

$$
\vw = \sum_n \alpha_n y_n \vx_n, \qquad \sum_n \alpha_n y_n = 0 .
$$

</v-click>
</div>
<div>

<v-click>

Sustituyendo:

$$
\max_{\boldsymbol\alpha \ge 0}\; \sum_n \alpha_n - \tfrac12 \sum_{n,m} \alpha_n\alpha_m y_n y_m\, \vx_n\T \vx_m
$$

con $\sum_n \alpha_n y_n = 0$.

</v-click>

<v-click>

**KKT:** $\alpha_n\big(y_n(\vw\T\vx_n + b) - 1\big) = 0$. Solo los puntos **sobre el margen** tienen $\alpha_n > 0$: son los vectores de soporte. En los pingüinos, $\boldsymbol\alpha = (4.37,\ 3.76,\ 0.60)$ y 271 ceros.

</v-click>

<v-click>

<div class="nota">

Los datos solo aparecen en **productos internos** $\vx_n\T\vx_m$. Esta es la puerta a los kernels.

</div>

</v-click>
</div>
</div>

---

# Pocos vectores de soporte, buena generalización

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="proposicion">

Para la SVM de margen duro,

$$
E_{\text{cv}} \le \frac{\#\,\text{vectores de soporte}}{N} ,
$$

donde $E_{\text{cv}}$ es el error de dejar uno fuera.

</Caja>

</div>
<div>

<v-click>

**Idea de la prueba.** Si sacamos un punto que **no** es vector de soporte, la solución no cambia (sus restricciones no estaban activas), y ese punto queda del lado correcto: error 0. Solo los vectores de soporte pueden contribuir.

</v-click>

<v-click>

Adelie contra Gentoo: cota $3/274 = 0.011$; error de dejar uno fuera real, **0**.

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# Margen suave

Cuando ninguna recta separa los datos.

---

# Variables de holgura

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Adelie y Chinstrap se enciman: el problema de margen duro **no tiene solución**. Permitimos violaciones $\xi_n \ge 0$, con un precio:

$$
\min_{\vw, b, \boldsymbol\xi}\; \tfrac12\norm{\vw}^2 + C\sum_n \xi_n
$$

$$
\text{s.a.}\quad y_n(\vw\T\vx_n + b) \ge 1 - \xi_n,\quad \xi_n \ge 0 .
$$

</div>
<div>

<v-clicks>

- $0 < \xi_n \le 1$: dentro del margen pero bien clasificado; $\xi_n > 1$: mal clasificado.
- En el dual cambia una sola cosa: $0 \le \alpha_n \le C$.
- $C$ grande: casi margen duro (poco sesgo, más varianza). $C$ chico: margen ancho, muchos vectores de soporte.

</v-clicks>
</div>
</div>

---

# El efecto de $C$

<MargenSuave />

---

# La SVM es pérdida bisagra más regularización

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

En el óptimo, $\xi_n = \max\big(0,\; 1 - y_n s_n\big)$ con $s_n = \vw\T\vx_n + b$. Sustituyendo y dividiendo entre $CN$:

$$
\min_{\vw, b}\; \frac1N \sum_n \max(0,\, 1 - y_n s_n) + \lambda\norm{\vw}^2, \quad \lambda = \frac{1}{2CN} .
$$

<v-click>

Es el mismo esquema de la Unidad 3: **error en la muestra + regularización**, con otra pérdida.

</v-click>
</div>
<div>

<Perdidas />

<v-click>

<div class="pequeno">

Bisagra y logística son cotas convexas de la pérdida 0-1. La bisagra vale **exactamente cero** pasado el margen: por eso la solución depende solo de algunos puntos.

</div>

</v-click>
</div>
</div>

---
layout: seccion
numero: 3
---

# El truco del kernel

Fronteras no lineales sin calcular las transformaciones.

---

# Transformar para separar

<Circulos />

<div class="cols" style="gap: 40px; margin-top: 12px">
<div>

En $(x_1, x_2)$ ninguna recta separa los anillos. En $\Phi(\vx) = (x_1, x_2, x_1^2 + x_2^2)$, un plano horizontal sí.

</div>
<div v-click>

Pero $\Phi$ puede tener muchísimas dimensiones. El dual solo necesita $\Phi(\vx)\T\Phi(\vx')$: si lo calculamos **sin construir** $\Phi$, no pagamos la dimensión.

</div>
</div>

---

# Kernels

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Un **kernel** es $K(\vx, \vx') = \Phi(\vx)\T\Phi(\vx')$. Polinomial de grado 2 en $\R^2$:

$$
(1 + \vx\T\vx')^2 = \Phi(\vx)\T\Phi(\vx'),
$$

$$
\Phi(\vx) = \big(1, \sqrt2 x_1, \sqrt2 x_2, x_1^2, x_2^2, \sqrt2 x_1x_2\big) .
$$

<v-click>

Con grado $Q$ en $\R^d$, $\Phi$ tiene $\binom{d+Q}{Q}$ coordenadas, pero $K$ cuesta $O(d)$.

</v-click>
</div>
<div>

<v-click>

**Gaussiano (RBF):** $K(\vx, \vx') = e^{-\gamma\norm{\vx - \vx'}^2}$. Expandiendo $e^{2\gamma\vx\T\vx'}$ en serie de Taylor aparecen monomios de **todos** los grados: $\Phi$ tiene dimensión infinita.

</v-click>

<v-click>

<Caja tipo="teorema">

**Mercer.** $K$ simétrica es un kernel válido si y solo si toda matriz de Gram $[K(\vx_n, \vx_m)]$ es semidefinida positiva.

</Caja>

</v-click>

<v-click>

<div class="pequeno">

La predicción: $g(\vx) = \operatorname{signo}\big(\sum_{n \in \text{VS}} \alpha_n y_n K(\vx_n, \vx) + b\big)$.

</div>

</v-click>
</div>
</div>

---

# El ancho del kernel gaussiano

<RBFGamma />

<div class="cols" style="gap: 40px; margin-top: 14px">
<div>

200 puntos de las «lunas» con ruido, $C = 1$. $\gamma$ es el inverso del radio de influencia de cada vector de soporte.

</div>
<div v-click>

$\gamma$ chico: casi lineal, **subajuste**. $\gamma$ grande: una isla alrededor de cada punto, casi todos son vectores de soporte, **sobreajuste**. Se elige con validación cruzada, junto con $C$.

</div>
</div>

---

# En la práctica: dígitos escritos a mano

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<MallaDigitos />

<div class="pequeno tenue">

Exactitud (%) por validación cruzada en 1,347 dígitos de 8 × 8 (Unidad 2), con kernel gaussiano.

</div>
</div>
<div>

<v-clicks>

- Mejor combinación en esta malla: $C = 10$, $\gamma = 0.1$. Exactitud de prueba **99.3 %** (3 errores en 450), contra 97.1 % de la logística.
- 545 de los 1,347 dígitos son vectores de soporte (40 %).
- **Escala los atributos**: $\gamma$ mide distancias.
- Entrenar cuesta entre $O(N^2)$ y $O(N^3)$: con cientos de miles de datos se usa una SVM lineal o características aleatorias de Fourier (Rahimi y Recht, 2007).

</v-clicks>
</div>
</div>

---

# El mapa de la unidad

| modelo | salida | pérdida | cómo se ajusta |
|---|---|---|---|
| regresión lineal | número real | cuadrática | forma cerrada (QR, SVD) |
| regresión logística | probabilidad | entropía cruzada | Newton (IRLS) |
| regresión de Poisson | conteo esperado | devianza de Poisson | Newton (IRLS) |
| LDA y QDA | probabilidad | verosimilitud conjunta | medias y covarianzas |
| Bayes ingenuo | probabilidad | verosimilitud conjunta | conteos |
| SVM | clase | bisagra + $\lambda\norm{\vw}^2$ | programación cuadrática (dual) |

<div v-click class="nota" style="margin-top: 20px">

Todos son lineales en algún espacio: en los atributos, en los log-momios, en el logaritmo de la media o, con un kernel, en un espacio que nunca se construye. La Unidad 5 deja de serlo.

</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Margen máximo: $\min \tfrac12\norm{\vw}^2$ con $y_n(\vw\T\vx_n + b) \ge 1$.
2. El dual depende solo de productos internos; los $\alpha_n > 0$ marcan los vectores de soporte.
3. $E_{\text{cv}} \le \#\text{VS}/N$.
4. Margen suave = pérdida bisagra + regularización; $C$ controla el compromiso.
5. Kernel: un producto interno en otro espacio. $C$ y $\gamma$ se eligen con validación cruzada.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Máquinas de vectores de soporte y kernels* (Unidad 4, ejemplo 4).

**Lectura:** *Learning From Data*, capítulo electrónico 8 (*Support Vector Machines*); *The Elements of Statistical Learning*, secciones 12.1–12.3.

</div>
</div>
</div>
