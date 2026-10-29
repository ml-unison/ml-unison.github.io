---
theme: none
title: Regresión logística y modelos lineales generalizados
unidad: Unidad 4
layout: portada
subtitulo: "Del Challenger a las bicicletas: probabilidades, conteos y la familia exponencial"
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

# Regresión logística y modelos lineales generalizados

---
layout: caso
caso: Caso · Responsabilidad
cifra: 31 °F
cifraNota: temperatura pronosticada para el despegue del Challenger; el vuelo más frío hasta entonces había sido a 53 °F.
fuente: Report of the Presidential Commission on the Space Shuttle Challenger Accident (Comisión Rogers, 1986).
---

# La noche del 27 de enero de 1986

El transbordador *Challenger* se desintegró 73 segundos después del despegue; murieron sus siete tripulantes. Falló una **junta tórica** de uno de los cohetes de combustible sólido.

<v-click>

La noche anterior, ingenieros de Morton Thiokol, el fabricante de los cohetes, recomendaron no lanzar por el frío. La gráfica que se discutió mostraba **solo los vuelos con incidentes** en las juntas.

</v-click>

---

# Lo que faltaba en la gráfica: los ceros

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<ChallengerPuntos v-if="$clicks < 1" :todos="false" />
<ChallengerPuntos v-else />

</div>
<div>

Los 23 vuelos anteriores: temperatura al despegue y número de juntas (de 6) con erosión o fuga de gases.

<v-click>

Con **todos** los vuelos, el patrón aparece: los 4 vuelos por debajo de 65 °F tuvieron daño; de los 19 restantes, solo 3.

</v-click>

<v-click>

<div class="nota">

Quitar los casos «sin evento» es un **sesgo de selección**: los datos que no vemos también son datos.

</div>

</v-click>
</div>
</div>

---

# Un modelo para una probabilidad

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Queremos $\Prob(y = 1 \mid \vx)$, la probabilidad de la clase 1 dados los atributos $\vx$. Una recta no sirve: se sale de $[0, 1]$. Modelamos los **log-momios** como lineales, con pesos $\vw$ y sesgo $b$:

$$
\ln \frac{\Prob(y = 1 \mid \vx)}{1 - \Prob(y = 1 \mid \vx)} = \vw\T \vx + b
$$

<v-click>

y despejando,

$$
\Prob(y = 1 \mid \vx) = \sigma(\vw\T\vx + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}} ,
$$

con $\sigma$ la **función logística**.

</v-click>
</div>
<div>

<Sigmoide />

<v-click>

<div class="pequeno">

**Interpretación:** subir el atributo $x_j$ en una unidad multiplica los momios por $e^{w_j}$, con $w_j$ su peso: la **razón de momios**.

</div>

</v-click>
</div>
</div>

---

# Máxima verosimilitud y entropía cruzada

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Con $M$ datos, $y^{(i)} \in \{0, 1\}$ y $p^{(i)} = \sigma(\vw\T \vx^{(i)} + b)$ la probabilidad predicha, la log-verosimilitud negativa promedio es la **entropía cruzada**:

$$
L(\vw, b) = -\frac1M \sum_{i=1}^{M} \big[ y^{(i)} \ln p^{(i)} + (1 - y^{(i)}) \ln(1 - p^{(i)}) \big] .
$$

<div class="pequeno tenue">

Con $y^{(i)} \in \{-1, +1\}$, como en *Learning From Data*, es lo mismo: $L = \frac1M \sum_i \ln\big(1 + e^{-y^{(i)} (\vw\T \vx^{(i)} + b)}\big)$.

</div>
</div>
<div>

<v-click>

Como $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, respecto a $\tilde\vw = (b, \vw)$:

$$
\nabla L = \tfrac1M \tilde\vX\T(\mathbf{p} - \vy), \quad \nabla^2 L = \tfrac1M \tilde\vX\T \mathbf{S} \tilde\vX ,
$$

con $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$ (datos con una columna de unos), $\mathbf{p} = (p^{(i)})$ y $\mathbf{S}$ la matriz diagonal con entradas $p^{(i)}(1 - p^{(i)})$.

</v-click>

<v-click>

$\mathbf{S}$ tiene diagonal positiva, así que la hessiana es semidefinida positiva: **convexa**, sin mínimos locales. Pero $\nabla L = \mathbf{0}$ no tiene solución en forma cerrada.

</v-click>
</div>
</div>

---

# El método de Newton es mínimos cuadrados repetidos

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

$$
\tilde\vw_{t+1} = \tilde\vw_t - \big(\tilde\vX\T \mathbf{S} \tilde\vX\big)^{-1} \tilde\vX\T(\mathbf{p} - \vy)
$$

con $\tilde\vw = (b, \vw)$, $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$, y las probabilidades $\mathbf{p}$ y la matriz $\mathbf{S} = \operatorname{diag}\big(p^{(i)}(1 - p^{(i)})\big)$ evaluadas en la iteración $t$.

<v-click>

Reacomodando,

$$
\tilde\vw_{t+1} = \big(\tilde\vX\T \mathbf{S} \tilde\vX\big)^{-1} \tilde\vX\T \mathbf{S}\, \vz, \quad \vz = \tilde\vX\tilde\vw_t + \mathbf{S}^{-1}(\vy - \mathbf{p}) :
$$

mínimos cuadrados **ponderados** con pesos $\mathbf{S}$ y respuesta de trabajo $\vz$. Por eso se llama IRLS.

</v-click>
</div>
<div v-click>

Challenger, un solo atributo $x$ = temperatura en °F, desde $b = w = 0$:

| iteración | $b$ | $w$ | log-verosimilitud |
|---|---|---|---|
| 0 | 0 | 0 | −15.94 |
| 1 | 9.62 | −0.150 | −10.59 |
| 2 | 13.66 | −0.211 | −10.18 |
| 3 | 14.94 | −0.231 | −10.158 |
| 4 | 15.04 | −0.232 | −10.1576 |

<div class="pequeno tenue">

Convergencia cuadrática: cada iteración duplica los dígitos correctos.

</div>
</div>
</div>

---

# La predicción para el 28 de enero

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<ChallengerCurva />

</div>
<div>

$e^{0.232} = 1.26$: **cada grado Fahrenheit menos multiplica los momios de daño por 1.26** (por 10 en 10 grados).

<v-click>

A 31 °F el modelo da $\Prob(\text{daño}) = 0.9996$. La banda contiene el 90 % central de 1,000 réplicas bootstrap; a 31 °F va de 0.96 a 1.

</v-click>

<v-click>

<div class="nota">

**Cuidado:** es una extrapolación de 22 grados fuera de los datos. Lo honesto no es «99.96 %», sino «no tenemos ninguna evidencia de que sea seguro, y todo indica lo contrario».

</div>

</v-click>
</div>
</div>

---

# Cuando los datos se separan, el estimador no existe

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

Si existen $\vw, b$ con $\vw\T\vx^{(i)} + b > 0$ para todos los $y^{(i)} = 1$ y $< 0$ para los $y^{(i)} = 0$, entonces $(c\vw, cb)$ con $c \to \infty$ hace que cada probabilidad predicha $p^{(i)} \to y^{(i)}$ y la entropía cruzada

$$
L(c\vw, cb) \to 0 \quad \text{sin alcanzarlo.}
$$

<v-click>

El mínimo no existe: el algoritmo hace crecer $\norm{\vw}$ para siempre y la curva se vuelve un escalón.

</v-click>

<v-click>

**Remedio:** regularizar, $L(\vw, b) + \lambda\norm{\vw}^2$ con $\lambda > 0$, que es lo que scikit-learn hace por omisión (`C = 1`, el inverso de la fuerza de la regularización).

</v-click>
</div>
<div>

<Separable />

<div class="pequeno tenue">

Challenger sin los tres vuelos con daño a 70 y 75 °F: los 20 restantes se separan en 64.5 °F.

</div>
</div>
</div>

---

# Más de dos clases: *softmax*

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Con $K$ clases, unos pesos y un sesgo por clase, $(\vw_k, b_k)$ para $k = 1, \dots, K$:

$$
\Prob(y = k \mid \vx) = \frac{e^{\vw_k\T \vx + b_k}}{\sum_{k'=1}^{K} e^{\vw_{k'}\T \vx + b_{k'}}} .
$$

<v-click>

Con $K = 2$ se recupera la logística con $\vw = \vw_1 - \vw_2$ y $b = b_1 - b_2$. La pérdida sigue siendo la entropía cruzada, ahora $-\frac1M\sum_{i=1}^{M} \ln \Prob(y^{(i)} \mid \vx^{(i)})$, y sigue siendo convexa.

</v-click>
</div>
<div v-click>

- Solo importan las **diferencias** $\vw_k - \vw_{k'}$ entre clases: los pesos no son identificables sin regularización o sin fijar una clase de referencia.
- Las fronteras entre cada par de clases son **hiperplanos**.
- Es la última capa de casi todas las redes neuronales clasificadoras.

</div>
</div>

---
layout: seccion
numero: 2
---

# Modelos lineales generalizados

Una sola receta para medias, probabilidades y conteos.

---

# Tres modelos, una misma forma

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

La **familia exponencial** (con dispersión fija), con parámetro natural $\theta$:

$$
p(y \mid \theta) = c(y) \exp\big(\theta\, y - A(\theta)\big), \qquad \E[y] = A'(\theta) ,
$$

donde $A$ normaliza la distribución y $c(y)$ no depende de $\theta$.

<v-click>

Un **GLM** hace lineal el parámetro natural, $\theta = \vw\T\vx + b,$ y la **liga** $g$ conecta la media $\mu = \E[y]$ con la misma parte lineal: $g(\mu) = \vw\T\vx + b$ también.

</v-click>
</div>
<div v-click>

| respuesta | distribución | $A(\theta)$ | liga canónica |
|---|---|---|---|
| real | normal | $\theta^2/2$ | identidad |
| 0/1 | Bernoulli | $\ln(1 + e^\theta)$ | logit |
| conteo | Poisson | $e^\theta$ | logaritmo |

</div>
</div>

<div v-click class="nota" style="margin-top: 20px">

**Un solo gradiente.** Con la liga canónica, $\theta^{(i)} = \vw\T\vx^{(i)} + b$ y la log-verosimilitud es $\ln \mathcal{L} = \sum_i \big[y^{(i)} \theta^{(i)} - A(\theta^{(i)})\big] + \text{cte}$. Respecto a $\tilde\vw = (b, \vw)$, con $\tilde\vX = [\,\mathbf{1} \;\; \vX\,]$ y $\boldsymbol\mu$ el vector de medias $\mu^{(i)} = A'(\theta^{(i)})$: $\nabla \ln \mathcal{L} = \tilde\vX\T(\vy - \boldsymbol\mu)$ y $\nabla^2 \ln \mathcal{L} = -\tilde\vX\T \operatorname{diag}\big(A''(\theta^{(i)})\big) \tilde\vX$. Mínimos cuadrados y logística son casos particulares, y Newton es IRLS en todos.

</div>

---

# Regresión de Poisson para las bicicletas

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

$$
y \mid \vx \sim \operatorname{Poisson}(\mu), \qquad \ln \mu = \vw\T \vx + b ,
$$

con $y$ las rentas en una hora y $\mu$ su media dados los atributos $\vx$.

<v-clicks>

- $\mu = e^{\vw\T\vx + b} > 0$ siempre: **no hay predicciones negativas**.
- $\Var[y] = \mu$: la varianza crece con la media, como el embudo de los residuos.
- Los efectos son **multiplicativos**: la lluvia no quita 72 bicicletas a toda hora; multiplica por $e^{w_j} = 0.61$ ($w_j$: el peso de la lluvia), un 39 % menos, a las 3 de la mañana y a las 8.

</v-clicks>
</div>
<div v-click>

| modelo | RECM | devianza | negativas |
|---|---|---|---|
| M3 lineal | 107.4 | 56.6 | 9.2 % |
| M3 Poisson | 98.6 | 39.7 | 0 % |
| M4 lineal | 86.4 | 40.0 | 10.0 % |
| M4 Poisson | **72.2** | **20.9** | 0 % |

<div class="pequeno tenue">

Mismos atributos, distinta familia. Devianza de Poisson media en prueba (a la lineal se le recortan las predicciones a 0.5). El año 2012 multiplica las rentas por 1.60.

</div>
</div>
</div>

---

# Un martes de prueba, hora por hora

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<DiaPoisson />

</div>
<div>

<v-clicks>

- En la madrugada, el modelo lineal predice entre −44 y −80 bicicletas por hora.
- Para compensar, en las horas pico se pasa: 474 contra 299 observadas a las 17 h.
- Poisson sigue la forma del día porque la suma en logaritmo es un **producto** en la escala original: hora × día × clima.

</v-clicks>

<div v-click class="nota">

Elegir la familia es elegir **qué ruido esperamos**. La pérdida cuadrática supone un ruido simétrico de varianza constante.

</div>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. La logística modela los **log-momios** como lineales; $e^{w_j}$, con $w_j$ el peso del atributo $j$, es una razón de momios.
2. Entropía cruzada convexa, sin forma cerrada: **Newton = IRLS**.
3. Con datos separables, sin regularizar, el estimador no existe.
4. GLM: familia exponencial + liga; el mismo gradiente para todos, datos por residuos $\vy - \boldsymbol\mu$ (salidas menos medias predichas).
5. Los ceros, la extrapolación y la familia también son decisiones con consecuencias.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Regresión logística y modelos lineales generalizados* (Unidad 4, ejemplo 2).

**Laboratorio 4:** logística con Newton y Bayes ingenuo, desde cero.

**Lectura:** *Learning From Data*, sección 3.3; S. R. Dalal, E. B. Fowlkes y B. Hoadley, «Risk analysis of the space shuttle», *JASA* 84 (1989).

</div>
</div>
</div>
