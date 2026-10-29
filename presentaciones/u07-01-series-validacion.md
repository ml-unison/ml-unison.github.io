---
theme: none
title: "Series de tiempo: estructura y validación"
unidad: Unidad 7
layout: portada
subtitulo: "El clima de Hermosillo: tendencia, estacionalidad, autocorrelación y cómo validar un pronóstico"
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
  Reconocimiento de Patrones · Unidad 7 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Series de tiempo: estructura y validación

---
layout: pregunta
---

# ¿Qué temperatura hará mañana en Hermosillo?

<v-click>

Dos respuestas sin modelo: **la de hoy**, o **la normal para esta fecha**. En 2021–2024, la primera se equivocó en promedio **1.28 °C** y la segunda **2.17 °C**.

</v-click>

<v-click>

Hoy veremos de dónde salen esos números, por qué son tan difíciles de vencer y cómo medir honestamente si un modelo los vence.

</v-click>

<!--
Dejar que el grupo proponga. Casi siempre salen las dos respuestas: «igual que hoy» y «lo normal para la fecha». Preguntar cuál creen que es mejor a un día y cuál a dos semanas: la respuesta cambia con el horizonte, y eso es lo que veremos.
-->

---

# 33 años de un termómetro en el aeropuerto

<SerieHermosillo v-if="$clicks < 1" :alto="380" />
<SerieHermosillo v-else zoom :alto="380" />

<div class="pequeno" style="margin-top: 8px">

<span v-if="$clicks < 1">Temperatura media diaria, estación MXM00076160 (GHCN-D, NOAA), 1992–2024: 12,054 días; faltan 82, que rellenamos por interpolación. Entrenamos con 1992–2020 y **no tocamos 2021–2024**.</span>
<span v-else>Un año de cerca: un **ciclo anual** de 18 °C, y encima una **variación de día a día** de un par de grados. Lo que se sale de la climatología es la **anomalía**.</span>

</div>

<v-click><span /></v-click>

<!--
Primer paso con cualquier serie: reindexar a su frecuencia. El archivo trae 11,972 renglones; si no reindexamos, «el renglón anterior» no siempre es «el día anterior» y todos los rezagos salen mal sin avisar. Casi todos los faltantes son de 1994.
Al hacer clic: el año 2023 con la climatología encima. Se ven el ciclo anual, el aplanamiento del verano del monzón y los frentes fríos del invierno.
-->

---

# Lo normal para cada fecha: la climatología

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<ClimatologiaHmo />

</div>
<div>

Promediar cada día del año da una curva ruidosa (29 años). Mejor, una **regresión lineal sobre armónicos** del día del año $d$, con $\omega = 2\pi/365.25$, sesgo $b$ y pesos $\alpha_j, \beta_j$:

$$
c(d) = b + \sum_{j=1}^{3}\big[\alpha_j \cos j\omega d + \beta_j \sin j\omega d\big]
$$

<v-click>

La **anomalía** es $a_t = y_t - c(d_t)$: la temperatura $y_t$ menos la climatología de su día del año $d_t$. La climatología explica el **82 %** de la varianza diaria; las anomalías tienen desviación estándar de **2.85 °C**.

</v-click>
</div>
</div>

<!--
La climatología se ajusta solo con 1992–2020 para poder usarla como pronóstico de 2021–2024 sin trampa. Mínimo de 16.4 °C a principios de enero, máximo de 34.0 °C a principios de julio. No es una senoide: sube rápido en primavera y se aplana en el verano; eso lo dan los armónicos 2 y 3. Son mínimos cuadrados con atributos no lineales en d.
-->

---

# Tendencia, estacionalidad y residuo: STL

<STLHermosillo />

<div class="pequeno tenue" style="margin-top: 6px">

STL (Cleveland et al., 1990) alterna suavizamientos locales: $y_t = \tau_t + s_t + r_t$ (tendencia, estacionalidad y residuo), y deja que el ciclo anual cambie poco a poco. La tendencia va de 24.3 a 28.0 °C: ¿calentamiento? Antes necesitamos medir la memoria de la serie.

</div>

<!--
STL: suaviza cada «subserie» de días equivalentes de distintos años para obtener la estacionalidad, suaviza lo que queda para la tendencia, y repite. La estacionalidad sale rugosa porque cada día del año se suaviza solo a través de los años, no con sus vecinos del calendario; por eso para pronosticar preferimos la climatología con armónicos.
Las otras dos componentes clásicas: el ciclo (oscilaciones sin periodo fijo, como El Niño) y el ruido.
-->

---
layout: seccion
numero: 1
---

# Autocorrelación

Cuánto se parece una serie a sí misma unos pasos atrás.

---

# La autocorrelación muestral

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Con rezago $k$:

$$
\hat\gamma(k) = \frac1T \sum_{t=k+1}^{T} (y_t - \bar y)(y_{t-k} - \bar y), \qquad \hat\rho(k) = \frac{\hat\gamma(k)}{\hat\gamma(0)}
$$

<div class="pequeno tenue">

$T$ es el número de días y $\bar y$ la media de la serie; $\hat\gamma$ es la autocovarianza y $\hat\rho$ la autocorrelación. Se divide entre $T$, no entre $T-k$: la matriz de autocovarianzas queda semidefinida positiva y los rezagos con pocos pares se encogen hacia cero.

</div>
</div>
<div>

<v-click>

<Caja tipo="definicion" nombre="ruido blanco">

$\E[\varepsilon_t] = 0$, varianza $\Var(\varepsilon_t) = \sigma^2$ y $\operatorname{Cov}(\varepsilon_t, \varepsilon_s) = 0$ si $t \neq s$. Sin memoria: nada mejor que su media para pronosticarlo.

</Caja>

</v-click>

<v-click>

Si la serie es ruido blanco, $\hat\rho(k) \approx \mathcal N(0, 1/T)$ para cada $k$ fijo: el 95 % cae en $\pm 1.96/\sqrt{T}$. Con $T = 10,593$, la banda es $\pm 0.019$.

<div class="pequeno tenue">

Es una banda **rezago por rezago**: con 40 rezagos, que se salgan uno o dos es normal.

</div>

</v-click>
</div>
</div>

<!--
Es la correlación de la serie con ella misma recorrida k pasos, usando la media global. En la libreta se programa en tres líneas y coincide con statsmodels. Con un ruido blanco simulado de la misma longitud, 1 de 40 rezagos queda fuera de la banda. Para probar muchos rezagos a la vez existe la prueba de Ljung–Box.
-->

---

# El ciclo anual tapa la memoria

<ACFHermosillo :anomalias="$clicks >= 1" />

<div class="cols pequeno" style="gap: 28px; margin-top: 6px">
<div>

La ACF de la temperatura oscila con periodo de un año: $\hat\rho(182) = -0.79$. No es memoria: es que julio siempre es caliente.

</div>
<div v-click>

Sin el ciclo, la anomalía de hoy tiene correlación **0.79** con la de ayer y **0.21** con la de hace una semana. Pero ni a 30 días llega a la banda: hay meses y años enteros anómalos.

</div>
</div>

<!--
La estacionalidad aparece en la ACF como una oscilación con el periodo m, y tapa todo lo demás. La memoria de verdad está en las anomalías: un frente frío dura unos días. Que la ACF no llegue a cero a 30 días dice que hay persistencia a escalas más largas (meses, años); lo usaremos al hablar de la tendencia.
-->

---

# Estacionariedad y la caminata aleatoria

<div class="cols" style="grid-template-columns: 2fr 3fr; gap: 40px">
<div>

<Caja tipo="definicion" nombre="estacionariedad débil">

La media $\E[y_t] = \mu$ y la autocovarianza $\operatorname{Cov}(y_t, y_{t-k}) = \gamma(k)$ no dependen de $t$, con varianza finita.

</Caja>

<div class="pequeno">

La temperatura no lo es (su media depende de la fecha); las anomalías, casi.

</div>

<v-click>

Caminata: $y_t = y_{t-1} + \varepsilon_t = \varepsilon_1 + \dots + \varepsilon_t$, así que

$$
\Var(y_t) = t\,\sigma^2 .
$$

<div class="pequeno">

No es estacionaria; su **diferencia** $y_t - y_{t-1} = \varepsilon_t$ sí.

</div>

</v-click>
</div>
<div v-click="1">

<div style="max-width: 540px; margin: 0 auto">

<CaminataAleatoria :varianza="false" />

</div>

<div class="pequeno tenue">

Con choques $\varepsilon_t$ de ruido blanco con varianza $\sigma^2 = 1$ y 2,000 caminatas: varianza empírica de 96.2, 253.7 y 528.3 en $t = 100, 250, 500$.

</div>
</div>
</div>

<!--
Toda la teoría de la ACF supone que la correlación entre y_t y y_{t-k} depende solo de k. La caminata aleatoria es el ejemplo clásico de lo contrario: la media es constante pero la varianza crece con t, y la covarianza (t-k)σ² depende de t. Su ACF muestral decae casi en línea recta, señal típica de no estacionariedad. Diferenciar es la otra forma clásica de quitar tendencias; es la base de los modelos ARIMA.
-->

---

# ¿Se está calentando Hermosillo?

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<TendenciaHmo />

</div>
<div>

Recta sobre las anomalías medias de cada año: **−0.05 °C por década**, intervalo de 95 % de **−0.36 a +0.26**. No se distingue de cero.

<v-click>

Con las 12,054 anomalías diarias y mínimos cuadrados ordinarios:

| error estándar | pendiente / e.e. |
|---|---|
| OLS ingenuo: 0.027 | −1.91 |
| HAC (Newey–West): 0.149 | −0.35 |

</v-click>

<v-click>

<div class="pequeno">

OLS cree tener 12,054 datos independientes. **La autocorrelación hace optimista cualquier fórmula que suponga independencia.**

</div>

</v-click>
</div>
</div>

<!--
Con cautela: esto no contradice el calentamiento documentado en el noroeste de México. Dice que una sola estación, con 33 años y variación interanual de ±1 °C (de −1.21 °C en 2006 a +2.08 °C en 2014), no alcanza para medir una tendencia de décimas de grado por década. Además, una estación de aeropuerto puede cambiar de instrumentos, ubicación o entorno; la racha fría de 2004–2008 podría ser en parte un caso así. Los estudios serios usan muchas estaciones homogeneizadas.
El error estándar ingenuo es 5.5 veces menor: con él casi anunciaríamos que Hermosillo se enfría. Lo mismo le pasará a la validación cruzada.
-->

---
layout: seccion
numero: 2
---

# Pronósticos de referencia

Lo más simple que se puede hacer, y cómo medir cuánto se equivoca.

---

# Cuatro referencias

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

| método | $\hat y_{T+h\mid T}$ |
|---|---|
| ingenuo | $y_T$ |
| estacional ingenuo | $y_{T+h-m}$, con $m = 365$ |
| climatología | $c(d_{T+h})$ |
| deriva | $y_T + h\,\dfrac{y_T - y_1}{T-1}$ |

<div class="pequeno tenue">

Con datos $y_1, \dots, y_T$ se pronostica $h$ días adelante; $m$ es el periodo estacional y $c(d)$ la climatología del día del año $d$.

</div>

</div>
<div>

<v-click>

La deriva sirve con tendencias claras (CO₂, población); aquí es casi el ingenuo.

</v-click>

<v-click>

Una quinta, sugerida por la ACF: partir de la anomalía de hoy y **dejar que se desvanezca**,

$$
\hat y_{T+h\mid T} = c(d_{T+h}) + \hat\phi^{\,h} a_T ,
$$

con $a_T$ la anomalía de hoy y $\hat\phi = \hat\rho_a(1) = 0.794$, la autocorrelación de las anomalías a un día. Es un **AR(1)** de las anomalías: la de mañana es $\phi$ veces la de hoy, más ruido.

</v-click>
</div>
</div>

<!--
Notación: serie y_1..y_T, pronóstico de y_{T+h} con información hasta T, horizonte h, periodo estacional m. Cualquier modelo que propongamos se compara contra estas referencias; si no las vence, no sirve.
-->

---

# Desde el 15 de enero de 2023

<ReferenciasHmo :mostrar="Math.min($clicks + 1, 4)" :observado="$clicks >= 4" />

<v-click><span /></v-click>
<v-click><span /></v-click>
<v-click><span /></v-click>
<v-click><span /></v-click>

<div class="pequeno" v-if="$clicks >= 4">

Nadie vio venir el frente frío. El AR(1) empieza en el ingenuo y en una semana se funde con la climatología; el estacional ingenuo repite el ruido de 2022.

</div>

<!--
Ir mostrando una referencia por clic: ingenuo (se queda en el valor del origen), climatología (ignora el origen, sigue lo normal de enero), AR(1) (arranca en el ingenuo y converge a la climatología), estacional ingenuo (repite el año pasado, con todo su ruido). Al final, lo que pasó: un frente frío que ninguna referencia anticipa. Pronosticar más allá de unos días requiere información de fuera de la serie (los modelos meteorológicos usan el estado de toda la atmósfera).
-->

---

# ¿Hasta dónde sirve cada una?

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<ErrorHorizonteHmo />

</div>
<div>

**Origen móvil:** cada día de 2021–2024 es un origen; la climatología y el coeficiente $\hat\phi$ del AR(1) se re-ajustan cada 1 de enero con todo el pasado.

<v-clicks>

- El ingenuo gana a un día (1.28 °C) y empata con la climatología a $h = 3$ días.
- El AR(1) es el mejor en todos los horizontes; a dos semanas ya es la climatología.
- Más allá, **la mejor información sobre un día es su fecha**.

</v-clicks>

</div>
</div>

<!--
Mover el deslizador: h = 1, 3, 7, 14, 30. El estacional ingenuo es el peor en todos los horizontes de menos de tres semanas: un solo año pasado trae todo el ruido de ese año. Moraleja: un modelo de temperatura diaria se reporta por horizonte. Un MAE de 2 °C es excelente a diez días y malo a uno.
-->

---

# Métricas de error

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 44px">
<div>

$$
\text{MAE} = \tfrac1M \textstyle\sum |e_t|, \qquad \text{RECM} = \sqrt{\tfrac1M \textstyle\sum e_t^2}
$$

$$
\text{MAPE} = \tfrac{100}{M}\textstyle\sum \left|\dfrac{e_t}{y_t}\right|, \qquad \text{MASE} = \dfrac{\text{MAE}}{q}
$$

<div class="pequeno">

$e_t = y_t - \hat y_{t \mid t-h}$ es el error de cada uno de los $M$ pronósticos evaluados. $q = \frac{1}{T-m}\sum_{t>m} |y_t - y_{t-m}|$: el MAE del estacional ingenuo **dentro del entrenamiento**, con periodo $m = 365$ (Hyndman y Koehler, 2006); aquí $q = 3.11$ °C.

</div>
</div>
<div>

Con $h = 1$, en los 1,461 días de 2021–2024:

| | MAE | RECM | MASE | MAPE °C | MAPE K |
|---|---|---|---|---|---|
| ingenuo | 1.28 | 1.69 | 0.41 | 5.5 % | 0.43 % |
| estac. ingenuo | 3.20 | 4.07 | 1.03 | 13.7 % | 1.07 % |
| climatología | 2.17 | 2.70 | 0.70 | 9.6 % | 0.73 % |
| AR(1) | **1.21** | **1.60** | **0.39** | 5.2 % | 0.41 % |

<v-click>

<div class="pequeno">

**Mismos pronósticos, MAPE 13 veces menor en kelvin.** MAPE depende del cero de la escala y explota cerca de él; MASE no.

</div>

</v-click>
</div>
</div>

<!--
MAE lo minimiza la mediana; RECM, la media, y castiga más los errores grandes. MASE < 1 quiere decir «mejor que el estacional ingenuo dentro de la muestra». Es invariante ante transformaciones afines de los datos (ejercicio de la libreta). El MAPE solo tiene sentido con escalas de razón (con un cero verdadero) y lejos de cero.
-->

---
layout: seccion
numero: 3
---

# Validar hacia adelante

Por qué no se baraja el tiempo.

---

# Barajar el tiempo filtra el futuro

<PliegosTiempo :mostrar="['kfold']" :fila="38" />

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 10px">
<div>

En *K-fold* aleatorio, el día $t$ queda en prueba y $t - 1$, $t + 1$ en entrenamiento.

</div>
<div v-click>

Sus anomalías tienen correlación **0.79** con la de $t$: el modelo puede ver la respuesta de prueba **a través de sus vecinos**. Es una fuga del futuro: información que no se tendría al pronosticar.

</div>
</div>

<!--
La validación cruzada usual supone renglones independientes. Con una serie no lo son. Cuánto se nota depende del modelo: uno que pueda «ubicar» un día en el tiempo y promediar a sus vecinos aprovecha la fuga; una regresión lineal casi no puede.
-->

---

# La demostración: pronosticar a 7 días

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<FugaKFold :grupos="$clicks < 1 ? 1 : ($clicks < 2 ? 2 : 3)" />

</div>
<div>

Atributos del día $t - 7$: dos rezagos, medias de 7 y 30 días, día del año y año. **Sin fuga en la matriz.**

<v-click>

KFold aleatorio elige el **bosque**: 1.75 contra 2.23 °C. El origen móvil elige la **lineal**: 2.28 contra 2.37.

</v-click>

<v-click>

En el futuro real gana la lineal (2.05). El bosque (2.19) ni siquiera vence a la climatología, y KFold le prometía 1.75.

</v-click>

<v-click>

<div class="nota">

La fuga favorece a los modelos **más flexibles**: justo los que más necesitamos evaluar bien.

</div>

</v-click>
</div>
</div>

<!--
Con el día del año y el año, el bosque puede ubicar cada día casi exactamente en el calendario; sus hojas terminan promediando a los vecinos t±1, t±2, que conocen la anomalía del día a predecir. La regresión lineal no puede memorizar vecindarios: su estimación casi no cambia entre esquemas. El origen móvil sale algo más alto que el futuro real porque 2016–2020 incluyó años difíciles como 2020: eso es variación de verdad, no fuga.
-->

---

# Origen móvil

<PliegosTiempo :mostrar="['creciente', 'deslizante']" />

<div class="cols pequeno" style="grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 8px">
<div>

**Ventana creciente:** entrena con todo el pasado. **Deslizante:** solo con una ventana fija de días recientes, para olvidar un pasado que ya no se parece al presente.

</div>
<div>

Se decide el **horizonte** (hueco entre el último dato y la prueba: `gap`), el tamaño del bloque de prueba y el **salto** entre orígenes. En scikit-learn: `TimeSeriesSplit`.

</div>
</div>

<!--
Azul: entrenamiento; rojo: prueba. También se le llama backtesting o evaluación con origen que rueda. En la libreta evaluamos las referencias así: re-ajuste anual con ventana creciente, cada día como origen.
-->

---
layout: caso
caso: Caso · Pronósticos que no se validan
cifra: 100 de 108
cifraNota: semanas en que Google Flu Trends sobreestimó la gripe en EE. UU., de agosto de 2011 a septiembre de 2013.
fuente: D. Lazer, R. Kennedy, G. King y A. Vespignani, «The parable of Google Flu», Science 343 (2014).
---

# Google Flu Trends

Estimaba las consultas por enfermedad tipo influenza a partir de las búsquedas en Google, antes que los CDC. En febrero de 2013 pronosticaba **más del doble** de lo que reportaron los CDC.

<v-click>

Se eligieron términos entre **50 millones** para ajustar 1,152 datos: «en parte detector de influenza, en parte detector de invierno». No vio la pandemia H1N1 de 2009, que no llegó en invierno.

</v-click>

<v-click>

Un modelo con datos de los CDC **rezagados** y estacionalidad tuvo menos error (MAE de 0.311 contra 0.486). Lección: comparar contra referencias y validar hacia adelante, siempre.

</v-click>

<!--
GFT se lanzó en 2008 (Ginsberg et al., Nature, 2009). Los errores de GFT estaban autocorrelacionados y tenían estacionalidad: el error de la semana pasada predecía el de esta. Los hábitos de búsqueda y el propio buscador cambiaron: las relaciones aprendidas en el pasado dejaron de valer. Google dejó de publicar las estimaciones en 2015.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Reindexar, y separar **ciclo, tendencia y anomalías** (climatología, STL).
2. La **ACF** mide la memoria; la estacionalidad la tapa y la no estacionariedad la estira.
3. Con autocorrelación, las fórmulas que suponen independencia son **optimistas**.
4. Todo pronóstico se compara con el **ingenuo y la climatología**, por horizonte, con MAE o MASE.
5. Se valida con **origen móvil**: nunca se baraja el tiempo.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *El clima de Hermosillo: tendencia, estacionalidad y cómo validar un pronóstico* (Unidad 7, ejemplo 1).

**Laboratorio 7:** validación temporal y modelos autorregresivos desde cero.

**Lectura:** R. J. Hyndman y G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed., capítulos 2, 3 y 5.

</div>
</div>
</div>
