---
theme: none
title: Pronosticar con aprendizaje automático
unidad: Unidad 7
layout: portada
subtitulo: "Rezagos, boosting y redes: la demanda de bicicletas hora por hora"
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
  Reconocimiento de Patrones · Unidad 7 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Pronosticar con aprendizaje automático

---
layout: pregunta
---

# Es domingo, 8 de la noche. ¿Cuántas bicicletas se rentarán mañana a las 8 de la mañana en Washington?

<v-click>

«Las mismas que el lunes pasado a esa hora» es una buena primera respuesta. Con ella nos equivocamos, en promedio, por **76 bicicletas por hora**. Hoy la bajaremos a 54 con un día de anticipación y a 27 con una hora, usando lo que ya sabemos: regresión, *boosting* y redes.

</v-click>

<!--
Dejar que el grupo proponga. Lo natural: "lo de ayer a esa hora" o "lo del lunes pasado".
Las dos son los pronósticos estacionales ingenuos; los dos dan MAE ≈ 76 en el último trimestre de 2012.
La idea de la clase: convertir el pronóstico en un problema de regresión supervisada.
-->

---
layout: caso
caso: Caso · Competencias de pronóstico
cifra: 22.4 %
cifraNota: menos error que el mejor método estadístico de referencia logró el ganador de la competencia M5 (2020), con promedios de modelos LightGBM.
fuente: 'S. Makridakis, E. Spiliotis y V. Assimakopoulos, «M5 accuracy competition: Results, findings, and conclusions», International Journal of Forecasting 38 (2022).'
---

# Las competencias M

Desde 1982, Spyros Makridakis organiza competencias abiertas de pronóstico. Por décadas, los métodos estadísticos simples y sus **combinaciones** fueron tan buenos como los sofisticados.

<v-click>

**M4 (2018), 100,000 series:** ganó un **híbrido** de suavizamiento exponencial y redes recurrentes; el segundo lugar usó *boosting* para **combinar** métodos estadísticos.

</v-click>

<v-click>

**M5 (2020), 42,840 series de ventas diarias de Walmart, 5,507 equipos de 101 países:** casi todos los mejores usaron **LightGBM con atributos rezagados y calendario**. El ganador, un estudiante de licenciatura, promedió modelos recursivos y no recursivos.

</v-click>

<!--
Cifras verificadas en el artículo (versión de preprint, oct. 2020): 42,840 series jerárquicas; 7,092 participantes en 5,507 equipos de 101 países; el ganador (YJ_STU, YeonJun Im, Corea del Sur) mejoró 22.4 % la WRMSSE del mejor benchmark (suavizamiento exponencial agregado de abajo hacia arriba, "ES bu"). Solo los 5 primeros superaron 20 %.
"Recursivos y no recursivos" es justo el tema de la sección 2.
-->

---

# Dos años de rentas, tres escalas de tiempo

<PronDosAnios v-if="$clicks < 1" vista="anios" />
<PronDosAnios v-else vista="semanas" />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

Capital Bikeshare, Washington, D. C., 2011–2012: 17,544 horas. **Tendencia** (el sistema crece) y **estación del año**.

</div>
<div v-click class="pequeno">

Por hora: **ciclo diario** con dos picos los días laborales, una loma el fin de semana: **ciclo semanal**.

</div>
</div>

<!--
Primer clic: dos semanas de septiembre de 2012 (fin de semana sombreado).
Entrenamos con ene 2011 – sep 2012; probamos en oct–dic 2012 (zona azul). Ojo con Sandy: volveremos al final.
-->

---

# Antes de modelar: ¿están todas las horas?

<div class="cols" style="grid-template-columns: 620px 1fr; gap: 44px">
<div>

<PronFaltantes />

</div>
<div>

Faltan **165** de 17,544 horas, casi todas entre las 2 y las 5 de la mañana. Y ninguna hora registrada tiene cero rentas: el mínimo es 1.

<v-click>

Una hora sin rentas **no aparece** en la tabla: el faltante es un cero. Los días con más huecos son una nevada (enero de 2011), los huracanes Irene (agosto de 2011) y Sandy (octubre de 2012).

</v-click>

<v-click>

<div class="nota">

**Decisión:** reindexar a todas las horas, rellenar con 0 y guardar una marca `faltante`. Sin reindexar, «el rezago 24» no sería «ayer a esta hora».

</div>

</v-click>
</div>
</div>

<!--
Énfasis: en series de tiempo, los rezagos se calculan por posición. Si faltan filas, y[t-24] ya no es ayer a la misma hora: fuga silenciosa de errores.
-->

---

# Del pronóstico a una regresión

<PronVentana />

$$
\mathbf{x}_t^{(h)} = \big(y_t, \dots, y_{t-23},\ y_{t+h-24},\ y_{t+h-168},\ \bar y_{24},\ \bar y_{168},\ \text{calendario de } t+h\big) \ \longmapsto\ y_{t+h}
$$

<!--
Mover el deslizador: el objetivo (rojo) avanza; los dos dorados (misma hora ayer y hace una semana) lo acompañan.
Cada origen del pasado da un renglón: unos 15,000 ejemplos de entrenamiento.
Ojo: y(t+h-24) solo es legítimo si h <= 24.
-->

---

# Tres reglas para no pronosticar con el futuro

<div class="cols" style="gap: 56px">
<div>

<v-clicks>

- Cada atributo se calcula **solo con lo conocido en el origen** $t$. Rezagos como $y_{t+h-24}$ valen si $h \le 24$.
- El **calendario** de $t+h$ sí se conoce: hora, día de la semana, feriado.
- La partición es **por tiempo**, y ninguna **respuesta** de entrenamiento cae en la prueba.

</v-clicks>

</div>
<div>

<v-click>

| | orígenes | periodo |
|---|---|---|
| entrenamiento | 14,977 | ene 2011 – sep 2012 |
| prueba | 2,185 | oct – dic 2012 |

<div class="pequeno tenue">

Un pronóstico de 1 a 24 horas desde **cada** hora de prueba, con el modelo entrenado una vez: **origen móvil**. Así medimos el error por horizonte.

</div>

</v-click>
</div>
</div>

<!--
La tercera regla es sutil: el último origen de entrenamiento es el 29 de sept. a las 23 h, para que su respuesta a 24 h no caiga en octubre.
-->

---

# Elegir hiperparámetros: el pasado valida al futuro

<PronOrigenMovil />

<div class="cols" style="gap: 40px; margin-top: 4px">
<div class="pequeno">

**Origen móvil con ventana creciente:** entrena con todo lo anterior y valida en el mes siguiente. Nunca al revés.

</div>
<div v-click class="pequeno">

Se eligen **300 árboles**. La diferencia **entre meses** (de 39 a 58) es mucho mayor que entre 300 y 800 árboles.

</div>
</div>

<!--
La validación cruzada aleatoria mezclaría horas vecinas, casi idénticas, entre entrenamiento y validación: optimista (libreta 1 de la unidad).
-->

---

# Los resultados: un modelo por horizonte

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

| pronóstico | MAE h = 1 | MAE h = 24 | MAE 1–24 | RECM 1–24 |
|---|---|---|---|---|
| ingenuo: ayer | 77.5 | 75.8 | 76.8 | 128.1 |
| ingenuo: semana pasada | 75.9 | 76.3 | 76.1 | 130.1 |
| lineal directa | 35.6 | 57.8 | 55.9 | 89.3 |
| LightGBM directa | **27.1** | **54.4** | **50.3** | **83.1** |

</div>
<div>

**Estrategia directa:** 24 modelos, $\hat y_{t+h} = g_h(\mathbf{x}_t^{(h)})$.

<v-click>

En prueba se rentan **215** bicicletas por hora en promedio. A una hora, el error es la mitad que a un día: lo último que se vio es la mejor pista.

</v-click>

<v-click>

<div class="pequeno tenue">

La lineal recibe la hora × tipo de día en *one-hot* (como el M4 de la Unidad 4). Pronósticos recortados en 0.

</div>

</v-click>
</div>
</div>

---

# ¿Qué mira LightGBM?

<PronImportancia />

<div class="cols" style="gap: 40px; margin-top: 10px">
<div class="pequeno">

A una hora, **el presente** $y_t$ y la misma hora de la semana pasada.

</div>
<div v-click class="pequeno">

A un día, dos terceras partes vienen de **la misma hora hace una semana**: un estacional ingenuo «inteligente» que se corrige con lo que ve hoy.

</div>
</div>

<!--
Fracción de la ganancia total de las divisiones (Unidad 5). Azul: rezagos; dorado: calendario.
La hora aporta poco sola porque el rezago semanal ya la lleva implícita: el lunes pasado a las 8 ya "sabe" que es lunes a las 8.
-->

---

# Una fuga tentadora: el clima

<div class="cols" style="gap: 48px">
<div>

La lluvia reduce las rentas un 39 % (Unidad 4). Agreguemos la temperatura y el clima **observados** en la hora objetivo $t+h$:

| h | sin clima | con clima | mejora |
|---|---|---|---|
| 1 | 27.1 | 25.9 | 4.5 % |
| 6 | 48.5 | 42.1 | 13.3 % |
| 24 | 54.4 | 46.2 | **14.9 %** |

</div>
<div>

<v-click>

<Caja tipo="cuidado">

A las 8 de la noche **no conocemos** el clima de mañana a las 8 de la noche. Ese atributo no existirá cuando haya que decidir: es **fuga del futuro**.

</Caja>

</v-click>

<v-click>

Formas honestas: solo **rezagos** del clima; el **pronóstico meteorológico** de ese momento (y entrenar con pronósticos históricos, no con lo observado); o **escenarios**. El 14.9 % es una **cota superior**.

</v-click>
</div>
</div>

<!--
Mismo tipo de error que la duración de la llamada en el ejemplo del banco (libreta de boosting, Unidad 5). En validación brilla; en producción, el atributo no está.
-->

---
layout: seccion
numero: 2
---

# Pronosticar varios pasos

Recursiva, directa o multisalida.

---

# Tres estrategias

<div class="cols" style="grid-template-columns: 1fr 1fr 1fr; gap: 36px">
<div>

### Recursiva

Un modelo a un paso, aplicado 24 veces, con cada pronóstico **metido en la ventana** como si fuera dato:

$$
\hat y_{t+2} = g(\hat y_{t+1}, y_t, \dots)
$$

<div class="pequeno tenue">

Es lo que hace ARIMA. Los errores **se acumulan**.

</div>
</div>
<div v-click>

### Directa

Un modelo por horizonte:

$$
\hat y_{t+h} = g_h\big(\mathbf{x}_t^{(h)}\big)
$$

<div class="pequeno tenue">

No acumula errores, pero son 24 modelos que no comparten nada.

</div>
</div>
<div v-click>

### Multisalida

Un modelo, 24 salidas:

$$
(\hat y_{t+1}, \dots, \hat y_{t+24}) = g(\mathbf{x}_t)
$$

<div class="pequeno tenue">

Natural en una red: la última capa tiene 24 neuronas.

</div>
</div>
</div>

<div v-click class="nota">

La red: perceptrón multicapa con la última semana en escala $\log(1+y)$, la hora y el día del origen y los días laborales de las 24 horas objetivo (223 entradas, dos capas de 256), entrenado con PyTorch en segundos.

</div>

<!--
¿Por qué acumula la recursiva? El modelo se entrenó con rezagos verdaderos y en prueba recibe rezagos pronosticados, con error y más suaves que los datos (cambio de distribución en las entradas).
-->

---

# Un pronóstico de 24 horas, visto de cerca

<div class="cols" style="grid-template-columns: 760px 1fr; gap: 36px">
<div>

<PronTrayectoria :mostrar="$clicks < 1 ? 1 : $clicks < 2 ? 2 : 3" />

</div>
<div>

Un martes de noviembre a las 6 de la mañana. El número junto a cada modelo es su MAE en esas 24 horas.

<v-click>

La **recursiva** arranca igual que la directa, pero ya no corrige: en un día flojo arrastra sus propios pronósticos altos.

</v-click>

<v-click>

<div class="pequeno tenue">

Es un solo origen: el veredicto lo da el promedio sobre los 2,185.

</div>

</v-click>
</div>
</div>

<!--
27 de noviembre de 2012, 6 h. Observado: un día con menos rentas que lo normal.
Directa 42.9, recursiva 60.0, red 30.6. En otros orígenes el orden cambia; por eso la siguiente lámina.
-->

---

# El error contra el horizonte

<div class="cols" style="grid-template-columns: 780px 1fr; gap: 30px">
<div>

<PronHorizonte :mostrar="$clicks + 2" />

</div>
<div>

<v-clicks>

- Todos empeoran en las primeras 6 horas y luego se estabilizan.
- La **recursiva** coincide con la directa en $h = 1$ y se separa: a 24 h tiene MAE 58.6 contra 54.4.
- La **red multisalida**, sin atributos diseñados, queda cerca (52.4 contra 50.3 de 1 a 24 h) y su RECM es la menor (82.9).

</v-clicks>

</div>
</div>

<!--
Clic 1: recursiva. Clic 2: red. Clic 3: lineal.
La red pierde a 1 hora (36.6): y_t es una de 223 entradas, mientras que a LightGBM se la dimos como atributo.
El ganador de la M5 promedió modelos recursivos y no recursivos: combinar ayuda. Ejercicio 3 de la libreta.
-->

---
layout: seccion
numero: 3
---

# Intervalos de predicción

Aprender un cuantil cambiando la pérdida.

---

# La pérdida cuantil

<div class="cols" style="gap: 56px">
<div>

Para decidir cuántas bicicletas llevar a una estación no basta $\hat y$: hace falta un intervalo, por ejemplo entre los cuantiles 0.1 y 0.9.

<Caja tipo="definicion" nombre="pérdida cuantil (pinball)">

$$
\rho_\tau(u) = u\,\big(\tau - \ind{u < 0}\big), \qquad u = y - q .
$$

Quedarse corto cuesta $\tau$ por unidad; pasarse, $1 - \tau$.

</Caja>

</div>
<div>

<v-click>

<Caja tipo="proposicion">

Si $Y$ tiene distribución continua $F$, el minimizador $q^*$ de $L(q) = \E[\rho_\tau(Y - q)]$ cumple $F(q^*) = \tau$: es el **cuantil** $\tau$ de $Y$.

</Caja>

</v-click>

<v-click>

$$
L(q) = \tau\!\int_q^\infty (y - q)\,dF + (1 - \tau)\!\int_{-\infty}^q (q - y)\,dF
$$

$$
L'(q) = -\tau\,(1 - F(q)) + (1 - \tau)\,F(q) = F(q) - \tau .
$$

$L'$ es creciente: $L$ es convexa y su mínimo está donde $F(q) = \tau$.

</v-click>
</div>
</div>

<!--
Con τ = 0.5, ρ es la mitad del error absoluto: la mediana minimiza el MAE.
Con datos: la pendiente de la pérdida empírica es (fracción de datos debajo de q) − τ.
-->

---

# Verificación con las rentas de las 8 de la mañana

<PronPinball />

<!--
Mover τ: el mínimo de la curva negra cae en el cuantil τ de los 438 días laborales de entrenamiento (histograma).
Lo rojo del histograma es la fracción τ de los datos.
Después: LightGBM con objective="quantile", alpha=τ hace lo mismo pero condicional a los atributos.
-->

---

# LightGBM con `objective="quantile"`: ¿cubre el 80 %?

<PronBanda cual="noviembre" />

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px; margin-top: 4px">
<div class="pequeno">

Semana del 12 de noviembre de 2012, pronosticada con un día de anticipación: banda del 10 % al 90 %, pronóstico (azul) y lo observado (negro). La banda se abre en las horas pico.

</div>
<div v-click class="pequeno">

Cobertura en prueba: **70.3 %** a 1 hora, **64.1 %** a 24 horas. Debajo del cuantil 0.1 cae el **21 %**: el modelo sobreestima la caída hacia el invierno.

</div>
</div>

<!--
Tabla completa (h: cobertura, debajo, arriba, ancho): 1: 70.3, 17.1, 12.7, 85 · 6: 68.3, 18.4, 13.4, 131 · 12: 66.9, 19.5, 13.6, 132 · 24: 64.1, 21.4, 14.5, 131.
Los cuantiles aprendidos describen la incertidumbre del pasado; el entrenamiento tiene un solo otoño, el de 2011, con un sistema más chico.
La cobertura se mide en datos posteriores; nunca se supone. Ejercicio 4: intervalos conformes.
-->

---

# El día que todos los modelos fallaron

<PronBanda cual="sandy" />

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 36px; margin-top: 4px">
<div class="pequeno">

29 y 30 de octubre de 2012: cerraron las oficinas federales y el metro. El modelo pronosticó un lunes normal, con su intervalo «del 80 %».

</div>
<div v-click class="pequeno">

Esas 48 horas son el 2 % de los pronósticos y el **9 %** del error a 24 h (MAE 54.4; sin ellas, 50.6). Nada en los rezagos anunciaba el huracán.

</div>
</div>

<!--
Lecciones: (1) un intervalo cuantifica la incertidumbre que se parece a la del pasado; (2) antes de reportar un error, ver de dónde viene, y decir si se excluyen eventos; (3) un sistema en producción necesita supervisión humana y una forma de incorporar información externa (alertas).
-->

---

# ¿Qué modelo para qué problema?

| situación | suele funcionar |
|---|---|
| Pocas series largas y suaves, estructura clara (CO₂, temperatura) | ARIMA, suavizamiento exponencial |
| Muchas series relacionadas, calendario y variables exógenas (ventas, demanda) | *Boosting* con atributos rezagados, un modelo **global** |
| Mucha historia, mucha estructura, muchos datos | Redes: MLP, recurrentes, convolucionales, *transformers* |

<div v-click class="nota">

En todos los casos: compararse con un **ingenuo estacional**, validar **hacia adelante**, ningún atributo del futuro y medir la **cobertura** en datos nuevos.

</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Un pronóstico es una regresión con **rezagos**, medias móviles y calendario, si cada atributo se conoce en el origen.
2. Se valida con **origen móvil**: el pasado predice al futuro.
3. Variables exógenas del futuro (el clima observado) son **fuga**: hay que usar sus pronósticos.
4. **Recursiva** acumula errores; **directa** y **multisalida**, no. Combinar ayuda.
5. La **pérdida cuantil** da intervalos; su cobertura se mide, no se supone.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Pronosticar la demanda de bicicletas hora por hora* (Unidad 7, ejemplo 3).

**Laboratorio 7:** validación temporal y modelos autorregresivos desde cero.

**Lectura:** R. J. Hyndman y G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed., capítulos 5 y 12; G. Bontempi, S. Ben Taieb e Y.-A. Le Borgne, «Machine learning strategies for time series forecasting» (2013).

</div>
</div>
</div>
