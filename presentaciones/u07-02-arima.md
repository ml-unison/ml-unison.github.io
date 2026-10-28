---
theme: none
title: Modelos ARIMA
unidad: Unidad 7
layout: portada
subtitulo: "La curva de Keeling: memoria, diferencias y pronósticos con incertidumbre"
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

# Modelos ARIMA

---
layout: caso
caso: Caso · Ciencia del clima
cifra: 315.7 <small>ppm</small>
cifraNota: concentración media de CO₂ en Mauna Loa en marzo de 1958, el primer mes de la serie. En mayo de 2026 fue de 432.3 ppm.
fuente: NOAA Global Monitoring Laboratory y Scripps Institution of Oceanography (P. Tans y R. Keeling); C. D. Keeling, Tellus 12 (1960).
---

# Un químico, un volcán y una pregunta

En 1958, Charles David Keeling empezó a medir el CO₂ del aire desde el observatorio de Mauna Loa, en Hawái, a 3,400 m de altura y lejos de fuentes locales. Quería saber si el CO₂ que emitimos al quemar combustibles fósiles **se queda en la atmósfera**.

<v-click>

Desde entonces se mide cada mes, sin interrupción. Es la serie de tiempo más famosa de la ciencia del clima: la **curva de Keeling**. En mayo de 2013, por primera vez, un promedio mensual pasó de 400 ppm.

</v-click>

<!--
Keeling murió en 2005; su hijo Ralph Keeling sigue la medición en Scripps, en paralelo con la NOAA. Los datos son de dominio público (se pide citar). La cifra de 400 ppm en mayo de 2013 es la del archivo de datos del curso (400.02).
-->

---

# La curva de Keeling

<CurvaKeeling v-if="$clicks < 1" :prueba="false" />
<CurvaKeeling v-else />

<div class="cols" style="gap: 40px; margin-top: 6px">
<div class="pequeno">

Tendencia que **se acelera** (0.86 ppm por año en los sesenta, 2.62 en esta década), un ciclo anual de unos 6 ppm y apenas unas décimas de ruido.

</div>
<div class="pequeno" v-click>

**Problema:** con los datos hasta diciembre de 2014 (682 meses), pronosticar cada mes de 2015 a 2026 (141 meses), **con intervalos**.

</div>
</div>

<!--
El ciclo anual: la vegetación del hemisferio norte absorbe CO₂ en primavera y verano y lo libera en otoño e invierno. El máximo es en mayo, el mínimo en septiembre u octubre.
Preguntar: ¿qué modelo de la Unidad 4 usarían? (regresión sobre tendencia + mes: la veremos al final).
-->

---
layout: pregunta
---

# Si este mes el CO₂ quedó medio ppm arriba de lo esperado, ¿qué esperas para el mes que entra?

<v-click>

Probablemente siga arriba, pero un poco menos: la sorpresa de hoy **se arrastra** y se va desvaneciendo. Esa memoria es lo que vamos a modelar: $y_t = \phi\, y_{t-1} + \varepsilon_t$.

</v-click>

---
layout: seccion
numero: 1
---

# Modelos con memoria

Autorregresivos, medias móviles y cómo reconocerlos.

---

# El modelo autorregresivo de orden 1

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

$$
y_t = c + \phi\, y_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \text{ruido blanco}(0, \sigma^2)
$$

Sustituyendo hacia atrás $k$ veces:

$$
y_t = c\textstyle\sum_{j<k}\phi^j + \phi^k y_{t-k} + \sum_{j<k}\phi^j\varepsilon_{t-j}
$$

<v-click>

Si $|\phi| < 1$, el pasado lejano se olvida ($\phi^k \to 0$): $y_t$ es una suma de choques con pesos que decaen. **Estacionario.** Con $|\phi| \ge 1$ no hay distribución estable.

</v-click>
</div>
<div>

<v-click>

**Media:** $\mu = c + \phi\mu \;\Rightarrow\; \mu = \dfrac{c}{1-\phi}$.

**Varianza:** $\varepsilon_t$ no se correlaciona con $y_{t-1}$, así que

$$
\gamma_0 = \phi^2 \gamma_0 + \sigma^2 \;\Rightarrow\; \gamma_0 = \frac{\sigma^2}{1-\phi^2} .
$$

</v-click>

<v-click>

**ACF:** multiplicando por $y_{t-k} - \mu$, $\gamma_k = \phi\,\gamma_{k-1}$, así que

$$
\rho_k = \phi^k .
$$

</v-click>

<v-click>

<div class="pequeno tenue">

50,000 pasos simulados con $\phi = 0.9$: varianza 5.29 (teoría 5.26), $\hat\rho_1 = 0.899$.

</div>

</v-click>
</div>
</div>

---

# Cinco procesos, cinco huellas

<ProcesosMemoria />

<!--
Recorrer los botones. Pedir al grupo que adivine la ACF antes de mostrarla.
Ruido blanco: con 20 rezagos, esperamos ~1 barra fuera de la banda por azar.
AR(1) con φ = 0.9: rachas largas, ACF geométrica (los puntos dorados son 0.9^k).
φ = −0.7: alterna.
MA(1): ρ1 = θ/(1+θ²) = 0.49; nunca puede pasar de 0.5 (ejercicio de la libreta).
Caminata: la ACF muestral apenas baja; esa es la señal de que hay que diferenciar.
-->

---

# Medias móviles y autocorrelación parcial

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

**MA($q$):** el choque de hoy deja ecos durante $q$ pasos,

$$
y_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} .
$$

$y_t$ y $y_{t-k}$ no comparten choques si $k > q$: **la ACF se corta en $q$**.

<v-click>

**PACF:** $\alpha_k$ es el último coeficiente de la mejor regresión de $y_t$ sobre $y_{t-1}, \dots, y_{t-k}$: la correlación con $y_{t-k}$ **descontando** los rezagos intermedios. En un AR($p$), **se corta en $p$**.

</v-click>
</div>
<div v-click>

Identificación de Box y Jenkins (1970):

| modelo | ACF | PACF |
|---|---|---|
| AR($p$) | decae | se corta en $p$ |
| MA($q$) | se corta en $q$ | decae |
| ARMA($p$, $q$) | decae | decae |
| no estacionario | casi no baja | $\alpha_1 \approx 1$ |

<div class="pequeno tenue">

Es una guía para proponer candidatos, no una regla: la decisión final la toman la verosimilitud y los residuos.

</div>
</div>
</div>

---

# Un AR($p$) es una regresión lineal sobre los rezagos

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 40px">
<div>

$$
y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t
$$

Los atributos son **la propia serie, desplazada**: una fila de $X$ por cada $t$, y mínimos cuadrados como en la Unidad 4, $\hat{\mathbf{w}} = (X\T X)^{-1} X\T \mathbf{y}$.

<div v-click class="tabla-chica">

AR(2) simulado, $T = 1{,}000$:

| | $c$ | $\phi_1$ | $\phi_2$ |
|---|---|---|---|
| verdadero | 2 | 0.75 | −0.50 |
| a mano | 2.076 | 0.731 | −0.507 |
| `AutoReg` | 2.076 | 0.731 | −0.507 |

</div>
</div>
<div>

<RezagosAR />

<div class="pequeno tenue">

Azul: atributos; rojo: respuesta; dorado: la predicción $\hat y_t$ del modelo ajustado.

</div>
</div>
</div>

<style>
.tabla-chica { font-size: 21px; }
.tabla-chica table { font-size: 20px; }
.tabla-chica td { padding: 7px 14px 7px 0; }
.tabla-chica th { padding-bottom: 8px; }
</style>

<!--
Mover el deslizador: cada posición de la ventana es una fila de la matriz de diseño.
Con errores gaussianos, mínimos cuadrados es la máxima verosimilitud condicional en las primeras p observaciones; ARIMA usa la verosimilitud exacta y da 0.730 y −0.507: casi igual.
Advertencia que conecta con la presentación anterior: no se valida con KFold aleatorio, porque las filas comparten valores.
-->

---

# ¿Cuándo es estacionario un AR($p$)?

<div class="pequeno" style="margin-bottom: 6px">

Con el operador de rezago $B y_t = y_{t-1}$, el modelo es $\phi(B)\, y_t = c + \varepsilon_t$, con $\phi(z) = 1 - \phi_1 z - \dots - \phi_p z^p$. Es estacionario si y solo si **todas las raíces de $\phi(z)$ están fuera del círculo unitario**. Para el AR(1), la raíz es $1/\phi$.

</div>

<RaicesAR2 con-acf style="margin-top: 4px" />

<div class="pequeno tenue">

AR(2) con $\phi(z) = 1 - 0.75z + 0.5z^2$: raíces de módulo $\sqrt2 = 1.41$. Al ser complejas, la ACF oscila con periodo de unos $2\pi/1.01 \approx 6$ pasos.

</div>

---
layout: seccion
numero: 2
---

# Tendencia y estacionalidad

Diferenciar, y luego modelar lo que queda: ARIMA y SARIMA.

---

# Diferenciar el CO₂

<DiferenciasCO2 />

<!--
∇ = 1 − B quita una tendencia lineal; ∇₁₂ = 1 − B¹² quita un ciclo anual fijo y es el crecimiento anual (el pico de 1998 es el gran El Niño de 1997–98).
En ∇∇₁₂y quedan dos barras: −0.29 en el rezago 1 y −0.46 en el 12. Con la tabla de Box–Jenkins: un MA(1) y un MA(1) estacional.
-->

---

# ARIMA y SARIMA

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

**ARIMA($p, d, q$):** un ARMA para la serie diferenciada $d$ veces,

$$
\phi(B)\,(1-B)^d\, y_t = \theta(B)\,\varepsilon_t .
$$

<v-click>

**SARIMA$(p,d,q)(P,D,Q)_m$:** polinomios en $B^m$ para la parte estacional, multiplicados con los de corto plazo:

$$
\phi(B)\Phi(B^m)(1-B)^d(1-B^m)^D y_t = \theta(B)\Theta(B^m)\varepsilon_t
$$

</v-click>
</div>
<div v-click>

La ACF de $\nabla\nabla_{12}\, y$ sugiere el $(0,1,1)(0,1,1)_{12}$:

$$
(1-B)(1-B^{12})\,y_t = (1+\theta B)(1+\Theta B^{12})\,\varepsilon_t
$$

Desarrollado, es una recursión que se puede programar:

$$
\begin{aligned}
y_t = {}& y_{t-1} + y_{t-12} - y_{t-13} \\
&+ \varepsilon_t + \theta\varepsilon_{t-1} + \Theta\varepsilon_{t-12} + \theta\Theta\varepsilon_{t-13}
\end{aligned}
$$

<div class="pequeno tenue">

Box y Jenkins lo usaron para los pasajeros aéreos mensuales: el **modelo de aerolíneas**.

</div>
</div>
</div>

---

# Máxima verosimilitud y AIC

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Con términos MA, $\varepsilon_{t-1}$ no se observa: ya no basta una regresión. Con errores gaussianos, la verosimilitud exacta se evalúa con el **filtro de Kalman** y se maximiza numéricamente (`SARIMAX`).

<v-click>

Para elegir órdenes, el **criterio de Akaike**:

$$
\mathrm{AIC} = -2 \ln \hat L + 2k
$$

Penaliza los parámetros, como $\Eout \approx \Ein + \text{complejidad}$ en la Unidad 3. Solo compara modelos con la **misma** diferenciación.

</v-click>
</div>
<div v-click>

36 modelos con $d = D = 1$, entrenamiento 1958–2014:

| SARIMA | $k$ | AIC |
|---|---|---|
| $(1,1,1)(0,1,1)_{12}$ | 4 | **327.2** |
| $(2,1,1)(0,1,1)_{12}$ | 5 | 327.9 |
| $(0,1,2)(0,1,1)_{12}$ | 4 | 327.9 |
| $(0,1,1)(0,1,1)_{12}$ | 3 | 328.0 |
| $(0,1,0)(0,1,0)_{12}$ | 1 | 758.7 |

<div class="pequeno">

Elegido: $\hat\phi = 0.19$, $\hat\theta = -0.54$, $\hat\Theta = -0.87$, $\hat\sigma = 0.30$ ppm.

</div>
</div>
</div>

<!--
Los primeros cuatro están a menos de 1 unidad de AIC: prácticamente empatan. Lo que todos los buenos comparten es Θ, el MA estacional; sin él, el AIC sube cientos de unidades.
Θ = −0.87 cerca de −1: (1 − B¹²) y (1 + ΘB¹²) casi se cancelan; el ciclo anual cambia, pero muy despacio.
-->

---

# ¿Quedó algo en los residuos?

<div class="cols" style="grid-template-columns: 560px 1fr; gap: 44px">
<div>

<ResiduosSARIMA />

</div>
<div>

Si el modelo capturó la estructura, los residuos deben ser **ruido blanco**.

<v-click>

**Ljung–Box** junta las primeras $K$ autocorrelaciones:

$$
Q = T(T+2)\sum_{k=1}^{K}\frac{\hat\rho_k^2}{T-k} \;\approx\; \chi^2_{K - 3} .
$$

</v-click>

<div v-click class="tabla-chica">

| $K$ | 12 | 24 | 36 |
|---|---|---|---|
| $Q$ | 6.2 | 15.7 | 30.0 |
| valor $p$ | 0.72 | 0.79 | 0.62 |

Las barras en 1 y en 12 desaparecieron: **pasa el diagnóstico**.

</div>
</div>
</div>

<style>
.tabla-chica { font-size: 22px; }
.tabla-chica table { font-size: 20px; margin-bottom: 14px; }
.tabla-chica td { padding: 7px 14px 7px 0; }
</style>

---
layout: seccion
numero: 3
---

# Pronósticos e intervalos

Cuánto sabemos del futuro, y cuánto creemos saber.

---

# La incertidumbre crece con el horizonte

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

AR(1) con media 0, iterando $h$ pasos:

$$
y_{T+h} = \phi^h y_T + \sum_{j=0}^{h-1}\phi^j \varepsilon_{T+h-j}
$$

<v-click>

El pronóstico es $\hat y_{T+h\mid T} = \phi^h y_T$; el error, los choques que aún no ocurren:

$$
\Var(\text{error}) = \sigma^2 \sum_{j=0}^{h-1}\phi^{2j} \xrightarrow[h\to\infty]{} \frac{\sigma^2}{1-\phi^2} .
$$

</v-click>

<v-click>

**Caminata aleatoria** ($\phi = 1$): $h\sigma^2$, **sin cota**. Todo ARIMA con $d \ge 1$ la hereda.

</v-click>
</div>
<div>

<VarianzaHorizonte />

<div class="pequeno tenue">

Líneas: teoría; puntos: 20,000 trayectorias simuladas.

</div>
</div>
</div>

<!--
En general, todo ARIMA es y_t = Σ ψ_j ε_{t−j} y la varianza del error a h pasos es σ² Σ_{j<h} ψ_j². El intervalo del 95 % es ŷ ± 1.96 σ √(Σ ψ_j²): supone errores gaussianos y parámetros conocidos.
-->

---

# Doce años de pronóstico contra la realidad

<div class="cols" style="grid-template-columns: 760px 1fr; gap: 36px">
<div>

<PronosticoCO2 v-if="$clicks < 1" :observado="false" />
<PronosticoCO2 v-else-if="$clicks < 3" />
<PronosticoCO2 v-else otros />

</div>
<div>

SARIMA desde diciembre de 2014, con intervalos del 80 % y 95 %.

<v-click>

Septiembre de 2026: se midieron **426.1** ppm; el pronóstico, **420.5** (95 %: 413.6 a 427.4).

</v-click>

<v-click>

El modelo extrapola **la pendiente de 2014**, y el CO₂ siguió acelerándose.

</v-click>

<v-click>

<div class="pequeno">

Otros modelos: la tendencia cuadrática acierta más a largo plazo.

</div>

</v-click>
</div>
</div>

<!--
Semiancho del intervalo del 95 %: 0.59 ppm a 1 mes, 1.27 a 12 meses, 6.90 a 141 meses.
De los 141 meses, el 94 % cae dentro del intervalo del 95 %, pero solo el 21 % dentro del de 80 %: los meses de una misma trayectoria no son pruebas independientes. Por eso la siguiente lámina evalúa desde muchos orígenes.
-->

---

# ¿Quién gana? Depende del horizonte

<div class="cols" style="grid-template-columns: 1fr 640px; gap: 36px">
<div>

**Un origen**, 141 meses (2015–2026):

| modelo | MAE | MASE |
|---|---|---|
| SARIMA | 3.12 | 2.08 |
| estacional ingenuo | 16.21 | 10.83 |
| lineal + mes | 13.12 | 8.76 |
| cuadrática + mes | **2.00** | **1.33** |

<v-click>

<div class="pequeno">

**Origen móvil**, 130 orígenes, $h = 1, \dots, 12$: el SARIMA gana en todos los horizontes, con MAE de 0.28 ppm a un mes y 0.64 a un año (MASE 0.30 en promedio).

</div>

</v-click>
</div>
<div v-click>

<OrigenMovilCO2 />

</div>
</div>

<!--
MASE: MAE dividido entre el del estacional ingenuo a un paso en entrenamiento (1.50 ppm, el crecimiento anual típico).
A corto plazo, el SARIMA aprende de los últimos meses; a muy largo plazo gana una forma global bien elegida, siempre que el mundo siga haciendo lo mismo.
-->

---

# ¿Son honestos los intervalos?

<div class="cols" style="grid-template-columns: 640px 1fr; gap: 40px">
<div>

<OrigenMovilCO2 vista="cobertura" />

</div>
<div>

En los 130 orígenes, el intervalo del 80 % contuvo el **75 %** de los valores y el del 95 %, el **90 %**.

<v-click>

- Los parámetros son los de 2014 y su propia incertidumbre no se incluye.
- La serie se aceleró: el error medio es positivo y crece con $h$.

</v-click>

<v-click>

<div class="nota">

Un intervalo de predicción vale lo que vale el supuesto de que el futuro se comporta como el pasado.

</div>

</v-click>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. AR(1): estacionario si $|\phi| < 1$, varianza $\sigma^2/(1-\phi^2)$, ACF $\phi^k$.
2. Un AR($p$) es una **regresión lineal sobre los rezagos**; ACF y PACF distinguen AR de MA.
3. **Diferenciar** quita tendencia y ciclo: ARIMA y SARIMA.
4. Máxima verosimilitud, AIC y residuos de ruido blanco (Ljung–Box).
5. La varianza del error crece con el horizonte; se valida **hacia adelante**, desde muchos orígenes.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *La curva de Keeling: pronosticar el CO₂ con ARIMA* (Unidad 7, ejemplo 2).

**Laboratorio 7:** validación temporal y modelos autorregresivos desde cero.

**Lectura:** R. J. Hyndman y G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed., capítulo 9; Box, Jenkins, Reinsel y Ljung, *Time Series Analysis*, capítulos 3, 4 y 9.

</div>
</div>
</div>

