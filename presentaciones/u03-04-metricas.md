---
theme: none
title: Métricas
unidad: Unidad 3
layout: portada
subtitulo: Qué medir cuando equivocarse no cuesta lo mismo en los dos sentidos
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

# Métricas de evaluación

---
layout: pregunta
---

# Un modelo para detectar quién no pagará su tarjeta acierta en 78 % de los casos. ¿Es bueno?

<v-click>

Depende: el modelo que dice «todos pagan» también acierta en 78 %, porque solo 22 % de los clientes incumple. Y no sirve para nada.

</v-click>

---

# El problema de hoy: riesgo de crédito

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div class="grande">

- 30,000 clientes de tarjeta de crédito de un banco de Taiwán, de abril a septiembre de 2005, en plena crisis de deuda de tarjetas.
- Límite de crédito, edad, sexo, escolaridad y seis meses de facturas, pagos y atrasos.
- **Problema:** estimar la probabilidad de que el cliente **no pague** el mes siguiente (22 %).

</div>
<div>

Dos modelos: **regresión logística** y un **bosque aleatorio** (muchos árboles de decisión que votan). Entrenados con 22,500 clientes y evaluados en 7,500.

<p class="pequeno tenue" style="margin-top: 26px">I.-C. Yeh y C. Lien, <em>Expert Systems with Applications</em> 36 (2009). UCI, CC BY 4.0.</p>

</div>
</div>

---
layout: seccion
numero: 1
---

# Del puntaje a la decisión

La matriz de confusión y el umbral.

---

# Un puntaje, un umbral, cuatro casillas

<UmbralInteractivo />

<!--
Mover el umbral desde 0.5 hacia abajo. Con 0.5 la exactitud es alta pero se detecta poco más de un tercio de los incumplidos. Al bajarlo, la exhaustividad sube y la precisión baja. Ningún umbral es «el correcto» sin hablar de costos.
-->

---

# Las métricas, con nombre y apellido

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

$$
\text{exactitud} = \frac{VP + VN}{M} \qquad \text{TFP} = \frac{FP}{FP + VN}
$$

$$
\text{precisión} = \frac{VP}{VP + FP} \qquad \text{TVP} = \frac{VP}{VP + FN}
$$

$$
F_1 = \frac{2 \cdot \text{precisión} \cdot \text{TVP}}{\text{precisión} + \text{TVP}}
$$

<div class="pequeno tenue">

$VP$, $VN$: verdaderos positivos y negativos; $FP$, $FN$: falsos positivos y negativos; $M$: número de clientes. TVP y TFP: tasas de verdaderos y de falsos positivos.

</div>

<div v-click="4">

La precisión depende de la **prevalencia** $\pi$ (la fracción de incumplidos); la TVP y la TFP no:

$$
\text{precisión} = \frac{\pi\,\text{TVP}}{\pi\,\text{TVP} + (1 - \pi)\,\text{TFP}}
$$

</div>
</div>
<div>

<v-clicks>

- **Exhaustividad (TVP):** de los que incumplieron, ¿a cuántos detectamos?
- **Precisión:** de los que marcamos, ¿cuántos sí incumplieron?
- **TFP:** de los buenos clientes, ¿a cuántos molestamos?

</v-clicks>

</div>
</div>

---
layout: seccion
numero: 2
---

# Todos los umbrales a la vez

Curvas ROC y de precisión–exhaustividad.

---

# Curvas ROC y de precisión–exhaustividad

<CurvasROCPR />

<div class="cols" style="gap: 40px">
<div class="pequeno">

Cada punto de la ROC es un umbral. El bosque domina a la logística en casi todo el rango.

</div>
<div class="pequeno" v-click>

Con clases desbalanceadas, la curva PR es más informativa: su referencia es la prevalencia, no 0.5.

</div>
</div>

---

# ¿Qué significa el área bajo la curva ROC?

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="proposicion">

Si $S_1$ es el puntaje de un incumplido al azar y $S_0$ el de un cumplido al azar, independientes,

$$
\text{AUC} = \Prob\big[S_1 > S_0\big] + \tfrac12\,\Prob\big[S_1 = S_0\big] .
$$

</Caja>

<v-click>

Para el bosque, $0.786$: en 79 de cada 100 parejas (incumplido, cumplido) el incumplido recibe el puntaje más alto.

</v-click>
</div>
<div v-click>

**Idea de la prueba.** Con umbral $t$, $\text{TVP}(t) = \Prob[S_1 \ge t]$ y $\text{TFP}(t) = \Prob[S_0 \ge t]$. Al recorrer $t$ de mayor a menor, $d\,\text{TFP} = f_{S_0}(t)\,dt$, con $f_{S_0}$ la densidad de $S_0$, y

$$
\text{AUC} = \int \text{TVP}\; d\,\text{TFP} = \int \Prob[S_1 \ge t]\, f_{S_0}(t)\, dt
$$

$$
= \Prob[S_1 \ge S_0] .
$$

<div class="pequeno tenue">

Es el estadístico de Mann–Whitney normalizado. En la libreta se verifica con los 9.7 millones de parejas; en el laboratorio se calcula con rangos en $O(M \log M)$, con $M$ el número de clientes.

</div>
</div>
</div>

---
layout: seccion
numero: 3
---

# Elegir el umbral

Lo decide el costo de cada error, no el modelo.

---

# El umbral óptimo de Bayes

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Un cliente con probabilidad de incumplir $p$. Marcarlo cuesta $c_{FP}$ si sí paga; dejarlo pasar cuesta $c_{FN}$ si incumple:

$$
\E[\text{costo} \mid \text{marcar}] = (1 - p)\,c_{FP}
$$

$$
\E[\text{costo} \mid \text{no marcar}] = p\,c_{FN}
$$

</div>
<div>

<div v-click>

Conviene marcar cuando $p\,c_{FN} > (1-p)\,c_{FP}$, es decir

$$
p \;>\; t^* = \frac{c_{FP}}{c_{FP} + c_{FN}} .
$$

</div>

<div v-click>

Con $c_{FP} = 1$ (molestar a un buen cliente) y $c_{FN} = 5$ (lo que se pierde de una deuda impagada), $t^* = 1/6$. Costos ilustrativos, no del banco.

</div>
</div>
</div>

<div v-click class="nota">

**Cuidado:** la regla supone que $p$ **es** una probabilidad, es decir, que el modelo está **calibrado**. Y $t$ se elige con datos de entrenamiento o validación, nunca con los de prueba.

</div>

---

# El costo según el umbral

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<CostoUmbral />

</div>
<div>

En los 7,500 clientes de prueba.

<v-click>

Con el bosque, pasar de $t = 0.5$ a $t^* = 1/6$ baja el costo promedio de 0.75 a 0.55 por cliente: **26 % menos**, con el mismo modelo.

</v-click>

<v-click>

El mínimo observado (0.54 en $t \approx 0.19$) está cerca de $t^*$; mirarlo en la prueba para elegir sería hacer trampa.

</v-click>
</div>
</div>

---

# Calibración: ¿el 30 % significa 30 %?

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<Confiabilidad />

</div>
<div>

Agrupamos a los clientes por decil de probabilidad predicha y comparamos con la fracción que **de verdad** incumplió.

<v-click>

La logística se aleja de la diagonal en los extremos. Se puede **recalibrar** con una función creciente $g$ ajustada en datos aparte (isotónica o sigmoide): no cambia el orden, así que el AUC casi no cambia, pero corrige lo que significan los números.

</v-click>

<v-click>

<div class="pequeno">

**Puntaje de Brier**, $\frac{1}{M}\sum_{i=1}^{M} \big(p^{(i)} - y^{(i)}\big)^2$, con $p^{(i)}$ la probabilidad predicha y $y^{(i)} \in \{0, 1\}$ lo que pasó: logística 0.144, bosque 0.133, bosque calibrado 0.132.

</div>

</v-click>
</div>
</div>

---
layout: seccion
numero: 4
---

# Equidad

Cuando las métricas no pueden ponerse de acuerdo.

---

# Las tasas por sexo

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<EquidadSexo />

</div>
<div>

Bosque calibrado, umbral $t^* = 1/6$.

<v-click>

Los hombres incumplen más en estos datos (25 % contra 20 %). Detectamos a los incumplidos casi por igual (TVP de 73 % y 72 %), pero marcamos a **más hombres cumplidos** (TFP de 33 % contra 30 %).

</v-click>

<v-click>

¿Bajamos el umbral para las mujeres o lo subimos para los hombres? Igualar la TFP descompone la TVP y la precisión.

</v-click>
</div>
</div>

---

# Un resultado de imposibilidad

<div class="cols" style="gap: 48px">
<div>

Para cada grupo, de la definición de precisión (VPP) se obtiene

$$
\text{TFP} = \frac{\pi}{1 - \pi} \cdot \frac{1 - \text{VPP}}{\text{VPP}} \cdot \text{TVP} .
$$

<v-click>

Si las prevalencias $\pi$ son **distintas**, no se pueden igualar a la vez la VPP, la TVP y la TFP entre grupos, salvo con un clasificador perfecto.

</v-click>

<p class="pequeno tenue" style="margin-top: 22px">A. Chouldechova, <em>Big Data</em> 5 (2017); J. Kleinberg, S. Mullainathan y M. Raghavan, ITCS (2017). Es el fondo matemático del caso COMPAS, un puntaje de riesgo de reincidencia usado en tribunales de EE. UU.</p>

</div>
<div v-click>

<Caja tipo="idea">

Elegir qué igualar **no es una pregunta técnica**: es decidir qué error es más injusto y para quién. La matemática solo dice que no se puede todo.

</Caja>

<div class="pequeno" style="margin-top: 16px">

Quitar la variable `sexo` tampoco resuelve el problema: otras variables (límite, escolaridad) cargan parte de esa información: funcionan como *proxies* del sexo.

</div>
</div>
</div>

---

# ¿Qué métrica responde qué pregunta?

<div style="margin-top: 6px">

| pregunta | métrica | ¿depende del umbral? | ¿depende de la prevalencia? |
|---|---|---|---|
| ¿ordena bien a los clientes por riesgo? | AUC | no | no |
| ¿encuentra a los que buscamos? | exhaustividad (TVP) | sí | no |
| ¿cuántas alarmas son ciertas? | precisión (VPP) | sí | **sí** |
| ¿cuánto nos cuesta? | costo esperado | sí | sí |
| ¿podemos creerle a sus probabilidades? | calibración, Brier | no | sí |

</div>

<div v-click class="nota">

**En regresión:** la RECM castiga los errores grandes; el error absoluto medio es más robusto; trabajar en logaritmo (como con las casas) mide errores relativos.

</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Con clases desbalanceadas, la exactitud engaña: comparar siempre con la **línea base**.
2. El modelo da un **puntaje**; la decisión necesita un **umbral**.
3. **AUC** = probabilidad de ordenar bien una pareja.
4. Umbral de Bayes $t^* = c_{FP}/(c_{FP} + c_{FN})$, con probabilidades **calibradas**.
5. Con prevalencias distintas, no todas las nociones de equidad se pueden cumplir a la vez.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Métricas de clasificación: el riesgo de crédito* (Unidad 3, ejemplo 4).

**Laboratorio 3:** validación cruzada, ROC, AUC y umbral por costos, desde cero.

**Lectura:** T. Fawcett, «An introduction to ROC analysis», *Pattern Recognition Letters* 27 (2006).

</div>
</div>
</div>
