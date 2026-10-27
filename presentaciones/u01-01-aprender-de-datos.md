---
theme: none
title: ¿Qué significa aprender de datos?
unidad: Unidad 1
layout: portada
subtitulo: Paradigmas, riesgo empírico, generalización y el flujo de trabajo de un proyecto
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
  Reconocimiento de Patrones · Unidad 1 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# ¿Qué significa aprender de datos?

---

# Plan de la sesión

<div class="cols cols-6-4 cols-centro" style="flex:1">
<div>

1. El problema de aprender: qué es y qué no es aprendizaje automático
2. Una formalización mínima: datos, hipótesis, pérdida y riesgo
3. Aprender no es memorizar: la generalización
4. El flujo de trabajo de un proyecto
5. Responsabilidad: datos, sesgos y decisiones

</div>
<div class="lema">
La pregunta central del curso no es <em>cómo ajustar</em> un modelo, sino <em>cuándo podemos confiar</em> en lo que aprendió.
</div>
</div>

---
layout: seccion
numero: 1
---

# El problema de aprender

¿Cuándo tiene sentido que un programa aprenda en lugar de que lo programemos?

---

# Programar *vs.* aprender

<DiagramaProgramar />

<div class="abajo">
<Caja tipo="idea">

Aprendemos de datos cuando **no sabemos escribir las reglas**, pero sí tenemos **ejemplos** del comportamiento que queremos.

</Caja>
</div>

---

# Una definición operativa

<Caja tipo="definicion" nombre="Mitchell (1997)">

Un programa **aprende** de la experiencia $E$, respecto a una clase de tareas $T$ y una medida de desempeño $P$, si su desempeño en $T$, medido por $P$, **mejora con la experiencia** $E$.

</Caja>

<div v-click>

| | Tarea $T$ | Experiencia $E$ | Desempeño $P$ |
|---|---|---|---|
| Filtro de *spam* | clasificar correos | correos etiquetados | falsos positivos y negativos |
| Presas de Sonora | pronosticar el almacenamiento | series históricas | error absoluto medio |
| Segmentar clientes | agrupar perfiles | compras sin etiquetar | cohesión de los grupos |

</div>

---

# Tres condiciones para que tenga sentido aprender

<div class="cols cols-6-4">
<div>

1. **Existe un patrón.** Hay alguna relación entre entradas y salidas, aunque sea ruidosa.
2. **No lo sabemos describir analíticamente.** Si tuviéramos la fórmula, la usaríamos.
3. **Tenemos datos** que reflejan ese patrón y que representan el uso futuro.

</div>
<div v-click>
<Caja tipo="cuidado">

Si falla la tercera condición, el modelo puede ser excelente en el laboratorio y **fallar en producción**.

<span class="pequeno tenue">Lo retomamos en la Unidad 3 (generalización) y en la Unidad 8 (*drift*).</span>

</Caja>
</div>
</div>

---

# Paradigmas de aprendizaje

| Paradigma | Observamos | Buscamos | En el curso |
|---|---|---|---|
| Supervisado | pares $(\vx_i, y_i)$ | $h(\vx) \approx y$ | Unidades 3 a 7 |
| No supervisado | solo $\vx_i$ | estructura de $P(\vx)$ | Unidad 2 |
| Autosupervisado | $\vx_i$, con etiquetas sacadas de los propios datos | representaciones útiles | Unidad 6 |
| Por refuerzo | estados, acciones y recompensas | una política $\pi(a \mid s)$ | fuera del alcance |

<div class="nota">

Este curso se concentra en el aprendizaje supervisado, y usa el no supervisado para explorar y representar datos.

</div>

---
layout: seccion
numero: 2
---

# Una formalización mínima

El vocabulario que usaremos todo el semestre.

---

# Los ingredientes del aprendizaje supervisado

<div style="margin: 6px 0 34px"><DiagramaIngredientes /></div>

<div class="cols" style="gap: 40px">
<div>

- Nunca conocemos $f$ ni $P$: **solo vemos la muestra** $\Dat$, obtenida i.i.d. de $P$.
- El algoritmo elige $g \in \Hyp$ usando únicamente $\Dat$.

</div>
<div>

- $\X$: espacio de entradas, típicamente $\R^d$.
- $\Y$: salidas; $\R$ en regresión, $\{1,\dots,K\}$ en clasificación.

</div>
</div>

---

# Pérdida y riesgo

<div class="cols cols-estirar">
<Caja tipo="definicion" nombre="riesgo (error fuera de muestra)">

Para una pérdida $\ell : \Y \times \Y \to \R_{\ge 0}$,

$$
\Eout(h) = \E_{(\vx, y) \sim P}\big[\, \ell(h(\vx), y) \,\big]
$$

</Caja>
<Caja tipo="definicion" nombre="riesgo empírico (error en muestra)">

Sobre los $n$ ejemplos de $\Dat$,

$$
\Ein(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(\vx_i), y_i)
$$

</Caja>
</div>

<div v-click>
<Caja tipo="algoritmo" nombre="minimización del riesgo empírico (ERM)">

Elegir $\; g = \argmin_{h \in \Hyp} \Ein(h) \;$ y **esperar** que $\Eout(g) \approx \Ein(g)$.

</Caja>
</div>

<div v-click class="nota">

Todo el curso gira alrededor de esa palabra: **esperar**. ¿Cuándo está justificado?

</div>

---

# Funciones de pérdida usuales

<div class="cols cols-4-6 cols-centro" style="flex:1">
<div>

- **Cuadrática** (regresión)<br>$\ell(\hat y, y) = (\hat y - y)^2$
- **0–1** (clasificación)<br>$\ell(\hat y, y) = \ind{\hat y \neq y}$
- **Logística**, $y \in \{-1, 1\}$<br>$\ell = \log_2\!\big(1 + e^{-y s}\big)$
- **Bisagra** (SVM)<br>$\ell = \max(0,\, 1 - y s)$

<div class="pequeno tenue">

Margen: $y\,s$, con $s = \vw\T\vx$.

</div>

</div>
<div>
<GraficaPerdidas />
</div>
</div>

<div class="nota">

La 0–1 es la que importa, pero no es diferenciable: optimizamos **sustitutos convexos** que la acotan por arriba.

</div>

---
layout: seccion
numero: 3
---

# Aprender no es memorizar

Un experimento con 15 puntos y polinomios de distinto grado.

---

# Un experimento: ajustar polinomios

<div class="tenue" style="margin-bottom: 8px">

Quince puntos de $y = \sin(2\pi x) + \varepsilon$, con $\varepsilon \sim \mathcal N(0,\, 0.25^2)$, ajustados por mínimos cuadrados.

</div>

<PolinomiosTres />

---

# Explóralo: el grado del polinomio

<PolinomioInteractivo />

<div class="nota">

$\Ein$ siempre baja al aumentar el grado; $\Eout$ baja y después se dispara. Pasa el ratón sobre la curva de error o mueve el control.

</div>

---

# ¿Por qué no basta con minimizar $\Ein$?

<div class="cols cols-centro">
<div>
<Caja tipo="teorema" nombre="desigualdad de Hoeffding">

Para $h$ fija **antes** de ver los datos y $\ell \in [0,1]$,

$$
\Prob\big[\, |\Ein(h) - \Eout(h)| > \epsilon \,\big] \le 2\, e^{-2 \epsilon^2 n}
$$

</Caja>

<div v-click>

- Pero el algoritmo **elige** $g$ mirando los datos, entre muchas hipótesis.
- Cuanto más rico es $\Hyp$, más fácil es ajustar el **ruido**.

</div>
</div>
<div>
<CotaHoeffding />
</div>
</div>

<div v-click class="nota">

En la Unidad 3: $\;\Eout(g) \le \Ein(g) + \Omega(\Hyp, n, \delta)$. Aprender es un compromiso entre un $\Hyp$ **expresivo** y uno **restringido**.

</div>

---
layout: seccion
numero: 4
---

# El flujo de trabajo

Lo que hace creíbles nuestros números.

---

# El ciclo de vida de un modelo

<div style="margin-top: 8px"><CicloVida /></div>

<div class="cols" style="gap: 40px; margin-top: 10px">
<div class="pequeno">

**La separación va primero**, antes de cualquier transformación que aprenda algo de los datos.

</div>
<div class="pequeno">

**El conjunto de prueba se usa una sola vez**, al final. Para elegir modelos se usa validación.

</div>
</div>

---

# Fuga de información: un error frecuente

<div class="cols codigo-chico" style="gap: 28px">
<Caja tipo="cuidado" etiqueta="Incorrecto">

```python
X = StandardScaler().fit_transform(X)  # ve todo
X_tr, X_te, y_tr, y_te = train_test_split(X, y)
modelo.fit(X_tr, y_tr)
modelo.score(X_te, y_te)      # optimista
```

</Caja>
<Caja tipo="correcto" etiqueta="Correcto · un Pipeline">

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y)
modelo = make_pipeline(StandardScaler(),
                       LogisticRegression())
modelo.fit(X_tr, y_tr)        # solo ve X_tr
modelo.score(X_te, y_te)      # honesto
```

</Caja>
</div>

Todo paso que **aprende parámetros de los datos** (escalar, imputar, codificar, seleccionar atributos) se ajusta **solo con el entrenamiento**. Los `Pipeline` de scikit-learn lo garantizan.

---
layout: seccion
numero: 5
---

# Responsabilidad

Un modelo hereda las decisiones de quien lo construye.

---

# Datos, sesgos y decisiones

<div class="cols cols-6-4">
<div>

- El modelo aprende la $P(\vx, y)$ **de los datos que le damos**, con sus sesgos históricos.
- La métrica que optimizamos **es** una decisión de valores: ¿qué cuesta más, un falso positivo o un falso negativo, y **a quién** le cuesta?
- Quitar un atributo sensible no basta: puede estar **codificado en otras variables**, como el código postal o la escuela de procedencia.

</div>
<Caja tipo="pregunta">

Antes de entrenar:

- ¿Quién generó los datos?
- ¿A quién representa la muestra, y a quién no?
- ¿A quién afectan las decisiones del modelo, y cómo podría impugnarlas?

</Caja>
</div>

---

# Para cerrar

<div class="cols cols-6-4" style="flex:1">
<div>

1. Aprender es elegir $g \in \Hyp$ con $\Ein$ pequeño **y** con garantías de que $\Eout$ también lo sea.
2. El problema central del curso es la **generalización**, no el ajuste.
3. Un flujo disciplinado (separar, validar y probar una sola vez) es lo que hace creíbles los resultados.

</div>
<div>

### Lecturas

<div class="pequeno">

Abu-Mostafa, Magdon-Ismail y Lin (2012), *Learning From Data*, cap. 1.

James *et al.* (2023), *An Introduction to Statistical Learning with Python*, cap. 2.

Barocas, Hardt y Narayanan (2023), *Fairness and Machine Learning*, cap. 1.

</div>

### Libreta

<p class="pequeno">Un primer flujo de trabajo completo con scikit-learn (Unidad 1, ejemplo 1).</p>

</div>
</div>
