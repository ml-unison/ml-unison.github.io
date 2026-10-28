---
theme: none
title: Árboles de decisión
unidad: Unidad 5
layout: portada
subtitulo: "Preguntas de sí o no: partición recursiva, impureza y poda"
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
  Reconocimiento de Patrones · Unidad 5 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Árboles de decisión

---
layout: pregunta
---

# Pienso en una palabra del diccionario. ¿Cuántas preguntas de sí o no necesitas para adivinarla?

<v-click>

Unas 17 bastan para 100,000 palabras, si cada pregunta parte en dos mitades lo que queda: $2^{17} = 131{,}072$. Con malas preguntas («¿es *gato*?») pueden ser 100,000. La clave es **elegir bien la siguiente pregunta**.

</v-click>

<!--
Jugar un par de rondas con el grupo: un estudiante piensa una palabra, el resto pregunta. Hacer notar que las preguntas buenas son las que parten el conjunto en partes parecidas («¿empieza con una letra antes de la M?»), y que cada pregunta depende de las respuestas anteriores. Eso es un árbol de decisión.
-->

---
layout: caso
caso: Caso · Medicina
cifra: 3 preguntas
cifraNota: de 19 variables medidas en las primeras 24 horas, bastaron tres para separar a los pacientes de alto riesgo.
fuente: L. Breiman, J. Friedman, R. Olshen y C. Stone, Classification and Regression Trees (1984), cap. 1.
---

# Pacientes de infarto en San Diego

En el centro médico de la Universidad de California en San Diego se buscaba identificar, con los datos de las primeras 24 horas, a los pacientes de infarto con alto riesgo de **no sobrevivir 30 días**. El árbol de CART:

<v-clicks>

1. ¿La presión sistólica mínima fue mayor que 91? Si **no**: alto riesgo.
2. Si sí, ¿la edad es mayor que 62.5 años? Si **no**: bajo riesgo.
3. Si sí, ¿hay taquicardia sinusal? Si **sí**: alto riesgo; si no, bajo.

</v-clicks>

<!--
Breiman, Friedman, Olshen y Stone abren su libro con este ejemplo (sección 1.1). Lo importante: un médico puede leer el árbol, aplicarlo sin computadora y discutirlo. Cada pregunta depende de la respuesta anterior: la edad solo importa si la presión no es baja. Eso ya es una interacción.
-->

---

# Los pingüinos en dos preguntas

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 40px">
<div>

Largo y profundidad del pico de 342 pingüinos de tres especies (Unidades 2 y 4).

<v-clicks>

- ¿largo ≤ 42.35 mm?
  - **sí** → ¿profundidad ≤ 15.1? 3 Gentoo; si no, 139 Adelie y 1 Chinstrap.
  - **no** → ¿profundidad ≤ 16.35? 111 Gentoo; si no, 67 Chinstrap, 12 Adelie y 9 Gentoo.
- Exactitud: **0.936**, con dos preguntas.

</v-clicks>

<div v-click class="nota">

Cada pregunta corta con una recta **paralela a un eje**. Las hojas son rectángulos con predicción constante.

</div>
</div>
<div>

<ArbolParticion prof="2" :botones="false" />

</div>
</div>

---

# Más profundidad, rectángulos más chicos

<ArbolParticion prof="1">

<div v-click class="pequeno" style="margin-top: 10px">

Sin límite, el árbol acierta en **todos** los de entrenamiento con 21 hojas, algunas para un solo pingüino. La validación cruzada se estanca desde la profundidad 4: es el sobreajuste de la Unidad 3, con forma geométrica.

</div>

</ArbolParticion>

<!--
Recorrer las profundidades con los botones: 1 (nunca predice Chinstrap), 2, 3, 4, 6, sin límite. Preguntar en qué momento el árbol deja de aprender de las especies y empieza a aprender de pingüinos individuales.
-->

---
layout: seccion
numero: 1
---

# ¿Qué pregunta hacer?

Impureza, ganancia y el algoritmo voraz.

---

# Impureza de un nodo

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

En un nodo, $p_k$ es la proporción de la clase $k$:

$$
\begin{aligned}
\text{Gini: } & G = \textstyle\sum_k p_k(1 - p_k) \\
\text{entropía: } & H = -\textstyle\sum_k p_k \log_2 p_k \\
\text{error: } & E = 1 - \max_k p_k
\end{aligned}
$$

<v-click>

Una división manda $n_I$ datos a la izquierda y $n_D$ a la derecha. Su **ganancia**:

$$
\Delta = i(\text{padre}) - \tfrac{n_I}{n}\, i(\text{izq}) - \tfrac{n_D}{n}\, i(\text{der})
$$

</v-click>
</div>
<div>

<ImpurezaCuerda />

<div class="pequeno tenue">

Dos clases: las tres valen 0 en un nodo puro y son máximas en $p = 1/2$.

</div>
</div>
</div>

---

# Dos lecturas del índice de Gini

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="proposicion" nombre="Error de un clasificador al azar">

Si a un dato del nodo le asignamos una etiqueta al azar con las proporciones del nodo, se equivoca con probabilidad

$$
\textstyle\sum_k \Prob(y = k)\,\Prob(\hat y \ne k) = \sum_k p_k (1 - p_k) = G .
$$

</Caja>

</div>
<div v-click>

<Caja tipo="proposicion" nombre="Varianza de las indicadoras">

Con $Z_k = \ind{y = k}$ para un dato al azar del nodo, $Z_k$ es de Bernoulli y $\Var(Z_k) = p_k(1 - p_k)$, así que

$$
G = \textstyle\sum_k \Var(Z_k) .
$$

</Caja>

<div class="pequeno">

Gini es una **varianza**. En regresión usaremos la varianza de $y$ en el nodo: es la misma idea.

</div>
</div>
</div>

---

# ¿Por qué no usar el error de clasificación?

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 36px">
<div>

Un nodo con 80 Adelie y 20 Chinstrap:

| división | error | Gini |
|---|---|---|
| A: (40, 20) \| (40, 0) | 0 | 0.053 |
| B: (70, 10) \| (10, 10) | 0 | 0.045 |

<div class="pequeno">

En las dos, las hijas siguen siendo de mayoría Adelie: el error no cambia. Pero A deja 40 Adelie en un nodo **puro**.

</div>

<v-click>

<div class="pequeno">

**Jensen:** las proporciones del padre son el promedio ponderado de las de las hijas. Si $i$ es **estrictamente cóncava**, el promedio de las impurezas de las hijas queda debajo: $\Delta > 0$ salvo que $p_I = p_D$. El error es lineal a trozos: $\Delta = 0$ si las hijas caen del mismo lado de $1/2$.

</div>

</v-click>
</div>
<div>

<ImpurezaCuerda v-if="$clicks < 1" />
<ImpurezaCuerda v-else cuerda />

</div>
</div>

<!--
Al hacer clic aparece la división A: círculo blanco = padre, puntos = hijas, rombo = promedio ponderado de las hijas, que está sobre la cuerda. En Gini el rombo queda debajo del padre (ganancia 0.053); en el error, la cuerda pasa justo por el padre (ganancia 0).
-->

---

# La mejor división: ordenar una vez

<BarridoUmbral>

<div class="nota">

Ordenando por $x_j$, los conteos de la izquierda son **sumas acumuladas**: todos los umbrales se evalúan en $O(n)$; con el orden, $O(n \log n)$ por atributo.

</div>

</BarridoUmbral>

<!--
Mover el deslizador desde 38 mm hacia la derecha: los Adelie pasan a la izquierda y la ganancia sube; pasado 42.35 mm empiezan a pasar Chinstrap y Gentoo y baja. Solo hay que probar los puntos medios entre valores consecutivos distintos.
-->

---

# CART: un algoritmo voraz

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="algoritmo" nombre="Partición recursiva">

1. Si el nodo es puro, tiene pocos datos o se llegó a la profundidad máxima: hoja (clase mayoritaria).
2. Para cada atributo $j$, el mejor umbral $t_j$.
3. Elegir el $(j, t_j)$ con mayor ganancia, partir y repetir en cada hija.

</Caja>

</div>
<div>

<v-clicks>

- **Costo:** $O(d\,N \log N)$ por nivel, porque los nodos de un mismo nivel se reparten los $N$ datos.
- **¿Por qué voraz?** Encontrar el árbol binario óptimo es **NP-completo** (Hyafil y Rivest, 1976). La regla voraz no ve el futuro: una división mediocre puede abrir paso a divisiones excelentes.
- En la libreta, unas 30 líneas de NumPy reproducen a `DecisionTreeClassifier` en los pingüinos.

</v-clicks>
</div>
</div>

---
layout: seccion
numero: 2
---

# Árboles de regresión y poda

Las bicicletas de la Unidad 4, sin construir nada a mano.

---

# Hojas con promedios

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

Cada hoja predice el **promedio** de sus $y$ (la constante que minimiza los cuadrados). La impureza es la varianza; se busca

$$
\min_{j,\,t} \sum_{x_{nj} \le t} (y_n - \bar y_I)^2 + \sum_{x_{nj} > t} (y_n - \bar y_D)^2 .
$$

<div v-click class="pequeno">

Bicicletas por hora, 10,886 horas de entrenamiento, siete atributos **crudos**: hora, laboral, clima, año, temperatura, humedad, viento. Sin *one-hot*, sin escalar.

</div>
</div>
<div v-click>

Profundidad 3, la rama de la madrugada:

- ¿hora ≤ 6.5? → ¿hora ≤ 5.5?
  - **no** (6 a. m.) → ¿laboral?
    - fin de semana: **20.0** bicicletas
    - día laboral: **102.6** bicicletas

<div class="nota">

**Una rama es una interacción.** En la U4 hubo que crear `hora × laboral` a mano; el árbol la encontró solo.

</div>
</div>
</div>

---

# Lo que aprendió el árbol

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 32px">
<div>

<EscalonesBicis />

</div>
<div>

Árbol podado (150 hojas), en 2012, despejado, 25 °C. Fin de semana incluye feriados.

<v-clicks>

- Los dos picos de los días laborales y la loma del mediodía en fin de semana.
- Es una **función escalonada**: constante en cada rectángulo; ni suave ni lineal.
- Fuera del rango de entrenamiento se queda **constante**: no extrapola.

</v-clicks>
</div>
</div>

---

# Más profundidad, menos error... hasta cierto punto

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 32px">
<div>

<ProfundidadError />

</div>
<div>

<v-clicks>

- Sin límite: profundidad 31, **10,403 hojas** para 10,886 horas. RECM de entrenamiento 6.0; de prueba, 80.5.
- El mínimo de prueba, 69.5, en profundidad 10: ya mejor que Poisson.
- Pero elegir la profundidad mirando la prueba es trampa, y detener el crecimiento tiene un defecto: una división que gana poco puede preparar otras muy buenas.

</v-clicks>
</div>
</div>

---

# Poda por costo–complejidad

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Crecer un árbol grande $T_0$ y **podarlo**. Para un subárbol $T$ con $|T|$ hojas:

$$
R_\alpha(T) = R(T) + \alpha\, |T|
$$

<div class="pequeno">

Error de entrenamiento + precio por hoja: el esquema de regularización de la Unidad 3.

</div>

<v-click>

Colapsar la rama $T_t$ de un nodo $t$ en una hoja conviene cuando

$$
\alpha \ge g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1} .
$$

</v-click>
</div>
<div v-click>

<Caja tipo="proposicion" nombre="Breiman et al., 1984">

Podar repetidamente el nodo con menor $g(t)$, el **eslabón más débil**, da una sucesión **anidada** $T_0 \supset T_1 \supset \cdots \supset \{\text{raíz}\}$ y valores $\alpha_1 < \alpha_2 < \cdots$ tales que $T_k$ minimiza $R_\alpha$ para $\alpha \in [\alpha_k, \alpha_{k+1})$.

</Caja>

<div class="pequeno">

En las bicicletas: 8,406 subárboles. Solo hay que elegir **un número**, $\alpha$, por validación cruzada.

</div>
</div>
</div>

---

# Elegir $\alpha$ y comparar

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 36px">
<div>

<PodaCV :ancho="600" :alto="420" />

<div class="pequeno tenue">

Validación cruzada de 5 pliegues **por días** (`GroupKFold`): las horas de un mismo día se parecen demasiado para separarlas.

</div>
</div>
<div v-click>

RECM de prueba (bicicletas por hora):

<RecmArboles :ancho="560" />

<div class="pequeno" style="margin-top: 10px">

Con atributos crudos, el árbol podado le gana al mejor modelo lineal de la U4 y a Poisson.

</div>
</div>
</div>

---

# Inestabilidad: alta varianza

<RemuestrasArbol>

<div v-click class="pequeno">

En 200 remuestras, la primera pregunta cambia de atributo **35 veces**; en el 30 % del plano, al menos 10 % de los árboles contradicen a la mayoría.

</div>

<div v-click class="nota">

Poco sesgo, mucha varianza: un cambio cerca de la raíz se propaga a todo el árbol. La Unidad 3 sugiere el remedio: **promediar**. Es la próxima clase.

</div>

</RemuestrasArbol>

<!--
Cada árbol se entrenó con una remuestra bootstrap de los 342 pingüinos (342 con reemplazo); los puntos dibujados son los originales. Recorrer las cuatro remuestras y terminar con la votación de 200 árboles: eso es bagging.
-->

---

# Árboles contra modelos lineales

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

| | lineal | árbol |
|---|---|---|
| función | hiperplano | escalones |
| interacciones | a mano | las descubre |
| escala de los atributos | importa | da igual |
| extrapolación | lineal | constante |
| varianza | baja | alta |

</div>
<div>

<v-clicks>

- **Transformaciones monótonas:** el árbol solo compara $x_j \le t$; cambiar $x_j$ por $\phi(x_j)$ creciente da las mismas particiones.
- **Categóricos:** con $q$ categorías hay $2^{q-1} - 1$ particiones; para dos clases o regresión basta ordenar por la media de $y$.
- **Faltantes:** CART usaba divisiones sustitutas; scikit-learn aprende a qué lado mandarlos.
- Una frontera **diagonal** necesita una escalera de muchos escalones.

</v-clicks>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Un árbol parte el espacio en **rectángulos** con preguntas $x_j \le t$; cada rama es una interacción.
2. Se crece de forma **voraz** con la impureza: Gini o entropía, estrictamente cóncavas; el error no sirve.
3. En regresión, hojas con promedios y reducción de la suma de cuadrados.
4. Se controla con **poda por costo–complejidad** $R(T) + \alpha|T|$ y validación cruzada.
5. Poco sesgo, **mucha varianza**: la materia prima ideal para promediar.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Árboles de decisión: de los pingüinos a las bicicletas* (Unidad 5, ejemplo 1).

**Laboratorio 5:** árboles y *boosting* desde cero.

**Lectura:** *The Elements of Statistical Learning*, sección 9.2; L. Breiman et al., *Classification and Regression Trees* (1984), caps. 1–3.

</div>
</div>
</div>
