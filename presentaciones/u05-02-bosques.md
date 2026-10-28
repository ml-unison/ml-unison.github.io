---
theme: none
title: Bagging y bosques aleatorios
unidad: Unidad 5
layout: portada
subtitulo: "La sabiduría de muchos árboles: promediar para quitar varianza"
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

# *Bagging* y bosques aleatorios

---
layout: caso
caso: Caso · La multitud
cifra: 1,207 lb
cifraNota: mediana de 787 estimaciones del peso de un buey ya sacrificado y limpiado. El peso real fue de 1,198 lb.
fuente: F. Galton, «Vox populi», Nature 75 (1907), 450–451.
---

# Adivinar el peso de un buey

En una exposición de ganado en Plymouth, Inglaterra, cualquiera podía comprar un boleto de seis peniques y apostar cuánto pesaría un buey después de sacrificado. Francis Galton consiguió los boletos.

<v-click>

La mediana de los 787 boletos legibles se equivocó por **9 libras**, menos del 1 %: mucho menos que la mayoría de los concursantes.

</v-click>

<v-click>

<div class="pequeno">

**Pregunta para el grupo:** ¿cuándo promediar opiniones **no** ayuda? Si todos leyeron el mismo pronóstico equivocado, el promedio hereda el error. Los errores independientes se cancelan; los compartidos, no.

</div>

</v-click>

<!--
Galton contó 800 boletos; descartó 13 ilegibles o defectuosos. Usó la mediana («el voto del medio») porque cada boleto es un voto y un disparate no la mueve. Cerró la nota diciendo que el resultado hablaba de la confiabilidad del juicio democrático mejor de lo que se habría esperado.
-->

---

# Un árbol profundo cambia con cada muestra

<DiaRemuestras :bagging="$clicks >= 1" />

<div class="pequeno tenue" style="margin-top: 8px">

Bicicletas de Washington (Unidad 4), martes 29 de marzo de 2011, de prueba. Puntos: bicicletas rentadas en cada hora. Cada curva se entrenó con una remuestra *bootstrap* distinta del entrenamiento.
<span v-click>A la derecha, en cada remuestra se promedian 50 árboles: las curvas casi se enciman.</span>

</div>

<!--
Primero solo el panel rojo: los árboles siguen la forma del día, pero en las horas pico se separan más de cien bicicletas entre una remuestra y otra. Poco sesgo, mucha varianza. Preguntar: ¿qué haríamos con varias estimaciones ruidosas? Promediarlas, como en la feria.
-->

---

# La varianza de un promedio

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<Caja tipo="proposicion">

Si $Z_1, \dots, Z_B$ tienen varianza $\sigma^2$ y correlación $\rho$ entre cada par,

$$
\Var(\bar Z) = \rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2 .
$$

</Caja>

<v-click>

<div class="pequeno">

**Prueba.** $\Var\big(\frac1B\sum_b Z_b\big) = \frac{1}{B^2}\sum_{b,c} \operatorname{Cov}(Z_b, Z_c) = \frac{1}{B^2}\big[B\sigma^2 + B(B-1)\rho\sigma^2\big]$, y se reacomoda.

</div>

</v-click>

<v-click>

<div class="nota">

Promediar se lleva la parte **independiente** del error; la **compartida**, $\rho\sigma^2$, se queda aunque $B \to \infty$.

</div>

</v-click>
</div>
<div>

<VarianzaPromedio />

<div class="pequeno tenue">

Líneas: la fórmula. Puntos: simulación con 4,000 repeticiones de $Z_b = \sqrt{\rho}\,U + \sqrt{1-\rho}\,E_b$, donde $U$ es el error que todos comparten.

</div>
</div>
</div>

---
layout: seccion
numero: 1
---

# *Bagging*

Muchos árboles a partir de una sola muestra.

---

# *Bootstrap* + *aggregating*

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="algoritmo" nombre="Breiman, 1996">

1. Para $b = 1, \dots, B$: saca una remuestra *bootstrap* $\mathcal{D}_b$ ($N$ datos con reemplazo; Efron, 1979) y entrena con ella un árbol profundo $T_b$.
2. Predice con el promedio $\hat f(\vx) = \frac1B \sum_b T_b(\vx)$.

</Caja>

<div class="pequeno">

En clasificación, $T_b(\vx)$ es la proporción de positivos en la hoja; con hojas puras, $\hat f(\vx)$ es la **fracción de votos**.

</div>
</div>
<div>

Censo de EE. UU. de 1994 (Laboratorio 1): ¿gana más de 50,000 dólares al año? 9,000 personas para entrenar, 3,000 de prueba.

<v-click>

| prueba | exactitud | AUC |
|---|---|---|
| un árbol profundo | 0.815 | 0.763 |
| *bagging*, 300 árboles | **0.841** | **0.884** |

</v-click>

<v-click>

<div class="pequeno tenue">

El árbol acierta el 98.9 % en entrenamiento. Dos árboles en remuestras distintas discrepan en el 17.1 % de las personas de prueba: esa es la varianza que el promedio quita.

</div>

</v-click>
</div>
</div>

<!--
Codificación: OrdinalEncoder para las categóricas; los árboles no necesitan escalar ni one-hot. La AUC del árbol es mala porque sus «probabilidades» son casi todas 0 o 1: no sabe ordenar.
-->

---

# Lo que queda fuera de la bolsa

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Un dato no sale en un sorteo con probabilidad $1 - \frac1N$, y hay $N$ sorteos independientes:

$$
P(n \notin \mathcal{D}_b) = \Big(1 - \frac1N\Big)^N \xrightarrow[N\to\infty]{} e^{-1} \approx 0.368 ,
$$

porque $N \ln\big(1 - \frac1N\big) = -1 - \frac{1}{2N} - \cdots \to -1$.

<v-click>

<div class="pequeno">

En los 300 árboles del censo ($N = 9{,}000$): quedó fuera el **36.80 %** en promedio (de 35.9 % a 37.5 %); la fórmula da 36.79 %.

</div>

</v-click>
</div>
<div>

<v-click>

<Caja tipo="definicion" nombre="estimación fuera de la bolsa (OOB)">

Para cada dato de entrenamiento, promedia solo los árboles que **no** lo vieron:

$$
\hat f_{\text{OOB}}(\vx_n) = \operatorname*{prom}_{b \,:\, n \notin \mathcal{D}_b} T_b(\vx_n) ,
$$

y mide el error de esas predicciones.

</Caja>

</v-click>

<v-click>

*Bagging* en el censo: error OOB **0.157**; error de prueba **0.159**. Cada dato recibió en promedio 110.4 votos ($\approx 0.368 \cdot 300$), sin entrenar nada más.

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# Bosques aleatorios

Árboles menos parecidos, mejor promedio.

---

# Sortear atributos en cada división

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Los árboles del *bagging* se parecen: todos ven casi los mismos datos y empiezan por las mismas divisiones fuertes. $\rho$ alta, piso $\rho\sigma^2$ alto.

<v-click>

<Caja tipo="definicion" nombre="bosque aleatorio (Breiman, 2001)">

*Bagging* en el que, en **cada división**, se sortean $m$ de los $d$ atributos y solo se busca entre ellos. Valores habituales: $m = \lfloor\sqrt d\rfloor$ en clasificación, $d/3$ en regresión. Con $m = d$ es *bagging*.

</Caja>

</v-click>
</div>
<div v-click>

Censo, $d = 12$, 300 árboles:

| | *bagging* | bosque, $m = 3$ |
|---|---|---|
| correlación entre árboles | 0.554 | **0.527** |
| AUC de un árbol | **0.740** | 0.737 |
| AUC del ensamble | 0.884 | **0.891** |
| error de prueba | 0.159 | **0.151** |

<div class="pequeno tenue">

Cada árbol es un poco peor ($\sigma^2$ sube), pero se parecen menos ($\rho$ baja), y el promedio gana.

</div>
</div>
</div>

<!--
La «correlación entre árboles» de la tabla es la correlación entre las predicciones de dos árboles a lo largo de las personas de prueba: una medida práctica de parecido. La ρ de la proposición (en un x fijo, a lo largo de conjuntos de entrenamiento) la medimos con las bicicletas más adelante.
-->

---

# Más árboles no sobreajustan

<div class="cols" style="grid-template-columns: 2fr 1fr; gap: 36px">
<div>

<CurvaArboles />

</div>
<div>

<v-clicks>

- Con un árbol el error es alto; en unas decenas se estabiliza.
- Ninguna curva vuelve a subir: $B$ no se afina, solo se ponen **suficientes** árboles. Agregar uno solo acerca el promedio a su límite.
- Las curvas OOB (punteadas) siguen a las de prueba **sin usar la prueba**.

</v-clicks>

<div v-click class="nota">

Los vaivenes son ruido: 0.003 de error son 9 personas de 3,000.

</div>
</div>
</div>

---

# $m$ y el tamaño de hoja, elegidos fuera de la bolsa

<OOBHiper />

<div class="cols" style="gap: 40px; margin-top: 6px">
<div class="pequeno">

$m = 3 = \lfloor\sqrt{12}\rfloor$ es el mejor, pero las diferencias son chicas: de 0.892 a 0.897.

</div>
<div v-click class="pequeno">

Hojas de al menos 5 datos suben la AUC de 0.897 a **0.916**: cada árbol da una proporción, no un voto 0/1, y el promedio ordena mejor.

</div>
</div>

---

# Fuera de la bolsa contra validación cruzada

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 48px">
<div>

Bosque elegido: 500 árboles, $m = 3$, hojas de al menos 5 datos.

| estimación de la AUC | AUC | bosques entrenados | tiempo |
|---|---|---|---|
| fuera de la bolsa | 0.9165 | 1 | 1.0 s |
| validación cruzada, 5 pliegues | 0.9154 | 5 | 4.9 s |
| prueba | 0.9145 | — | — |

<div class="pequeno tenue">

Los tiempos dependen de la máquina; la proporción, alrededor de 1 a 5, no.

</div>
</div>
<div>

<v-click>

El OOB sale **del mismo ajuste** que produce el modelo. Por eso con bosques casi siempre se usa en lugar de validación cruzada.

</v-click>

<v-click>

<Caja tipo="cuidado">

Cada dato recibe solo $\approx B/3$ votos: con pocos árboles el OOB es pesimista. Y solo existe si cada modelo deja datos fuera.

</Caja>

</v-click>
</div>
</div>

---

# Bicicletas: sesgo y varianza, medidos

<div class="cols" style="grid-template-columns: 2fr 3fr; gap: 40px">
<div>

<div class="pequeno">

| modelo | RECM de prueba |
|---|---|
| M4 lineal (Unidad 4) | 86.4 |
| un árbol profundo | 80.5 |
| M4 Poisson (Unidad 4) | 72.2 |
| bosque, $m = 3$ | 62.8 |
| *bagging*, $m = 7$ | **61.7** |

</div>

<div class="pequeno tenue">

Los atributos de M4, **sin** la interacción hora × laboral hecha a mano: el árbol la descubre solo.

</div>
</div>
<div>

<v-click>

<SesgoVarianzaBarras />

<div class="pequeno tenue">

20 remuestras *bootstrap* del entrenamiento; en cada una, un árbol, un *bagging* de 50 y un bosque de 50. Promediado sobre las 6,493 horas de prueba.

</div>

</v-click>
</div>
</div>

<!--
Bagging divide la varianza entre 5.7 (2,619 → 463). El término sesgo² + ruido también baja un 8 %: un promedio de funciones escalonadas es más suave que cualquiera de ellas. El bosque con m = 3 tiene menos varianza, pero más sesgo: con solo 7 atributos y la hora dominando, a veces no puede usar la hora. Moraleja: m se elige con OOB o CV, no se fija de antemano. No conocemos la f verdadera, así que sesgo² y ruido no se separan.
-->

---

# $\rho$ y $\sigma^2$ en datos reales

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

En cada hora de prueba $\vx$, con las 20 remuestras:

- $\sigma^2(\vx)$: varianza de **un árbol** del ensamble;
- $\rho\sigma^2(\vx)$: covarianza entre **dos árboles del mismo ensamble**, que comparten los datos de entrenamiento.

<v-click>

Esa covarianza es el **piso**: la varianza del promedio con $B \to \infty$.

</v-click>
</div>
<div v-click>

| bicicletas | $\sigma^2$ | $\rho$ | piso $\rho\sigma^2$ | $B = 50$ |
|---|---|---|---|---|
| *bagging* | 3,156 | 0.129 | 408 | 463 |
| bosque, $m = 3$ | 6,445 | 0.030 | 195 | 320 |

<div class="pequeno">

$408 + (1 - 0.129)\cdot 3{,}156/50 = 463$, la varianza medida en la lámina anterior.

</div>

<div class="nota">

Sortear atributos dividió el piso entre dos, pero duplicó $\sigma^2$ y subió el sesgo. Con pocos atributos y uno dominante, gana el *bagging*.

</div>
</div>
</div>

---
layout: seccion
numero: 3
---

# ¿Qué atributos usa el bosque?

Dos medidas de importancia, y una trampa.

---

# Dos importancias

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="definicion" nombre="impureza media (MDI)">

Cada vez que un árbol divide con el atributo $j$, el Gini baja. Se suman esas bajadas, ponderadas por la fracción de datos del nodo, y se promedian sobre los árboles. Sale gratis del entrenamiento.

</Caja>

<v-click>

<Caja tipo="definicion" nombre="importancia por permutación">

En datos que el modelo **no** vio, se revuelve la columna $j$ y se mide cuánto empeora la métrica. Si el modelo no la usa, no cambia nada.

</Caja>

</v-click>
</div>
<div v-click>

**El experimento.** Agregamos al censo dos columnas inútiles por construcción:

- `ruido`: un número normal al azar;
- `id`: un número de identificación distinto para cada persona.

Ninguna tiene relación con el ingreso. ¿Qué importancia les da cada medida?

</div>
</div>

---

# MDI premia lo que tiene muchos valores

<ImportanciasDos :permutacion="$clicks >= 1" />

<div v-click class="pequeno tenue" style="margin-top: 4px">

Bosque de 300 árboles con hojas de un dato. Por permutación, `id` y `ruido` valen −0.001: nada.

</div>

<!--
Primero solo MDI: preguntar al grupo si creen que el id es el tercer atributo más importante, por encima de relationship y capital-gain. Luego la permutación, con el clic. El −0.001 es azar: revolver una columna inútil a veces mejora un poco la AUC.
-->

---

# Por qué MDI se equivoca

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Los árboles crecen hasta hojas puras. En los nodos profundos quedan pocos datos, y un atributo con **muchos valores** ofrece muchos umbrales: alguno separa esos pocos datos **por casualidad**.

<v-click>

Esa división baja la impureza **en entrenamiento**, y MDI la cuenta como mérito. `id` y `ruido` tienen 9,000 valores distintos; `sex`, 2. (C. Strobl y colaboradores, *BMC Bioinformatics*, 2007.)

</v-click>
</div>
<div>

<v-click>

<Caja tipo="idea">

Para decir qué usa un modelo, **permutación en datos de prueba**. MDI solo como primera mirada, nunca para comparar atributos con números de valores muy distintos.

</Caja>

</v-click>

<v-click>

<div class="nota">

La permutación da casi nada a `sex` (0.002), y no porque el sexo no importe para el ingreso: `relationship` (esposo / esposa) lo contiene. Y lo que aprende el modelo describe la desigualdad de 1994, no un criterio para decidir sobre personas.

</div>

</v-click>
</div>
</div>

---

# Comparación en la prueba del censo

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

| modelo | AUC | exactitud |
|---|---|---|
| un árbol profundo | 0.763 | 0.815 |
| *bagging* | 0.886 | 0.840 |
| bosque, valores por omisión | 0.891 | 0.847 |
| regresión logística (Unidad 4) | 0.903 | 0.846 |
| *extra trees*, $m = 3$, hoja 5 | 0.906 | 0.845 |
| bosque, $m = 3$, hoja 5 | **0.915** | **0.860** |

<div class="pequeno tenue">

Ensambles de 500 árboles. Logística con *one-hot*, logaritmo de las ganancias de capital y estandarización, como en el Laboratorio 1.

</div>
</div>
<div>

<v-clicks>

- El bosque por omisión **no** le gana a la logística en AUC; ajustado con OOB, sí.
- ***Extra trees*** (Geurts y colaboradores, 2006): también sortean el **umbral** de cada atributo y no usan *bootstrap*. Más rápidos, árboles aún menos correlacionados.
- La logística sigue siendo una línea base muy digna. La próxima clase: *boosting*.

</v-clicks>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Promediar $B$ estimaciones deja $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$: se va lo independiente, queda lo compartido.
2. *Bagging*: árboles profundos (poco sesgo) en remuestras *bootstrap*, promediados (menos varianza).
3. Bosque aleatorio: sortear $m$ atributos por división baja $\rho$; $m$ y el tamaño de hoja se eligen.
4. Cada árbol deja fuera $\approx e^{-1}$ de los datos: el error **OOB** es validación casi gratis.
5. Importancia: **permutación en prueba**; MDI favorece atributos con muchos valores.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Bosques aleatorios: la sabiduría de muchos árboles* (Unidad 5, ejemplo 2).

**Laboratorio 5:** árboles y *boosting* desde cero, con *bagging* y estimación OOB.

**Lectura:** *The Elements of Statistical Learning*, secciones 8.7 y 15.1–15.3; L. Breiman, «Random forests», *Machine Learning* 45 (2001).

</div>
</div>
</div>
