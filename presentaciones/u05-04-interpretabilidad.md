---
theme: none
title: "Interpretabilidad: qué aprendió el modelo"
unidad: Unidad 5
layout: portada
subtitulo: "Importancia por permutación, dependencia parcial y valores de Shapley, con las casas de Ames"
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

# Interpretabilidad: qué aprendió el modelo

---
layout: pregunta
---

# Un modelo predice muy bien qué pacientes con neumonía van a morir. ¿Lo usarías para decidir a quién hospitalizar?

<v-click>

Antes habría que saber **qué aprendió**. Un buen error de prueba no basta: los datos de prueba tienen los mismos sesgos que los de entrenamiento.

</v-click>

<!--
Dejar que el grupo responda. Lo normal es que digan que sí, si el error de prueba es bueno. La siguiente lámina muestra por qué no.
-->

---
layout: caso
caso: Caso · Salud
cifra: asma
cifraNota: según el modelo, tener asma reducía el riesgo de morir de neumonía.
fuente: G. F. Cooper et al., Artificial Intelligence in Medicine 9 (1997); R. Caruana et al., «Intelligible models for healthcare», KDD (2015).
---

# El asma que «protegía»

A mediados de los noventa se entrenaron varios modelos para estimar el riesgo de muerte de pacientes con neumonía y decidir a quién internar. Un sistema de reglas aprendió: **tener asma ⇒ menor riesgo**.

<v-click>

Era cierto en los datos: a los pacientes con asma se les internaba de inmediato, a menudo en terapia intensiva, y ese cuidado los salvaba. Usado para decidir, el modelo los habría mandado a casa.

</v-click>

<v-click>

Las redes neuronales del proyecto eran más exactas y **no se usaron**: nadie podía revisar qué habían aprendido. Veinte años después, Caruana et al. encontraron la misma regla con un modelo interpretable.

</v-click>

<!--
El modelo no está "mal": describe correctamente los datos. El problema es que los datos son el resultado de una política de tratamiento, y el modelo se iba a usar para cambiar esa política. Es un sesgo que ninguna métrica de prueba delata. Caruana et al. usaron modelos aditivos generalizados con interacciones de pares (GA2M) y encontraron además que el dolor de pecho y las enfermedades del corazón también "bajaban" el riesgo, por la misma razón.
-->

---

# ¿Para qué interpretar un modelo?

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 56px">
<div>

<v-clicks>

- **Depurar:** encontrar el asma, las fugas de información (Unidad 1) o un atributo mal codificado.
- **Confianza:** un médico o un valuador no usa lo que no entiende.
- **Ciencia:** descubrir qué patrones hay en los datos.
- **Regulación:** en crédito, por ejemplo, hay que darle al solicitante rechazado las razones principales.

</v-clicks>

</div>
<div v-click>

**Global o local.** Una explicación global describe el modelo completo («la calidad es lo que más pesa»); una local explica una predicción («esta casa vale 267 mil porque…»).

**Por diseño o *post hoc*.** Un modelo lineal o un árbol chico se leen directamente. Las explicaciones *post hoc* se calculan después, mirando la caja negra desde afuera: son **aproximaciones** de lo que hace.

</div>
</div>

<!--
Todo lo que veremos hoy es post hoc: funciona para cualquier modelo, pero hay que recordar que es una descripción del modelo, no el modelo.
-->

---

# El problema de hoy: el precio de las casas de Ames

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div>

- Las 1,456 ventas de casas de Ames, Iowa (2006–2010) de las unidades 1 y 3; 1,164 para entrenar y 292 de prueba.
- 12 atributos legibles: área habitable, calidad general, año de construcción, sótano, cochera, terreno, baños, barrio…
- Predecimos $\ln(\text{precio})$: el error se lee como error relativo y las explicaciones, como **factores**.
- Modelo: *gradient boosting* (LightGBM, 600 árboles de 8 hojas). Rival: regresión ridge.

</div>
<div>

| modelo | RECM (CV) | RECM (prueba) |
|---|---|---|
| ridge | 0.125 ± 0.002 | 0.138 |
| LightGBM | 0.128 ± 0.004 | 0.141 |

<p class="pequeno tenue">Validación cruzada de 5 particiones en entrenamiento (± error estándar). Una RECM de 0.13 en logaritmo es un error típico de 13 % del precio.</p>

</div>
</div>

<!--
Usamos la misma partición de la Unidad 3. El barrio entra al boosting como el precio mediano de su barrio en entrenamiento (codificación por la respuesta, solo con entrenamiento). Las áreas, en metros cuadrados.
-->

---
layout: cita
autor: C. Rudin, Nature Machine Intelligence 1 (2019)
nota: "En Ames, la regresión ridge empata con el boosting, y sus coeficientes ya son la explicación: con todo lo demás fijo, un punto más de calidad sube el precio 6.7 %, y un 10 % más de área, 3.6 %. Aun así, aprendamos a abrir la caja negra: no siempre empatan."
---

*Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead.*

<!--
El título del artículo es la tesis. Rudin argumenta que en datos tabulares con atributos bien construidos, los modelos interpretables suelen empatar con las cajas negras, y que las explicaciones post hoc pueden ser engañosas. Aquí lo comprobamos: empate. La moraleja práctica es siempre probar el modelo interpretable antes.
-->

---
layout: seccion
numero: 1
---

# Qué atributos usa el modelo

Importancia por permutación.

---

# Importancia por permutación

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 40px">
<div>

**Barajar** la columna $j$ en prueba rompe su relación con $y$ sin cambiar su distribución:

$$
I_j = \frac1K \sum_{k=1}^{K} L\big(\mathbf{y}, \hat f(X^{\pi_k}_{(j)})\big) - L\big(\mathbf{y}, \hat f(X)\big)
$$

<v-clicks>

- Sirve para **cualquier** modelo (Breiman, 2001; Fisher, Rudin y Dominici, 2019).
- Se mide **en prueba**: lo que el modelo usa para generalizar, no lo que memorizó.
- Barajar el área sube la RECM de 0.141 a 0.212. Barajar los baños no cambia nada: el modelo ya sabe el área.

</v-clicks>

</div>
<div>

<PermutacionGemela />

</div>
</div>

<!--
K = 10 permutaciones por atributo. La pérdida L es la RECM en logaritmo del precio. La importancia de los baños casi nula no quiere decir que los baños no se relacionen con el precio: quiere decir que, dados los demás atributos (sobre todo el área), no agregan información.
-->

---

# Dos atributos casi iguales se reparten la importancia

<PermutacionGemela :gemela="true" />

<div class="cols" style="gap: 40px; margin-top: 4px">
<div class="pequeno">

Agregamos el área **declarada por el vendedor** (error de 3 %, correlación 0.996). Mismo error, RECM 0.142; la importancia del área cae de 0.071 a 0.042.

</div>
<div class="pequeno" v-click>

Al barajar una, queda la otra: se mide lo que se pierde **dados los demás**, no la información que aporta. Remedio: barajar **grupos**.

</div>
</div>

<!--
Además, barajar el área crea casas que no existen: 300 metros cuadrados con un baño. El modelo nunca vio casas así y su respuesta ahí es arbitraria. Eso lo vemos con más claridad en la sección siguiente.
-->

---
layout: seccion
numero: 2
---

# Cómo cambia la predicción

Dependencia parcial y curvas ICE.

---

# Dependencia parcial: el mismo valor para todas las casas

<div class="cols" style="grid-template-columns: 4fr 6fr; gap: 36px">
<div>

$$
\hat f_S(x_S) = \frac1N \sum_{i=1}^{N} \hat f\big(x_S, \vx_C^{(i)}\big)
$$

<div class="pequeno">

Friedman (2001). Estima $\E_{X_C}\big[\hat f(x_S, X_C)\big]$.

<v-clicks>

- Cada sumando, como función de $x_S$, es una curva **ICE**: qué le pasaría a **esa** casa si solo cambiara su área (Goldstein et al., 2015).
- PDP = promedio de las ICE. Curvas paralelas: sin interacciones.
- De 64 a 286 m², el precio se multiplica por **1.63**; de 1896 a 2009, por 1.23.

</v-clicks>

</div>
</div>
<div>

<CurvasICE />

</div>
</div>

<!--
Usar los botones para cambiar al año de construcción y para ocultar las curvas ICE. Las escaleras son la huella de los árboles: solo aprenden funciones escalonadas. Arriba de 260 metros cuadrados se aplana porque no hay casas: ningún árbol tiene datos para seguir subiendo. Las ICE son casi paralelas en escala logarítmica: el efecto del área es un factor común, que es justo lo que supone el modelo lineal en logaritmo.
-->

---

# El precio de promediar: casas que no existen

<div class="cols cols-6-4" style="gap: 36px">
<div>

<CasasSinteticas />

</div>
<div>

Para calcular $\hat f_S(250\text{ m}^2)$, la PDP le pone 250 m² a **todas** las casas, también a las chicas y sencillas.

<v-clicks>

- Casas reales de 250 m² o más: 31; con calidad 5 o menos: **una**.
- Puntos que la PDP promedia en 250 m² con calidad 5 o menos: **36 %**.

</v-clicks>

<div v-click class="nota">

Con atributos correlacionados, la PDP mezcla extrapolaciones con predicciones razonables. Los efectos locales acumulados (ALE; Apley y Zhu, 2020) promedian solo sobre casas parecidas.

</div>
</div>
</div>

<!--
Cada punto rojo es una casa real a la que se le cambió el área por 250 o 300 metros cuadrados. El modelo nunca vio esa región; lo que predice ahí lo dictan los árboles, que simplemente extienden el último escalón. La importancia por permutación y SHAP intervencional tienen el mismo problema.
-->

---
layout: seccion
numero: 3
---

# Valores de Shapley

Repartir una predicción entre los atributos.

---

# Un juego cooperativo

<div class="cols" style="grid-template-columns: 11fr 9fr; gap: 44px">
<div>

¿Por qué **esta** casa vale lo que el modelo dice? Los atributos son **jugadores**; una **coalición** $S$ es lo que sabemos de la casa. Lo que «gana» una coalición:

$$
v(S) = \E\big[\hat f(\vx_S, X_C)\big] \approx \frac{1}{100} \sum_{i} \hat f\big(\vx_S, \vx_C^{(i)}\big)
$$

con los demás atributos tomados de 100 casas de **fondo**.

<v-click>

Modelo de juguete: tres atributos, precio en miles de dólares. La casa: 180 m², calidad 8, construida en 2009. Hay que repartir $322.3 - 179.8 = 142.5$.

</v-click>

</div>
<div class="tabla-chica" v-click>

| coalición $S$ | $v(S)$ |
|---|---|
| ∅ | 179.8 |
| área | 200.0 |
| calidad | 234.2 |
| año | 226.5 |
| área, calidad | 264.8 |
| área, año | 254.2 |
| calidad, año | 281.0 |
| las tres | **322.3** = $\hat f(\vx)$ |

</div>
</div>

<style>
.tabla-chica table { font-size: 20px; }
.tabla-chica td { padding: 6px 14px 6px 0; }
.tabla-chica th { padding-bottom: 8px; }
</style>

<!--
v(vacío) es la predicción promedio del modelo en el fondo; v de todos es la predicción para la casa. Es la misma idea de la dependencia parcial, pero fijando varios atributos a la vez.
-->

---

# Seis órdenes de llegada, un promedio

<OrdenesShapley />

<div class="pequeno" style="margin-top: 8px">

Cada atributo recibe lo que su llegada **agrega**. El área aporta 20.2 si llega primero y 41.3 si llega al último: en dólares, el metro cuadrado vale más en una casa de calidad 8. El valor de Shapley promedia los $3! = 6$ órdenes: área 30.2, calidad 60.7, año 51.6.

</div>

<!--
Leer un renglón: la casa promedio vale 179.8; si primero sabemos el área, sube a 200.0 (contribución 20.2); luego la calidad, a 264.8 (64.8); luego el año, a 322.3 (57.5). Todos los renglones terminan en el mismo punto: cambia el reparto, no el total. La biblioteca shap da exactamente los mismos tres números.
-->

---

# El valor de Shapley (1953)

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Cuatro propiedades de un reparto justo $\phi_1, \dots, \phi_d$:

<v-clicks>

1. **Eficiencia:** $\sum_j \phi_j = v(\text{todos}) - v(\varnothing)$.
2. **Simetría:** si $j$ y $k$ aportan lo mismo a toda coalición, $\phi_j = \phi_k$.
3. **Jugador nulo:** si $j$ nunca aporta, $\phi_j = 0$.
4. **Aditividad:** el reparto de $v + w$ es la suma de los repartos.

</v-clicks>

</div>
<div v-click>

<Caja tipo="teorema" nombre="Shapley, 1953">

Hay un **único** reparto que cumple las cuatro:

$$
\phi_j = \sum_{S \subseteq D \setminus \{j\}} \frac{|S|!\,(d - |S| - 1)!}{d!}\,\big[v(S \cup \{j\}) - v(S)\big]
$$

</Caja>

<div class="pequeno">

El peso es la fracción de los $d!$ órdenes en los que, justo antes de $j$, llegaron exactamente los de $S$: la fórmula es el promedio sobre órdenes de la lámina anterior.

</div>
</div>
</div>

<!--
D = {1, ..., d}. Shapley la propuso para repartir ganancias entre jugadores de un juego cooperativo; le dieron el Nobel de Economía en 2012 (junto con A. Roth), aunque por su trabajo en asignaciones estables. Idea de la unicidad: los juegos "de unanimidad" (ganan 1 si S contiene a un conjunto T) forman una base de todos los juegos; simetría, jugador nulo y eficiencia fijan el reparto en cada uno, y la aditividad lo extiende.
-->

---

# SHAP: Shapley para explicar predicciones

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Lundberg y Lee (2017): con el juego $v_{\vx}(S) = \E\big[\hat f(\vx_S, X_C)\big]$ (**intervencional**),

$$
\hat f(\vx) = \phi_0 + \sum_{j=1}^{d} \phi_j(\vx), \qquad \phi_0 = \E\big[\hat f(X)\big].
$$

<v-click>

<Caja tipo="ejemplo" nombre="modelo lineal">

Si $\hat f(\vx) = w_0 + \vw\T\vx$, la contribución de $j$ a **cualquier** coalición es $w_j\big(x_j - \E[X_j]\big)$, así que

$$
\phi_j = w_j\big(x_j - \E[X_j]\big).
$$

</Caja>

</v-click>
</div>
<div>

<v-click>

**El costo.** Con 12 atributos hay $2^{12} = 4{,}096$ coaliciones; con 100 casas de fondo, 409,600 predicciones por casa explicada: unos 4 segundos.

</v-click>

<v-click>

**TreeSHAP** (Lundberg et al., 2020) aprovecha que cada camino de un árbol usa pocos atributos: valores **exactos** en tiempo polinomial, en unos milisegundos.

</v-click>

<v-click>

<div class="nota">

Verificado en la libreta: la enumeración y `shap` difieren en a lo más $2.5 \times 10^{-7}$.

</div>

</v-click>
</div>
</div>

<!--
Intervencional: fijamos los atributos de S y tomamos los demás de su distribución marginal, como si interviniéramos la casa. La alternativa "observacional" condiciona en X_S = x_S, que es más difícil de estimar. Sin fondo, TreeExplainer usa la versión tree_path_dependent, que aproxima la condicional con las cuentas de los árboles. Para modelos que no son árboles, KernelSHAP muestrea coaliciones (mucho más lento).
-->

---

# Por qué esta casa vale 267 mil dólares

<CascadaSHAP />

<div class="pequeno tenue" style="margin-top: 4px">

245 m², construida en 1880 y remodelada en 2003, en un barrio barato. Los $\phi_j$ se suman en logaritmo; en dólares, cada atributo **multiplica** el precio por $e^{\phi_j}$.

</div>

<!--
Leer de arriba abajo: la referencia es e^{phi_0}, el precio "promedio" (geométrico) del modelo. El área la encarece 34 %, la antigüedad la abarata 11 %. Preguntar: ¿le daríamos esta explicación a un comprador? ¿Qué podría discutir? Por ejemplo, que la remodelación de 2003 apenas cuenta (×1.02, dentro de "otros"). Los factores no dependen del orden de las barras; los dólares que "suma" cada uno sí dependerían.
-->

---

# Todas las casas a la vez: el enjambre

<div class="cols" style="grid-template-columns: 7fr 3fr; gap: 28px">
<div>

<EnjambreSHAP />

</div>
<div class="pequeno">

Un punto por casa de prueba y atributo, ordenados por $\frac1N \sum_i |\phi_j(\vx_i)|$.

<v-clicks>

- **Dirección:** en área y calidad, lo rojo (valores altos) sube el precio.
- **Forma:** las casas muy grandes reciben hasta $e^{0.32} \approx 1.38$.
- **Dispersión:** un mismo valor con efectos distintos es la huella de las **interacciones**.

</v-clicks>

</div>
</div>

<!--
La importancia SHAP media y la de permutación ordenan parecido pero no igual: una mide cuánto se mueve la predicción, la otra cuánto empeora el error sin el atributo. La remodelación, por ejemplo, mueve bastante la predicción pero casi no empeora el error al barajarla: la información se recupera del año de construcción.
-->

---

# Un atributo, casa por casa

<div class="cols" style="grid-template-columns: 6fr 4fr; gap: 36px">
<div>

<DependenciaSHAP />

</div>
<div>

$\phi_{\text{área}}$ contra el área: como la dependencia parcial, pero con **casas reales** y con la dispersión a la vista.

<v-click>

Arriba de 180 m², las casas de mejor calidad quedan **por encima** de la curva: en una casa buena, cada metro cuadrado vale un poco más. Es una **interacción** que la PDP promedia y esconde.

</v-click>

</div>
</div>

<!--
Por debajo de 150 metros cuadrados los colores se mezclan: ahí los dos efectos son casi independientes. La PDP está desplazada verticalmente para que su promedio en las casas de prueba coincida con el de los valores SHAP (la PDP está en otra escala: no tiene el término phi_0 restado).
-->

---

# Lo que una explicación no dice

<div class="cols" style="grid-template-columns: 1fr 1fr 1fr; gap: 36px">
<div>

### Modelo ≠ mundo

SHAP describe $\hat f$, no el mecanismo que produce los precios. Que la chimenea sume no quiere decir que **construir** una suba el precio: va junto con otras cosas que el modelo no ve. El asma no protege.

</div>
<div v-click>

### Muchos modelos, muchas explicaciones

Con el área del vendedor, el modelo es igual de bueno y $\overline{|\phi|}$ del área baja de 0.085 a 0.062; la gemela se lleva 0.022. Sumadas, 0.085. Breiman lo llamó el **efecto Rashomon**.

</div>
<div v-click>

### Se pueden manipular

Slack et al. (2020) construyeron clasificadores que discriminan por raza en los datos reales y se portan bien en los puntos sintéticos que usan LIME y SHAP: las explicaciones esconden la raza.

</div>
</div>

<div v-click class="nota">

Una explicación *post hoc* no es una auditoría. Si lo que está en juego es alto, mejor un modelo que se pueda leer directamente.

</div>

<!--
Para afirmaciones causales ("si agrego una chimenea, el precio sube tanto") se necesita un experimento o un modelo causal explícito. Slack, Hilgard, Jia, Singh y Lakkaraju, AIES 2020, "Fooling LIME and SHAP". La razón de fondo es la misma de las casas imposibles: los métodos evalúan el modelo fuera de los datos.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. Exacto no quiere decir seguro: hay que **mirar qué aprendió** el modelo.
2. Primero, probar un modelo **interpretable por diseño**; si empata, usarlo.
3. Permutación: cuánto **depende** el modelo de un atributo, dados los demás.
4. PDP e ICE: **cómo** cambia la predicción; cuidado con las casas que no existen.
5. SHAP: el único reparto **aditivo** con los axiomas de Shapley. Explica el modelo, no el mundo.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Interpretar un modelo de precios de casas* (Unidad 5, ejemplo 4).

**Laboratorio 5:** árboles y *boosting* desde cero.

**Lectura:** C. Molnar, *Interpretable Machine Learning*, 2.ª ed. (2022), caps. 8.1, 8.5 y 9.5–9.6; C. Rudin, *Nature Machine Intelligence* 1 (2019).

</div>
</div>
</div>
