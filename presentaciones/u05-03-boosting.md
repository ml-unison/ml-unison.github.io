---
theme: none
title: "Boosting: de AdaBoost a XGBoost, LightGBM y CatBoost"
unidad: Unidad 5
layout: portada
subtitulo: Sumar modelos débiles, uno tras otro, para corregir los errores de los anteriores
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

# *Boosting*: de AdaBoost a XGBoost, LightGBM y CatBoost

---
layout: pregunta
---

# Tienes un clasificador que acierta apenas un poco más que una moneda. ¿Puedes convertirlo en uno tan bueno como quieras?

<v-click>

Kearns y Valiant (1988–89) lo plantearon como problema abierto de la teoría PAC. Schapire (1990) demostró que **sí**; Freund y Schapire (1997) lo volvieron práctico con **AdaBoost**, y recibieron el premio Gödel en 2003. La idea: entrenar muchos clasificadores débiles **en secuencia**, cada uno concentrado en lo que los anteriores hicieron mal.

</v-click>

<!--
Pedir ideas antes de mostrar la respuesta. Suelen proponer «promediar muchos», como en el bagging. Hacer notar la diferencia: si todos los clasificadores débiles se equivocan en los mismos casos, promediarlos no ayuda. Hay que obligar a cada uno a mirar donde los otros fallan.
-->

---

# AdaBoost: concentrarse en los errores

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<Caja tipo="algoritmo" nombre="AdaBoost">

Etiquetas $y^{(i)} \in \{-1, +1\}$; cada dato empieza con peso $w^{(i)} = 1/M$. Para cada ronda $t = 1, \dots, B$:

1. ajustar el clasificador débil $h_t$ a los datos con pesos $w^{(i)}$;
2. error ponderado $\varepsilon_t = \sum_i w^{(i)}\, \ind{h_t(\vx^{(i)}) \ne y^{(i)}}$;
3. voto $\alpha_t = \frac12 \ln \frac{1 - \varepsilon_t}{\varepsilon_t}$;
4. $w^{(i)} \leftarrow w^{(i)}\, e^{-\alpha_t y^{(i)} h_t(\vx^{(i)})}$ y normalizar.

Clasificador final, tras $B$ rondas: $\hat y = \operatorname{signo}\big(F_B(\vx)\big)$, con $F_B = \sum_{t=1}^{B} \alpha_t h_t$.

</Caja>

</div>
<div>

<v-clicks>

- $y^{(i)} h_t(\vx^{(i)}) = +1$ si acertó: el peso se multiplica por $e^{-\alpha_t} < 1$. Si falló, por $e^{\alpha_t} > 1$.
- Un clasificador mejor que la moneda ($\varepsilon_t < \frac12$) recibe voto $\alpha_t > 0$; uno casi perfecto, un voto enorme.
- **Aprendiz débil** típico: un **tocón**, un árbol de una sola pregunta $x_j > u$ (atributo $j$, umbral $u$). Malo solo, con mucho sesgo y poca varianza.

</v-clicks>
</div>
</div>

<!--
Comparar con el bagging: allí los árboles son profundos (poco sesgo, mucha varianza) y se entrenan en paralelo e independientes. Aquí son tocones (mucho sesgo, poca varianza) y se entrenan en secuencia, cada uno dependiendo de los anteriores.
-->

---

# AdaBoost en los pingüinos, ronda por ronda

<AdaBoostPasos />

<!--
Barbijo (Chinstrap) contra Adelia y papúa, con longitud y profundidad del pico. Los barbijo tienen el pico largo y profundo a la vez: ninguna pregunta sola los separa.
Recorrer las rondas: el tocón 1 dice «barbijo si longitud > 50.5»; se equivoca con los barbijo de pico más corto y los papúa de pico largo, que en la ronda 2 cargan la mitad del peso total (siempre es así: ver el ejercicio 1 de la libreta). El tocón 2 pregunta por la profundidad; el 3 vuelve a la longitud con otro umbral. Con 50 rondas los pesos se concentran en la frontera; con 200 aparecen entrantes alrededor de puntos aislados y franjas en zonas sin datos: memoriza.
-->

---

# El error de entrenamiento cae exponencialmente

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

<Caja tipo="proposicion" nombre="Freund y Schapire, 1997">

Con $\varepsilon_t = \frac12 - \gamma_t$ ($\gamma_t$: ventaja de $h_t$ sobre la moneda), el error de entrenamiento cumple

$$
\frac1M \sum_{i=1}^{M} \ind{\hat y^{(i)} \ne y^{(i)}} \le \frac1M\sum_{i=1}^{M} e^{-y^{(i)} F_B(\vx^{(i)})}
$$

$$
= \prod_{t=1}^{B} 2\sqrt{\varepsilon_t(1 - \varepsilon_t)} \le e^{-2\sum_t \gamma_t^2} .
$$

</Caja>

<v-click>

<div class="pequeno">

**Esbozo.** $\ind{y F \le 0} \le e^{-yF}$. Desenrollando los pesos, la pérdida exponencial media es $\prod_t Z_t$, con $Z_t = (1-\varepsilon_t)e^{-\alpha_t} + \varepsilon_t e^{\alpha_t}$. El $\alpha_t$ de AdaBoost **minimiza** $Z_t$, y el mínimo es $2\sqrt{\varepsilon_t(1-\varepsilon_t)}$.

</div>

</v-click>
</div>
<div>

<CotaAdaBoost :ancho="600" :alto="380" />

<div class="pequeno tenue">

Pingüinos: con 200 tocones, cota 0.108 y error 0.029. No llega a 0: hay pingüinos de especies distintas con picos casi iguales, y $\varepsilon_t$ sube hacia 0.5.

</div>
</div>
</div>

<!--
Si cada aprendiz débil tiene una ventaja γ fija sobre la moneda, el error de entrenamiento cae como exp(−2γ²B) tras B rondas: ésa es la respuesta a Kearns y Valiant (para el error de entrenamiento; la generalización se controla aparte).
En la gráfica, la igualdad entre la pérdida exponencial media y el producto se verifica en la libreta a 1e-16.
-->

---

# ¿Qué optimiza AdaBoost?

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 48px">
<div>

Friedman, Hastie y Tibshirani (2000): un **modelo aditivo por etapas**. En cada etapa, sin tocar el modelo $F_{t-1}$ de las rondas anteriores,

$$
(\alpha_t, h_t) = \argmin_{\alpha, h} \sum_i e^{-y^{(i)} (F_{t-1}(\vx^{(i)}) + \alpha h(\vx^{(i)}))} .
$$

<v-click>

Los factores $e^{-y^{(i)} F_{t-1}(\vx^{(i)})}$ **son** los pesos $w^{(i)}$ (normalizados). Separando aciertos y errores, con $\varepsilon$ el error ponderado de $h$:

$$
\sum_i w^{(i)} e^{-\alpha y^{(i)} h(\vx^{(i)})} = e^{-\alpha} + \big(e^{\alpha} - e^{-\alpha}\big)\,\varepsilon .
$$

</v-click>
</div>
<div>

<v-click>

- Para cualquier $\alpha > 0$, la mejor $h$ es la de **menor error ponderado**.
- Derivando en $\alpha$: $e^{2\alpha} = \frac{1 - \varepsilon}{\varepsilon}$, es decir, $\alpha = \frac12 \ln \frac{1 - \varepsilon}{\varepsilon}$.

</v-click>

<v-click>

<div class="pequeno">

Ronda 5 de los pingüinos: el $\alpha$ que minimiza numéricamente la pérdida es 0.514108; la fórmula da **0.514108**.

</div>

</v-click>

<v-click>

<div class="nota">

Si AdaBoost es «sumar modelos por etapas para bajar una pérdida», ¿por qué no usar **cualquier** pérdida? La exponencial castiga muchísimo a los puntos muy mal clasificados: es sensible a etiquetas erróneas.

</div>

</v-click>
</div>
</div>

<!--
Una consecuencia que vale mencionar: el minimizador poblacional de E[exp(−yF)|x] es la mitad del log-momio, así que AdaBoost estima una logística aditiva (ejercicio 2 de la libreta).
-->


---
layout: seccion
numero: 2
---

# *Gradient boosting*

Descenso por gradiente donde los parámetros son funciones.

---
layout: caso
caso: Caso · Bicicletas compartidas
cifra: 17,379
cifraNota: horas de rentas de Capital Bikeshare en Washington, D. C., de enero de 2011 a diciembre de 2012.
fuente: H. Fanaee-T y J. Gama, Progress in Artificial Intelligence 2 (2014); Bike Sharing, UCI (CC BY 4.0).
---

# ¿Cuántas bicicletas se rentarán en esta hora?

Un sistema de bicicletas compartidas necesita anticipar la demanda para repartir bicicletas entre estaciones y programar el mantenimiento.

<v-click>

**Datos:** por cada hora, las bicicletas rentadas $y$ y siete atributos: hora del día, si es día laboral, año, clima (de 1, despejado, a 3, lluvia o nieve), temperatura, humedad y viento.

**Partición:** días 1 a 19 de cada mes para entrenar (10,886 horas); del 20 en adelante para probar (6,493).

</v-click>

<v-click>

**Métrica:** la raíz del error cuadrático medio, en bicicletas por hora,

$$
\text{RECM} = \sqrt{\frac1M \sum_{i=1}^{M} \big(y^{(i)} - \hat y^{(i)}\big)^2} ,
$$

con $M$ las horas de prueba, $y^{(i)}$ las rentas reales y $\hat y^{(i)}$ las predichas. Predecir siempre el promedio da **181.8**.

</v-click>

---

# Con la pérdida cuadrática, el residuo es menos la derivada

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 48px">
<div>

Sea $F$ la predicción actual para un dato con respuesta $y$, y su pérdida cuadrática

$$
\ell(y, F) = \tfrac12 (y - F)^2 \quad\Longrightarrow\quad \frac{\partial \ell}{\partial F} = -(y - F) .
$$

El residuo $y - F$ es **menos la derivada** de la pérdida respecto a la predicción.

<v-clicks>

- Residuo positivo: **subir** $F$ baja la pérdida; negativo: **bajarla**.
- Sumarle a $F$ algo parecido a los residuos es moverse **en contra de la derivada**: un paso de descenso por gradiente.

</v-clicks>
</div>
<div v-click>

<Caja tipo="ejemplo" nombre="días laborales">

Primera predicción: el promedio, $F_0 = 193$ bicicletas. Un dato típico de cuatro horas:

| hora | $y$ | residuo $y - F_0$ | $\partial \ell / \partial F$ |
|---|---|---|---|
| 3:00 | 5 | −188 | +188 |
| 8:00 | 480 | +287 | −287 |
| 13:00 | 197 | +4 | −4 |
| 17:00 | 529 | +336 | −336 |

</Caja>

</div>
</div>

<!--
A las 8 y a las 17 hay que subir mucho la predicción; a las 3, bajarla; a las 13, casi nada.
Los valores de y son los promedios de rentas a esa hora en los días laborales de entrenamiento; F_0 = 193 es el promedio de todas las horas laborales.
Insistir en el signo: la derivada dice hacia dónde sube la pérdida; el residuo, hacia dónde baja. Son la misma información con signo contrario.
-->

---

# Las predicciones como parámetros

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 44px">
<div>

Tomemos como parámetros las $M$ predicciones en los datos, $F^{(i)} = F(\vx^{(i)})$, y minimicemos

$$
L\big(F^{(1)}, \dots, F^{(M)}\big) = \frac1M \sum_{i=1}^{M} \ell\big(y^{(i)}, F^{(i)}\big) .
$$

<v-click>

Cada derivada parcial solo involucra a un dato. Un paso de descenso por gradiente en la ronda $t$, con tasa $\eta$ (que absorbe el $\frac1M$):

$$
\begin{aligned}
F^{(i)}_t &= F^{(i)}_{t-1} - \eta\, \frac{\partial \ell\big(y^{(i)}, F\big)}{\partial F}\bigg|_{F = F^{(i)}_{t-1}} \\
&= F^{(i)}_{t-1} + \eta\, r^{(i)}_t .
\end{aligned}
$$

</v-click>
</div>
<div>

<v-click>

<Caja tipo="definicion" nombre="pseudo-residuo">

$$
r^{(i)}_t = -\frac{\partial \ell\big(y^{(i)}, F\big)}{\partial F}\bigg|_{F = F_{t-1}(\vx^{(i)})}
$$

Hacia dónde y cuánto conviene mover la predicción del dato $i$ en la ronda $t$. Con la pérdida cuadrática es exactamente el residuo $y^{(i)} - F_{t-1}(\vx^{(i)})$; con otras pérdidas no, de ahí el «pseudo».

</Caja>

</v-click>

<v-click>

<div class="nota">

Dos problemas: el paso solo dice qué hacer en los $M$ datos de entrenamiento, no en un $\vx$ nuevo; y con $M$ parámetros libres, repetirlo lleva cada $F^{(i)}$ hasta $y^{(i)}$: **memoriza**.

</div>

</v-click>
</div>
</div>

---

# La idea de Friedman: un árbol que imita al gradiente

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 44px">
<div>

En la ronda $t$, ajustar un **árbol de regresión** $h_t$ con los pares $\big(\vx^{(i)}, r^{(i)}_t\big)$ como datos, de modo que $h_t(\vx^{(i)}) \approx r^{(i)}_t$ en cada dato. Después se avanza:

$$
F_t(\vx) = F_{t-1}(\vx) + \eta\, h_t(\vx) .
$$

<v-clicks>

- **En los datos**, es el paso de gradiente: $F_t(\vx^{(i)}) \approx F_{t-1}(\vx^{(i)}) + \eta\, r^{(i)}_t$.
- **En un $\vx$ nuevo**, la corrección es la de la hoja a la que llega.
- **Regulariza**: una hoja da a todos sus datos la misma corrección, el promedio de sus $r^{(i)}_t$; no persigue el ruido de cada uno.

</v-clicks>
</div>
<div v-click>

<Caja tipo="algoritmo" nombre="gradient boosting (Friedman, 2001)">

1. $F_0 = \argmin_c \sum_{i} \ell(y^{(i)}, c)$: la mejor constante (el promedio, con la cuadrática).
2. Para $t = 1, \dots, B$ (número de árboles):
   1. pseudo-residuos $r^{(i)}_t$, $i = 1, \dots, M$;
   2. ajustar el árbol $h_t$ a los pares $\big(\vx^{(i)}, r^{(i)}_t\big)$;
   3. $F_t = F_{t-1} + \eta\, h_t$, con $\eta \in (0, 1]$ la tasa de aprendizaje.

</Caja>
</div>
</div>

---

# Cada tocón corrige lo que falta

<BoostingEtapas />

<!--
Bicicletas, solo días laborales de entrenamiento, un solo atributo: la hora. Tocones (árboles de una sola pregunta) y η = 0.5.
Avanzar t: 0 (solo el promedio), 1, 2, 3... A la derecha están los pseudo-residuos r = y − F_t y, en rojo, el tocón h_{t+1} que se les ajusta: es el árbol que imita al gradiente. Siempre corta donde los residuos cambian de signo en promedio. Ese tocón, multiplicado por η, se suma a la curva azul de la izquierda.
Con 300 tocones, la suma de escalones casi coincide con el promedio por hora (puntos dorados): es lo mejor que se puede con pérdida cuadrática y un atributo discreto.
-->

---

# Otras pérdidas, el mismo algoritmo

<div class="cols" style="grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 36px">
<div>

| pérdida | $\ell(y, F)$ | $r = -\partial \ell / \partial F$ |
|---|---|---|
| cuadrática | $\frac12 (y - F)^2$ | $y - F$ |
| absoluta | $\lvert y - F \rvert$ | $\operatorname{signo}(y - F)$ |
| Poisson | $e^{F} - yF$ | $y - e^{F}$ |
| logística | $\ln(1 + e^{F}) - yF$ | $y - p$ |
| exponencial | $e^{-yF}$ | $y\, e^{-yF}$ |


</div>
<div>

<v-clicks>

- Solo cambia el paso 2.1: el árbol siempre se ajusta por mínimos cuadrados a los $r^{(i)}_t$.
- Sin la cuadrática, el promedio de los $r$ de una hoja no es el mejor valor: Friedman **reoptimiza cada hoja** (mediana con la absoluta, paso de Newton con la logística). Generalizada, esa idea es XGBoost.

</v-clicks>

<div class="pequeno">

**Absoluta:** regresión robusta a atípicos. **Poisson:** conteos; $F$ es el logaritmo de la media $\mu = e^{F}$. **Logística:** clases $y \in \{0, 1\}$; $F$ es el log-momio y $p = \sigma(F) = 1/(1 + e^{-F})$ la probabilidad. **Exponencial:** la de AdaBoost, con $y \in \{-1, +1\}$.

</div>
</div>
</div>

---

# *Gradient boosting* desde cero

<div class="cols codigo-chico" style="grid-template-columns: 6fr 5fr; gap: 36px">
<div>

```python
def gradient_boosting(X, y, B, eta, profundidad):
    F = np.full(len(y), y.mean())       # F_0: el promedio
    arboles = []
    for t in range(B):
        r = y - F      # pseudo-residuos: -dℓ/dF, pérdida cuadrática
        h = DecisionTreeRegressor(max_depth=profundidad).fit(X, r)
        F += eta * h.predict(X)          # F_t = F_{t-1} + η h_t
        arboles.append(h)
    return arboles
```

<div class="pequeno tenue" style="margin-top: 14px">

$B = 300$ árboles de profundidad 4, $\eta = 0.1$. Contra `GradientBoostingRegressor` de scikit-learn: diferencia máxima de $1.3 \times 10^{-12}$ bicicletas.

</div>
</div>
<div v-click>

RECM de prueba, mismos siete atributos:

| modelo | RECM |
|---|---|
| predecir el promedio | 181.8 |
| regresión lineal | 143.2 |
| un árbol de profundidad 10 | 70.2 |
| *gradient boosting*, cuadrática | **59.9** |
| `HistGradientBoosting`, Poisson | **56.5** |

</div>
</div>

<!--
La regresión lineal no puede representar los dos picos del día laboral con la hora como número. Los árboles de profundidad 4 combinan hora, tipo de día y temperatura en una misma hoja: encuentran solos las interacciones. Para estos conteos, la pérdida de Poisson baja un poco más el error. HistGradientBoosting es la versión con histogramas de scikit-learn (sección 4).
Para que coincidan también los desempates, en la libreta se le pasa a cada árbol el mismo generador aleatorio que usa scikit-learn.
-->

---
layout: seccion
numero: 3
---

# Una campaña de mercadeo bancario

Seis modelos, una fuga de información y la parada temprana.

---
layout: caso
caso: Caso · Mercadeo telefónico
cifra: 11.7 %
cifraNota: de 45,211 llamadas de un banco portugués (mayo de 2008 a noviembre de 2010) terminaron en la contratación de un depósito a plazo.
fuente: S. Moro, P. Cortez y P. Rita, Decision Support Systems 62 (2014); Bank Marketing, UCI (CC BY 4.0).
---

# ¿A quién llamar primero?

Un centro de llamadas ofrece depósitos a plazo. Cada llamada cuesta tiempo de un ejecutivo y paciencia del cliente, y casi nueve de cada diez dicen que no.

<v-click>

**Datos:** edad, trabajo, estado civil, escolaridad, saldo, hipoteca, préstamo; día, mes, tipo de contacto y **duración** de la llamada; cuántas veces se le llamó y el resultado de campañas anteriores.

**Problema:** antes de marcar, estimar la probabilidad de que el cliente contrate, para ordenar la lista. Partición estratificada 75/25.

</v-click>

<v-click>

**Métrica:** la **AUC**, el área bajo la curva ROC. Es la probabilidad de que, al tomar un cliente que contrata y uno que no, el modelo le dé mayor probabilidad al primero: 0.5 es ordenar al azar; 1, un orden perfecto.

</v-click>

---

# Una AUC demasiado buena

<div class="cols" style="grid-template-columns: 6fr 5fr; gap: 40px">
<div>

<FugaDuracion />

</div>
<div>

Con todos los atributos, AUC de prueba:

| modelo | AUC |
|---|---|
| regresión logística | 0.905 |
| `HistGradientBoosting` | **0.934** |
| solo la duración | 0.800 |

<v-click>

<div class="nota">

¿Cuándo se conoce la duración de una llamada? **Al colgar**, cuando ya se sabe la respuesta. Es una **fuga de información**: un atributo que no se tendrá al momento de predecir. La documentación de UCI pide descartarla «si la intención es tener un modelo predictivo realista».

</div>

</v-click>
</div>
</div>

<!--
Dejar que el grupo se entusiasme con el 0.93 antes de preguntar cuándo se conoce cada columna. Quien contrata habla más (hay que tomar sus datos); quien no, cuelga pronto. Las 3 llamadas de duración 0 son todas «no». La pregunta que hay que hacerse ante cada columna: ¿la tendré en el momento de decidir?
-->

---

# Seis modelos sin trampa

<div class="cols" style="grid-template-columns: 6fr 5fr; gap: 40px">
<div>

| modelo | AUC | captura al 20 % |
|---|---|---|
| regresión logística | 0.773 | 55.6 % |
| bosque aleatorio | 0.802 | 61.8 % |
| `HistGradientBoosting` | 0.804 | 62.8 % |
| XGBoost | 0.808 | 62.9 % |
| LightGBM | 0.806 | 62.9 % |
| CatBoost | **0.810** | **63.2 %** |

<div class="pequeno tenue">

Sin la duración. **Captura:** de los que contratan, cuántos quedan en el 20 % de la lista con mayor probabilidad (al azar, 20 %). **Bosque:** 500 árboles profundos con remuestras.

</div>
</div>
<div>

<v-clicks>

- Los cuatro *boostings* empatan: las milésimas entre ellos cambiarían con otra partición.
- Sobre la logística, siete puntos más de captura: cientos de clientes en una campaña.
- Todos entrenan en segundos o menos; CatBoost es el más lento (unos 6 s).

</v-clicks>

<div v-click class="nota">

Lo más importante para el modelo es el **mes** y el tipo de **contacto**: identifican *qué campaña* era. Las filas están en orden temporal; entrenar con campañas anteriores y probar con las siguientes sería más honesto.

</div>
</div>
</div>

<!--
XGBoost, LightGBM y CatBoost con tasa 0.05 y el número de árboles por parada temprana en un 20 % de validación (365, 132 y 557 árboles), reentrenados con todo el entrenamiento. XGBoost con profundidad 4, LightGBM con 15 hojas, CatBoost con profundidad 6 (su valor por omisión). HistGradientBoosting con sus valores por omisión.
Captura al 10 %: todos alrededor del 42–44 %; los clientes más obvios (los que ya contrataron en una campaña anterior) son fáciles para cualquier modelo. La diferencia aparece más abajo en la lista.
Importancia por permutación en prueba (caída de AUC): mes 0.089, contacto 0.065, resultado previo 0.027, día 0.024. Las llamadas con contacto «unknown» son casi todas de la primera campaña.
-->

---

# Tasa de aprendizaje y parada temprana

<div class="cols" style="grid-template-columns: 7fr 4fr; gap: 30px">
<div>

<CurvasTasa v-if="$clicks < 1" :entrena="false" />
<CurvasTasa v-else />

<div class="pequeno tenue" style="margin-top: 10px">

XGBoost, profundidad 4. Error: entropía cruzada, $-\frac1M\sum_i \big[y^{(i)} \ln p^{(i)} + (1 - y^{(i)}) \ln(1 - p^{(i)})\big]$, con $p^{(i)}$ la probabilidad estimada. Mínimos: 0.2925 (33 árboles, $\eta = 0.3$) y 0.2899 (175 con $\eta = 0.1$; 406 con $\eta = 0.03$).

</div>
</div>
<div>

<v-clicks>

- **El *boosting* sobreajusta**: el error de entrenamiento baja sin fin; el de validación tiene un mínimo.
- $\eta$ chica: más árboles, pero una curva plana cerca del mínimo.
- **Parada temprana**: detenerse en el mínimo de validación. Es regularización.

</v-clicks>
</div>
</div>

<!--
Primero solo las curvas de validación. Con el primer clic aparecen las de entrenamiento (punteadas): siempre bajan.
A diferencia de un bosque, donde más árboles nunca empeoran el error esperado, aquí cada árbol persigue los residuos que quedan y tarde o temprano persigue ruido. Regla práctica: η tan chica como el tiempo permita y el número de árboles por parada temprana.
-->

---
layout: seccion
numero: 4
---

# XGBoost, LightGBM y CatBoost por dentro

Segundo orden, histogramas y categorías sin fugas.

---

# XGBoost: usar también la curvatura

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 44px">
<div>

El *gradient boosting* solo usa la **pendiente** de la pérdida. XGBoost (Chen y Guestrin, 2016) usa también la **curvatura**. Si a la predicción actual $F$ de un dato le sumamos una corrección $w$, por Taylor:

$$
\ell(y, F + w) \approx \ell(y, F) + g\, w + \tfrac12 h\, w^2 ,
$$

con $g = \partial \ell / \partial F$ y $h = \partial^2 \ell / \partial F^2$, ambas evaluadas en $F$.

<v-click>

Es una parábola en $w$, con mínimo en $w^* = -g/h$: un **paso de Newton**. La pendiente dice hacia dónde moverse; la curvatura, cuánto.

</v-click>
</div>
<div v-click>

| pérdida | $g$ | $h$ |
|---|---|---|
| cuadrática | $F - y$ | $1$ |
| logística | $p - y$ | $p(1 - p)$ |
| Poisson | $e^{F} - y$ | $e^{F}$ |

<div class="pequeno">

$p = \sigma(F)$, la probabilidad estimada. Con la cuadrática, $w^* = y - F$: el residuo completo. Con la logística, la curvatura $p(1 - p)$ es casi cero cuando el modelo ya está muy seguro ($p$ cerca de 0 o de 1); si además se equivoca, $g$ es cercano a $\pm 1$ y el paso de Newton es enorme. Por eso XGBoost lo regulariza.

</div>
</div>
</div>

---

# El objetivo de cada ronda

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 44px">
<div>

Un árbol con $J$ hojas es constante por pedazos: $f(\vx) = w_{q(\vx)}$, con $q(\vx) \in \{1, \dots, J\}$ la hoja a la que llega $\vx$ y $w_q$ su valor. En la ronda $t$, XGBoost elige el árbol $f$ que minimiza

$$
\sum_{i=1}^{M} \ell\big(y^{(i)}, F_{t-1}(\vx^{(i)}) + f(\vx^{(i)})\big) + \Omega(f) ,
$$

$$
\Omega(f) = \gamma J + \tfrac{\lambda}{2} \textstyle\sum_{q=1}^{J} w_q^2 .
$$

<div class="pequeno">

$\gamma$: precio de cada hoja, castiga los árboles grandes. $\lambda$: encoge los $w_q$ hacia 0, frena los pasos enormes.

</div>
</div>
<div v-click>

Con Taylor alrededor de $F_{t-1}(\vx^{(i)})$ en cada dato, y quitando los términos que no dependen de $f$, queda:

$$
\sum_{i=1}^{M} \Big[ g^{(i)} f(\vx^{(i)}) + \tfrac12 h^{(i)} f(\vx^{(i)})^2 \Big] + \Omega(f) ,
$$

con $g^{(i)}$ y $h^{(i)}$ la primera y la segunda derivadas de la pérdida del dato $i$ en $F_{t-1}(\vx^{(i)})$.

<div class="nota">

Se calculan una vez por ronda; después, construir el árbol solo requiere **sumar** $g$ y $h$. Sirve para cualquier pérdida con dos derivadas.

</div>
</div>
</div>

---

# El valor óptimo de cada hoja

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 44px">
<div>

Sea $I_q = \{i : q(\vx^{(i)}) = q\}$, los datos que llegan a la hoja $q$. Todos reciben el mismo valor $w_q$; con las sumas

$$
G_q = \sum_{i \in I_q} g^{(i)}, \qquad H_q = \sum_{i \in I_q} h^{(i)} ,
$$

el objetivo se separa en una parábola por hoja:

$$
\sum_{q=1}^{J} \Big[ G_q w_q + \tfrac12 (H_q + \lambda)\, w_q^2 \Big] + \gamma J .
$$

</div>
<div>

<v-click>

Derivando cada parábola respecto a $w_q$ e igualando a cero:

$$
w_q^* = -\frac{G_q}{H_q + \lambda} .
$$

</v-click>

<v-clicks>

- **Cuadrática:** $h^{(i)} = 1$, $H_q$ es el número de datos de la hoja y $-G_q$ la suma de sus residuos: $w_q^*$ es su **promedio, encogido** por $\lambda$.
- **Logística con $\lambda = 0$:** un paso de Newton por hoja.
- Con $w_q^*$, cada hoja baja el objetivo en $\frac12 G_q^2 / (H_q + \lambda)$: con eso se comparan árboles.

</v-clicks>
</div>
</div>

---

# La ganancia de una división

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

Una hoja se parte en izquierda y derecha, con sumas $(G_I, H_I)$ y $(G_D, H_D)$. El objetivo mínimo antes de partirla, menos el de después:

$$
\tfrac12 \left[ \frac{G_I^2}{H_I + \lambda} + \frac{G_D^2}{H_D + \lambda} - \frac{(G_I + G_D)^2}{H_I + H_D + \lambda} \right] - \gamma .
$$

<v-clicks>

- Se calcula para cada atributo y umbral, y gana la mayor: es el criterio de división, en lugar de una impureza.
- Si ninguna es positiva, la hoja no se divide: **poda incorporada**.

</v-clicks>
</div>
<div v-click>

<Caja tipo="ejemplo" nombre="verificación con XGBoost">

<div class="pequeno">

Banco, un árbol de profundidad 1 sobre la edad, logística, $\lambda = 1$, desde $p = 0.5$: $g^{(i)} = 0.5 - y^{(i)}$ y $h^{(i)} = 0.25$.

| hoja | datos | contratan | $G$ | $H$ | $w^*$ |
|---|---|---|---|---|---|
| < 61 | 32,996 | 3,588 | 12,910 | 8,249 | −1.565 |
| ≥ 61 | 912 | 379 | 77 | 228 | −0.336 |

Ganancia a mano: **167.0**. XGBoost reporta 334.0 (omite el $\frac12$) y las mismas hojas.

</div>

</Caja>

</div>
</div>

<!--
H es la cuarta parte de los datos de la hoja y G la mitad de los datos menos los que contratan.
Las dos hojas son negativas porque partimos de p = 0.5 y solo el 11.7 % contrata; la de 61 años o más baja mucho menos el log-momio porque ahí contrata el 42 %, contra el 11 % de los menores. Por eso en la práctica se parte del log-momio de la prevalencia.
-->

---

# Los hiperparámetros de XGBoost, en la fórmula

<div class="cols" style="grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: 40px">
<div class="pequeno">

| hiperparámetro | qué controla |
|---|---|
| `learning_rate` | la tasa $\eta$ en $F_t = F_{t-1} + \eta f_t$ |
| `n_estimators` | las rondas $B$ (mejor, con parada temprana) |
| `max_depth` | la profundidad de cada árbol |
| `reg_lambda` | $\lambda$: encoge los $w_q$ |
| `gamma` | $\gamma$: ganancia mínima para dividir |
| `min_child_weight` | el $H_q$ mínimo de una hoja |
| `subsample`, `colsample_bytree` | fracción de datos y de atributos que ve cada árbol |

</div>
<div>

<v-clicks>

- **Faltantes:** cada división prueba mandarlos a la izquierda y a la derecha, y se queda con la de mayor ganancia. No hace falta imputar.
- **Submuestreo:** cada árbol ve otra parte de los datos y atributos: menos varianza y más rapidez, como en un bosque aleatorio.

</v-clicks>
</div>
</div>

<!--
Con la pérdida cuadrática, h = 1 y min_child_weight es el número mínimo de datos por hoja.
En el banco usamos max_depth=4, learning_rate=0.05 y parada temprana; lo demás con sus valores por omisión (λ = 1, γ = 0).
-->

---

# Histogramas: buscar divisiones sin ordenar

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 44px">
<div>

El algoritmo **exacto** ordena cada atributo y prueba cada umbral entre valores distintos: hasta $M - 1$ por atributo, en cada nodo de cada árbol.

<v-click>

Con **histogramas** (LightGBM, Ke et al., 2017):

1. Antes de entrenar, cada atributo se parte una vez en ≤ 255 intervalos (por cuantiles).
2. En cada nodo, una pasada por los datos suma $g$ y $h$ en cada intervalo.
3. Solo se prueban los bordes; $G_I$ y $H_I$ son sumas acumuladas.

</v-click>
</div>
<div>

<v-clicks>

- El saldo del banco toma **6,421** valores distintos; con histogramas, a lo más 255 umbrales.
- El histograma de una hija es el del padre menos el de su hermana: solo se calcula el de la más chica.
- Con millones de filas, es la diferencia entre segundos y horas.
- Hoy lo usan también XGBoost (`tree_method="hist"`, por omisión) y scikit-learn (`HistGradientBoosting`).

</v-clicks>
</div>
</div>

<!--
En la libreta, con 34,000 llamadas y 100 árboles de profundidad 4, el gradient boosting «exacto» de scikit-learn tarda varias veces más que HistGradientBoosting (los tiempos dependen de la máquina).
Perder precisión en los umbrales casi no cuesta: la tasa de aprendizaje y la suma de cientos de árboles suavizan la diferencia.
-->

---

# LightGBM: crecer por hojas y muestrear por gradiente

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

<div style="display: flex; gap: 8px; justify-content: center">
<FormasArbol tipo="niveles" :ancho="270" />
<FormasArbol tipo="hojas" :ancho="270" />
</div>

<div class="pequeno">

Tres divisiones, numeradas en orden. **Por hojas** (LightGBM) se divide siempre la hoja de mayor ganancia, hasta `num_leaves` hojas: con las mismas hojas baja más la pérdida, pero puede hacer ramas muy profundas.

</div>
</div>
<div v-click>

**GOSS:** un dato con $\lvert g^{(i)} \rvert$ chico ya está bien ajustado y aporta poco a la ganancia. En cada ronda:

1. se conserva el $a$ % de los datos con mayor $\lvert g^{(i)} \rvert$;
2. de los demás se sortea un $b$ %, con $g$ y $h$ multiplicados por $(1 - a)/b$ para no sesgar $G$ y $H$.

<div class="pequeno">

Con $a = 20$ % y $b = 10$ %, cada árbol usa el 30 % de los datos. **EFB:** junta en uno los atributos ralos que nunca son distintos de cero a la vez, como las columnas de un *one-hot*.

</div>
</div>
</div>

<!--
GOSS: gradient-based one-side sampling. Se activa con data_sample_strategy="goss". Las ramas profundas del crecimiento por hojas se limitan con num_leaves y min_child_samples.
XGBoost crece por niveles por omisión (grow_policy="depthwise").
-->

---

# CatBoost: el problema de las categorías

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

CatBoost (Prokhorenkova et al., 2018) nació para categóricas con muchos valores, como un código postal: con *one-hot* serían miles de columnas. La alternativa es cambiar cada categoría por la **proporción de positivos** que tiene en entrenamiento:

$$
\hat x^{(i)} = \frac{\sum_{j:\, c^{(j)} = c^{(i)}} y^{(j)} + a\, \bar p}{\#\{j : c^{(j)} = c^{(i)}\} + a}
$$

<div class="pequeno">

$c^{(i)}$: categoría del dato $i$; $\bar p$: proporción global de positivos; $a > 0$: cuánto pesa $\bar p$ cuando la categoría tiene pocos datos.

</div>
</div>
<div>

<v-click>

<div class="nota">

La suma incluye a $j = i$: **la codificación contiene la propia respuesta**. Con un solo dato en la categoría, $\hat x^{(i)} = (y^{(i)} + a \bar p)/(1 + a)$.

</div>

</v-click>

<v-click>

<Caja tipo="ejemplo" nombre="una columna sin información">

<div class="pequeno">

Al banco le agregamos un identificador **al azar** (7,888 valores) codificado así. `HistGradientBoosting`, AUC:

| | entrenamiento | prueba |
|---|---|---|
| sin el identificador | 0.859 | 0.804 |
| con el identificador | 0.959 | **0.539** |

</div>

</Caja>

</v-click>
</div>
</div>

<!--
En entrenamiento la columna «predice»; en prueba no dice nada, y el modelo confió en ella. Es una fuga de información creada por el preprocesamiento.
Con a = 1 y la proporción global de 11.7 %, un cliente que contrató y es el único de su identificador queda codificado como 0.56; uno que no, como 0.06.
-->

---

# CatBoost: estadísticas ordenadas

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 40px">
<div>

Se barajan los datos de entrenamiento al azar, y cada dato se codifica **solo con los que van antes** que él:

$$
\hat x^{(i)} = \frac{\sum_{j \prec i,\; c^{(j)} = c^{(i)}} y^{(j)} + a\, \bar p}{\#\{j \prec i : c^{(j)} = c^{(i)}\} + a}
$$

$j \prec i$: $j$ va antes que $i$ en ese orden. La respuesta del propio dato nunca entra. En prueba se usan todos los datos de entrenamiento.

<v-click>

<div class="pequeno">

**Boosting ordenado:** la misma idea con los pseudo-residuos; el $r^{(i)}$ de un dato se calcula con un modelo que no lo vio (`boosting_type="Ordered"`, útil con pocos datos).

</div>

</v-click>
</div>
<div v-click>

<Caja tipo="ejemplo" nombre="el mismo identificador al azar">

AUC de prueba en el banco:

| | AUC |
|---|---|
| sin el identificador | 0.804 |
| codificación con todos los datos | **0.539** |
| codificación ordenada | 0.805 |
| CatBoost, como categoría | 0.806 |

Con la codificación ordenada, la columna sin información ya no engaña al modelo.

</Caja>

</div>
</div>

<!--
Las tres primeras filas usan HistGradientBoosting; la última, CatBoost con el identificador entre sus categorías.
El precio: los primeros datos del orden se codifican con muy poca información. CatBoost usa varios órdenes al azar para promediar ese efecto.
-->

---

# CatBoost: árboles simétricos

<div class="cols" style="grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 44px">
<div>

<FormasArbol tipo="simetrico" :ancho="480" />

</div>
<div>

En cada nivel, **la misma pregunta** en todos los nodos. Un árbol de profundidad $d$ es una lista de $d$ preguntas, y la hoja de $\vx$ son sus $d$ respuestas sí/no: un número binario entre 0 y $2^d - 1$.

<v-click>

<div class="pequeno">

Con el árbol de la izquierda, un dato con $x_1 = 3$ y $x_3 = 4$ responde «sí» y «no»: hoja $10$.

</div>

</v-click>

<v-clicks>

- **Predicción muy rápida:** $d$ comparaciones, sin recorrer ramas.
- **Árboles más restringidos:** sobreajustan menos y funcionan bien con los valores por omisión.

</v-clicks>
</div>
</div>

<!--
CatBoost usa profundidad 6 por omisión: 64 hojas por árbol. También crea combinaciones de categorías (por ejemplo, trabajo × mes) codificadas con la misma estadística ordenada.
-->

---

# Tres bibliotecas, la misma idea

<div class="pequeno">

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| origen | 2016, Chen y Guestrin | 2017, Microsoft (Ke et al.) | 2018, Yandex (Prokhorenkova et al.) |
| forma del árbol | por niveles | por hojas | simétrico |
| divisiones | histogramas o exacta | histogramas, GOSS, EFB | histogramas |
| categóricas | nativas | nativas | estadísticas ordenadas |
| destaca por | regularización, faltantes | rapidez con muchos datos | categóricas, valores por omisión |
| banco: AUC de prueba | 0.808 | 0.806 | 0.810 |

</div>

<v-click>

<div class="nota" style="margin-top: 18px">

Las tres usan las derivadas $g$ y $h$, hojas $w^* = -G/(H + \lambda)$ y la ganancia de segundo orden. En el banco quedaron a milésimas: importan más la tasa chica con parada temprana y revisar qué atributos se tendrán al decidir.

</div>

</v-click>

<!--
Las «categóricas nativas» de XGBoost (enable_categorical), LightGBM y HistGradientBoosting parten las categorías en dos grupos ordenándolas por G/H, sin one-hot.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 22px; line-height: 1.35">

1. *Boosting*: un modelo aditivo **por etapas**; cada término corrige a los anteriores. Reduce sesgo, no varianza.
2. AdaBoost = pérdida exponencial; su error de entrenamiento cae como $\prod_t 2\sqrt{\varepsilon_t(1 - \varepsilon_t)}$, con $\varepsilon_t$ el error ponderado de la ronda $t$.
3. *Gradient boosting*: descenso por gradiente sobre las predicciones. Un árbol imita los **pseudo-residuos** $r = -\partial \ell/\partial F$ y $F_t = F_{t-1} + \eta h_t$.
4. XGBoost: segundo orden y penalización, $w^* = -G/(H + \lambda)$. LightGBM: histogramas y crecimiento por hojas. CatBoost: categorías codificadas sin su propia respuesta.
5. Tasa chica y **parada temprana**. Ningún algoritmo protege de una **fuga**: ¿tendré este dato al decidir?

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Boosting y una campaña de mercadeo bancario* (Unidad 5, ejemplo 3).

**Laboratorio 5:** árboles y *boosting* desde cero.

**Lectura:** *The Elements of Statistical Learning*, cap. 10; Chen y Guestrin, «XGBoost», *KDD* (2016); Ke et al., «LightGBM», *NeurIPS* (2017); Prokhorenkova et al., «CatBoost: unbiased boosting with categorical features», *NeurIPS* (2018).

</div>
</div>
</div>
