---
theme: none
title: "Boosting: de AdaBoost a XGBoost"
unidad: Unidad 5
layout: portada
subtitulo: Sumar modelos débiles, uno tras otro, para corregir los errores de los anteriores
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
  Reconocimiento de Patrones · Unidad 5 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# *Boosting*: de AdaBoost a XGBoost

---
layout: pregunta
---

# Tienes un clasificador que acierta apenas un poco más que una moneda. ¿Puedes convertirlo en uno tan bueno como quieras?

<v-click>

Kearns y Valiant (1988–89) lo plantearon como problema abierto de la teoría PAC. Schapire (1990) demostró que **sí**; Freund y Schapire (1997) lo volvieron práctico con **AdaBoost**, y recibieron el premio Gödel en 2003. La idea: entrenar muchos clasificadores débiles **en secuencia**, cada uno concentrado en lo que los anteriores hicieron mal.

</v-click>

<!--
Pedir ideas antes de mostrar la respuesta. Suelen proponer «promediar muchos», como el bagging de la clase pasada. Hacer notar la diferencia: si todos los clasificadores débiles se equivocan en los mismos casos, promediarlos no ayuda. Hay que obligar a cada uno a mirar donde los otros fallan.
-->

---

# AdaBoost: concentrarse en los errores

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="algoritmo" nombre="AdaBoost">

Etiquetas $y_n \in \{-1, +1\}$, pesos $w_n = 1/N$. Para $t = 1, \dots, T$:

1. ajustar $h_t$ a los datos con pesos $w$;
2. $\varepsilon_t = \sum_n w_n\, \ind{h_t(\vx_n) \ne y_n}$;
3. $\alpha_t = \frac12 \ln \frac{1 - \varepsilon_t}{\varepsilon_t}$;
4. $w_n \leftarrow w_n\, e^{-\alpha_t y_n h_t(\vx_n)}$ y normalizar.

Clasificador final:

$$
g(\vx) = \operatorname{signo}\big(F_T(\vx)\big), \qquad F_T = \sum_t \alpha_t h_t .
$$

</Caja>

</div>
<div>

<v-clicks>

- $y_n h_t(\vx_n) = +1$ si acertó: el peso se multiplica por $e^{-\alpha_t} < 1$. Si falló, por $e^{\alpha_t} > 1$.
- Un clasificador mejor que la moneda ($\varepsilon_t < \frac12$) recibe voto $\alpha_t > 0$; uno casi perfecto, un voto enorme.
- **Aprendiz débil** típico: un **tocón**, un árbol de una sola pregunta $x_j > u$. Malo solo, con mucho sesgo y poca varianza.

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

Con $\varepsilon_t = \frac12 - \gamma_t$, el error de entrenamiento cumple

$$
\frac1N \sum_n \ind{g(\vx_n) \ne y_n} \le \frac1N\sum_n e^{-y_n F_T(\vx_n)}
$$

$$
= \prod_{t} 2\sqrt{\varepsilon_t(1 - \varepsilon_t)} \le e^{-2\sum_t \gamma_t^2} .
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
Si cada aprendiz débil tiene una ventaja γ fija sobre la moneda, el error de entrenamiento cae como exp(−2γ²T): ésa es la respuesta a Kearns y Valiant (para el error de entrenamiento; la generalización se controla aparte, como en la Unidad 3).
En la gráfica, la igualdad entre la pérdida exponencial media y el producto se verifica en la libreta a 1e-16.
-->

---

# ¿Qué optimiza AdaBoost?

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Friedman, Hastie y Tibshirani (2000): un **modelo aditivo por etapas**. En cada etapa, sin tocar lo anterior,

$$
(\alpha_t, h_t) = \argmin_{\alpha, h} \sum_n e^{-y_n (F_{t-1}(\vx_n) + \alpha h(\vx_n))} .
$$

<v-click>

Los factores $e^{-y_n F_{t-1}(\vx_n)}$ **son** los pesos $w_n$. Separando aciertos y errores:

$$
\sum_n w_n e^{-\alpha y_n h(\vx_n)} = e^{-\alpha} + \big(e^{\alpha} - e^{-\alpha}\big)\,\varepsilon .
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

# Descenso por gradiente en el espacio de funciones

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 40px">
<div>

Minimizar $\sum_n L\big(y_n, F(\vx_n)\big)$ tratando los valores $F(\vx_n)$ como parámetros:

$$
r_{nm} = -\frac{\partial L(y_n, F)}{\partial F}\Big|_{F = F_{m-1}(\vx_n)} .
$$

<v-click>

Los **pseudo-residuos** solo dicen qué hacer en los datos. Friedman (2001): **ajustarles un árbol** $h_m$, que generaliza la dirección a todo $\vx$, y avanzar:

$$
F_m(\vx) = F_{m-1}(\vx) + \eta\, h_m(\vx) .
$$

</v-click>
</div>
<div v-click>

| pérdida | $L(y, F)$ | pseudo-residuo |
|---|---|---|
| cuadrática | $\frac12 (y - F)^2$ | $y - F$ |
| absoluta | $\lvert y - F \rvert$ | $\operatorname{signo}(y - F)$ |
| logística | $\ln(1 + e^{F}) - yF$ | $y - p$ |
| exponencial | $e^{-yF}$ | $y\, e^{-yF}$ |

<div class="pequeno tenue">

Logística con $y \in \{0, 1\}$, $F$ el log-momio y $p = \theta(F)$: «etiqueta menos probabilidad», el mismo residuo de los GLM de la Unidad 4. $\eta$ es la **tasa de aprendizaje**.

</div>
</div>
</div>

<!--
Con la pérdida cuadrática, el algoritmo es: cada árbol se ajusta a lo que los anteriores no explicaron. Con la absoluta solo importa el signo del residuo, así que un atípico no jala más que cualquier otro punto. Para pérdidas que no son la cuadrática, el valor de cada hoja se reoptimiza (mediana para la absoluta, un paso de Newton para la logística); eso lleva directo a XGBoost.
-->

---

# Cada tocón corrige lo que falta

<BoostingEtapas />

<!--
Bicicletas de Washington (Unidad 4), solo días laborales de entrenamiento, un solo atributo: la hora. Tocones y η = 0.5.
Avanzar m: 0 (solo el promedio), 1, 2, 3... A la derecha, los residuos y el tocón rojo que se ajusta en la siguiente ronda: siempre corta donde los residuos cambian de signo en promedio. Con 300 tocones, la suma de escalones casi coincide con el promedio por hora (puntos dorados): es lo mejor que se puede con pérdida cuadrática y un atributo discreto.
-->

---

# *Gradient boosting* desde cero

<div class="cols codigo-chico" style="grid-template-columns: 6fr 5fr; gap: 36px">
<div>

```python
def gradient_boosting(X, y, M, eta, profundidad):
    F = np.full(len(y), y.mean())
    arboles = []
    for m in range(M):
        r = y - F                        # pseudo-residuos
        h = DecisionTreeRegressor(max_depth=profundidad).fit(X, r)
        F += eta * h.predict(X)
        arboles.append(h)
    return arboles
```

<div class="pequeno tenue" style="margin-top: 14px">

300 árboles de profundidad 4, $\eta = 0.1$. Contra `GradientBoostingRegressor`: diferencia máxima de $1.3 \times 10^{-12}$ bicicletas.

</div>
</div>
<div v-click>

Bicicletas de Washington, RECM de prueba con los atributos crudos (sin interacciones a mano):

| modelo | RECM |
|---|---|
| M4 lineal (Unidad 4) | 86.4 |
| M4 Poisson (Unidad 4) | 72.2 |
| *gradient boosting*, cuadrática | **59.9** |
| `HistGradientBoosting`, Poisson | **56.5** |

</div>
</div>

<!--
Los árboles de profundidad 4 combinan hora, tipo de día y temperatura en una misma hoja: descubren solos la interacción que en la Unidad 4 hubo que construir a mano. Y la lección de la Unidad 4 sigue valiendo: con pérdida de Poisson baja un poco más.
Para que coincidan también los desempates, en la libreta se le pasa a cada árbol el mismo generador aleatorio que usa scikit-learn.
-->

---
layout: seccion
numero: 3
---

# Una campaña de mercadeo bancario

Cinco modelos, una fuga de información y la parada temprana.

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

**Problema:** antes de marcar, estimar la probabilidad de que el cliente contrate, para ordenar la lista. Métrica: **AUC**. Partición estratificada 75/25.

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
| logística | 0.905 |
| `HistGradientBoosting` | **0.934** |
| solo la duración | 0.800 |

<v-click>

<div class="nota">

¿Cuándo se conoce la duración de una llamada? **Al colgar**, cuando ya se sabe la respuesta. Es una **fuga de información** (Unidad 1); la documentación de UCI pide descartarla «si la intención es tener un modelo predictivo realista».

</div>

</v-click>
</div>
</div>

<!--
Dejar que el grupo se entusiasme con el 0.93 antes de preguntar cuándo se conoce cada columna. Quien contrata habla más (hay que tomar sus datos); quien no, cuelga pronto. Las 3 llamadas de duración 0 son todas «no». La pregunta que hay que hacerse ante cada columna: ¿la tendré en el momento de decidir?
-->

---

# Cinco modelos sin trampa

<div class="cols" style="grid-template-columns: 6fr 5fr; gap: 40px">
<div>

| modelo | AUC | captura al 20 % |
|---|---|---|
| logística | 0.773 | 55.6 % |
| bosque aleatorio | 0.802 | 61.8 % |
| `HistGradientBoosting` | 0.804 | 62.8 % |
| XGBoost | **0.808** | 62.9 % |
| LightGBM | 0.806 | 62.9 % |

<div class="pequeno tenue">

Sin la duración. Captura: proporción de los que contratan que están en el 20 % de la lista con mayor probabilidad (al azar sería 20 %).

</div>
</div>
<div>

<v-clicks>

- Los tres *boostings* empatan: las milésimas entre ellos cambiarían con otra partición.
- Sobre la logística: siete puntos más de captura, cientos de clientes en una campaña.
- Todos entrenan en segundos o menos; XGBoost y LightGBM, en menos de uno.

</v-clicks>

<div v-click class="nota">

Lo más importante para el modelo es el **mes** y el tipo de **contacto**: identifican *qué campaña* era. Las filas están en orden temporal; una validación temporal (Unidad 7) sería más honesta.

</div>
</div>
</div>

<!--
XGBoost y LightGBM con tasa 0.05 y el número de árboles por parada temprana en un 20 % de validación (365 y 132 árboles), reentrenados con todo el entrenamiento. HistGradientBoosting con sus valores por omisión.
Captura al 10 %: todos alrededor del 42 %; los clientes más obvios (los que ya contrataron en una campaña anterior) son fáciles para cualquier modelo. La diferencia aparece más abajo en la lista.
Importancia por permutación en prueba (caída de AUC): mes 0.089, contacto 0.065, resultado previo 0.027, día 0.024. Las llamadas con contacto «unknown» son casi todas de la primera campaña.
-->

---

# Tasa de aprendizaje y parada temprana

<div class="cols" style="grid-template-columns: 7fr 4fr; gap: 30px">
<div>

<CurvasTasa v-if="$clicks < 1" :entrena="false" />
<CurvasTasa v-else />

<div class="pequeno tenue" style="margin-top: 10px">

XGBoost, árboles de profundidad 4, en un 20 % de validación. Mínimos: 0.2925 con 33 árboles ($\eta = 0.3$); 0.2899 con 175 ($\eta = 0.1$) y con 406 ($\eta = 0.03$).

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

# XGBoost y LightGBM por dentro

Segundo orden, regularización e histogramas.

---

# El peso óptimo de una hoja

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 44px">
<div>

XGBoost (Chen y Guestrin, 2016) agrega al árbol $h$, con hojas de valores $w_1, \dots, w_J$, una penalización

$$
\Omega(h) = \gamma J + \tfrac{\lambda}{2} \textstyle\sum_j w_j^2
$$

<v-click>

y aproxima la pérdida a **segundo orden** alrededor de $F_{m-1}$, con $g_n = \partial_F L$ y $h_n = \partial^2_F L$:

$$
\sum_n \Big[ g_n h(\vx_n) + \tfrac12 h_n h(\vx_n)^2 \Big] + \Omega(h) .
$$

</v-click>
</div>
<div>

<v-click>

Agrupando por hojas, con $G_j = \sum_{n \in j} g_n$ y $H_j = \sum_{n \in j} h_n$, cada hoja es una parábola independiente:

$$
\sum_{j=1}^{J} \Big[ G_j w_j + \tfrac12 (H_j + \lambda)\, w_j^2 \Big] + \gamma J
$$

</v-click>

<v-click>

$$
w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad \text{pérdida} = -\frac12 \sum_j \frac{G_j^2}{H_j + \lambda} + \gamma J .
$$

</v-click>

<v-click>

<div class="pequeno">

Logística: $g = p - y$, $h = p(1-p)$; con $\lambda = 0$ es un **paso de Newton** por hoja. Cuadrática: $h = 1$, el promedio de los residuos encogido por $\lambda$.

</div>

</v-click>
</div>
</div>

---

# La ganancia de una división

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 44px">
<div>

Si una hoja se divide en dos, con sumas $(G_I, H_I)$ a la izquierda y $(G_D, H_D)$ a la derecha, la pérdida baja en

$$
\frac12 \left[ \frac{G_I^2}{H_I + \lambda} + \frac{G_D^2}{H_D + \lambda} - \frac{(G_I + G_D)^2}{H_I + H_D + \lambda} \right] - \gamma .
$$

<v-click>

Es el criterio para elegir divisiones (en vez de Gini o suma de cuadrados). Si ninguna gana más que $\gamma$, la hoja no se divide: **poda incorporada**, como el $\alpha$ del costo–complejidad.

</v-click>
</div>
<div v-click>

<Caja tipo="ejemplo" nombre="verificación con XGBoost">

Un árbol de profundidad 1 sobre la edad, logística, $\lambda = 1$, desde $p = 0.5$:

| | $G$ | $H$ | $w^*$ |
|---|---|---|---|
| edad < 61 | 12,910.0 | 8,249.00 | −1.5648 |
| edad ≥ 61 | 77.0 | 228.00 | −0.3362 |

Ganancia a mano: **167.0**. XGBoost reporta 334.0 (omite el $\frac12$) y las mismas hojas.

</Caja>

</div>
</div>

<!--
Las dos hojas son negativas porque partimos de p = 0.5 y solo el 11.7 % contrata; la de 61 años o más baja mucho menos el log-momio (en ese grupo contrata una proporción mucho mayor). Por eso en la práctica se parte del log-momio de la prevalencia.
-->

---

# Histogramas, crecimiento por hojas y GOSS

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<v-clicks>

- **Histogramas.** Cada atributo se discretiza una vez en ≤ 255 intervalos. Buscar la división es recorrer 255 sumas de $g$ y $h$, no $N$ umbrales; el histograma de una hija es el del padre menos el de su hermana.
- **Crecimiento por hojas** (LightGBM). Se divide siempre la hoja de mayor ganancia, hasta `num_leaves`. Baja más la pérdida con las mismas hojas; puede sobreajustar.
- **GOSS** (LightGBM). Los datos con $\lvert g_n\rvert$ chico ya están bien ajustados: se usan todos los de gradiente grande y una muestra, reponderada, de los demás.

</v-clicks>
</div>
<div>

<v-click>

El saldo toma **6,421** valores distintos en el entrenamiento; con histogramas, a lo más 255.

</v-click>

<v-click>

| biblioteca | histogramas | categóricas nativas |
|---|---|---|
| XGBoost | `tree_method="hist"` | sí |
| LightGBM | siempre | sí |
| scikit-learn | `HistGradientBoosting` | sí |

<div class="pequeno tenue">

LightGBM: Ke et al. (2017). Con cientos de miles o millones de filas, los histogramas son la diferencia entre entrenar en segundos o en horas.

</div>

</v-click>
</div>
</div>

<!--
En la libreta, con 34,000 llamadas, 100 árboles de profundidad 4: el gradient boosting «exacto» de scikit-learn tarda varias veces más que HistGradientBoosting (los tiempos dependen de la máquina).
Las categóricas nativas parten las categorías en dos grupos ordenándolas por G/H, sin one-hot.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. *Boosting*: un modelo aditivo **por etapas**; cada término corrige a los anteriores. Reduce sesgo, no varianza.
2. AdaBoost = pérdida exponencial; su error de entrenamiento cae como $\prod_t 2\sqrt{\varepsilon_t(1 - \varepsilon_t)}$.
3. *Gradient boosting*: árboles ajustados a los **pseudo-residuos**, $-\partial L/\partial F$.
4. XGBoost: segundo orden, $w^* = -G/(H + \lambda)$; tasa chica y **parada temprana**.
5. Ningún algoritmo protege de una **fuga**: ¿tendré este dato al decidir?

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Boosting y una campaña de mercadeo bancario* (Unidad 5, ejemplo 3).

**Laboratorio 5:** árboles y *boosting* desde cero.

**Lectura:** *The Elements of Statistical Learning*, cap. 10 (secciones 10.1–10.6 y 10.9–10.12); T. Chen y C. Guestrin, «XGBoost: a scalable tree boosting system», *KDD* (2016).

</div>
</div>
</div>
