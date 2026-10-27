---
theme: none
title: ¿Qué significa aprender de datos?
unidad: Unidad 1
layout: portada
subtitulo: Ejemplos, generalización, la navaja de Occam y lo que puede salir mal
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

# Las preguntas de hoy

<div class="cols cols-6-4 cols-centro" style="flex:1">
<div class="grande">

1. ¿Qué significa **aprender**?
2. ¿Cuándo conviene que una computadora aprenda en lugar de programarla?
3. ¿Por qué aprender **no es memorizar**?
4. ¿Cómo evitamos **engañarnos** al medir?
5. ¿Qué puede **salir mal**?

</div>
<div class="lema">
Al final de la clase deberían poder explicar por qué un modelo que acierta en <em>todos</em> sus datos puede ser un mal modelo.
</div>
</div>

---
layout: seccion
numero: 1
---

# ¿Qué significa aprender?

Empecemos por lo que hacemos las personas.

---
layout: pregunta
---

# ¿Cómo aprendió usted a reconocer el número siete?

<v-click>

Nadie le dio una definición formal del «siete». Vio muchos ejemplos, todos distintos, y alguien le dijo cuál era cuál.

</v-click>

<!--
Dejar que respondan. Lo que buscamos: aprendimos de ejemplos, no de una regla. Nadie sabe escribir "la regla del siete".
-->

---

# Mismo número, mil formas de escribirlo

<div class="figura-centro"><MosaicoDigitos :tam="66" /></div>

<div class="nota">

Dígitos escritos a mano por 13 personas, digitalizados en 8 × 8 pixeles (*UCI Optical Recognition of Handwritten Digits*, incluido en scikit-learn).

</div>

---

# Actividad: escriban la regla

<div class="cols" style="gap: 70px; margin-top: 4px">
<div>

### Esto es un 7

<MosaicoDigitos conjunto="sietes" :tam="82" :columnas="4" />

</div>
<div>

### Esto es un 1

<MosaicoDigitos conjunto="unos" :tam="82" :columnas="4" />

</div>
</div>

<div v-click class="nota">

En parejas, 3 minutos: escriban una regla con pixeles («si el renglón de arriba…») que separe los sietes de los unos. Luego busquemos un ejemplo donde falle.

</div>

<!--
Casi todas las reglas fallan con algún ejemplo: hay unos con "sombrero" y sietes muy inclinados. La conclusión es que escribir la regla a mano es frágil.
-->

---
layout: center
---

<div class="lema" style="font-size: 44px; max-width: 1000px; text-align: center; line-height: 1.3">

Aprender es construir, a partir de <em>ejemplos</em>, una regla que funcione en casos que <em>nunca hemos visto</em>.

</div>

---

# Programar o aprender

<DosCaminos />

---

# ¿Aprender o programar?

<p class="tenue" style="margin-bottom: 26px">Para cada problema: ¿escribirían el programa o dejarían que la computadora aprenda de ejemplos?</p>

<ul class="casos grande">
  <li>Calcular el IVA de una factura <Veredicto v-click /></li>
  <li>Detectar correos fraudulentos <Veredicto v-click si /></li>
  <li>Ordenar una lista de calificaciones <Veredicto v-click /></li>
  <li>Pronosticar el almacenamiento de la presa El Novillo <Veredicto v-click si /></li>
  <li>Reconocer quién habla en una grabación <Veredicto v-click si /></li>
</ul>

<!--
Ir caso por caso y pedir que justifiquen. El IVA y el ordenamiento tienen una regla exacta y conocida: aprender sería absurdo.
-->

---

# Cuándo tiene sentido aprender

<div class="cols cols-6-4">
<div class="grande">

1. **Hay un patrón**: las entradas dicen algo sobre la salida.
2. **No sabemos escribirlo**: si conociéramos la fórmula, la programaríamos.
3. **Tenemos datos** que muestran ese patrón y se parecen a los casos futuros.

</div>
<div v-click>
<Caja tipo="cuidado">

La tercera condición es la que más se olvida. Un modelo entrenado con datos que no se parecen a su uso real **falla en silencio**.

</Caja>
</div>
</div>

---

# Ahora sí, una definición

<Caja tipo="definicion" nombre="Tom Mitchell (1997)">

<div class="grande">

Un programa **aprende** de la experiencia $E$, en una tarea $T$, con una medida de desempeño $P$, si su desempeño en $T$, medido con $P$, **mejora con la experiencia** $E$.

</div>

</Caja>

<div v-click class="cols" style="margin-top: 22px; gap: 56px">
<div>

### En los dígitos

- **Tarea:** decir qué número es una imagen.
- **Experiencia:** 1,797 imágenes ya etiquetadas.
- **Desempeño:** porcentaje de aciertos en imágenes **nuevas**.

</div>
<div>

### En la presa

- **Tarea:** pronosticar el almacenamiento del próximo mes.
- **Experiencia:** años de registros de lluvia y almacenamiento.
- **Desempeño:** error promedio en meses que el modelo no vio.

</div>
</div>

---

# Tres formas de aprender

<Paradigmas />

---
layout: seccion
numero: 2
---

# Aprender no es memorizar

Un experimento con quince puntos.

---
layout: pregunta
---

# Un estudiante memoriza las respuestas de los exámenes de años anteriores.

<v-click>

Saca 10 en todos esos exámenes. ¿Aprendió? ¿Qué calificación esperan que saque en el examen de este año?

</v-click>

---

# El memorizador

<div class="cols cols-4-6 cols-centro" style="flex:1; gap: 70px">
<Memorizador />
<div class="grande">

Un programa que **guarda todos los ejemplos** en una tabla y, ante una imagen, busca si ya la vio:

- en los datos de entrenamiento acierta el **100 %**;
- ante una imagen nueva **no tiene nada que decir**.

<div v-click class="lema" style="margin-top: 28px">

Memorizar es fácil. Lo difícil, y lo valioso, es <em>generalizar</em>.

</div>
</div>
</div>

---

# Quince observaciones

<p class="tenue">Una variable x y una respuesta y medida con ruido. Queremos una curva que prediga y a partir de x.</p>

<div class="figura-centro"><EscenaPolinomio :ancho="1100" :alto="420" /></div>

<div class="pequeno tenue">¿Qué curva dibujarían ustedes? Tracen una en su cuaderno.</div>

---

# Tres propuestas

<p class="tenue" style="margin-bottom: 8px">Ajustamos polinomios de grado 1, 4 y 12 por mínimos cuadrados.</p>

<PolinomiosTres />

<div v-click class="nota">

El de grado 12 pasa por casi todos los puntos. ¿Es el mejor?

</div>

---

# La prueba de fuego: datos nuevos

<p class="tenue" style="margin-bottom: 10px">Llegan quince observaciones nuevas del mismo fenómeno.</p>

<div class="cols" style="gap: 36px">
<div>

### Grado 4

<EscenaPolinomio :grado="4" nuevos :ancho="560" :alto="360" />

</div>
<div>

### Grado 12

<EscenaPolinomio :grado="12" nuevos :ancho="560" :alto="360" />

</div>
</div>

<LeyendaPolinomio nuevos ajuste />

---

# Para comparar, necesitamos medir el error

<div class="cols cols-6-4 cols-centro">
<div><EscenaPolinomio :grado="4" residuos="tr" :ancho="700" :alto="420" /></div>
<div>

En cada punto, la curva se equivoca por la distancia vertical <span style="color:#D5433C; font-weight:650">en rojo</span>.

<v-click>

Elevamos al cuadrado (para que no importe el signo y pesen más los errores grandes) y promediamos:

$$
\frac{1}{n}\sum_{i=1}^{n}\big(\,\text{predicción}_i - y_i\,\big)^2
$$

</v-click>
<v-click>

Esa forma de medir qué tan mal lo hace un modelo en un ejemplo es una **función de pérdida**.

</v-click>
</div>
</div>

<!--
Aquí aparece por primera vez la función de pérdida, como respuesta a una necesidad: comparar curvas con un número. Mencionar que la elección de la pérdida depende del problema (no es lo mismo equivocarse por arriba que por abajo en una presa).
-->

---

# Dos errores distintos

<div class="cols" style="gap: 56px; margin-top: 8px">
<Caja tipo="definicion" nombre="error en la muestra">

El promedio de la pérdida en los datos que **vio** el modelo:

$$
\Ein(h) = \frac{1}{n}\sum_{i=1}^{n} \ell\big(h(\vx_i),\, y_i\big)
$$

<span class="tenue">Lo podemos calcular.</span>

</Caja>
<Caja tipo="definicion" nombre="error fuera de la muestra">

La pérdida esperada en datos **nuevos** del mismo fenómeno:

$$
\Eout(h) = \E\big[\, \ell\big(h(\vx),\, y\big) \,\big]
$$

<span class="tenue">Es lo que nos importa, y no lo conocemos.</span>

</Caja>
</div>

<div v-click class="lema" style="margin-top: 20px">

Aprender bien es lograr un $\Ein$ pequeño <em>y</em> que $\Eout$ se parezca a $\Ein$.

</div>

---

# Explórenlo: el grado del polinomio

<PolinomioInteractivo />

<div class="nota">

Punteada: la regla verdadera, que en la práctica nunca conocemos. $\Ein$ siempre baja; $\Eout$ baja y luego se dispara.

</div>

<!--
Mover el control en clase: grado 0, 1, 3, 9, 12. Preguntar antes de mover: ¿qué creen que pasará con E_out?
-->

---
layout: pregunta
---

# Si dos explicaciones se ajustan igual de bien a los datos, ¿con cuál se quedan?

---
layout: cita
autor: Guillermo de Ockham (c. 1287–1347)
nota: «No hay que multiplicar las entidades sin necesidad.» Entre explicaciones que dan cuenta igual de bien de lo observado, conviene preferir la más simple.
---

*Pluralitas non est ponenda sine necessitate.*

---

# La navaja de Occam en la historia de la ciencia

<Epiciclos />

---

# Occam en el aprendizaje automático

<PolinomiosTres diagnostico />

<div class="cols" style="gap: 48px; margin-top: 14px">
<div class="pequeno">

**Por qué funciona.** Un modelo simple no puede ajustarse al ruido; si aun así se ajusta a los datos, probablemente encontró el patrón.

</div>
<div class="pequeno">

**Su límite.** «Simple» no es «simplista»: el grado 1 tampoco sirve. En la Unidad 3 convertiremos esta idea en herramientas: regularización y validación.

</div>
</div>

---
layout: seccion
numero: 3
---

# Cómo no engañarnos

El flujo de trabajo que hace creíbles nuestros números.

---

# El ciclo de vida de un modelo

<div class="figura-centro"><CicloAnillo /></div>

<div class="nota">

En dorado, los pasos que protegen la evaluación: **separar** antes de tocar los datos y **evaluar en prueba una sola vez**.

</div>

---
layout: pregunta
---

# ¿Qué pasa si el profesor deja ver el examen el día anterior?

<v-click>

La calificación deja de medir lo que aprendieron. Con los datos de prueba pasa exactamente lo mismo: si el modelo los «vio», su desempeño en ellos ya no dice nada.

</v-click>

---

# Una trampa real: la columna «bote»

<p class="tenue" style="margin-bottom: 20px">Exactitud (validación cruzada) al predecir quién sobrevivió al Titanic.</p>

<BarrasFuga />

<div v-click class="cols" style="gap: 48px; margin-top: 10px">
<div class="pequeno">

**Qué pasó.** La columna `boat` dice en qué bote salvavidas subió cada pasajero. Se registró **después** del naufragio: quien subió a un bote casi siempre sobrevivió.

</div>
<div class="pequeno">

**La lección.** Un resultado demasiado bueno es una señal de alarma. Esa información no existiría al momento de predecir: es una **fuga de información**.

</div>
</div>

---

# La fuga más común es más sutil

<div class="cols codigo-chico" style="gap: 32px">
<Caja tipo="cuidado" etiqueta="Así no">

```python
X = StandardScaler().fit_transform(X)  # ve todo
X_tr, X_te, y_tr, y_te = train_test_split(X, y)
modelo.fit(X_tr, y_tr)
modelo.score(X_te, y_te)      # optimista
```

</Caja>
<Caja tipo="correcto" etiqueta="Así sí">

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y)
modelo = make_pipeline(StandardScaler(),
                       LogisticRegression())
modelo.fit(X_tr, y_tr)        # solo ve X_tr
modelo.score(X_te, y_te)      # honesto
```

</Caja>
</div>

<div class="grande" style="margin-top: 10px">

Escalar con la media de **todos** los datos le pasa al modelo información de la prueba. Regla: **primero se separa, después se transforma**. Los `Pipeline` de scikit-learn lo garantizan.

</div>

---
layout: seccion
numero: 4
---

# Lo que puede salir mal

Un modelo hereda los datos, la métrica y las decisiones de quien lo construye.

---
layout: pregunta
---

# Un modelo aprende de datos históricos. ¿Qué aprende si la historia fue injusta?

---
layout: caso
caso: Caso 1 · Justicia penal
cifra: 45 % <small>vs.</small> 23 %
cifraNota: de las personas que no volvieron a delinquir fueron marcadas como de «alto riesgo», según fueran afroamericanas o blancas.
fuente: J. Angwin et al., «Machine Bias», ProPublica (2016).
---

# Puntajes de riesgo de reincidencia

Jueces de Estados Unidos usan un puntaje (COMPAS) para decidir fianzas y sentencias. ProPublica analizó más de 7,000 casos de un condado de Florida.

La exactitud era parecida en ambos grupos, pero **los errores no se repartían igual**: unos cargaban con los falsos positivos y otros con los falsos negativos.

<span class="falla">Dónde falló</span> en qué significa «ser justo». Hay definiciones razonables de equidad que no se pueden cumplir todas a la vez.

---
layout: caso
caso: Caso 2 · Contratación
cifra: 10 años
cifraNota: de currículums recibidos, en su mayoría de hombres, sirvieron para entrenar el modelo.
fuente: J. Dastin, Reuters (2018).
---

# Un filtro automático de currículums

Amazon desarrolló un sistema para calificar candidatos a puestos técnicos imitando las contrataciones pasadas.

El modelo aprendió a **penalizar** los currículums que incluían la palabra *women's* (por ejemplo, «capitana del club de ajedrez femenil»). La empresa abandonó el proyecto.

<span class="falla">Dónde falló</span> en los datos: el modelo reprodujo con fidelidad un patrón histórico injusto.

---
layout: caso
caso: Caso 3 · Salud
cifra: 17.7 % → 46.5 %
cifraNota: proporción de pacientes afroamericanos que recibirían atención adicional si se corrige el sesgo del algoritmo.
fuente: Z. Obermeyer et al., Science 366 (2019).
---

# ¿Quién necesita más atención médica?

Un algoritmo usado en hospitales de Estados Unidos decidía qué pacientes entraban a programas de cuidado especial. En lugar de qué tan enfermo estaba alguien, predecía **cuánto gastaría** en salud.

Como históricamente se ha gastado menos en pacientes afroamericanos, con el mismo puntaje estaban **más enfermos**.

<span class="falla">Dónde falló</span> en la variable objetivo: el gasto era un mal sustituto de la necesidad.

---
layout: caso
caso: Caso 4 · Visión por computadora
cifra: 34.7 % <small>vs.</small> 0.8 %
cifraNota: error al clasificar el género de mujeres de piel oscura y de hombres de piel clara.
fuente: J. Buolamwini y T. Gebru, «Gender Shades», FAT* (2018).
---

# Reconocimiento facial comercial

Se evaluaron sistemas comerciales de análisis facial con un conjunto de fotografías **balanceado** por género y tono de piel.

Los sistemas eran excelentes con los grupos bien representados en sus datos de entrenamiento y malos con los demás.

<span class="falla">Dónde falló</span> en datos no representativos, y en evaluar con datos que tampoco lo eran.

---
layout: pregunta
etiqueta: Antes de entrenar cualquier modelo
---

# Preguntas que siempre hay que hacerse

<div class="grande" style="max-width: 980px">

- ¿Quién generó estos datos, y a quién **no** representan?
- ¿La variable que predigo mide lo que realmente me importa?
- ¿Qué cuesta más, un falso positivo o un falso negativo, y **a quién** le cuesta?
- ¿Quién se ve afectado por las decisiones del modelo, y cómo puede impugnarlas?

</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex:1">
<div class="grande">

1. Aprender es **generalizar** a partir de ejemplos, no memorizarlos.
2. Lo que importa es el error en **datos nuevos**; el error en los datos de entrenamiento engaña.
3. Entre modelos que explican igual, conviene el **más simple** (Occam).
4. Se separan los datos **antes** de tocarlos y la prueba se usa **una vez**.
5. Los datos y la métrica son **decisiones**, con consecuencias para personas.

</div>
<div>

### Para la próxima clase

<div class="pequeno">

**Libretas:** *Explorar datos con pandas* y *Un primer pipeline sin trampas* (Unidad 1, ejemplos 1 y 3).

**Lectura:** Abu-Mostafa, Magdon-Ismail y Lin (2012), *Learning From Data*, cap. 1.

**Para profundizar:** Barocas, Hardt y Narayanan (2023), *Fairness and Machine Learning*, cap. 1.

</div>
</div>
</div>
