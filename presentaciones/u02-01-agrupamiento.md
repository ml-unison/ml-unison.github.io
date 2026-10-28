---
theme: none
title: Agrupamiento
unidad: Unidad 2
layout: portada
subtitulo: Encontrar grupos cuando nadie nos dice las respuestas
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
  Reconocimiento de Patrones · Unidad 2 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Agrupamiento

---
layout: pregunta
---

# Si nadie nos dice las respuestas, ¿qué podemos aprender de los datos?

<!--
Recordar la Unidad 1: en supervisado cada ejemplo trae su etiqueta. Aquí solo tenemos las x. Dejar que propongan: grupos, cosas raras, resúmenes, dimensiones importantes.
-->

---

# ¿Cuántos grupos ven?

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div>

<PinguinosPico v-if="$clicks < 1" />
<PinguinosPico v-else especies />

</div>
<div>

342 pingüinos del archipiélago Palmer, en la Antártida. De cada uno se midió el **largo** y la **profundidad** del pico.

<v-click>

Son tres especies. Pero sin los colores, mucha gente ve **dos** grupos, y otra mucha ve **cuatro**.

</v-click>

<p class="pequeno tenue" style="margin-top: 30px">K. Gorman, Palmer Station LTER; paquete <em>palmerpenguins</em> de A. Horst, A. Hill y K. Gorman (2020).</p>

</div>
</div>

<!--
Votación a mano alzada antes del clic: ¿2, 3, 4? Anotar en el pizarrón. La discusión es el punto: «grupo» no está definido hasta que decimos qué es parecido.
-->

---

# Para qué sirve agrupar

<div class="cols" style="gap: 56px; margin-top: 10px">
<div>

### Entender

- **Segmentar clientes** para diseñar campañas distintas.
- **Subtipos de enfermedades.** Al agrupar tumores de mama por la expresión de sus genes aparecieron subtipos con pronósticos distintos.<sup>1</sup>
- **Organizar documentos** por tema, sin leerlos todos.

</div>
<div>

### Resumir

- **Comprimir.** Una imagen de millones de colores, con 16.
- **Detectar lo raro.** Lo que no cae en ningún grupo.
- **Preparar** un problema supervisado: los grupos como atributo nuevo.

</div>
</div>

<p class="pequeno tenue abajo"><sup>1</sup> C. Perou et al., «Molecular portraits of human breast tumours», <em>Nature</em> 406 (2000).</p>

---
layout: seccion
numero: 1
---

# k-medias

Representar cada grupo por un solo punto.

---

# La idea: un representante por grupo

<div class="cols cols-centro" style="flex: 1; gap: 56px">
<div class="grande">

Imaginen que solo pueden guardar **$k$ puntos** y que cada dato se sustituye por el más cercano de ellos.

<v-click>

¿Qué $k$ puntos elegirían para **perder lo menos posible**?

</v-click>
</div>
<div v-click>

Si medimos la pérdida con la distancia al cuadrado, buscamos centros $\mu_1, \dots, \mu_k$ y una asignación $c(i)$ que minimicen

$$
J = \sum_{i=1}^{n} \norm{\vx_i - \mu_{c(i)}}^2 .
$$

<div class="pequeno" style="margin-top: 16px">

Es la misma idea que en la Unidad 1: definir qué es «hacerlo bien» con una función que se pueda minimizar.

</div>
</div>
</div>

---

# Dos mitades fáciles de un problema difícil

<div class="cols" style="gap: 56px">
<div>

Minimizar $J$ sobre centros **y** asignaciones a la vez es NP-difícil.<sup>1</sup> Pero cada mitad por separado es fácil:

<v-clicks>

- **Centros fijos.** La mejor asignación es mandar cada punto a su centro más cercano.
- **Asignación fija.** El mejor centro de cada grupo es su **media**:
$$
\nabla_{\mu_j} \sum_{c(i)=j} \norm{\vx_i - \mu_j}^2 = 0 \;\Rightarrow\; \mu_j = \bar\vx_j .
$$

</v-clicks>

<p class="pequeno tenue" style="margin-top: 30px"><sup>1</sup> D. Aloise et al., «NP-hardness of Euclidean sum-of-squares clustering», <em>Machine Learning</em> 75 (2009).</p>

</div>
<div v-click>

<Caja tipo="algoritmo" nombre="Lloyd, 1957">

1. Elegir $k$ centros iniciales.
2. **Asignar:** cada punto al centro más cercano.
3. **Mover:** cada centro a la media de sus puntos.
4. Repetir 2–3 hasta que nada cambie.

</Caja>

</div>
</div>

---

# k-medias, paso a paso

<KMediasPasos />

<!--
Mover el deslizador despacio. Preguntar en la iteración 1: ¿por qué el grupo de arriba se lleva casi todo? Luego ver cómo J baja rápido al principio y después casi nada.
-->

---

# ¿Por qué siempre termina?

<div class="cols" style="gap: 56px">
<div>

<Caja tipo="proposicion">

$J$ nunca aumenta de una iteración a la siguiente, y el algoritmo termina en un número finito de pasos.

</Caja>

<v-click>

**Idea.** Cada paso minimiza $J$ respecto a una parte de las variables, con la otra fija: no puede subir. Hay un número finito de particiones de $n$ puntos en $k$ grupos, y no se repite ninguna (salvo empates), así que se acaba.

</v-click>
</div>
<div v-click>

### Pero…

Termina en un **mínimo local**: un punto donde ni asignar ni mover mejoran, pero no necesariamente el mejor.

El resultado depende del **inicio**.

</div>
</div>

---

# El inicio importa

<MinimosLocales />

<div v-click class="nota">

**k-medias++** elige el primer centro al azar y cada uno de los siguientes con probabilidad proporcional a $D(\vx)^2$, la distancia al cuadrado al centro más cercano ya elegido: tiende a repartirlos. En la práctica, además, se corre varias veces (`n_init`) y se queda el menor $J$. D. Arthur y S. Vassilvitskii, *SODA* (2007).

</div>

---
layout: pregunta
---

# ¿Y cuántos grupos pedimos? ¿Cómo elegimos $k$?

<v-click>

$J$ no sirve para eso: siempre baja al aumentar $k$. Con $k = n$, $J = 0$.

</v-click>

---

# Clientes de una tienda en línea

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div class="grande">

- 25,900 facturas de una tienda en línea del Reino Unido, de diciembre de 2010 a diciembre de 2011.
- Resumimos a cada uno de los **4,317 clientes** con tres números:
  - **R**ecencia: días desde su última compra.
  - **F**recuencia: cuántas compras hizo.
  - **M**onto: cuánto gastó en total.

</div>
<div>

<div class="lema" style="font-size: 25px">Las tres son muy sesgadas: logaritmo y estandarizar, como en la Unidad 1.</div>

<p class="pequeno tenue" style="margin-top: 30px">D. Chen, <em>Online Retail</em>, UCI Machine Learning Repository (2015). CC BY 4.0.</p>

</div>
</div>

---

# Dos criterios para elegir $k$

<CurvasK />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

**Codo:** buscar dónde $J$ deja de bajar mucho. Aquí no hay un codo claro, lo que pasa casi siempre con datos reales.

</div>
<div class="pequeno" v-click>

**Silueta** de un punto: $s = \dfrac{b - a}{\max(a, b)}$, con $a$ su distancia media a su grupo y $b$ al grupo vecino más cercano. Cerca de 1, bien ubicado; cerca de 0, en la frontera.

</div>
</div>

---

# La silueta prefiere $k = 2$. ¿Nos sirve?

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 48px">
<div>

Con $k = 2$ el corte separa casi exactamente a quienes compraron **una sola vez**. En logaritmo, entre 1 y 2 compras hay un hueco: esa estructura la pusimos nosotros al representar los datos.

<div v-click="2" style="margin-top: 22px">
<Caja tipo="idea">

No hay un $k$ «verdadero». Los criterios internos descartan valores malos; la decisión depende de **para qué** se quieren los grupos.

</Caja>
</div>
</div>
<div v-click="1">

Con $k = 4$ salen grupos que mercadotecnia puede **usar**:

<TablaMini :columnas="['segmento', 'clientes', 'días sin comprar', 'compras', '% ventas']" :filas="[['leales de alto valor', 763, 7, 9, '66 %'], ['regulares', '1,214', 53, 4, '23 %'], ['nuevos u ocasionales', 777, 16, 2, '5 %'], ['dormidos', '1,563', 173, 1, '6 %']]" :tam="19" />

<p class="pequeno tenue">Medianas por grupo. Los nombres los ponemos nosotros.</p>

</div>
</div>

<!--
La libreta 1 de la unidad hace este análisis completo y nombra los segmentos.
-->

---

# Lo que k-medias supone sin decirlo

<Fallas />

<div v-click class="nota">

k-medias parte el espacio en regiones de Voronoi: **fronteras rectas** a medio camino entre centros. Funciona bien con grupos redondos, de tamaño parecido y en la misma escala. Si no es así, se equivoca con toda confianza.

</div>

---
layout: seccion
numero: 2
---

# Agrupamiento jerárquico

No uno, sino todos los agrupamientos a la vez.

---

# De abajo hacia arriba

<div class="cols" style="gap: 56px">
<div>

<Caja tipo="algoritmo" nombre="aglomerativo">

1. Cada punto es su propio grupo.
2. Unir los **dos grupos más cercanos**.
3. Repetir hasta que quede uno solo.

</Caja>

<v-click>

El resultado es un **dendrograma**: un árbol que registra a qué distancia se unió cada par de grupos. Cortarlo a una altura da un agrupamiento; no hay que fijar $k$ antes.

</v-click>
</div>
<div v-click>

### ¿Distancia entre grupos?

Entre dos puntos la sabemos medir. Entre dos **grupos** hay que decidirlo, y la decisión cambia el resultado.

</div>
</div>

---

# Cuatro maneras de medir la distancia entre grupos

<Enlaces />

<div v-click class="nota">

El **simple** encadena: dos grupos lejanos se unen si hay una fila de puntos entre ellos. El **completo** prefiere grupos compactos. **Ward** minimiza lo mismo que k-medias, $J$, un paso a la vez.

</div>

---

# Las presas de Sonora, agrupadas por cómo se llenan

<Dendrograma />

<!--
Cada presa se describe con su porcentaje de llenado mes a mes (42 meses). Enlace promedio. Mover el corte: con un corte alto quedan pocos grupos. Notar que las presas del Yaqui NO quedan juntas: el algoritmo agrupa por comportamiento, no por río. ¿Qué tienen en común las que sí quedaron juntas?
-->

---
layout: caso
caso: Caso · Evolución humana
cifra: 200 mil años
cifraNota: edad aproximada de la ancestra común, por línea materna, de todas las personas vivas.
fuente: R. Cann, M. Stoneking y A. Wilson, «Mitochondrial DNA and human evolution», Nature 325 (1987).
---

# La Eva mitocondrial

Las mitocondrias tienen su propio ADN y lo heredamos **solo de la madre**, casi sin cambios: cada mutación es una marca que pasa a toda la descendencia.

<v-click>

Cann, Stoneking y Wilson compararon el ADN mitocondrial de **147 personas** de África, Asia, Europa, Australia y Nueva Guinea (133 tipos distintos) y los acomodaron en un **árbol según su parecido**, como el dendrograma de las presas. Al método no le dijeron de dónde venía cada persona.

</v-click>

<!--
«Eva» no era la única mujer de su época: es la ancestra común más reciente por línea materna. Otras mujeres de su tiempo también son ancestras nuestras, por otras líneas. El árbol se construyó por parsimonia (el árbol con menos mutaciones), un método de agrupamiento jerárquico propio de la biología.
-->

---

# Lo que dijo el árbol

<div class="cols" style="grid-template-columns: 11fr 9fr; gap: 44px">
<div>

<ArbolMitocondrial />

<p class="tenue" style="font-size: 16px; margin-top: 2px">Esquema de la forma del árbol, no los 133 tipos del artículo.</p>

</div>
<div>

<v-clicks>

- La raíz se abre en dos ramas: una **solo con africanos** y otra con **las cinco poblaciones**.
- Los africanos son el grupo **más diverso**: la variación dentro de África es tan grande como entre África y cualquier otra región.
- Conclusión: todos descendemos de una población que vivió **en África** hace unos 140 a 290 mil años; las demás regiones se poblaron después, varias veces.

</v-clicks>

<div v-click class="pequeno tenue" style="margin-top: 22px">

El método del árbol recibió críticas, y la conclusión se confirmó después con genomas mitocondriales completos: M. Ingman et al., *Nature* 408 (2000).

</div>
</div>
</div>

---

# Agrupar personas: lo que dicen los datos sobre las «razas»

<div class="cols" style="gap: 48px">
<div>

<div class="comparacion">
<div>
<span class="comparacion__etiqueta">variación genética dentro de cada población</span>
<span class="comparacion__cifra">93–95 %</span>
<span class="comparacion__nota">Entre los grandes grupos geográficos, solo de 3 a 5 %.</span>
</div>
</div>

<p class="pequeno tenue" style="margin-top: 20px">N. Rosenberg et al., «Genetic structure of human populations», <em>Science</em> 298 (2002): 1,056 personas de 52 poblaciones, agrupadas sin decirle al algoritmo su origen. R. Lewontin había encontrado algo parecido en 1972: alrededor de 85 % dentro de las poblaciones.</p>

</div>
<div>

<v-clicks>

- La mayor parte de lo que nos distingue genéticamente está **entre personas de una misma población**, no entre poblaciones.
- Los grupos que encuentran los algoritmos siguen la **geografía** y cambian de forma **gradual**, sin fronteras nítidas.
- Somos una especie joven, salida de África hace poco. Las «razas» no son grupos biológicos separados.

</v-clicks>
</div>
</div>

<div v-click class="nota">

**Cuidado:** esos mismos grupos se han citado como «prueba» de que las razas existen, algo que sus autores no afirman. Lo que significa un grupo lo decide quien lo interpreta.

</div>

<!--
Conectar con la cita de Estivill-Castro que viene más adelante: los grupos los pone el algoritmo, la interpretación la ponemos nosotros. Este es uno de los mejores ejemplos de aprendizaje no supervisado usado para desmontar un prejuicio con datos.
-->

---

# El costo de ver todo el árbol

<div class="cols" style="gap: 56px">
<div class="grande">

- Guardar todas las distancias entre pares: memoria $O(n^2)$.
- Tiempo entre $O(n^2)$ y $O(n^3)$, según el enlace.
- Con 100 mil puntos, 5 mil millones de distancias.

</div>
<div v-click>

<Caja tipo="idea">

Jerárquico para **cientos o pocos miles** de puntos, cuando el árbol mismo es informativo (especies, documentos, presas). k-medias cuando $n$ es grande: cada iteración cuesta $O(nkd)$.

</Caja>

</div>
</div>

---
layout: seccion
numero: 3
---

# DBSCAN

Un grupo es una región densa.

---

# Grupos como regiones densas

<div class="cols" style="gap: 56px">
<div>

Dos parámetros: un radio $\varepsilon$ y un mínimo de puntos $m$.

<v-clicks>

- **Punto central:** tiene al menos $m$ vecinos a distancia $\le \varepsilon$.
- Dos puntos centrales a distancia $\le \varepsilon$ están en el **mismo grupo**, y así en cadena.
- **Borde:** no es central, pero está cerca de uno.
- **Ruido:** ni central ni cerca de uno. No pertenece a nadie.

</v-clicks>

<p class="pequeno tenue" style="margin-top: 24px">M. Ester, H.-P. Kriegel, J. Sander y X. Xu, <em>KDD</em> (1996).</p>

</div>
<div v-click>

### Lo que gana

- No hay que fijar $k$.
- Grupos de **cualquier forma**.
- Separa el **ruido** en lugar de forzarlo a un grupo.

### Lo que cuesta

- Elegir $\varepsilon$, y uno solo para todo el espacio: le cuestan las densidades distintas.

</div>
</div>

---

# Los mismos tres conjuntos, con DBSCAN

<Fallas metodo="db" />

<div v-click class="nota">

En gris, los puntos que DBSCAN declara **ruido**. Resuelve las formas no convexas, pero en el tercer conjunto el grupo disperso se le deshace en ruido: ningún $\varepsilon$ sirve para las dos densidades a la vez.

</div>

---
layout: seccion
numero: 4
---

# ¿Cómo sabemos si salió bien?

---

# Evaluar sin respuestas

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 40px">
<div>

### Internos

Solo usan los datos: silueta, $J$, índices de Davies–Bouldin o Calinski–Harabasz.

<span class="tenue pequeno">Cada uno premia su propia idea de grupo (casi siempre compacto y redondo).</span>

</div>
<div v-click>

### Externos

Comparan contra etiquetas conocidas, p. ej. el **índice de Rand ajustado** (ARI): 1 es idéntico y 0 es lo que daría el azar.

<span class="tenue pequeno">Sirven para estudiar métodos; en un problema real, si tuviéramos etiquetas no estaríamos agrupando.</span>

</div>
<div v-click>

### El de verdad

¿Los grupos son **estables** (cambian poco con otra muestra) y **útiles** para la decisión que motivó el análisis?

<span class="tenue pequeno">Esto lo juzga quien conoce el problema.</span>

</div>
</div>

---
layout: cita
autor: V. Estivill-Castro, «Why so many clustering algorithms», SIGKDD Explorations 4 (2002)
nota: Cada algoritmo trae su propia definición de grupo. No hay una que sea correcta para todos los problemas.
---

*Clustering is in the eye of the beholder.*

---
layout: caso
caso: Ética · Segmentar personas
cifra: Código postal
cifraNota: bastaba para decidir qué precio veía cada visitante de una tienda en línea.
fuente: J. Valentino-DeVries, J. Singer-Vine y A. Soltani, The Wall Street Journal (2012).
---

# Precios distintos según dónde vives

El sitio de una cadena de papelerías mostraba precios más altos a quienes vivían lejos de una tienda de la competencia. El periódico encontró que las zonas con precios más bajos tenían, en promedio, **ingresos más altos**.

<v-click>

Nadie programó «cobrar más a los pobres». Bastó con segmentar por una variable que se relaciona con el ingreso. Un segmento se convierte fácilmente en un **estereotipo con consecuencias**.

</v-click>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div class="grande">

1. Agrupar exige decidir qué es **parecido**: variables, escala y distancia.
2. **k-medias** minimiza $J$ alternando asignar y mover; mínimos locales, k-medias++ y varios inicios.
3. **Jerárquico** da el árbol completo; el enlace cambia el resultado.
4. **DBSCAN** encuentra regiones densas de cualquier forma y separa el ruido.
5. No hay $k$ verdadero: el criterio final es la utilidad.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Segmentar clientes con k-medias, jerárquico y DBSCAN* (Unidad 2, ejemplo 1).

**Laboratorio 2:** programar k-medias desde cero.

**Lectura:** James et al., *An Introduction to Statistical Learning*, sección 12.4.

</div>
</div>
</div>
