---
theme: none
title: Reducción de dimensión
unidad: Unidad 2
layout: portada
subtitulo: Pocas coordenadas que conserven lo importante, para ver, comprimir y preparar datos
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

# Reducción de dimensión

---
layout: pregunta
---

# Una foto de 12 megapíxeles es un punto en un espacio de 36 millones de dimensiones. ¿Las fotos de verdad llenan ese espacio?

<v-click>

Si eligen 36 millones de números al azar, obtienen ruido, nunca una cara. Las fotos de verdad viven en una parte diminuta del espacio.

</v-click>

---

# Un dígito es un punto en $\R^{64}$

<div class="cols cols-centro" style="flex: 1; grid-template-columns: 1fr 1fr; gap: 56px">
<div>

<MosaicoDigitos />

</div>
<div style="font-size: 26px">

1,797 dígitos escritos a mano por 43 personas, en imágenes de $8 \times 8$ con intensidades de 0 a 16.

<v-click>

Cada imagen es un vector de **64 números**. No podemos dibujar 64 dimensiones, y muchas de ellas casi no varían: las esquinas siempre están en blanco.

</v-click>

<p class="tenue" style="margin-top: 16px; font-size: 17px">E. Alpaydin y C. Kaynak, UCI (1998). Incluido en scikit-learn.</p>

</div>
</div>

---

# La maldición de la dimensión

<div class="cols cols-centro" style="flex: 1; gap: 56px">
<div>

500 puntos al azar en el cubo $[0,1]^d$. Para un punto cualquiera, ¿cuánto más lejos está el vecino **más lejano** que el **más cercano**?

$$
\frac{d_{\max} - d_{\min}}{d_{\min}}
$$

</div>
<div>

<DistanciasDimension />

<div v-click class="pequeno" style="margin-top: 26px">

En dimensión alta **todos los puntos están casi a la misma distancia**. «Vecino más cercano» y «grupo compacto» pierden sentido, justo lo que usan k-medias y DBSCAN.

</div>
</div>
</div>

---
layout: seccion
numero: 1
---

# Análisis de componentes principales

La mejor sombra.

---

# La mejor sombra

<div class="cols" style="grid-template-columns: 5fr 4fr; gap: 48px">
<div>

<ProyeccionPCA :proyectar="$clicks >= 1" />

</div>
<div>

Aleta y masa de los pingüinos, estandarizadas. Casi toda la variación va **en una dirección**.

<v-click>

Si proyectamos cada punto sobre esa recta (en rojo), pasamos de 2 números a 1 y **conservamos el 94 % de la varianza**.

</v-click>

<v-click>

Como elegir desde dónde fotografiar un objeto para que su sombra muestre lo más posible.

</v-click>
</div>
</div>

---

# Dos maneras de pedir lo mismo

<div class="cols" style="gap: 48px">
<div>

Datos centrados $\vx_1, \dots, \vx_n \in \R^d$. Buscamos una dirección unitaria $\mathbf{u}$.

<Caja tipo="idea" nombre="máxima varianza">

Que la sombra quede lo más **extendida** posible:

$$
\max_{\norm{\mathbf u} = 1} \; \frac{1}{n} \sum_i (\mathbf u\T \vx_i)^2
$$

</Caja>

</div>
<div v-click>

<Caja tipo="idea" nombre="mínimo error de reconstrucción">

Que los puntos queden lo más **cerca** de su sombra:

$$
\min_{\norm{\mathbf u} = 1} \; \frac{1}{n} \sum_i \norm{\vx_i - (\mathbf u\T \vx_i)\, \mathbf u}^2
$$

</Caja>

<div class="pequeno" style="margin-top: 16px">

Son lo mismo por Pitágoras: $\;\norm{\vx_i}^2 = (\mathbf u\T\vx_i)^2 + \norm{\vx_i - (\mathbf u\T\vx_i)\mathbf u}^2$, y el lado izquierdo no depende de $\mathbf u$.

</div>
</div>
</div>

---

# La solución: vectores propios de la covarianza

<div class="cols" style="gap: 48px">
<div>

Con $S = \frac{1}{n} \sum_i \vx_i \vx_i\T$, la varianza de la sombra es $\mathbf u\T S \mathbf u$. Con un multiplicador de Lagrange:

$$
\nabla_{\mathbf u}\big[\mathbf u\T S \mathbf u - \lambda(\mathbf u\T \mathbf u - 1)\big] = 0 \;\Rightarrow\; S\mathbf u = \lambda \mathbf u .
$$

<v-click>

$\mathbf u$ es un **vector propio** de $S$, y la varianza que conserva es $\mathbf u\T S \mathbf u = \lambda$. Gana el de **mayor valor propio**.

</v-click>
</div>
<div v-click>

<Caja tipo="algoritmo" nombre="PCA con la SVD">

1. Centrar: restar la media de cada columna.
2. Descomponer $X = U \Sigma V\T$.
3. Las filas de $V\T$ son las componentes; la varianza de la $j$-ésima es $\sigma_j^2 / (n-1)$.
4. Coordenadas nuevas: $Z = X V_m$ (las primeras $m$ columnas de $V$).

</Caja>

<div class="pequeno tenue">

La SVD evita formar $S$ y es más estable numéricamente.

</div>

</div>
</div>

---

# ¿Cuántas componentes necesitan los dígitos?

<div class="cols cols-6-4" style="gap: 40px">
<div>

<VarianzaAcumulada />

</div>
<div>

De 64 dimensiones, **21** conservan el 90 % de la varianza.

<v-click>

La curva crece rápido y luego se aplana: los dígitos viven cerca de un subespacio mucho más pequeño que $\R^{64}$.

</v-click>

<v-click>

<Caja tipo="cuidado">

90 % de la varianza no es 90 % de la información útil para una tarea. Una dirección con poca varianza puede ser justo la que distingue un 3 de un 8.

</Caja>

</v-click>
</div>
</div>

---

# Las componentes también son imágenes

<ComponentesPCA />

<div class="cols" style="gap: 40px; margin-top: 30px">
<div class="pequeno">

Cada componente es una dirección en $\R^{64}$: un patrón de píxeles. En **azul** suma tinta, en **rojo** la quita.

</div>
<div class="pequeno" v-click>

Todo dígito se escribe como el promedio más una combinación de estos patrones:

$$
\vx \approx \bar\vx + z_1 \mathbf u_1 + z_2 \mathbf u_2 + \dots + z_m \mathbf u_m .
$$

</div>
</div>

---

# Reconstruir con $m$ componentes

<Reconstrucciones />

<div class="cols" style="gap: 40px; margin-top: 36px">
<div class="pequeno">

Con 10 números ya se reconoce el dígito; con 40 casi no se distingue del original.

</div>
<div class="pequeno" v-click>

Lo que se descarta son las direcciones de menor varianza: muchas veces ruido. Por eso PCA también se usa para **limpiar** y para **comprimir** antes de entrenar.

</div>
</div>

---

# La escala manda

<div class="cols" style="gap: 56px">
<div>

### Sin estandarizar

Pingüinos con pico y aleta en milímetros y masa en **gramos**.

<div class="comparacion" style="margin-top: 18px">
<div>
<span class="comparacion__etiqueta">varianza en la PC1</span>
<span class="comparacion__cifra">99.99 %</span>
<span class="comparacion__nota">La PC1 es, prácticamente, la masa.</span>
</div>
</div>

</div>
<div v-click>

### Estandarizadas

Cada medida con media 0 y desviación 1.

<div class="comparacion" style="margin-top: 18px">
<div>
<span class="comparacion__etiqueta">varianza en la PC1</span>
<span class="comparacion__cifra">69 %</span>
<span class="comparacion__nota">Combina las cuatro medidas: el «tamaño» del pingüino.</span>
</div>
</div>

</div>
</div>

<div v-click class="nota">

PCA busca varianza, y la varianza depende de las unidades. Igual que en k-medias: estandarizar es una decisión de modelado, no un detalle técnico.

</div>

---

# PCA para ver: los dígitos en dos dimensiones

<div class="cols" style="grid-template-columns: 5fr 3fr; gap: 40px">
<div>

<Embebido metodo="pca" :ancho="640" :alto="480" />

</div>
<div>

Las dos primeras componentes conservan el **29 %** de la varianza.

<v-click>

Algunos dígitos se separan (0, 6, 4), pero la mayoría se enciman.

</v-click>

<v-click>

PCA es **lineal**: solo puede rotar y proyectar. Si los datos viven en una superficie curva, ninguna sombra plana la desdobla.

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# Métodos no lineales

Conservar a los vecinos, no las distancias.

---

# t-SNE: que los vecinos sigan siendo vecinos

<div class="cols" style="gap: 48px">
<div>

<v-clicks>

1. En el espacio original, convertir distancias en probabilidades de ser vecinos: $p_{j \mid i} \propto \exp\!\big(-\norm{\vx_i - \vx_j}^2 / 2\sigma_i^2\big)$.
2. Colocar puntos $\mathbf y_i$ en el plano, con probabilidades $q_{ij} \propto \big(1 + \norm{\mathbf y_i - \mathbf y_j}^2\big)^{-1}$.
3. Mover los $\mathbf y_i$ para que $q$ se parezca a $p$: minimizar $\mathrm{KL}(P \,\|\, Q)$ por descenso de gradiente.

</v-clicks>

<p class="pequeno tenue" style="margin-top: 20px">L. van der Maaten y G. Hinton, <em>JMLR</em> 9 (2008).</p>

</div>
<div v-click>

<Caja tipo="idea">

La distribución $t$ de colas pesadas en el plano deja que los puntos **no vecinos** se alejen mucho. Así hay espacio para todos y los grupos se separan.

La **perplejidad** fija $\sigma_i$: es, aproximadamente, cuántos vecinos considera cada punto.

</Caja>

</div>
</div>

---

# Los mismos dígitos: PCA contra t-SNE

<div class="cols" style="gap: 40px">
<div><Embebido metodo="pca" :ancho="560" :alto="440" titulo="PCA" :radio="2.6" /></div>
<div><Embebido metodo="tsne" :ancho="560" :alto="440" titulo="t-SNE, perplejidad 30" :radio="2.6" /></div>
</div>

<!--
Nadie le dijo a t-SNE qué dígito es cada punto: los colores se agregan después. Que los grupos salgan separados dice que en R^64 los vecindarios ya estaban bien formados.
-->

---

# La perplejidad cambia el dibujo

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 24px">
<div><Embebido metodo="perp5" :ancho="380" :alto="340" titulo="perplejidad 5" :radio="2.6" /></div>
<div><Embebido metodo="perp30" :ancho="380" :alto="340" titulo="perplejidad 30" :radio="2.6" /></div>
<div><Embebido metodo="perp100" :ancho="380" :alto="340" titulo="perplejidad 100" :radio="2.6" /></div>
</div>

<div v-click class="nota">

600 dígitos. Con perplejidad baja aparecen grupitos que no existen; con alta, el dibujo se parece más a PCA. No hay un valor correcto: hay que probar varios.

</div>

---

# t-SNE no conserva los tamaños

<TrampaTSNE />

<div v-click class="nota">

Dos nubes en 10 dimensiones, la de la derecha **8 veces más dispersa**. t-SNE las dibuja casi del mismo tamaño. Tampoco significan nada las **distancias entre grupos**. M. Wattenberg, F. Viégas e I. Johnson, «How to Use t-SNE Effectively», *Distill* (2016).

</div>

---

# UMAP

<div class="cols" style="grid-template-columns: 5fr 3fr; gap: 40px">
<div>

<Embebido metodo="umap" :ancho="620" :alto="460" />

</div>
<div>

La misma idea, con otra matemática: construir un **grafo de vecinos** y buscar un dibujo en el plano que lo respete.

<v-clicks>

- Suele conservar mejor la estructura **global**.
- Escala mejor a cientos de miles de puntos.
- Tiene `transform`: proyecta **puntos nuevos** sin rehacer todo. t-SNE no.

</v-clicks>

<p class="pequeno tenue" style="margin-top: 20px">L. McInnes, J. Healy y J. Melville, arXiv:1802.03426 (2018).</p>

</div>
</div>

---

# ¿Cuál usar?

<div style="margin-top: 10px">

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| tipo | lineal | no lineal | no lineal |
| conserva | varianza global | vecindarios locales | vecindarios, algo de lo global |
| puntos nuevos | sí, `transform` | no | sí, `transform` |
| para preprocesar | sí | no | con cuidado |
| ¿distancias y tamaños interpretables? | sí | no | no del todo |
| determinista | sí | no (semilla) | no (semilla) |

</div>

<div v-click class="nota">

Regla práctica: empezar con PCA. Si hace falta un mapa para **explorar**, t-SNE o UMAP, sin sacar conclusiones de tamaños ni distancias entre grupos.

</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 26px; line-height: 1.4">

1. En dimensión alta las distancias se parecen todas: reducir ayuda a ver, comprimir y aprender.
2. **PCA**: la proyección de máxima varianza, que es la de mínimo error de reconstrucción; vectores propios o SVD.
3. Cuántas componentes: varianza acumulada, pero pensando en la tarea.
4. La **escala** decide qué es varianza.
5. **t-SNE** y **UMAP** conservan vecindarios; son para mirar, no para medir.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Reducir dimensión: PCA, t-SNE y UMAP* (Unidad 2, ejemplo 3).

**Laboratorio 2:** k-medias y PCA desde cero, y cuantizar los colores de una foto.

**Lectura:** James et al., *An Introduction to Statistical Learning*, sección 12.2.

</div>
</div>
</div>
