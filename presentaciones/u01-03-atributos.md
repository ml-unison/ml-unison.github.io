---
theme: none
title: Ingeniería de atributos y pipelines
unidad: Unidad 1
layout: portada
subtitulo: Convertir datos reales en una matriz que un modelo pueda usar, sin hacer trampa
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
  Reconocimiento de Patrones · Unidad 1 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Ingeniería de atributos y pipelines

---
layout: pregunta
---

# Un modelo solo entiende matrices de números sin huecos. ¿Qué hacemos con «Excelente», con el año 1958 o con un dato que falta?

---

# Del registro a un vector

<div class="cols cols-centro" style="grid-template-columns: 5fr 1fr 4fr; gap: 20px; flex: 1">
<div>

### Un registro, tal como llega

<TablaMini :columnas="['columna', 'valor']" :filas="[['GrLivArea', 1710], ['YearBuilt', '2003'], ['Neighborhood', 'CollgCr'], ['KitchenQual', 'Gd'], ['PoolQC', '(vacío)'], ['LotFrontage', 65]]" :tam="20" />

</div>
<div class="centrado lema" style="font-size: 40px">

$\longrightarrow$

$\phi$

</div>
<div>

### Un vector de números

$$
\vx = \big(\, 0.52,\; -1.01,\; 0,\, 1,\, 0, \dots,\; 2,\; 0,\; -0.21,\; 0 \,\big)
$$

<div v-click class="pequeno" style="margin-top: 24px">

La **ingeniería de atributos** es diseñar $\phi$: la función que convierte cada registro en un vector de números. Los modelos de todo el curso trabajan sobre la matriz $X \in \R^{n \times d}$ cuyas filas son esos vectores.

</div>
</div>
</div>

---

# Los datos de hoy: casas de Ames, Iowa

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div class="grande">

- 1,460 ventas de casas entre 2006 y 2010.
- 79 atributos: áreas, años, barrio, calidades, alberca, cochera…
- **Problema:** estimar el precio de venta.

<p class="pequeno tenue">D. De Cock, «Ames, Iowa: Alternative to the Boston Housing Data», <em>Journal of Statistics Education</em> 19 (2011). Versión de OpenML.</p>

</div>
<div class="lema" style="font-size: 26px">
Tiene, en una sola tabla, casi todos los problemas que encontrarán en datos reales.
</div>
</div>

---
layout: seccion
numero: 1
---

# El contrato `fit` / `transform`

La regla que evita casi todas las fugas.

---

# Aprender los parámetros, aplicarlos después

<FitTransform />

<div v-click class="nota">

Toda transformación de scikit-learn sigue este contrato: escaladores, imputadores, codificadores y también los modelos (`fit` / `predict`). Los parámetros $\theta$ salen **solo** del entrenamiento.

</div>

---
layout: pregunta
---

# Para estandarizar el conjunto de prueba, ¿qué media usamos: la del entrenamiento o la de la prueba?

<v-click>

La del entrenamiento. En producción llega **una casa a la vez**: no hay «media de la prueba» que calcular. Evaluar como si la hubiera es hacer trampa.

</v-click>

---
layout: seccion
numero: 2
---

# Atributos numéricos

Escalas, sesgos y fechas.

---

# ¿Qué casa se parece más a A?

<DistanciaEscala />

---

# Estandarizar

<div class="cols cols-6-4" style="gap: 56px">
<div>

Para cada columna $j$, con $\bar x_j$ y $s_j$ calculadas **en el entrenamiento**:

$$
z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}
$$

Cada columna queda con media 0 y desviación 1 (en el entrenamiento; en la prueba, aproximadamente).

<v-click>

Alternativa: llevar al intervalo $[0, 1]$ con mínimo y máximo. Es más sensible a los valores extremos.

</v-click>
</div>
<div v-click>

### Importa en

- métodos basados en **distancias** ($k$ vecinos, agrupamiento, SVM);
- modelos con **regularización**, que castigan igual a todos los coeficientes;
- **descenso de gradiente** y redes neuronales.

### No importa en

- árboles y sus ensambles: solo comparan valores dentro de una misma columna.

</div>
</div>

---

# Variables sesgadas: el logaritmo

<HistogramasPrecio />

<div class="cols" style="gap: 48px; margin-top: 4px">
<div class="pequeno">

Muchas casas modestas y pocas muy caras. Con el logaritmo, la distribución es casi simétrica.

</div>
<div class="pequeno">

Y los errores se vuelven **relativos**: $\;\log \hat y - \log y \approx \dfrac{\hat y - y}{y}$. Equivocarse por 20 mil dólares pesa distinto en una casa de 100 mil que en una de 500 mil.

</div>
</div>

---

# Fechas: convertirlas en algo con sentido

<div class="cols cols-centro" style="grid-template-columns: 3fr 2fr; gap: 48px; flex: 1">
<div>

<TablaMini :columnas="['YearBuilt', 'YearRemodAdd', 'YrSold', 'edad', 'desde_remodelación']" :filas="[['2003', '2003', '2008', 5, 5], ['1976', '1976', '2007', 31, 31], ['1915', '1970', '2006', 91, 36]]" :nueva-col="3" :tam="20" />

```python
def edades(t):
    return pd.DataFrame({
        "edad": t["YrSold"] - t["YearBuilt"],
        "desde_remodelación": t["YrSold"] - t["YearRemodAdd"]})
```

</div>
<div class="grande">

«1958» no es una cantidad que tenga sentido multiplicar por un coeficiente. La **edad al venderse**, sí.

<v-click>

Es una resta: **no aprende** nada de los datos. Aun así va dentro del `Pipeline`, para aplicarse igual a cada casa nueva.

</v-click>
</div>
</div>

---
layout: seccion
numero: 3
---

# Atributos categóricos

Con orden y sin orden.

---

# Sin orden: *one-hot*

<div class="cols cols-centro" style="grid-template-columns: 1fr 1fr; gap: 48px; flex: 1">
<div style="display: flex; gap: 22px; align-items: center">

<TablaMini :columnas="['Neighborhood']" :filas="[['CollgCr'], ['Veenker'], ['CollgCr'], ['NoRidge']]" :tam="21" />

<span style="font-size: 30px; color: #8DA3C9">⟶</span>

<TablaMini :columnas="['CollgCr', 'NoRidge', 'Veenker']" :filas="[[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]" :tam="21" :marcar-cols="[0, 1, 2]" />

</div>
<div class="pequeno">

Una columna binaria por categoría: $k$ categorías → $k$ columnas.

<v-click>

- `handle_unknown="ignore"`: un barrio nuevo queda en ceros.
- `min_frequency=10`: los barrios con pocas casas se agrupan.
- Sin regularización, las $k$ columnas suman 1 y son colineales con el intercepto: se quita una (`drop="first"`).

</v-click>
</div>
</div>

---

# Con orden: codificación ordinal

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<CalidadCocina />
<div class="grande">

Precio mediano según la calidad de la cocina. Codificamos Regular = 0, Típica = 1, Buena = 2, Excelente = 3.

<v-click>

<Caja tipo="pregunta">

¿Es lo mismo pasar de Regular a Típica que de Buena a Excelente? La codificación ordinal **supone pasos iguales**; los datos dicen otra cosa.

</Caja>

</v-click>
</div>
</div>

---

# ¿Cuál codificación?

| La columna… | Codificación | Ejemplo |
|---|---|---|
| no tiene orden | *one-hot* | barrio, tipo de construcción |
| tiene orden, y el efecto crece con él | ordinal, con el orden **explícito** | calidades, nivel de estudios |
| tiene orden, pero los pasos son muy desiguales | *one-hot*, o ordinal con un modelo de árboles | calidad, si el modelo es lineal y hay muchos datos |
| tiene cientos de categorías | agrupar las raras, o técnicas que veremos después | código postal, modelo de auto |

<div class="nota">

Cuidado: sin orden explícito, `OrdinalEncoder` usa el **alfabético**, y queda Excelente < Regular < Buena < Típica.

</div>

---
layout: seccion
numero: 4
---

# Faltantes y atípicos

Antes de rellenar o quitar, preguntar por qué.

---
layout: pregunta
---

# La columna «calidad de la alberca» tiene 99.5 % de faltantes. ¿La borramos, o rellenamos con la categoría más común?

<v-click>

Ninguna de las dos. Ese faltante significa **«no tiene alberca»**. Es información, y se codifica como una categoría propia.

</v-click>

---

# Dos clases de faltante

<div class="cols" style="gap: 56px; margin-top: 6px">
<div>

### El faltante significa algo

`PoolQC`, `FireplaceQu`, `BsmtQual`: la casa **no tiene** alberca, chimenea o sótano.

<TablaMini :columnas="['FireplaceQu', '→', 'codificado']" :filas="[['Gd', '', 'Gd'], ['(vacío)', '', 'Ninguna'], ['TA', '', 'TA']]" :tam="20" :nueva-col="2" />

<p class="pequeno tenue" style="margin-top: 14px">Precio mediano: 192 mil dólares con chimenea y 135 mil sin chimenea. Imputar con «la más común» habría borrado esa diferencia.</p>

</div>
<div v-click>

### El dato se perdió

`LotFrontage`, el frente del terreno: toda casa tiene uno.

<TablaMini :columnas="['LotFrontage', '→', 'imputado', 'faltaba']" :filas="[[65, '', 65, 0], ['(vacío)', '', 70, 1], [80, '', 80, 0]]" :tam="20" :nueva-col="3" />

<p class="pequeno tenue" style="margin-top: 14px">Mediana aprendida en el entrenamiento (70), más una columna que indica qué se imputó: a veces, que falte un dato también es información.</p>

</div>
</div>

---

# Valores atípicos: entender antes de quitar

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<DispersionCasas />
<div class="pequeno">

Las cuatro casas de más de 4,000 pies² no son iguales:

- dos son **ventas parciales** de casas sin terminar, que no representan el mercado;
- dos son casas grandes y caras, con precios razonables.

<v-click>

El autor de los datos recomienda excluirlas todas al enseñar.

**Reglas:** entender el origen; quitar solo del **entrenamiento**; nunca quitar «porque empeoran la métrica»; documentar la decisión.

</v-click>
</div>
</div>

---
layout: seccion
numero: 5
---

# Pipelines

Todo junto, y sin fugas.

---

# Cada columna con su tratamiento

<DiagramaPipeline />

---

# En código

<div class="codigo-chico">

```python
imputar = SimpleImputer(strategy="median", add_indicator=True)
preprocesamiento = ColumnTransformer([
    ("sesgadas",  make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), sesgadas),
    ("numericas", make_pipeline(imputar, StandardScaler()), numericas),
    ("edades",    make_pipeline(FunctionTransformer(edades), StandardScaler()), fechas),
    ("nominales", OneHotEncoder(handle_unknown="ignore", min_frequency=10), nominales),
    ("ordinales", make_pipeline(calidad, StandardScaler()), ordinales),
])
modelo = TransformedTargetRegressor(regressor=make_pipeline(preprocesamiento, RidgeCV()),
                                    func=np.log, inverse_func=np.exp)

modelo.fit(X_entrenamiento, y_entrenamiento)   # aprende TODO: medianas, categorías, medias, coeficientes
modelo.predict(X_nuevo)                        # aplica TODO, en el mismo orden
```

</div>

<div class="nota">

Un solo objeto: se entrena, se valida, se guarda y se despliega completo. En la Unidad 8 lo serviremos así.

</div>

---
layout: pregunta
---

# Si nunca tocamos el conjunto de prueba, ¿ya no hay fugas?

<v-click>

Falta un lugar: **dentro de la validación cruzada**. Ahí también hay una parte que el modelo no debería ver.

</v-click>

---

# Fugas dentro de la validación cruzada

<div class="cols" style="gap: 56px">
<div>

### Escalar antes de validar

<PliegosCV />

<p class="pequeno">En cada iteración, la parte de validación (<span style="color:#A87C12; font-weight:650">dorado</span>) <b>ya influyó</b> en la media y la desviación. La estimación sale optimista.</p>

</div>
<div v-click>

### Escalar dentro del `Pipeline`

<PliegosCV correcto />

<p class="pequeno">En cada iteración, el escalador se ajusta solo con las cuatro partes de entrenamiento (marcadas arriba). Es exactamente lo que pasará en producción.</p>

</div>
</div>

---

# ¿Cuánto ayuda preparar bien los datos?

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div class="comparacion">
  <div><span class="comparacion__etiqueta">ingenuo</span><span class="comparacion__cifra tenue">17,072</span></div>
  <div><span class="comparacion__etiqueta">con preprocesamiento</span><span class="comparacion__cifra">11,659</span></div>
  <div class="comparacion__nota">dólares de error mediano · <b>32 % menos</b></div>
</div>
<div class="grande">

Error mediano, con validación cruzada, del **mismo modelo**.

<v-click>

**Ingenuo:** solo columnas numéricas, faltantes con la media y precio en dólares.

**Con preprocesamiento:** todo lo de hoy.

</v-click>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div class="grande">

1. Ingeniería de atributos es diseñar $\phi$: del registro a un vector.
2. `fit` aprende con el entrenamiento; `transform` aplica lo aprendido a todo.
3. El **tipo** de cada columna decide su tratamiento.
4. Antes de imputar o quitar, preguntar **por qué**.
5. Todo dentro de un `Pipeline`: también la validación cruzada queda honesta.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libretas:** *Preprocesamiento: preparar datos reales para un modelo* y *Un primer pipeline sin trampas* (Unidad 1, ejemplos 2 y 3).

**Laboratorio 1:** construir y evaluar un `Pipeline` completo.

**Referencia:** guía de usuario de scikit-learn, secciones *Pipelines and composite estimators* y *Preprocessing data*.

</div>
</div>
</div>
