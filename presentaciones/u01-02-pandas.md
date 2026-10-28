---
theme: none
title: Datos tabulares con pandas
unidad: Unidad 1
layout: portada
subtitulo: Cinco operaciones, tres errores silenciosos y una pregunta sobre las presas de Sonora
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

# Datos tabulares con pandas

---
layout: pregunta
---

# ¿Cuánta agua tienen hoy las presas de Sonora? ¿Y cómo cambia a lo largo del año?

<v-click>

Antes de pensar en un modelo para pronosticarlo, hay que poder **responder preguntas con los datos**. Esa es la habilidad de hoy.

</v-click>

<!--
Esta pregunta acompaña toda la clase y es la base del proyecto. Preguntar si alguien sabe de qué presa sale el agua de Hermosillo.
-->

---

# Los datos: presas de Sonora

<div class="cols cols-6-4" style="gap: 50px">
<div>

### Registros diarios · 11,533 filas

<TablaMini :columnas="['clave', 'fecha', 'almacenamiento']" :filas="[['LCDSO', '2020/01/01', 738.54], ['PECSO', '2020/01/01', 2283.69], ['AOBSO', '2020/01/01', 1683.92], ['AGZCH', '2020/01/01', 79.1], ['…', '…', '…']]" />

### Catálogo · una fila por presa

<TablaMini :columnas="['Clave', 'Nombre_comun', 'Río', 'cap_namo']" :filas="[['LCDSO', 'La Angostura', 'Río Yaqui', 703], ['PECSO', 'El Novillo', 'Río Yaqui', 2833], ['AOBSO', 'Oviachic', 'Río Yaqui', 3023], ['…', '…', '…', '…']]" />

</div>
<div class="pequeno">

Almacenamiento diario de 10 presas, de enero de 2020 a junio de 2023, en **hm³** (millones de m³).

<div style="font-size: 19px; color: var(--tenue); margin: -6px 0 22px">Fuente: portal de datos abiertos del Gobierno de Sonora (datos.sonora.gob.mx), conjunto «Recursos Hídricos», SAGARHPA, CC BY.</div>

**`cap_namo`**: capacidad al nivel de aguas máximas **ordinarias**, lo que la presa guarda en operación normal. El porcentaje de llenado se calcula respecto a ella.

<div v-click class="lema" style="font-size: 26px; margin-top: 30px">
Fíjense en la fecha: ¿es una fecha o un <em>texto</em> que parece fecha?
</div>

</div>
</div>

---
layout: seccion
numero: 1
---

# La tabla

Qué es una tabla «ordenada» y cómo la guarda pandas.

---

# Datos ordenados

<div class="cols cols-4-6 cols-centro" style="flex: 1">
<div class="grande">

1. Cada **variable** es una columna.
2. Cada **observación** es una fila.
3. Cada **tipo de unidad** observada es una tabla.

<p class="pequeno tenue" style="margin-top: 20px">H. Wickham, «Tidy Data», <em>Journal of Statistical Software</em> 59 (2014).</p>

</div>
<div>

<TablaMini :columnas="['clave', 'fecha', 'almacenamiento']" :filas="[['LCDSO', '2020-01-01', 738.5], ['PECSO', '2020-01-01', 2283.7], ['LCDSO', '2020-01-02', 738.2], ['PECSO', '2020-01-02', 2284.1]]" :tam="22" />

<div class="pequeno tenue" style="margin-top: 18px">

Observación: «una presa en un día». Variables: qué presa, qué día, cuánta agua. Las propiedades fijas de cada presa (nombre, río, capacidad) son **otro tipo de unidad**: van en otra tabla, el catálogo.

</div>
</div>
</div>

<!--
Preguntar: ¿por qué no poner el nombre de la presa en cada registro diario? Respuesta: se repetiría 1,273 veces y, si se corrige, hay que corregirlo en todos lados.
-->

---

# Cada columna tiene un tipo

| Tipo | Ejemplo en nuestros datos | Qué se puede hacer |
|---|---|---|
| Numérica continua | almacenamiento: 738.54 hm³ | sumar, promediar, graficar |
| Numérica discreta | número de recámaras de una casa | contar, promediar con cuidado |
| Categórica nominal | río: Yaqui, Mayo, Sonora | contar, agrupar; **no** ordenar |
| Categórica ordinal | calidad: mala < regular < buena | ordenar, comparar |
| Fecha | 2022-09-15 | restar, extraer el mes, agrupar por periodos |
| Texto libre | el nombre de un pasajero | extraer partes (p. ej. el título) |

<div class="nota">

El **tipo** decide qué operaciones tienen sentido, y más adelante cómo se codifica la columna para un modelo.

</div>

---

# Cómo guarda pandas una tabla

<div class="cols cols-centro" style="gap: 60px; flex: 1">
<div>

```python
df.dtypes
```

```text
clave                     str
fecha              datetime64
almacenamiento        float64
presa                     str
cap_namo                int64
```

</div>
<div class="grande">

Un `DataFrame` es un conjunto de **columnas**; cada una es un arreglo de NumPy con **un solo tipo**.

<v-click>

Por eso las operaciones con columnas completas son rápidas y los ciclos `for` sobre filas son lentos:

```python
# 11,533 multiplicaciones en una línea, sin ciclos
df["llenado"] = 100 * df["almacenamiento"] / df["cap_namo"]
```

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# Cinco operaciones

Con ellas se responde casi cualquier pregunta sobre una tabla.

---

# Seleccionar y filtrar

<div class="cols" style="gap: 56px">
<div>

### Columnas

```python
df[["presa", "llenado"]]
```

<TablaMini :columnas="['clave', 'fecha', 'presa', 'llenado']" :filas="[['LCDSO', '2022-09-15', 'La Angostura', 89.2], ['PECSO', '2022-09-15', 'El Novillo', 99], ['AOBSO', '2022-09-15', 'Oviachic', 36.9], ['ARLSO', '2022-09-15', 'Hermosillo', 17.1]]" :marcar-cols="[2, 3]" />

</div>
<div v-click>

### Filas que cumplen una condición

```python
df.loc[df["llenado"] > 50]
df.query("llenado > 50")
```

<TablaMini :columnas="['clave', 'fecha', 'presa', 'llenado']" :filas="[['LCDSO', '2022-09-15', 'La Angostura', 89.2], ['PECSO', '2022-09-15', 'El Novillo', 99], ['AOBSO', '2022-09-15', 'Oviachic', 36.9], ['ARLSO', '2022-09-15', 'Hermosillo', 17.1]]" :atenuar-filas="[2, 3]" />

</div>
</div>

<div class="nota">

`df["llenado"] > 50` no es un número: es una **columna de verdaderos y falsos** (una máscara) que selecciona filas.

</div>

---

# Crear columnas

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<div>

```python
df["llenado"] = 100 * df["almacenamiento"] / df["cap_namo"]
df["mes"] = df["fecha"].dt.month
```

<TablaMini :columnas="['presa', 'almacenamiento', 'cap_namo', 'llenado']" :filas="[['La Angostura', 627.1, 703, 89.2], ['El Novillo', 2805.8, 2833, 99], ['Oviachic', 1115.7, 3023, 36.9], ['Hermosillo', 37.7, 220, 17.1]]" :nueva-col="3" :tam="20" />

</div>
<div class="grande">

¿Cuál presa tiene **más agua**? ¿Cuál está **más llena**?

<v-click>

Son preguntas distintas. El Novillo guarda 75 veces más agua que Hermosillo; para compararlas hace falta una **escala común**.

</v-click>
</div>
</div>

---

# Agrupar: separar, aplicar, combinar

<div class="figura-centro"><SepararAplicarCombinar /></div>

```python
df.groupby("rio")["llenado"].mean()
```

<!--
Datos del 15 de septiembre de 2022. Mostrar que groupby no es más que tres pasos. Preguntar: ¿qué cambia si en lugar de mean aplicamos max? ¿y count?
-->

---

# Unir dos tablas

<div class="figura-centro"><UnionTablas /></div>

<div class="nota">

`how="left"` conserva todos los registros aunque no tengan pareja en el catálogo; `how="inner"` solo los que tienen.

</div>

---

# Reformatear: largo y ancho

<div class="figura-centro"><LargoAncho /></div>

<div class="cols" style="gap: 48px">
<div class="pequeno">

**Largo**: cómodo para filtrar, agrupar y guardar. Es el formato «ordenado».

</div>
<div class="pequeno">

**Ancho**: cómodo para graficar series y para ver huecos día por día, como veremos ahora.

</div>
</div>

---
layout: seccion
numero: 3
---

# Errores silenciosos

Lo peligroso no es lo que truena, sino lo que funciona mal sin avisar.

---

# 1 · La fecha que era texto

<div class="cols codigo-chico" style="gap: 40px">
<div>

```python
df["fecha"].dtype       # str
sorted(["2020/10/01", "2020/2/01", "2020/9/01"])
```

```text
['2020/10/01', '2020/2/01', '2020/9/01']
```

<p class="tenue">Orden alfabético: octubre antes que febrero.</p>

</div>
<div v-click>

```python
df["fecha"] = pd.to_datetime(
    df["fecha"], format="%Y/%m/%d")
df["fecha"].dt.month      # ahora sí
df.resample("MS", on="fecha").mean()
```

<div class="lema" style="margin-top: 24px; font-size: 26px">
Revisen siempre <code>df.dtypes</code> después de leer un archivo.
</div>

</div>
</div>

---
layout: pregunta
---

# `df.isna().sum()` da cero en todas las columnas. ¿Faltan datos?

---

# 2 · Faltantes que no se ven

<p class="tenue" style="margin-bottom: 12px">Días con registro de cada presa, enero de 2020 a junio de 2023. <span style="color:#D5433C">En rojo, los días sin registro.</span></p>

<CoberturaPresas />

<div class="cols" style="gap: 48px">
<div class="pequeno">

En una serie diaria, un día faltante **no es un `NaN`: es una fila que no existe**. Se descubre comparando contra todas las fechas del periodo con `reindex`.

</div>
<div class="pequeno">

Los Pilares no tiene huecos: **entró tarde** al registro. El Novillo y El Molinito tienen un hueco de ocho días en marzo de 2023.

</div>
</div>

---

# 3 · Uniones que duplican

<div class="cols cols-centro" style="grid-template-columns: 7fr 3fr; gap: 36px; flex: 1">
<UnionTablas duplicado :tam="16" />
<div class="pequeno">

Alguien agregó al catálogo El Novillo con su nombre oficial, «Plutarco Elías Calles».

Nada truena: **las filas se duplican en silencio** y las sumas de El Novillo salen al doble.

<v-click>

**Defensa:** `validate="many_to_one"` en el `merge`, y contar filas antes y después.

</v-click>
</div>
</div>

---

# Un valor raro no siempre es un error

<PresasMensual />

<div class="nota">

La Angostura pasó del 100 %. ¿Error de captura? No: en avenidas el agua sube por encima del nivel **ordinario** sin pasar del extraordinario. Antes de «limpiar» un valor, pregúntense si es **imposible** o solo **inusual**.

</div>

---

# Un promedio sin contexto miente

<div class="cols cols-6-4 cols-centro" style="flex: 1">
<PresasPromedio />
<div class="grande">

Llenado promedio de 2020 a 2023.

<v-click>

Los Pilares solo tiene datos de **tres meses**: su promedio no se compara con el de las demás.

Hermosillo, en cambio, sí estuvo casi vacía todo el periodo: es un dato real, y una mala noticia.

</v-click>
</div>
</div>

---
layout: seccion
numero: 4
---

# De la tabla a la respuesta

Volvamos a la pregunta del principio.

---

# ¿Cómo cambia el agua a lo largo del año?

<PresasAnual :ancho="1100" :alto="380" />

<div class="cols codigo-chico" style="gap: 40px; margin-top: 8px">

```python
df.groupby(df["fecha"].dt.month)["llenado"].mean()
```

<div class="pequeno">

Baja de enero a julio (riego y evaporación) y sube en agosto y septiembre, con el monzón. Esta **estacionalidad** será clave en la Unidad 7.

</div>
</div>

---

# Antes de modelar, revisen

<div class="cols" style="gap: 60px; margin-top: 10px">
<div class="grande">

1. ¿De dónde vienen los datos y qué significa **cada columna**?
2. ¿Los **tipos** son los correctos?
3. ¿Qué **falta**, incluyendo filas que no existen?
4. ¿Las **uniones** conservaron el número de filas?
5. ¿Los valores raros son **imposibles** o solo **inusuales**?

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Explorar datos con pandas: las presas de Sonora* (Unidad 1, ejemplo 1).

**Referencia:** la [guía de usuario de pandas](https://pandas.pydata.org/docs/user_guide/index.html), en especial «10 minutes to pandas» y «Group by».

**Lectura:** H. Wickham (2014), *Tidy Data*.

</div>
</div>
</div>
