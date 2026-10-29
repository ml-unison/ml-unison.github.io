# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <small>Reconocimiento de Patrones · Licenciatura en Ciencias de la Computación · Universidad de Sonora<br>
# Julio Waissman · <a href="https://ml-unison.github.io">ml-unison.github.io</a> · Licencia CC BY-SA 4.0</small>
#
# ---
#
# # Explorar datos con pandas: las presas de Sonora
#
# **Unidad 1 · Ejemplo 1**
#
# Sonora es un estado semiárido. El agua para las ciudades y para los distritos de riego de los valles del Yaqui y del Mayo depende en buena parte de unas cuantas presas. Saber cuánta agua tienen, y cómo cambia a lo largo del año, es una pregunta con consecuencias económicas y sociales.
#
# Antes de pensar en cualquier modelo, hay que **entender los datos**. En esta libreta vamos a:
#
# 1. leer los datos y revisar qué tipo tiene cada columna,
# 2. unir dos tablas (registros diarios y catálogo de presas),
# 3. crear columnas nuevas, filtrar y agrupar,
# 4. descubrir **datos faltantes que no se ven a simple vista**,
# 5. y resumir series de tiempo por mes.
#
# Todo con [pandas](https://pandas.pydata.org/docs/user_guide/index.html), la biblioteca estándar de Python para datos tabulares.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rdp

rdp.estilo.usar()
pd.set_option("display.max_columns", 20)

# %% [markdown]
# ## 1. De dónde vienen los datos
#
# El Gobierno del Estado de Sonora publica en su portal de datos abiertos el conjunto [Recursos Hídricos](https://datos.sonora.gob.mx/dataset/Recursos%20H%C3%ADdricos), a cargo de la Secretaría de Agricultura, Ganadería, Recursos Hidráulicos, Pesca y Acuacultura (SAGARHPA), con licencia CC BY. Contiene el almacenamiento diario de las presas de Sonora desde 1941, en un archivo por década, más un catálogo de las presas.
#
# Para el curso usamos un extracto de dos tablas, guardado en el repositorio para que la libreta no dependa de que el portal esté disponible:
#
# - **`almac_reciente`**: un registro por presa y por día, de enero de 2020 a junio de 2023, con el almacenamiento en hectómetros cúbicos (1 hm³ = un millón de m³).
# - **`catalogo_estatal`**: una fila por presa, con su nombre, el río, el municipio y sus capacidades.
#
# Dos capacidades aparecen en el catálogo y conviene entenderlas antes de usarlas:
#
# | Columna | Significado |
# |---|---|
# | `cap_namo` | Capacidad al **nivel de aguas máximas ordinarias**: lo que la presa almacena en operación normal. |
# | `cap_name` | Capacidad al **nivel de aguas máximas extraordinarias**: el máximo físico, que solo se alcanza en avenidas. |
#
# Por eso, el porcentaje de llenado de una presa se calcula respecto al NAMO.

# %%
registros = rdp.datos.cargar("almac_reciente")
catalogo = rdp.datos.cargar("catalogo_estatal", sheet_name="Catálogo_estatal")

registros.head()

# %% [markdown]
# ## 2. Primer vistazo
#
# Antes de calcular cualquier cosa, tres preguntas: ¿cuántas filas y columnas hay?, ¿de qué tipo es cada columna?, ¿qué valores toman?

# %%
print(registros.shape)
registros.info()

# %% [markdown]
# Hay un problema: la columna `fecha` es **texto**, no una fecha. Mientras sea texto no podremos ordenar bien, extraer el mes ni agrupar por periodos. Es uno de los errores más comunes al leer datos.

# %%
registros["fecha"] = pd.to_datetime(registros["fecha"], format="%Y/%m/%d")
registros.dtypes

# %%
registros.describe()

# %% [markdown]
# El mínimo es 0: alguna presa estuvo prácticamente vacía. ¿Cuántas presas hay y cuántos registros tiene cada una?

# %%
registros["clave"].value_counts()

# %% [markdown]
# La presa `BICSO` tiene solo 92 registros, cuando las demás tienen alrededor de 1,270. Anotamos esto; lo investigaremos en la sección 6.

# %% [markdown]
# ## 3. Unir con el catálogo
#
# Las claves como `PECSO` no le dicen nada a nadie. El catálogo tiene el nombre común de cada presa, el río y su capacidad. Nos quedamos con las columnas útiles y les damos nombres consistentes.

# %%
presas = (
    catalogo[["Clave", "Nombre_comun", "Río", "Municipio", "Estado", "cap_namo", "cap_name"]]
    .rename(columns={"Clave": "clave", "Nombre_comun": "presa", "Río": "rio",
                     "Municipio": "municipio", "Estado": "estado"})
)
presas

# %% [markdown]
# Unimos las dos tablas por la columna `clave` con `merge`. El argumento `validate="many_to_one"` le pide a pandas que **verifique** que cada clave aparece una sola vez en el catálogo: si no fuera así, la unión duplicaría registros sin avisar.

# %%
df = registros.merge(presas, on="clave", how="left", validate="many_to_one")
print(df.shape)
df.head()

# %% [markdown]
# Después de una unión siempre hay que comprobar dos cosas: que no cambió el número de filas y que no quedaron registros sin pareja.

# %%
assert len(df) == len(registros)
df["presa"].isna().sum()

# %% [markdown]
# ## 4. Columnas nuevas, filtros y selección
#
# El almacenamiento en hm³ no permite comparar presas de tamaños muy distintos: El Novillo guarda casi 3,000 hm³ y Punta de Agua apenas 16. Lo comparable es el **porcentaje de llenado**.

# %%
df["llenado"] = 100 * df["almacenamiento"] / df["cap_namo"]
df[["fecha", "presa", "almacenamiento", "cap_namo", "llenado"]].sample(5, random_state=1)

# %% [markdown]
# Para filtrar filas usamos una máscara booleana con `.loc`, o bien `.query`, que se lee casi como una oración.

# %%
novillo_2022 = df.loc[(df["presa"] == "El Novillo") & (df["fecha"].dt.year == 2022)]
novillo_2022[["fecha", "almacenamiento", "llenado"]].head()

# %%
df.query("llenado > 100")[["fecha", "presa", "llenado"]]

# %% [markdown]
# Algunas presas superaron el 100 % de su NAMO. No es un error de captura: en temporada de avenidas el agua puede subir por encima del nivel ordinario, sin pasar del extraordinario (`cap_name`). **Antes de «limpiar» un valor raro, hay que entender si es imposible o solo inusual.**

# %% [markdown]
# ## 5. Agrupar y resumir
#
# `groupby` divide la tabla en grupos, aplica una función a cada uno y junta los resultados. Es la operación más útil de pandas.

# %%
resumen = (
    df.groupby("presa")["llenado"]
    .agg(["mean", "min", "max"])
    .round(1)
    .sort_values("mean", ascending=False)
)
resumen

# %%
fig, ax = plt.subplots(figsize=(7, 4))
orden = resumen.sort_values("mean")
ax.barh(orden.index, orden["mean"], color=rdp.estilo.AZUL)
ax.set(title="Llenado promedio, 2020–2023", xlabel="% del NAMO", xlim=(0, 100))
ax.grid(axis="y", visible=False)
for y, v in enumerate(orden["mean"]):
    ax.text(v + 1.5, y, f"{v:.0f} %", va="center", fontsize=10, color=rdp.estilo.TENUE)
plt.show()

# %% [markdown]
# Dos observaciones. La presa Abelardo Rodríguez Luján («Hermosillo» en el catálogo), que históricamente abasteció a la ciudad, pasó casi todo el periodo prácticamente vacía: un modelo que pronostique su almacenamiento lidiaría con una serie casi constante en cero. Y el promedio de Los Pilares es engañoso: como veremos en la siguiente sección, solo tiene tres meses de datos. **Un promedio sin contexto puede decir poco o mentir.**

# %% [markdown]
# ## 6. Datos faltantes que no se ven
#
# `df.isna().sum()` dice que no hay faltantes. Pero en una serie diaria un día faltante no aparece como `NaN`: **simplemente no está la fila**. Para verlo hay que comparar las fechas que tenemos con las que deberíamos tener.

# %%
df.isna().sum().sum()

# %%
dias_esperados = (df["fecha"].max() - df["fecha"].min()).days + 1
cobertura = (
    df.groupby("presa")["fecha"]
    .agg(primer_dia="min", ultimo_dia="max", registros="count")
    .assign(faltantes=lambda t: dias_esperados - t["registros"])
)
cobertura

# %% [markdown]
# Los Pilares solo tiene datos desde finales de marzo de 2023: no es una presa con huecos, es una presa que **entró tarde** al registro. El Novillo y El Molinito sí tienen días faltantes.
#
# Una forma de hacer visibles los huecos es pasar a formato **ancho** (una columna por presa, una fila por día) y reindexar con todas las fechas del periodo.

# %%
ancho = df.pivot(index="fecha", columns="presa", values="llenado")
todas = pd.date_range(ancho.index.min(), ancho.index.max(), freq="D")
ancho = ancho.reindex(todas)
ancho.isna().sum()

# %%
huecos_novillo = ancho.index[ancho["El Novillo"].isna()]
huecos_novillo

# %% [markdown]
# Es un solo hueco de ocho días seguidos, en las mismas fechas para El Novillo y El Molinito: probablemente un problema en la captura, no en las presas.
#
# ¿Qué hacer con él? Depende de para qué usaremos los datos. Para un promedio mensual, ocho días no cambian mucho. Para un modelo que use el valor del día anterior, sí importan. En una serie que cambia lentamente, una opción razonable es **interpolar**, con un límite para no inventar semanas enteras, y **anotar que se hizo**.

# %%
ancho_interp = ancho.drop(columns="Los Pilares").interpolate(limit=10)
ancho_interp.isna().sum().sum()

# %% [markdown]
# ## 7. Series de tiempo: resumir por mes
#
# Con un índice de fechas, `resample` agrupa por periodos de tiempo. Veamos el llenado mensual del sistema del río Yaqui, que abastece al valle del Yaqui.

# %%
yaqui = ["La Angostura", "El Novillo", "Oviachic"]
mensual = ancho_interp[yaqui].resample("MS").mean()

fig, ax = plt.subplots(figsize=(8.5, 4))
ax.axhline(100, color=rdp.estilo.GRIS, lw=1, ls="--")
ax.text(pd.Timestamp("2021-02-01"), 103, "100 % del NAMO", color=rdp.estilo.TENUE, fontsize=10)
for nombre in yaqui:
    ax.plot(mensual.index, mensual[nombre])
ax.set(title="Sistema del río Yaqui: llenado mensual", ylabel="% del NAMO", ylim=(0, 125))
rdp.estilo.etiquetar(ax, mensual.index[-1] + pd.Timedelta(days=25), mensual.iloc[-1].to_dict())
ax.set_xlim(right=mensual.index[-1] + pd.Timedelta(days=200))
plt.show()

# %% [markdown]
# Se ve una caída sostenida hasta mediados de 2022 y una recuperación con las lluvias de ese verano. Para separar el **ciclo anual** de la tendencia, agrupamos por mes del año, sin importar el año.

# %%
por_mes = ancho_interp[yaqui].groupby(ancho_interp.index.month).mean()
por_mes.index.name = "mes"

fig, ax = plt.subplots(figsize=(8.5, 4))
for nombre in yaqui:
    ax.plot(por_mes.index, por_mes[nombre], marker="o")
ax.set(title="Llenado promedio según el mes del año", xlabel="mes", ylabel="% del NAMO",
       xticks=range(1, 13), xticklabels=list("EFMAMJJASOND"), xlim=(0.6, 14.2))
rdp.estilo.etiquetar(ax, 12.35, por_mes.iloc[-1].to_dict())
plt.show()

# %% [markdown]
# El almacenamiento baja de enero a julio (riego y evaporación) y sube en agosto y septiembre, con el monzón. Esta **estacionalidad** será importante en la Unidad 7, cuando pronostiquemos series de tiempo.

# %% [markdown]
# ## 8. Formato largo y formato ancho
#
# Los mismos datos pueden guardarse de dos formas:
#
# - **largo** (como `df`): una fila por observación; cómodo para agrupar y filtrar;
# - **ancho** (como `ancho`): una columna por serie; cómodo para graficar y comparar.
#
# `pivot` va de largo a ancho y `melt` regresa.

# %%
largo = (
    mensual.reset_index(names="mes")
    .melt(id_vars="mes", var_name="presa", value_name="llenado")
)
largo.head()

# %% [markdown]
# ## 9. Autoevaluación

# %%
rdp.quiz.mostrar("u01", ids=["u01-p11", "u01-p12", "u01-p13"])

# %% [markdown]
# ## 10. Ejercicios
#
# 1. ¿Qué día de todo el periodo tuvo el **menor almacenamiento total** (en hm³) sumando todas las presas de Sonora? Excluye Guadalupe, que está en Chihuahua.
# 2. Para cada presa, calcula el cambio de llenado entre el 1 de enero y el 31 de diciembre de 2022. ¿Cuáles ganaron agua ese año?
# 3. Construye una tabla con una fila por año y una columna por presa con el llenado **máximo** del año. Pista: `groupby` con `df["fecha"].dt.year`, o `pivot_table`.
# 4. El catálogo tiene la hoja `Municipios`, con la región de Sonora a la que pertenece cada municipio. Úsala para agregar la región a cada presa. ¿Qué problema encuentras al unir por nombre de municipio?
