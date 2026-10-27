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
# # Regresión lineal: cuántas bicicletas se rentarán
#
# **Unidad 4 · Ejemplo 1**
#
# La regresión lineal es el modelo más viejo del curso y, probablemente, el más usado en el mundo. Parece sencillo (una suma ponderada de las variables), pero detrás hay mucha más matemática y mucho más poder de lo que aparenta. En esta libreta, con datos reales de un sistema de bicicletas compartidas, vamos a:
#
# 1. **derivar** la solución de mínimos cuadrados (las ecuaciones normales), verla como una **proyección** y como un estimador de **máxima verosimilitud**;
# 2. ver **cómo se calcula** en la práctica: por qué no se invierte $\mathbf{X}^\top\mathbf{X}$, qué es el número de condición y por qué el descenso por gradiente necesita variables en escalas comparables;
# 3. entender que el modelo es lineal **en los parámetros, no en los datos**: con buenas características (codificación *one-hot*, interacciones) el error de prueba baja casi a la mitad;
# 4. estudiar la **colinealidad**: cuando dos variables dicen casi lo mismo, los coeficientes se vuelven inestables aunque las predicciones no;
# 5. **diagnosticar** el modelo con sus residuos y descubrir qué supuesto falla, lo que nos llevará a la siguiente libreta.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado.

# %%
import os, sys

RAMA = "rediseno-2027"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Datos.** Capital Bikeshare es el sistema público de bicicletas compartidas de Washington, D. C.: cientos de estaciones repartidas por la ciudad donde uno toma una bicicleta con una tarjeta y la deja en cualquier otra estación. El sistema publica el registro de **cada viaje** (dónde y cuándo empezó y terminó). Hadi Fanaee-T y João Gama (Universidad de Porto) tomaron los viajes de 2011 y 2012, los **contaron por hora** y les agregaron el clima de esa hora (temperatura, sensación térmica, humedad, viento y una clasificación del estado del tiempo) y el calendario (día de la semana, feriados). El resultado son 17,379 horas, publicadas en el repositorio de la UCI (*Progress in Artificial Intelligence*, 2014).
#
# | columna | significado |
# |---|---|
# | `fecha`, `hora` | día y hora (0 a 23) |
# | `laboral` | 1 si no es fin de semana ni feriado |
# | `clima` | 1 despejado, 2 nublado o neblina, 3 lluvia o nieve ligera (4, intensa: solo 3 horas, la unimos a la 3) |
# | `temp`, `sensacion` | temperatura y sensación térmica (°C) |
# | `humedad`, `viento` | humedad relativa (%) y velocidad del viento (km/h) |
# | `casuales`, `registrados`, `total` | rentas en esa hora: usuarios ocasionales, suscriptores y la suma |
#
# **Problema.** Predecir cuántas bicicletas se rentarán en una hora (`total`) a partir de la hora, el calendario y el clima. Para el operador del sistema no es una curiosidad: las bicicletas se acumulan en las estaciones de destino (el centro en la mañana, los barrios en la tarde) y hay que **redistribuirlas** con camionetas. Saber cuánta demanda habrá, y cuándo, permite planear esos recorridos y el número de bicicletas en circulación.

# %%
b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
b["anio"] = b.fecha.dt.year - 2011          # 0 = 2011, 1 = 2012 (el sistema creció)
b["clima"] = b.clima.clip(upper=3)          # la categoría 4 tiene 3 horas: se une a la 3
b["hora_lab"] = b.hora.astype(str) + "_" + b.laboral.astype(str)
print(b.shape)
b.head()

# %% [markdown]
# ### Cómo partimos los datos
#
# Lo habitual sería separar al azar un 25 % de las horas para prueba. Aquí sería un error: dos horas consecutivas del mismo día se parecen muchísimo (mismo clima, misma gente), así que si la 8:00 de un martes queda en entrenamiento y la 9:00 en prueba, el modelo «ve» casi la respuesta y el error de prueba sale optimista. Hacemos lo mismo que hicieron Fanaee-T y Gama (y la competencia de Kaggle que usó estos datos): **entrenamos con los días 1 a 19 de cada mes y probamos con los días 20 en adelante**. Así, el conjunto de prueba son días completos que el modelo nunca vio, de todas las temporadas.
#
# No es perfecto: el día 19 y el día 20 siguen siendo vecinos, y una partición más honesta para predecir el futuro sería entrenar con el pasado y probar con lo que sigue. Los datos con estructura temporal se tratan con más cuidado en la Unidad 7.

# %%
entrena = b[b.fecha.dt.day <= 19]           # días 1–19 de cada mes: 10,886 horas
prueba  = b[b.fecha.dt.day >= 20]           # días 20 en adelante: 6,493 horas
num = ["temp", "humedad", "viento"]
print(f"entrenamiento: {len(entrena):,} horas · prueba: {len(prueba):,} horas")
entrena[["total", "temp", "sensacion", "humedad", "viento"]].describe().round(1)

# %% [markdown]
# Antes de modelar, miremos los datos. La gráfica que más dice es la renta media por hora del día, separando días laborales de fines de semana y feriados.

# %%
perfil = entrena.groupby(["laboral", "hora"]).total.mean().unstack(0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(perfil.index, perfil[1], "o-", color=AZUL, ms=4)
ax.plot(perfil.index, perfil[0], "o-", color=ROJO, ms=4)
ax.set(xticks=range(0, 24, 2), xlim=(-0.5, 23.5), xlabel="hora del día", ylabel="rentas por hora (media)",
       title="Renta media de bicicletas por hora del día (entrenamiento)")
ax.text(18.6, 460, "días laborales", color=AZUL, fontsize=12)
ax.text(13.3, 420, "fines de semana y feriados", color=ROJO, fontsize=12, ha="center")
ax.annotate("entrada al trabajo", (8, perfil.loc[8, 1]), xytext=(1.5, 480), color=AZUL, fontsize=11,
            arrowprops=dict(arrowstyle="->", color=AZUL))
ax.annotate("salida", (17.5, perfil.loc[17, 1]), xytext=(20, 520), color=AZUL, fontsize=11,
            arrowprops=dict(arrowstyle="->", color=AZUL))
plt.show()

# %% [markdown]
# Son dos comportamientos completamente distintos:
#
# - En **días laborales**, dos picos agudos: a las 8 h y a las 17–18 h. Es gente que va y vuelve del trabajo; entre semana la bicicleta es transporte.
# - En **fines de semana y feriados**, una sola joroba ancha alrededor del mediodía: es paseo.
#
# Guarda esta figura en la cabeza. Un buen modelo tiene que poder dibujar estas dos curvas, y veremos que una regresión lineal puede hacerlo, si le damos las características adecuadas. Pero empecemos por lo más simple.

# %% [markdown]
# ## 2. Mínimos cuadrados desde cero
#
# ### El modelo y el error
#
# Empezamos con una sola variable: ¿cuánto explica la **temperatura**? Es razonable que con buen clima se renten más bicicletas.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(entrena.temp, entrena.total, s=6, alpha=0.15, color=AZUL, edgecolors="none")
ax.set(xlabel="temperatura (°C)", ylabel="rentas en la hora", title="Rentas por hora contra temperatura (entrenamiento)")
plt.show()

# %% [markdown]
# Hay tendencia, pero también una dispersión enorme: a la misma temperatura hay horas con 5 rentas (la madrugada) y con 800 (la salida del trabajo). Ajustemos de todos modos una recta, para entender la maquinaria.
#
# Con la notación de *Learning From Data*: cada ejemplo es $\mathbf{x}_n = (1, x_{n1}, \dots, x_{nd})^\top \in \mathbb{R}^{d+1}$ (el 1 inicial acompaña al término constante $w_0$), y la hipótesis es $h(\mathbf{x}) = \mathbf{w}^\top\mathbf{x}$. Apilamos los $N$ ejemplos como renglones de la **matriz de diseño** $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ y las etiquetas en $\mathbf{y} \in \mathbb{R}^N$. El error dentro de la muestra es
#
# $$
# E_{\text{in}}(\mathbf{w}) = \frac{1}{N}\sum_{n=1}^{N}\big(\mathbf{w}^\top\mathbf{x}_n - y_n\big)^2 = \frac{1}{N}\,\lVert \mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2 .
# $$
#
# ### Las ecuaciones normales
#
# Desarrollando el cuadrado, $E_{\text{in}}(\mathbf{w}) = \frac{1}{N}\big(\mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{w}^\top\mathbf{X}^\top\mathbf{y} + \mathbf{y}^\top\mathbf{y}\big)$, una función cuadrática de $\mathbf{w}$. Su gradiente y su hessiana son
#
# $$
# \nabla E_{\text{in}}(\mathbf{w}) = \frac{2}{N}\big(\mathbf{X}^\top\mathbf{X}\mathbf{w} - \mathbf{X}^\top\mathbf{y}\big),
# \qquad
# \nabla^2 E_{\text{in}} = \frac{2}{N}\,\mathbf{X}^\top\mathbf{X}.
# $$
#
# La hessiana es semidefinida positiva ($\mathbf{v}^\top\mathbf{X}^\top\mathbf{X}\mathbf{v} = \lVert\mathbf{X}\mathbf{v}\rVert^2 \ge 0$), así que $E_{\text{in}}$ es convexa y todo punto donde el gradiente se anula es un mínimo global. Igualando a cero obtenemos las **ecuaciones normales**:
#
# $$
# \boxed{\;\mathbf{X}^\top\mathbf{X}\,\mathbf{w} = \mathbf{X}^\top\mathbf{y}\;}
# $$
#
# Si las columnas de $\mathbf{X}$ son linealmente independientes, $\mathbf{X}^\top\mathbf{X}$ es invertible (definida positiva, de hecho) y la solución es única: $\mathbf{w}_{\text{lin}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$. Resolvámoslas de tres maneras.

# %%
X = np.column_stack([np.ones(len(entrena)), entrena.temp])
y = entrena.total.to_numpy(dtype=float)
N = len(y)

w_normales = np.linalg.solve(X.T @ X, X.T @ y)                 # resolver el sistema de 2 × 2
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]                 # mínimos cuadrados directamente sobre X
reg = LinearRegression().fit(entrena[["temp"]], y)             # scikit-learn
w_sklearn = np.r_[reg.intercept_, reg.coef_]

print(pd.DataFrame([w_normales, w_lstsq, w_sklearn], columns=["w0", "w1 (temp)"],
                   index=["ecuaciones normales", "np.linalg.lstsq", "scikit-learn"]).round(4))
E_in = np.mean((X @ w_lstsq - y) ** 2)
print(f"\nE_in = {E_in:,.0f}   (RECM de entrenamiento = {np.sqrt(E_in):.1f} rentas)")

# %% [markdown]
# Las tres coinciden: **cada grado centígrado más se asocia con unas 9.2 rentas más por hora**, y el término constante (6 rentas a 0 °C) es una extrapolación sin mucho sentido porque casi no hay horas tan frías. El error típico, unas 166 rentas por hora, es enorme comparado con la media (192): la temperatura sola explica poco. Lo arreglaremos en la sección 4.

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(entrena.temp, entrena.total, s=6, alpha=0.15, color=AZUL, edgecolors="none")
tt = np.linspace(entrena.temp.min(), entrena.temp.max(), 2)
ax.plot(tt, w_lstsq[0] + w_lstsq[1] * tt, color=ROJO, lw=3)
ax.text(41, 470, f"$\\hat y = {w_lstsq[0]:.1f} + {w_lstsq[1]:.2f}\\,$temp", color=ROJO,
        ha="right", va="bottom", fontsize=13, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
ax.set(xlabel="temperatura (°C)", ylabel="rentas en la hora", title="La recta de mínimos cuadrados")
plt.show()

# %% [markdown]
# ### La geometría: una proyección
#
# Las predicciones $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ son combinaciones lineales de las columnas de $\mathbf{X}$: viven en el **espacio columna** de $\mathbf{X}$, un subespacio de dimensión $d+1$ dentro de $\mathbb{R}^N$. Minimizar $\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert$ es buscar el punto de ese subespacio más cercano a $\mathbf{y}$, y ese punto es la **proyección ortogonal**. Las ecuaciones normales dicen exactamente eso, reescritas como
#
# $$
# \mathbf{X}^\top\big(\mathbf{y} - \hat{\mathbf{y}}\big) = \mathbf{0}:
# $$
#
# el vector de residuos es perpendicular a cada columna de $\mathbf{X}$. En particular es perpendicular a la columna de unos, así que **los residuos suman cero**; y es perpendicular a la columna de temperaturas, así que no queda ninguna tendencia lineal con la temperatura sin explicar.
#
# Sustituyendo $\mathbf{w}_{\text{lin}}$, $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ con la **matriz sombrero** (le «pone el sombrero» a $\mathbf{y}$)
#
# $$
# \mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top \in \mathbb{R}^{N\times N}.
# $$
#
# Por ser una proyección ortogonal, $\mathbf{H}$ es **simétrica** ($\mathbf{H}^\top = \mathbf{H}$), **idempotente** ($\mathbf{H}^2 = \mathbf{H}$: proyectar dos veces es proyectar una) y su traza es la dimensión del subespacio: $\operatorname{tr}\mathbf{H} = \operatorname{tr}\big((\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}\big) = \operatorname{tr}\mathbf{I}_{d+1} = d + 1$ (usando $\operatorname{tr}(AB) = \operatorname{tr}(BA)$). Verifiquémoslo. Con $N = 10{,}886$, $\mathbf{H}$ tendría casi 120 millones de entradas, así que la formamos con una submuestra de 500 horas.

# %%
residuos = y - X @ w_lstsq
print("X^T (y - ŷ) =", np.round(X.T @ residuos, 6), f"   (compárese con ‖X^T y‖ = {np.linalg.norm(X.T @ y):.2e})")

rng = np.random.default_rng(SEMILLA)
idx = rng.choice(N, size=500, replace=False)
Xs = X[idx]
H = Xs @ np.linalg.solve(Xs.T @ Xs, Xs.T)
print(f"\nH es de {H.shape[0]} × {H.shape[1]}")
print(f"simétrica:    max |H - H^T| = {np.abs(H - H.T).max():.1e}")
print(f"idempotente:  max |H² - H|  = {np.abs(H @ H - H).max():.1e}")
print(f"traza:        tr H = {np.trace(H):.6f}   (d + 1 = {Xs.shape[1]})")
print(f"valores propios: {np.sum(np.isclose(np.linalg.eigvalsh(H), 1))} iguales a 1, el resto ≈ 0")

# %% [markdown]
# Todo cuadra hasta el error de redondeo. (Las entradas de $\mathbf{X}^\top(\mathbf{y}-\hat{\mathbf{y}})$ no son exactamente cero, pero son unas $10^{-12}$ veces más chicas que $\mathbf{X}^\top\mathbf{y}$.) Los valores propios de una proyección son solo 0 y 1: $d+1$ direcciones que se conservan y $N - d - 1$ que se aniquilan. Este hecho ya lo usamos en la Unidad 3 para demostrar que $\mathbb{E}[E_{\text{in}}] = \sigma^2\big(1 - \frac{d+1}{N}\big)$.
#
# ### La lectura probabilística: máxima verosimilitud
#
# ¿Por qué el error **cuadrático** y no el absoluto, o cualquier otro? Una respuesta es la comodidad (es derivable y da un sistema lineal). Otra, más profunda, es probabilística. Supongamos que los datos se generan como
#
# $$
# y_n = \mathbf{w}^\top\mathbf{x}_n + \varepsilon_n, \qquad \varepsilon_n \sim \mathcal{N}(0, \sigma^2) \text{ independientes.}
# $$
#
# Entonces $y_n \mid \mathbf{x}_n \sim \mathcal{N}(\mathbf{w}^\top\mathbf{x}_n, \sigma^2)$ y la log-verosimilitud de los datos es
#
# $$
# \ell(\mathbf{w}, \sigma^2) = \sum_{n=1}^N \log\Big[\frac{1}{\sqrt{2\pi\sigma^2}}\,e^{-(y_n - \mathbf{w}^\top\mathbf{x}_n)^2/2\sigma^2}\Big]
# = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^N\big(y_n - \mathbf{w}^\top\mathbf{x}_n\big)^2 .
# $$
#
# Para cualquier $\sigma^2$ fija, el único término que depende de $\mathbf{w}$ es la suma de cuadrados, con signo negativo: **maximizar la verosimilitud en $\mathbf{w}$ es exactamente minimizar $E_{\text{in}}$**. Derivando ahora respecto a $\sigma^2$ se obtiene $\hat\sigma^2 = E_{\text{in}}(\mathbf{w}_{\text{lin}})$: la varianza estimada del ruido es el error cuadrático medio.
#
# Esta lectura es útil por dos razones. Primero, nos da intervalos y errores estándar (sección 5). Segundo, hace explícitos los **supuestos**: ruido gaussiano, con la misma varianza en todas partes, alrededor de una media lineal. Si cambiamos el supuesto sobre el ruido, cambia el error que hay que minimizar; esa es la idea de los modelos lineales generalizados de la siguiente libreta. Ya podemos sospechar que aquí algo no encaja: las rentas son enteros no negativos, y la nube de arriba es mucho más ancha a 30 °C que a 0 °C.

# %% [markdown]
# ## 3. Cómo se calcula
#
# ### El número de condición
#
# La fórmula $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ es perfecta en el pizarrón y mala en la computadora. La razón es el **número de condición**. Para una matriz $\mathbf{X}$ con valores singulares $s_1 \ge \dots \ge s_{d+1} > 0$,
#
# $$
# \kappa(\mathbf{X}) = \frac{s_1}{s_{d+1}} ,
# $$
#
# y mide cuánto puede amplificarse un error relativo en los datos al resolver el problema: con aritmética de 16 cifras significativas (doble precisión) y $\kappa \approx 10^k$, se pueden perder hasta unas $k$ cifras de la solución. Si $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^\top$, entonces $\mathbf{X}^\top\mathbf{X} = \mathbf{V}\mathbf{S}^2\mathbf{V}^\top$: sus valores singulares son los **cuadrados** de los de $\mathbf{X}$, así que
#
# $$
# \kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2 .
# $$
#
# Formar $\mathbf{X}^\top\mathbf{X}$ **duplica el número de cifras que se pueden perder**. Veámoslo con la temperatura y la sensación térmica, que en estos datos tienen correlación 0.988 (dicen casi lo mismo; volveremos a esto en la sección 5).

# %%
print(f"corr(temp, sensacion) = {b.temp.corr(b.sensacion):.3f}\n")


def matriz_diseno(df, columnas, estandarizar=False):
    Z = df[columnas].to_numpy(dtype=float)
    if estandarizar:
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    return np.column_stack([np.ones(len(df)), Z])


filas = []
for columnas in [["temp"], ["temp", "sensacion"]]:
    for est in [False, True]:
        Xc = matriz_diseno(entrena, columnas, est)
        filas.append({"columnas": " + ".join(columnas), "estandarizadas": "sí" if est else "no",
                      "κ(X)": np.linalg.cond(Xc), "κ(XᵀX)": np.linalg.cond(Xc.T @ Xc)})
pd.DataFrame(filas).round(1)

# %% [markdown]
# Dos lecciones:
#
# - La relación $\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2$ se cumple en cada renglón.
# - Hay dos fuentes de mal condicionamiento. Una es la **escala**: la columna de unos y la de temperaturas (de 0 a 40, con media 20) apuntan en direcciones parecidas; estandarizar la columna la vuelve ortogonal a la de unos, y $\kappa$ baja a 1. La otra es la **colinealidad**: temp y sensación siguen casi alineadas después de estandarizar, y eso ninguna escala lo arregla ($\kappa(\mathbf{X}) \approx 11$).
#
# Con $\kappa(\mathbf{X}^\top\mathbf{X}) \approx 10^4$ y 16 cifras, la doble precisión sobra. Para ver el efecto, repitamos el cálculo en **precisión simple** (`float32`, unas 7 cifras), como en una GPU, y comparemos con la respuesta de doble precisión.

# %%
for columnas in [["temp"], ["temp", "sensacion"]]:
    Xc = matriz_diseno(entrena, columnas)
    w64 = np.linalg.lstsq(Xc, y, rcond=None)[0]
    X32, y32 = Xc.astype(np.float32), y.astype(np.float32)
    w_ne = np.linalg.solve(X32.T @ X32, X32.T @ y32)
    w_qr = np.linalg.lstsq(X32, y32, rcond=None)[0]
    err = lambda w: np.linalg.norm(w - w64) / np.linalg.norm(w64)
    print(f"{' + '.join(columnas):>16}:  error relativo en float32 — ecuaciones normales {err(w_ne):.1e} · "
          f"lstsq sobre X {err(w_qr):.1e}")

# %% [markdown]
# Resolver las ecuaciones normales pierde **entre 50 y 2,600 veces más precisión** que trabajar directamente sobre $\mathbf{X}$: es el orden del factor $\kappa(\mathbf{X})$ extra que se paga al elevarlo al cuadrado (la cota es para el peor caso, por eso no se alcanza exactamente). Por eso las bibliotecas nunca forman $\mathbf{X}^\top\mathbf{X}$. Usan una de dos factorizaciones:
#
# - **QR**: $\mathbf{X} = \mathbf{Q}\mathbf{R}$, con $\mathbf{Q}$ de columnas ortonormales y $\mathbf{R}$ triangular superior. Sustituyendo en las ecuaciones normales, $\mathbf{R}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{R}\mathbf{w} = \mathbf{R}^\top\mathbf{Q}^\top\mathbf{y}$, es decir, $\mathbf{R}\mathbf{w} = \mathbf{Q}^\top\mathbf{y}$, que se resuelve por sustitución hacia atrás. Como $\kappa(\mathbf{R}) = \kappa(\mathbf{X})$, no se eleva nada al cuadrado.
# - **SVD**: $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^\top$ y $\mathbf{w} = \mathbf{V}\mathbf{S}^{-1}\mathbf{U}^\top\mathbf{y}$. Es más cara, pero también funciona si $\mathbf{X}$ no tiene rango completo: se descartan los valores singulares casi nulos y se obtiene la solución de **norma mínima** (la pseudoinversa). Es lo que usan `np.linalg.lstsq` y, a través de SciPy, `LinearRegression` de scikit-learn.

# %%
Q, R = np.linalg.qr(X)
w_qr = np.linalg.solve(R, Q.T @ y)   # R es triangular: en serio se usaría sustitución hacia atrás
print("QR:", np.round(w_qr, 4), "  κ(R) =", round(np.linalg.cond(R), 1), "  κ(X) =", round(np.linalg.cond(X), 1))

# %% [markdown]
# ### Descenso por gradiente
#
# Con millones de datos o de variables, ni siquiera QR es viable, y se usa **descenso por gradiente**: partir de algún $\mathbf{w}(0)$ y dar pasos contra el gradiente,
#
# $$
# \mathbf{w}(t+1) = \mathbf{w}(t) - \eta\,\nabla E_{\text{in}}\big(\mathbf{w}(t)\big),
# $$
#
# con **tasa de aprendizaje** $\eta > 0$. Para regresión lineal podemos analizarlo exactamente. Sea $\mathbf{A} = \nabla^2 E_{\text{in}} = \frac{2}{N}\mathbf{X}^\top\mathbf{X}$, con valores propios $\lambda_{\min} \le \dots \le \lambda_{\max}$. Como el gradiente es $\mathbf{A}(\mathbf{w} - \mathbf{w}_{\text{lin}})$, el error $\mathbf{e}(t) = \mathbf{w}(t) - \mathbf{w}_{\text{lin}}$ cumple
#
# $$
# \mathbf{e}(t+1) = (\mathbf{I} - \eta\mathbf{A})\,\mathbf{e}(t) \quad\Longrightarrow\quad \mathbf{e}(t) = (\mathbf{I} - \eta\mathbf{A})^t\,\mathbf{e}(0).
# $$
#
# En la base de vectores propios de $\mathbf{A}$, cada componente se multiplica en cada paso por $1 - \eta\lambda_i$. Por lo tanto:
#
# - **Converge si y solo si** $|1 - \eta\lambda_i| < 1$ para toda $i$, es decir, $\boxed{\eta < 2/\lambda_{\max}}$. Con un paso más grande, la dirección más empinada oscila con amplitud creciente y el algoritmo diverge.
# - La componente **más lenta** es la de $\lambda_{\min}$, que se reduce por el factor $1 - \eta\lambda_{\min} > 1 - 2\lambda_{\min}/\lambda_{\max} = 1 - 2/\kappa(\mathbf{A})$. Incluso con la mejor $\eta$ posible, $\eta^\ast = 2/(\lambda_{\min} + \lambda_{\max})$, el factor de contracción es $\frac{\kappa - 1}{\kappa + 1}$. Con $\kappa$ grande, eso está muy cerca de 1 y hacen falta del orden de $\kappa$ iteraciones por cada cifra decimal ganada.
#
# El número de condición de la hessiana manda: si $\kappa(\mathbf{A})$ es grande, las curvas de nivel de $E_{\text{in}}$ son elipses muy alargadas, el gradiente apunta casi perpendicular al eje largo, y el algoritmo zigzaguea a través del valle en lugar de avanzar por él. Comparemos el modelo `total ~ temp` con la temperatura **sin centrar** (en °C) y **estandarizada**. Los dos parten de la misma hipótesis, $h = 0$ (es decir, $\mathbf{w}(0) = \mathbf{0}$), y declaramos convergencia cuando $E_{\text{in}}$ está a menos de una millonésima relativa del mínimo.

# %%
def descenso(Xd, y, eta, max_iter=100_000, tol=1e-6, guardar=False):
    """Descenso por gradiente para E_in = ‖Xw − y‖²/N desde w = 0. Regresa (iteraciones, trayectoria)."""
    A = Xd.T @ Xd / len(y)
    c = Xd.T @ y / len(y)
    yy = y @ y / len(y)
    w_opt = np.linalg.solve(A, c)
    E_opt = w_opt @ A @ w_opt - 2 * c @ w_opt + yy
    w = np.zeros(Xd.shape[1])
    camino = [w.copy()]
    for t in range(max_iter):
        if (w @ A @ w - 2 * c @ w + yy - E_opt) / E_opt < tol:
            return t, np.array(camino)
        w = w - eta * 2 * (A @ w - c)
        if guardar:
            camino.append(w.copy())
    return None, np.array(camino)


casos = {"sin centrar": matriz_diseno(entrena, ["temp"]), "estandarizada": matriz_diseno(entrena, ["temp"], True)}
FRACCIONES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]   # η como fracción del límite 2/λ_max
filas, mejor = [], {}
for nombre, Xd in casos.items():
    lam = np.linalg.eigvalsh(2 * Xd.T @ Xd / N)
    print(f"temp {nombre}: valores propios de la hessiana = {lam.round(3)}   κ = {lam[-1] / lam[0]:,.1f}   "
          f"2/λ_max = {2 / lam[-1]:.5f}")
    iters = {f: descenso(Xd, y, f * 2 / lam[-1])[0] for f in FRACCIONES}
    f_mejor = min(iters, key=iters.get)
    mejor[nombre] = (f_mejor * 2 / lam[-1], iters[f_mejor])
    filas.append({"temp": nombre, **{f"{f}·2/λmax": iters[f] for f in FRACCIONES}})
print("\nIteraciones hasta converger, según la tasa de aprendizaje:")
pd.DataFrame(filas).set_index("temp")

# %%
for nombre, (eta, it) in mejor.items():
    print(f"temp {nombre:>13}: mejor η de la tabla = {eta:.5f}  →  {it:,} iteraciones")

# %% [markdown]
# La diferencia es abismal: con la temperatura en °C, la hessiana tiene $\kappa \approx 3{,}650$ y el mejor descenso de la tabla necesita 4,585 iteraciones; estandarizada, $\kappa = 1$ y basta **una sola** (con $\eta = 1/\lambda$, exactamente la mitad del límite).
#
# ¿Por qué una? Con la columna estandarizada, $\frac{1}{N}\mathbf{X}^\top\mathbf{X} = \mathbf{I}$ (la columna tiene media 0, así que es ortogonal a la de unos, y varianza 1), la hessiana es $2\mathbf{I}$, las curvas de nivel son **círculos** y el gradiente apunta directo al centro. Es un caso ideal, que se da porque solo hay una variable; con varias variables correlacionadas $\kappa$ ya no es 1 (recuerda el 131 de temp y sensación), pero estandarizar sigue eliminando la parte del problema que se debe a la escala.
#
# Hay que notar también que en el caso sin centrar el límite $2/\lambda_{\max}$ es diminuto (0.002) y, aun así, la mejor tasa es la que está pegada a él: con $\kappa$ grande, el paso lo limita la dirección empinada y el avance lo frena la plana. Veámoslo en las curvas de nivel, con $\eta = 0.9 \cdot 2/\lambda_{\max}$ en ambos casos.

# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 5.8))
for ax, (nombre, Xd), color in zip(axs, casos.items(), [ROJO, AZUL]):
    A, c, yy = Xd.T @ Xd / N, Xd.T @ y / N, y @ y / N
    lam_max = np.linalg.eigvalsh(2 * A)[-1]
    it, camino = descenso(Xd, y, 0.9 * 2 / lam_max, guardar=True)
    w_opt = np.linalg.solve(A, c)
    E_opt = w_opt @ A @ w_opt - 2 * c @ w_opt + yy
    if nombre == "sin centrar":
        lim0, lim1 = (-12, 24), (-1, 19)
    else:
        lim0, lim1 = (-40, 380), (-60, 200)
    W0, W1 = np.meshgrid(np.linspace(*lim0, 400), np.linspace(*lim1, 400))
    E = A[0, 0] * W0**2 + 2 * A[0, 1] * W0 * W1 + A[1, 1] * W1**2 - 2 * (c[0] * W0 + c[1] * W1) + yy
    ax.contour(W0, W1, E / E_opt - 1, levels=[0.0005, 0.005, 0.05, 0.25, 1, 3], colors=GRIS, linewidths=0.9)
    ax.plot(camino[:, 0], camino[:, 1], "-", color=color, lw=1.2, alpha=0.9)
    ax.plot(camino[:40, 0], camino[:40, 1], "o", color=color, ms=3)
    ax.plot(*w_opt, "*", color=TINTA, ms=17, mec="white", zorder=5)
    ax.plot(0, 0, "s", color=TINTA, ms=7)
    ax.set(xlim=lim0, ylim=lim1, xlabel="$w_0$", ylabel="$w_1$",
           title=f"temp {nombre}: κ = {np.linalg.cond(A):,.0f}, {it:,} iteraciones")
    if nombre == "sin centrar":
        for t, pos in [(100, (-6, 7.0)), (1000, (4.5, 6.4)), (2500, (10, 7.6))]:
            ax.plot(*camino[t], "o", color=TINTA, ms=6)
            ax.annotate(f"t = {t:,}", camino[t], xytext=pos, fontsize=10.5, ha="center",
                        arrowprops=dict(arrowstyle="-", color=TENUE, lw=0.8))
axs[0].text(-0.6, -0.6, "inicio", fontsize=11, ha="right")
axs[0].text(7.2, 10.3, "$\\mathbf{w}_{\\mathrm{lin}}$", fontsize=13)
axs[1].text(-12, -40, "inicio", fontsize=11)
axs[1].text(205, 45, "$\\mathbf{w}_{\\mathrm{lin}}$", fontsize=13)
fig.suptitle("Descenso por gradiente con η = 0.9·2/λ$_{\\max}$ desde w = 0 (curvas: E$_{\\mathrm{in}}$ a 0.05 %, 0.5 %, 5 %, 25 %, 100 % y 300 % del mínimo)",
             fontsize=12, color=TENUE, y=1.0)
plt.tight_layout()
plt.show()

# %% [markdown]
# - **Sin centrar** (izquierda): las curvas de nivel son elipses tan alargadas que parecen rectas paralelas. Los primeros pasos rebotan de una pared del valle a la otra en dirección casi vertical (llegan hasta $w_1 \approx 17$); esa oscilación se amortigua en unas decenas de iteraciones, porque en la dirección empinada el factor es $|1 - \eta\lambda_{\max}| = 0.8$. Después el algoritmo se arrastra por el fondo del valle hacia $\mathbf{w}_{\text{lin}}$ con factor $1 - \eta\lambda_{\min} \approx 0.9995$ por paso: recorrer esas seis unidades de $w_0$ consume casi todas las iteraciones (los puntos negros marcan dónde va en $t = 100$, $1{,}000$ y $2{,}500$). Un cambio de 10 unidades en $w_0$ casi no cambia $E_{\text{in}}$, pero uno de 0.5 en $w_1$ sí: la columna de unos y la de temperaturas tienen escalas muy distintas.
# - **Estandarizada** (derecha): círculos. El algoritmo oscila sobre la recta que une el inicio con el óptimo, porque $\eta$ es mayor que la ideal, pero cada oscilación se reduce por un factor $|1 - 0.9\cdot 2| = 0.8$ y llega en 32 iteraciones.
#
# **Moraleja práctica:** antes de usar descenso por gradiente (y en las redes neuronales de la Unidad 6 siempre se usa), **estandariza las variables**. Por eso nuestros modelos de scikit-learn llevan `StandardScaler`, aunque para `LinearRegression` la solución exacta no lo necesita.

# %% [markdown]
# ## 4. Lineal en los parámetros, no en los datos
#
# La palabra «lineal» de regresión lineal se refiere a $\mathbf{w}$, no a $\mathbf{x}$. Todo lo que hicimos (ecuaciones normales, proyección, máxima verosimilitud) funciona igual si en lugar de $\mathbf{x}$ usamos cualquier transformación fija $\boldsymbol{\Phi}(\mathbf{x})$: indicadoras de categorías, potencias, productos de variables. En LFD es la transformación no lineal $\mathbf{z} = \boldsymbol{\Phi}(\mathbf{x})$. El arte está en elegir las características. Construimos una escalera de cuatro modelos, todos con `LinearRegression`:
#
# | modelo | categóricas (*one-hot*) | numéricas (estandarizadas) |
# |---|---|---|
# | M1 | — | `temp` |
# | M2 | — | `temp`, `hora` (la hora como número: **un error a propósito**) |
# | M3 | `hora`, `clima`, `anio`, `laboral` | `temp`, `humedad`, `viento` |
# | M4 | `hora_lab` (hora × laboral), `clima`, `anio` | `temp`, `humedad`, `viento` |
#
# La codificación *one-hot* con `drop="first"` crea una columna indicadora por categoría, salvo la primera, que queda absorbida en el término constante (si las pusiéramos todas, sumarían la columna de unos y $\mathbf{X}$ no tendría rango completo). La variable `anio` vale 0 en 2011 y 1 en 2012: el sistema creció mucho de un año al otro y queremos que el modelo lo sepa.

# %%
def modelo(categoricas, numericas):
    transformador = make_column_transformer((OneHotEncoder(drop="first"), categoricas), (StandardScaler(), numericas))
    return make_pipeline(transformador, LinearRegression())


ESCALERA = {
    "M1": ([], ["temp"]),
    "M2": ([], ["temp", "hora"]),
    "M3": (["hora", "clima", "anio", "laboral"], num),
    "M4": (["hora_lab", "clima", "anio"], num),
}
recm = lambda real, pred: np.sqrt(np.mean((real - pred) ** 2))

modelos, filas = {}, []
for nombre, (cat, nums) in ESCALERA.items():
    m = modelo(cat, nums).fit(entrena, entrena.total)
    modelos[nombre] = m
    filas.append({"modelo": nombre, "parámetros": m[0].transform(entrena.head()).shape[1] + 1,
                  "RECM entrenamiento": recm(entrena.total, m.predict(entrena)),
                  "RECM prueba": recm(prueba.total, m.predict(prueba))})
tabla_escalera = pd.DataFrame(filas).set_index("modelo")
tabla_escalera.round(1)

# %% [markdown]
# El error de prueba pasa de 165 a **86.4** rentas por hora, casi la mitad, sin salir de la regresión lineal. Y las RECM de entrenamiento y de prueba son parecidas en los cuatro modelos: con 10,886 horas y a lo más 56 parámetros, la varianza no es problema (recuerda la brecha $\approx 2\sigma^2(d+1)/N$ de la Unidad 3); lo que mejora es el **sesgo**, porque cada modelo puede representar más formas.
#
# Para ver qué ganó cada escalón, comparemos el perfil por hora que predicen M2, M3 y M4 con el real, en los días laborales del conjunto de prueba.

# %%
lab = prueba[prueba.laboral == 1]
perfiles = pd.DataFrame({"real": lab.total, **{k: modelos[k].predict(lab) for k in ["M2", "M3", "M4"]}}).groupby(lab.hora).mean()

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(perfiles.index, perfiles["real"], "o-", color=TINTA, lw=3, ms=5)
for k, color in zip(["M2", "M3", "M4"], [GRIS, ROJO, AZUL]):
    ax.plot(perfiles.index, perfiles[k], "-", color=color, lw=2.2)
ax.set(xticks=range(0, 24, 2), xlim=(-0.5, 27), xlabel="hora del día", ylabel="rentas por hora (media)",
       title="Días laborales de prueba: perfil real y perfil predicho por cada modelo")
rdp.estilo.etiquetar(ax, 23.4, {"real": perfiles["real"].iloc[-1], "M2: hora como número": perfiles["M2"].iloc[-1],
                                "M3: hora one-hot": perfiles["M3"].iloc[-1],
                                "M4: hora × laboral": perfiles["M4"].iloc[-1]}, separacion=34)
plt.show()

# %% [markdown]
# - **M2** trata la hora como un número: el modelo es $w_0 + w_1\,\text{temp} + w_2\,\text{hora}$, y lo único que puede decir de la hora es «a más tarde, más (o menos) rentas», con la misma pendiente todo el día. Una **recta**, que no puede tener picos. Además pone la hora 23 lejísimos de la 0, cuando en realidad son vecinas. La hora es una variable **categórica** que casualmente se escribe con números.
# - **M3** le da a cada hora su propio coeficiente (23 indicadoras): ahora sí aparecen los picos. Pero el perfil tiene que ser **el mismo** para todos los días, desplazado hacia arriba o abajo por `laboral`. El modelo es aditivo: el efecto de la hora no depende del tipo de día. El resultado es un compromiso entre las dos curvas de la sección 1: los picos de los laborales se quedan cortos y el mediodía se sobreestima (en los fines de semana pasa al revés).
# - **M4** usa la **interacción** hora × laboral: una indicadora por cada combinación (las 8 de un día laboral, las 8 de un sábado, …), 47 en total. Ahora cada tipo de día tiene su propio perfil y M4 sigue de cerca la curva real.
#
# Una interacción entre dos variables categóricas es simplemente una categórica nueva con todas las combinaciones. Entre una categórica y una numérica (por ejemplo, que el efecto de la temperatura dependa de la hora) sería el producto de la indicadora por la variable. En todos los casos el modelo sigue siendo lineal en $\mathbf{w}$, y todo lo de las secciones 2 y 3 aplica sin cambios.

# %% [markdown]
# ## 5. Colinealidad
#
# ### Errores estándar
#
# Con el modelo probabilístico de la sección 2 y $\mathbf{X}$ fija, $\mathbf{w}_{\text{lin}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ es una transformación lineal del vector aleatorio $\mathbf{y}$, cuya covarianza es $\sigma^2\mathbf{I}$. Como $\operatorname{Cov}(\mathbf{M}\mathbf{y}) = \mathbf{M}\operatorname{Cov}(\mathbf{y})\mathbf{M}^\top$,
#
# $$
# \operatorname{Cov}(\mathbf{w}_{\text{lin}}) = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\,\sigma^2\mathbf{I}\,\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1} = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}.
# $$
#
# El **error estándar** del coeficiente $j$ es la raíz del elemento diagonal correspondiente, con $\sigma^2$ estimada por $\hat\sigma^2 = \frac{1}{N - d - 1}\lVert\mathbf{y} - \hat{\mathbf{y}}\rVert^2$ (el divisor $N - d - 1$ la hace insesgada, por el resultado de la Unidad 3).
#
# ### El factor de inflación de varianza
#
# ¿De qué depende ese elemento diagonal? Se puede demostrar (es el ejercicio 4) que, para una variable $x_j$ que no sea la constante,
#
# $$
# \operatorname{Var}(w_j) = \frac{\sigma^2}{N\,s_j^2}\cdot\underbrace{\frac{1}{1 - R_j^2}}_{\text{VIF}_j},
# $$
#
# donde $s_j^2$ es la varianza de $x_j$ y $R_j^2$ es el coeficiente de determinación de la regresión de $x_j$ **sobre las demás variables**. El primer factor es la varianza que tendría $w_j$ si $x_j$ fuera la única variable; el segundo, el **factor de inflación de varianza** (VIF), dice cuántas veces la multiplica el hecho de que $x_j$ se pueda predecir con las otras. Si $x_j$ no tiene nada que ver con las demás, $R_j^2 = 0$ y $\text{VIF} = 1$; si es casi una combinación de ellas, $R_j^2 \to 1$ y la varianza explota.
#
# La intuición: el coeficiente $w_j$ mide el efecto de mover $x_j$ **manteniendo fijas las demás**. Si temp y sensación casi siempre se mueven juntas, los datos tienen muy poca información sobre qué pasa cuando una se mueve y la otra no.
#
# Comparemos el modelo M1 (`total ~ temp`) con su extensión `total ~ temp + sensacion`, en unidades originales para que los coeficientes se lean en rentas por °C.

# %%
def ajuste_con_errores(df, columnas):
    Xd = matriz_diseno(df, columnas)
    yd = df.total.to_numpy(dtype=float)
    w = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    sigma2 = np.sum((yd - Xd @ w) ** 2) / (len(yd) - Xd.shape[1])
    ee = np.sqrt(sigma2 * np.diag(np.linalg.inv(Xd.T @ Xd)))
    return pd.DataFrame({"coeficiente": w, "error estándar": ee}, index=["constante"] + columnas), np.sqrt(sigma2)


t1, s1 = ajuste_con_errores(entrena, ["temp"])
t2, s2 = ajuste_con_errores(entrena, ["temp", "sensacion"])
R2_temp = entrena.temp.corr(entrena.sensacion) ** 2      # con un solo regresor, R² = corr²
vif = 1 / (1 - R2_temp)
print("total ~ temp");               print(t1.round(3)); print(f"σ̂ = {s1:.1f}\n")
print("total ~ temp + sensacion");   print(t2.round(3)); print(f"σ̂ = {s2:.1f}\n")
print(f"R² de temp contra sensación = {R2_temp:.4f}   →   VIF = {vif:.1f},  √VIF = {np.sqrt(vif):.2f}")
print(f"cociente de errores estándar de w_temp: {t2.loc['temp', 'error estándar'] / t1.loc['temp', 'error estándar']:.2f}")

# %% [markdown]
# Al agregar la sensación térmica, el error estándar del coeficiente de la temperatura se multiplica por 5.77, exactamente $\sqrt{\text{VIF}}$ (porque $\hat\sigma$ no cambió). La regla empírica habitual es preocuparse cuando el VIF pasa de 5 o 10; aquí vale 33. Y el modelo no mejoró nada: $\hat\sigma$ sigue en 166.5, y el coeficiente de la sensación ni siquiera es claramente distinto de cero (1.08 ± 1.09). La sensación térmica casi no aporta información nueva, pero sí mucha incertidumbre al **reparto** entre las dos variables.
#
# (Un matiz honesto: estas fórmulas suponen que las horas son independientes, y no lo son; horas vecinas tienen errores parecidos. Los errores estándar verdaderos son más grandes. Lo que nos interesa aquí es la comparación entre los dos modelos, que no depende de ese detalle.)
#
# ### Verlo con *bootstrap*
#
# Las fórmulas dependen de supuestos. Una verificación que no los necesita es el *bootstrap*: remuestreamos con reemplazo las 10,886 horas de entrenamiento, reajustamos el modelo y repetimos 200 veces. La dispersión de los coeficientes entre réplicas aproxima su variabilidad de muestreo.

# %%
rng = np.random.default_rng(SEMILLA)
X1, X2 = matriz_diseno(entrena, ["temp"]), matriz_diseno(entrena, ["temp", "sensacion"])
X1p, X2p = matriz_diseno(prueba, ["temp"]), matriz_diseno(prueba, ["temp", "sensacion"])
W1, W2, P1, P2 = [], [], [], []
for _ in range(200):
    i = rng.integers(0, N, N)
    w1 = np.linalg.lstsq(X1[i], y[i], rcond=None)[0]
    w2 = np.linalg.lstsq(X2[i], y[i], rcond=None)[0]
    W1.append(w1); W2.append(w2); P1.append(X1p @ w1); P2.append(X2p @ w2)
W1, W2, P1, P2 = map(np.array, (W1, W2, P1, P2))

print("Desviación estándar entre réplicas bootstrap:")
print(f"  w_temp, modelo solo con temp:     {W1[:, 1].std():.3f}")
print(f"  w_temp, modelo con temp + sens.:  {W2[:, 1].std():.3f}")
print(f"  w_sensacion:                       {W2[:, 2].std():.3f}")
print(f"  w_temp + w_sensacion:              {(W2[:, 1] + W2[:, 2]).std():.3f}")
print(f"  correlación entre w_temp y w_sensacion: {np.corrcoef(W2[:, 1], W2[:, 2])[0, 1]:.3f}")
print("\nDesviación estándar de las predicciones en prueba (promedio sobre las horas):")
print(f"  solo temp: {P1.std(axis=0).mean():.2f} rentas    temp + sensación: {P2.std(axis=0).mean():.2f} rentas")

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
ax.scatter(W2[:, 1], W2[:, 2], s=22, color=AZUL, alpha=0.7)
ax.axline((t2.loc["temp", "coeficiente"], t2.loc["sensacion", "coeficiente"]), slope=-1, color=GRIS, ls="--", lw=1.2)
ax.set(xlabel="$w_{\\mathrm{temp}}$ (rentas por °C)", ylabel="$w_{\\mathrm{sensación}}$ (rentas por °C)",
       title="200 réplicas bootstrap de total ~ temp + sensación")
ax.text(0.97, 0.95, "recta $w_{\\mathrm{temp}} + w_{\\mathrm{sens}}$ = constante", transform=ax.transAxes, ha="right",
        va="top", color=GRIS, fontsize=11)
ax = axs[1]
partes = [W1[:, 1], W2[:, 1], W2[:, 2], W2[:, 1] + W2[:, 2]]
nombres = ["$w_{\\mathrm{temp}}$\n(solo temp)", "$w_{\\mathrm{temp}}$\n(con sens.)", "$w_{\\mathrm{sens}}$", "suma"]
for k, (v, color) in enumerate(zip(partes, [GRIS, AZUL, ROJO, DORADO])):
    ax.scatter(np.full(len(v), k) + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=color, alpha=0.5)
ax.set(xticks=range(4), xticklabels=nombres, ylabel="coeficiente (rentas por °C)",
       title="Cada coeficiente por separado varía mucho; la suma, no")
plt.tight_layout()
plt.show()

# %% [markdown]
# La nube de la izquierda es una elipse estrecha y **inclinada hacia abajo**: cuando una réplica le da más peso a la temperatura, le quita casi lo mismo a la sensación, y viceversa. Los datos determinan muy bien la **suma** $w_{\text{temp}} + w_{\text{sens}}$ (el efecto de que suban ambas, que es lo que pasa casi siempre) y muy mal el **reparto**. A la derecha se ve en números: el coeficiente de la temperatura varía más de seis veces más que en el modelo sin sensación, mientras que la suma es tan estable como el coeficiente único del modelo simple.
#
# Esto no es casualidad. La matriz de covarianza $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$ tiene sus valores propios grandes en las direcciones donde $\mathbf{X}^\top\mathbf{X}$ tiene los pequeños, y la dirección casi degenerada es justo $(0, 1, -1)$: subir temp y bajar sensación en la misma cantidad cambia muy poco las predicciones, porque las dos columnas casi coinciden.
#
# Y las **predicciones** apenas se mueven: su desviación entre réplicas es de unas 2 rentas por hora con una o con dos variables. **La colinealidad es un problema para interpretar coeficientes, no para predecir.** Si lo que se quiere es interpretar («cada grado más da tantas rentas»), hay que quitar una de las variables redundantes, combinarlas, o regularizar con *ridge* (Unidad 3), que encoge justamente esas direcciones mal determinadas.

# %% [markdown]
# ## 6. Diagnóstico: los residuos de M4
#
# Un error de prueba bajo no garantiza que el modelo esté bien planteado. La herramienta clásica para revisar los supuestos es graficar los **residuos** $y_n - \hat y_n$ contra las predicciones $\hat y_n$. Si el modelo de la sección 2 fuera correcto, deberíamos ver una franja horizontal de ancho constante alrededor de cero, sin estructura.

# %%
m4 = modelos["M4"]
pred4 = m4.predict(prueba)
res4 = prueba.total.to_numpy() - pred4
negativas = np.mean(pred4 < 0)
print(f"RECM de prueba de M4: {recm(prueba.total, pred4):.1f}")
print(f"predicciones negativas: {np.sum(pred4 < 0):,} de {len(pred4):,} horas de prueba ({100 * negativas:.1f} %)")
print("horas del día con más predicciones negativas:",
      pd.Series(prueba.hora[pred4 < 0]).value_counts().head(6).to_dict())

# %%
bins = pd.qcut(pred4, 20)
banda = pd.DataFrame({"pred": pred4, "res": res4}).groupby(bins, observed=True).agg(centro=("pred", "mean"), de=("res", "std"))

fig, axs = plt.subplots(1, 2, figsize=(14, 5.3))
ax = axs[0]
ax.scatter(pred4, res4, s=6, alpha=0.2, color=AZUL, edgecolors="none")
ax.axhline(0, color=TINTA, lw=1)
ax.axvline(0, color=ROJO, lw=1.2, ls="--")
ax.plot(banda.centro, 2 * banda.de, color=DORADO, lw=2.5)
ax.plot(banda.centro, -2 * banda.de, color=DORADO, lw=2.5)
ax.text(-10, -400, f"{100 * negativas:.1f} % de las predicciones\nson negativas", color=ROJO, fontsize=11, ha="right")
ax.text(banda.centro.iloc[-1], -2 * banda.de.iloc[-1] - 25, "±2 desv. est.", color=DORADO, ha="right", va="top", fontsize=11)
ax.set(xlabel="predicción $\\hat y$ (M4)", ylabel="residuo $y - \\hat y$", title="Residuos contra predicciones (prueba)")
ax = axs[1]
ax.plot(banda.centro, banda.de, "o-", color=DORADO)
ax.set(xlabel="predicción $\\hat y$ (media del grupo)", ylabel="desviación estándar del residuo",
       title="La dispersión crece con la media", ylim=(0, None))
plt.tight_layout()
plt.show()

# %% [markdown]
# (La frontera diagonal de abajo a la izquierda es la recta $y = 0$, es decir, residuo $= -\hat y$: no se pueden rentar menos de cero bicicletas.) El diagnóstico tiene dos hallazgos, y los dos violan los supuestos:
#
# - **Embudo.** Los residuos se abren conforme crece la predicción: la desviación estándar pasa de unas pocas decenas de rentas cuando se esperan pocas bicicletas a más de cien en las horas pico. La varianza **no** es constante; crece con la media. El error cuadrático trata igual un error de 50 rentas a las 3 de la mañana (cuando se esperan 5) que a las 6 de la tarde (cuando se esperan 600), aunque el primero es grotesco y el segundo, razonable.
# - **Predicciones imposibles.** El **10.0 %** de las horas de prueba recibe una predicción negativa, casi todas de madrugada. Una recta no sabe que las rentas no pueden ser menores que cero.
#
# Las dos cosas tienen la misma raíz: las rentas son **conteos**. Para conteos, lo natural es que la varianza crezca con la media (en la distribución de Poisson son iguales) y que la media sea positiva. Además, los efectos se ven más **multiplicativos** que aditivos: la lluvia no quita 50 rentas a todas horas, quita una fracción de las que habría. Todo eso apunta a un modelo de la forma
#
# $$
# \log \mathbb{E}[y \mid \mathbf{x}] = \mathbf{w}^\top\mathbf{x}, \qquad y \mid \mathbf{x} \sim \text{Poisson},
# $$
#
# que sigue siendo lineal en $\mathbf{w}$ (ahora en la escala del logaritmo de la media) y se ajusta por máxima verosimilitud, igual que aquí, solo que con otro supuesto sobre el ruido. Es la **regresión de Poisson**, un modelo lineal generalizado, y es el tema de la siguiente libreta.

# %% [markdown]
# ## 7. Cierre
#
# - Mínimos cuadrados minimiza $E_{\text{in}}(\mathbf{w}) = \frac{1}{N}\lVert\mathbf{X}\mathbf{w} - \mathbf{y}\rVert^2$; anular el gradiente da las **ecuaciones normales** $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$.
# - Geométricamente, $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ es la **proyección ortogonal** de $\mathbf{y}$ sobre el espacio columna; la matriz sombrero es simétrica, idempotente y de traza $d + 1$. Probabilísticamente, con ruido gaussiano de varianza constante, mínimos cuadrados es **máxima verosimilitud**.
# - En la computadora no se forma $\mathbf{X}^\top\mathbf{X}$, porque $\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2$: se usa QR o SVD. El **descenso por gradiente** converge si $\eta < 2/\lambda_{\max}$, y su velocidad la dicta $\kappa$ de la hessiana: con temp en °C ($\kappa \approx 3{,}650$) tardó unas 4,600 iteraciones; estandarizada ($\kappa = 1$), una.
# - El modelo es lineal **en los parámetros**: con *one-hot* para la hora y la interacción hora × laboral, la RECM de prueba bajó de 165 a 86.4.
# - La **colinealidad** (temp y sensación, correlación 0.988, VIF ≈ 33) hace inestables los coeficientes individuales, pero no las predicciones.
# - Los residuos de M4 muestran un embudo y 10 % de predicciones negativas: las rentas son conteos, y el supuesto gaussiano no les queda. La siguiente libreta lo resuelve con regresión de Poisson.

# %% [markdown]
# ## 8. Autoevaluación

# %%
rdp.quiz.mostrar("u04", ids=["u04-p01", "u04-p02", "u04-p03", "u04-p04", "u04-p05"])

# %% [markdown]
# ## 9. Ejercicios
#
# 1. **La matriz sombrero.** Demuestra que $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ es simétrica e idempotente, que sus valores propios solo pueden ser 0 o 1, y que $\mathbf{I} - \mathbf{H}$ también es una proyección. ¿Sobre qué subespacio proyecta $\mathbf{I} - \mathbf{H}$? Usa esto para explicar por qué los residuos de un modelo con término constante siempre suman cero.
# 2. **Descenso por gradiente.** (a) Demuestra que con $\eta = 2/(\lambda_{\min} + \lambda_{\max})$ el factor de contracción del error es $(\kappa - 1)/(\kappa + 1)$ y que ninguna otra $\eta$ fija lo mejora. (b) Usa esa fórmula para estimar cuántas iteraciones necesita el caso sin centrar para reducir el error en $\mathbf{w}$ por un factor de $10^{3}$, y compáralo con la tabla de la sección 3. (c) Centra la temperatura pero **no** la escales (resta la media, deja los °C). ¿Cuánto vale $\kappa$ de la hessiana? ¿Cuántas iteraciones hacen falta?
# 3. **Una interacción numérica.** Agrega a M4 el producto de la temperatura por `laboral` (pista: crea la columna en el `DataFrame` y añádela a las numéricas). ¿Mejora la RECM de prueba? Interpreta el coeficiente: ¿la temperatura importa más en los días laborales o en los fines de semana? ¿Tiene sentido?
# 4. **El VIF.** Demuestra la fórmula $\operatorname{Var}(w_j) = \sigma^2 / \big(N s_j^2 (1 - R_j^2)\big)$. Pista: por el teorema de Frisch–Waugh–Lovell, $w_j$ es el coeficiente de la regresión de $\mathbf{y}$ sobre $\tilde{\mathbf{x}}_j$, el residuo de regresar $\mathbf{x}_j$ sobre las demás columnas, y $\lVert\tilde{\mathbf{x}}_j\rVert^2 = N s_j^2 (1 - R_j^2)$. Verifica numéricamente con el modelo `total ~ temp + sensacion`.
# 5. **Partir al azar.** Repite la escalera de la sección 4 partiendo las horas **al azar** (`train_test_split` con 37 % para prueba, `random_state=2011`). ¿Cómo cambian las RECM de prueba, sobre todo la de M4? ¿Por qué el error al azar es más optimista? ¿Qué pasaría si agregáramos al modelo la renta de la hora anterior como variable?
