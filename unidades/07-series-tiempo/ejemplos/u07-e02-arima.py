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
# # La curva de Keeling: pronosticar el CO₂ con ARIMA
#
# **Unidad 7 · Ejemplo 2**
#
# En el ejemplo anterior vimos que una serie de tiempo tiene memoria: el valor de hoy se parece al de ayer, y el de este mes al del mismo mes del año pasado. Los modelos **ARIMA** convierten esa memoria en un modelo probabilístico explícito, con pocos parámetros, que se ajusta por máxima verosimilitud y que da, además del pronóstico, un **intervalo de predicción** honesto. En esta libreta vamos a:
#
# 1. presentar la serie de CO₂ de Mauna Loa y plantear el problema: con los datos hasta 2014, pronosticar 2015–2026;
# 2. simular los procesos básicos (ruido blanco, AR(1), MA(1), caminata aleatoria) y aprender a reconocerlos por su **autocorrelación** (ACF) y su **autocorrelación parcial** (PACF);
# 3. demostrar las propiedades del AR(1) y comprobar que un AR($p$) es **una regresión lineal sobre los rezagos**, que podemos ajustar a mano por mínimos cuadrados;
# 4. usar la **diferenciación** para quitar tendencia y estacionalidad, ajustar un **SARIMA** elegido por AIC y revisar sus residuos;
# 5. derivar cómo crece la incertidumbre con el horizonte y comparar el pronóstico con lo que de verdad pasó, contra otros métodos y con validación de origen móvil.

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
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

import rdp

rdp.estilo.usar()
SEMILLA = 1958
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE
AZUL_TENUE = rdp.estilo.PALETA["azul-tenue"]
# statsmodels avisa de cosas inofensivas (frecuencia del índice, puntos de arranque); las silenciamos
warnings.filterwarnings("ignore", module="statsmodels")

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Contexto.** En marzo de 1958, Charles David Keeling, un joven químico del Instituto Scripps de Oceanografía, empezó a medir la concentración de dióxido de carbono en el aire desde el observatorio de Mauna Loa, en Hawái, a 3,400 m de altura y lejos de fuentes locales de contaminación. Quería saber si el CO₂ que la humanidad emitía al quemar combustibles fósiles se quedaba en la atmósfera. En pocos años sus mediciones mostraron dos cosas que hoy son la imagen más famosa de la ciencia del clima: un **ciclo anual** (las plantas del hemisferio norte absorben CO₂ en primavera y verano y lo liberan en otoño e invierno) montado sobre una **tendencia** que no deja de crecer. La serie se conoce como la **curva de Keeling** y la siguen midiendo Scripps y la NOAA.
#
# **Datos.** Promedios mensuales de la fracción molar de CO₂ en aire seco, en partes por millón (ppm), desde marzo de 1958, publicados por el Laboratorio de Monitoreo Global de la NOAA (P. Tans y R. Keeling; dominio público, se pide citar). Las columnas son `anio`, `mes`, `fecha_decimal`, `co2` y `desestacionalizado` (la serie sin el ciclo anual, calculada por la NOAA; no la usaremos). Los pocos meses de los primeros años sin medición vienen ya interpolados por la NOAA, así que la serie no tiene huecos.
#
# **Problema.** Usando solo los datos **hasta diciembre de 2014**, pronosticar la concentración mensual de **enero de 2015 en adelante**, con intervalos de predicción, y comparar con lo que de verdad se midió.

# %%
co2 = rdp.datos.cargar("co2_mauna_loa")
co2.index = pd.period_range(f"{co2.anio[0]}-{co2.mes[0]:02d}", periods=len(co2), freq="M")
assert (co2.index.year == co2.anio).all() and (co2.index.month == co2.mes).all()  # meses consecutivos, sin huecos
y = co2["co2"]
entrena, prueba = y[:"2014-12"], y["2015-01":]
print(f"{len(y)} meses, de {y.index[0]} a {y.index[-1]}; entrenamiento: {len(entrena)}, prueba: {len(prueba)}")
print(f"marzo de 1958: {y['1958-03']:.2f} ppm · máximo: {y.max():.2f} ppm ({y.idxmax()}) · último: {y.iloc[-1]:.2f} ppm ({y.index[-1]})")

# %%
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
fechas = co2.fecha_decimal
ax = ejes[0]
ax.plot(fechas[: len(entrena)], entrena, color=AZUL, lw=1.3)
ax.plot(fechas[len(entrena):], prueba, color=GRIS, lw=1.3)
ax.axvline(2015, color=TENUE, ls="--", lw=1)
ax.text(2014, 418, "entrenamiento", ha="right", color=AZUL, fontsize=11)
ax.text(2016, 395, "prueba", ha="left", color=GRIS, fontsize=11)
ax.set_title("CO₂ en Mauna Loa, promedio mensual")
ax.set_xlabel("año")
ax.set_ylabel("ppm")
ax = ejes[1]
tramo = co2.loc["2011-01":"2014-12"]
ax.plot(tramo.fecha_decimal, tramo.co2, color=AZUL, marker="o", ms=3.5, lw=1.3)
ax.set_title("Cuatro años de cerca: el ciclo anual")
ax.set_xlabel("año")
ax.set_xticks([2011, 2012, 2013, 2014, 2015])
fig.tight_layout()
plt.show()

# %%
anual = y.groupby(y.index.year).mean()
crecimiento = anual.diff()
completo = y.groupby(y.index.year).size() == 12
crecimiento = crecimiento[completo & completo.shift(1, fill_value=False)]
amplitud = (y.groupby(y.index.year).max() - y.groupby(y.index.year).min())[completo]
print("crecimiento medio (ppm por año) por década:")
print(crecimiento.groupby(crecimiento.index // 10 * 10).mean().round(2).to_string())
print(f"amplitud media del ciclo anual (máx − mín de cada año): {amplitud.mean():.1f} ppm")

# %% [markdown]
# Tres cosas saltan a la vista y las tendremos que modelar:
#
# - una **tendencia** creciente y que además **se acelera**: el aumento anual pasó de menos de 1 ppm por año en los años sesenta a 2.4 en la década de 2010 y a más de 2.6 en la actual;
# - un **ciclo anual** muy regular, de unos 6 ppm entre el máximo (mayo) y el mínimo (septiembre u octubre);
# - poco ruido: los meses se desvían de la curva suave apenas unas décimas de ppm.
#
# Un modelo para esta serie tiene que capturar la tendencia, el ciclo y la memoria de corto plazo. Antes de atacarla, construyamos las piezas con series simuladas, donde sabemos la verdad.

# %% [markdown]
# ## 2. Modelos con memoria
#
# ### Ruido blanco y estacionariedad
#
# El ladrillo básico es el **ruido blanco**: variables $\varepsilon_t$ con media 0, varianza $\sigma^2$ y no correlacionadas entre sí. No tiene memoria: saber $\varepsilon_{t-1}$ no ayuda a predecir $\varepsilon_t$.
#
# Recordemos de la libreta anterior que una serie es **débilmente estacionaria** si su media $\mathbb{E}[y_t] = \mu$ no depende de $t$ y su autocovarianza $\gamma_k = \operatorname{Cov}(y_t, y_{t-k})$ depende solo del rezago $k$. Entonces la **autocorrelación** es $\rho_k = \gamma_k / \gamma_0$, y su estimador, la ACF muestral, es
#
# $$\hat\rho_k = \frac{\sum_{t=k+1}^{T} (y_t - \bar y)(y_{t-k} - \bar y)}{\sum_{t=1}^{T} (y_t - \bar y)^2}.$$
#
# Para ruido blanco, $\hat\rho_k$ es aproximadamente normal con media 0 y varianza $1/T$; por eso dibujamos bandas en $\pm 1.96/\sqrt{T}$.
#
# ### El modelo autorregresivo de orden 1
#
# La forma más simple de darle memoria a una serie es que cada valor sea una fracción del anterior más una sorpresa nueva:
#
# $$y_t = c + \phi\, y_{t-1} + \varepsilon_t, \qquad \varepsilon_t \text{ ruido blanco con varianza } \sigma^2 .$$
#
# **Cuándo es estacionario.** Sustituyendo hacia atrás $k$ veces,
#
# $$y_t = c\,(1 + \phi + \dots + \phi^{k-1}) + \phi^k y_{t-k} + \sum_{j=0}^{k-1} \phi^j \varepsilon_{t-j}.$$
#
# Si $|\phi| < 1$, el término $\phi^k y_{t-k}$ desaparece cuando $k\to\infty$ y queda $y_t = \frac{c}{1-\phi} + \sum_{j\ge 0}\phi^j\varepsilon_{t-j}$: el proceso es una suma de choques pasados cuyo peso decae geométricamente, y es estacionario. Si $|\phi| \ge 1$, el pasado lejano nunca se olvida (o se amplifica) y no hay distribución estable.
#
# **Media, varianza y ACF.** Con $|\phi|<1$, tomando esperanza en el modelo, $\mu = c + \phi\mu$, así que $\mu = c/(1-\phi)$. Tomando varianzas, y como $\varepsilon_t$ no está correlacionada con $y_{t-1}$ (que solo depende de choques anteriores),
#
# $$\gamma_0 = \phi^2\gamma_0 + \sigma^2 \quad\Longrightarrow\quad \gamma_0 = \frac{\sigma^2}{1-\phi^2}.$$
#
# Multiplicando $y_t - \mu = \phi(y_{t-1}-\mu) + \varepsilon_t$ por $y_{t-k}-\mu$ con $k\ge1$ y tomando esperanza, $\gamma_k = \phi\,\gamma_{k-1}$, de donde
#
# $$\rho_k = \phi^k .$$
#
# La autocorrelación decae **geométricamente**: lentamente si $\phi$ está cerca de 1, y alternando de signo si $\phi < 0$.
#
# ### La caminata aleatoria
#
# El caso frontera $\phi = 1$ es la **caminata aleatoria** $y_t = y_{t-1} + \varepsilon_t$. Partiendo de $y_0 = 0$, $y_t = \sum_{j=1}^t \varepsilon_j$ tiene varianza $t\sigma^2$, que crece sin límite: no es estacionaria. Pero su **diferencia** $\nabla y_t = y_t - y_{t-1} = \varepsilon_t$ es ruido blanco. Esa observación es la «I» de ARIMA.
#
# ### Medias móviles
#
# Otra forma de memoria: que el choque de hoy deje un eco mañana,
#
# $$y_t = \mu + \varepsilon_t + \theta\,\varepsilon_{t-1} \qquad \text{(MA(1))}.$$
#
# Aquí $\gamma_0 = (1+\theta^2)\sigma^2$, $\gamma_1 = \theta\sigma^2$ y $\gamma_k = 0$ para $k\ge2$, porque $y_t$ y $y_{t-2}$ no comparten ningún choque. Un MA($q$), $y_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1}+\dots+\theta_q\varepsilon_{t-q}$, tiene ACF que **se corta** después del rezago $q$. (El nombre «media móvil» es histórico; no tiene que ver con suavizar promediando.)
#
# Simulemos los cinco procesos con 500 observaciones cada uno.

# %%
def simular_arma(phi=(), theta=(), n=500, sigma=1.0, rng=None, quemado=300):
    """Simula y_t = Σ φ_i y_{t-i} + ε_t + Σ θ_j ε_{t-j}; descarta los primeros `quemado` valores."""
    phi, theta = np.atleast_1d(phi), np.atleast_1d(theta)
    e = rng.normal(0, sigma, n + quemado)
    y = np.zeros(n + quemado)
    for t in range(n + quemado):
        ar = sum(phi[i] * y[t - 1 - i] for i in range(len(phi)) if t - 1 - i >= 0)
        ma = sum(theta[j] * e[t - 1 - j] for j in range(len(theta)) if t - 1 - j >= 0)
        y[t] = ar + e[t] + ma
    return y[quemado:]


rng = np.random.default_rng(SEMILLA)
T = 500
procesos = {
    "ruido blanco": rng.normal(0, 1, T),
    "AR(1), φ = 0.9": simular_arma(phi=0.9, rng=rng),
    "AR(1), φ = −0.7": simular_arma(phi=-0.7, rng=rng),
    "MA(1), θ = 0.8": simular_arma(theta=0.8, rng=rng),
    "caminata aleatoria": np.cumsum(rng.normal(0, 1, T)),
}
teorica = {
    "ruido blanco": lambda k: 0 * k,
    "AR(1), φ = 0.9": lambda k: 0.9**k,
    "AR(1), φ = −0.7": lambda k: (-0.7) ** k,
    "MA(1), θ = 0.8": lambda k: np.where(k == 1, 0.8 / (1 + 0.8**2), 0.0),
    "caminata aleatoria": None,
}
K = 20
rezagos = np.arange(1, K + 1)
banda = 1.96 / np.sqrt(T)

fig, ejes = plt.subplots(5, 3, figsize=(13, 13.5), gridspec_kw={"width_ratios": [2, 1, 1]})
for fila, (nombre, serie) in zip(ejes, procesos.items()):
    fila[0].plot(serie[:200], color=AZUL, lw=1.1)
    fila[0].set_title(nombre)
    for ax, valores, titulo in ((fila[1], acf(serie, nlags=K)[1:], "ACF"), (fila[2], pacf(serie, nlags=K)[1:], "PACF")):
        ax.axhspan(-banda, banda, color=AZUL_TENUE, alpha=0.5, lw=0)
        ax.vlines(rezagos, 0, valores, color=AZUL, lw=2.4)
        ax.axhline(0, color=TENUE, lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_title(titulo)
    if teorica[nombre] is not None:
        fila[1].plot(rezagos, teorica[nombre](rezagos), "o", color=ROJO, ms=4, zorder=4)
for ax in ejes[-1]:
    ax.set_xlabel("t" if ax is ejes[-1, 0] else "rezago k")
ejes[0, 1].text(K, 0.8, "● teórica", color=ROJO, ha="right", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# Las barras son las autocorrelaciones muestrales; los puntos rojos, las teóricas $\rho_k$; la franja azul, las bandas $\pm1.96/\sqrt T$.
#
# - **Ruido blanco:** casi todas las barras dentro de la banda (con 20 rezagos, esperamos que una caiga fuera por azar).
# - **AR(1) con $\phi=0.9$:** la serie se queda arriba o abajo por rachas largas; la ACF decae lentamente como $0.9^k$.
# - **AR(1) con $\phi=-0.7$:** la serie zigzaguea; la ACF alterna de signo.
# - **MA(1):** una sola autocorrelación grande, $\rho_1 = 0.8/1.64 = 0.49$, y luego nada.
# - **Caminata aleatoria:** la serie deambula; la ACF muestral apenas baja. Una ACF que decae así de lento es la firma de una serie **no estacionaria** que hay que diferenciar.
#
# ### La autocorrelación parcial
#
# La ACF de un AR(1) no se corta: $y_t$ está correlacionada con $y_{t-2}$ aunque el modelo solo use $y_{t-1}$, porque la correlación «pasa» a través de $y_{t-1}$. La **autocorrelación parcial** $\alpha_k$ mide la correlación entre $y_t$ y $y_{t-k}$ **descontando** lo que explican los rezagos intermedios. Una definición operativa: $\alpha_k$ es el último coeficiente $\phi_{kk}$ de la mejor regresión lineal de $y_t$ sobre $y_{t-1}, \dots, y_{t-k}$. Por eso, en un AR($p$), $\alpha_k = 0$ para $k > p$: la PACF **se corta** en $p$. En la figura, la PACF de los dos AR(1) tiene una sola barra y la del MA(1) decae alternando.
#
# Esta dualidad es la base del método de identificación de Box y Jenkins (1970):
#
# | modelo | ACF | PACF |
# |---|---|---|
# | AR($p$) | decae (geométrica u oscilante) | se corta después de $p$ |
# | MA($q$) | se corta después de $q$ | decae |
# | ARMA($p$, $q$) | decae a partir de $q$ | decae a partir de $p$ |
# | no estacionaria | decae muy lento, casi lineal | primera barra cerca de 1 |
#
# ### Comprobación numérica del AR(1)
#
# Verifiquemos $\gamma_0 = \sigma^2/(1-\phi^2)$ y $\rho_k = \phi^k$ con series más largas (50,000 pasos), para que el error de muestreo sea pequeño.

# %%
filas = []
for phi in (0.5, 0.9, -0.7):
    s = simular_arma(phi=phi, n=50_000, rng=rng)
    r = acf(s, nlags=3)
    filas.append({"φ": phi, "varianza muestral": s.var(), "σ²/(1−φ²)": 1 / (1 - phi**2),
                  "ρ̂₁": r[1], "ρ̂₂": r[2], "φ²": phi**2, "ρ̂₃": r[3], "φ³": phi**3})
pd.DataFrame(filas).round(3)

# %% [markdown]
# Coinciden hasta la segunda o tercera cifra. Con $\phi = 0.9$ la varianza es 5.3 veces la del ruido: la memoria acumula choques.
#
# ## 3. Un AR($p$) es una regresión lineal
#
# El AR($p$) general es
#
# $$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \varepsilon_t .$$
#
# Visto con los ojos de la Unidad 4, es **un modelo lineal** cuyos atributos son los valores pasados de la propia serie. Si armamos la matriz de diseño con una fila por tiempo $t = p+1, \dots, T$,
#
# $$X = \begin{pmatrix} 1 & y_p & y_{p-1} & \cdots & y_1 \\ 1 & y_{p+1} & y_p & \cdots & y_2 \\ \vdots & & & & \vdots \\ 1 & y_{T-1} & y_{T-2} & \cdots & y_{T-p}\end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} y_{p+1} \\ \vdots \\ y_T\end{pmatrix},$$
#
# los coeficientes salen de mínimos cuadrados, $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top\mathbf{y}$, y $\hat\sigma^2$ es la varianza de los residuos. Con errores gaussianos esto es la máxima verosimilitud **condicional** en las primeras $p$ observaciones; la verosimilitud exacta (la que usa `ARIMA`) las incluye también, y para series largas la diferencia es mínima.
#
# **¿Cuándo es estacionario?** Escribiendo el **operador de rezago** $B y_t = y_{t-1}$, el modelo es $\phi(B)\,y_t = c + \varepsilon_t$ con el polinomio característico $\phi(z) = 1 - \phi_1 z - \dots - \phi_p z^p$. El AR($p$) es estacionario si y solo si **todas las raíces de $\phi(z)$ están fuera del círculo unitario** (para el AR(1), la raíz es $1/\phi$, y $|1/\phi| > 1$ es lo mismo que $|\phi| < 1$). Raíces complejas dan una ACF que oscila amortiguándose: ciclos «casi periódicos».
#
# Probemos con un AR(2) con $\phi_1 = 0.75$, $\phi_2 = -0.5$ y $c = 2$, cuyas raíces son complejas.

# %%
phi_verd = np.array([0.75, -0.5])
c_verd = 2.0
s = simular_arma(phi=phi_verd, n=1_000, rng=rng) + c_verd / (1 - phi_verd.sum())  # media c/(1 − φ1 − φ2)


def matriz_rezagos(serie, p):
    """Matriz de diseño [1, y_{t-1}, …, y_{t-p}] y respuesta y_t para t = p+1, …, T."""
    T = len(serie)
    X = np.column_stack([np.ones(T - p)] + [serie[p - i : T - i] for i in range(1, p + 1)])
    return X, serie[p:]


X, objetivo = matriz_rezagos(s, 2)
w, *_ = np.linalg.lstsq(X, objetivo, rcond=None)
sigma2 = np.mean((objetivo - X @ w) ** 2)

autoreg = AutoReg(s, lags=2, trend="c").fit()
mv = ARIMA(s, order=(2, 0, 0)).fit()
mu_mv = mv.params[0]  # ARIMA reporta la media, no la constante c
comparacion = pd.DataFrame({
    "verdadero": [c_verd, *phi_verd],
    "mínimos cuadrados (a mano)": w,
    "AutoReg": autoreg.params,
    "ARIMA (verosimilitud exacta)": [mu_mv * (1 - mv.params[1:3].sum()), *mv.params[1:3]],
}, index=["c", "φ1", "φ2"])
comparacion.round(4)

# %%
raices = np.roots([-phi_verd[1], -phi_verd[0], 1])  # φ(z) = 1 − 0.75 z + 0.5 z², coeficientes de mayor a menor grado
print("raíces de φ(z):", np.round(raices, 3), " módulo:", np.round(np.abs(raices), 3))
print("raíces estimadas:", np.round(np.roots([-w[2], -w[1], 1]), 3))
print(f"σ² estimada a mano: {sigma2:.3f} (verdadera: 1)")

# %% [markdown]
# Los mínimos cuadrados hechos a mano coinciden con `AutoReg` hasta la última cifra (es exactamente el mismo cálculo) y la verosimilitud exacta de `ARIMA` difiere apenas en la cuarta decimal. Las raíces del polinomio característico son $0.75 \pm 1.20\,i$, de módulo $\sqrt{2} = 1.41 > 1$: el proceso es estacionario y, por ser complejas, su ACF oscila. Su argumento, $\pm 1.01$ radianes, da un «periodo» aproximado de $2\pi/1.01 \approx 6$ pasos.
#
# Todo lo que aprendimos de modelos lineales se traslada: los errores estándar, la regularización, y también los peligros. En particular, **no** se puede validar un AR($p$) con validación cruzada aleatoria (ejemplo anterior): los atributos son la propia serie desplazada.

# %% [markdown]
# ## 4. Diferenciar: quitar la tendencia y el ciclo
#
# La serie de CO₂ no es estacionaria: su media sube. La receta de Box y Jenkins es **diferenciar** hasta obtener algo estacionario y modelar eso con un ARMA:
#
# - la **diferencia** $\nabla y_t = y_t - y_{t-1} = (1-B)y_t$ quita una tendencia lineal (y $\nabla^2$, una cuadrática);
# - la **diferencia estacional** $\nabla_{12}\, y_t = y_t - y_{t-12} = (1-B^{12})y_t$ quita un ciclo anual que se repite; además, como compara el mismo mes de dos años seguidos, es el **crecimiento anual**;
# - aplicadas ambas, $\nabla\nabla_{12}\, y_t = (1-B)(1-B^{12})\,y_t$: el cambio en el crecimiento anual de un mes al siguiente.
#
# Veamos cada versión de la serie de entrenamiento con su ACF hasta 48 meses.

# %%
versiones = {
    r"$y$ (original)": entrena,
    r"$\nabla y$ (diferencia mensual)": entrena.diff(),
    r"$\nabla_{12}\, y$ (crecimiento anual)": entrena.diff(12),
    r"$\nabla\nabla_{12}\, y$": entrena.diff(12).diff(),
}
K = 48
fig, ejes = plt.subplots(4, 2, figsize=(13, 12), gridspec_kw={"width_ratios": [1.6, 1]})
for fila, (nombre, serie) in zip(ejes, versiones.items()):
    serie = serie.dropna()
    fila[0].plot(serie.index.year + (serie.index.month - 0.5) / 12, serie, color=AZUL, lw=0.9)
    fila[0].set_title(nombre)
    r = acf(serie, nlags=K)[1:]
    fila[1].axhspan(-1.96 / np.sqrt(len(serie)), 1.96 / np.sqrt(len(serie)), color=AZUL_TENUE, alpha=0.5, lw=0)
    fila[1].vlines(np.arange(1, K + 1), 0, r, color=AZUL, lw=2)
    fila[1].vlines([12, 24, 36, 48], 0, r[[11, 23, 35, 47]], color=ROJO, lw=2.6)
    fila[1].axhline(0, color=TENUE, lw=0.8)
    fila[1].set_ylim(-1, 1)
    fila[1].set_xticks([0, 12, 24, 36, 48])
    fila[1].set_title("ACF (rojo: múltiplos de 12)")
ejes[-1, 0].set_xlabel("año")
ejes[-1, 1].set_xlabel("rezago (meses)")
fig.tight_layout()
plt.show()

# %%
d2 = entrena.diff(12).diff().dropna()
r = acf(d2, nlags=36)
a = pacf(d2, nlags=36)
print(f"∇∇₁₂y: {len(d2)} valores, media {d2.mean():.4f}, desviación estándar {d2.std():.3f} ppm")
print("ACF  en rezagos 1, 2, 3, 12, 24:", np.round(r[[1, 2, 3, 12, 24]], 2))
print("PACF en rezagos 1, 2, 3, 12, 24:", np.round(a[[1, 2, 3, 12, 24]], 2))

# %% [markdown]
# - La serie original tiene una ACF que casi no baja: no estacionaria, como la caminata aleatoria.
# - $\nabla y$ quita la tendencia, pero el ciclo anual domina: la ACF es una onda de periodo 12 que no se apaga.
# - $\nabla_{12} y$ quita el ciclo; lo que queda es el crecimiento anual, que todavía deambula (sube de ~1 a ~2 ppm por año) y su ACF decae lento.
# - $\nabla\nabla_{12} y$ ya parece estacionaria, con media prácticamente cero. Su ACF tiene **dos barras claras**: en el rezago 1 ($-0.29$) y en el 12 ($-0.46$), y casi nada más. Leído con la tabla de Box y Jenkins: un MA(1) para la parte de corto plazo y un MA(1) **estacional** (rezago 12). La PACF que decae en los múltiplos de 12 confirma la parte estacional de media móvil.
#
# ## 5. ARIMA y SARIMA: ajuste y selección
#
# Juntando las piezas con el operador de rezago, un **ARIMA($p,d,q$)** es un ARMA($p,q$) para la serie diferenciada $d$ veces:
#
# $$\phi(B)\,(1-B)^d\, y_t = \theta(B)\,\varepsilon_t, \qquad \phi(z) = 1-\sum_{i=1}^p\phi_i z^i,\quad \theta(z) = 1+\sum_{j=1}^q \theta_j z^j .$$
#
# El **SARIMA$(p,d,q)(P,D,Q)_m$** agrega polinomios en $B^m$ para la parte estacional, que se multiplican con los de corto plazo:
#
# $$\phi(B)\,\Phi(B^m)\,(1-B)^d(1-B^m)^D\, y_t = \theta(B)\,\Theta(B^m)\,\varepsilon_t .$$
#
# Por ejemplo, el SARIMA$(0,1,1)(0,1,1)_{12}$, que la identificación anterior sugiere, es $(1-B)(1-B^{12})\,y_t = (1+\theta B)(1+\Theta B^{12})\,\varepsilon_t$; desarrollado,
#
# $$y_t = y_{t-1} + y_{t-12} - y_{t-13} + \varepsilon_t + \theta\varepsilon_{t-1} + \Theta\varepsilon_{t-12} + \theta\Theta\varepsilon_{t-13}.$$
#
# Box y Jenkins lo usaron para el número mensual de pasajeros aéreos, y desde entonces se le llama el **modelo de aerolíneas**.
#
# **Ajuste.** Los términos MA hacen que $\varepsilon_{t-1}$ no se observe directamente, así que ya no basta una regresión. Con errores gaussianos, la verosimilitud exacta se calcula con el filtro de Kalman (es lo que hace `SARIMAX` de `statsmodels`) y se maximiza numéricamente.
#
# **Selección.** Más parámetros siempre aumentan la verosimilitud; el **criterio de información de Akaike** penaliza la complejidad:
#
# $$\mathrm{AIC} = -2\,\ln \hat L + 2k ,$$
#
# con $k$ el número de parámetros estimados. Se elige el modelo de menor AIC. (Es una estimación del error fuera de la muestra en escala de log-verosimilitud, como $E_{\text{out}} \approx E_{\text{in}} + $ penalización en la Unidad 3; el BIC usa $k\ln T$ y castiga más.) Los órdenes de diferenciación $d$ y $D$ los fijamos antes: el AIC solo compara modelos para la **misma** serie diferenciada.
#
# Recorramos una malla pequeña con $d = D = 1$.

# %%
malla = []
for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    ajuste = SARIMAX(entrena, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
    malla.append({"modelo": f"({p},1,{q})({P},1,{Q})₁₂", "parámetros": p + q + P + Q + 1, "AIC": ajuste.aic})
malla = pd.DataFrame(malla).sort_values("AIC").reset_index(drop=True)
malla["Δ AIC"] = malla.AIC - malla.AIC[0]
print(f"{len(malla)} modelos")
pd.concat([malla.head(6), malla.tail(2)]).round(2)

# %% [markdown]
# El mejor es el SARIMA$(1,1,1)(0,1,1)_{12}$, aunque los primeros cinco están a menos de 1.1 unidades de AIC y prácticamente empatan (una regla usual: diferencias menores que 2 no son concluyentes). Lo que **todos** los buenos tienen en común es el término $\Theta$ (MA estacional); los que no lo tienen están decenas o cientos de unidades atrás. Nos quedamos con el ganador.

# %%
modelo = SARIMAX(entrena, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
pd.DataFrame({"estimación": modelo.params, "error estándar": modelo.bse}).rename(
    index={"ar.L1": "φ (AR 1)", "ma.L1": "θ (MA 1)", "ma.S.L12": "Θ (MA estacional 12)", "sigma2": "σ²"}).round(4)

# %% [markdown]
# El término estacional $\Theta = -0.87$ es el más importante y está muy bien determinado. Su interpretación: $(1-B^{12})$ y $(1 + \Theta B^{12})$ casi se cancelan, así que el modelo dice «el ciclo anual cambia, pero muy despacio»; un $\Theta = -1$ sería un ciclo exactamente fijo. La desviación estándar del ruido es $\hat\sigma = \sqrt{0.092} = 0.30$ ppm: un mes cualquiera se sale del patrón apenas unas décimas.
#
# ## 6. Diagnóstico: ¿los residuos son ruido blanco?
#
# Si el modelo capturó toda la estructura, sus residuos (errores de pronóstico a un paso dentro de la muestra) deben parecer ruido blanco. Lo revisamos a ojo con su ACF y formalmente con la **prueba de Ljung–Box**, que junta las primeras $K$ autocorrelaciones en
#
# $$Q = T(T+2)\sum_{k=1}^{K}\frac{\hat\rho_k^2}{T-k},$$
#
# que, bajo la hipótesis de ruido blanco, tiene aproximadamente distribución $\chi^2$ con $K - (\text{parámetros ARMA})$ grados de libertad. Un valor $p$ chico indica autocorrelación que el modelo no explicó.

# %%
residuos = modelo.resid[13:]  # los primeros 13 meses se pierden al diferenciar (1 + 12)
lb = acorr_ljungbox(residuos, lags=[12, 24, 36], model_df=3)
fig, ejes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.6, 1, 0.9]})
ejes[0].plot(residuos.index.year + (residuos.index.month - 0.5) / 12, residuos, color=AZUL, lw=0.7)
ejes[0].set_title("Residuos del SARIMA (ppm)")
r = acf(residuos, nlags=36)[1:]
ejes[1].axhspan(-1.96 / np.sqrt(len(residuos)), 1.96 / np.sqrt(len(residuos)), color=AZUL_TENUE, alpha=0.5, lw=0)
ejes[1].vlines(np.arange(1, 37), 0, r, color=AZUL, lw=2)
ejes[1].axhline(0, color=TENUE, lw=0.8)
ejes[1].set_ylim(-0.5, 0.5)
ejes[1].set_xticks([0, 12, 24, 36])
ejes[1].set_title("ACF de los residuos")
ejes[2].hist(residuos, bins=30, color=AZUL_TENUE, edgecolor="white")
ejes[2].set_title("Histograma")
fig.tight_layout()
plt.show()
lb.round(3)

# %% [markdown]
# Las autocorrelaciones de los residuos caen dentro de las bandas (la más grande es de 0.08), el histograma es simétrico y la prueba de Ljung–Box no encuentra estructura en ningún horizonte ($p$ entre 0.6 y 0.8). El modelo pasa el diagnóstico. Compáralo con la ACF de $\nabla\nabla_{12}y$: las barras en 1 y en 12 desaparecieron, que es justo lo que $\theta$ y $\Theta$ debían explicar.

# %% [markdown]
# ## 7. Pronósticos: cómo crece la incertidumbre
#
# Para pronosticar $h$ pasos adelante con información hasta $T$, se usa la esperanza condicional $\hat y_{T+h\mid T} = \mathbb{E}[y_{T+h}\mid y_1,\dots,y_T]$: en la ecuación del modelo, los $y$ futuros se sustituyen por sus pronósticos y los choques futuros por cero.
#
# **AR(1).** Con media cero, iterando el modelo,
#
# $$y_{T+h} = \phi^h y_T + \sum_{j=0}^{h-1}\phi^j\varepsilon_{T+h-j},$$
#
# así que $\hat y_{T+h\mid T} = \phi^h y_T$ (el pronóstico regresa a la media geométricamente) y el error de pronóstico es la suma de los choques que todavía no ocurren. Como son independientes,
#
# $$\operatorname{Var}\big(y_{T+h} - \hat y_{T+h\mid T}\big) = \sigma^2\sum_{j=0}^{h-1}\phi^{2j} = \sigma^2\,\frac{1-\phi^{2h}}{1-\phi^2} \;\xrightarrow[h\to\infty]{}\; \frac{\sigma^2}{1-\phi^2}=\gamma_0 .$$
#
# En el horizonte largo la incertidumbre se estabiliza en la varianza del proceso: pronosticar lejos equivale a pronosticar la media.
#
# **Caminata aleatoria** ($\phi=1$): $\hat y_{T+h\mid T} = y_T$ y la varianza del error es $h\sigma^2$: **crece sin límite**, y el ancho del intervalo crece como $\sqrt h$. Todo modelo con $d \ge 1$ hereda este comportamiento; los ARIMA de la serie de CO₂ tienen $d = D = 1$, y sus intervalos se abren sin cota.
#
# En general, todo ARIMA se puede escribir como $y_t = \sum_{j\ge0}\psi_j\varepsilon_{t-j}$ (los **pesos $\psi$**), y la varianza del error a $h$ pasos es $\sigma^2\sum_{j<h}\psi_j^2$. El intervalo de predicción del 95 % es $\hat y_{T+h\mid T} \pm 1.96\,\sigma\sqrt{\sum_{j<h}\psi_j^2}$, suponiendo errores gaussianos y **parámetros conocidos** (la incertidumbre de estimar $\phi$, $\theta$… no se incluye).
#
# Comprobémoslo por simulación: muchas trayectorias que comparten el pasado y se separan después de $T$.

# %%
def varianza_error(phi, h):
    """Varianza teórica (σ² = 1) del error de pronóstico a h pasos de un AR(1); φ = 1 es la caminata aleatoria."""
    return h if phi == 1 else (1 - phi ** (2 * h)) / (1 - phi**2)


H = 24
horizontes = np.arange(1, H + 1)
replicas = 20_000
casos = {"AR(1), φ = 0.5": 0.5, "AR(1), φ = 0.9": 0.9, "caminata aleatoria": 1.0}
y_T = 2.0  # último valor observado, igual para todas las réplicas
fig, ax = plt.subplots(figsize=(10, 5))
valores_finales = {}
for (nombre, phi), color in zip(casos.items(), [DORADO_OSC, AZUL, ROJO]):
    trayectorias = np.empty((replicas, H))
    actual = np.full(replicas, y_T)
    for h in range(H):
        actual = phi * actual + rng.normal(0, 1, replicas)
        trayectorias[:, h] = actual
    error = trayectorias - (phi**horizontes) * y_T  # error contra el pronóstico φ^h y_T
    ax.plot(horizontes, np.sqrt(varianza_error(phi, horizontes)), color=color, lw=2.4)
    ax.plot(horizontes[::2], error.std(axis=0)[::2], "o", color=color, ms=5)
    valores_finales[nombre] = np.sqrt(varianza_error(phi, H))
rdp.estilo.etiquetar(ax, H + 0.6, valores_finales)
ax.set_xlim(0, H + 6)
ax.set_xlabel("horizonte h")
ax.set_ylabel("desviación estándar del error (σ = 1)")
ax.set_title("Error de pronóstico contra horizonte (línea: teoría; puntos: 20,000 simulaciones)")
plt.show()
print({k: round(float(v), 2) for k, v in valores_finales.items()})

# %% [markdown]
# Con $\phi = 0.5$ el error se estabiliza casi de inmediato en $1/\sqrt{1-0.25} = 1.15$; con $\phi = 0.9$ tarda más y llega a $1/\sqrt{0.19} = 2.29$; la caminata aleatoria sigue creciendo como $\sqrt h$ ($\sqrt{24} = 4.9$ a 24 pasos). Las simulaciones caen sobre las curvas teóricas.
#
# ## 8. El pronóstico del CO₂, contra la realidad
#
# Pronostiquemos desde diciembre de 2014 los 141 meses de prueba (casi doce años) con el SARIMA, y comparemos con tres alternativas:
#
# - **estacional ingenuo:** cada mes futuro vale lo mismo que ese mes en 2014;
# - **regresión lineal sobre tendencia + mes:** $y_t = w_0 + w_1 t + \sum_{m} w_m\,\mathbb{1}[\text{mes}=m] + \varepsilon_t$, ajustada por mínimos cuadrados con todos los datos de entrenamiento (la Unidad 4 tal cual);
# - **regresión con tendencia cuadrática + mes:** igual, agregando $t^2$.
#
# Medimos con el error absoluto medio (MAE), la raíz del error cuadrático medio (RECM) y el MASE, que divide el MAE entre el del estacional ingenuo a un paso en el entrenamiento (libreta anterior).

# %%
pron = modelo.get_forecast(len(prueba))
media = pron.predicted_mean
int80, int95 = pron.conf_int(alpha=0.20), pron.conf_int(alpha=0.05)

t = np.arange(len(y), dtype=float)
meses = pd.get_dummies(co2.mes).to_numpy(dtype=float)  # 12 indicadoras, sin intersección aparte
X_lin = np.column_stack([t, meses])
X_cuad = np.column_stack([t, (t / 100) ** 2, meses])
n_ent = len(entrena)


def regresion(X, hasta):
    """Ajusta por mínimos cuadrados con las filas [0, hasta) y predice todas."""
    w, *_ = np.linalg.lstsq(X[:hasta], y.to_numpy()[:hasta], rcond=None)
    return X @ w


pronosticos = {
    "SARIMA": media.to_numpy(),
    "estacional ingenuo": np.resize(entrena.to_numpy()[-12:], len(prueba)),
    "tendencia lineal + mes": regresion(X_lin, n_ent)[n_ent:],
    "tendencia cuadrática + mes": regresion(X_cuad, n_ent)[n_ent:],
}
escala = np.mean(np.abs(entrena.to_numpy()[12:] - entrena.to_numpy()[:-12]))  # MAE del estacional ingenuo a un paso, en entrenamiento


def metricas(real, pred):
    e = real - pred
    return {"MAE": np.mean(np.abs(e)), "RECM": np.sqrt(np.mean(e**2)), "MASE": np.mean(np.abs(e)) / escala}


print(f"escala del MASE: {escala:.3f} ppm")
tabla_larga = pd.DataFrame({k: metricas(prueba.to_numpy(), v) for k, v in pronosticos.items()}).T
tabla_larga.round(2)

# %%
fx = co2.fecha_decimal.to_numpy()
fx_p = fx[n_ent:]
fig, ejes = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [1.5, 1]})
ax = ejes[0]
ax.fill_between(fx_p, int95.iloc[:, 0], int95.iloc[:, 1], color=AZUL_TENUE, alpha=0.55, lw=0)
ax.fill_between(fx_p, int80.iloc[:, 0], int80.iloc[:, 1], color=AZUL_TENUE, lw=0)
ax.plot(fx[n_ent - 60 : n_ent], entrena[-60:], color=TINTA, lw=1.2)
ax.plot(fx_p, prueba, color=TINTA, lw=1.2)
ax.plot(fx_p, media, color=AZUL, lw=1.6)
ax.axvline(2015, color=TENUE, ls="--", lw=1)
ax.text(2018.6, 416.5, "observado", color=TINTA, fontsize=11, ha="right")
ax.text(2026.9, 408, "SARIMA\n(intervalos de 80 % y 95 %)", color=AZUL, fontsize=11, ha="right", va="top")
ax.set_xticks([2010, 2014, 2018, 2022, 2026])
ax.set_title("Pronóstico desde diciembre de 2014")
ax.set_xlabel("año")
ax.set_ylabel("ppm")
ax = ejes[1]
anios = prueba.index.year
for (nombre, pred), color in zip(pronosticos.items(), [AZUL, GRIS, ROJO, DORADO_OSC]):
    error_anual = pd.Series(np.abs(prueba.to_numpy() - pred)).groupby(anios).mean()
    ax.plot(error_anual.index, error_anual, marker="o", ms=4, color=color)
ax.set_yscale("log")
ax.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
finales = {n: np.abs(prueba["2026"].to_numpy() - p[-9:]).mean() for n, p in pronosticos.items()}
rdp.estilo.etiquetar(ax, 2026.4, {"SARIMA": finales["SARIMA"], "estacional\ningenuo": finales["estacional ingenuo"],
                                  "lineal + mes": finales["tendencia lineal + mes"], "cuadrática\n+ mes": finales["tendencia cuadrática + mes"]}, separacion=0)
ax.set_xlim(2014.5, 2029.5)
ax.set_xticks([2015, 2018, 2021, 2024])
ax.set_title("MAE de cada año de prueba (ppm, escala log)")
ax.set_xlabel("año")
fig.tight_layout()
plt.show()

# %%
dentro80 = ((prueba >= int80.iloc[:, 0]) & (prueba <= int80.iloc[:, 1])).to_numpy()
dentro95 = ((prueba >= int95.iloc[:, 0]) & (prueba <= int95.iloc[:, 1])).to_numpy()
ancho = (int95.iloc[:, 1] - int95.iloc[:, 0]) / 2
print(f"semiancho del intervalo del 95 %: {ancho.iloc[0]:.2f} ppm a 1 mes, {ancho.iloc[11]:.2f} a 12 meses, {ancho.iloc[-1]:.2f} a {len(prueba)} meses")
print(f"meses de prueba dentro del intervalo: 80 % → {dentro80.mean():.0%}, 95 % → {dentro95.mean():.0%}")
print(f"septiembre de 2026: observado {prueba.iloc[-1]:.2f}, SARIMA {media.iloc[-1]:.2f} [{int95.iloc[-1, 0]:.1f}, {int95.iloc[-1, 1]:.1f}]")
print(f"error medio (observado − SARIMA) en 2025: {(prueba['2025'] - media['2025']).mean():.2f} ppm")

# %% [markdown]
# Hay mucho que leer aquí.
#
# - **El SARIMA reproduce el ciclo y la tendencia**, y a un año su error es de unas décimas. Pero se va quedando abajo: en septiembre de 2026 pronosticó 420.5 ppm y se midieron 426.1. ¿Por qué? Con $d = D = 1$ y sin constante, el modelo extrapola **la pendiente reciente** (la de 2014, de algo más de 2 ppm por año), y el CO₂ siguió acelerándose. Ningún término del modelo puede anticipar una aceleración que no vio.
# - **Los intervalos se abren** como predijo la sección 7, de ±0.6 ppm a un mes a ±1.3 a un año y ±6.9 a casi doce años. El 95 % contiene el 94 % de los meses, pero el 80 % solo el 21 %: los meses de una misma trayectoria no son 141 pruebas independientes; si el pronóstico se desvía, se desvía para todos los meses siguientes. Para evaluar la calibración de los intervalos hay que pronosticar desde muchos orígenes (sección 9).
# - **La regresión con tendencia lineal** falla por mucho (13 ppm de MAE; MASE 8.8: casi nueve veces el error típico del estacional ingenuo a un año dentro de la muestra): una recta no puede seguir una curva que se acelera. **El estacional ingenuo** es todavía peor a largo plazo, porque no tiene tendencia.
# - **La tendencia cuadrática + mes gana a largo plazo** (MAE 2.0 contra 3.1): su forma fija sí captura la aceleración de 1958–2014, y esa aceleración continuó. Es una lección importante: el modelo más flexible no es el mejor en todos los horizontes. El SARIMA es excelente para el corto plazo (aprende de los últimos meses), y una forma global bien elegida puede ganar a muy largo plazo, siempre que el mundo siga haciendo lo mismo, lo cual nadie garantiza.

# %% [markdown]
# ## 9. Validación con origen móvil a 12 meses
#
# Un solo origen de pronóstico es una sola muestra. Lo correcto, como vimos en el ejemplo anterior, es pronosticar desde **muchos orígenes**: para cada mes $o$ de enero de 2015 a septiembre de 2025, usar los datos hasta $o$, pronosticar los siguientes 12 meses y medir el error en cada horizonte $h = 1,\dots,12$. Para el SARIMA mantenemos los parámetros estimados con los datos hasta 2014 y solo actualizamos los datos (`apply`), que es barato; las regresiones las reajustamos en cada origen, y el estacional ingenuo usa el valor de 12 meses antes del mes pronosticado.

# %%
H = 12
origenes = range(n_ent, len(y) - H + 1)
errores = {k: [] for k in pronosticos}
cubre80, cubre95 = [], []
yv = y.to_numpy()
for o in origenes:
    real = yv[o : o + H]
    p = modelo.apply(y.iloc[:o]).get_forecast(H)
    errores["SARIMA"].append(real - p.predicted_mean.to_numpy())
    lo, hi = p.conf_int(alpha=0.20).to_numpy().T
    cubre80.append((real >= lo) & (real <= hi))
    lo, hi = p.conf_int(alpha=0.05).to_numpy().T
    cubre95.append((real >= lo) & (real <= hi))
    errores["estacional ingenuo"].append(real - yv[o - 12 : o - 12 + H])
    errores["tendencia lineal + mes"].append(real - regresion(X_lin, o)[o : o + H])
    errores["tendencia cuadrática + mes"].append(real - regresion(X_cuad, o)[o : o + H])
mae_h = pd.DataFrame({k: np.abs(np.array(v)).mean(axis=0) for k, v in errores.items()}, index=pd.Index(range(1, H + 1), name="h"))
cobertura = pd.DataFrame({"80 %": np.mean(cubre80, axis=0), "95 %": np.mean(cubre95, axis=0)}, index=mae_h.index)
print(f"{len(origenes)} orígenes; MAE promedio sobre h = 1…12:")
print((mae_h.mean()).round(2).to_string())
print(f"cobertura promedio: 80 % → {np.mean(cubre80):.0%}, 95 % → {np.mean(cubre95):.0%}")

# %%
fig, ejes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
ax = ejes[0]
for (nombre, color) in zip(mae_h.columns, [AZUL, GRIS, ROJO, DORADO_OSC]):
    ax.plot(mae_h.index, mae_h[nombre], marker="o", ms=4, color=color)
ax.set_yscale("log")
ax.set_yticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
rdp.estilo.etiquetar(ax, 12.4, {"SARIMA": mae_h.iloc[-1, 0], "estacional ingenuo": mae_h.iloc[-1, 1],
                                 "lineal + mes": mae_h.iloc[-1, 2], "cuadrática + mes": mae_h.iloc[-1, 3]}, separacion=0)
ax.set_xlim(0.5, 16.5)
ax.set_xticks(range(1, 13))
ax.set_xlabel("horizonte h (meses)")
ax.set_title(f"MAE por horizonte, {len(origenes)} orígenes (ppm, escala log)")
ax = ejes[1]
ax.plot(cobertura.index, cobertura["80 %"], marker="o", ms=4, color=AZUL)
ax.plot(cobertura.index, cobertura["95 %"], marker="o", ms=4, color=ROJO)
ax.axhline(0.80, color=AZUL, ls=":", lw=1.2)
ax.axhline(0.95, color=ROJO, ls=":", lw=1.2)
ax.text(12.4, 0.80, "80 %", color=AZUL, va="center", backgroundcolor="white")
ax.text(12.4, 0.95, "95 %", color=ROJO, va="center", backgroundcolor="white")
ax.set_ylim(0.6, 1.0)
ax.set_xlim(0.5, 13.8)
ax.set_xticks(range(1, 13))
ax.set_xlabel("horizonte h (meses)")
ax.set_title("Cobertura de los intervalos del SARIMA")
fig.tight_layout()
plt.show()
mae_h.round(2).T

# %% [markdown]
# Ahora el panorama es otro:
#
# - A horizontes de hasta un año, el **SARIMA es por mucho el mejor**: 0.28 ppm a un mes y 0.64 a doce, menos de la mitad que la regresión cuadrática y una cuarta parte del estacional ingenuo (en promedio sobre los 12 horizontes, MAE 0.45 ppm y MASE $0.45/1.50 = 0.30$). Su error crece con $h$, como debe.
# - El estacional ingenuo tiene error casi constante, unos 2.6 ppm: el crecimiento de un año, que ignora.
# - Los intervalos del SARIMA cubren un poco **menos** de lo prometido (≈75 % el de 80 % y ≈90 % el de 95 %). Las razones probables: los parámetros se estimaron hasta 2014 y no incluyen su propia incertidumbre, y la serie se aceleró después (el error medio del SARIMA es positivo y crece con $h$). Un intervalo de predicción es tan bueno como el supuesto de que el futuro se comporta como el pasado.
#
# ## 10. Cierre
#
# - Un **AR($p$)** es una regresión lineal sobre los rezagos; un **MA($q$)** modela ecos de choques pasados; la ACF y la PACF los distinguen.
# - **Diferenciar** convierte tendencias y ciclos en algo estacionario; la «I» de ARIMA y la parte estacional de SARIMA no son más que eso.
# - Se ajusta por **máxima verosimilitud**, se elige con **AIC** y se valida **revisando los residuos** (ACF, Ljung–Box) y, sobre todo, **pronosticando hacia adelante** desde muchos orígenes.
# - La incertidumbre crece con el horizonte, sin cota si hay diferencias. Y ningún intervalo protege contra un cambio de régimen: la curva de Keeling se aceleró más de lo que los datos hasta 2014 permitían prever.
#
# *Referencias:* C. D. Keeling, «The concentration and isotopic abundances of carbon dioxide in the atmosphere», *Tellus* 12 (1960). G. E. P. Box, G. M. Jenkins, G. C. Reinsel y G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5.ª ed., Wiley (2015). R. J. Hyndman y G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed., OTexts (2021), capítulo 9. G. M. Ljung y G. E. P. Box, «On a measure of lack of fit in time series models», *Biometrika* 65 (1978).

# %% [markdown]
# ## 11. Autoevaluación

# %%
rdp.quiz.mostrar("u07", ids=["u07-p05", "u07-p06", "u07-p07", "u07-p08"])

# %% [markdown]
# ## 12. Ejercicios
#
# 1. **ACF del MA(1) y del AR(2).** Demuestra que para el MA(1), $\rho_1 = \theta/(1+\theta^2)$, y que $|\rho_1| \le 1/2$ para cualquier $\theta$. ¿Qué dice esto sobre una serie cuya ACF muestral tiene una sola barra de 0.7? Para el AR(2), deduce las ecuaciones de Yule–Walker $\rho_k = \phi_1\rho_{k-1} + \phi_2\rho_{k-2}$ y úsalas para calcular $\rho_1$ y $\rho_2$ del AR(2) de la sección 3; compáralos con la ACF muestral de la serie simulada.
# 2. **Pronóstico del AR(1) con media.** Para $y_t = c + \phi y_{t-1} + \varepsilon_t$ con $|\phi|<1$, demuestra que $\hat y_{T+h\mid T} = \mu + \phi^h(y_T - \mu)$ con $\mu = c/(1-\phi)$. Luego, para un AR(2) ajustado a mano (sección 3), programa el pronóstico recursivo de $h$ pasos y compáralo con `AutoReg(...).fit().predict`.
# 3. **Un AR grande contra un SARIMA chico.** Ajusta por mínimos cuadrados (con tu `matriz_rezagos`) modelos AR($p$) a $\nabla\nabla_{12}y$ con $p = 1, \dots, 36$ y elige $p$ con el AIC aproximado $T\ln\hat\sigma^2 + 2(p+1)$. ¿Qué $p$ sale y por qué tiene que ser al menos 12? ¿Cuántos parámetros usa, contra los 4 del SARIMA? Compara sus pronósticos a 12 meses con origen móvil.
# 4. **Una deriva para la aceleración.** Ajusta el SARIMA con `trend="t"` (con $d = D = 1$, una constante en la serie diferenciada dos veces equivale a una tendencia cuadrática en la original) y repite las secciones 8 y 9. ¿Mejora el pronóstico a largo plazo? ¿Y la cobertura de los intervalos? Discute por qué agregar una tendencia determinista a un modelo diferenciado es una decisión fuerte.
# 5. **Reestimar o no reestimar.** En la sección 9 los parámetros se quedaron fijos en los de 2014. Repite la validación reestimando el SARIMA una vez al año (cada 12 orígenes). ¿Cambian el MAE y la cobertura? ¿Cuánto tiempo de cómputo cuesta?
